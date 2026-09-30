#!/usr/bin/env python3
"""Small local web UI for generating stock-analysis YouTube Shorts.

Run with:  python3 webapp.py
Then open: http://127.0.0.1:5050
"""
import os
import threading
import traceback
import uuid

from flask import Flask, request, redirect, url_for, jsonify, send_file, abort, Response

from pipeline import config, prompt_script, producer

app = Flask(__name__)

# Defense-in-depth on top of the Hugging Face Space's own "Private" visibility:
# if APP_PASSWORD is set (e.g. via a Space secret), gate every route behind HTTP
# Basic Auth. Unset locally, so `python3 webapp.py` on your own machine needs no auth.
APP_USERNAME = os.environ.get("APP_USERNAME", "admin")
APP_PASSWORD = os.environ.get("APP_PASSWORD")


@app.before_request
def _check_auth():
    if not APP_PASSWORD:
        return None
    auth = request.authorization
    if not auth or auth.username != APP_USERNAME or auth.password != APP_PASSWORD:
        return Response(
            "Authentication required", 401,
            {"WWW-Authenticate": 'Basic realm="Video Generator"'},
        )
    return None

# In-memory job store — fine for a single-user local tool.
JOBS = {}
JOBS_LOCK = threading.Lock()

DURATIONS = [30, 45, 60, 90, 180]

FORM_PAGE = """
<!doctype html>
<html>
<head>
<meta charset="utf-8">
<title>Video Generator</title>
<style>
  body { font-family: -apple-system, sans-serif; max-width: 640px; margin: 40px auto; padding: 0 16px; color: #1a1a1a; }
  h1 { font-size: 1.4rem; }
  label { display: block; margin-top: 14px; font-weight: 600; font-size: 0.9rem; }
  input, textarea, select { width: 100%; padding: 8px; margin-top: 4px; box-sizing: border-box; font-size: 0.95rem; }
  textarea { min-height: 160px; font-family: inherit; }
  button { margin-top: 20px; padding: 10px 20px; font-size: 1rem; cursor: pointer; }
  .hint { color: #666; font-size: 0.8rem; margin-top: 2px; }
  .disclaimer { background: #fff3cd; padding: 10px; border-radius: 6px; font-size: 0.85rem; margin-bottom: 20px; }
  .row { display: flex; gap: 12px; }
  .row > div { flex: 1; }
</style>
</head>
<body>
  <h1>AI Video Generator</h1>
  <div class="disclaimer">
    Describe any topic, or paste your own analysis/facts (e.g. a stock's revenue,
    profit, P/E). If you include specific facts/numbers, the model will only
    narrate those — it won't invent figures. For open topics, it'll write freely.
  </div>
  <form method="post" action="/generate">
    <label>What should this video be about?</label>
    <textarea name="prompt" required placeholder="e.g. Give a neutral analysis of XYZ Industries Ltd: manufactures specialty chemicals for textiles and agrochemicals, based in Gujarat. Revenue ₹412 Cr (FY24), profit ₹38 Cr (FY24), P/E 22.5, revenue grew 18% YoY. Discuss whether it looks good for long-term growth, no buy/sell recommendation.

or just: The history of the Roman aqueducts, or: 5 weird facts about octopuses."></textarea>

    <div class="row">
      <div>
        <label>Format</label>
        <select name="orientation">
          <option value="shorts" selected>Vertical (YouTube Shorts)</option>
          <option value="landscape">Landscape (regular YouTube)</option>
        </select>
      </div>
      <div>
        <label>Video length</label>
        <select name="duration">
          {% for d in durations %}
          <option value="{{ d }}" {% if d == 60 %}selected{% endif %}>{{ d }} seconds</option>
          {% endfor %}
        </select>
      </div>
    </div>

    <button type="submit">Generate Video</button>
  </form>
</body>
</html>
"""

STATUS_PAGE = """
<!doctype html>
<html>
<head>
<meta charset="utf-8">
<title>Generating...</title>
<style>
  body { font-family: -apple-system, sans-serif; max-width: 640px; margin: 40px auto; padding: 0 16px; }
  #log { background: #111; color: #0f0; padding: 12px; border-radius: 6px; font-family: monospace; font-size: 0.85rem; white-space: pre-wrap; min-height: 120px; }
  video { width: 100%; max-width: 360px; margin-top: 16px; border-radius: 8px; }
  .error { color: #b00020; white-space: pre-wrap; }
  a.button { display: inline-block; margin: 6px 8px 0 0; padding: 8px 14px; background: #1a1a1a; color: white; text-decoration: none; border-radius: 6px; }
</style>
</head>
<body>
  <h1>Job {{ job_id[:8] }}</h1>
  <div id="log">Starting...</div>
  <div id="result"></div>
  <script>
    async function poll() {
      const r = await fetch("/job/{{ job_id }}/status");
      const data = await r.json();
      document.getElementById("log").textContent = data.log.join("\\n");
      if (data.status === "done") {
        document.getElementById("result").innerHTML = `
          <video controls src="/job/{{ job_id }}/video"></video><br>
          <a class="button" href="/job/{{ job_id }}/video" download>Download MP4</a>
          <a class="button" href="/job/{{ job_id }}/srt" download>Download Captions (.srt)</a>
          <a class="button" href="/job/{{ job_id }}/metadata" download>Download Title/Description/Tags</a>
        `;
        return;
      }
      if (data.status === "error") {
        document.getElementById("result").innerHTML = '<p class="error">' + data.error + '</p>';
        return;
      }
      setTimeout(poll, 1500);
    }
    poll();
  </script>
</body>
</html>
"""


def _run_job(job_id: str, form: dict) -> None:
    def log(msg):
        with JOBS_LOCK:
            JOBS[job_id]["log"].append(msg)

    try:
        log("Writing script from your prompt...")
        script = prompt_script.write_script_from_prompt(
            prompt=form["prompt"],
            duration_sec=int(form.get("duration", 60)),
        )
        orientation = config.LANDSCAPE if form.get("orientation") == "landscape" else config.SHORTS
        result = producer.produce_from_script(script, orientation, on_progress=log)
        with JOBS_LOCK:
            JOBS[job_id]["status"] = "done"
            JOBS[job_id]["result"] = result
    except Exception as e:
        with JOBS_LOCK:
            JOBS[job_id]["status"] = "error"
            JOBS[job_id]["error"] = f"{e}\n\n{traceback.format_exc()}"


@app.route("/")
def index():
    from flask import render_template_string
    return render_template_string(FORM_PAGE, durations=DURATIONS)


@app.route("/generate", methods=["POST"])
def generate():
    job_id = uuid.uuid4().hex
    form = request.form.to_dict()
    with JOBS_LOCK:
        JOBS[job_id] = {"status": "running", "log": [], "result": None, "error": None}
    thread = threading.Thread(target=_run_job, args=(job_id, form), daemon=True)
    thread.start()
    return redirect(url_for("job_page", job_id=job_id))


@app.route("/job/<job_id>")
def job_page(job_id):
    if job_id not in JOBS:
        abort(404)
    from flask import render_template_string
    return render_template_string(STATUS_PAGE, job_id=job_id)


@app.route("/job/<job_id>/status")
def job_status(job_id):
    job = JOBS.get(job_id)
    if not job:
        abort(404)
    with JOBS_LOCK:
        return jsonify({"status": job["status"], "log": job["log"], "error": job["error"]})


@app.route("/job/<job_id>/video")
def job_video(job_id):
    job = JOBS.get(job_id)
    if not job or job["status"] != "done":
        abort(404)
    return send_file(job["result"]["video"], mimetype="video/mp4")


@app.route("/job/<job_id>/srt")
def job_srt(job_id):
    job = JOBS.get(job_id)
    if not job or job["status"] != "done":
        abort(404)
    return send_file(job["result"]["srt"], as_attachment=True)


@app.route("/job/<job_id>/metadata")
def job_metadata(job_id):
    job = JOBS.get(job_id)
    if not job or job["status"] != "done":
        abort(404)
    return send_file(job["result"]["metadata"], as_attachment=True)


if __name__ == "__main__":
    # HF Spaces (Docker SDK) expects the app to listen on 0.0.0.0:7860.
    port = int(os.environ.get("PORT", 7860))
    app.run(host="0.0.0.0", port=port, debug=False)
