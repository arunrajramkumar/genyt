# ytagent — fully-automated YouTube video generator

Give it a theme, it produces a finished, ready-to-upload video:

```
theme  -->  local Ollama LLM writes script  -->  Piper narrates each scene
       -->  Pexels fetches matching stock footage/photos
       -->  ffmpeg assembles everything into one MP4
```

Output per video (in `output/`):
- `<slug>.mp4` — the finished video (1920x1080, H.264 + AAC)
- `<slug>.srt` — captions file (upload to YouTube separately as closed captions —
  this ffmpeg build has no subtitle-burn-in support, see Notes below)
- `<slug>.metadata.txt` — title, description, tags to paste into YouTube Studio

This tool does **not** upload to YouTube — you review and publish the output yourself.

## One-time setup

```bash
pip3 install -r requirements.txt
```

Edit `.env` and fill in:
- `PEXELS_API_KEY` — free, from https://www.pexels.com/api/

Ollama and the `llama3.2` model are already installed on this machine (in
`~/.local/bin`, since Homebrew's `ollama` formula fails to build from source
here — see Notes below). Before running the pipeline, make sure the Ollama
server is running:

```bash
ollama serve &     # starts the local model server; leave it running
```

If it's not already running, `run.py` will fail with a clear connection error
telling you to start it.

The Piper voice model and ffmpeg are already installed for this machine.

## Usage

Edit `themes.yaml` to add topics you want videos for, then:

```bash
python3 run.py            # produce a video for the next un-done theme
python3 run.py --all      # produce videos for every un-done theme
python3 run.py --theme "3 weird facts about octopuses" --duration 120   # one-off, skip themes.yaml
```

Each theme in `themes.yaml` is marked `done: true` after its video is produced, so
you can re-run `run.py` on a schedule (e.g. via `cron`) and it will only ever pick
up new themes you add.

### Running on a schedule (cron)

```
# every day at 9am, generate one video from themes.yaml
0 9 * * * cd /Users/aramkumar/ytagent && /usr/bin/python3 run.py >> cron.log 2>&1
```

## Notes / limitations

- **Ollama is installed manually, not via Homebrew.** `brew install ollama` tries
  to compile from source on this machine (its Homebrew lives at a non-standard
  prefix, so no precompiled bottles are available) and that build is flaky. Instead,
  the official precompiled app was downloaded from ollama.com and its CLI binary
  plus runtime libraries were extracted into `~/.local/bin`. If you ever need to
  reinstall/upgrade, redownload `https://ollama.com/download/Ollama-darwin.zip`,
  unzip it, and copy everything from `Ollama.app/Contents/Resources/` into
  `~/.local/bin/` again.
- **`ollama serve` must be running** for script generation to work. It's not set
  up as an auto-starting service — start it manually (`ollama serve &`) each
  session, or set up a `launchd` agent if you want it always running.
- **Captions aren't burned into the video.** This machine's ffmpeg was built from
  source (non-standard Homebrew prefix) without `libass`, so the `subtitles` filter
  isn't available. The `.srt` file is generated separately — upload it in YouTube
  Studio under Subtitles for the same effect, with the benefit that viewers can
  toggle it off.
- **Script quality is local-LLM quality.** `llama3.2` (3B) is fast but less
  capable than a hosted frontier model — expect occasionally repetitive or
  generic scripts. Swap models via `OLLAMA_MODEL` in `.env` (e.g. a larger
  `llama3.1:8b` or `qwen2.5:14b` if your machine can run it) for better output
  at the cost of slower generation.
- **Stock visuals** come from Pexels based on the local model's `visual_query` per
  scene. Quality depends on how well the query matches available stock footage —
  review output before publishing.
- No upload automation is included by design (per your choice) — you stay in
  control of what actually gets published to your channel.
