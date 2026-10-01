"""Telegram bot integration: lets a mobile user submit a video prompt as a chat
message and receive the finished video back in the same chat, instead of using
the web UI. Wired into webapp.py via an incoming-webhook route.
"""
import os
import threading

import requests

from . import config, prompt_script, producer

BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN", "")
WEBHOOK_SECRET = os.environ.get("TELEGRAM_WEBHOOK_SECRET", "")
ALLOWED_CHAT_IDS = {
    c.strip() for c in os.environ.get("TELEGRAM_ALLOWED_CHAT_IDS", "").split(",") if c.strip()
}

API_BASE = f"https://api.telegram.org/bot{BOT_TOKEN}"

HELP_TEXT = (
    "Send me a topic or prompt (e.g. \"5 facts about octopuses\" or a stock's "
    "name + numbers) and I'll generate a vertical YouTube Short and send it "
    "back here — usually takes a few minutes."
)


def _send_message(chat_id, text: str) -> None:
    requests.post(f"{API_BASE}/sendMessage", data={"chat_id": chat_id, "text": text}, timeout=30)


def _send_video(chat_id, path, caption: str = "") -> None:
    with open(path, "rb") as f:
        requests.post(
            f"{API_BASE}/sendVideo",
            data={"chat_id": chat_id, "caption": caption[:1024]},
            files={"video": f},
            timeout=300,
        )


def _send_photo(chat_id, path, caption: str = "") -> None:
    with open(path, "rb") as f:
        requests.post(
            f"{API_BASE}/sendPhoto",
            data={"chat_id": chat_id, "caption": caption[:1024]},
            files={"photo": f},
            timeout=60,
        )


def _send_document(chat_id, path, caption: str = "") -> None:
    with open(path, "rb") as f:
        requests.post(
            f"{API_BASE}/sendDocument",
            data={"chat_id": chat_id, "caption": caption[:1024]},
            files={"document": f},
            timeout=120,
        )


def _run_job(chat_id, prompt: str) -> None:
    try:
        _send_message(chat_id, "Got it — generating your video now, this can take a few minutes.")
        script = prompt_script.write_script_from_prompt(prompt=prompt, duration_sec=60)
        result = producer.produce_from_script(script, config.SHORTS, on_progress=lambda msg: None)
        _send_video(chat_id, result["video"], caption=script["title"])
        _send_photo(chat_id, result["thumbnail"], caption="Thumbnail")
        _send_document(chat_id, result["srt"], caption="Captions (.srt)")
        _send_document(chat_id, result["metadata"], caption="Title / description / tags")
    except Exception as e:
        _send_message(chat_id, f"Sorry, video generation failed: {e}")


def handle_update(update: dict) -> None:
    """Processes one Telegram webhook update. Fire-and-forget: always returns
    immediately, kicking off generation in a background thread, since Telegram
    expects a fast webhook response."""
    message = update.get("message") or {}
    chat_id = message.get("chat", {}).get("id")
    text = (message.get("text") or "").strip()
    if not chat_id or not text:
        return

    if ALLOWED_CHAT_IDS and str(chat_id) not in ALLOWED_CHAT_IDS:
        _send_message(chat_id, "Sorry, you're not authorized to use this bot.")
        return

    if text in ("/start", "/help"):
        _send_message(chat_id, HELP_TEXT)
        return

    threading.Thread(target=_run_job, args=(chat_id, text), daemon=True).start()
