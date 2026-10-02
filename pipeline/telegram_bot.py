"""Telegram bot integration: lets a mobile user submit a video prompt as a chat
message and receive the finished video back in the same chat, instead of using
the web UI. Wired into webapp.py via an incoming-webhook route.
"""
import os
import re
import threading

import requests

from . import chat_prefs, config, prompt_script, producer

BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN", "")
WEBHOOK_SECRET = os.environ.get("TELEGRAM_WEBHOOK_SECRET", "")
ALLOWED_CHAT_IDS = {
    c.strip() for c in os.environ.get("TELEGRAM_ALLOWED_CHAT_IDS", "").split(",") if c.strip()
}

API_BASE = f"https://api.telegram.org/bot{BOT_TOKEN}"

# Some Telegram clients (e.g. Desktop with the default "Enter sends message"
# setting) send each line of a multi-paragraph prompt as a separate message
# rather than one message with embedded newlines. Without buffering, each
# fragment would kick off its own tiny, nonsensical video instead of the one
# full prompt the user intended. Fragments from the same chat arriving within
# this window are joined back into a single prompt before generation starts.
PROMPT_DEBOUNCE_SECONDS = 3.0

_pending_lock = threading.Lock()
_pending_fragments = {}  # chat_id -> list[str], messages waiting to be joined
_pending_timers = {}  # chat_id -> threading.Timer, armed to flush the buffer

HELP_TEXT = (
    "Send me a topic or prompt (e.g. \"5 facts about octopuses\" or a stock's "
    "name + numbers) and I'll generate a vertical YouTube Short and send it "
    "back here — usually takes a few minutes.\n\n"
    "Want a different narration language? Send /voice to see your options."
)

# Friendly name -> edge-tts voice ID, for the handful of languages most likely
# to be requested. Not exhaustive — advanced users can also send any raw
# edge-tts voice ID directly (e.g. "cy-GB-NiaNeural") and it's used as-is.
VOICE_ALIASES = {
    "english": "en-IN-PrabhatNeural",
    "english-in": "en-IN-PrabhatNeural",
    "english-us": "en-US-AriaNeural",
    "english-uk": "en-GB-SoniaNeural",
    "hindi": "hi-IN-SwaraNeural",
    "tamil": "ta-IN-PallaviNeural",
    "telugu": "te-IN-ShrutiNeural",
    "kannada": "kn-IN-SapnaNeural",
    "malayalam": "ml-IN-SobhanaNeural",
    "marathi": "mr-IN-AarohiNeural",
    "bengali": "bn-IN-TanishaaNeural",
    "gujarati": "gu-IN-DhwaniNeural",
    "spanish": "es-ES-ElviraNeural",
    "french": "fr-FR-DeniseNeural",
    "german": "de-DE-KatjaNeural",
    "arabic": "ar-SA-ZariyahNeural",
    "chinese": "zh-CN-XiaoxiaoNeural",
    "japanese": "ja-JP-NanamiNeural",
    "portuguese": "pt-BR-FranciscaNeural",
}

_RAW_VOICE_RE = re.compile(r"^[a-z]{2}-[A-Z]{2}-\w+Neural$")

VOICE_HELP_TEXT = (
    "Current languages: " + ", ".join(sorted(VOICE_ALIASES)) + "\n\n"
    "Send /voice <language> to switch, e.g. \"/voice hindi\". "
    "Or send any exact edge-tts voice ID (e.g. \"/voice ta-IN-ValluvarNeural\") "
    "for finer control — run `edge-tts --list-voices` for the full catalog."
)


def _resolve_voice(arg: str) -> str:
    arg = arg.strip()
    alias = VOICE_ALIASES.get(arg.lower())
    if alias:
        return alias
    if _RAW_VOICE_RE.match(arg):
        return arg
    return ""


def _check(response: requests.Response) -> None:
    """Telegram returns 200 with {"ok": false, "description": ...} on failure
    rather than an HTTP error status — raise explicitly so callers don't
    silently swallow a failed send (e.g. a video over the 50MB bot upload cap)."""
    try:
        data = response.json()
    except ValueError:
        response.raise_for_status()
        return
    if not data.get("ok"):
        raise RuntimeError(f"Telegram API error: {data.get('description', data)}")


def _send_message(chat_id, text: str) -> None:
    _check(requests.post(f"{API_BASE}/sendMessage", data={"chat_id": chat_id, "text": text}, timeout=30))


def _send_video(chat_id, path, caption: str = "") -> None:
    with open(path, "rb") as f:
        _check(requests.post(
            f"{API_BASE}/sendVideo",
            data={"chat_id": chat_id, "caption": caption[:1024]},
            files={"video": f},
            timeout=300,
        ))


def _send_photo(chat_id, path, caption: str = "") -> None:
    with open(path, "rb") as f:
        _check(requests.post(
            f"{API_BASE}/sendPhoto",
            data={"chat_id": chat_id, "caption": caption[:1024]},
            files={"photo": f},
            timeout=60,
        ))


def _send_document(chat_id, path, caption: str = "") -> None:
    with open(path, "rb") as f:
        _check(requests.post(
            f"{API_BASE}/sendDocument",
            data={"chat_id": chat_id, "caption": caption[:1024]},
            files={"document": f},
            timeout=120,
        ))


def _run_job(chat_id, prompt: str) -> None:
    try:
        _send_message(chat_id, "Got it — generating your video now, this can take a few minutes.")
        script = prompt_script.write_script_from_prompt(prompt=prompt, duration_sec=60)
        voice = chat_prefs.get_voice(chat_id) or None
        result = producer.produce_from_script(script, config.SHORTS, on_progress=lambda msg: None, voice=voice)
        _send_video(chat_id, result["video"], caption=script["title"])
        _send_photo(chat_id, result["thumbnail"], caption="Thumbnail")
        _send_document(chat_id, result["srt"], caption="Captions (.srt)")
        _send_document(chat_id, result["metadata"], caption="Title / description / tags")
    except Exception as e:
        _send_message(chat_id, f"Sorry, video generation failed: {e}")


def _flush_prompt(chat_id) -> None:
    with _pending_lock:
        fragments = _pending_fragments.pop(chat_id, [])
        _pending_timers.pop(chat_id, None)
    if not fragments:
        return
    threading.Thread(target=_run_job, args=(chat_id, "\n\n".join(fragments)), daemon=True).start()


def _queue_prompt(chat_id, text: str) -> None:
    with _pending_lock:
        _pending_fragments.setdefault(chat_id, []).append(text)
        old_timer = _pending_timers.get(chat_id)
        if old_timer is not None:
            old_timer.cancel()
        timer = threading.Timer(PROMPT_DEBOUNCE_SECONDS, _flush_prompt, args=(chat_id,))
        timer.daemon = True
        _pending_timers[chat_id] = timer
    # Started outside the lock: a timer that fires synchronously/immediately
    # (as in tests) would otherwise re-enter _flush_prompt's `with _pending_lock`
    # while this thread still holds it, deadlocking on this non-reentrant lock.
    timer.start()


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

    if text == "/voice":
        current = chat_prefs.get_voice(chat_id) or config.TTS_VOICE
        _send_message(chat_id, f"Current voice: {current}\n\n{VOICE_HELP_TEXT}")
        return

    if text.startswith("/voice "):
        arg = text[len("/voice "):]
        voice = _resolve_voice(arg)
        if not voice:
            _send_message(chat_id, f"Didn't recognize \"{arg}\".\n\n{VOICE_HELP_TEXT}")
            return
        chat_prefs.set_voice(chat_id, voice)
        _send_message(chat_id, f"Voice set to {voice}. Your next video will use it.")
        return

    _queue_prompt(chat_id, text)
