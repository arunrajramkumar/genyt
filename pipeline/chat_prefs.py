"""Tiny JSON-backed store for per-Telegram-chat preferences (currently just the
narration voice/language). Persists across requests within a running instance;
lost on redeploy/restart since the host filesystem is ephemeral — acceptable
since a user can just re-run /voice.
"""
import json
from pathlib import Path

from . import config

_PREFS_PATH = config.CACHE_DIR / "chat_prefs.json"


def _load() -> dict:
    if not _PREFS_PATH.exists():
        return {}
    try:
        return json.loads(_PREFS_PATH.read_text(encoding="utf-8"))
    except (ValueError, OSError):
        return {}


def _save(data: dict) -> None:
    _PREFS_PATH.write_text(json.dumps(data), encoding="utf-8")


def get_voice(chat_id) -> str:
    return _load().get(str(chat_id), {}).get("voice", "")


def set_voice(chat_id, voice: str) -> None:
    data = _load()
    data.setdefault(str(chat_id), {})["voice"] = voice
    _save(data)
