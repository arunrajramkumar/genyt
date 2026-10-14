"""Synthesizes narration using Microsoft Edge's free neural TTS (via edge-tts),
which offers a genuine Indian-English voice — something Piper's voice catalog
doesn't have (Piper only ships en_US/en_GB).

This machine sits behind a TLS-intercepting corporate proxy, and edge-tts
hardcodes its SSL context to certifi's bundle (ignoring SSL_CERT_FILE/etc), so
we patch its SSL context at import time to trust the proxy's CA bundle.
"""
import asyncio
import re
import shutil
import ssl
import subprocess
import time
from pathlib import Path

from . import config

# macOS: this machine's proxy-intercepted cert bundle. Linux (Docker): the
# system CA store, which on a TLS-intercepting network has the proxy's CA
# appended to it via update-ca-certificates (see Dockerfile.local) — on a
# non-intercepted network (e.g. Hugging Face Spaces) it's just the normal
# public root list, so this is a no-op there, not a security downgrade.
_PROXY_CA_CANDIDATES = [
    "/usr/local/etc/openssl/certs/combined_cacerts.pem",
    "/etc/ssl/certs/ca-certificates.crt",
]

import edge_tts.communicate as _comm
import edge_tts.voices as _voices_mod

for _bundle in _PROXY_CA_CANDIDATES:
    if Path(_bundle).exists():
        _ctx = ssl.create_default_context(cafile=_bundle)
        _comm._SSL_CTX = _ctx
        _voices_mod._SSL_CTX = _ctx
        break

import base64
import json
from datetime import datetime, timezone

import edge_tts
import aiohttp
import requests

FFPROBE_BIN = shutil.which("ffprobe")

# Abbreviations that TTS engines tend to spell out letter-by-letter instead of
# reading as words. Expanded before synthesis so narration sounds natural.
#
# These patterns need a trailing (?![a-zA-Z]) (not just a leading \b), because
# without it "Corp"/"Inc" etc. match as a bare prefix of ordinary words too —
# "Corp\.?" alone matches the first 4 letters of "Corporation"/"Corporate" and
# "Inc\.?" matches the first 3 letters of "Incorporation"/"Increase"/"Income"/
# "Incredible", splicing the expansion into the leftover letters and producing
# garbage like "Corporationoration" or "Incorporatedorporation".
_ABBREVIATIONS = [
    (r"\bLtd\.?(?![a-zA-Z])", "Limited"),
    (r"\bPvt\.?(?![a-zA-Z])", "Private"),
    (r"\bCorp\.?(?![a-zA-Z])", "Corporation"),
    (r"\bInc\.?(?![a-zA-Z])", "Incorporated"),
    (r"\bCo\.(?![a-zA-Z])", "Company"),
    (r"\bP/E\b", "price to earnings"),
    (r"\bYoY\b", "year over year"),
    (r"\bQoQ\b", "quarter over quarter"),
    (r"\bRs\.?(?=\s?\d)", "Rupees"),
    (r"₹", "Rupees "),
    (r"~\s?(?=\d)", "approximately "),
    (r"\bCr\.?\b", "Crore"),
    (r"\bROCE\b", "return on capital employed"),
    (r"\bROE\b", "return on equity"),
    (r"\bCAGR\b", "compound annual growth rate"),
    (r"\bEBITDA\b", "earnings before interest, tax, depreciation and amortization"),
    (r"\bNPA\b", "non-performing assets"),
    (r"\bCASA\b", "current account savings account"),
    (r"\bNIM\b", "net interest margin"),
]


def _normalize_for_speech(text: str) -> str:
    for pattern, replacement in _ABBREVIATIONS:
        text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
    return text


def _probe_duration(path: Path) -> float:
    result = subprocess.run(
        [FFPROBE_BIN, "-v", "error", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", str(path)],
        check=True, capture_output=True, text=True,
    )
    return float(result.stdout.strip())


async def _synthesize_async(text: str, voice: str, out_path: Path) -> None:
    communicate = edge_tts.Communicate(text, voice)
    await communicate.save(str(out_path))


def _synthesize_google(text: str, out_path: Path) -> None:
    response = requests.post(
        "https://texttospeech.googleapis.com/v1/text:synthesize",
        params={"key": config.GOOGLE_TTS_API_KEY},
        json={
            "input": {"text": text},
            "voice": {"languageCode": "en-IN", "name": config.GOOGLE_TTS_VOICE},
            "audioConfig": {"audioEncoding": "MP3"},
        },
        timeout=30,
    )
    response.raise_for_status()
    out_path.write_bytes(base64.b64decode(response.json()["audioContent"]))


def _current_month() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m")


def _load_google_tts_usage() -> dict:
    try:
        data = json.loads(config.GOOGLE_TTS_USAGE_FILE.read_text())
    except (FileNotFoundError, ValueError):
        data = {}
    if data.get("month") != _current_month():
        return {"month": _current_month(), "characters": 0}
    return data


def _record_google_tts_usage(num_chars: int) -> None:
    usage = _load_google_tts_usage()
    usage["characters"] += num_chars
    config.GOOGLE_TTS_USAGE_FILE.write_text(json.dumps(usage))


class TTSQuotaExceededError(RuntimeError):
    """Raised when a TTS engine's monthly usage guardrail has tripped (or
    would trip), instead of risking a billed API call. Callers in run.py
    catch this specifically to stop a scheduled job cleanly rather than
    crashing with a traceback every day the quota stays exhausted."""


def google_tts_within_cap(num_chars: int) -> bool:
    """True if synthesizing num_chars more characters this month would stay
    within config.GOOGLE_TTS_MONTHLY_CHAR_CAP (minus a fixed safety buffer).
    Always True for non-Google engines, since no cap applies to them."""
    if config.TTS_ENGINE != "google":
        return True
    usable_cap = config.GOOGLE_TTS_MONTHLY_CHAR_CAP - config.GOOGLE_TTS_SAFETY_BUFFER_CHARS
    return _load_google_tts_usage()["characters"] + num_chars <= usable_cap


def ensure_google_tts_budget(total_chars: int) -> None:
    """Pre-flight guardrail: raises TTSQuotaExceededError if synthesizing
    total_chars characters (e.g. a whole video's narration, summed across all
    its scenes) would exceed this month's Google TTS cap. Callers should run
    this BEFORE starting per-scene synthesis, since checking only inside each
    synthesize() call risks a video with some scenes narrated and others not,
    once the cap is crossed mid-video."""
    if not google_tts_within_cap(total_chars):
        raise TTSQuotaExceededError(
            f"Google TTS monthly character cap ({config.GOOGLE_TTS_MONTHLY_CHAR_CAP}) "
            f"would be exceeded by this video's ~{total_chars} narration characters -- "
            "skipping before any synthesis, rendering, or upload work begins."
        )


def synthesize(text: str, out_path: Path, voice: str = None) -> float:
    """Render `text` to an MP3 file at `out_path`. Returns duration in seconds.

    `voice` overrides config.TTS_VOICE for this call (e.g. a per-chat language
    preference from the Telegram bot) — defaults to the configured voice.
    Only applies to the edge-tts engine; the Google engine always uses
    config.GOOGLE_TTS_VOICE, since Google's voice catalog/IDs don't overlap
    with edge-tts's.

    edge-tts talks to an unofficial Microsoft endpoint that occasionally drops
    the connection without sending audio (NoAudioReceived/WebSocketError).
    Reproduced locally: isolated calls fail intermittently (~1 in 10) and
    normally recover on the very next attempt, but on GitHub Actions runners
    a video's 6th-or-so consecutive call would fail and then *every* retry
    for the next ~31s would also fail — consistent with a short burst-rate
    limit on that IP that outlasts our old backoff window. Retry with a
    backoff long enough to ride out that kind of sustained throttle.

    On a sustained block (e.g. a GitHub Actions runner IP getting a run of
    403s), edge-tts's own single built-in retry-after-token-refresh can also
    fail, which raises aiohttp's own exception (e.g. WSServerHandshakeError)
    rather than one of edge_tts's wrapped EdgeTTSException types -- caught
    here too so our longer backoff still applies instead of crashing the run.

    config.TTS_ENGINE == "google" switches to Google Cloud Text-to-Speech's
    officially authenticated API instead -- added because Microsoft's
    edge-tts endpoint blocks GitHub Actions' published IP ranges outright
    (confirmed across both ubuntu-latest and macos-latest runners), which no
    amount of retrying the same blocked network path can fix.

    Google's free tier for the voice used here is 1,000,000 characters/month;
    config.GOOGLE_TTS_MONTHLY_CHAR_CAP keeps a local, git-committed running
    total safely under that so this never silently tips into billed usage --
    raises instead of calling the API once the cap would be exceeded.
    """
    if not FFPROBE_BIN:
        raise RuntimeError("ffprobe not found on PATH. Install ffmpeg first.")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    normalized = _normalize_for_speech(text)

    if config.TTS_ENGINE == "google":
        # Callers (producer.produce_from_script) are expected to call
        # ensure_google_tts_budget() with the whole video's character count
        # before this loop starts -- this per-call check is defense-in-depth
        # in case synthesize() is ever called directly without that preflight.
        ensure_google_tts_budget(len(normalized))
        # Record the attempt before the request, same reasoning as Tavily's
        # guardrail: undercounting here would risk the cap never actually biting.
        _record_google_tts_usage(len(normalized))

    attempts = 8
    for attempt in range(1, attempts + 1):
        try:
            if config.TTS_ENGINE == "google":
                _synthesize_google(normalized, out_path)
            else:
                asyncio.run(_synthesize_async(normalized, voice or config.TTS_VOICE, out_path))
            return _probe_duration(out_path)
        except (edge_tts.exceptions.EdgeTTSException, aiohttp.ClientError, requests.RequestException):
            if attempt == attempts:
                raise
            time.sleep(min(2 ** (attempt - 1), 45))
