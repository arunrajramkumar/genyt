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

import edge_tts
import aiohttp

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


def synthesize(text: str, out_path: Path, voice: str = None) -> float:
    """Render `text` to an MP3 file at `out_path`. Returns duration in seconds.

    `voice` overrides config.TTS_VOICE for this call (e.g. a per-chat language
    preference from the Telegram bot) — defaults to the configured voice.

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
    """
    if not FFPROBE_BIN:
        raise RuntimeError("ffprobe not found on PATH. Install ffmpeg first.")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    normalized = _normalize_for_speech(text)

    attempts = 8
    for attempt in range(1, attempts + 1):
        try:
            asyncio.run(_synthesize_async(normalized, voice or config.TTS_VOICE, out_path))
            return _probe_duration(out_path)
        except (edge_tts.exceptions.EdgeTTSException, aiohttp.ClientError):
            if attempt == attempts:
                raise
            time.sleep(min(2 ** (attempt - 1), 45))
