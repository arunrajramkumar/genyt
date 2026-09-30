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

FFPROBE_BIN = shutil.which("ffprobe")

# Abbreviations that TTS engines tend to spell out letter-by-letter instead of
# reading as words. Expanded before synthesis so narration sounds natural.
_ABBREVIATIONS = [
    (r"\bLtd\.?", "Limited"),
    (r"\bPvt\.?", "Private"),
    (r"\bCorp\.?", "Corporation"),
    (r"\bInc\.?", "Incorporated"),
    (r"\bCo\.", "Company"),
    (r"\bP/E\b", "price to earnings"),
    (r"\bYoY\b", "year over year"),
    (r"\bQoQ\b", "quarter over quarter"),
    (r"\bRs\.?(?=\s?\d)", "Rupees"),
    (r"\bCr\.?\b", "Crore"),
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


def synthesize(text: str, out_path: Path) -> float:
    """Render `text` to an MP3 file at `out_path`. Returns duration in seconds."""
    if not FFPROBE_BIN:
        raise RuntimeError("ffprobe not found on PATH. Install ffmpeg first.")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    asyncio.run(_synthesize_async(_normalize_for_speech(text), config.TTS_VOICE, out_path))
    return _probe_duration(out_path)
