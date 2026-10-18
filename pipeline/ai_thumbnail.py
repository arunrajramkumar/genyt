"""Generates a free AI thumbnail/hook-card background image — a purpose-built
"hookable" visual (e.g. a dramatic trading-floor scene) reads as more
attention-grabbing on a feed than whatever stock-footage frame happened to get
picked, and doesn't depend on Pexels having a matching clip for an obscure
small-cap stock.

Primary generator: Gemini 2.5 Flash Image ("Nano Banana"), Google AI Studio's
free-tier image model (set GEMINI_API_KEY) — it renders legible on-screen
text/numbers far more reliably than Pollinations.ai, and is called via plain
`requests` (no new SDK/pip dependency), the same way pipeline/tts.py calls
Google Cloud TTS. A git-committed daily usage counter
(config.GEMINI_IMAGE_USAGE_FILE) is checked BEFORE each call so a run
proactively falls back to Pollinations once the free daily allowance is used
up, instead of only reacting to a 429.

Fallback generator: Pollinations.ai — needs no API key/account/billing, just an
unauthenticated GET request that returns the image bytes directly. Used
whenever Gemini is unconfigured, over its daily cap, or fails outright.

Both are public, best-effort services with no SLA, so this stays best-effort
throughout: any failure in either tier just means producer.py falls back to
its existing Pexels-frame background — never a blocked video.
"""
import base64
import json
import urllib.parse
from datetime import datetime, timezone
from pathlib import Path

import requests

from . import config

POLLINATIONS_ENDPOINT = "https://image.pollinations.ai/prompt/{prompt}"
GEMINI_GENERATE_ENDPOINT = (
    "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
)


def _build_prompt(company_name: str) -> str:
    """A generic "hookable" finance-thumbnail scene. Deliberately avoids
    asking for any text/numbers/logos — text-to-image models render legible
    text unreliably, and the bold stock name/stat is already burned in
    separately by textcard.render_hook_card on top of this background."""
    return (
        f"A dramatic, cinematic photo evoking {company_name}'s business and the stock "
        "market — a dynamic trading floor, stock chart on a big screen, or relevant "
        "industrial/corporate scene. Bold lighting, high contrast, vertical 9:16 "
        "composition, no text, no logos, no watermarks, no numbers, photorealistic "
        "financial news thumbnail style."
    )


def _current_day() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")


def _load_gemini_image_usage() -> dict:
    try:
        data = json.loads(config.GEMINI_IMAGE_USAGE_FILE.read_text())
    except (FileNotFoundError, ValueError):
        data = {}
    if data.get("day") != _current_day():
        return {"day": _current_day(), "requests": 0}
    return data


def _record_gemini_image_usage() -> None:
    usage = _load_gemini_image_usage()
    usage["requests"] += 1
    config.GEMINI_IMAGE_USAGE_FILE.write_text(json.dumps(usage))


def gemini_image_within_cap() -> bool:
    """True if one more Gemini image request today would stay within
    config.GEMINI_IMAGE_DAILY_CAP (minus a fixed safety buffer)."""
    usable_cap = config.GEMINI_IMAGE_DAILY_CAP - config.GEMINI_IMAGE_SAFETY_BUFFER
    return _load_gemini_image_usage()["requests"] < usable_cap


def _generate_via_gemini(company_name: str, out_path: Path, on_progress, width: int, height: int):
    """Returns out_path on success, None on any failure (never raises)."""
    aspect_ratio = "9:16" if height >= width else "16:9"
    try:
        response = requests.post(
            GEMINI_GENERATE_ENDPOINT.format(model=config.GEMINI_IMAGE_MODEL),
            params={"key": config.GEMINI_API_KEY},
            json={
                "contents": [{"parts": [{"text": _build_prompt(company_name)}]}],
                "generationConfig": {
                    "responseModalities": ["IMAGE"],
                    "imageConfig": {"aspectRatio": aspect_ratio},
                },
            },
            timeout=60,
        )
        response.raise_for_status()
        candidates = response.json().get("candidates", [])
        for part in candidates[0]["content"]["parts"]:
            inline_data = part.get("inlineData") or part.get("inline_data")
            if inline_data and inline_data.get("data"):
                out_path.parent.mkdir(parents=True, exist_ok=True)
                out_path.write_bytes(base64.b64decode(inline_data["data"]))
                _record_gemini_image_usage()
                on_progress(f"[ai_thumbnail] generated Gemini thumbnail background for {company_name!r}")
                return out_path
        on_progress("[ai_thumbnail] Gemini returned no image data; falling back to Pollinations")
        return None
    except Exception as e:
        on_progress(f"[ai_thumbnail] Gemini thumbnail request failed ({e}); falling back to Pollinations")
        return None


def _generate_via_pollinations(company_name: str, out_path: Path, on_progress, width: int, height: int):
    """Returns out_path on success, None on any failure (never raises)."""
    prompt = urllib.parse.quote(_build_prompt(company_name))
    url = POLLINATIONS_ENDPOINT.format(prompt=prompt)
    params = {"width": width, "height": height, "nologo": "true"}
    try:
        resp = requests.get(url, params=params, timeout=45)
        resp.raise_for_status()
        content_type = resp.headers.get("content-type", "").split(";")[0].strip()
        if not resp.content or not content_type.startswith("image/"):
            on_progress("[ai_thumbnail] Pollinations returned no usable image; falling back to stock-footage background")
            return None
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_bytes(resp.content)
        on_progress(f"[ai_thumbnail] generated Pollinations thumbnail background for {company_name!r}")
        return out_path
    except Exception as e:
        on_progress(f"[ai_thumbnail] Pollinations thumbnail request failed ({e}); falling back to stock-footage background")
        return None


def generate_hook_background(company_name: str, out_path: Path, on_progress=print,
                              width: int = 720, height: int = 1280):
    """Returns `out_path` on success, or None (never raises) if there's no
    company name to work with or every generator tier fails.

    Tries Gemini 2.5 Flash Image first (if GEMINI_API_KEY is configured and
    today's usage is under its daily cap), then falls back to Pollinations.ai,
    then finally lets the caller fall back to a Pexels stock-footage frame."""
    if not company_name:
        on_progress("[ai_thumbnail] no company name available; skipping AI thumbnail background")
        return None

    if config.GEMINI_API_KEY:
        if gemini_image_within_cap():
            result = _generate_via_gemini(company_name, out_path, on_progress, width, height)
            if result:
                return result
        else:
            on_progress("[ai_thumbnail] Gemini daily image cap reached; using Pollinations fallback")

    return _generate_via_pollinations(company_name, out_path, on_progress, width, height)
