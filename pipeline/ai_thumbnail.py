"""Generates a free AI thumbnail/hook-card background image — a purpose-built
"hookable" visual (e.g. a dramatic trading-floor scene) reads as more
attention-grabbing on a feed than whatever stock-footage frame happened to get
picked, and doesn't depend on Pexels having a matching clip for an obscure
small-cap stock.

Three tiers, tried in order, each never raising (always returns None on any
failure so the next tier/caller can take over):

1. Gemini 2.5 Flash Image ("Nano Banana") — Google AI Studio's free-tier image
   model (set GEMINI_API_KEY). Renders legible on-screen text/numbers far more
   reliably than the other two tiers.
2. Cloudflare Workers AI (set CLOUDFLARE_ACCOUNT_ID + CLOUDFLARE_API_TOKEN) —
   a genuinely free, no-credit-card-required daily allowance, used when Gemini
   is unconfigured, over its daily cap, or fails outright.
3. Pollinations.ai — needs no API key/account/billing at all, just an
   unauthenticated GET request. Always available as the last resort.

All three are called via plain `requests` (no new SDK/pip dependency), the
same way pipeline/tts.py calls Google Cloud TTS. Tiers 1 and 2 each check a
git-committed daily usage counter BEFORE calling, so a run proactively moves
to the next tier once that day's free allowance is used up, instead of only
reacting to a 429 -- and a 429 itself still immediately maxes out that
counter, so a same-day retry doesn't repeat a call already known to fail.

All three are public/free services with no real SLA, so this stays
best-effort throughout: any failure cascades down, and failure of all three
just means producer.py falls back to its existing Pexels-frame background —
never a blocked video.
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
CLOUDFLARE_RUN_ENDPOINT = (
    "https://api.cloudflare.com/client/v4/accounts/{account_id}/ai/run/{model}"
)


# company name -> one-sentence description of what it does, set by
# generate_hook_background for the duration of one generation.
_BUSINESS_HINTS = {}


def _build_prompt(company_name: str) -> str:
    """A "hookable" finance-thumbnail scene. Deliberately avoids asking for any
    text/numbers/logos — text-to-image models render legible text unreliably,
    and the bold stock name/stat is already burned in separately by textcard
    on top of this background.

    With a business description it shows the company's real operations
    (a zinc smelter, a solar panel line, a jewellery showroom): every video
    otherwise got the same generic trading-floor image, which made the
    channel's thumbnails indistinguishable from each other and from every
    other finance channel's."""
    business = _BUSINESS_HINTS.get(company_name)
    if business:
        return (
            f"A dramatic, cinematic photo of the real-world operations of {company_name}. "
            f"About the company: {business} Show what it really does — its products, plants, "
            "mines, stores, branches, or customers using its service (for a lender or "
            "insurer: people, homes, shops and vehicles it finances, not factories) — not a "
            "stock chart or trading floor. Bold lighting, high contrast, rich "
            "colour, vertical 9:16 composition with the main subject in the middle third, "
            "no text, no logos, no watermarks, no numbers, photorealistic."
        )
    return (
        f"A dramatic, cinematic photo evoking {company_name}'s business and the stock "
        "market — a dynamic trading floor, stock chart on a big screen, or relevant "
        "industrial/corporate scene. Bold lighting, high contrast, vertical 9:16 "
        "composition, no text, no logos, no watermarks, no numbers, photorealistic "
        "financial news thumbnail style."
    )


def _current_day() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")


def _load_usage(usage_file: Path) -> dict:
    try:
        data = json.loads(usage_file.read_text())
    except (FileNotFoundError, ValueError):
        data = {}
    if data.get("day") != _current_day():
        return {"day": _current_day(), "requests": 0}
    return data


def _record_usage(usage_file: Path) -> None:
    usage = _load_usage(usage_file)
    usage["requests"] += 1
    usage_file.write_text(json.dumps(usage))


def _record_quota_exhausted(usage_file: Path, daily_cap: int) -> None:
    """Called when a provider itself returns 429 (quota/rate-limit exceeded)
    -- bumps today's local counter straight to the cap so any later call this
    same day (e.g. a manual workflow re-run, or a per-channel run that
    generates more than one thumbnail) skips this tier instead of making a
    call already known to fail."""
    usage = _load_usage(usage_file)
    usage["requests"] = max(usage["requests"], daily_cap)
    usage_file.write_text(json.dumps(usage))


def _within_cap(usage_file: Path, daily_cap: int, safety_buffer: int) -> bool:
    usable_cap = daily_cap - safety_buffer
    return _load_usage(usage_file)["requests"] < usable_cap


def _load_gemini_image_usage() -> dict:
    return _load_usage(config.GEMINI_IMAGE_USAGE_FILE)


def _record_gemini_image_usage() -> None:
    _record_usage(config.GEMINI_IMAGE_USAGE_FILE)


def _record_gemini_quota_exhausted() -> None:
    _record_quota_exhausted(config.GEMINI_IMAGE_USAGE_FILE, config.GEMINI_IMAGE_DAILY_CAP)


def gemini_image_within_cap() -> bool:
    """True if one more Gemini image request today would stay within
    config.GEMINI_IMAGE_DAILY_CAP (minus a fixed safety buffer)."""
    return _within_cap(config.GEMINI_IMAGE_USAGE_FILE, config.GEMINI_IMAGE_DAILY_CAP, config.GEMINI_IMAGE_SAFETY_BUFFER)


def _load_cloudflare_image_usage() -> dict:
    return _load_usage(config.CLOUDFLARE_IMAGE_USAGE_FILE)


def _record_cloudflare_image_usage() -> None:
    _record_usage(config.CLOUDFLARE_IMAGE_USAGE_FILE)


def _record_cloudflare_quota_exhausted() -> None:
    _record_quota_exhausted(config.CLOUDFLARE_IMAGE_USAGE_FILE, config.CLOUDFLARE_IMAGE_DAILY_CAP)


def cloudflare_image_within_cap() -> bool:
    """True if one more Cloudflare Workers AI image request today would stay
    within config.CLOUDFLARE_IMAGE_DAILY_CAP (minus a fixed safety buffer)."""
    return _within_cap(config.CLOUDFLARE_IMAGE_USAGE_FILE, config.CLOUDFLARE_IMAGE_DAILY_CAP, config.CLOUDFLARE_IMAGE_SAFETY_BUFFER)


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
        on_progress("[ai_thumbnail] Gemini returned no image data; trying next fallback")
        return None
    except requests.HTTPError as e:
        status = e.response.status_code if e.response is not None else None
        if status in (429, 402):
            # 429 = real rate/quota limit hit. 402 = Google gating this model
            # behind billing being enabled on the project, independent of the
            # daily free-tier cap -- in both cases, retrying again today is
            # known to fail, so skip straight to the next tier for the rest
            # of the day instead of wasting another call per thumbnail.
            _record_gemini_quota_exhausted()
        on_progress(f"[ai_thumbnail] Gemini thumbnail request failed ({e}); trying next fallback")
        return None
    except Exception as e:
        on_progress(f"[ai_thumbnail] Gemini thumbnail request failed ({e}); trying next fallback")
        return None


def _generate_via_cloudflare(company_name: str, out_path: Path, on_progress, width: int, height: int):
    """Returns out_path on success, None on any failure (never raises).

    Workers AI's response shape varies by model: Stable Diffusion-family
    models return the image as raw bytes (content-type image/*), while
    FLUX-family models wrap a base64-encoded image in a JSON envelope
    ({"result": {"image": "..."}}) -- handled generically here so swapping
    config.CLOUDFLARE_IMAGE_MODEL doesn't require a code change."""
    try:
        response = requests.post(
            CLOUDFLARE_RUN_ENDPOINT.format(
                account_id=config.CLOUDFLARE_ACCOUNT_ID, model=config.CLOUDFLARE_IMAGE_MODEL,
            ),
            headers={"Authorization": f"Bearer {config.CLOUDFLARE_API_TOKEN}"},
            json={"prompt": _build_prompt(company_name)},
            timeout=60,
        )
        response.raise_for_status()
        content_type = response.headers.get("content-type", "").split(";")[0].strip()
        if content_type.startswith("image/"):
            image_bytes = response.content
        else:
            image_b64 = response.json().get("result", {}).get("image")
            if not image_b64:
                on_progress("[ai_thumbnail] Cloudflare Workers AI returned no image data; trying next fallback")
                return None
            image_bytes = base64.b64decode(image_b64)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_bytes(image_bytes)
        _record_cloudflare_image_usage()
        on_progress(f"[ai_thumbnail] generated Cloudflare Workers AI thumbnail background for {company_name!r}")
        return out_path
    except requests.HTTPError as e:
        if e.response is not None and e.response.status_code == 429:
            _record_cloudflare_quota_exhausted()
        on_progress(f"[ai_thumbnail] Cloudflare Workers AI request failed ({e}); trying next fallback")
        return None
    except Exception as e:
        on_progress(f"[ai_thumbnail] Cloudflare Workers AI request failed ({e}); trying next fallback")
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
                              width: int = 720, height: int = 1280, business_hint: str = ""):
    """Returns `out_path` on success, or None (never raises) if there's no
    company name to work with or every generator tier fails.

    Tries Gemini 2.5 Flash Image first (if GEMINI_API_KEY is configured and
    today's usage is under its daily cap), then Cloudflare Workers AI (if
    configured and under cap), then Pollinations.ai, then finally lets the
    caller fall back to a Pexels stock-footage frame."""
    if not company_name:
        on_progress("[ai_thumbnail] no company name available; skipping AI thumbnail background")
        return None
    _BUSINESS_HINTS.clear()
    if business_hint:
        _BUSINESS_HINTS[company_name] = business_hint

    if config.GEMINI_API_KEY:
        if gemini_image_within_cap():
            result = _generate_via_gemini(company_name, out_path, on_progress, width, height)
            if result:
                return result
        else:
            on_progress("[ai_thumbnail] Gemini daily image cap reached; trying next fallback")

    if config.CLOUDFLARE_ACCOUNT_ID and config.CLOUDFLARE_API_TOKEN:
        if cloudflare_image_within_cap():
            result = _generate_via_cloudflare(company_name, out_path, on_progress, width, height)
            if result:
                return result
        else:
            on_progress("[ai_thumbnail] Cloudflare Workers AI daily image cap reached; using Pollinations fallback")

    return _generate_via_pollinations(company_name, out_path, on_progress, width, height)
