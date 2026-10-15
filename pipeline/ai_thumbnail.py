"""Generates an AI thumbnail/hook-card background image via Google's Imagen
API, when configured — a purpose-built "hookable" visual (e.g. a dramatic
trading-floor scene) reads as more attention-grabbing on a feed than whatever
stock-footage frame happened to get picked, and doesn't depend on Pexels
having a matching clip for an obscure small-cap stock.

Best-effort throughout: a missing key, quota error, or API failure just means
producer.py falls back to its existing Pexels-frame background — never a
blocked video.
"""
import base64
from pathlib import Path

import requests

from . import config

IMAGEN_ENDPOINT = (
    "https://generativelanguage.googleapis.com/v1beta/models/{model}:predict?key={key}"
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


def generate_hook_background(company_name: str, out_path: Path, on_progress=print):
    """Returns `out_path` on success, or None (never raises) if Imagen isn't
    configured or the request fails for any reason."""
    if not config.GOOGLE_IMAGEN_API_KEY:
        on_progress("[ai_thumbnail] GOOGLE_IMAGEN_API_KEY not set; skipping AI thumbnail background")
        return None
    if not company_name:
        on_progress("[ai_thumbnail] no company name available; skipping AI thumbnail background")
        return None

    url = IMAGEN_ENDPOINT.format(model=config.GOOGLE_IMAGEN_MODEL, key=config.GOOGLE_IMAGEN_API_KEY)
    body = {
        "instances": [{"prompt": _build_prompt(company_name)}],
        "parameters": {"sampleCount": 1, "aspectRatio": "9:16"},
    }
    try:
        resp = requests.post(url, json=body, timeout=45)
        resp.raise_for_status()
        predictions = (resp.json() or {}).get("predictions") or []
        image_b64 = predictions[0].get("bytesBase64Encoded") if predictions else None
        if not image_b64:
            on_progress("[ai_thumbnail] Imagen returned no image; falling back to stock-footage background")
            return None
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_bytes(base64.b64decode(image_b64))
        on_progress(f"[ai_thumbnail] generated AI thumbnail background for {company_name!r}")
        return out_path
    except Exception as e:
        on_progress(f"[ai_thumbnail] Imagen request failed ({e}); falling back to stock-footage background")
        return None
