"""Generates a free AI thumbnail/hook-card background image via Pollinations.ai
— a purpose-built "hookable" visual (e.g. a dramatic trading-floor scene)
reads as more attention-grabbing on a feed than whatever stock-footage frame
happened to get picked, and doesn't depend on Pexels having a matching clip
for an obscure small-cap stock.

Pollinations.ai needs no API key/account/billing (unlike Google Imagen,
OpenAI, etc.) — just an unauthenticated GET request that returns the image
bytes directly. It's a public, best-effort service with no SLA, so this is
still wired as best-effort throughout: any network/rate-limit/content
failure just means producer.py falls back to its existing Pexels-frame
background — never a blocked video.
"""
import urllib.parse
from pathlib import Path

import requests

POLLINATIONS_ENDPOINT = "https://image.pollinations.ai/prompt/{prompt}"


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


def generate_hook_background(company_name: str, out_path: Path, on_progress=print,
                              width: int = 720, height: int = 1280):
    """Returns `out_path` on success, or None (never raises) if there's no
    company name to work with or the request fails for any reason."""
    if not company_name:
        on_progress("[ai_thumbnail] no company name available; skipping AI thumbnail background")
        return None

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
        on_progress(f"[ai_thumbnail] generated AI thumbnail background for {company_name!r}")
        return out_path
    except Exception as e:
        on_progress(f"[ai_thumbnail] AI thumbnail request failed ({e}); falling back to stock-footage background")
        return None
