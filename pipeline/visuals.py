"""Fetches stock video clips (falling back to photos) from Pexels for a scene query."""
import hashlib
from pathlib import Path

import requests

from . import config

VIDEO_SEARCH_URL = "https://api.pexels.com/videos/search"
PHOTO_SEARCH_URL = "https://api.pexels.com/v1/search"


def _cache_path(query: str, orientation: str, suffix: str) -> Path:
    key = hashlib.sha1(f"{orientation}:{query}".encode("utf-8")).hexdigest()[:16]
    return config.CACHE_DIR / f"{key}.{suffix}"


def _download(url: str, dest: Path) -> Path:
    resp = requests.get(url, timeout=60)
    resp.raise_for_status()
    dest.write_bytes(resp.content)
    return dest


def fetch_visual(query: str, orientation: str = "landscape", target_width: int = None) -> tuple[Path, str]:
    """Returns (local_file_path, kind) where kind is 'video' or 'image'.

    orientation: "landscape" or "portrait" (portrait for YouTube Shorts / 9:16).
    """
    if not config.PEXELS_API_KEY:
        raise RuntimeError("PEXELS_API_KEY is not set in .env")

    target_width = target_width or config.VIDEO_WIDTH
    headers = {"Authorization": config.PEXELS_API_KEY}

    video_cache = _cache_path(query, orientation, "mp4")
    if video_cache.exists():
        return video_cache, "video"
    image_cache = _cache_path(query, orientation, "jpg")
    if image_cache.exists():
        return image_cache, "image"

    resp = requests.get(
        VIDEO_SEARCH_URL,
        headers=headers,
        params={"query": query, "orientation": orientation, "per_page": 5},
        timeout=30,
    )
    resp.raise_for_status()
    videos = resp.json().get("videos", [])
    if videos:
        files = sorted(
            videos[0]["video_files"],
            key=lambda f: abs((f.get("width") or 0) - target_width),
        )
        best = next((f for f in files if f.get("width", 0) >= 720), files[0])
        return _download(best["link"], video_cache), "video"

    resp = requests.get(
        PHOTO_SEARCH_URL,
        headers=headers,
        params={"query": query, "orientation": orientation, "per_page": 1},
        timeout=30,
    )
    resp.raise_for_status()
    photos = resp.json().get("photos", [])
    if photos:
        url = photos[0]["src"]["large2x"]
        return _download(url, image_cache), "image"

    raise RuntimeError(f"No Pexels results (video or photo) for query: {query!r}")
