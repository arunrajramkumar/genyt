"""Fetches stock video clips (falling back to photos) from Pexels for a scene query."""
import hashlib
import random
from pathlib import Path

import requests

from . import config

VIDEO_SEARCH_URL = "https://api.pexels.com/videos/search"
PHOTO_SEARCH_URL = "https://api.pexels.com/v1/search"

# Scenes across different videos often reuse the same generic query (e.g.
# "financial documents calculator" from a fallback) — caching only the single
# top Pexels result per query meant every video with that query got the exact
# same clip forever. Caching a small pool instead and picking randomly from it
# gives visual variety on reuse without paying for a fresh API search each time.
POOL_SIZE = 4


def _cache_path(query: str, orientation: str, suffix: str, index: int = 0) -> Path:
    key = hashlib.sha1(f"{orientation}:{query}".encode("utf-8")).hexdigest()[:16]
    return config.CACHE_DIR / f"{key}_{index}.{suffix}"


def _pool_paths(query: str, orientation: str, suffix: str) -> list[Path]:
    return [_cache_path(query, orientation, suffix, i) for i in range(POOL_SIZE)]


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

    video_pool = _pool_paths(query, orientation, "mp4")
    cached_videos = [p for p in video_pool if p.exists()]
    if cached_videos:
        return random.choice(cached_videos), "video"
    image_pool = _pool_paths(query, orientation, "jpg")
    cached_images = [p for p in image_pool if p.exists()]
    if cached_images:
        return random.choice(cached_images), "image"

    resp = requests.get(
        VIDEO_SEARCH_URL,
        headers=headers,
        params={"query": query, "orientation": orientation, "per_page": POOL_SIZE},
        timeout=30,
    )
    resp.raise_for_status()
    videos = resp.json().get("videos", [])
    if videos:
        downloaded = []
        for video, dest in zip(videos[:POOL_SIZE], video_pool):
            files = sorted(
                video["video_files"],
                key=lambda f: abs((f.get("width") or 0) - target_width),
            )
            best = next((f for f in files if f.get("width", 0) >= 720), files[0])
            downloaded.append(_download(best["link"], dest))
        return random.choice(downloaded), "video"

    resp = requests.get(
        PHOTO_SEARCH_URL,
        headers=headers,
        params={"query": query, "orientation": orientation, "per_page": POOL_SIZE},
        timeout=30,
    )
    resp.raise_for_status()
    photos = resp.json().get("photos", [])
    if photos:
        downloaded = []
        for photo, dest in zip(photos[:POOL_SIZE], image_pool):
            downloaded.append(_download(photo["src"]["large2x"], dest))
        return random.choice(downloaded), "image"

    raise RuntimeError(f"No Pexels results (video or photo) for query: {query!r}")
