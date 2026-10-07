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


def _pick(paths: list[Path], exclude: set) -> Path:
    """Random choice that avoids paths already used elsewhere in the same video
    when possible, falling back to a repeat only if every pooled option has
    already been used (unavoidable when a query's pool is smaller than the
    number of scenes that share it)."""
    unused = [p for p in paths if p not in exclude]
    return random.choice(unused or paths)


def fetch_visual(
    query: str, orientation: str = "landscape", target_width: int = None, exclude: set = None,
) -> tuple[Path, str]:
    """Returns (local_file_path, kind) where kind is 'video' or 'image'.

    orientation: "landscape" or "portrait" (portrait for YouTube Shorts / 9:16).
    exclude: file paths already used by other scenes in the same video — two
    scenes sharing a generic query (e.g. a fallback like "business professional
    presentation") would otherwise have a real chance of landing on the exact
    same clip via random.choice over the same small cached pool.
    """
    if not config.PEXELS_API_KEY:
        raise RuntimeError("PEXELS_API_KEY is not set in .env")

    target_width = target_width or config.VIDEO_WIDTH
    headers = {"Authorization": config.PEXELS_API_KEY}
    exclude = exclude or set()

    video_pool = _pool_paths(query, orientation, "mp4")
    cached_videos = [p for p in video_pool if p.exists()]
    if cached_videos:
        return _pick(cached_videos, exclude), "video"
    image_pool = _pool_paths(query, orientation, "jpg")
    cached_images = [p for p in image_pool if p.exists()]
    if cached_images:
        return _pick(cached_images, exclude), "image"

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
        return _pick(downloaded, exclude), "video"

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
        return _pick(downloaded, exclude), "image"

    raise RuntimeError(f"No Pexels results (video or photo) for query: {query!r}")
