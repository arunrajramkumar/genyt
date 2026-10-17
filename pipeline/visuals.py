"""Fetches stock video clips (falling back to photos) from Pexels for a scene query."""
import hashlib
import json
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
# A single education video can easily have 5+ scenes sharing a near-identical
# query (e.g. several "mathematics"/"numbers" scenes in one chapter) -- a
# pool of 4 was getting exhausted mid-video, forcing _pick()'s repeat
# fallback and producing the "same visuals repeating" symptom. 16 (Pexels
# allows per_page up to 80) gives `exclude` much more headroom to avoid
# repeats even in long videos, at the cost of a slightly larger one-time
# download per cache miss -- still a single Pexels API request either way.
POOL_SIZE = 16

# Pexels orders results by relevance, so always requesting page 1 means every
# video asking for the same query (e.g. "stock market chart") draws from the
# exact same top-8 results every other channel doing the same query also gets
# -- a major contributor to finance-channel b-roll looking interchangeable.
# Drawing from a random page multiplies the effective pool ~4x at no extra
# API-call cost (still one search request per cache miss).
SEARCH_PAGE_CHOICES = [1, 2, 3, 4]

# Deliberately NOT persisted across CI runs (e.g. via actions/cache): a cache
# hit skips the network call entirely and reuses whatever page got picked the
# first time, forever -- persisting would freeze each query's pool to its
# first random page instead of letting the page roll fresh (see
# SEARCH_PAGE_CHOICES) on every scheduled run, net *less* variety over time.

# Every scheduled video runs in a fresh CI VM with an empty CACHE_DIR, so the
# pool/exclude machinery above only ever prevents repeats WITHIN one video --
# it has no memory of what earlier videos already showed, and a generic
# fallback query (e.g. "business professional presentation") draws from the
# same small set of Pexels results every single day. config.VISUAL_USAGE_FILE
# is git-committed (unlike CACHE_DIR) specifically so this list survives
# across runs -- see config.py. ~800 entries is roughly 60-80 videos' worth
# of scenes, long enough that a daily upload schedule won't visibly repeat a
# clip within a week or two, without the file growing unbounded forever.
MAX_RECENT_USAGE = 800


def _load_recent_usage() -> list:
    try:
        return json.loads(config.VISUAL_USAGE_FILE.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return []


def _record_usage(asset_key: str) -> None:
    recent = _load_recent_usage()
    recent.append(asset_key)
    config.VISUAL_USAGE_FILE.write_text(json.dumps(recent[-MAX_RECENT_USAGE:]))


def _drop_recently_used(items: list, key_fn, recent: set) -> list:
    """Filters out items whose identity was used in a recent video, falling
    back to the full unfiltered list when every candidate has already been
    seen recently (a thin/niche query can have fewer live results than the
    recent-usage window covers) -- reusing a clip is still better than
    leaving a scene without any visual at all."""
    fresh = [item for item in items if key_fn(item) not in recent]
    return fresh or items


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

    On a fresh (non-cached) search, results are also filtered against
    config.VISUAL_USAGE_FILE's record of assets used in recent PRIOR videos
    (see _drop_recently_used) -- `exclude` alone only prevents repeats within
    one video, which doesn't help since every scheduled video runs in a fresh
    CI VM with no cache, so a generic query would otherwise draw from the same
    handful of Pexels results every single day.
    """
    if not config.PEXELS_API_KEY:
        raise RuntimeError("PEXELS_API_KEY is not set in .env")

    target_width = target_width or config.VIDEO_WIDTH
    headers = {"Authorization": config.PEXELS_API_KEY}
    exclude = exclude or set()
    recent_usage = set(_load_recent_usage())

    video_pool = _pool_paths(query, orientation, "mp4")
    cached_videos = [p for p in video_pool if p.exists()]
    if cached_videos:
        return _pick(cached_videos, exclude), "video"
    image_pool = _pool_paths(query, orientation, "jpg")
    cached_images = [p for p in image_pool if p.exists()]
    if cached_images:
        return _pick(cached_images, exclude), "image"

    page = random.choice(SEARCH_PAGE_CHOICES)
    resp = requests.get(
        VIDEO_SEARCH_URL,
        headers=headers,
        params={"query": query, "orientation": orientation, "per_page": POOL_SIZE, "page": page},
        timeout=30,
    )
    resp.raise_for_status()
    videos = resp.json().get("videos", [])
    if not videos and page != 1:
        # A niche query (e.g. a sector-specific term) may simply not have
        # results that deep -- retry page 1 before giving up on video entirely,
        # so variety-seeking never makes a thin query worse off than before.
        resp = requests.get(
            VIDEO_SEARCH_URL,
            headers=headers,
            params={"query": query, "orientation": orientation, "per_page": POOL_SIZE, "page": 1},
            timeout=30,
        )
        resp.raise_for_status()
        videos = resp.json().get("videos", [])
    videos = _drop_recently_used(videos, lambda v: f"video:{v.get('id')}", recent_usage)
    if videos:
        downloaded = []
        ids = []
        for video, dest in zip(videos[:POOL_SIZE], video_pool):
            files = sorted(
                video["video_files"],
                key=lambda f: abs((f.get("width") or 0) - target_width),
            )
            best = next((f for f in files if f.get("width", 0) >= 720), files[0])
            downloaded.append(_download(best["link"], dest))
            ids.append(video.get("id"))
        chosen_path = _pick(downloaded, exclude)
        _record_usage(f"video:{ids[downloaded.index(chosen_path)]}")
        return chosen_path, "video"

    photo_page = random.choice(SEARCH_PAGE_CHOICES)
    resp = requests.get(
        PHOTO_SEARCH_URL,
        headers=headers,
        params={"query": query, "orientation": orientation, "per_page": POOL_SIZE, "page": photo_page},
        timeout=30,
    )
    resp.raise_for_status()
    photos = resp.json().get("photos", [])
    if not photos and photo_page != 1:
        resp = requests.get(
            PHOTO_SEARCH_URL,
            headers=headers,
            params={"query": query, "orientation": orientation, "per_page": POOL_SIZE, "page": 1},
            timeout=30,
        )
        resp.raise_for_status()
        photos = resp.json().get("photos", [])
    photos = _drop_recently_used(photos, lambda p: f"photo:{p.get('id')}", recent_usage)
    if photos:
        downloaded = []
        ids = []
        for photo, dest in zip(photos[:POOL_SIZE], image_pool):
            downloaded.append(_download(photo["src"]["large2x"], dest))
            ids.append(photo.get("id"))
        chosen_path = _pick(downloaded, exclude)
        _record_usage(f"photo:{ids[downloaded.index(chosen_path)]}")
        return chosen_path, "image"

    raise RuntimeError(f"No Pexels results (video or photo) for query: {query!r}")
