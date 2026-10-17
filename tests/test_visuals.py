import json
import random
from unittest import mock

import pytest
import requests

from pipeline import visuals, config


def test_fetch_visual_raises_without_api_key(no_api_keys):
    with pytest.raises(RuntimeError, match="PEXELS_API_KEY"):
        visuals.fetch_visual("octopus")


def test_fetch_visual_returns_cached_video_without_network_call(isolated_dirs, monkeypatch):
    monkeypatch.setattr(config, "PEXELS_API_KEY", "dummy")
    cache_path = visuals._cache_path("octopus", "landscape", "mp4", 0)
    cache_path.write_bytes(b"fake video bytes")

    def fail_if_called(*a, **k):
        raise AssertionError("should not hit network when cache has a hit")
    monkeypatch.setattr(visuals.requests, "get", fail_if_called)

    path, kind = visuals.fetch_visual("octopus", orientation="landscape")
    assert path == cache_path
    assert kind == "video"


def test_fetch_visual_falls_back_to_cached_image(isolated_dirs, monkeypatch):
    monkeypatch.setattr(config, "PEXELS_API_KEY", "dummy")
    image_cache = visuals._cache_path("octopus", "landscape", "jpg", 0)
    image_cache.write_bytes(b"fake image bytes")

    monkeypatch.setattr(visuals.requests, "get", lambda *a, **k: (_ for _ in ()).throw(AssertionError("no network")))
    path, kind = visuals.fetch_visual("octopus", orientation="landscape")
    assert path == image_cache
    assert kind == "image"


def test_fetch_visual_picks_randomly_among_cached_pool(isolated_dirs, monkeypatch):
    monkeypatch.setattr(config, "PEXELS_API_KEY", "dummy")
    pool = visuals._pool_paths("octopus", "landscape", "mp4")
    for p in pool:
        p.write_bytes(b"fake video bytes")
    monkeypatch.setattr(visuals.requests, "get", lambda *a, **k: (_ for _ in ()).throw(AssertionError("no network")))

    # POOL_SIZE draws uniformly at random: 30 samples made this flaky once
    # POOL_SIZE grew from 4 to 8 (~1-in-5 chance some item never comes up);
    # 200 samples keeps the odds of missing an item astronomically small.
    seen = {visuals.fetch_visual("octopus", orientation="landscape")[0] for _ in range(200)}
    assert seen == set(pool)


def test_fetch_visual_avoids_excluded_paths_when_pool_has_alternatives(isolated_dirs, monkeypatch):
    monkeypatch.setattr(config, "PEXELS_API_KEY", "dummy")
    pool = visuals._pool_paths("octopus", "landscape", "mp4")
    for p in pool:
        p.write_bytes(b"fake video bytes")
    monkeypatch.setattr(visuals.requests, "get", lambda *a, **k: (_ for _ in ()).throw(AssertionError("no network")))

    exclude = set(pool[:-1])
    seen = {visuals.fetch_visual("octopus", orientation="landscape", exclude=exclude)[0] for _ in range(30)}
    assert seen == {pool[-1]}


def test_fetch_visual_repeats_only_when_every_pooled_option_is_excluded(isolated_dirs, monkeypatch):
    monkeypatch.setattr(config, "PEXELS_API_KEY", "dummy")
    pool = visuals._pool_paths("octopus", "landscape", "mp4")
    for p in pool:
        p.write_bytes(b"fake video bytes")
    monkeypatch.setattr(visuals.requests, "get", lambda *a, **k: (_ for _ in ()).throw(AssertionError("no network")))

    path, kind = visuals.fetch_visual("octopus", orientation="landscape", exclude=set(pool))
    assert path in pool
    assert kind == "video"


def _fake_get_sequence(responses):
    it = iter(responses)
    def fake_get(url, headers=None, params=None, timeout=None):
        resp = next(it)
        return resp
    return fake_get


def _fixed_page_then_real_choice(page: int):
    """Pins only the FIRST random.choice call (the page selection) to a fixed
    value, then delegates any later calls (e.g. _pick's pool selection, which
    shares the same random.choice) to the real implementation -- these tests
    care about which page got requested, not which pooled item got picked."""
    real_choice = random.choice
    calls = iter([page])
    def fake_choice(seq):
        try:
            return next(calls)
        except StopIteration:
            return real_choice(seq)
    return fake_choice


def _ok_response(json_body, content=b"bytes"):
    resp = mock.Mock()
    resp.raise_for_status.side_effect = None
    resp.json.return_value = json_body
    resp.content = content
    return resp


def test_fetch_visual_prefers_video_results_closest_to_target_width(isolated_dirs, monkeypatch):
    monkeypatch.setattr(config, "PEXELS_API_KEY", "dummy")
    video_search_response = _ok_response({
        "videos": [{
            "video_files": [
                {"width": 320, "link": "https://example.com/small.mp4"},
                {"width": 1280, "link": "https://example.com/exact.mp4"},
                {"width": 4000, "link": "https://example.com/huge.mp4"},
            ],
        }],
    })
    monkeypatch.setattr(visuals.requests, "get", _fake_get_sequence([video_search_response]))
    monkeypatch.setattr(visuals, "_download", lambda url, dest: (dest.write_bytes(b"x"), dest)[1])

    path, kind = visuals.fetch_visual("factory", orientation="landscape", target_width=1280)
    assert kind == "video"


def test_fetch_visual_falls_back_to_photo_when_no_videos(isolated_dirs, monkeypatch):
    monkeypatch.setattr(config, "PEXELS_API_KEY", "dummy")
    monkeypatch.setattr(visuals.random, "choice", _fixed_page_then_real_choice(1))
    video_response = _ok_response({"videos": []})
    photo_response = _ok_response({"photos": [{"src": {"large2x": "https://example.com/photo.jpg"}}]})
    monkeypatch.setattr(visuals.requests, "get", _fake_get_sequence([video_response, photo_response]))
    monkeypatch.setattr(visuals, "_download", lambda url, dest: (dest.write_bytes(b"x"), dest)[1])

    path, kind = visuals.fetch_visual("abstract concept", orientation="portrait")
    assert kind == "image"


def test_fetch_visual_raises_when_nothing_found(isolated_dirs, monkeypatch):
    monkeypatch.setattr(config, "PEXELS_API_KEY", "dummy")
    monkeypatch.setattr(visuals.random, "choice", lambda choices: 1)
    video_response = _ok_response({"videos": []})
    photo_response = _ok_response({"photos": []})
    monkeypatch.setattr(visuals.requests, "get", _fake_get_sequence([video_response, photo_response]))

    with pytest.raises(RuntimeError, match="No Pexels results"):
        visuals.fetch_visual("a very obscure query", orientation="landscape")


def test_fetch_visual_uses_a_random_page_to_diversify_results(isolated_dirs, monkeypatch):
    # Pexels orders by relevance, so always requesting page 1 means every
    # video asking for the same query draws from the same top-8 results as
    # every other channel doing the same query -- a random page multiplies
    # the effective pool for free.
    monkeypatch.setattr(config, "PEXELS_API_KEY", "dummy")
    monkeypatch.setattr(visuals.random, "choice", _fixed_page_then_real_choice(3))
    video_search_response = _ok_response({
        "videos": [{"video_files": [{"width": 1280, "link": "https://example.com/clip.mp4"}]}],
    })
    seen_params = []

    def fake_get(url, headers=None, params=None, timeout=None):
        seen_params.append(params)
        return video_search_response
    monkeypatch.setattr(visuals.requests, "get", fake_get)
    monkeypatch.setattr(visuals, "_download", lambda url, dest: (dest.write_bytes(b"x"), dest)[1])

    visuals.fetch_visual("stock market chart", orientation="landscape")
    assert seen_params[0]["page"] == 3


def test_fetch_visual_retries_page_one_when_random_page_has_no_videos(isolated_dirs, monkeypatch):
    # A niche/sector-specific query may not have results as deep as page 3+
    # -- retrying page 1 before falling back to photos keeps variety-seeking
    # from making a thin query worse off than before.
    monkeypatch.setattr(config, "PEXELS_API_KEY", "dummy")
    monkeypatch.setattr(visuals.random, "choice", _fixed_page_then_real_choice(3))
    empty_page = _ok_response({"videos": []})
    page_one_response = _ok_response({
        "videos": [{"video_files": [{"width": 1280, "link": "https://example.com/clip.mp4"}]}],
    })
    monkeypatch.setattr(visuals.requests, "get", _fake_get_sequence([empty_page, page_one_response]))
    monkeypatch.setattr(visuals, "_download", lambda url, dest: (dest.write_bytes(b"x"), dest)[1])

    path, kind = visuals.fetch_visual("niche sector term", orientation="landscape")
    assert kind == "video"


def test_fetch_visual_seeds_a_pool_of_distinct_clips_for_variety_on_reuse(isolated_dirs, monkeypatch):
    monkeypatch.setattr(config, "PEXELS_API_KEY", "dummy")
    video_search_response = _ok_response({
        "videos": [
            {"video_files": [{"width": 1280, "link": f"https://example.com/clip{i}.mp4"}]}
            for i in range(visuals.POOL_SIZE)
        ],
    })
    monkeypatch.setattr(visuals.requests, "get", _fake_get_sequence([video_search_response]))
    downloaded_urls = []

    def fake_download(url, dest):
        downloaded_urls.append(url)
        dest.write_bytes(b"x")
        return dest

    monkeypatch.setattr(visuals, "_download", fake_download)

    visuals.fetch_visual("recurring generic query", orientation="landscape", target_width=1280)

    assert len(set(downloaded_urls)) == visuals.POOL_SIZE
    pool = visuals._pool_paths("recurring generic query", "landscape", "mp4")
    assert all(p.exists() for p in pool)


# -- cross-video dedup: config.VISUAL_USAGE_FILE -------------------------------
#
# Every scheduled video runs in a fresh CI VM with an empty cache, so the pool/
# exclude machinery above only ever stops repeats WITHIN one video -- these
# tests cover the separate, persisted mechanism that avoids repeating a clip
# shown in a recent PRIOR video (see pipeline/config.py's VISUAL_USAGE_FILE).

def _video_response_with_ids(ids):
    return _ok_response({
        "videos": [
            {"id": i, "video_files": [{"width": 1280, "link": f"https://example.com/clip{i}.mp4"}]}
            for i in ids
        ],
    })


def test_fetch_visual_avoids_a_video_id_used_in_a_recent_prior_video(isolated_dirs, monkeypatch):
    monkeypatch.setattr(config, "PEXELS_API_KEY", "dummy")
    config.VISUAL_USAGE_FILE.write_text('["video:1"]')
    monkeypatch.setattr(visuals.requests, "get", _fake_get_sequence([_video_response_with_ids([1, 2])]))
    monkeypatch.setattr(visuals, "_download", lambda url, dest: (dest.write_bytes(b"x"), dest)[1])

    seen_urls = {visuals.fetch_visual("recurring generic query", orientation="landscape")[0] for _ in range(30)}
    assert seen_urls == {visuals._pool_paths("recurring generic query", "landscape", "mp4")[0]}


def test_fetch_visual_records_the_chosen_asset_as_recently_used(isolated_dirs, monkeypatch):
    monkeypatch.setattr(config, "PEXELS_API_KEY", "dummy")
    monkeypatch.setattr(visuals.requests, "get", _fake_get_sequence([_video_response_with_ids([1])]))
    monkeypatch.setattr(visuals, "_download", lambda url, dest: (dest.write_bytes(b"x"), dest)[1])

    visuals.fetch_visual("recurring generic query", orientation="landscape")
    assert visuals._load_recent_usage() == ["video:1"]


def test_fetch_visual_reuses_a_clip_when_every_result_was_recently_used(isolated_dirs, monkeypatch):
    # A thin/niche query can have fewer live Pexels results than the
    # recent-usage window covers -- falling back to a repeat is still better
    # than leaving a scene without any visual at all.
    monkeypatch.setattr(config, "PEXELS_API_KEY", "dummy")
    config.VISUAL_USAGE_FILE.write_text('["video:1", "video:2"]')
    monkeypatch.setattr(visuals.requests, "get", _fake_get_sequence([_video_response_with_ids([1, 2])]))
    monkeypatch.setattr(visuals, "_download", lambda url, dest: (dest.write_bytes(b"x"), dest)[1])

    path, kind = visuals.fetch_visual("recurring generic query", orientation="landscape")
    assert kind == "video"


def test_record_usage_trims_to_the_max_recent_window(isolated_dirs):
    config.VISUAL_USAGE_FILE.write_text(json.dumps([f"video:{i}" for i in range(visuals.MAX_RECENT_USAGE)]))
    visuals._record_usage("video:new")
    recent = visuals._load_recent_usage()
    assert len(recent) == visuals.MAX_RECENT_USAGE
    assert recent[-1] == "video:new"
    assert "video:0" not in recent
