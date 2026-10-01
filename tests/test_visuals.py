from unittest import mock

import pytest
import requests

from pipeline import visuals, config


def test_fetch_visual_raises_without_api_key(no_api_keys):
    with pytest.raises(RuntimeError, match="PEXELS_API_KEY"):
        visuals.fetch_visual("octopus")


def test_fetch_visual_returns_cached_video_without_network_call(isolated_dirs, monkeypatch):
    monkeypatch.setattr(config, "PEXELS_API_KEY", "dummy")
    cache_path = visuals._cache_path("octopus", "landscape", "mp4")
    cache_path.write_bytes(b"fake video bytes")

    def fail_if_called(*a, **k):
        raise AssertionError("should not hit network when cache has a hit")
    monkeypatch.setattr(visuals.requests, "get", fail_if_called)

    path, kind = visuals.fetch_visual("octopus", orientation="landscape")
    assert path == cache_path
    assert kind == "video"


def test_fetch_visual_falls_back_to_cached_image(isolated_dirs, monkeypatch):
    monkeypatch.setattr(config, "PEXELS_API_KEY", "dummy")
    image_cache = visuals._cache_path("octopus", "landscape", "jpg")
    image_cache.write_bytes(b"fake image bytes")

    monkeypatch.setattr(visuals.requests, "get", lambda *a, **k: (_ for _ in ()).throw(AssertionError("no network")))
    path, kind = visuals.fetch_visual("octopus", orientation="landscape")
    assert path == image_cache
    assert kind == "image"


def _fake_get_sequence(responses):
    it = iter(responses)
    def fake_get(url, headers=None, params=None, timeout=None):
        resp = next(it)
        return resp
    return fake_get


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
    video_response = _ok_response({"videos": []})
    photo_response = _ok_response({"photos": [{"src": {"large2x": "https://example.com/photo.jpg"}}]})
    monkeypatch.setattr(visuals.requests, "get", _fake_get_sequence([video_response, photo_response]))
    monkeypatch.setattr(visuals, "_download", lambda url, dest: (dest.write_bytes(b"x"), dest)[1])

    path, kind = visuals.fetch_visual("abstract concept", orientation="portrait")
    assert kind == "image"


def test_fetch_visual_raises_when_nothing_found(isolated_dirs, monkeypatch):
    monkeypatch.setattr(config, "PEXELS_API_KEY", "dummy")
    video_response = _ok_response({"videos": []})
    photo_response = _ok_response({"photos": []})
    monkeypatch.setattr(visuals.requests, "get", _fake_get_sequence([video_response, photo_response]))

    with pytest.raises(RuntimeError, match="No Pexels results"):
        visuals.fetch_visual("a very obscure query", orientation="landscape")
