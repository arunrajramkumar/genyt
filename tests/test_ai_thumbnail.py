import base64
import json
from unittest import mock

from pipeline import ai_thumbnail, config


def test_generate_hook_background_skips_when_no_company_name(tmp_path):
    result = ai_thumbnail.generate_hook_background("", tmp_path / "bg.png", on_progress=lambda m: None)
    assert result is None


def test_generate_hook_background_writes_image_bytes_on_success(monkeypatch, tmp_path):
    fake_response = mock.Mock()
    fake_response.raise_for_status = lambda: None
    fake_response.headers = {"content-type": "image/jpeg"}
    fake_response.content = b"fake-jpeg-bytes"
    monkeypatch.setattr(ai_thumbnail.requests, "get", lambda *a, **k: fake_response)

    out_path = tmp_path / "bg.png"
    result = ai_thumbnail.generate_hook_background("Arrow Greentech", out_path, on_progress=lambda m: None)

    assert result == out_path
    assert out_path.read_bytes() == b"fake-jpeg-bytes"


def test_generate_hook_background_passes_requested_dimensions(monkeypatch, tmp_path):
    captured = {}

    def fake_get(url, params=None, timeout=None):
        captured["params"] = params
        resp = mock.Mock()
        resp.raise_for_status = lambda: None
        resp.headers = {"content-type": "image/jpeg"}
        resp.content = b"x"
        return resp

    monkeypatch.setattr(ai_thumbnail.requests, "get", fake_get)
    ai_thumbnail.generate_hook_background(
        "Arrow Greentech", tmp_path / "bg.png", on_progress=lambda m: None, width=720, height=1280,
    )
    assert captured["params"]["width"] == 720
    assert captured["params"]["height"] == 1280


def test_generate_hook_background_returns_none_when_request_fails(monkeypatch, tmp_path):
    monkeypatch.setattr(
        ai_thumbnail.requests, "get", mock.Mock(side_effect=RuntimeError("timeout")),
    )
    logs = []
    result = ai_thumbnail.generate_hook_background("Arrow Greentech", tmp_path / "bg.png", on_progress=logs.append)
    assert result is None
    assert any("timeout" in msg for msg in logs)


def test_generate_hook_background_returns_none_when_response_is_not_an_image(monkeypatch, tmp_path):
    fake_response = mock.Mock()
    fake_response.raise_for_status = lambda: None
    fake_response.headers = {"content-type": "text/html"}
    fake_response.content = b"<html>rate limited</html>"
    monkeypatch.setattr(ai_thumbnail.requests, "get", lambda *a, **k: fake_response)

    result = ai_thumbnail.generate_hook_background("Arrow Greentech", tmp_path / "bg.png", on_progress=lambda m: None)
    assert result is None


def test_build_prompt_excludes_text_and_logo_instructions():
    prompt = ai_thumbnail._build_prompt("Arrow Greentech")
    assert "Arrow Greentech" in prompt
    assert "no text" in prompt
    assert "no logos" in prompt


def _gemini_image_response(data=b"fake-png-bytes"):
    resp = mock.Mock()
    resp.raise_for_status = lambda: None
    resp.json.return_value = {
        "candidates": [{"content": {"parts": [
            {"inlineData": {"mimeType": "image/png", "data": base64.b64encode(data).decode()}},
        ]}}],
    }
    return resp


def test_generate_hook_background_skips_gemini_when_api_key_unset(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "GEMINI_API_KEY", "")
    monkeypatch.setattr(ai_thumbnail.requests, "post", mock.Mock(side_effect=AssertionError("should not call Gemini")))
    fake_response = mock.Mock()
    fake_response.raise_for_status = lambda: None
    fake_response.headers = {"content-type": "image/jpeg"}
    fake_response.content = b"fake-jpeg-bytes"
    monkeypatch.setattr(ai_thumbnail.requests, "get", lambda *a, **k: fake_response)

    result = ai_thumbnail.generate_hook_background("Arrow Greentech", tmp_path / "bg.png", on_progress=lambda m: None)
    assert result == tmp_path / "bg.png"


def test_generate_hook_background_uses_gemini_when_configured(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "GEMINI_API_KEY", "dummy")
    monkeypatch.setattr(ai_thumbnail.requests, "post", lambda *a, **k: _gemini_image_response(b"gemini-bytes"))
    monkeypatch.setattr(ai_thumbnail.requests, "get", mock.Mock(side_effect=AssertionError("should not fall back to Pollinations")))

    out_path = tmp_path / "bg.png"
    result = ai_thumbnail.generate_hook_background("Arrow Greentech", out_path, on_progress=lambda m: None)

    assert result == out_path
    assert out_path.read_bytes() == b"gemini-bytes"
    assert ai_thumbnail._load_gemini_image_usage()["requests"] == 1


def test_generate_hook_background_falls_back_to_pollinations_when_gemini_fails(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "GEMINI_API_KEY", "dummy")
    monkeypatch.setattr(ai_thumbnail.requests, "post", mock.Mock(side_effect=RuntimeError("503")))
    fake_response = mock.Mock()
    fake_response.raise_for_status = lambda: None
    fake_response.headers = {"content-type": "image/jpeg"}
    fake_response.content = b"pollinations-bytes"
    monkeypatch.setattr(ai_thumbnail.requests, "get", lambda *a, **k: fake_response)

    out_path = tmp_path / "bg.png"
    result = ai_thumbnail.generate_hook_background("Arrow Greentech", out_path, on_progress=lambda m: None)

    assert result == out_path
    assert out_path.read_bytes() == b"pollinations-bytes"
    assert ai_thumbnail._load_gemini_image_usage()["requests"] == 0


def test_generate_hook_background_falls_back_to_pollinations_when_gemini_over_daily_cap(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "GEMINI_API_KEY", "dummy")
    config.GEMINI_IMAGE_USAGE_FILE.write_text(json.dumps({
        "day": ai_thumbnail._current_day(),
        "requests": config.GEMINI_IMAGE_DAILY_CAP,
    }))
    monkeypatch.setattr(ai_thumbnail.requests, "post", mock.Mock(side_effect=AssertionError("should not call Gemini over cap")))
    fake_response = mock.Mock()
    fake_response.raise_for_status = lambda: None
    fake_response.headers = {"content-type": "image/jpeg"}
    fake_response.content = b"pollinations-bytes"
    monkeypatch.setattr(ai_thumbnail.requests, "get", lambda *a, **k: fake_response)

    out_path = tmp_path / "bg.png"
    result = ai_thumbnail.generate_hook_background("Arrow Greentech", out_path, on_progress=lambda m: None)

    assert result == out_path
    assert out_path.read_bytes() == b"pollinations-bytes"


def test_gemini_image_within_cap_resets_on_a_new_day(isolated_dirs):
    config.GEMINI_IMAGE_USAGE_FILE.write_text(json.dumps({"day": "2020-01-01", "requests": 10**9}))
    assert ai_thumbnail.gemini_image_within_cap() is True


def test_record_gemini_image_usage_increments_requests(isolated_dirs):
    ai_thumbnail._record_gemini_image_usage()
    ai_thumbnail._record_gemini_image_usage()
    assert ai_thumbnail._load_gemini_image_usage()["requests"] == 2
