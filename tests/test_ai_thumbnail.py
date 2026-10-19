import base64
import json
from unittest import mock

import requests

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


def test_generate_hook_background_marks_quota_exhausted_on_429_so_same_day_retries_skip_gemini(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "GEMINI_API_KEY", "dummy")
    rate_limited = mock.Mock()
    http_error = requests.HTTPError("429 Client Error: Too Many Requests")
    http_error.response = mock.Mock(status_code=429)
    rate_limited.raise_for_status.side_effect = http_error
    monkeypatch.setattr(ai_thumbnail.requests, "post", lambda *a, **k: rate_limited)
    fake_response = mock.Mock()
    fake_response.raise_for_status = lambda: None
    fake_response.headers = {"content-type": "image/jpeg"}
    fake_response.content = b"pollinations-bytes"
    monkeypatch.setattr(ai_thumbnail.requests, "get", lambda *a, **k: fake_response)

    ai_thumbnail.generate_hook_background("Arrow Greentech", tmp_path / "bg.png", on_progress=lambda m: None)
    assert ai_thumbnail._load_gemini_image_usage()["requests"] == config.GEMINI_IMAGE_DAILY_CAP
    assert ai_thumbnail.gemini_image_within_cap() is False

    # A second call the same day must skip straight to Pollinations without
    # hitting Gemini again.
    monkeypatch.setattr(ai_thumbnail.requests, "post", mock.Mock(side_effect=AssertionError("should not retry Gemini")))
    result = ai_thumbnail.generate_hook_background("Arrow Greentech", tmp_path / "bg2.png", on_progress=lambda m: None)
    assert result == tmp_path / "bg2.png"


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


# -- Cloudflare Workers AI: second-tier fallback (Gemini -> Cloudflare -> Pollinations) --

def _cloudflare_raw_image_response(data=b"fake-sdxl-bytes"):
    resp = mock.Mock()
    resp.raise_for_status = lambda: None
    resp.headers = {"content-type": "image/png"}
    resp.content = data
    return resp


def _cloudflare_json_image_response(data=b"fake-flux-bytes"):
    resp = mock.Mock()
    resp.raise_for_status = lambda: None
    resp.headers = {"content-type": "application/json"}
    resp.json.return_value = {"result": {"image": base64.b64encode(data).decode()}, "success": True}
    return resp


def test_generate_hook_background_skips_cloudflare_when_unconfigured(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "GEMINI_API_KEY", "")
    monkeypatch.setattr(config, "CLOUDFLARE_ACCOUNT_ID", "")
    monkeypatch.setattr(config, "CLOUDFLARE_API_TOKEN", "")
    monkeypatch.setattr(ai_thumbnail.requests, "post", mock.Mock(side_effect=AssertionError("should not call Cloudflare")))
    fake_response = mock.Mock()
    fake_response.raise_for_status = lambda: None
    fake_response.headers = {"content-type": "image/jpeg"}
    fake_response.content = b"fake-jpeg-bytes"
    monkeypatch.setattr(ai_thumbnail.requests, "get", lambda *a, **k: fake_response)

    result = ai_thumbnail.generate_hook_background("Arrow Greentech", tmp_path / "bg.png", on_progress=lambda m: None)
    assert result == tmp_path / "bg.png"


def test_generate_hook_background_uses_cloudflare_when_gemini_unconfigured(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "GEMINI_API_KEY", "")
    monkeypatch.setattr(config, "CLOUDFLARE_ACCOUNT_ID", "acct123")
    monkeypatch.setattr(config, "CLOUDFLARE_API_TOKEN", "dummy")
    monkeypatch.setattr(ai_thumbnail.requests, "post", lambda *a, **k: _cloudflare_raw_image_response(b"sdxl-bytes"))
    monkeypatch.setattr(ai_thumbnail.requests, "get", mock.Mock(side_effect=AssertionError("should not fall back to Pollinations")))

    out_path = tmp_path / "bg.png"
    result = ai_thumbnail.generate_hook_background("Arrow Greentech", out_path, on_progress=lambda m: None)

    assert result == out_path
    assert out_path.read_bytes() == b"sdxl-bytes"
    assert ai_thumbnail._load_cloudflare_image_usage()["requests"] == 1


def test_generate_hook_background_parses_json_wrapped_base64_image(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "GEMINI_API_KEY", "")
    monkeypatch.setattr(config, "CLOUDFLARE_ACCOUNT_ID", "acct123")
    monkeypatch.setattr(config, "CLOUDFLARE_API_TOKEN", "dummy")
    monkeypatch.setattr(ai_thumbnail.requests, "post", lambda *a, **k: _cloudflare_json_image_response(b"flux-bytes"))

    out_path = tmp_path / "bg.png"
    result = ai_thumbnail.generate_hook_background("Arrow Greentech", out_path, on_progress=lambda m: None)

    assert result == out_path
    assert out_path.read_bytes() == b"flux-bytes"


def test_generate_hook_background_falls_back_to_pollinations_when_cloudflare_fails(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "GEMINI_API_KEY", "")
    monkeypatch.setattr(config, "CLOUDFLARE_ACCOUNT_ID", "acct123")
    monkeypatch.setattr(config, "CLOUDFLARE_API_TOKEN", "dummy")
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
    assert ai_thumbnail._load_cloudflare_image_usage()["requests"] == 0


def test_generate_hook_background_marks_cloudflare_quota_exhausted_on_429(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "GEMINI_API_KEY", "")
    monkeypatch.setattr(config, "CLOUDFLARE_ACCOUNT_ID", "acct123")
    monkeypatch.setattr(config, "CLOUDFLARE_API_TOKEN", "dummy")
    rate_limited = mock.Mock()
    http_error = requests.HTTPError("429 Client Error: Too Many Requests")
    http_error.response = mock.Mock(status_code=429)
    rate_limited.raise_for_status.side_effect = http_error
    monkeypatch.setattr(ai_thumbnail.requests, "post", lambda *a, **k: rate_limited)
    fake_response = mock.Mock()
    fake_response.raise_for_status = lambda: None
    fake_response.headers = {"content-type": "image/jpeg"}
    fake_response.content = b"pollinations-bytes"
    monkeypatch.setattr(ai_thumbnail.requests, "get", lambda *a, **k: fake_response)

    ai_thumbnail.generate_hook_background("Arrow Greentech", tmp_path / "bg.png", on_progress=lambda m: None)
    assert ai_thumbnail._load_cloudflare_image_usage()["requests"] == config.CLOUDFLARE_IMAGE_DAILY_CAP
    assert ai_thumbnail.cloudflare_image_within_cap() is False


def test_generate_hook_background_falls_back_to_pollinations_when_cloudflare_over_daily_cap(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "GEMINI_API_KEY", "")
    monkeypatch.setattr(config, "CLOUDFLARE_ACCOUNT_ID", "acct123")
    monkeypatch.setattr(config, "CLOUDFLARE_API_TOKEN", "dummy")
    config.CLOUDFLARE_IMAGE_USAGE_FILE.write_text(json.dumps({
        "day": ai_thumbnail._current_day(),
        "requests": config.CLOUDFLARE_IMAGE_DAILY_CAP,
    }))
    monkeypatch.setattr(ai_thumbnail.requests, "post", mock.Mock(side_effect=AssertionError("should not call Cloudflare over cap")))
    fake_response = mock.Mock()
    fake_response.raise_for_status = lambda: None
    fake_response.headers = {"content-type": "image/jpeg"}
    fake_response.content = b"pollinations-bytes"
    monkeypatch.setattr(ai_thumbnail.requests, "get", lambda *a, **k: fake_response)

    result = ai_thumbnail.generate_hook_background("Arrow Greentech", tmp_path / "bg.png", on_progress=lambda m: None)
    assert result == tmp_path / "bg.png"
    assert result.read_bytes() == b"pollinations-bytes"


def test_generate_hook_background_falls_back_to_cloudflare_when_gemini_fails(monkeypatch, tmp_path):
    # Full three-tier cascade: Gemini fails -> Cloudflare succeeds -> Pollinations never called.
    monkeypatch.setattr(config, "GEMINI_API_KEY", "dummy")
    monkeypatch.setattr(config, "CLOUDFLARE_ACCOUNT_ID", "acct123")
    monkeypatch.setattr(config, "CLOUDFLARE_API_TOKEN", "dummy")

    def fake_post(url, **kwargs):
        if "generativelanguage" in url:
            raise RuntimeError("Gemini down")
        return _cloudflare_raw_image_response(b"cloudflare-bytes")

    monkeypatch.setattr(ai_thumbnail.requests, "post", fake_post)
    monkeypatch.setattr(ai_thumbnail.requests, "get", mock.Mock(side_effect=AssertionError("should not fall back to Pollinations")))

    out_path = tmp_path / "bg.png"
    result = ai_thumbnail.generate_hook_background("Arrow Greentech", out_path, on_progress=lambda m: None)

    assert result == out_path
    assert out_path.read_bytes() == b"cloudflare-bytes"


def test_cloudflare_image_within_cap_resets_on_a_new_day(isolated_dirs):
    config.CLOUDFLARE_IMAGE_USAGE_FILE.write_text(json.dumps({"day": "2020-01-01", "requests": 10**9}))
    assert ai_thumbnail.cloudflare_image_within_cap() is True


def test_record_cloudflare_image_usage_increments_requests(isolated_dirs):
    ai_thumbnail._record_cloudflare_image_usage()
    ai_thumbnail._record_cloudflare_image_usage()
    assert ai_thumbnail._load_cloudflare_image_usage()["requests"] == 2
