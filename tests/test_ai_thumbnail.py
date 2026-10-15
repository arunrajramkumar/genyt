from unittest import mock

from pipeline import ai_thumbnail, config


def test_generate_hook_background_skips_when_no_api_key(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "GOOGLE_IMAGEN_API_KEY", "")
    result = ai_thumbnail.generate_hook_background("Arrow Greentech", tmp_path / "bg.png", on_progress=lambda m: None)
    assert result is None


def test_generate_hook_background_skips_when_no_company_name(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "GOOGLE_IMAGEN_API_KEY", "fake-key")
    result = ai_thumbnail.generate_hook_background("", tmp_path / "bg.png", on_progress=lambda m: None)
    assert result is None


def test_generate_hook_background_writes_decoded_image_on_success(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "GOOGLE_IMAGEN_API_KEY", "fake-key")
    fake_response = mock.Mock()
    fake_response.raise_for_status = lambda: None
    fake_response.json = lambda: {"predictions": [{"bytesBase64Encoded": "aGVsbG8="}]}  # "hello"
    monkeypatch.setattr(ai_thumbnail.requests, "post", lambda *a, **k: fake_response)

    out_path = tmp_path / "bg.png"
    result = ai_thumbnail.generate_hook_background("Arrow Greentech", out_path, on_progress=lambda m: None)

    assert result == out_path
    assert out_path.read_bytes() == b"hello"


def test_generate_hook_background_returns_none_when_api_errors(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "GOOGLE_IMAGEN_API_KEY", "fake-key")
    monkeypatch.setattr(
        ai_thumbnail.requests, "post", mock.Mock(side_effect=RuntimeError("timeout")),
    )
    logs = []
    result = ai_thumbnail.generate_hook_background("Arrow Greentech", tmp_path / "bg.png", on_progress=logs.append)
    assert result is None
    assert any("timeout" in msg for msg in logs)


def test_generate_hook_background_returns_none_when_no_predictions(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "GOOGLE_IMAGEN_API_KEY", "fake-key")
    fake_response = mock.Mock()
    fake_response.raise_for_status = lambda: None
    fake_response.json = lambda: {"predictions": []}
    monkeypatch.setattr(ai_thumbnail.requests, "post", lambda *a, **k: fake_response)

    result = ai_thumbnail.generate_hook_background("Arrow Greentech", tmp_path / "bg.png", on_progress=lambda m: None)
    assert result is None


def test_build_prompt_excludes_text_and_logo_instructions():
    prompt = ai_thumbnail._build_prompt("Arrow Greentech")
    assert "Arrow Greentech" in prompt
    assert "no text" in prompt
    assert "no logos" in prompt
