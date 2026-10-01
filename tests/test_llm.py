from unittest import mock

import pytest
import requests

from pipeline import llm, config


def test_chat_json_raises_without_api_key(no_api_keys):
    with pytest.raises(RuntimeError, match="GROQ_API_KEY"):
        llm.chat_json("system", "user", max_tokens=100, timeout=10)


def _fake_response(status_code=200, json_body=None):
    resp = mock.Mock()
    resp.status_code = status_code
    resp.json.return_value = json_body or {}
    if status_code >= 400:
        resp.raise_for_status.side_effect = requests.HTTPError(f"{status_code} error")
    else:
        resp.raise_for_status.side_effect = None
    return resp


def test_chat_json_returns_message_content(monkeypatch):
    monkeypatch.setattr(config, "GROQ_API_KEY", "dummy-key")
    payload = {"choices": [{"message": {"content": '{"ok": true}'}}]}
    monkeypatch.setattr(llm.requests, "post", lambda *a, **k: _fake_response(200, payload))
    result = llm.chat_json("system", "user", max_tokens=100, timeout=10)
    assert result == '{"ok": true}'


def test_chat_json_sends_reasoning_effort_low(monkeypatch):
    """Regression test: openai/gpt-oss-120b is a reasoning model that will burn
    the whole max_tokens budget on hidden chain-of-thought and return a 400
    (json_validate_failed) unless reasoning_effort is capped — see the
    'Fix Groq script generation failing with 400' commit."""
    monkeypatch.setattr(config, "GROQ_API_KEY", "dummy-key")
    captured = {}

    def fake_post(url, headers, json, timeout):
        captured["json"] = json
        return _fake_response(200, {"choices": [{"message": {"content": "{}"}}]})

    monkeypatch.setattr(llm.requests, "post", fake_post)
    llm.chat_json("system", "user", max_tokens=512, timeout=10)

    assert captured["json"]["reasoning_effort"] == "low"
    assert captured["json"]["max_tokens"] == 512
    assert captured["json"]["response_format"] == {"type": "json_object"}


def test_chat_json_raises_on_http_error(monkeypatch):
    monkeypatch.setattr(config, "GROQ_API_KEY", "dummy-key")
    monkeypatch.setattr(llm.requests, "post", lambda *a, **k: _fake_response(400, {"error": "bad"}))
    with pytest.raises(requests.HTTPError):
        llm.chat_json("system", "user", max_tokens=100, timeout=10)


def test_chat_json_wraps_connection_error(monkeypatch):
    monkeypatch.setattr(config, "GROQ_API_KEY", "dummy-key")

    def raise_connection_error(*a, **k):
        raise requests.ConnectionError("network down")

    monkeypatch.setattr(llm.requests, "post", raise_connection_error)
    with pytest.raises(RuntimeError, match="Could not reach the Groq API"):
        llm.chat_json("system", "user", max_tokens=100, timeout=10)
