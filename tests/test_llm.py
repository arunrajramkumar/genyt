from unittest import mock

import pytest
import requests

from pipeline import llm, config


def test_chat_json_raises_without_api_key(no_api_keys):
    with pytest.raises(RuntimeError, match="GROQ_API_KEY"):
        llm.chat_json("system", "user", max_tokens=100, timeout=10)


def _fake_response(status_code=200, json_body=None, text="", headers=None):
    resp = mock.Mock()
    resp.status_code = status_code
    resp.json.return_value = json_body or {}
    resp.text = text
    resp.headers = headers or {}
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
    """The raw Groq error body must surface in the message — raise_for_status()
    alone discards it, which made production 400s undiagnosable."""
    monkeypatch.setattr(config, "GROQ_API_KEY", "dummy-key")
    monkeypatch.setattr(
        llm.requests, "post",
        lambda *a, **k: _fake_response(400, {"error": "bad"}, text='{"error": {"message": "context_length_exceeded"}}'),
    )
    with pytest.raises(RuntimeError, match="context_length_exceeded"):
        llm.chat_json("system", "user", max_tokens=100, timeout=10)


def test_chat_json_retries_with_bigger_budget_on_truncated_generation(monkeypatch):
    """Regression test: reasoning_effort 'low' alone isn't always enough — a
    fact-dense segment can still burn the whole max_tokens budget on hidden
    chain-of-thought and get a 400 json_validate_failed. That should be
    retried with a bigger token budget instead of failing the whole job."""
    monkeypatch.setattr(config, "GROQ_API_KEY", "dummy-key")
    truncated = _fake_response(
        400, {"error": {"code": "json_validate_failed", "message": "max completion tokens reached"}},
    )
    success = _fake_response(200, {"choices": [{"message": {"content": '{"ok": true}'}}]})
    calls = []

    def fake_post(url, headers, json, timeout):
        calls.append(json["max_tokens"])
        return truncated if len(calls) == 1 else success

    monkeypatch.setattr(llm.requests, "post", fake_post)
    result = llm.chat_json("system", "user", max_tokens=512, timeout=10)

    assert result == '{"ok": true}'
    assert calls == [512, 1024]


def test_chat_json_gives_up_after_hitting_token_cap(monkeypatch):
    monkeypatch.setattr(config, "GROQ_API_KEY", "dummy-key")
    truncated = _fake_response(
        400, {"error": {"code": "json_validate_failed", "message": "max completion tokens reached"}},
        text='{"error": {"code": "json_validate_failed"}}',
    )
    calls = []

    def fake_post(url, headers, json, timeout):
        calls.append(json["max_tokens"])
        return truncated

    monkeypatch.setattr(llm.requests, "post", fake_post)
    with pytest.raises(RuntimeError, match="json_validate_failed"):
        llm.chat_json("system", "user", max_tokens=1500, timeout=10)

    assert calls == [1500, llm.MAX_TOKENS_CAP, llm.MAX_TOKENS_CAP]


def test_chat_json_resamples_on_malformed_json_already_at_cap(monkeypatch):
    """Regression test: a production 400 json_validate_failed showed a fully
    completed generation (not truncated) with a "tags" array collapsed into
    one mis-escaped string -- a one-off sampling glitch, not a budget problem.
    The call site (prompt_script.py) already passes max_tokens=4096, above
    MAX_TOKENS_CAP, so the old "bump budget while under cap" retry loop never
    even ran once for it. A plain resample at the same budget should still be
    tried instead of failing outright."""
    monkeypatch.setattr(config, "GROQ_API_KEY", "dummy-key")
    malformed = _fake_response(
        400, {"error": {"code": "json_validate_failed", "message": "Failed to generate JSON. Please adjust your prompt."}},
    )
    success = _fake_response(200, {"choices": [{"message": {"content": '{"ok": true}'}}]})
    calls = []

    def fake_post(url, headers, json, timeout):
        calls.append(json["max_tokens"])
        return malformed if len(calls) == 1 else success

    monkeypatch.setattr(llm.requests, "post", fake_post)
    result = llm.chat_json("system", "user", max_tokens=4096, timeout=120)

    assert result == '{"ok": true}'
    assert calls == [4096, 4096]


def test_chat_json_wraps_connection_error(monkeypatch):
    monkeypatch.setattr(config, "GROQ_API_KEY", "dummy-key")

    def raise_connection_error(*a, **k):
        raise requests.ConnectionError("network down")

    monkeypatch.setattr(llm.requests, "post", raise_connection_error)
    with pytest.raises(RuntimeError, match="Could not reach the Groq API"):
        llm.chat_json("system", "user", max_tokens=100, timeout=10)


def test_chat_json_retries_on_rate_limit_using_suggested_delay(monkeypatch):
    """Regression test: a real production 429 from Groq ('Rate limit reached
    ... tokens per minute (TPM) ... Please try again in 1.4625s') previously
    failed the whole run instantly instead of waiting out a few-second TPM
    window that would have succeeded on retry."""
    monkeypatch.setattr(config, "GROQ_API_KEY", "dummy-key")
    rate_limited = _fake_response(
        429,
        {"error": {"message": "Rate limit reached ... Please try again in 1.4625s", "code": "rate_limit_exceeded"}},
    )
    success = _fake_response(200, {"choices": [{"message": {"content": '{"ok": true}'}}]})
    calls = []

    def fake_post(url, headers, json, timeout):
        calls.append(json["max_tokens"])
        return rate_limited if len(calls) == 1 else success

    sleeps = []
    monkeypatch.setattr(llm.requests, "post", fake_post)
    monkeypatch.setattr(llm.time, "sleep", lambda s: sleeps.append(s))

    result = llm.chat_json("system", "user", max_tokens=512, timeout=10)

    assert result == '{"ok": true}'
    assert len(calls) == 2
    assert sleeps == [1.9625]


def test_chat_json_gives_up_after_max_rate_limit_retries(monkeypatch):
    monkeypatch.setattr(config, "GROQ_API_KEY", "dummy-key")
    rate_limited = _fake_response(429, {"error": {"message": "rate limited"}}, text="rate limited")
    calls = []

    def fake_post(url, headers, json, timeout):
        calls.append(1)
        return rate_limited

    monkeypatch.setattr(llm.requests, "post", fake_post)
    monkeypatch.setattr(llm.time, "sleep", lambda s: None)

    with pytest.raises(RuntimeError, match="Groq API error 429"):
        llm.chat_json("system", "user", max_tokens=100, timeout=10)

    assert len(calls) == 1 + llm.MAX_RATE_LIMIT_RETRIES
