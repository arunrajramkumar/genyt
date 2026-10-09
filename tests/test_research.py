import json

import pytest
import requests

from pipeline import config, research


@pytest.fixture(autouse=True)
def _usage_check_unreachable_by_default(monkeypatch):
    """Most tests exercise the local call-count guardrail in isolation --
    default the live Tavily /usage check to "unreachable" (as it would be
    with the fake API key these tests use) so it doesn't interfere unless a
    test explicitly overrides it."""
    monkeypatch.setattr(research.requests, "get", lambda *a, **k: (_ for _ in ()).throw(requests.ConnectionError("boom")))


@pytest.mark.parametrize("theme", [
    "TATA MOTORS Stock analysis for long term growth",
    "Is Reliance Industries Ltd a good dividend stock?",
    "Upcoming IPO analysis for this month",
    "NSE: CDSL quarterly results breakdown",
])
def test_needs_grounding_true_for_finance_themes(theme):
    assert research.needs_grounding(theme) is True


@pytest.mark.parametrize("theme", [
    "5 Facts About Octopuses",
    "History of the Roman Empire",
    "Explain Pythagoras theorem to a 12 year old",
])
def test_needs_grounding_false_for_generic_themes(theme):
    assert research.needs_grounding(theme) is False


def test_fetch_grounding_facts_returns_empty_string_when_not_configured(monkeypatch):
    monkeypatch.setattr(config, "TAVILY_API_KEY", "")
    assert research.fetch_grounding_facts("Tata Motors stock") == ""


class _FakeResponse:
    def __init__(self, payload):
        self._payload = payload

    def raise_for_status(self):
        pass

    def json(self):
        return self._payload


_SEARCH_RESULTS_PAYLOAD = {
    "results": [
        {"title": "Tata Motors Q2 results", "content": "Revenue grew 12%.", "url": "https://example.com/a"},
        {"title": "Empty snippet", "content": "", "url": "https://example.com/b"},
    ]
}


def test_fetch_grounding_facts_formats_results_with_sources(monkeypatch):
    monkeypatch.setattr(config, "TAVILY_API_KEY", "fake-key")
    monkeypatch.setattr(research.requests, "post", lambda *a, **k: _FakeResponse(_SEARCH_RESULTS_PAYLOAD))
    result = research.fetch_grounding_facts("Tata Motors stock")

    assert "Revenue grew 12%." in result
    assert "https://example.com/a" in result
    assert "Empty snippet" not in result


def test_fetch_grounding_facts_returns_empty_string_on_request_failure(monkeypatch):
    monkeypatch.setattr(config, "TAVILY_API_KEY", "fake-key")

    def fake_post(*a, **k):
        raise requests.ConnectionError("boom")

    monkeypatch.setattr(research.requests, "post", fake_post)
    assert research.fetch_grounding_facts("Tata Motors stock") == ""


def test_fetch_grounding_facts_uses_basic_search_depth_to_minimize_credit_cost(monkeypatch):
    monkeypatch.setattr(config, "TAVILY_API_KEY", "fake-key")
    captured = {}

    def fake_post(url, json, timeout):
        captured["search_depth"] = json["search_depth"]
        return _FakeResponse(_SEARCH_RESULTS_PAYLOAD)

    monkeypatch.setattr(research.requests, "post", fake_post)
    research.fetch_grounding_facts("Tata Motors stock")
    assert captured["search_depth"] == "basic"


# -- local call-count guardrail (fallback when the live /usage check fails) ---

def test_fetch_grounding_facts_stops_calling_tavily_once_monthly_cap_reached(monkeypatch):
    monkeypatch.setattr(config, "TAVILY_API_KEY", "fake-key")
    monkeypatch.setattr(config, "TAVILY_MONTHLY_CALL_CAP", 2)
    calls = []
    monkeypatch.setattr(research.requests, "post", lambda *a, **k: (calls.append(1), _FakeResponse(_SEARCH_RESULTS_PAYLOAD))[1])

    assert research.fetch_grounding_facts("a") != ""
    assert research.fetch_grounding_facts("b") != ""
    assert research.fetch_grounding_facts("c") == ""  # cap hit -- no third call
    assert len(calls) == 2


def test_fetch_grounding_facts_records_each_call_in_the_usage_file(monkeypatch):
    monkeypatch.setattr(config, "TAVILY_API_KEY", "fake-key")
    monkeypatch.setattr(research.requests, "post", lambda *a, **k: _FakeResponse(_SEARCH_RESULTS_PAYLOAD))

    research.fetch_grounding_facts("a")
    research.fetch_grounding_facts("b")

    usage = json.loads(config.TAVILY_USAGE_FILE.read_text())
    assert usage["calls"] == 2
    assert usage["month"] == research._current_month()


def test_usage_counter_resets_for_a_new_calendar_month(monkeypatch):
    monkeypatch.setattr(config, "TAVILY_API_KEY", "fake-key")
    monkeypatch.setattr(config, "TAVILY_MONTHLY_CALL_CAP", 1)
    monkeypatch.setattr(research.requests, "post", lambda *a, **k: _FakeResponse(_SEARCH_RESULTS_PAYLOAD))

    config.TAVILY_USAGE_FILE.write_text(json.dumps({"month": "2000-01", "calls": 1}))

    assert research.fetch_grounding_facts("a") != ""  # old month's count doesn't carry over


# -- live Tavily /usage credit check -------------------------------------------

def test_fetch_remaining_credits_returns_none_when_usage_check_fails(monkeypatch):
    monkeypatch.setattr(research.requests, "get", lambda *a, **k: (_ for _ in ()).throw(requests.ConnectionError("boom")))
    assert research._fetch_remaining_credits() is None


def test_fetch_remaining_credits_computes_plan_limit_minus_usage(monkeypatch):
    monkeypatch.setattr(
        research.requests, "get",
        lambda *a, **k: _FakeResponse({"account": {"plan_usage": 1800, "plan_limit": 2000}}),
    )
    assert research._fetch_remaining_credits() == 200


def test_fetch_remaining_credits_returns_none_when_plan_is_unbounded(monkeypatch):
    monkeypatch.setattr(
        research.requests, "get",
        lambda *a, **k: _FakeResponse({"account": {"plan_usage": 500, "plan_limit": None}}),
    )
    assert research._fetch_remaining_credits() is None


def test_fetch_remaining_credits_returns_zero_on_any_paygo_overage_usage(monkeypatch):
    monkeypatch.setattr(
        research.requests, "get",
        lambda *a, **k: _FakeResponse({"account": {"plan_usage": 100, "plan_limit": 2000, "paygo_usage": 5}}),
    )
    assert research._fetch_remaining_credits() == 0


def test_fetch_grounding_facts_skips_the_search_when_live_credits_below_safety_buffer(monkeypatch):
    monkeypatch.setattr(config, "TAVILY_API_KEY", "fake-key")
    monkeypatch.setattr(config, "TAVILY_SAFETY_BUFFER_CREDITS", 50)
    monkeypatch.setattr(research, "_fetch_remaining_credits", lambda: 10)
    calls = []
    monkeypatch.setattr(research.requests, "post", lambda *a, **k: (calls.append(1), _FakeResponse(_SEARCH_RESULTS_PAYLOAD))[1])

    assert research.fetch_grounding_facts("Tata Motors stock") == ""
    assert calls == []


def test_fetch_grounding_facts_proceeds_when_live_credits_comfortably_above_buffer(monkeypatch):
    monkeypatch.setattr(config, "TAVILY_API_KEY", "fake-key")
    monkeypatch.setattr(config, "TAVILY_SAFETY_BUFFER_CREDITS", 50)
    monkeypatch.setattr(research, "_fetch_remaining_credits", lambda: 1000)
    monkeypatch.setattr(research.requests, "post", lambda *a, **k: _FakeResponse(_SEARCH_RESULTS_PAYLOAD))

    assert research.fetch_grounding_facts("Tata Motors stock") != ""
