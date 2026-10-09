import pytest

from pipeline import config, research


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


def test_fetch_grounding_facts_formats_results_with_sources(monkeypatch):
    monkeypatch.setattr(config, "TAVILY_API_KEY", "fake-key")

    class FakeResponse:
        def raise_for_status(self):
            pass

        def json(self):
            return {
                "results": [
                    {"title": "Tata Motors Q2 results", "content": "Revenue grew 12%.", "url": "https://example.com/a"},
                    {"title": "Empty snippet", "content": "", "url": "https://example.com/b"},
                ]
            }

    monkeypatch.setattr(research.requests, "post", lambda *a, **k: FakeResponse())
    result = research.fetch_grounding_facts("Tata Motors stock")

    assert "Revenue grew 12%." in result
    assert "https://example.com/a" in result
    assert "Empty snippet" not in result


def test_fetch_grounding_facts_returns_empty_string_on_request_failure(monkeypatch):
    monkeypatch.setattr(config, "TAVILY_API_KEY", "fake-key")

    import requests

    def fake_post(*a, **k):
        raise requests.ConnectionError("boom")

    monkeypatch.setattr(research.requests, "post", fake_post)
    assert research.fetch_grounding_facts("Tata Motors stock") == ""


# -- monthly credit guardrail --------------------------------------------------

class _FakeResponse:
    def raise_for_status(self):
        pass

    def json(self):
        return {"results": [{"title": "T", "content": "Some fact.", "url": "https://example.com"}]}


def test_fetch_grounding_facts_uses_basic_search_depth_to_minimize_credit_cost(monkeypatch):
    monkeypatch.setattr(config, "TAVILY_API_KEY", "fake-key")
    captured = {}

    def fake_post(url, json, timeout):
        captured["search_depth"] = json["search_depth"]
        return _FakeResponse()

    monkeypatch.setattr(research.requests, "post", fake_post)
    research.fetch_grounding_facts("Tata Motors stock")
    assert captured["search_depth"] == "basic"


def test_fetch_grounding_facts_stops_calling_tavily_once_monthly_cap_reached(monkeypatch):
    monkeypatch.setattr(config, "TAVILY_API_KEY", "fake-key")
    monkeypatch.setattr(config, "TAVILY_MONTHLY_CALL_CAP", 2)
    calls = []
    monkeypatch.setattr(research.requests, "post", lambda *a, **k: (calls.append(1), _FakeResponse())[1])

    assert research.fetch_grounding_facts("a") != ""
    assert research.fetch_grounding_facts("b") != ""
    assert research.fetch_grounding_facts("c") == ""  # cap hit -- no third call
    assert len(calls) == 2


def test_fetch_grounding_facts_records_each_call_in_the_usage_file(monkeypatch):
    monkeypatch.setattr(config, "TAVILY_API_KEY", "fake-key")
    monkeypatch.setattr(research.requests, "post", lambda *a, **k: _FakeResponse())

    research.fetch_grounding_facts("a")
    research.fetch_grounding_facts("b")

    import json
    usage = json.loads(config.TAVILY_USAGE_FILE.read_text())
    assert usage["calls"] == 2
    assert usage["month"] == research._current_month()


def test_usage_counter_resets_for_a_new_calendar_month(monkeypatch):
    monkeypatch.setattr(config, "TAVILY_API_KEY", "fake-key")
    monkeypatch.setattr(config, "TAVILY_MONTHLY_CALL_CAP", 1)
    monkeypatch.setattr(research.requests, "post", lambda *a, **k: _FakeResponse())

    import json
    config.TAVILY_USAGE_FILE.write_text(json.dumps({"month": "2000-01", "calls": 1}))

    assert research.fetch_grounding_facts("a") != ""  # old month's count doesn't carry over

