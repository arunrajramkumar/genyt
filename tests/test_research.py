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
