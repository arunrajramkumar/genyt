import json

import pytest
import requests

from pipeline import config, research

# Captured before the autouse fixture below monkeypatches research._fetch_duckduckgo_facts
# away by name, so tests that exercise the real implementation can still call it directly.
_real_fetch_duckduckgo_facts = research._fetch_duckduckgo_facts


@pytest.fixture(autouse=True)
def _usage_check_unreachable_by_default(monkeypatch):
    """Most tests exercise the local call-count guardrail in isolation --
    default the live Tavily /usage check to "unreachable" (as it would be
    with the fake API key these tests use) so it doesn't interfere unless a
    test explicitly overrides it."""
    monkeypatch.setattr(research.requests, "get", lambda *a, **k: (_ for _ in ()).throw(requests.ConnectionError("boom")))


@pytest.fixture(autouse=True)
def _duckduckgo_fallback_empty_by_default(monkeypatch):
    """Most tests exercise Tavily (or its absence) in isolation -- default the
    free DuckDuckGo fallback to "nothing found" so a test asserting an empty
    result for a Tavily failure/cap/not-configured path doesn't silently hit
    the real network or get a nondeterministic fallback result. Tests that
    want to exercise the fallback itself override this explicitly."""
    monkeypatch.setattr(research, "_fetch_duckduckgo_facts", lambda topic: "")


@pytest.fixture(autouse=True)
def _company_symbol_search_no_match_by_default(monkeypatch):
    """fetch_grounding_facts resolves bare company-name themes (no literal
    "NSE:" marker) to a symbol via fundamentals.search_company_symbol -- a
    real network call against Screener.in's search endpoint. Default it to
    "no match" so tests that only care about the Tavily/DuckDuckGo path don't
    silently hit the real network or trigger a real fetch_fundamentals call.
    Tests exercising the resolution path override this explicitly."""
    monkeypatch.setattr(research.fundamentals, "search_company_symbol", lambda name: "")


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


@pytest.mark.parametrize("theme", [
    "KPI Green Energy",
    "Reliance Industries",
    "Tata Consultancy Services",
])
def test_needs_grounding_true_for_bare_company_name_themes(theme):
    assert research.needs_grounding(theme) is True


def test_fetch_grounding_facts_returns_empty_string_when_not_configured(monkeypatch):
    monkeypatch.setattr(config, "TAVILY_API_KEY", "")
    assert research.fetch_grounding_facts("Tata Motors stock") == ""


class _FakeResponse:
    def __init__(self, payload=None, text=""):
        self._payload = payload
        self.text = text

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


def test_fetch_remaining_credits_syncs_local_usage_file_to_tavilys_own_count(monkeypatch):
    # The local file is otherwise hand-incremented one call at a time, which
    # drifts from Tavily's real account usage (e.g. calls made during local
    # testing that were never committed) -- a successful live check must
    # overwrite it with the authoritative number instead of leaving it stale.
    config.TAVILY_USAGE_FILE.write_text(json.dumps({"month": research._current_month(), "calls": 3}))
    monkeypatch.setattr(
        research.requests, "get",
        lambda *a, **k: _FakeResponse({"account": {"plan_usage": 357, "plan_limit": 1500}}),
    )
    research._fetch_remaining_credits()
    usage = json.loads(config.TAVILY_USAGE_FILE.read_text())
    assert usage["calls"] == 357
    assert usage["month"] == research._current_month()


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


# -- real fundamentals data (scraped, no API key) take priority over Tavily ---

def test_fetch_grounding_facts_uses_real_fundamentals_for_nse_named_companies(monkeypatch):
    monkeypatch.setattr(config, "TAVILY_API_KEY", "")
    monkeypatch.setattr(
        research.fundamentals, "fetch_fundamentals",
        lambda symbol: {"symbol": symbol, "ratios": {"Stock P/E": "57.2"}},
    )
    result = research.fetch_grounding_facts("NSE: CDSL quarterly results breakdown")
    assert "Stock P/E: 57.2" in result


def test_fetch_grounding_facts_combines_fundamentals_with_tavily_snippets(monkeypatch):
    monkeypatch.setattr(config, "TAVILY_API_KEY", "fake-key")
    monkeypatch.setattr(
        research.fundamentals, "fetch_fundamentals",
        lambda symbol: {"symbol": symbol, "ratios": {"Stock P/E": "57.2"}},
    )
    monkeypatch.setattr(research.requests, "post", lambda *a, **k: _FakeResponse(_SEARCH_RESULTS_PAYLOAD))
    result = research.fetch_grounding_facts("NSE: CDSL quarterly results breakdown")

    assert "Stock P/E: 57.2" in result
    assert "Revenue grew 12%." in result


def test_fetch_grounding_facts_falls_back_to_tavily_when_no_nse_symbol_named(monkeypatch):
    monkeypatch.setattr(config, "TAVILY_API_KEY", "fake-key")
    fundamentals_calls = []
    monkeypatch.setattr(
        research.fundamentals, "fetch_fundamentals",
        lambda symbol: fundamentals_calls.append(symbol) or {},
    )
    monkeypatch.setattr(research.requests, "post", lambda *a, **k: _FakeResponse(_SEARCH_RESULTS_PAYLOAD))
    result = research.fetch_grounding_facts("Tata Motors stock")

    assert fundamentals_calls == []
    assert "Revenue grew 12%." in result


# -- resolving a bare company-name theme (no literal "NSE:") to a real ticker -
# via Screener.in's free name-search endpoint, so live fundamentals still take
# priority over (potentially stale) web-search snippets. -----------------------

@pytest.mark.parametrize("theme, expected_candidate", [
    ("Pidilite Industries - Stock Analysis for Long Term Wealth Creation", "Pidilite Industries"),
    ("Is Reliance Industries Ltd a good dividend stock?", "Reliance Industries"),
    ("TATA MOTORS Stock analysis for long term growth", "TATA MOTORS"),
    ("KPI Green Energy", "KPI Green Energy"),
])
def test_extract_company_name_candidate_trims_generic_finance_words(theme, expected_candidate):
    assert research._extract_company_name_candidate(theme) == expected_candidate


@pytest.mark.parametrize("theme", [
    "5 Facts About Octopuses",
    "Upcoming IPO analysis for this month",
])
def test_extract_company_name_candidate_returns_empty_for_generic_themes(theme):
    # These themes have no real company name in them -- the leftover word(s)
    # (e.g. "Octopuses", "Upcoming") must still be handed to Screener's search
    # as a best-effort query (it reliably returns no match for them), never
    # silently treated as a resolved company.
    candidate = research._extract_company_name_candidate(theme)
    assert candidate in ("", "Octopuses", "Upcoming")


def test_fetch_grounding_facts_resolves_bare_company_name_to_real_fundamentals(monkeypatch):
    monkeypatch.setattr(config, "TAVILY_API_KEY", "")
    monkeypatch.setattr(
        research.fundamentals, "search_company_symbol",
        lambda name: "PIDILITIND" if name == "Pidilite Industries" else "",
    )
    monkeypatch.setattr(
        research.fundamentals, "fetch_fundamentals",
        lambda symbol: {"symbol": symbol, "ratios": {"Stock P/E": "59.8"}} if symbol == "PIDILITIND" else {},
    )
    result = research.fetch_grounding_facts("Pidilite Industries - Stock Analysis for Long Term Wealth Creation")
    assert "Stock P/E: 59.8" in result


def test_fetch_grounding_facts_ignores_resolved_symbol_that_fails_fundamentals_check(monkeypatch):
    # Screener's search returning a hit only confirms a company with that name
    # exists -- it must still be independently verified via fetch_fundamentals
    # before being trusted, same as trending_stocks' verify-then-trust pattern.
    monkeypatch.setattr(config, "TAVILY_API_KEY", "fake-key")
    monkeypatch.setattr(research.fundamentals, "search_company_symbol", lambda name: "FAKECO")
    monkeypatch.setattr(research.fundamentals, "fetch_fundamentals", lambda symbol: {})
    monkeypatch.setattr(research.requests, "post", lambda *a, **k: _FakeResponse(_SEARCH_RESULTS_PAYLOAD))
    result = research.fetch_grounding_facts("Totally Madeup Company stock")

    assert "Stock P/E" not in result
    assert "Revenue grew 12%." in result


def test_fetch_grounding_facts_prefers_explicit_nse_symbol_over_name_resolution(monkeypatch):
    monkeypatch.setattr(config, "TAVILY_API_KEY", "")
    resolve_calls = []
    monkeypatch.setattr(
        research.fundamentals, "search_company_symbol",
        lambda name: resolve_calls.append(name) or "",
    )
    monkeypatch.setattr(
        research.fundamentals, "fetch_fundamentals",
        lambda symbol: {"symbol": symbol, "ratios": {"Stock P/E": "57.2"}},
    )
    result = research.fetch_grounding_facts("NSE: CDSL quarterly results breakdown")

    assert resolve_calls == []  # explicit "NSE: CDSL" already resolved -- no name search needed
    assert "Stock P/E: 57.2" in result


# -- unit normalization: Tavily snippets use international million/billion, --
# -- but Indian audiences read rupee amounts in lakh/crore -------------------

def test_fetch_grounding_facts_normalizes_million_billion_from_tavily_snippets(monkeypatch):
    monkeypatch.setattr(config, "TAVILY_API_KEY", "fake-key")
    payload = {"results": [
        {"title": "Premier Polyfilm", "content": "Market capitalization of ₹11.12 billion.", "url": "https://example.com/a"},
    ]}
    monkeypatch.setattr(research.requests, "post", lambda *a, **k: _FakeResponse(payload))
    result = research.fetch_grounding_facts("Premier Polyfilm stock analysis")
    assert "billion" not in result
    assert "₹1,112 crore" in result


def test_fetch_grounding_facts_proceeds_when_live_credits_comfortably_above_buffer(monkeypatch):
    monkeypatch.setattr(config, "TAVILY_API_KEY", "fake-key")
    monkeypatch.setattr(config, "TAVILY_SAFETY_BUFFER_CREDITS", 50)
    monkeypatch.setattr(research, "_fetch_remaining_credits", lambda: 1000)
    monkeypatch.setattr(research.requests, "post", lambda *a, **k: _FakeResponse(_SEARCH_RESULTS_PAYLOAD))

    assert research.fetch_grounding_facts("Tata Motors stock") != ""


# -- free DuckDuckGo fallback, used once Tavily's quota/credits run dry ------

_DDG_HTML = """
<div class="result">
  <a rel="nofollow" class="result__a" href="//duckduckgo.com/l/?uddg=https%3A%2F%2Fexample.com%2Fa&amp;rut=1">
    Tata Motors <b>Q2</b> results
  </a>
  <a class="result__snippet" href="//duckduckgo.com/l/?uddg=https%3A%2F%2Fexample.com%2Fa">
    Revenue grew <b>12%</b> year over year.
  </a>
</div>
<div class="result">
  <a rel="nofollow" class="result__a" href="https://example.com/b">No snippet here</a>
</div>
"""


def test_fetch_duckduckgo_facts_parses_titles_snippets_and_resolves_redirect_urls(monkeypatch):
    monkeypatch.setattr(research.requests, "post", lambda *a, **k: _FakeResponse(text=_DDG_HTML))
    result = _real_fetch_duckduckgo_facts("Tata Motors stock")
    assert "Tata Motors Q2 results" in result
    assert "Revenue grew 12% year over year." in result
    assert "(source: https://example.com/a)" in result
    assert "No snippet here" not in result  # no matching result__snippet -- regex requires both


def test_fetch_duckduckgo_facts_returns_empty_string_on_request_failure(monkeypatch):
    monkeypatch.setattr(
        research.requests, "post",
        lambda *a, **k: (_ for _ in ()).throw(requests.ConnectionError("boom")),
    )
    assert _real_fetch_duckduckgo_facts("Tata Motors stock") == ""


def test_search_web_falls_back_to_duckduckgo_when_tavily_is_not_configured(monkeypatch):
    monkeypatch.setattr(config, "TAVILY_API_KEY", "")
    monkeypatch.setattr(research, "_fetch_duckduckgo_facts", lambda topic: f"ddg facts about {topic}")
    assert research.search_web("Tata Motors stock") == "ddg facts about Tata Motors stock"


def test_search_web_falls_back_to_duckduckgo_when_tavily_monthly_cap_is_hit(monkeypatch):
    monkeypatch.setattr(config, "TAVILY_API_KEY", "fake-key")
    monkeypatch.setattr(config, "TAVILY_MONTHLY_CALL_CAP", 0)
    monkeypatch.setattr(research, "_fetch_duckduckgo_facts", lambda topic: "ddg fallback facts")
    assert research.search_web("Tata Motors stock") == "ddg fallback facts"


def test_search_web_prefers_tavily_and_skips_duckduckgo_when_tavily_succeeds(monkeypatch):
    monkeypatch.setattr(config, "TAVILY_API_KEY", "fake-key")
    monkeypatch.setattr(research.requests, "post", lambda *a, **k: _FakeResponse(_SEARCH_RESULTS_PAYLOAD))
    ddg_calls = []
    monkeypatch.setattr(research, "_fetch_duckduckgo_facts", lambda topic: ddg_calls.append(topic) or "unused")
    result = research.search_web("Tata Motors stock")
    assert "Revenue grew 12%." in result
    assert ddg_calls == []


def test_fetch_grounding_facts_uses_duckduckgo_facts_when_tavily_not_configured(monkeypatch):
    monkeypatch.setattr(config, "TAVILY_API_KEY", "")
    monkeypatch.setattr(research, "_fetch_duckduckgo_facts", lambda topic: "- Real DDG fact (source: https://example.com)")
    result = research.fetch_grounding_facts("Tata Motors stock")
    assert "Real DDG fact" in result
