import json

from pipeline import trending_stocks as ts


def _mock_response(payload: dict) -> str:
    return json.dumps(payload)


# -- _search_trending_stock_news: aggregates Tavily snippets across queries --

def test_search_trending_stock_news_joins_nonempty_results(monkeypatch):
    monkeypatch.setattr(ts.research, "_fetch_tavily_facts", lambda q: f"facts about {q}" if "today" in q else "")
    text = ts._search_trending_stock_news(["NSE stocks today", "other query"])
    assert "facts about NSE stocks today" in text
    assert "other query" not in text


# -- _extract_candidates: grounds LLM output against the source text ---------

def test_extract_candidates_returns_empty_for_empty_text():
    assert ts._extract_candidates("") == []


def test_extract_candidates_keeps_candidates_named_in_the_text(monkeypatch):
    monkeypatch.setattr(
        ts.llm, "chat_json",
        lambda *a, **k: _mock_response({"candidates": [{"symbol": "TCS", "name": "Tata Consultancy Services"}]}),
    )
    text = "Tata Consultancy Services shares rallied today after strong earnings."
    assert ts._extract_candidates(text) == [{"symbol": "TCS", "name": "Tata Consultancy Services"}]


def test_extract_candidates_drops_hallucinated_company_not_in_text(monkeypatch):
    monkeypatch.setattr(
        ts.llm, "chat_json",
        lambda *a, **k: _mock_response({"candidates": [{"symbol": "RELIANCE", "name": "Reliance Industries"}]}),
    )
    text = "Tata Consultancy Services shares rallied today after strong earnings."
    assert ts._extract_candidates(text) == []


def test_extract_candidates_returns_empty_after_bad_responses(monkeypatch):
    monkeypatch.setattr(ts.llm, "chat_json", lambda *a, **k: "not json at all")
    assert ts._extract_candidates("some real snippet text") == []


# -- select_trending_symbols: dedupe, exclusion, fundamentals verification ---

def test_select_trending_symbols_excludes_already_covered_symbols(monkeypatch):
    monkeypatch.setattr(ts, "_search_trending_stock_news", lambda queries=None: "text")
    monkeypatch.setattr(
        ts, "_extract_candidates",
        lambda text: [{"symbol": "CDSL", "name": "CDSL"}, {"symbol": "TCS", "name": "TCS"}],
    )
    monkeypatch.setattr(ts.fundamentals, "fetch_fundamentals", lambda symbol: {"symbol": symbol})

    result = ts.select_trending_symbols(exclude_symbols=["CDSL"], limit=5)
    assert result == [{"symbol": "TCS", "name": "TCS"}]


def test_select_trending_symbols_skips_candidates_that_fail_fundamentals_check(monkeypatch):
    monkeypatch.setattr(ts, "_search_trending_stock_news", lambda queries=None: "text")
    monkeypatch.setattr(
        ts, "_extract_candidates",
        lambda text: [{"symbol": "FAKECO", "name": "Fake Co"}, {"symbol": "TCS", "name": "TCS"}],
    )
    monkeypatch.setattr(ts.fundamentals, "fetch_fundamentals", lambda symbol: {} if symbol == "FAKECO" else {"symbol": symbol})

    result = ts.select_trending_symbols(exclude_symbols=[], limit=5)
    assert result == [{"symbol": "TCS", "name": "TCS"}]


def test_select_trending_symbols_respects_limit(monkeypatch):
    monkeypatch.setattr(ts, "_search_trending_stock_news", lambda queries=None: "text")
    monkeypatch.setattr(
        ts, "_extract_candidates",
        lambda text: [{"symbol": s, "name": s} for s in ["A", "B", "C"]],
    )
    monkeypatch.setattr(ts.fundamentals, "fetch_fundamentals", lambda symbol: {"symbol": symbol})

    result = ts.select_trending_symbols(exclude_symbols=[], limit=2)
    assert [c["symbol"] for c in result] == ["A", "B"]


def test_select_trending_symbols_ranks_by_market_cap_descending(monkeypatch):
    # Candidates come back in LLM-extraction order (small company first), but
    # the smaller company must not win a limited slot over a bigger, more
    # widely-recognized one.
    monkeypatch.setattr(ts, "_search_trending_stock_news", lambda queries=None: "text")
    monkeypatch.setattr(
        ts, "_extract_candidates",
        lambda text: [{"symbol": s, "name": s} for s in ["SMALLCO", "BIGCO", "MIDCO"]],
    )
    market_caps = {"SMALLCO": 500.0, "BIGCO": 50000.0, "MIDCO": 5000.0}
    monkeypatch.setattr(ts.fundamentals, "fetch_fundamentals", lambda symbol: {"symbol": symbol})
    monkeypatch.setattr(ts.fundamentals, "market_cap_cr", lambda data: market_caps[data["symbol"]])

    result = ts.select_trending_symbols(exclude_symbols=[], limit=2)
    assert [c["symbol"] for c in result] == ["BIGCO", "MIDCO"]


def test_select_trending_symbols_returns_empty_on_unexpected_failure(monkeypatch):
    def boom(queries=None):
        raise RuntimeError("network down")

    monkeypatch.setattr(ts, "_search_trending_stock_news", boom)
    assert ts.select_trending_symbols() == []
