import json

from pipeline import trending_stocks as ts


def _mock_response(payload: dict) -> str:
    return json.dumps(payload)


# -- _search_trending_stock_news: aggregates Tavily snippets across queries --

def test_search_trending_stock_news_joins_nonempty_results(monkeypatch):
    monkeypatch.setattr(ts.research, "search_web", lambda q: f"facts about {q}" if "today" in q else "")
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
    assert len(result) == 2
    assert {c["symbol"] for c in result} <= {"A", "B", "C"}


def test_select_trending_symbols_ranks_by_market_cap_descending(monkeypatch):
    # Candidates come back in LLM-extraction order (small company first), but
    # the smaller company must not win a limited slot over a bigger, more
    # widely-recognized one -- stub out the weighted-random pick itself
    # (tested separately below) so this test only verifies the final result
    # is market-cap-sorted, not which random draw happened to occur.
    monkeypatch.setattr(ts, "_search_trending_stock_news", lambda queries=None: "text")
    monkeypatch.setattr(
        ts, "_extract_candidates",
        lambda text: [{"symbol": s, "name": s} for s in ["SMALLCO", "BIGCO", "MIDCO"]],
    )
    market_caps = {"SMALLCO": 500.0, "BIGCO": 50000.0, "MIDCO": 5000.0}
    monkeypatch.setattr(ts.fundamentals, "fetch_fundamentals", lambda symbol: {"symbol": symbol})
    monkeypatch.setattr(ts.fundamentals, "market_cap_cr", lambda data: market_caps[data["symbol"]])
    monkeypatch.setattr(
        ts, "_weighted_pick_without_replacement",
        lambda pairs, k: sorted(pairs, key=lambda p: p[1], reverse=True)[:k],
    )

    result = ts.select_trending_symbols(exclude_symbols=[], limit=2)
    assert [c["symbol"] for c in result] == ["BIGCO", "MIDCO"]


# -- _weighted_pick_without_replacement: market-cap-weighted diversity -------

def test_weighted_pick_without_replacement_never_exceeds_k():
    pairs = [({"symbol": s}, 100.0) for s in ["A", "B", "C"]]
    assert len(ts._weighted_pick_without_replacement(pairs, 2)) == 2


def test_weighted_pick_without_replacement_returns_all_when_k_exceeds_pool_size():
    pairs = [({"symbol": s}, 100.0) for s in ["A", "B"]]
    assert len(ts._weighted_pick_without_replacement(pairs, 5)) == 2


def test_weighted_pick_without_replacement_never_repeats_a_candidate():
    pairs = [({"symbol": s}, 100.0) for s in ["A", "B", "C", "D"]]
    chosen = ts._weighted_pick_without_replacement(pairs, 4)
    assert len({c["symbol"] for c, _cap in chosen}) == 4


def test_weighted_pick_without_replacement_gives_zero_market_cap_a_nonzero_chance(monkeypatch):
    # A candidate with an unparseable/zero market cap (weight 0) must still be
    # pickable (weight floored at 1), not permanently excluded.
    pairs = [({"symbol": "ZERO"}, 0.0), ({"symbol": "HUGE"}, 1_000_000.0)]
    monkeypatch.setattr(ts.random, "choices", lambda pool, weights, k: [pool[0]] if weights[0] > 0 else [pool[1]])
    chosen = ts._weighted_pick_without_replacement(pairs, 1)
    assert chosen[0][0]["symbol"] == "ZERO"


def test_select_trending_symbols_excludes_companies_already_on_the_channel(monkeypatch):
    monkeypatch.setattr(ts, "_search_trending_stock_news", lambda queries=None: "text")
    monkeypatch.setattr(
        ts, "_extract_candidates",
        lambda text: [
            {"symbol": "TATAMOTORS", "name": "Tata Motors Ltd"},
            {"symbol": "CDSL", "name": "CDSL"},
            {"symbol": "TCS", "name": "Tata Consultancy Services"},
        ],
    )
    monkeypatch.setattr(ts.fundamentals, "fetch_fundamentals", lambda symbol: {"symbol": symbol})
    channel = [
        "Tata Motors: Is the EV bet working?\nA deep dive.\nstocks",
        "Some title\nDeep dive: CDSL (NSE: CDSL)\n",
    ]
    result = ts.select_trending_symbols(limit=5, channel_texts=channel)
    assert [c["symbol"] for c in result] == ["TCS"]


def test_select_trending_symbols_widens_search_when_all_candidates_are_covered(monkeypatch):
    batches_seen = []

    def fake_news(queries=None):
        batches_seen.append(queries)
        return "text"

    def fake_extract(text):
        return [{"symbol": "CDSL", "name": "CDSL"}] if len(batches_seen) == 1 else [{"symbol": "TCS", "name": "TCS"}]

    monkeypatch.setattr(ts, "_search_trending_stock_news", fake_news)
    monkeypatch.setattr(ts, "_extract_candidates", fake_extract)
    monkeypatch.setattr(ts.fundamentals, "fetch_fundamentals", lambda symbol: {"symbol": symbol})

    result = ts.select_trending_symbols(exclude_symbols=["CDSL"], limit=1)
    assert result == [{"symbol": "TCS", "name": "TCS"}]
    assert batches_seen == [None, ts.EXTRA_QUERY_BATCHES[0]]


def test_select_trending_symbols_returns_empty_when_every_batch_is_covered(monkeypatch):
    monkeypatch.setattr(ts, "_search_trending_stock_news", lambda queries=None: "text")
    monkeypatch.setattr(ts, "_extract_candidates", lambda text: [{"symbol": "CDSL", "name": "CDSL"}])
    monkeypatch.setattr(ts.fundamentals, "fetch_fundamentals", lambda symbol: {"symbol": symbol})
    assert ts.select_trending_symbols(exclude_symbols=["CDSL"]) == []


def test_select_trending_symbols_prefers_well_known_companies_when_enough_exist(monkeypatch):
    monkeypatch.setattr(ts, "_search_trending_stock_news", lambda queries=None: "text")
    monkeypatch.setattr(
        ts, "_extract_candidates",
        lambda text: [{"symbol": s, "name": s} for s in ["TINY", "BIG1", "BIG2"]],
    )
    caps = {"TINY": 300.0, "BIG1": 90000.0, "BIG2": 50000.0}
    monkeypatch.setattr(ts.fundamentals, "fetch_fundamentals", lambda symbol: {"symbol": symbol})
    monkeypatch.setattr(ts.fundamentals, "market_cap_cr", lambda data: caps[data["symbol"]])
    for _ in range(20):
        result = ts.select_trending_symbols(limit=2)
        assert {c["symbol"] for c in result} == {"BIG1", "BIG2"}


def test_select_trending_symbols_returns_empty_on_unexpected_failure(monkeypatch, capsys):
    def boom(queries=None):
        raise RuntimeError("network down")

    monkeypatch.setattr(ts, "_search_trending_stock_news", boom)
    assert ts.select_trending_symbols() == []
    assert "network down" in capsys.readouterr().out


def test_select_trending_symbols_logs_why_nothing_was_selected(monkeypatch, capsys):
    monkeypatch.setattr(ts, "_search_trending_stock_news", lambda queries=None: "text")
    monkeypatch.setattr(
        ts, "_extract_candidates",
        lambda text: [{"symbol": "CDSL", "name": "CDSL"}, {"symbol": "FAKECO", "name": "Fake Co"}],
    )
    monkeypatch.setattr(ts.fundamentals, "fetch_fundamentals", lambda symbol: {})

    assert ts.select_trending_symbols(exclude_symbols=["CDSL"]) == []
    out = capsys.readouterr().out
    assert "2 grounded candidate(s)" in out
    assert "1 already covered/duplicate, 1 failed Screener verification, 0 verified" in out
