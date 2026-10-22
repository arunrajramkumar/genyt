import json
import random

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
    assert ts._extract_candidates(text) == [
        {"symbol": "TCS", "name": "Tata Consultancy Services", "event": "other", "event_quote": ""},
    ]


def test_extract_candidates_drops_hallucinated_company_not_in_text(monkeypatch):
    monkeypatch.setattr(
        ts.llm, "chat_json",
        lambda *a, **k: _mock_response({"candidates": [{"symbol": "RELIANCE", "name": "Reliance Industries"}]}),
    )
    text = "Tata Consultancy Services shares rallied today after strong earnings."
    assert ts._extract_candidates(text) == []


def test_chunk_text_respects_limit_and_loses_nothing():
    text = "\n".join(f"line {i} " + "x" * 40 for i in range(100)) + "\n" + "y" * 250
    chunks = ts._chunk_text(text, 500)
    assert all(len(c) <= 500 for c in chunks) and len(chunks) > 5
    assert "".join(chunks).replace("\n", "") == text.replace("\n", "")


def test_extract_candidates_splits_large_input_and_merges_results(monkeypatch):
    monkeypatch.setattr(ts, "MAX_EXTRACT_CHARS", 200)
    sizes = []

    def fake(system, user, **k):
        sizes.append(len(user))
        name = "Tata Consultancy Services" if "Tata" in user else "Infosys"
        return json.dumps({"candidates": [{"symbol": "X", "name": name}] if name in user else []})
    monkeypatch.setattr(ts.llm, "chat_json", fake)
    text = "Tata Consultancy Services posted results.\n" + "filler words here. " * 12 + "\nInfosys won an order.\n" + "filler words here. " * 12
    out = ts._extract_candidates(text)
    assert len(sizes) > 1 and max(sizes) < 400
    assert {c["name"] for c in out} == {"Tata Consultancy Services", "Infosys"}


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


def test_trending_queries_target_fresh_events_and_ipos():
    joined = " ".join(ts.TRENDING_QUERIES).lower()
    assert "results" in joined and "ipo" in joined


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


# -- _event_is_verified / event grounding: an LLM-claimed event must be in the text --

def test_event_is_verified_accepts_ipo_word_near_the_name():
    text = "Acme India IPO opens for subscription on Monday, price band Rs 300-320."
    assert ts._event_is_verified("Acme India", "ipo", text)


def test_event_is_verified_rejects_ipo_claim_when_ipo_word_is_far_from_the_name():
    text = "Acme India reported strong Q1 results today. " + "filler words. " * 30 + "IPO market is slow."
    assert not ts._event_is_verified("Acme India", "ipo", text)


def test_event_is_verified_results_needs_results_words_near_the_name():
    assert ts._event_is_verified("Apollo Micro", "results", "Apollo Micro Q1 revenue jumped 88%.")
    assert not ts._event_is_verified("Apollo Micro", "results", "Apollo Micro was added to a midcap index.")


def test_event_is_verified_order_win_needs_both_order_and_win_words():
    assert ts._event_is_verified("Cupid", "order_win", "Cupid bags a Rs 40 crore export order.")
    assert not ts._event_is_verified("Cupid", "order_win", "Cupid is an export order-heavy business.")


def test_event_is_verified_other_always_passes_and_unknown_never_does():
    assert ts._event_is_verified("Anything", "other", "")
    assert not ts._event_is_verified("Anything", "merger", "Anything merger announced")


def test_extract_candidates_keeps_verified_event_and_quote(monkeypatch):
    monkeypatch.setattr(
        ts.llm, "chat_json",
        lambda *a, **k: _mock_response({"candidates": [{
            "symbol": "ACME", "name": "Acme India", "event": "IPO",
            "event_quote": "Acme India IPO opens Monday",
        }]}),
    )
    text = "Acme India IPO opens Monday for subscription with a Rs 300 price band."
    assert ts._extract_candidates(text) == [{
        "symbol": "ACME", "name": "Acme India", "event": "ipo", "event_quote": "Acme India IPO opens Monday",
    }]


def test_extract_candidates_downgrades_unverifiable_event_to_other(monkeypatch):
    monkeypatch.setattr(
        ts.llm, "chat_json",
        lambda *a, **k: _mock_response({"candidates": [{
            "symbol": "TCS", "name": "Tata Consultancy Services", "event": "ipo",
            "event_quote": "Tata Consultancy Services IPO",
        }]}),
    )
    text = "Tata Consultancy Services shares rallied today after strong earnings."
    assert ts._extract_candidates(text) == [{
        "symbol": "TCS", "name": "Tata Consultancy Services", "event": "other", "event_quote": "",
    }]


def test_extract_candidates_drops_event_quote_that_is_not_in_the_text(monkeypatch):
    monkeypatch.setattr(
        ts.llm, "chat_json",
        lambda *a, **k: _mock_response({"candidates": [{
            "symbol": "CUPID", "name": "Cupid", "event": "order_win", "event_quote": "Cupid wins a huge order",
        }]}),
    )
    text = "Cupid bags a Rs 40 crore export order from a European buyer."
    out = ts._extract_candidates(text)
    assert out[0]["event"] == "order_win" and out[0]["event_quote"] == ""


def test_extract_candidates_prefers_verified_event_over_other_for_same_company(monkeypatch):
    responses = [
        {"candidates": [{"symbol": "CUPID", "name": "Cupid", "event": "other", "event_quote": ""}]},
        {"candidates": [{"symbol": "CUPID", "name": "Cupid", "event": "results", "event_quote": "Cupid Q1 results"}]},
    ]
    monkeypatch.setattr(ts, "MAX_EXTRACT_CHARS", 60)
    monkeypatch.setattr(ts.llm, "chat_json", lambda *a, **k: _mock_response(responses.pop(0)))
    text = "Cupid is a soap maker known to many households.\nCupid Q1 results were out today, profit up."
    out = ts._extract_candidates(text)
    assert len(out) == 1 and out[0]["event"] == "results"


# -- _weighted_pick_without_replacement: event tier, then others, mega-caps last --

def _cand(symbol, event="other"):
    return {"symbol": symbol, "name": symbol, "event": event, "event_quote": ""}


def test_weighted_pick_takes_event_candidates_before_other_candidates():
    pairs = [(_cand("OLDBIG"), 40_000.0), (_cand("NEWSMALL", "results"), 300.0), (_cand("IPOCO", "ipo"), 1_500.0)]
    chosen = [c["symbol"] for c, _ in ts._weighted_pick_without_replacement(pairs, 2)]
    assert set(chosen) == {"NEWSMALL", "IPOCO"}


def test_weighted_pick_draws_from_other_tier_only_after_event_tier_is_exhausted():
    pairs = [(_cand("PLAIN"), 800.0), (_cand("EVENT", "order_win"), 200.0)]
    assert [c["symbol"] for c, _ in ts._weighted_pick_without_replacement(pairs, 1)] == ["EVENT"]


def test_weighted_pick_puts_mega_cap_without_event_last_resort():
    pairs = [(_cand("MEGA"), 900_000.0), (_cand("MIDCAP"), 3_000.0)]
    assert [c["symbol"] for c, _ in ts._weighted_pick_without_replacement(pairs, 1)] == ["MIDCAP"]
    assert [c["symbol"] for c, _ in ts._weighted_pick_without_replacement(pairs, 2)] == ["MIDCAP", "MEGA"]


def test_weighted_pick_mega_cap_with_fresh_event_is_still_eligible_first():
    pairs = [(_cand("MEGA", "results"), 900_000.0), (_cand("MIDCAP"), 3_000.0)]
    assert [c["symbol"] for c, _ in ts._weighted_pick_without_replacement(pairs, 1)] == ["MEGA"]


def test_weighted_pick_keeps_randomness_within_a_tier(monkeypatch):
    monkeypatch.setattr(ts, "random", random.Random(7))
    pairs = [(_cand("AAA"), 1_000.0), (_cand("BBB"), 1_000.0), (_cand("CCC"), 1_000.0)]
    seen = {ts._weighted_pick_without_replacement(pairs, 1)[0][0]["symbol"] for _ in range(300)}
    assert seen == {"AAA", "BBB", "CCC"}


def test_weighted_pick_flattens_market_cap_lead_within_a_tier(monkeypatch):
    # Log weighting: a Rs 50,000 cr name is ~1.5x a Rs 500 cr name, not ~10x.
    monkeypatch.setattr(ts, "random", random.Random(11))
    pairs = [(_cand("BIG"), 50_000.0), (_cand("SMALL"), 500.0)]
    picks = [ts._weighted_pick_without_replacement(pairs, 1)[0][0]["symbol"] for _ in range(2000)]
    assert 0.5 < picks.count("BIG") / len(picks) < 0.7


def test_select_trending_symbols_prefers_event_candidate_over_bigger_plain_one(monkeypatch):
    monkeypatch.setattr(ts, "_search_trending_stock_news", lambda queries=None: "text")
    monkeypatch.setattr(ts, "_extract_candidates", lambda text: [_cand("BIGCO"), _cand("ACME", "ipo")])
    caps = {"BIGCO": 60_000.0, "ACME": 2_000.0}
    monkeypatch.setattr(ts.fundamentals, "fetch_fundamentals", lambda symbol: {"symbol": symbol})
    monkeypatch.setattr(ts.fundamentals, "market_cap_cr", lambda data: caps[data["symbol"]])

    assert ts.select_trending_symbols(exclude_symbols=[], limit=1) == [_cand("ACME", "ipo")]


def test_select_trending_symbols_passes_event_through_and_logs_event_count(monkeypatch, capsys):
    monkeypatch.setattr(ts, "_search_trending_stock_news", lambda queries=None: "text")
    monkeypatch.setattr(ts, "_extract_candidates", lambda text: [_cand("ACME", "ipo")])
    monkeypatch.setattr(ts.fundamentals, "fetch_fundamentals", lambda symbol: {"symbol": symbol})
    monkeypatch.setattr(ts.fundamentals, "market_cap_cr", lambda data: 2_000.0)

    result = ts.select_trending_symbols(exclude_symbols=[], limit=1)
    assert result[0]["event"] == "ipo"
    assert "1 with a fresh news event" in capsys.readouterr().out


def test_select_trending_symbols_drops_unlisted_ipo_before_ranking(monkeypatch):
    # An IPO that isn't listed yet has no Screener page, so verification drops it
    # (same as any unverifiable name) and ranking never sees it.
    monkeypatch.setattr(ts, "_search_trending_stock_news", lambda queries=None: "text")
    monkeypatch.setattr(ts, "_extract_candidates", lambda text: [_cand("UNLISTED", "ipo"), _cand("PLAIN")])
    monkeypatch.setattr(
        ts.fundamentals, "fetch_fundamentals",
        lambda symbol: {} if symbol == "UNLISTED" else {"symbol": symbol},
    )
    monkeypatch.setattr(ts.fundamentals, "market_cap_cr", lambda data: 1_000.0)

    assert [c["symbol"] for c in ts.select_trending_symbols(exclude_symbols=[], limit=1)] == ["PLAIN"]
