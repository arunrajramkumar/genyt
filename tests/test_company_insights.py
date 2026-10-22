import json

import pytest
import requests

from pipeline import company_insights as ci
from pipeline import fundamentals, prompt_script, segmented_script
from pipeline.segmented_script import parse_segments
from pipeline.youtube_insights import filter_title_guidance
from scripts.build_finance_queue import build_theme_entry

# conftest stubs these to "nothing found" for every test; capture the real ones
# at import time so did-you-know tests can re-point them.
REAL_FETCH_PEERS = fundamentals.fetch_peers
REAL_DID_YOU_KNOW_FACTS = ci.did_you_know_facts
REAL_FIND_DYK = prompt_script._find_did_you_know
REAL_ARTICULATE = ci.articulate_facts
REAL_BEAT = ci.did_you_know_beat

SNIPPETS = "Pidilite Industries was founded in 1959 and its first product was Fevicol, a white adhesive for carpenters."
QUOTE = "its first product was Fevicol, a white adhesive for carpenters"


def _llm(payload):
    return lambda *a, **k: json.dumps(payload)


# ---- story grounding ----

def test_story_fact_accepts_grounded_fact(monkeypatch):
    monkeypatch.setattr(ci.llm, "chat_json", _llm({"fact": "Pidilite's first product was Fevicol.", "quote": QUOTE}))
    assert ci._extract_story_fact("Pidilite Industries Ltd", SNIPPETS) == "Pidilite's first product was Fevicol."


def test_story_fact_rejects_quote_not_in_snippets(monkeypatch):
    monkeypatch.setattr(ci.llm, "chat_json", _llm({"fact": "It invented glue on Mars.", "quote": "it invented glue on Mars in space"}))
    assert ci._extract_story_fact("Pidilite Industries Ltd", SNIPPETS) == ""


def test_story_fact_rejects_number_not_in_quote(monkeypatch):
    monkeypatch.setattr(ci.llm, "chat_json", _llm({"fact": "Fevicol sold 500 crore packs.", "quote": QUOTE}))
    assert ci._extract_story_fact("Pidilite Industries Ltd", SNIPPETS) == ""


def test_story_fact_rejects_number_word_not_in_quote(monkeypatch):
    monkeypatch.setattr(ci.llm, "chat_json", _llm({"fact": "Pidilite sold five adhesives.", "quote": QUOTE}))
    assert ci._extract_story_fact("Pidilite Industries Ltd", SNIPPETS) == ""


def test_story_fact_rejects_advice_and_short_quotes(monkeypatch):
    monkeypatch.setattr(ci.llm, "chat_json", _llm({"fact": "Investors should buy Fevicol.", "quote": QUOTE}))
    assert ci._extract_story_fact("Pidilite Industries Ltd", SNIPPETS) == ""
    monkeypatch.setattr(ci.llm, "chat_json", _llm({"fact": "Fevicol.", "quote": "Fevicol"}))
    assert ci._extract_story_fact("Pidilite Industries Ltd", SNIPPETS) == ""


def test_story_fact_rejects_when_company_not_in_snippets(monkeypatch):
    monkeypatch.setattr(ci.llm, "chat_json", _llm({"fact": "Fevicol was a first product.", "quote": QUOTE}))
    assert ci._extract_story_fact("Asian Paints Ltd", SNIPPETS) == ""


def test_story_fact_empty_source_skips_llm(monkeypatch):
    def boom(*a, **k):
        raise AssertionError("LLM must not be called without a source")
    monkeypatch.setattr(ci.llm, "chat_json", boom)
    assert ci._extract_story_fact("Pidilite Industries Ltd", "   ") == ""


def test_story_fact_tolerates_invalid_json(monkeypatch):
    monkeypatch.setattr(ci.llm, "chat_json", lambda *a, **k: "not json at all")
    assert ci._extract_story_fact("Pidilite Industries Ltd", SNIPPETS) == ""


def test_norm_folds_typography():
    assert ci._norm("It’s a “great” – firm") == 'it\'s a "great" - firm'
    assert ci._norm("a b c") == "a b c"


def test_core_name_strips_suffix_and_parentheticals():
    assert ci._core_name("Pidilite Industries Ltd") == "pidilite industries"
    assert ci._core_name("Steelcast Limited (India)") == "steelcast"


# ---- story source order ----

def test_find_story_fact_uses_wikipedia_first_and_skips_web(monkeypatch):
    monkeypatch.setattr(ci, "_wikipedia_extract", lambda n: SNIPPETS)
    monkeypatch.setattr(ci.llm, "chat_json", _llm({"fact": "Pidilite's first product was Fevicol.", "quote": QUOTE}))

    def boom(q):
        raise AssertionError("web search must not be used when Wikipedia verified a fact")
    monkeypatch.setattr(ci.research, "search_web", boom)
    assert ci.find_story_fact("Pidilite Industries Ltd") == "Pidilite's first product was Fevicol."


def test_find_story_fact_falls_back_to_web(monkeypatch):
    monkeypatch.setattr(ci, "_wikipedia_extract", lambda n: "")
    monkeypatch.setattr(ci.research, "search_web", lambda q: SNIPPETS)
    monkeypatch.setattr(ci.llm, "chat_json", _llm({"fact": "Pidilite's first product was Fevicol.", "quote": QUOTE}))
    assert ci.find_story_fact("Pidilite Industries Ltd") == "Pidilite's first product was Fevicol."


def test_find_story_fact_returns_empty_on_failure(monkeypatch):
    def boom(q):
        raise RuntimeError("down")
    monkeypatch.setattr(ci, "_wikipedia_extract", lambda n: "")
    monkeypatch.setattr(ci.research, "search_web", boom)
    assert ci.find_story_fact("Pidilite") == ""


# ---- wikipedia ----

class _Resp:
    def __init__(self, payload):
        self._payload = payload

    def raise_for_status(self):
        pass

    def json(self):
        return self._payload


def _wiki_get(title, extract, hit_title=None):
    def get(url, params=None, **kw):
        if params.get("list") == "search":
            return _Resp({"query": {"search": [{"title": hit_title or title}]}})
        return _Resp({"query": {"pages": {"1": {"extract": extract}}}})
    return get


def test_wikipedia_extract_returns_article_for_matching_title(monkeypatch):
    extract = "Pidilite Industries Limited is an Indian manufacturing company of adhesives."
    monkeypatch.setattr(ci.requests, "get", _wiki_get("Pidilite Industries", extract))
    out = ci._wikipedia_extract("Pidilite Industries Ltd")
    assert "Pidilite Industries Limited is an Indian" in out


def test_wikipedia_extract_rejects_differently_named_article(monkeypatch):
    monkeypatch.setattr(ci.requests, "get", _wiki_get("Pidilite", "A company.", hit_title="Pidilite Fan Club"))
    assert ci._wikipedia_extract("Asian Paints Ltd") == ""


def test_wikipedia_extract_rejects_non_company_article(monkeypatch):
    monkeypatch.setattr(ci.requests, "get", _wiki_get("Steelcast", "Steelcast is a small village known for its river."))
    assert ci._wikipedia_extract("Steelcast Ltd") == ""


def test_wikipedia_extract_survives_network_error(monkeypatch):
    def boom(*a, **k):
        raise requests.ConnectionError("x")
    monkeypatch.setattr(ci.requests, "get", boom)
    assert ci._wikipedia_extract("Pidilite Industries Ltd") == ""


def test_wikipedia_extract_ignores_tiny_names(monkeypatch):
    def boom(*a, **k):
        raise AssertionError("no request for a too-short name")
    monkeypatch.setattr(ci.requests, "get", boom)
    assert ci._wikipedia_extract("Ab") == ""


# ---- peer comparison ----

def _peers(own_roce, own_pe, others):
    peers = [{"name": "Me", "is_self": True, "roce": own_roce, "pe": own_pe}]
    peers += [{"name": f"P{i}", "is_self": False, "roce": r, "pe": p} for i, (r, p) in enumerate(others)]
    return peers


def test_peer_fact_highest_roce():
    fact = ci.peer_comparison_fact(_peers(27.4, 50, [(10, 40), (12, 45), (15, 48), (9, 52)]))
    assert "27.4%" in fact and "highest" in fact and "4 listed peers" in fact


def test_peer_fact_lowest_pe():
    fact = ci.peer_comparison_fact(_peers(15, 10, [(15, 40), (15, 45), (15, 48)]))
    assert "P/E ratio of 10" in fact and "lowest" in fact


def test_peer_fact_partial_rank():
    fact = ci.peer_comparison_fact(_peers(20, 50, [(10, 40), (12, 45), (15, 48), (30, 52)]))
    assert "higher than 3 of its 4 listed peers" in fact


def test_peer_fact_skips_when_company_in_lower_half():
    assert ci.peer_comparison_fact(_peers(5, 90, [(10, 40), (12, 45), (15, 48), (30, 52)])) == ""


def test_peer_fact_needs_enough_peers():
    assert ci.peer_comparison_fact(_peers(27.4, 50, [(10, 40), (12, 45)])) == ""
    assert ci.peer_comparison_fact([]) == ""
    assert ci.peer_comparison_fact(None) == ""


def test_peer_fact_requires_own_row():
    peers = [{"name": f"P{i}", "is_self": False, "roce": r, "pe": 10} for i, r in enumerate([1, 2, 3, 4])]
    assert ci.peer_comparison_fact(peers) == ""


def test_peer_fact_ignores_negative_pe_and_missing_values():
    # Loss-making peers (negative P/E) must not make the company look "cheapest".
    peers = _peers(None, 10, [(None, -5), (None, -8), (None, -2), (None, 40)])
    assert ci.peer_comparison_fact(peers) == ""


# ---- did_you_know_facts ----

def test_did_you_know_facts_combines_story_and_peer(monkeypatch):
    monkeypatch.setattr(ci, "find_story_fact", lambda n: "A story.")
    monkeypatch.setattr(ci.fundamentals, "fetch_peers", lambda d: _peers(27.4, 50, [(10, 40), (12, 45), (15, 48)]))
    facts = REAL_DID_YOU_KNOW_FACTS("Me Ltd", {})
    assert facts[0] == "A story." and "highest" in facts[1]


def test_did_you_know_facts_never_raises_and_drops_missing(monkeypatch):
    monkeypatch.setattr(ci, "find_story_fact", lambda n: "")

    def boom(d):
        raise RuntimeError("down")
    monkeypatch.setattr(ci.fundamentals, "fetch_peers", boom)
    assert REAL_DID_YOU_KNOW_FACTS("Me Ltd", {}) == []


# ---- screener peers parsing ----

PEER_HTML = """
<table><thead><tr><th>S.No.</th><th>Name</th><th>CMP Rs.</th><th>P/E</th><th>Mar Cap Rs.Cr.</th><th>ROCE %</th></tr></thead>
<tr><td>1</td><td><a href="/company/CDSL/consolidated/">CDSL</a></td><td>100</td><td>57.2</td><td>26,334</td><td>31.3</td></tr>
<tr><td>2</td><td><a href="/company/BSE/">BSE</a></td><td>200</td><td>40.0</td><td>50,000</td><td>20.5</td></tr>
<tr><td></td><td>Median: 5 Co.</td><td>1</td><td>45</td><td>1</td><td>25</td></tr></table>
"""


def test_parse_peers_reads_columns_by_header_and_flags_self():
    peers = fundamentals._parse_peers(PEER_HTML, "CDSL")
    assert [p["symbol"] for p in peers] == ["CDSL", "BSE"]
    assert peers[0]["is_self"] and not peers[1]["is_self"]
    assert peers[1]["pe"] == 40.0 and peers[1]["roce"] == 20.5 and peers[1]["market_cap_cr"] == 50000.0


def test_parse_peers_returns_empty_without_table():
    assert fundamentals._parse_peers("<html></html>") == []


def test_parse_peers_ignores_unrelated_tables():
    html = "<table><tr><th>Quarter</th><th>Sales</th></tr><tr><td>Mar</td><td>5</td></tr></table>" + PEER_HTML
    assert [p["symbol"] for p in fundamentals._parse_peers(html, "CDSL")] == ["CDSL", "BSE"]


def test_parse_peers_non_numeric_cells_become_none():
    html = ("<table><tr><th>Name</th><th>P/E</th><th>ROCE %</th></tr>"
            '<tr><td><a href="/company/AAA/">AAA</a></td><td>--</td><td>12</td></tr></table>')
    peer = fundamentals._parse_peers(html, "AAA")[0]
    assert peer["pe"] is None and peer["roce"] == 12.0


def test_parse_warehouse_id_both_forms():
    assert fundamentals._parse_warehouse_id('<div data-warehouse-id="6596">') == "6596"
    assert fundamentals._parse_warehouse_id("fetch('/api/company/6596/peers/')") == "6596"
    assert fundamentals._parse_warehouse_id("<div>") is None


def test_fetch_peers_without_warehouse_id_is_empty():
    assert REAL_FETCH_PEERS({"symbol": "CDSL"}) == []
    assert REAL_FETCH_PEERS({}) == []
    assert REAL_FETCH_PEERS(None) == []


def test_fetch_peers_prefers_inline_peers(monkeypatch):
    def boom(*a, **k):
        raise AssertionError("no request when peers are inline")
    monkeypatch.setattr(fundamentals.requests, "get", boom)
    assert REAL_FETCH_PEERS({"peers": [{"name": "x"}]}) == [{"name": "x"}]


def test_fetch_peers_uses_endpoint(monkeypatch):
    seen = {}

    class R:
        text = PEER_HTML

        def raise_for_status(self):
            pass

    def get(url, **kw):
        seen["url"] = url
        seen["headers"] = kw["headers"]
        return R()
    monkeypatch.setattr(fundamentals.requests, "get", get)
    peers = REAL_FETCH_PEERS({"symbol": "CDSL", "warehouse_id": "6596"})
    assert seen["url"] == "https://www.screener.in/api/company/6596/peers/"
    assert seen["headers"]["X-Requested-With"] == "XMLHttpRequest"
    assert [p["symbol"] for p in peers] == ["CDSL", "BSE"]


def test_fetch_peers_endpoint_failure_is_empty(monkeypatch):
    def boom(*a, **k):
        raise requests.ConnectionError("x")
    monkeypatch.setattr(fundamentals.requests, "get", boom)
    assert REAL_FETCH_PEERS({"symbol": "CDSL", "warehouse_id": "6596"}) == []


RATIO_HTML = (
    '<div data-warehouse-id="6596"><ul id="top-ratios">'
    '<li class="flex flex-space-between"><span class="name">ROCE</span><span class="value">31 %</span></li></ul></div>'
)


def test_fetch_fundamentals_stores_warehouse_id_and_inline_peers(monkeypatch):
    class R:
        text = RATIO_HTML + PEER_HTML

        def raise_for_status(self):
            pass
    monkeypatch.setattr(fundamentals.requests, "get", lambda *a, **k: R())
    data = fundamentals.fetch_fundamentals("cdsl")
    assert data["warehouse_id"] == "6596"
    assert [p["symbol"] for p in data["peers"]] == ["CDSL", "BSE"]


def test_fetch_fundamentals_without_peers_omits_keys(monkeypatch):
    class R:
        text = RATIO_HTML.replace(' data-warehouse-id="6596"', "")

        def raise_for_status(self):
            pass
    monkeypatch.setattr(fundamentals.requests, "get", lambda *a, **k: R())
    data = fundamentals.fetch_fundamentals("cdsl")
    assert "peers" not in data and "warehouse_id" not in data


# ---- build_theme_entry (segmented path) ----

_DATA = {
    "symbol": "CDSL",
    "ratios": {"Market Cap": "₹ 26,334 Cr.", "Stock P/E": "57.2", "ROE": "31.3 %"},
    "quarterly": {"quarter_dates": ["2026-06-30"], "sales_cr": ["222"], "net_profit_cr": ["152"]},
}


def _entry(monkeypatch, facts, duration=45):
    monkeypatch.setattr("scripts.build_finance_queue.fundamentals.fetch_fundamentals", lambda s: _DATA)
    monkeypatch.setattr("scripts.build_finance_queue.company_insights.did_you_know_facts", lambda n, d: facts)
    return build_theme_entry("CDSL", "CDSL", duration)


@pytest.mark.parametrize("duration", [40, 45, 50])
def test_build_theme_entry_adds_did_you_know_after_hook(monkeypatch, duration):
    entry = _entry(monkeypatch, ["A story.", "A peer fact."], duration)
    segments = parse_segments(entry["theme"])
    assert [s["label"] for s in segments][:2] == ["Hook", "Did You Know"]
    assert "A story." in segments[1]["content"] and "A peer fact." in segments[1]["content"]
    assert segments[1]["duration"] == 10
    assert sum(s["duration"] for s in segments) == entry["duration_sec"]
    assert 35 <= entry["duration_sec"] <= 50


def test_build_theme_entry_single_fact_beat_is_shorter(monkeypatch):
    entry = _entry(monkeypatch, ["A peer fact."])
    segments = parse_segments(entry["theme"])
    assert segments[1]["label"] == "Did You Know" and segments[1]["duration"] == 7
    assert "A peer fact." in segments[1]["content"]


def test_build_theme_entry_without_facts_is_unchanged(monkeypatch):
    entry = _entry(monkeypatch, [])
    labels = [s["label"] for s in parse_segments(entry["theme"])]
    assert labels == ["Hook", "Key Metrics", "Revenue & Profit Growth", "Risk & Verdict", "Recap"]
    assert 35 <= entry["duration_sec"] <= 50


# ---- freeform path helpers ----

def test_did_you_know_scene_is_built_from_facts_only():
    scene = prompt_script._did_you_know_scene({"facts": ["A story.", "A peer fact."], "narration": ""})
    assert scene["narration"].startswith("Did you know?")
    assert "A story." in scene["narration"] and "A peer fact." in scene["narration"]
    assert scene["on_screen_text"] == "DID YOU KNOW?"
    assert scene["diagram"] is None and scene["visual_query"]


def test_find_dyk_uses_resolved_company_name(monkeypatch):
    seen = {}
    monkeypatch.setattr(prompt_script.fundamentals, "fetch_fundamentals", lambda s: {"symbol": s})

    def facts(name, data):
        seen["args"] = (name, data)
        return ["F"]
    monkeypatch.setattr(prompt_script.company_insights, "did_you_know_facts", facts)
    assert REAL_FIND_DYK({"name": "Steelcast Ltd"}, "STEELCAST")["facts"] == ["F"]
    assert seen["args"] == ("Steelcast Ltd", {"symbol": "STEELCAST"})


def test_find_dyk_resolves_name_from_symbol(monkeypatch):
    monkeypatch.setattr(prompt_script.fundamentals, "search_company",
                        lambda s: {"symbol": "CDSL", "name": "Central Depository Services (India) Ltd"})
    monkeypatch.setattr(prompt_script.fundamentals, "fetch_fundamentals", lambda s: {"symbol": s})
    monkeypatch.setattr(prompt_script.company_insights, "did_you_know_facts", lambda n, d: [n])
    assert REAL_FIND_DYK({}, "CDSL")["facts"] == ["Central Depository Services (India) Ltd"]


def test_find_dyk_rejects_mismatched_symbol_lookup(monkeypatch):
    monkeypatch.setattr(prompt_script.fundamentals, "search_company", lambda s: {"symbol": "OTHER", "name": "Other Ltd"})
    assert REAL_FIND_DYK({}, "CDSL")["facts"] == []


def test_find_dyk_without_symbol_or_name_is_empty():
    assert REAL_FIND_DYK({}, None)["facts"] == []
    assert REAL_FIND_DYK({"name": "X Ltd"}, "")["facts"] == []


def test_find_dyk_swallows_errors(monkeypatch):
    def boom(s):
        raise RuntimeError("down")
    monkeypatch.setattr(prompt_script.fundamentals, "fetch_fundamentals", boom)
    assert REAL_FIND_DYK({"name": "X Ltd"}, "X")["facts"] == []


# ---- title / guidance hygiene ----

def test_strip_title_emoji_removes_emoji_and_keeps_text():
    out = segmented_script.strip_title_emoji("32.3% ROCE Steelcast Ltd – Wealth \U0001F680")
    assert "\U0001F680" not in out and "32.3% ROCE Steelcast Ltd" in out and out == out.strip()


def test_filter_title_guidance_drops_emoji_and_wealth_advice():
    text = "Use emoji in titles.\nShow the exact number up front.\nTry Long-Term Wealth in every title."
    out = filter_title_guidance(text)
    assert "emoji" not in out.lower() and "wealth" not in out.lower()
    assert "exact number" in out


# ---- LLM articulation of the facts ----

FACTS = ["Its P/E of 103 means 103 years of profit.", "Its latest quarterly profit of Rs 666 crore is Rs 7.3 crore a day."]


def _narr(monkeypatch, narration):
    monkeypatch.setattr(ci.llm, "chat_json", lambda *a, **k: json.dumps({"narration": narration}))


def test_articulate_accepts_grounded_rewrite(monkeypatch):
    text = "Solar Industries earns about Rs 7.3 crore every single day, yet its P/E of 103 asks you to wait 103 years."
    _narr(monkeypatch, text)
    assert REAL_ARTICULATE("Solar Industries India Ltd", FACTS) == text


@pytest.mark.parametrize("bad", [
    "Solar Industries earns Rs 8 crore every single day, which is a lot to think about today.",
    "Solar Industries doubled its profit to Rs 666 crore, which is a lot to think about today.",
    "Solar Industries is a stock you should buy at a P/E of 103 given Rs 666 crore profit.",
    "Solar Industries makes Rs 666 crore, and Reliance makes more than that every quarter easily.",
    "Rs 666 crore.",
    "Did you know Solar Industries made Rs 666 crore in profit during the latest quarter here?",
])
def test_articulate_rejects_ungrounded_rewrites(monkeypatch, bad):
    _narr(monkeypatch, bad)
    assert REAL_ARTICULATE("Solar Industries India Ltd", FACTS) == ""


def test_articulate_survives_llm_errors_and_garbage(monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("down")
    monkeypatch.setattr(ci.llm, "chat_json", boom)
    assert REAL_ARTICULATE("X Ltd", FACTS) == ""
    monkeypatch.setattr(ci.llm, "chat_json", lambda *a, **k: "not json")
    assert REAL_ARTICULATE("X Ltd", FACTS) == ""
    assert REAL_ARTICULATE("X Ltd", []) == ""


def test_beat_uses_all_candidates_but_caps_raw_facts(monkeypatch):
    seen = {}

    def articulate(n, f):
        seen["f"] = f
        return "hooky line"
    monkeypatch.setattr(ci, "did_you_know_facts", lambda n, d: ["a", "b", "c"])
    monkeypatch.setattr(ci, "articulate_facts", articulate)
    beat = REAL_BEAT("X Ltd", {})
    assert seen["f"] == ["a", "b", "c"] and beat == {"facts": ["a", "b"], "narration": "hooky line"}
    monkeypatch.setattr(ci, "did_you_know_facts", lambda n, d: [])
    assert REAL_BEAT("X Ltd", {}) == {"facts": [], "narration": ""}


def test_scene_prefers_validated_narration():
    scene = prompt_script._did_you_know_scene({"facts": ["Raw."], "narration": "Polished hook line."})
    assert scene["narration"] == "Did you know? Polished hook line."


def test_build_theme_entry_speaks_articulated_narration_verbatim(monkeypatch):
    monkeypatch.setattr("scripts.build_finance_queue.fundamentals.fetch_fundamentals", lambda s: _DATA)
    monkeypatch.setattr("scripts.build_finance_queue.company_insights.did_you_know_beat",
                        lambda n, d: {"facts": ["F1."], "narration": "A hooky grounded line."})
    dyk = parse_segments(build_theme_entry("CDSL", "CDSL", 45)["theme"])[1]
    assert dyk["label"] == "Did You Know" and dyk["duration"] == 7
    assert segmented_script._extract_verbatim(dyk["content"]) == "Did you know? A hooky grounded line."
