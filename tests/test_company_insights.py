import json

from pipeline import company_insights as ci
from pipeline import fundamentals
from pipeline.segmented_script import parse_segments
from scripts.build_finance_queue import build_theme_entry

SNIPPETS = "Pidilite Industries was founded in 1959 and its first product was Fevicol, a white adhesive for carpenters."
QUOTE = "its first product was Fevicol, a white adhesive for carpenters"


def _llm(payload):
    return lambda *a, **k: json.dumps(payload)


def test_story_fact_accepts_grounded_fact(monkeypatch):
    monkeypatch.setattr(ci.llm, "chat_json", _llm({"fact": "Pidilite's first product was Fevicol.", "quote": QUOTE}))
    assert ci._extract_story_fact("Pidilite Industries Ltd", SNIPPETS) == "Pidilite's first product was Fevicol."


def test_story_fact_rejects_quote_not_in_snippets(monkeypatch):
    monkeypatch.setattr(ci.llm, "chat_json", _llm({"fact": "It invented glue on Mars.", "quote": "it invented glue on Mars in space"}))
    assert ci._extract_story_fact("Pidilite Industries Ltd", SNIPPETS) == ""


def test_story_fact_rejects_number_not_in_quote(monkeypatch):
    monkeypatch.setattr(ci.llm, "chat_json", _llm({"fact": "Fevicol sold 500 crore packs.", "quote": QUOTE}))
    assert ci._extract_story_fact("Pidilite Industries Ltd", SNIPPETS) == ""


def test_story_fact_rejects_advice_and_short_quotes(monkeypatch):
    monkeypatch.setattr(ci.llm, "chat_json", _llm({"fact": "Investors should buy Fevicol.", "quote": QUOTE}))
    assert ci._extract_story_fact("Pidilite Industries Ltd", SNIPPETS) == ""
    monkeypatch.setattr(ci.llm, "chat_json", _llm({"fact": "Fevicol.", "quote": "Fevicol"}))
    assert ci._extract_story_fact("Pidilite Industries Ltd", SNIPPETS) == ""


def test_story_fact_rejects_when_company_not_in_snippets(monkeypatch):
    monkeypatch.setattr(ci.llm, "chat_json", _llm({"fact": "Fevicol was a first product.", "quote": QUOTE}))
    assert ci._extract_story_fact("Asian Paints Ltd", SNIPPETS) == ""


def test_find_story_fact_returns_empty_on_failure(monkeypatch):
    def boom(q):
        raise RuntimeError("down")
    monkeypatch.setattr(ci.research, "search_web", boom)
    assert ci.find_story_fact("Pidilite") == ""


def _peers(own_roce, own_pe, others):
    peers = [{"name": "Me", "is_self": True, "roce": own_roce, "pe": own_pe}]
    peers += [{"name": f"P{i}", "is_self": False, "roce": r, "pe": p} for i, (r, p) in enumerate(others)]
    return peers


def test_peer_fact_highest_roce():
    fact = ci.peer_comparison_fact(_peers(27.4, 50, [(10, 40), (12, 45), (15, 48), (9, 52)]))
    assert "27.4%" in fact and "highest" in fact and "4" in fact


def test_peer_fact_lowest_pe():
    fact = ci.peer_comparison_fact(_peers(15, 10, [(15, 40), (15, 45), (15, 48)]))
    assert "P/E of 10" in fact and "lowest" in fact


def test_peer_fact_needs_enough_peers():
    assert ci.peer_comparison_fact(_peers(27.4, 50, [(10, 40), (12, 45)])) == ""
    assert ci.peer_comparison_fact([]) == ""


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


def test_parse_warehouse_id():
    assert fundamentals._parse_warehouse_id('<div data-warehouse-id="6596">') == "6596"
    assert fundamentals._parse_warehouse_id("<div>") is None


def test_fetch_peers_without_warehouse_id_is_empty(monkeypatch):
    monkeypatch.undo()
    assert fundamentals.fetch_peers({"symbol": "CDSL"}) == []


_DATA = {
    "symbol": "CDSL",
    "ratios": {"Market Cap": "₹ 26,334 Cr.", "Stock P/E": "57.2", "ROE": "31.3 %"},
    "quarterly": {"quarter_dates": ["2026-06-30"], "sales_cr": ["222"], "net_profit_cr": ["152"]},
}


def test_build_theme_entry_adds_did_you_know_after_hook(monkeypatch):
    monkeypatch.setattr("scripts.build_finance_queue.fundamentals.fetch_fundamentals", lambda s: _DATA)
    monkeypatch.setattr("scripts.build_finance_queue.company_insights.find_story_fact", lambda n: "A story.")
    monkeypatch.setattr("scripts.build_finance_queue.company_insights.peer_comparison_fact", lambda p: "A peer fact.")
    entry = build_theme_entry("CDSL", "CDSL")
    segments = parse_segments(entry["theme"])
    assert [s["label"] for s in segments][:2] == ["Hook", "Did You Know"]
    assert "A story." in segments[1]["content"] and "A peer fact." in segments[1]["content"]
    assert segments[1]["duration"] == 10
    assert sum(s["duration"] for s in segments) == entry["duration_sec"]
    assert 50 <= entry["duration_sec"] <= 65


def test_build_theme_entry_peer_fact_alone_is_fallback_beat(monkeypatch):
    monkeypatch.setattr("scripts.build_finance_queue.fundamentals.fetch_fundamentals", lambda s: _DATA)
    monkeypatch.setattr("scripts.build_finance_queue.company_insights.peer_comparison_fact", lambda p: "A peer fact.")
    entry = build_theme_entry("CDSL", "CDSL")
    segments = parse_segments(entry["theme"])
    assert segments[1]["label"] == "Did You Know" and segments[1]["duration"] == 7
    assert "A peer fact." in segments[1]["content"]
