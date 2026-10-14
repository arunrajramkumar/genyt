from unittest import mock

import yaml

from pipeline.segmented_script import parse_segments
from scripts import build_finance_queue
from scripts.build_finance_queue import build_theme_entry, MIN_DURATION, MAX_DURATION


_FAKE_DATA = {
    "symbol": "CDSL",
    "ratios": {"Market Cap": "₹ 26,334 Cr.", "Stock P/E": "57.2", "ROE": "31.3 %"},
    "quarterly": {"quarter_dates": ["2026-06-30"], "sales_cr": ["222"], "net_profit_cr": ["152"]},
    "revenue_growth_yoy_pct": 12.1,
    "profit_growth_yoy_pct": 44.8,
}


def test_build_theme_entry_returns_none_when_fundamentals_unavailable(monkeypatch):
    monkeypatch.setattr("scripts.build_finance_queue.fundamentals.fetch_fundamentals", lambda symbol: {})
    assert build_theme_entry("UNKNOWNCO", "Unknown Co") is None


def test_build_theme_entry_emits_five_timed_headers(monkeypatch):
    monkeypatch.setattr("scripts.build_finance_queue.fundamentals.fetch_fundamentals", lambda symbol: _FAKE_DATA)
    entry = build_theme_entry("CDSL", "CDSL")
    segments = parse_segments(entry["theme"])
    labels = [s["label"] for s in segments]
    assert labels == ["Hook", "Key Metrics", "Revenue & Profit Growth", "Risk & Verdict", "Recap"]


def test_build_theme_entry_segments_carry_real_fetched_numbers(monkeypatch):
    monkeypatch.setattr("scripts.build_finance_queue.fundamentals.fetch_fundamentals", lambda symbol: _FAKE_DATA)
    entry = build_theme_entry("CDSL", "CDSL")
    segments = {s["label"]: s["content"] for s in parse_segments(entry["theme"])}

    assert "57.2" in segments["Key Metrics"]
    assert "31.3 %" in segments["Key Metrics"]
    assert "12.1% year-on-year" in segments["Revenue & Profit Growth"]
    assert "44.8% year-on-year" in segments["Revenue & Profit Growth"]


def test_build_theme_entry_risk_segment_forbids_new_numbers_and_recommendations(monkeypatch):
    monkeypatch.setattr("scripts.build_finance_queue.fundamentals.fetch_fundamentals", lambda symbol: _FAKE_DATA)
    entry = build_theme_entry("CDSL", "CDSL")
    segments = {s["label"]: s["content"] for s in parse_segments(entry["theme"])}
    assert "do not" in segments["Risk & Verdict"].lower()
    assert "recommendation" in segments["Risk & Verdict"].lower()


def test_build_theme_entry_duration_clamped_between_90_and_120(monkeypatch):
    monkeypatch.setattr("scripts.build_finance_queue.fundamentals.fetch_fundamentals", lambda symbol: _FAKE_DATA)
    short_entry = build_theme_entry("CDSL", "CDSL", duration_sec=30)
    long_entry = build_theme_entry("CDSL", "CDSL", duration_sec=300)
    assert MIN_DURATION <= short_entry["duration_sec"] <= MAX_DURATION
    assert MIN_DURATION <= long_entry["duration_sec"] <= MAX_DURATION


def test_build_theme_entry_duration_sec_matches_sum_of_segment_durations(monkeypatch):
    monkeypatch.setattr("scripts.build_finance_queue.fundamentals.fetch_fundamentals", lambda symbol: _FAKE_DATA)
    entry = build_theme_entry("CDSL", "CDSL")
    segments = parse_segments(entry["theme"])
    assert sum(s["duration"] for s in segments) == entry["duration_sec"]


def test_build_theme_entry_sets_shorts_and_extra_tags(monkeypatch):
    monkeypatch.setattr("scripts.build_finance_queue.fundamentals.fetch_fundamentals", lambda symbol: _FAKE_DATA)
    entry = build_theme_entry("CDSL", "CDSL Ltd")
    assert entry["shorts"] is True
    assert entry["done"] is False
    assert "CDSL Ltd" in entry["extra_tags"]
    assert "CDSL" in entry["extra_tags"]


# -- existing_symbols: reads already-covered NSE symbols out of a queue file -

def testexisting_symbols_returns_empty_set_for_missing_file():
    assert build_finance_queue.existing_symbols("/no/such/file.yaml") == set()


def testexisting_symbols_extracts_symbols_from_theme_text(tmp_path):
    path = tmp_path / "themes.yaml"
    path.write_text(yaml.safe_dump({"themes": [
        {"theme": "Central Depository Services India Limited (NSE: CDSL)."},
        {"theme": "Kalyan Jewellers India Limited (NSE: KALYANKJIL)."},
    ]}))
    assert build_finance_queue.existing_symbols(str(path)) == {"CDSL", "KALYANKJIL"}


# -- main: --trending path wires discovery into the same entry builder -------

def test_main_requires_symbols_or_trending(monkeypatch, capsys):
    monkeypatch.setattr("sys.argv", ["build_finance_queue.py", "--output", "out.yaml"])
    try:
        build_finance_queue.main()
        assert False, "expected SystemExit"
    except SystemExit as e:
        assert e.code != 0


def test_main_trending_queues_discovered_candidates(monkeypatch, tmp_path):
    output = tmp_path / "out.yaml"
    monkeypatch.setattr(
        "sys.argv",
        ["build_finance_queue.py", "--trending", "--limit", "2", "--output", str(output)],
    )
    monkeypatch.setattr(
        build_finance_queue.trending_stocks, "select_trending_symbols",
        lambda exclude_symbols, limit: [{"symbol": "TCS", "name": "TCS"}],
    )
    monkeypatch.setattr(build_finance_queue.fundamentals, "fetch_fundamentals", lambda symbol: _FAKE_DATA)

    build_finance_queue.main()

    written = yaml.safe_load(output.read_text())
    assert len(written["themes"]) == 1
    assert "TCS" in written["themes"][0]["_label"]


def test_main_trending_skips_fundamentals_query_when_nothing_discovered(monkeypatch, tmp_path):
    output = tmp_path / "out.yaml"
    monkeypatch.setattr(
        "sys.argv",
        ["build_finance_queue.py", "--trending", "--output", str(output)],
    )
    monkeypatch.setattr(build_finance_queue.trending_stocks, "select_trending_symbols", lambda exclude_symbols, limit: [])

    def fail_if_called(symbol):
        raise AssertionError("fetch_fundamentals should not be called with no candidates")

    monkeypatch.setattr(build_finance_queue.fundamentals, "fetch_fundamentals", fail_if_called)

    build_finance_queue.main()

    written = yaml.safe_load(output.read_text())
    assert written["themes"] == []
