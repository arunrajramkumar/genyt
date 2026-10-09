from unittest import mock

from pipeline.segmented_script import parse_segments
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
