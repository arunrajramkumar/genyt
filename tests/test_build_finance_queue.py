from unittest import mock

import pytest
import yaml

from pipeline.segmented_script import parse_segments
from scripts import build_finance_queue
from scripts.build_finance_queue import build_theme_entry, MIN_DURATION, MAX_DURATION


_FAKE_DATA = {
    "symbol": "CDSL",
    "ratios": {"Market Cap": "₹ 26,334 Cr.", "Stock P/E": "57.2", "ROE": "31.3 %"},
    "quarterly": {"quarter_dates": ["2026-06-30"], "sales_cr": ["200", "205", "210", "215", "222"],
                  "net_profit_cr": ["100", "110", "120", "130", "152"]},
    "revenue_growth_yoy_pct": 11.0,
    "profit_growth_yoy_pct": 52.0,
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
    assert "11% year-on-year" in segments["Revenue & Profit Growth"]
    assert "52% year-on-year" in segments["Revenue & Profit Growth"]


def test_build_theme_entry_risk_segment_forbids_new_numbers_and_recommendations(monkeypatch):
    monkeypatch.setattr("scripts.build_finance_queue.fundamentals.fetch_fundamentals", lambda symbol: _FAKE_DATA)
    entry = build_theme_entry("CDSL", "CDSL")
    segments = {s["label"]: s["content"] for s in parse_segments(entry["theme"])}
    assert "do not" in segments["Risk & Verdict"].lower()
    assert "recommendation" in segments["Risk & Verdict"].lower()


def test_build_theme_entry_duration_clamped_to_min_and_max(monkeypatch):
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


def test_build_theme_entry_appends_competitor_tags(monkeypatch):
    monkeypatch.setattr("scripts.build_finance_queue.fundamentals.fetch_fundamentals", lambda symbol: _FAKE_DATA)
    entry = build_theme_entry("CDSL", "CDSL Ltd", competitor_tags=["mutual funds", "share market"])
    assert entry["extra_tags"][-2:] == ["mutual funds", "share market"]


# -- hook/metrics content: lead with the most striking number, cap data dump -

def test_build_theme_entry_hook_leads_with_headline_number_not_company_name(monkeypatch):
    monkeypatch.setattr("scripts.build_finance_queue.fundamentals.fetch_fundamentals", lambda symbol: _FAKE_DATA)
    entry = build_theme_entry("CDSL", "CDSL")
    segments = {s["label"]: s["content"] for s in parse_segments(entry["theme"])}
    hook = segments["Hook"]
    assert hook.index("52% year-on-year") < hook.index("CDSL (NSE: CDSL)")


def test_build_theme_entry_hook_prefers_growth_pct_over_a_scraped_ratio(monkeypatch):
    monkeypatch.setattr("scripts.build_finance_queue.fundamentals.fetch_fundamentals", lambda symbol: _FAKE_DATA)
    entry = build_theme_entry("CDSL", "CDSL")
    segments = {s["label"]: s["content"] for s in parse_segments(entry["theme"])}
    assert "Profit growth: 52% year-on-year" in segments["Hook"]


def test_build_theme_entry_key_metrics_caps_to_three_most_striking_ratios(monkeypatch):
    data = dict(_FAKE_DATA, ratios={
        "Market Cap": "₹ 26,334 Cr.", "Face Value": "2", "Book Value": "120",
        "Stock P/E": "57.2", "ROE": "31.3 %", "ROCE": "28.1 %", "Dividend Yield": "0.5 %",
    })
    monkeypatch.setattr("scripts.build_finance_queue.fundamentals.fetch_fundamentals", lambda symbol: data)
    entry = build_theme_entry("CDSL", "CDSL")
    segments = {s["label"]: s["content"] for s in parse_segments(entry["theme"])}
    metrics = segments["Key Metrics"]
    # Priority ratios survive the cap...
    assert "28.1 %" in metrics  # ROCE
    assert "31.3 %" in metrics  # ROE
    assert "57.2" in metrics  # Stock P/E
    # ...while a lower-priority ratio (7 were given, cap is 3) is dropped.
    assert "Face Value" not in metrics
    assert "Book Value" not in metrics


def test_build_theme_entry_key_metrics_instructs_plain_english_translation(monkeypatch):
    monkeypatch.setattr("scripts.build_finance_queue.fundamentals.fetch_fundamentals", lambda symbol: _FAKE_DATA)
    entry = build_theme_entry("CDSL", "CDSL")
    segments = {s["label"]: s["content"] for s in parse_segments(entry["theme"])}
    assert "plain-English" in segments["Key Metrics"]
    assert "2 or 3" in segments["Key Metrics"]


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
    monkeypatch.setattr(build_finance_queue.market_trends, "get_competitor_tags", lambda limit=5: [])

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
    monkeypatch.setattr(build_finance_queue.market_trends, "get_competitor_tags", lambda limit=5: [])

    def fail_if_called(symbol):
        raise AssertionError("fetch_fundamentals should not be called with no candidates")

    monkeypatch.setattr(build_finance_queue.fundamentals, "fetch_fundamentals", fail_if_called)

    build_finance_queue.main()

    written = yaml.safe_load(output.read_text())
    assert written["themes"] == []


# -- short-form timing: 35-50s, headline first, Risk folds into Recap when short -

def test_default_duration_is_45_and_range_is_35_to_50():
    assert (MIN_DURATION, MAX_DURATION) == (35, 50)
    assert build_finance_queue.DEFAULT_DURATION == 45


def test_build_theme_entry_short_video_merges_risk_into_recap(monkeypatch):
    monkeypatch.setattr("scripts.build_finance_queue.fundamentals.fetch_fundamentals", lambda symbol: _FAKE_DATA)
    entry = build_theme_entry("CDSL", "CDSL", duration_sec=40)
    segments = {s["label"]: s["content"] for s in parse_segments(entry["theme"])}
    assert list(segments) == ["Hook", "Key Metrics", "Revenue & Profit Growth", "Recap"]
    assert "do not" in segments["Recap"].lower() and "recommendation" in segments["Recap"].lower()


@pytest.mark.parametrize("duration", range(35, 51))
@pytest.mark.parametrize("facts", [[], ["A peer fact."], ["A peer fact.", "Another fact."]])
def test_build_theme_entry_segments_sum_to_duration_with_sane_minimums(monkeypatch, duration, facts):
    monkeypatch.setattr("scripts.build_finance_queue.fundamentals.fetch_fundamentals", lambda symbol: _FAKE_DATA)
    monkeypatch.setattr("scripts.build_finance_queue.company_insights.did_you_know_beat",
                        lambda n, d: {"facts": facts, "narration": ""})
    entry = build_theme_entry("CDSL", "CDSL", duration_sec=duration)
    segments = parse_segments(entry["theme"])
    assert entry["duration_sec"] == duration
    assert sum(s["duration"] for s in segments) == duration
    assert min(s["duration"] for s in segments) >= 6
    assert segments[-1]["label"] == "Recap"


def test_build_theme_entry_long_did_you_know_is_shrunk_to_fit_a_35s_video(monkeypatch):
    monkeypatch.setattr("scripts.build_finance_queue.fundamentals.fetch_fundamentals", lambda symbol: _FAKE_DATA)
    monkeypatch.setattr("scripts.build_finance_queue.company_insights.did_you_know_beat",
                        lambda n, d: {"facts": ["A", "B"], "narration": ""})
    entry = build_theme_entry("CDSL", "CDSL", duration_sec=35)
    dyk = {s["label"]: s for s in parse_segments(entry["theme"])}["Did You Know"]
    assert dyk["duration"] == 7


def test_build_theme_entry_hook_demands_headline_in_first_6_to_8_words(monkeypatch):
    monkeypatch.setattr("scripts.build_finance_queue.fundamentals.fetch_fundamentals", lambda symbol: _FAKE_DATA)
    entry = build_theme_entry("CDSL", "CDSL")
    hook = {s["label"]: s["content"] for s in parse_segments(entry["theme"])}["Hook"]
    assert "first 6-8 words" in hook
