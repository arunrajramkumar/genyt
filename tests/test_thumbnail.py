"""Stock thumbnails: every number from scraped fundamentals, fixed labels, and
the renderer copes with any spec (2026-10-23 review: junk labels, a "117%"
ROCE copied from a prompt example, a five-line duplicated-name title)."""
import json
from datetime import date
from pathlib import Path

import pytest
from PIL import Image

from pipeline import ai_thumbnail, producer, segmented_script, textcard
from pipeline import company_insights as ci
from pipeline import thumbnail_spec as ts

# conftest stubs these per test; capture the real ones at import time.
REAL_SPEC = producer._stock_thumbnail_spec
REAL_SUMMARY = ci.business_summary
DATES = ["2025-09-30", "2025-12-31", "2026-03-31", "2026-06-30", "2026-09-30"]
TODAY = date(2026, 10, 23)


def _data(sales, profit, ratios=None, dates=DATES):
    return {"ratios": ratios or {"ROCE": "69.2 %", "ROE": "76.4 %", "Stock P/E": "10.6"},
            "quarterly": {"quarter_dates": dates, "sales_cr": sales, "net_profit_cr": profit}}


HZ = _data(["8525", "1", "1", "1", "14961"], ["2632", "1", "1", "1", "6317"])


def test_spec_from_fundamentals_only():
    spec = ts.build(HZ, "Hindustan Zinc Ltd", "Hindustan Zinc Q2 Results: Profit +140% | Can It Last?", today=TODAY)
    assert spec == {"name": "HINDUSTAN ZINC", "tag": "Q2 FY27 RESULTS", "hero_value": "+140%",
                    "hero_label": "NET PROFIT YoY", "direction": "up", "chips": ["P/E 10.6", "ROCE 69.2%"],
                    "hook": "CAN IT LAST?"}


def test_decline_is_red_with_exact_precision():
    spec = ts.build(_data(["500", "1", "1", "1", "560"], ["60", "1", "1", "1", "47"]), "DOMS Industries Ltd",
                    "DOMS Industries Q2 Profit Down 21.7%", today=TODAY)
    assert spec["hero_value"] == "-21.7%" and spec["direction"] == "down" and spec["hook"] == "WHAT HAPPENED?"


def test_one_off_profit_falls_back_to_revenue():
    spec = ts.build(_data(["40", "1", "1", "1", "69"], ["1", "1", "1", "1", "567"]), "Swan Defence Ltd", today=TODAY)
    assert spec["hero_value"] == "+72.5%" and spec["hero_label"] == "REVENUE YoY"


def test_no_growth_data_uses_ratio_without_arrow_and_skips_duplicate_chip():
    spec = ts.build(_data([], []), "X Ltd", today=TODAY)
    assert spec["hero_value"] == "69.2%" and spec["hero_label"] == "ROCE" and spec["direction"] == ""
    assert spec["chips"] == ["P/E 10.6", "ROE 76.4%"]


def test_nothing_to_headline_gives_empty_spec():
    assert ts.build(_data([], [], ratios={"Stock P/E": "10"}), "X Ltd", today=TODAY) == {}
    assert ts.build({}, "X Ltd") == {} and ts.build(HZ, "") == {}


@pytest.mark.parametrize("quarter_end, label", [("2026-06-30", "Q1 FY27"), ("2026-09-30", "Q2 FY27"),
                                                ("2026-12-31", "Q3 FY27"), ("2027-03-31", "Q4 FY27"),
                                                ("2026-05-31", ""), ("bad", "")])
def test_fiscal_quarter_label(quarter_end, label):
    assert ts.fiscal_quarter_label(quarter_end) == label


def test_tag_ipo_and_stale_results():
    assert ts.build(HZ, "Acme Ltd", "Acme IPO: GMP 71%", today=TODAY)["tag"] == "IPO"
    assert ts.build(HZ, "Hindustan Zinc", today=date(2027, 6, 1))["tag"] == ""


def test_long_title_tail_or_numbers_never_become_the_hook():
    assert ts.build(HZ, "X Ltd", "X: Is This Growth Story Really Going To Continue?", today=TODAY)["hook"] == "WHAT'S BEHIND IT?"
    assert ts.build(HZ, "X Ltd", "X: Up 140%?", today=TODAY)["hook"] == "WHAT'S BEHIND IT?"


@pytest.mark.parametrize("spec", [
    ts.build(HZ, "Hindustan Zinc Ltd", "Hindustan Zinc: Can It Last?", today=TODAY),
    {"name": "AVERY VERY LONG NAMED INFRASTRUCTURE AND ENGINEERING CORPORATION", "tag": "IPO",
     "hero_value": "+12345.6%", "hero_label": "REVENUE YoY", "direction": "up", "chips": ["P/E 1234.5", "ROCE 99.9%"],
     "hook": "A VERY LONG HOOK QUESTION THAT HAS TO WRAP?"},
    {"name": "X", "hero_value": "9%", "hero_label": "ROCE", "direction": "", "chips": [], "hook": "", "tag": ""},
])
@pytest.mark.parametrize("size", [(1080, 1920), (720, 1280), (1280, 720)])
def test_renderer_handles_any_spec_and_size(tmp_path, spec, size):
    out = textcard.render_stock_thumbnail(spec, *size, tmp_path / "t.png")
    assert Image.open(out).size == size


def test_renderer_uses_background(tmp_path):
    bg = tmp_path / "bg.png"
    Image.new("RGB", (400, 400), (200, 0, 0)).save(bg)
    out = textcard.render_stock_thumbnail({"hero_value": "+5%", "direction": "up"}, 360, 640, tmp_path / "t.png", bg)
    r, g, b = Image.open(out).convert("RGB").getpixel((5, 5))
    assert r > g and r > b  # red background shows through the dim


# ---- producer wiring ----

def test_spec_only_for_stock_scripts(monkeypatch):
    monkeypatch.setattr(producer.fundamentals, "fetch_fundamentals", lambda s: HZ)
    assert REAL_SPEC({"title": "t"}) == {}
    assert REAL_SPEC({"symbol": "HINDZINC", "company_name": "Hindustan Zinc", "title": "t"})["name"] == "HINDUSTAN ZINC"


def test_spec_failure_falls_back_quietly(monkeypatch):
    def boom(s):
        raise RuntimeError("screener down")
    monkeypatch.setattr(producer.fundamentals, "fetch_fundamentals", boom)
    logs = []
    assert REAL_SPEC({"symbol": "X", "company_name": "X", "title": "t"}, logs.append) == {}
    assert "generic hook card" in logs[0]


# ---- business-specific AI background ----

def test_prompt_uses_business_description_when_known(monkeypatch, tmp_path):
    prompts = []

    def fake_pollinations(company_name, out_path, on_progress, width, height):
        prompts.append(ai_thumbnail._build_prompt(company_name))
    monkeypatch.setattr(ai_thumbnail.config, "GEMINI_API_KEY", "")
    monkeypatch.setattr(ai_thumbnail.config, "CLOUDFLARE_ACCOUNT_ID", "")
    monkeypatch.setattr(ai_thumbnail, "_generate_via_pollinations", fake_pollinations)
    ai_thumbnail.generate_hook_background("Hindustan Zinc", tmp_path / "a.png", lambda m: None,
                                          business_hint="Hindustan Zinc is an Indian zinc mining company.")
    ai_thumbnail.generate_hook_background("Other Co", tmp_path / "b.png", lambda m: None)
    assert "zinc mining" in prompts[0] and "not a stock chart" in prompts[0]
    assert "trading floor" in prompts[1] and "zinc" not in prompts[1]


def test_business_summary_first_sentence_without_parentheticals(monkeypatch):
    monkeypatch.setattr(ci, "_wikipedia_extract", lambda n: (
        "Wikipedia article 'Hindustan Zinc':\nHindustan Zinc Limited (HZL) is an Indian integrated zinc producer. "
        "It is based in Udaipur."))
    assert REAL_SUMMARY("Hindustan Zinc") == "Hindustan Zinc Limited is an Indian integrated zinc producer."
    monkeypatch.setattr(ci, "_wikipedia_extract", lambda n: "")
    assert REAL_SUMMARY("Nobody") == ""




# ---- prompt examples are never facts ----

def test_example_numbers_are_not_extracted_or_grounded():
    content = ('For JNK (NSE: JNKINDIA), translate each metric (e.g. "ROCE of 117% -- meaning huge profit"). '
               "Facts:\nROCE: 17.5 %")
    stripped = segmented_script._strip_examples(content)
    assert "117" not in stripped
    facts = segmented_script._extract_facts(stripped)
    assert facts and not any("117" in f for f in facts)
    assert not segmented_script._is_grounded("ROCE 117%", stripped)
