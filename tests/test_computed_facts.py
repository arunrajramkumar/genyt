import pytest

from pipeline import company_insights as ci
from scripts.build_finance_queue import build_theme_entry

REAL_DID_YOU_KNOW_FACTS = ci.did_you_know_facts

# Solar Industries-like figures: profit 666 cr now vs 352 a year ago on sales 3668 vs 2158.
SOLAR = {
    "symbol": "SOLARINDS",
    "ratios": {
        "Stock P/E": "103", "Current Price": "₹ 19,132", "High / Low": "₹ 21,500 / 14,000",
        "Market Cap": "₹ 1,75,858 Cr.",
    },
    "quarterly": {
        "sales_cr": ["2,158", "2,200", "2,300", "3,000", "3,668"],
        "net_profit_cr": ["352", "340", "360", "500", "666"],
    },
}


def test_payback_fact_for_rich_valuation():
    fact = ci.payback_fact(SOLAR)
    assert "P/E ratio of 103" in fact and "103 years" in fact and "if profit stayed flat" in fact


@pytest.mark.parametrize("pe", ["39.9", "", "n/a", "5000"])
def test_payback_fact_skipped_for_ordinary_or_bad_pe(pe):
    assert ci.payback_fact({"ratios": {"Stock P/E": pe}}) == ""
    assert ci.payback_fact({}) == "" and ci.payback_fact(None) == ""


def test_profit_per_day_in_crore():
    fact = ci.profit_per_day_fact(SOLAR)
    assert "Rs 666 crore" in fact and "Rs 7.3 crore" in fact and "every single day" in fact


def test_profit_per_day_large_and_lakh_scales():
    big = ci.profit_per_day_fact({"quarterly": {"net_profit_cr": ["9,100"]}})
    assert "Rs 9,100 crore" in big and "Rs 100 crore" in big
    small = ci.profit_per_day_fact({"quarterly": {"net_profit_cr": ["45"]}})
    assert "Rs 49 lakh" in small
    assert ci.profit_per_day_fact({"quarterly": {"net_profit_cr": ["5"]}}) == ""


@pytest.mark.parametrize("profit", [["-20"], [], ["0"], ["x"]])
def test_profit_per_day_skipped_for_loss_or_missing(profit):
    assert ci.profit_per_day_fact({"quarterly": {"net_profit_cr": profit}}) == ""
    assert ci.profit_per_day_fact({}) == ""


def test_margin_fact_up():
    fact = ci.margin_fact(SOLAR)
    assert "Rs 18 became profit" in fact and "up from Rs 16" in fact


def test_margin_fact_down_and_flat_and_loss():
    # latest is the last element: now = 20/200 = 10%, year ago = values[-5] = 30/100 = 30%
    fact = ci.margin_fact({"quarterly": {"sales_cr": ["100", "1", "1", "1", "200"],
                                         "net_profit_cr": ["30", "1", "1", "1", "20"]}})
    assert "Rs 10 became profit" in fact and "down from Rs 30" in fact
    flat = {"quarterly": {"sales_cr": ["100", "1", "1", "1", "100"], "net_profit_cr": ["10", "1", "1", "1", "10.4"]}}
    assert ci.margin_fact(flat) == ""
    loss = {"quarterly": {"sales_cr": ["100", "1", "1", "1", "100"], "net_profit_cr": ["10", "1", "1", "1", "-5"]}}
    assert ci.margin_fact(loss) == ""
    assert ci.margin_fact({"quarterly": {"sales_cr": ["1"], "net_profit_cr": ["1"]}}) == ""


def test_high_low_below_high():
    assert "about 11% below its 52-week high of Rs 21,500" in ci._high_low_fact(SOLAR)


def test_high_low_near_high_and_above_low_and_none():
    def d(price, hl):
        return {"ratios": {"Current Price": price, "High / Low": hl}}
    assert "within 3% of its 52-week high of Rs 1,000" in ci._high_low_fact(d("₹ 980", "₹ 1,000 / 500"))
    assert "up about 100% from its 52-week low of Rs 100" in ci._high_low_fact(d("₹ 200", "₹ 215 / 100"))
    assert ci._high_low_fact(d("₹ 150", "₹ 160 / 120")) == ""  # unremarkable


@pytest.mark.parametrize("price,hl", [("₹ 50", "₹ 100 / 60"), ("₹ 500", "₹ 100 / 60"), ("", "₹ 100 / 60"),
                                      ("₹ 80", "₹ 100"), ("₹ 80", "")])
def test_high_low_skipped_for_inconsistent_data(price, hl):
    assert ci._high_low_fact({"ratios": {"Current Price": price, "High / Low": hl}}) == ""


def test_computed_facts_order_and_resilience(monkeypatch):
    facts = ci.computed_facts(SOLAR)
    markers = ["P/E", "every single day", "became profit", "52-week"]
    assert len(facts) == 4 and all(m in f for m, f in zip(markers, facts))
    assert ci.computed_facts({}) == []

    def boom(data):
        raise RuntimeError("x")
    monkeypatch.setattr(ci, "payback_fact", boom)
    assert len(ci.computed_facts(SOLAR)) == 3  # one failing fact doesn't drop the rest


def test_did_you_know_facts_lists_all_candidates_story_first(monkeypatch):
    monkeypatch.setattr(ci, "find_story_fact", lambda n: "A story.")
    monkeypatch.setattr(ci.fundamentals, "fetch_peers", lambda d: [])
    assert REAL_DID_YOU_KNOW_FACTS("Solar Industries India Ltd", SOLAR)[0] == "A story."
    assert len(REAL_DID_YOU_KNOW_FACTS("Solar Industries India Ltd", SOLAR)) == 5


def test_did_you_know_facts_falls_back_to_computed_when_story_and_peers_missing(monkeypatch):
    monkeypatch.setattr(ci, "find_story_fact", lambda n: "")
    monkeypatch.setattr(ci.fundamentals, "fetch_peers", lambda d: [])
    facts = REAL_DID_YOU_KNOW_FACTS("Solar Industries India Ltd", SOLAR)
    assert len(facts) == 4 and "P/E ratio of 103" in facts[0] and "every single day" in facts[1]


def test_recap_asks_a_comment_question(monkeypatch):
    monkeypatch.setattr("scripts.build_finance_queue.fundamentals.fetch_fundamentals", lambda s: SOLAR)
    entry = build_theme_entry("SOLARINDS", "Solar Industries India Ltd")
    recap = entry["theme"].split("Recap")[-1]
    assert "yes-or-no question" in recap and "never a recommendation" in recap
