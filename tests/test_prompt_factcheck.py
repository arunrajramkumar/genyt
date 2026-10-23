"""Prompt figures vs Screener (2026-10-23: stale Cummins/Lloyds quarters,
Bajaj P/E mismatch)."""
import pytest

from pipeline import prompt_factcheck as pf

D = ["2025-06-30", "2025-09-30", "2025-12-31", "2026-03-31", "2026-06-30"]
CUMMINS = {"ratios": {"Stock P/E": "50.6", "Market Cap": "₹ 1,19,000 Cr.", "Current Price": "₹ 4,290"},
           "quarterly": {"quarter_dates": D, "sales_cr": ["2860", "1", "1", "1", "3375"],
                         "net_profit_cr": ["589", "1", "1", "1", "543"]}}
OLD_CUMMINS = ("“In Q2 FY2027, total revenue was approximately ₹3,170 crore, up around 26.4% year-on-year. "
               "Net profit reached approximately ₹622 crore, increasing around 38.5% year-on-year.”")


def test_stale_quarter_growth_and_amounts_are_flagged():
    warnings = pf.check(OLD_CUMMINS, CUMMINS)
    text = "\n".join(warnings)
    assert "Q2 FY27" in text and "Q1 FY27" in text
    assert "+38.5%" in text and "-7.8%" in text
    assert "+26.4%" in text and "+18%" in text
    assert "₹622 crore" in text and "₹543 crore" in text


def test_matching_prompt_has_no_warnings():
    good = ("“In Q1 FY27, revenue rose about 18% to ₹3,375 crore, but net profit fell 8% to about ₹543 crore. "
            "The stock trades at a P/E of about 50, with a market value of roughly ₹1.2 lakh crore.”")
    assert pf.check(good, CUMMINS) == []


@pytest.mark.parametrize("phrase, expected", [
    ("profit slipped 8% to ₹543 crore", []),
    ("profit grew 8% to ₹543 crore", ["+8%"]),
])
def test_fall_words_give_the_quoted_growth_its_sign(phrase, expected):
    warnings = pf.check(phrase, CUMMINS)
    assert all(e in "\n".join(warnings) for e in expected) and bool(warnings) == bool(expected)


def test_pe_market_cap_and_price_mismatches():
    prompt = ("The stock is trading at ₹5,200 per share. Its P/E ratio is approximately 70, "
              "with a market capitalisation of around ₹2 lakh crore.")
    text = "\n".join(pf.check(prompt, CUMMINS))
    assert "P/E of about 70" in text and "50.6" in text
    assert "₹2,00,000" in text or "₹200,000 crore" in text
    assert "share price" in text


def test_times_earnings_form_and_tolerance():
    assert pf.check("It trades at about 27 times earnings.", {"ratios": {"Stock P/E": "29.5"}}) == []
    assert pf.check("It trades at about 22 times earnings.", {"ratios": {"Stock P/E": "29.5"}})


def test_older_quarter_flagged_as_stale():
    data = dict(CUMMINS, quarterly=dict(CUMMINS["quarterly"], quarter_dates=D[:-1] + ["2026-12-31"]))
    assert "stale" in "\n".join(pf.check("In Q1 FY27 profit rose.", data))


def test_nothing_to_compare_or_bad_data_never_raises():
    assert pf.check("", CUMMINS) == [] and pf.check("text", {}) == []
    assert pf.check("profit rose 10% to ₹5 crore", {"quarterly": {"net_profit_cr": ["x"]}}) == []
