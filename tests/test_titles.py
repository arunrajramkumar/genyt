"""Search-friendly title cleanup for stock videos: no doubled company name,
no "Long-Term Wealth" filler, no emoji or dangling separators, and the short
name (not the legal name) leads the title."""
import json

import pytest

from pipeline import segmented_script as ss


def _polish(title: str, company_name: str) -> str:
    """Mirrors the production order: clean -> emoji -> polish -> truncate."""
    title = ss.clean_title(title)
    title = ss.strip_title_emoji(title)
    return ss._truncate_title(ss.polish_stock_title(title, company_name))


# The production titles from the YouTube Analytics review, each with the
# company name the pipeline actually carries (the registered name).
BAD_TITLES = [
    ("Karur Vysya Bank Ltd: Karur Vysya Bank: Profit Up 37.3% YoY – Can…",
     "Karur Vysya Bank Ltd", "Karur Vysya Bank: Profit Up 37.3% YoY – Can…"),
    ("Solar Industries India Ltd: Solar Industries India 89% Net Profit…",
     "Solar Industries India Ltd", "Solar Industries India 89% Net Profit…"),
    ("Waaree Energies Ltd: Waaree Energies Q3 Profit Jumps 133%: Margin…",
     "Waaree Energies Ltd", "Waaree Energies Q3 Profit Jumps 133%: Margin…"),
    ("24% Revenue Surge: Titan Company Ltd Long‑Term Wealth Play",
     "Titan Company Ltd", "24% Revenue Surge: Titan Company Ltd"),
    ("Asian Paints: Long‑Term Wealth Potential 📈",
     "Asian Paints", "Asian Paints Share Price Analysis"),
    ("Gabriel India Ltd: 26.5% ROCE Boost – Gabriel India Long‑Term Wealth?",
     "Gabriel India Ltd", "Gabriel India Ltd: 26.5% ROCE Boost?"),
]


@pytest.mark.parametrize("raw, company, expected", BAD_TITLES)
def test_bad_production_titles_are_cleaned(raw, company, expected):
    assert _polish(raw, company) == expected


def test_long_term_wealth_variants_are_all_stripped():
    for variant in ("Long-Term Wealth", "Long‑Term Wealth", "Long–Term Wealth", "long term wealth"):
        result = ss.strip_title_boilerplate(f"Titan Company Ltd: Is It {variant} Builder?")
        assert "wealth" not in result.lower(), variant


def test_long_term_growth_filler_tag_is_stripped_but_mid_sentence_use_is_kept():
    assert ss.strip_title_boilerplate("Titan Company Ltd – Long-Term Growth") == "Titan Company Ltd"
    kept = "Titan Company Ltd: Long-Term Growth Rate Hits 12%"
    assert ss.strip_title_boilerplate(kept) == kept


def test_strip_title_boilerplate_keeps_negative_numbers():
    title = "Premier Polyfilm: Profit -12% YoY"
    assert ss.strip_title_boilerplate(title) == title


def test_doubled_name_with_colon_collapses_to_single_name():
    assert ss._ensure_company_name_in_title("Karur Vysya Bank: Profit Up", "Karur Vysya Bank Ltd") == "Karur Vysya Bank: Profit Up"


def test_doubled_name_without_colon_after_first_collapses_to_single_name():
    result = ss._ensure_company_name_in_title("Solar Industries India Ltd: Solar Industries India 89% Net Profit", "Solar Industries India Ltd")
    assert result == "Solar Industries India 89% Net Profit"


def test_legal_name_present_in_title_is_not_prepended_again():
    title = "Karur Vysya Bank Q2 Results: Profit +37%"
    assert ss._ensure_company_name_in_title(title, "Karur Vysya Bank Ltd") == title


def test_missing_name_is_prepended_as_short_name_not_legal_name():
    result = ss._ensure_company_name_in_title("Profit Up 37% In A Quarter", "Karur Vysya Bank Ltd")
    assert result == "Karur Vysya Bank: Profit Up 37% In A Quarter"


def test_name_match_is_case_insensitive_and_word_bounded():
    # "Titan" must not count as present inside an unrelated word.
    assert ss._ensure_company_name_in_title("Titanium Stocks Rally", "Titan Company Ltd") == "Titan: Titanium Stocks Rally"


def test_polish_leaves_non_company_titles_untouched():
    title = "Long-Term Wealth Play: 5 Facts About Octopuses"
    assert ss.polish_stock_title(title, None) == title


def test_polish_is_a_no_op_for_clean_search_title():
    title = "Karur Vysya Bank Q2 Results: Profit +37% | Share Price"
    assert ss.polish_stock_title(title, "Karur Vysya Bank Ltd") == title


def test_polish_does_not_leave_a_dangling_separator_when_only_filler_remains():
    assert ss.polish_stock_title("Long-Term Wealth Play", "Asian Paints Ltd") == "Asian Paints Share Price Analysis"


def test_polish_never_invents_numbers():
    title = "Karur Vysya Bank Q2 Results: Profit +37%"
    result = ss.polish_stock_title(title, "Karur Vysya Bank Ltd")
    assert set(_numbers(result)) <= set(_numbers(title))


def _numbers(text):
    import re
    return re.findall(r"\d+(?:\.\d+)?", text)


def test_write_metadata_applies_polish_to_llm_title(monkeypatch):
    monkeypatch.setattr(
        ss.llm, "chat_json",
        lambda *a, **k: json.dumps({"title": "Asian Paints: Long‑Term Wealth Potential 📈", "description": "d", "tags": []}),
    )
    meta = ss._write_metadata([{"narration": "n"}], company_name="Asian Paints")
    assert meta["title"] == "Asian Paints Share Price Analysis"


def test_write_metadata_collapses_doubled_legal_name(monkeypatch):
    monkeypatch.setattr(
        ss.llm, "chat_json",
        lambda *a, **k: json.dumps({"title": "Karur Vysya Bank: Profit Up 37.3% YoY", "description": "d", "tags": []}),
    )
    meta = ss._write_metadata([{"narration": "n"}], company_name="Karur Vysya Bank Ltd")
    assert meta["title"] == "Karur Vysya Bank: Profit Up 37.3% YoY"


def test_metadata_prompt_steers_to_short_name_search_pattern():
    prompt = " ".join(ss.SYSTEM_PROMPT_METADATA.split())  # wrapping-agnostic
    assert "short name" in prompt
    assert "60 characters or fewer" in prompt
    assert "Karur Vysya Bank Q2 Results" in prompt


@pytest.mark.parametrize("raw, company, expected", [
    ("Lloyds Metals Profit Triples to ₹1,726 cr", "Lloyds Metals & Energy Ltd", "Lloyds Metals Profit Triples to ₹1,726 cr"),
    ("Tata Consumer Q2 Results", "Tata Consumer Products Ltd", "Tata Consumer Q2 Results"),
    ("Q2 Profit Up 12%", "Bank of Baroda", "Bank of Baroda: Q2 Profit Up 12%"),
    ("Profit Up 37%", "Karur Vysya Bank Ltd", "Karur Vysya Bank: Profit Up 37%"),
])
def test_shortened_company_name_counts_as_present(raw, company, expected):
    assert ss.polish_stock_title(raw, company) == expected
