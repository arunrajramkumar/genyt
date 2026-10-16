from unittest import mock

import pytest

from pipeline import segmented_script as ss


# -- parse_segments: structured timed-prompt detection -------------------------

STOCK_PROMPT = """0-5 seconds - Hook
"This steel company just signed a ₹2,400 crore order book."

5-20 seconds - Company overview
Mid-cap steel manufacturer with plants in Odisha and Jharkhand.

20-35 seconds - Financials
Revenue grew 18% YoY to ₹3,120 crore. Debt-free with ROE of 22%.

35-50 seconds - Growth drivers
Government infra spending should drive utilization higher.

End screen:
This is not investment advice.
"""


def test_parse_segments_extracts_labels_and_durations():
    segments = ss.parse_segments(STOCK_PROMPT)
    assert [s["label"] for s in segments] == [
        "Hook", "Company overview", "Financials", "Growth drivers", "End screen",
    ]
    assert segments[0]["duration"] == 5
    assert segments[1]["duration"] == 15
    assert segments[-1]["duration"] is not None  # End screen has no header range -> fallback duration


def test_parse_segments_requires_at_least_three_headers():
    short_prompt = "0-5 seconds - Hook\nSomething.\n\n5-10 seconds - Body\nMore."
    assert ss.parse_segments(short_prompt) == []


def test_parse_segments_ignores_freeform_prompt():
    assert ss.parse_segments("Tell me about the history of Rome, no particular structure.") == []


def test_parse_segments_skips_empty_sections():
    prompt = (
        "0-5 seconds - Hook\n\n"
        "5-10 seconds - Body\nReal content here.\n\n"
        "10-15 seconds - More\nMore content.\n\n"
        "15-20 seconds - End\nFinal content.\n"
    )
    segments = ss.parse_segments(prompt)
    assert "Hook" not in [s["label"] for s in segments]


# -- grounding / no-fabrication guardrails -------------------------------------

def test_extract_verbatim_pulls_quote_that_starts_its_own_line():
    # Real themes wrap a whole pre-written narration paragraph in quotes on its
    # own line (see CDSL/Kalyan Jewellers entries in themes.yaml).
    content = 'Some lead-in text.\n\n"This steel company just signed a huge deal."\n\nMore text.'
    assert ss._extract_verbatim(content) == "This steel company just signed a huge deal."


def test_extract_verbatim_ignores_quote_embedded_mid_sentence():
    # A section/chapter title quoted inline (e.g. 'Section 1.1 "What is
    # Mathematics?" -- explain this...') is not a pre-written narration to
    # reproduce verbatim -- regression test for a bug that collapsed education
    # scene narration down to just the section title.
    content = 'Section 1.1 "What is Mathematics?" -- explain this step by step.'
    assert ss._extract_verbatim(content) == ""


def test_extract_verbatim_returns_empty_when_no_quote():
    assert ss._extract_verbatim("No quotes in this content at all.") == ""


@pytest.mark.parametrize("content,expected", [
    ("Revenue grew to ₹3,120 crore this year.", "₹3,120 crore"),
    ("ROE 18.5 reported this quarter, no other figures.", "ROE"),
    ("The company is debt-free with export exposure.", "Debt-Free Balance Sheet"),
    ("Just a plain sentence with nothing extractable.", ""),
])
def test_extract_facts_finds_known_patterns(content, expected):
    facts = ss._extract_facts(content)
    if expected:
        assert any(expected.split()[0].lower() in f.lower() or expected in f for f in facts)
    else:
        assert facts == []


def test_extract_facts_returns_all_distinct_metrics_in_order():
    content = "P/E ratio stands at 20.09, Price-to-Book at 5.51, EV/EBITDA 13.48, and ROE of 30.2%."
    facts = ss._extract_facts(content, limit=4)
    assert len(facts) == 4
    assert any("20.09" in f for f in facts)
    assert any("5.51" in f for f in facts)
    assert any("13.48" in f for f in facts)
    assert any("30.2" in f for f in facts)


def test_extract_facts_longer_match_wins_over_contained_shorter_one():
    # The generic "%" pattern would also match the "30.2%" inside "ROE 30.2%"
    # at the same position -- the longer, more specific ROE match should win
    # and the bare percentage should not show up as a separate duplicate fact.
    content = "Strong ROE of 30.2% this quarter."
    facts = ss._extract_facts(content)
    assert facts == ["ROE of 30.2%"] or facts == ["ROE 30.2%"] or "30.2%" in facts[0]
    assert len(facts) == 1


def test_extract_facts_caps_at_limit():
    content = "P/E 10, P/B 2, EV/EBITDA 5, ROE 20%, ROCE 25%, NSE: ABC"
    facts = ss._extract_facts(content, limit=3)
    assert len(facts) == 3


def test_extract_facts_plain_price_is_captured():
    content = "The price has climbed from ₹889.25 on Sep 18 to ₹913.05 on Sep 22."
    facts = ss._extract_facts(content, limit=4)
    assert any("889.25" in f for f in facts)
    assert any("913.05" in f for f in facts)


def test_extract_facts_does_not_truncate_lakh_crore_to_lakh():
    # Regression: "₹15 lakh crore" (how large-cap market caps are quoted)
    # must not be truncated to "₹15 lakh" -- a 100,000x magnitude error.
    content = "TCS has a market cap of ₹15 lakh crore."
    facts = ss._extract_facts(content)
    assert any("lakh crore" in f for f in facts)
    assert not any(f.strip().endswith("₹15 lakh") for f in facts)


def test_extract_facts_bare_percent_keeps_negative_sign():
    # Regression: fundamentals.py phrases declines as "grew -12.3% YoY" -- the
    # bare "%" fallback pattern must not drop the minus sign and silently
    # display a decline as a positive figure.
    content = "Revenue grew -12.3% year-on-year this quarter."
    facts = ss._extract_facts(content)
    assert any("-12.3%" in f for f in facts)


def test_extract_facts_bare_percent_without_sign_still_works():
    content = "Revenue grew 12.3% year-on-year this quarter."
    facts = ss._extract_facts(content)
    assert any("12.3%" in f for f in facts)


def test_is_grounded_accepts_numbers_present_in_source():
    content = "Revenue grew 18% YoY to ₹3,120 crore."
    assert ss._is_grounded("₹3,120 crore revenue", content) is True


def test_is_grounded_rejects_fabricated_numbers():
    content = "Revenue grew 18% YoY to ₹3,120 crore."
    # 9999 never appears in the source content -> must be flagged as ungrounded.
    assert ss._is_grounded("₹9999 crore revenue", content) is False


def test_is_grounded_rejects_text_containing_a_question_mark():
    assert ss._is_grounded("Is this real?", "Anything") is False


def test_is_grounded_allows_text_with_no_digits():
    assert ss._is_grounded("Debt-free balance sheet", "Some unrelated content") is True


def test_segment_system_prompt_forbids_answering_unanswered_exercise_questions():
    # Regression for a real production bug: source textbook content posed an
    # unanswered "Figure it Out" question ("Which sequence do you get when you
    # add the All 1's sequence up and down?") and the model invented a
    # plausible-but-wrong answer ("gives square numbers") rather than treating
    # it as an open question for the viewer.
    prompt = ss._segment_system_prompt(target_words=40, label="Exercises")
    assert "never invent a resolution" in prompt.lower()


def test_fix_double_escaped_unicode_decodes_literal_escapes():
    assert ss._fix_double_escaped_unicode("\\u20b9412 crore") == "₹412 crore"


def test_fix_double_escaped_unicode_leaves_normal_text_untouched():
    assert ss._fix_double_escaped_unicode("plain text") == "plain text"


@pytest.mark.parametrize("label,expected_options", [
    ("Hook", ["industrial factory establishing shot", "city skyline aerial view", "busy trading floor"]),
    ("Company overview", ["steel factory workers", "office team meeting", "business professional presentation"]),
    ("Financials", ["stock market chart growth", "stock ticker screen", "financial analyst at desk"]),
    ("Risk factors", ["warning sign caution", "stormy stock market graph", "red downward chart"]),
])
def test_fallback_visual_matches_label_keywords(label, expected_options):
    assert ss._fallback_visual(label) in expected_options


def test_fallback_visual_defaults_for_unmatched_label():
    assert ss._fallback_visual("Something unrelated") == "business professional presentation"


# -- _write_segment_scene: LLM call shape + grounding enforcement --------------

def _mock_llm_response(payload: dict):
    import json
    return json.dumps(payload)


def test_write_segment_scene_uses_model_output_when_grounded(monkeypatch):
    monkeypatch.setattr(
        ss.llm, "chat_json",
        lambda *a, **k: _mock_llm_response({
            "narration": "Revenue grew to ₹3,120 crore this year.",
            "visual_query": "factory production line",
            "on_screen_text": "₹3,120 crore",
        }),
    )
    scene = ss._write_segment_scene("Financials", "Revenue grew to ₹3,120 crore this year.", 15)
    assert scene["on_screen_text"] == "₹3,120 crore"
    assert scene["visual_query"] == "factory production line"


def test_write_segment_scene_strips_ungrounded_on_screen_text_and_uses_fact_fallback(monkeypatch):
    monkeypatch.setattr(
        ss.llm, "chat_json",
        lambda *a, **k: _mock_llm_response({
            "narration": "Revenue grew nicely this year.",
            "visual_query": "factory",
            "on_screen_text": "₹9999999 crore",  # fabricated, not present in source
        }),
    )
    scene = ss._write_segment_scene("Financials", "Revenue grew 18% YoY to ₹3,120 crore.", 15)
    # The fabricated figure must never reach on_screen_text; the real figure
    # extracted directly from the source content should be used instead.
    assert "9999999" not in scene["on_screen_text"]
    assert "3,120" in scene["on_screen_text"]


def test_write_segment_scene_shows_all_metrics_when_content_has_several(monkeypatch):
    """Regression test: a real published video's valuation scene spoke P/E,
    P/B, EV/EBITDA, and ROE, but the model's on_screen_text only ever named
    one of them (e.g. P/E), so P/B and the others were spoken but never shown
    on screen. When the source content names multiple metrics, all of them
    should end up on screen, not just whichever single one the model picked."""
    monkeypatch.setattr(
        ss.llm, "chat_json",
        lambda *a, **k: _mock_llm_response({
            "narration": "P/E ratio stands at 20.09, Price-to-Book at 5.51, EV/EBITDA 13.48, ROE 30.2%.",
            "visual_query": "stock market chart",
            "on_screen_text": "P/E 20.09",
        }),
    )
    content = "P/E ratio stands at 20.09, Price-to-Book at 5.51, EV/EBITDA 13.48, and ROE of 30.2%."
    scene = ss._write_segment_scene("Valuation", content, 15)
    assert "20.09" in scene["on_screen_text"]
    assert "5.51" in scene["on_screen_text"]


def test_write_segment_scene_overrides_narration_with_verbatim_quote(monkeypatch):
    monkeypatch.setattr(
        ss.llm, "chat_json",
        lambda *a, **k: _mock_llm_response({
            "narration": "A paraphrased version the model made up.",
            "visual_query": "factory",
            "on_screen_text": "",
        }),
    )
    content = '"This steel company just signed a huge order book deal." extra context'
    scene = ss._write_segment_scene("Hook", content, 5)
    assert scene["narration"] == "This steel company just signed a huge order book deal."


def test_write_segment_scene_retries_on_empty_narration_then_succeeds(monkeypatch):
    responses = iter([
        _mock_llm_response({"narration": "", "visual_query": "x", "on_screen_text": ""}),
        _mock_llm_response({"narration": "Second attempt works.", "visual_query": "factory", "on_screen_text": ""}),
    ])
    monkeypatch.setattr(ss.llm, "chat_json", lambda *a, **k: next(responses))
    scene = ss._write_segment_scene("Overview", "Some facts.", 10)
    assert scene["narration"] == "Second attempt works."


def test_write_segment_scene_raises_after_three_bad_attempts(monkeypatch):
    monkeypatch.setattr(ss.llm, "chat_json", lambda *a, **k: "not json at all")
    with pytest.raises(RuntimeError, match="3 times in a row"):
        ss._write_segment_scene("Overview", "Some facts.", 10)


def test_write_segment_scene_keeps_grounded_diagram(monkeypatch):
    diagram = {"kind": "number_line", "min": 1, "max": 10, "highlight": [5]}
    monkeypatch.setattr(
        ss.llm, "chat_json",
        lambda *a, **k: _mock_llm_response({
            "narration": "The numbers go from 1 to 10.",
            "visual_query": "classroom",
            "on_screen_text": "",
            "diagram": diagram,
        }),
    )
    scene = ss._write_segment_scene("Numbers", "Numbers from 1 to 10, with 5 in the middle.", 10)
    assert scene["diagram"] == diagram


def test_write_segment_scene_drops_ungrounded_diagram(monkeypatch):
    # 999 never appears in the source content -> the diagram must be rejected
    # entirely rather than shipping a fabricated number on screen.
    diagram = {"kind": "number_line", "min": 1, "max": 999}
    monkeypatch.setattr(
        ss.llm, "chat_json",
        lambda *a, **k: _mock_llm_response({
            "narration": "The numbers go from 1 to 10.",
            "visual_query": "classroom",
            "on_screen_text": "",
            "diagram": diagram,
        }),
    )
    scene = ss._write_segment_scene("Numbers", "Numbers from 1 to 10.", 10)
    assert scene["diagram"] is None


def test_write_segment_scene_defaults_diagram_to_none_when_absent(monkeypatch):
    monkeypatch.setattr(
        ss.llm, "chat_json",
        lambda *a, **k: _mock_llm_response({"narration": "Plain scene.", "visual_query": "classroom", "on_screen_text": ""}),
    )
    scene = ss._write_segment_scene("Overview", "Some facts.", 10)
    assert scene["diagram"] is None


def test_write_segment_scene_includes_global_instructions_in_user_prompt(monkeypatch):
    captured = {}

    def fake_chat_json(system_prompt, user_prompt, max_tokens, timeout):
        captured["user_prompt"] = user_prompt
        return _mock_llm_response({"narration": "n", "visual_query": "v", "on_screen_text": ""})

    monkeypatch.setattr(ss.llm, "chat_json", fake_chat_json)
    ss._write_segment_scene("Overview", "Some facts.", 10, global_instructions="curious and upbeat teacher tone")
    assert "curious and upbeat teacher tone" in captured["user_prompt"]


def test_write_segment_scene_omits_global_instructions_section_when_not_given(monkeypatch):
    captured = {}

    def fake_chat_json(system_prompt, user_prompt, max_tokens, timeout):
        captured["user_prompt"] = user_prompt
        return _mock_llm_response({"narration": "n", "visual_query": "v", "on_screen_text": ""})

    monkeypatch.setattr(ss.llm, "chat_json", fake_chat_json)
    ss._write_segment_scene("Overview", "Some facts.", 10)
    assert "Overall tone/style" not in captured["user_prompt"]


def test_write_segmented_script_assembles_title_description_tags(monkeypatch):
    monkeypatch.setattr(
        ss, "_write_segment_scene",
        lambda label, content, duration, global_instructions=None: {"narration": f"N:{label}", "visual_query": "v", "on_screen_text": ""},
    )
    monkeypatch.setattr(ss.youtube_insights, "get_combined_guidance", lambda: "")
    monkeypatch.setattr(
        ss, "_write_metadata",
        lambda scenes, style_guidance="", company_name=None: {"title": "Premier Polyfilm: What Long-Term Investors Watch",
                         "description": "desc", "tags": ["premier polyfilm", "stock analysis"]},
    )
    segments = [{"label": "Hook", "content": "c", "duration": 5}]
    script = ss.write_segmented_script("some prompt", segments)
    assert script["title"] == "Premier Polyfilm: What Long-Term Investors Watch"
    assert script["description"] == "desc"
    assert script["tags"] == ["premier polyfilm", "stock analysis"]
    assert script["scenes"][0]["narration"].startswith("N:Hook")


def test_write_segmented_script_extracts_company_name_from_prompt_for_metadata(monkeypatch):
    """build_finance_queue.py always writes the theme as
    "Company Name (NSE: SYMBOL)..." -- that should be threaded into
    _write_metadata so it can force the company name into the title even if
    the metadata model drifts onto a vaguer hook phrase."""
    monkeypatch.setattr(
        ss, "_write_segment_scene",
        lambda label, content, duration, global_instructions=None: {"narration": "n", "visual_query": "v", "on_screen_text": ""},
    )
    monkeypatch.setattr(ss.youtube_insights, "get_combined_guidance", lambda: "")
    captured = {}

    def fake_write_metadata(scenes, style_guidance="", company_name=None):
        captured["company_name"] = company_name
        return {"title": "T", "description": "d", "tags": []}

    monkeypatch.setattr(ss, "_write_metadata", fake_write_metadata)
    segments = [{"label": "Hook", "content": "c", "duration": 5}]
    script = ss.write_segmented_script("Premier Polyfilm (NSE: PREMIER). Write an opening question.", segments)
    assert captured["company_name"] == "Premier Polyfilm"
    # Also surfaced on the returned script dict -- producer.py's AI-thumbnail
    # background generation needs it to build an image prompt.
    assert script["company_name"] == "Premier Polyfilm"


def test_write_segmented_script_passes_channel_style_guidance_into_metadata(monkeypatch):
    monkeypatch.setattr(
        ss, "_write_segment_scene",
        lambda label, content, duration, global_instructions=None: {"narration": "n", "visual_query": "v", "on_screen_text": ""},
    )
    monkeypatch.setattr(ss.youtube_insights, "get_combined_guidance", lambda: "- Lead with a number")
    captured = {}

    def fake_write_metadata(scenes, style_guidance="", company_name=None):
        captured["style_guidance"] = style_guidance
        return {"title": "T", "description": "d", "tags": []}

    monkeypatch.setattr(ss, "_write_metadata", fake_write_metadata)
    segments = [{"label": "Hook", "content": "c", "duration": 5}]
    ss.write_segmented_script("some prompt", segments)
    assert captured["style_guidance"] == "- Lead with a number"


def test_write_segmented_script_threads_global_instructions_into_every_scene_call(monkeypatch):
    captured = []
    monkeypatch.setattr(
        ss, "_write_segment_scene",
        lambda label, content, duration, global_instructions=None: captured.append(global_instructions) or
        {"narration": "n", "visual_query": "v", "on_screen_text": ""},
    )
    monkeypatch.setattr(ss.youtube_insights, "get_combined_guidance", lambda: "")
    monkeypatch.setattr(ss, "_write_metadata", lambda scenes, style_guidance="", company_name=None: {"title": "T", "description": "d", "tags": []})
    segments = [{"label": "Hook", "content": "c1", "duration": 5}, {"label": "Body", "content": "c2", "duration": 10}]
    ss.write_segmented_script("some prompt", segments, global_instructions="curious and upbeat teacher tone")
    assert captured == ["curious and upbeat teacher tone", "curious and upbeat teacher tone"]


# -- clean_title: strips hashtags and collapses whitespace ---------------------

def test_clean_title_strips_hashtags():
    assert ss.clean_title("Premier Polyfilm Stock Analysis #stocks #investing") == "Premier Polyfilm Stock Analysis"


def test_clean_title_collapses_double_spaces():
    assert ss.clean_title("Premier Polyfilm Limited  Stock Analysis") == "Premier Polyfilm Limited Stock Analysis"


def test_clean_title_leaves_clean_title_untouched():
    assert ss.clean_title("What's Next For ITC") == "What's Next For ITC"


# -- _extract_company_name / _ensure_company_name_in_title ---------------------

def test_extract_company_name_pulls_name_before_nse_ticker():
    prompt = "0-12 seconds - Hook\nPremier Polyfilm (NSE: PREMIER). Write an exciting opening question."
    assert ss._extract_company_name(prompt) == "Premier Polyfilm"


def test_extract_company_name_returns_none_when_no_nse_ticker_present():
    assert ss._extract_company_name("How octopuses see the world") is None


def test_ensure_company_name_in_title_leaves_title_untouched_when_already_present():
    title = "Premier Polyfilm: Is This Multibagger Overvalued?"
    assert ss._ensure_company_name_in_title(title, "Premier Polyfilm") == title


def test_ensure_company_name_in_title_prepends_when_missing():
    title = "This Multibagger Stock Nobody's Talking About"
    result = ss._ensure_company_name_in_title(title, "Premier Polyfilm")
    assert result == "Premier Polyfilm: This Multibagger Stock Nobody's Talking About"


def test_ensure_company_name_in_title_is_case_insensitive():
    title = "premier polyfilm is up big this quarter"
    assert ss._ensure_company_name_in_title(title, "Premier Polyfilm") == title


def test_ensure_company_name_in_title_noop_when_no_company_name_known():
    title = "5 Facts About Octopuses"
    assert ss._ensure_company_name_in_title(title, None) == title


def test_ensure_company_name_in_title_does_not_truncate():
    """Truncation is _truncate_title's job (applied once, after this prepend,
    so it can back off to a word boundary) -- this function just prepends."""
    title = "Why This Stock Could Be A Massive Multibagger For Long Term Investors"
    result = ss._ensure_company_name_in_title(title, "Premier Polyfilm")
    assert result == f"Premier Polyfilm: {title}"


# -- _truncate_title: backs off to a word boundary instead of cutting mid-word -

def test_truncate_title_leaves_short_title_untouched():
    assert ss._truncate_title("Short title") == "Short title"


def test_truncate_title_backs_off_to_last_word_boundary():
    """Regression test: a production title was hard-cut to exactly 70 chars
    mid-word ("...Is It a Long-Term Wea" instead of stopping before "Wea")."""
    title = "Tips Music (TIPSMUSIC): 5-Year Profit CAGR 89% - Is It a Long-Term Wealth Builder?"
    result = ss._truncate_title(title)
    assert len(result) <= 70
    assert not result.endswith("Wea")
    assert result == "Tips Music (TIPSMUSIC): 5-Year Profit CAGR 89% - Is It a Long-Term"


def test_truncate_title_hard_cuts_when_no_word_boundary_available():
    title = "x" * 100
    result = ss._truncate_title(title)
    assert len(result) == 70


def test_write_metadata_cleans_hashtags_and_double_spaces_from_title(monkeypatch):
    monkeypatch.setattr(
        ss.llm, "chat_json",
        lambda *a, **k: _mock_llm_response({
            "title": "Premier Polyfilm Limited  Stock Analysis #stocks",
            "description": "d", "tags": [],
        }),
    )
    meta = ss._write_metadata([{"narration": "n"}])
    assert meta["title"] == "Premier Polyfilm Limited Stock Analysis"


# -- append_subscribe_cta: spoken + on-screen subscribe ask on the final scene -

def test_append_subscribe_cta_appends_to_narration_of_last_scene():
    scenes = [{"narration": "Scene one.", "on_screen_text": ""}, {"narration": "Scene two.", "on_screen_text": "Final fact"}]
    ss.append_subscribe_cta(scenes)
    assert scenes[0]["narration"] == "Scene one."
    assert scenes[1]["narration"] == "Scene two. If this helped, subscribe for more videos like it."


def test_append_subscribe_cta_sets_on_screen_text_when_empty():
    scenes = [{"narration": "Only scene.", "on_screen_text": ""}]
    ss.append_subscribe_cta(scenes)
    assert scenes[0]["on_screen_text"] == "SUBSCRIBE FOR MORE"


def test_append_subscribe_cta_preserves_existing_on_screen_text():
    scenes = [{"narration": "Only scene.", "on_screen_text": "Final fact"}]
    ss.append_subscribe_cta(scenes)
    assert scenes[0]["on_screen_text"] == "Final fact"


def test_append_subscribe_cta_handles_empty_scene_list():
    scenes = []
    ss.append_subscribe_cta(scenes)
    assert scenes == []


def test_write_segmented_script_appends_subscribe_cta_to_last_scene(monkeypatch):
    monkeypatch.setattr(ss.youtube_insights, "get_combined_guidance", lambda: "")
    monkeypatch.setattr(
        ss, "_write_segment_scene",
        lambda label, content, duration, global_instructions=None: {"narration": f"{label} narration.", "visual_query": "q", "on_screen_text": ""},
    )
    monkeypatch.setattr(
        ss, "_write_metadata",
        lambda scenes, style_guidance="", company_name=None: {"title": "T", "description": "d", "tags": ["a"]},
    )
    segments = [{"label": "Hook", "content": "c1", "duration": 5}, {"label": "End screen", "content": "c2", "duration": 5}]
    result = ss.write_segmented_script("prompt", segments)
    assert result["scenes"][-1]["narration"].endswith("subscribe for more videos like it.")
    assert result["scenes"][-1]["on_screen_text"] == "SUBSCRIBE FOR MORE"
    assert result["scenes"][0]["narration"] == "Hook narration."


# -- _write_metadata: LLM-derived title/description/tags -----------------------

def test_write_metadata_uses_scene_narration_to_name_the_actual_subject(monkeypatch):
    monkeypatch.setattr(
        ss.llm, "chat_json",
        lambda *a, **k: _mock_llm_response({
            "title": "Premier Polyfilm: What Long-Term Investors Watch",
            "description": "A neutral look at Premier Polyfilm's fundamentals.",
            "tags": ["premier polyfilm", "stock analysis", "long term investing"],
        }),
    )
    scenes = [{"narration": "Premier Polyfilm has a 44.5% five-year CAGR."}]
    meta = ss._write_metadata(scenes)
    assert meta["title"] == "Premier Polyfilm: What Long-Term Investors Watch"
    assert "Premier Polyfilm" in meta["description"]
    assert "premier polyfilm" in meta["tags"]


def test_write_metadata_truncates_long_title(monkeypatch):
    monkeypatch.setattr(
        ss.llm, "chat_json",
        lambda *a, **k: _mock_llm_response({"title": "x" * 100, "description": "d", "tags": []}),
    )
    meta = ss._write_metadata([{"narration": "n"}])
    assert len(meta["title"]) == 70


def test_write_metadata_retries_on_empty_title_then_succeeds(monkeypatch):
    responses = iter([
        _mock_llm_response({"title": "", "description": "d", "tags": []}),
        _mock_llm_response({"title": "Second attempt", "description": "d", "tags": []}),
    ])
    monkeypatch.setattr(ss.llm, "chat_json", lambda *a, **k: next(responses))
    meta = ss._write_metadata([{"narration": "n"}])
    assert meta["title"] == "Second attempt"


def test_write_metadata_raises_after_three_bad_attempts(monkeypatch):
    monkeypatch.setattr(ss.llm, "chat_json", lambda *a, **k: "not json at all")
    with pytest.raises(RuntimeError, match="3 times in a row"):
        ss._write_metadata([{"narration": "n"}])


def test_write_metadata_includes_style_guidance_in_prompt_when_given(monkeypatch):
    captured = {}

    def fake_chat_json(system_prompt, user_prompt, max_tokens, timeout):
        captured["user_prompt"] = user_prompt
        return _mock_llm_response({"title": "T", "description": "d", "tags": []})

    monkeypatch.setattr(ss.llm, "chat_json", fake_chat_json)
    ss._write_metadata([{"narration": "n"}], style_guidance="- Lead with a number")
    assert "- Lead with a number" in captured["user_prompt"]


def test_write_metadata_omits_style_guidance_section_when_empty(monkeypatch):
    captured = {}

    def fake_chat_json(system_prompt, user_prompt, max_tokens, timeout):
        captured["user_prompt"] = user_prompt
        return _mock_llm_response({"title": "T", "description": "d", "tags": []})

    monkeypatch.setattr(ss.llm, "chat_json", fake_chat_json)
    ss._write_metadata([{"narration": "n"}])
    assert "currently drawing viewers" not in captured["user_prompt"]


# -- merge_tags: required tags must survive even if the LLM's list is long -----

def test_merge_tags_puts_required_tags_first_and_dedupes():
    result = ss.merge_tags(["NCERT", "CBSE"], ["cbse", "number patterns", "NCERT"])
    assert result == ["NCERT", "CBSE", "number patterns"]


def test_merge_tags_with_no_required_tags_just_dedupes_generated():
    assert ss.merge_tags([], ["a", "a", "b"]) == ["a", "b"]


def test_merge_tags_trims_to_character_budget_keeping_required_tags():
    required = ["NCERT", "CBSE"]
    generated = ["x" * 100, "y" * 100, "z" * 100, "w" * 100, "v" * 100]
    result = ss.merge_tags(required, generated, char_budget=250)
    assert result[0] == "NCERT"
    assert result[1] == "CBSE"
    assert sum(len(t) for t in result) <= 250


def test_write_segmented_script_includes_extra_tags(monkeypatch):
    monkeypatch.setattr(
        ss, "_write_segment_scene",
        lambda label, content, duration, global_instructions=None: {"narration": "n", "visual_query": "v", "on_screen_text": ""},
    )
    monkeypatch.setattr(ss.youtube_insights, "get_combined_guidance", lambda: "")
    monkeypatch.setattr(
        ss, "_write_metadata",
        lambda scenes, style_guidance="", company_name=None: {"title": "T", "description": "d", "tags": ["number theory"]},
    )
    segments = [{"label": "Hook", "content": "c", "duration": 5}]
    script = ss.write_segmented_script("prompt", segments, extra_tags=["NCERT", "CBSE"])
    assert script["tags"] == ["NCERT", "CBSE", "number theory"]


# -- normalize_indian_currency_units: Indian audiences read rupee amounts in --
# -- lakh/crore, not million/billion -- regression for a real published video --
# -- where narration said "market capitalization of ₹11.12 billion" but the ---
# -- on-screen text showed only "₹11.12" (the "billion" got silently dropped) -

def test_normalize_indian_currency_units_converts_billion_to_crore():
    assert ss.normalize_indian_currency_units("a market cap of ₹11.12 billion") == "a market cap of ₹1,112 crore"


def test_normalize_indian_currency_units_converts_large_million_to_crore():
    assert ss.normalize_indian_currency_units("earnings of ₹49 million") == "earnings of ₹4.9 crore"


def test_normalize_indian_currency_units_converts_small_million_to_lakh():
    assert ss.normalize_indian_currency_units("revenue of ₹5 million") == "revenue of ₹50 lakh"


def test_normalize_indian_currency_units_converts_large_thousand_to_lakh():
    assert ss.normalize_indian_currency_units("profit of ₹250 thousand") == "profit of ₹2.5 lakh"


def test_normalize_indian_currency_units_converts_small_thousand_to_plain_rupees():
    assert ss.normalize_indian_currency_units("fee of ₹50 thousand") == "fee of ₹50,000"


def test_normalize_indian_currency_units_boundary_ten_million_is_exactly_one_crore():
    assert ss.normalize_indian_currency_units("₹10 million") == "₹1 crore"


def test_normalize_indian_currency_units_leaves_text_without_million_billion_untouched():
    assert ss.normalize_indian_currency_units("Revenue grew to ₹3,120 crore this year.") == "Revenue grew to ₹3,120 crore this year."


def test_normalize_indian_currency_units_never_touches_dollar_amounts():
    assert ss.normalize_indian_currency_units("a market cap of $11.12 billion") == "a market cap of $11.12 billion"


def test_normalize_indian_currency_units_is_case_insensitive():
    assert ss.normalize_indian_currency_units("₹49 Million") == "₹4.9 crore"


def test_normalize_indian_currency_units_converts_trillion_to_lakh_crore():
    assert ss.normalize_indian_currency_units("₹2.1 trillion") == "₹2.1 lakh crore"


def test_normalize_indian_currency_units_converts_abbreviated_billion():
    assert ss.normalize_indian_currency_units("₹11.12B market cap") == "₹1,112 crore market cap"


def test_normalize_indian_currency_units_converts_abbreviated_million_with_space():
    assert ss.normalize_indian_currency_units("₹500 Mn revenue") == "₹50 crore revenue"


def test_normalize_indian_currency_units_converts_abbreviated_thousand():
    assert ss.normalize_indian_currency_units("₹500K fee") == "₹5 lakh fee"


def test_normalize_indian_currency_units_converts_large_billion_to_lakh_crore():
    assert ss.normalize_indian_currency_units("₹1500 billion") == "₹1.5 lakh crore"


def test_write_segment_scene_normalizes_million_billion_in_source_content(monkeypatch):
    # Regression for a real published video: narration said "11.2 million"
    # but the on-screen text showed only "Rs.11.2" -- the unit got dropped.
    # The source content itself uses the international convention; the scene's
    # narration and on-screen text must both come out in crore/lakh instead.
    monkeypatch.setattr(
        ss.llm, "chat_json",
        lambda *a, **k: _mock_llm_response({
            "narration": "The company posted a market cap of ₹11.12 billion.",
            "visual_query": "stock chart",
            "on_screen_text": "₹11.12 billion",
        }),
    )
    scene = ss._write_segment_scene("Valuation", "Market cap stands at ₹11.12 billion.", 10)
    assert "billion" not in scene["narration"]
    assert "1,112 crore" in scene["narration"]
    assert "billion" not in scene["on_screen_text"]
    assert "1,112 crore" in scene["on_screen_text"]


def test_metadata_prompt_steers_away_from_verdict_style_titles():
    # The channel's existing narration-level neutrality rule (no buy/sell
    # recommendation) doesn't automatically cover title phrasing -- a model
    # could still write a verdict-style title ("Cheap or Overvalued?") even
    # with neutral narration underneath. Lock in the explicit title-level rule.
    assert "Apply or Avoid" in ss.SYSTEM_PROMPT_METADATA
    assert "data-driven" in ss.SYSTEM_PROMPT_METADATA


def test_segment_prompt_steers_visual_query_away_from_generic_finance_imagery():
    # Pexels search terms like "stock market chart" return the same top
    # results every finance channel querying the same term also gets -- push
    # the model toward sector-specific terms instead, so footage actually
    # differs company to company.
    assert "stock market chart" in ss.SYSTEM_PROMPT_SEGMENT
    assert "sector" in ss.SYSTEM_PROMPT_SEGMENT


def test_segment_prompt_prefers_diagram_over_stock_footage_for_numeric_scenes():
    # A diagram built from this company's own numbers is unique to the video;
    # generic stock footage of a chart/building looks the same everywhere --
    # the model should prefer the diagram even when filmable stock exists,
    # not reserve it only for when stock footage has no equivalent at all.
    assert "prefer setting this" in ss.SYSTEM_PROMPT_SEGMENT
    assert "ONLY set this" not in ss.SYSTEM_PROMPT_SEGMENT
