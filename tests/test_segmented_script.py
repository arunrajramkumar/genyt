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

def test_extract_verbatim_pulls_quoted_sentence():
    content = 'Some lead-in text. "This steel company just signed a huge deal." More text.'
    assert ss._extract_verbatim(content) == "This steel company just signed a huge deal."


def test_extract_verbatim_returns_empty_when_no_quote():
    assert ss._extract_verbatim("No quotes in this content at all.") == ""


@pytest.mark.parametrize("content,expected", [
    ("Revenue grew to ₹3,120 crore this year.", "₹3,120 crore"),
    ("ROE 18.5 reported this quarter, no other figures.", "ROE"),
    ("The company is debt-free with export exposure.", "Debt-Free Balance Sheet"),
    ("Just a plain sentence with nothing extractable.", ""),
])
def test_extract_fact_finds_known_patterns(content, expected):
    fact = ss._extract_fact(content)
    if expected:
        assert expected.split()[0].lower() in fact.lower() or expected in fact
    else:
        assert fact == ""


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


def test_fix_double_escaped_unicode_decodes_literal_escapes():
    assert ss._fix_double_escaped_unicode("\\u20b9412 crore") == "₹412 crore"


def test_fix_double_escaped_unicode_leaves_normal_text_untouched():
    assert ss._fix_double_escaped_unicode("plain text") == "plain text"


@pytest.mark.parametrize("label,expected_fragment", [
    ("Hook", "factory"),
    ("Company overview", "steel factory"),
    ("Financials", "stock market chart growth"),
    ("Risk factors", "warning"),
    ("Something unrelated", "business professional presentation"),
])
def test_fallback_visual_matches_label_keywords(label, expected_fragment):
    assert expected_fragment in ss._fallback_visual(label)


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
    scene = ss._write_segment_scene("Financials", "Revenue grew 18% YoY to ₹3,120 crore.", 15)
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


def test_write_segmented_script_assembles_title_description_tags(monkeypatch):
    monkeypatch.setattr(
        ss, "_write_segment_scene",
        lambda label, content, duration: {"narration": f"N:{label}", "visual_query": "v", "on_screen_text": ""},
    )
    monkeypatch.setattr(ss.youtube_insights, "get_combined_guidance", lambda: "")
    monkeypatch.setattr(
        ss, "_write_metadata",
        lambda scenes, style_guidance="": {"title": "Premier Polyfilm: What Long-Term Investors Watch",
                         "description": "desc", "tags": ["premier polyfilm", "stock analysis"]},
    )
    segments = [{"label": "Hook", "content": "c", "duration": 5}]
    script = ss.write_segmented_script("some prompt", segments)
    assert script["title"] == "Premier Polyfilm: What Long-Term Investors Watch"
    assert script["description"] == "desc"
    assert script["tags"] == ["premier polyfilm", "stock analysis"]
    assert script["scenes"][0]["narration"] == "N:Hook"


def test_write_segmented_script_passes_channel_style_guidance_into_metadata(monkeypatch):
    monkeypatch.setattr(
        ss, "_write_segment_scene",
        lambda label, content, duration: {"narration": "n", "visual_query": "v", "on_screen_text": ""},
    )
    monkeypatch.setattr(ss.youtube_insights, "get_combined_guidance", lambda: "- Lead with a number")
    captured = {}

    def fake_write_metadata(scenes, style_guidance=""):
        captured["style_guidance"] = style_guidance
        return {"title": "T", "description": "d", "tags": []}

    monkeypatch.setattr(ss, "_write_metadata", fake_write_metadata)
    segments = [{"label": "Hook", "content": "c", "duration": 5}]
    ss.write_segmented_script("some prompt", segments)
    assert captured["style_guidance"] == "- Lead with a number"


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
