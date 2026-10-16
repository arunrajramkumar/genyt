import json

import pytest

from pipeline import prompt_script


def _json_response(payload: dict) -> str:
    return json.dumps(payload)


VALID_SCRIPT = {
    "title": "5 Facts About Octopuses",
    "description": "A quick dive into octopus facts.",
    "tags": ["octopus", "ocean", "facts"],
    "scenes": [
        {"narration": "Octopuses have three hearts.", "visual_query": "octopus swimming", "on_screen_text": ""},
    ],
}


def test_extract_json_parses_plain_json():
    assert prompt_script._extract_json(json.dumps(VALID_SCRIPT)) == VALID_SCRIPT


def test_extract_json_strips_markdown_fences():
    fenced = f"```json\n{json.dumps(VALID_SCRIPT)}\n```"
    assert prompt_script._extract_json(fenced) == VALID_SCRIPT


def test_extract_json_extracts_object_from_surrounding_prose():
    wrapped = f"Sure, here's the script:\n{json.dumps(VALID_SCRIPT)}\nHope that helps!"
    assert prompt_script._extract_json(wrapped) == VALID_SCRIPT


def test_write_script_from_prompt_dispatches_to_segmented_for_structured_prompt(monkeypatch):
    called = {}

    def fake_write_segmented(prompt, segments, duration_sec, extra_tags=None, global_instructions=None):
        called["ran"] = True
        return VALID_SCRIPT

    monkeypatch.setattr(prompt_script.segmented_script, "write_segmented_script", fake_write_segmented)
    structured_prompt = (
        "0-5 seconds - Hook\nFact one.\n\n"
        "5-10 seconds - Body\nFact two.\n\n"
        "10-15 seconds - End\nFact three.\n"
    )
    result = prompt_script.write_script_from_prompt(structured_prompt)
    assert called.get("ran") is True
    assert result == VALID_SCRIPT


def test_write_script_from_prompt_falls_back_to_freeform_for_plain_prompt(monkeypatch):
    monkeypatch.setattr(prompt_script, "_write_freeform_script", lambda prompt, duration_sec, extra_tags=None: VALID_SCRIPT)
    result = prompt_script.write_script_from_prompt("Tell me about octopuses")
    assert result == VALID_SCRIPT


def test_write_script_from_prompt_threads_style_into_segmented_path(monkeypatch):
    captured = {}

    def fake_write_segmented(prompt, segments, duration_sec, extra_tags=None, global_instructions=None):
        captured["global_instructions"] = global_instructions
        return VALID_SCRIPT

    monkeypatch.setattr(prompt_script.segmented_script, "write_segmented_script", fake_write_segmented)
    structured_prompt = (
        "0-5 seconds - Hook\nFact one.\n\n"
        "5-10 seconds - Body\nFact two.\n\n"
        "10-15 seconds - End\nFact three.\n"
    )
    prompt_script.write_script_from_prompt(structured_prompt, style="curious and upbeat")
    assert captured["global_instructions"] == "curious and upbeat"


def test_write_script_from_prompt_folds_style_into_freeform_prompt(monkeypatch):
    captured = {}

    def fake_freeform(prompt, duration_sec, extra_tags=None):
        captured["prompt"] = prompt
        return VALID_SCRIPT

    monkeypatch.setattr(prompt_script, "_write_freeform_script", fake_freeform)
    prompt_script.write_script_from_prompt("Tell me about octopuses", style="curious and upbeat")
    assert "Tell me about octopuses" in captured["prompt"]
    assert "curious and upbeat" in captured["prompt"]


def test_write_freeform_script_succeeds_on_valid_response(monkeypatch):
    monkeypatch.setattr(prompt_script.llm, "chat_json", lambda *a, **k: _json_response(VALID_SCRIPT))
    result = prompt_script._write_freeform_script("octopuses")
    assert result["title"] == VALID_SCRIPT["title"]
    assert result["description"] == VALID_SCRIPT["description"]
    assert result["tags"] == VALID_SCRIPT["tags"]
    assert result["scenes"][0]["narration"].startswith(VALID_SCRIPT["scenes"][0]["narration"])


def test_write_freeform_script_retries_then_succeeds(monkeypatch):
    responses = iter(["not json", _json_response(VALID_SCRIPT)])
    monkeypatch.setattr(prompt_script.llm, "chat_json", lambda *a, **k: next(responses))
    result = prompt_script._write_freeform_script("octopuses")
    assert result["title"] == VALID_SCRIPT["title"]
    assert result["scenes"][0]["narration"].startswith(VALID_SCRIPT["scenes"][0]["narration"])


@pytest.mark.parametrize("missing_key", ["title", "description", "tags", "scenes"])
def test_write_freeform_script_rejects_missing_required_fields(monkeypatch, missing_key):
    bad_script = {k: v for k, v in VALID_SCRIPT.items() if k != missing_key}
    monkeypatch.setattr(prompt_script.llm, "chat_json", lambda *a, **k: _json_response(bad_script))
    with pytest.raises(RuntimeError, match="3 times in a row"):
        prompt_script._write_freeform_script("octopuses")


def test_write_freeform_script_rejects_zero_scenes(monkeypatch):
    bad_script = dict(VALID_SCRIPT, scenes=[])
    monkeypatch.setattr(prompt_script.llm, "chat_json", lambda *a, **k: _json_response(bad_script))
    with pytest.raises(RuntimeError, match="3 times in a row"):
        prompt_script._write_freeform_script("octopuses")


def test_write_freeform_script_includes_channel_style_guidance_when_available(monkeypatch):
    monkeypatch.setattr(prompt_script.youtube_insights, "get_combined_guidance", lambda: "- Lead with a number")
    captured = {}

    def fake_chat_json(system_prompt, user_prompt, max_tokens, timeout):
        captured["user_prompt"] = user_prompt
        return _json_response(VALID_SCRIPT)

    monkeypatch.setattr(prompt_script.llm, "chat_json", fake_chat_json)
    prompt_script._write_freeform_script("octopuses")
    assert "- Lead with a number" in captured["user_prompt"]


def test_write_freeform_script_omits_style_guidance_section_when_unavailable(monkeypatch):
    monkeypatch.setattr(prompt_script.youtube_insights, "get_combined_guidance", lambda: "")
    captured = {}

    def fake_chat_json(system_prompt, user_prompt, max_tokens, timeout):
        captured["user_prompt"] = user_prompt
        return _json_response(VALID_SCRIPT)

    monkeypatch.setattr(prompt_script.llm, "chat_json", fake_chat_json)
    prompt_script._write_freeform_script("octopuses")
    assert "currently drawing viewers" not in captured["user_prompt"]


def test_write_freeform_script_cleans_hashtags_and_double_spaces_from_title(monkeypatch):
    dirty_script = dict(VALID_SCRIPT, title="5 Facts  About Octopuses #ocean #facts")
    monkeypatch.setattr(prompt_script.llm, "chat_json", lambda *a, **k: _json_response(dirty_script))
    result = prompt_script._write_freeform_script("octopuses")
    assert result["title"] == "5 Facts About Octopuses"


def test_write_freeform_script_falls_back_to_default_tag_when_empty(monkeypatch):
    untagged_script = dict(VALID_SCRIPT, tags=[])
    monkeypatch.setattr(prompt_script.llm, "chat_json", lambda *a, **k: _json_response(untagged_script))
    result = prompt_script._write_freeform_script("octopuses")
    assert result["tags"] == ["short video"]


def test_write_freeform_script_merges_extra_tags_ahead_of_generated_ones(monkeypatch):
    tagged_script = dict(VALID_SCRIPT, tags=["octopuses", "ocean facts"])
    monkeypatch.setattr(prompt_script.llm, "chat_json", lambda *a, **k: _json_response(tagged_script))
    result = prompt_script._write_freeform_script("octopuses", extra_tags=["NCERT", "CBSE"])
    assert result["tags"] == ["NCERT", "CBSE", "octopuses", "ocean facts"]


def test_write_freeform_script_injects_grounding_facts_for_finance_themes(monkeypatch):
    monkeypatch.setattr(prompt_script.research, "needs_grounding", lambda theme: True)
    monkeypatch.setattr(prompt_script.research, "fetch_grounding_facts", lambda theme, symbol=None: "- Revenue grew 12% (source: example.com)")
    captured = {}

    def fake_chat_json(system_prompt, user_prompt, max_tokens, timeout):
        captured["user_prompt"] = user_prompt
        return _json_response(VALID_SCRIPT)

    monkeypatch.setattr(prompt_script.llm, "chat_json", fake_chat_json)
    prompt_script._write_freeform_script("Tata Motors stock analysis")
    assert "Revenue grew 12%" in captured["user_prompt"]
    assert "use ONLY" in captured["user_prompt"]


def test_write_freeform_script_warns_against_fabrication_when_grounding_unavailable(monkeypatch):
    monkeypatch.setattr(prompt_script.research, "needs_grounding", lambda theme: True)
    monkeypatch.setattr(prompt_script.research, "fetch_grounding_facts", lambda theme, symbol=None: "")
    captured = {}

    def fake_chat_json(system_prompt, user_prompt, max_tokens, timeout):
        captured["user_prompt"] = user_prompt
        return _json_response(VALID_SCRIPT)

    monkeypatch.setattr(prompt_script.llm, "chat_json", fake_chat_json)
    prompt_script._write_freeform_script("Tata Motors stock analysis")
    assert "Do NOT" in captured["user_prompt"]
    assert "invent specific numbers" in captured["user_prompt"]


def test_write_freeform_script_skips_grounding_for_non_finance_themes(monkeypatch):
    called = []
    monkeypatch.setattr(prompt_script.research, "needs_grounding", lambda theme: False)
    monkeypatch.setattr(prompt_script.research, "fetch_grounding_facts", lambda theme, symbol=None: called.append(theme))
    monkeypatch.setattr(prompt_script.llm, "chat_json", lambda *a, **k: _json_response(VALID_SCRIPT))
    prompt_script._write_freeform_script("octopuses")
    assert called == []


def test_write_freeform_script_scrubs_ticker_from_narration(monkeypatch):
    """Same TTS-mispronunciation concern as the segmented path (see
    test_segmented_script.test_write_segmented_script_scrubs_ticker_from_narration)
    -- the freeform path resolves its own symbol via
    research.resolve_symbol_for_theme and must scrub it out of narration too,
    keyed off the company name the model put in the first scene's
    on_screen_text (see _extract_freeform_company_name)."""
    monkeypatch.setattr(prompt_script.research, "needs_grounding", lambda theme: True)
    monkeypatch.setattr(prompt_script.research, "fetch_grounding_facts", lambda theme, symbol=None: "- P/E is 59.8 (source: example.com)")
    monkeypatch.setattr(prompt_script.research, "resolve_symbol_for_theme", lambda theme: "PIDILITIND")
    ticker_script = dict(VALID_SCRIPT, scenes=[
        {"narration": "PIDILITIND trades at a P/E of 59.8.", "visual_query": "v", "on_screen_text": "Pidilite Industries Ltd"},
    ])
    monkeypatch.setattr(prompt_script.llm, "chat_json", lambda *a, **k: _json_response(ticker_script))
    result = prompt_script._write_freeform_script("Pidilite Industries Ltd stock analysis")
    assert "PIDILITIND" not in result["scenes"][0]["narration"]
    assert "Pidilite Industries Ltd" in result["scenes"][0]["narration"]


def test_write_freeform_script_appends_subscribe_cta_to_last_scene(monkeypatch):
    two_scene_script = dict(VALID_SCRIPT, scenes=[
        {"narration": "Fact one.", "visual_query": "q1", "on_screen_text": ""},
        {"narration": "Fact two.", "visual_query": "q2", "on_screen_text": ""},
    ])
    monkeypatch.setattr(prompt_script.llm, "chat_json", lambda *a, **k: _json_response(two_scene_script))
    result = prompt_script._write_freeform_script("octopuses")
    assert result["scenes"][-1]["narration"] == "Fact two. If this helped, subscribe for more videos like it."
    assert result["scenes"][-1]["on_screen_text"] == "SUBSCRIBE FOR MORE"
    assert result["scenes"][0]["narration"] == "Fact one."


# -- on_screen_text grounding: freeform path must not show a number that was --
# -- spoken in a DIFFERENT scene (or never spoken at all) ---------------------

def test_ground_scene_on_screen_text_clears_text_not_grounded_in_its_own_narration():
    # Regression for a real published video where on-screen metrics had
    # "completely no correlation" with the narration -- the on_screen_text
    # here quotes a number that was never said in THIS scene's narration.
    scene = {"narration": "The company's margins improved this quarter.", "on_screen_text": "Revenue: Rs 412 Cr"}
    prompt_script._ground_scene_on_screen_text(scene)
    assert scene["on_screen_text"] == ""


def test_ground_scene_on_screen_text_keeps_text_grounded_in_its_own_narration():
    scene = {"narration": "Revenue grew to Rs 412 Cr this quarter.", "on_screen_text": "Revenue: Rs 412 Cr"}
    prompt_script._ground_scene_on_screen_text(scene)
    assert scene["on_screen_text"] == "Revenue: Rs 412 Cr"


def test_ground_scene_on_screen_text_fills_in_a_fact_when_on_screen_text_is_empty():
    scene = {"narration": "ROE stands at 30.2% this year.", "on_screen_text": ""}
    prompt_script._ground_scene_on_screen_text(scene)
    assert "30.2" in scene["on_screen_text"]


def test_ground_scene_on_screen_text_shows_all_facts_when_narration_has_several():
    scene = {
        "narration": "P/E ratio stands at 20.09, while ROE is 30.2%.",
        "on_screen_text": "P/E: 20.09",
    }
    prompt_script._ground_scene_on_screen_text(scene)
    assert "|" in scene["on_screen_text"]


def test_ground_scene_on_screen_text_leaves_qualitative_text_alone():
    # No 2+ digit numbers in the on_screen_text -- always considered grounded,
    # same as the segmented path's _is_grounded, so company names/labels with
    # no digits are never wiped out.
    scene = {"narration": "A quick look at the company's history.", "on_screen_text": "Arrow Greentech"}
    prompt_script._ground_scene_on_screen_text(scene)
    assert scene["on_screen_text"] == "Arrow Greentech"


def test_ground_scene_on_screen_text_preserves_first_scene_subject_name_despite_multiple_facts():
    # Regression for the real reported bug's own first scene: SYSTEM_PROMPT
    # requires scene 1's on_screen_text to be the subject's name even though
    # the hook narration also states 2+ numeric facts -- those facts must not
    # clobber the name the way they would for any other scene.
    scene = {
        "narration": "Mazagon Dock trades at a current price of ₹2,335.50 and a market cap of ₹81,765 cr.",
        "on_screen_text": "Mazagon Dock",
    }
    prompt_script._ground_scene_on_screen_text(scene, is_first_scene=True)
    assert scene["on_screen_text"] == "Mazagon Dock"


def test_ground_scene_on_screen_text_still_clears_fabricated_first_scene_text():
    scene = {"narration": "Mazagon Dock is a major Indian shipbuilder.", "on_screen_text": "Revenue: ₹412 Cr"}
    prompt_script._ground_scene_on_screen_text(scene, is_first_scene=True)
    assert scene["on_screen_text"] == ""


def test_ground_scene_on_screen_text_fills_empty_first_scene_text_with_a_fact():
    scene = {"narration": "Revenue grew to ₹412 Cr this quarter.", "on_screen_text": ""}
    prompt_script._ground_scene_on_screen_text(scene, is_first_scene=True)
    assert "412" in scene["on_screen_text"]


def test_ground_scene_on_screen_text_normalizes_million_billion_before_grounding_check():
    # Mirrors _write_freeform_script's real call order: narration is already
    # normalized by the time this runs, so the on-screen text's normalized
    # figure must still line up with it for the grounding check to pass.
    scene = {
        "narration": "The company posted a market cap of ₹1,112 crore.",
        "on_screen_text": "₹11.12 billion",
    }
    prompt_script._ground_scene_on_screen_text(scene)
    assert scene["on_screen_text"] == "₹1,112 crore"


def test_write_freeform_script_clears_on_screen_text_not_grounded_in_its_own_scene(monkeypatch):
    # Three scenes so the subscribe-CTA rewrite of the LAST scene's
    # on_screen_text doesn't mask what we're actually testing here.
    mismatched_script = dict(VALID_SCRIPT, scenes=[
        {"narration": "Octopuses have three hearts.", "visual_query": "octopus", "on_screen_text": "Revenue: ₹412 Cr"},
        {"narration": "Revenue grew to ₹412 Cr this quarter.", "visual_query": "chart", "on_screen_text": ""},
        {"narration": "That's a strong result overall.", "visual_query": "office", "on_screen_text": ""},
    ])
    monkeypatch.setattr(prompt_script.llm, "chat_json", lambda *a, **k: _json_response(mismatched_script))
    result = prompt_script._write_freeform_script("octopuses")
    # The figure was spoken in scene 2, not scene 1 -- scene 1 must not show it.
    assert result["scenes"][0]["on_screen_text"] == ""
    assert "412" in result["scenes"][1]["on_screen_text"]


def test_write_freeform_script_preserves_first_scene_subject_name_through_full_pipeline(monkeypatch):
    multi_fact_hook_script = dict(VALID_SCRIPT, scenes=[
        {
            "narration": "Mazagon Dock trades at a current price of ₹2,335.50 and a market cap of ₹81,765 cr.",
            "visual_query": "shipyard",
            "on_screen_text": "Mazagon Dock",
        },
        {"narration": "That's a strong result overall.", "visual_query": "office", "on_screen_text": ""},
    ])
    monkeypatch.setattr(prompt_script.llm, "chat_json", lambda *a, **k: _json_response(multi_fact_hook_script))
    result = prompt_script._write_freeform_script("Mazagon Dock stock analysis")
    assert result["scenes"][0]["on_screen_text"] == "Mazagon Dock"


# -- company_name: freeform path must also surface it (for ai_thumbnail) ------

def test_write_freeform_script_sets_company_name_from_first_scene_for_finance_theme(monkeypatch):
    nalco_script = dict(VALID_SCRIPT, scenes=[
        {
            "narration": "National Aluminium is the focus today.",
            "visual_query": "aluminium plant",
            "on_screen_text": "National Aluminium Company",
        },
        {"narration": "Its price-to-earnings looks cheap.", "visual_query": "chart", "on_screen_text": ""},
    ])
    monkeypatch.setattr(prompt_script.llm, "chat_json", lambda *a, **k: _json_response(nalco_script))
    result = prompt_script._write_freeform_script("National Aluminium Company - Stock Analysis For Long Term Growth")
    assert result["company_name"] == "National Aluminium Company"


def test_write_freeform_script_omits_company_name_for_non_finance_theme(monkeypatch):
    monkeypatch.setattr(prompt_script.llm, "chat_json", lambda *a, **k: _json_response(VALID_SCRIPT))
    result = prompt_script._write_freeform_script("Tell me about octopuses")
    assert result["company_name"] is None


def test_write_freeform_script_omits_company_name_when_first_scene_has_no_text(monkeypatch):
    untitled_hook_script = dict(VALID_SCRIPT, scenes=[
        {"narration": "A quick look at the numbers.", "visual_query": "chart", "on_screen_text": ""},
    ])
    monkeypatch.setattr(prompt_script.llm, "chat_json", lambda *a, **k: _json_response(untitled_hook_script))
    result = prompt_script._write_freeform_script("Tata Motors stock analysis")
    assert result["company_name"] is None


# -- unit normalization: narration/description/on_screen_text must read in --
# -- lakh/crore, not million/billion, for Indian finance themes --------------

def test_write_freeform_script_normalizes_million_billion_in_narration_and_description(monkeypatch):
    # Regression for a real published video: narration said "market
    # capitalization of ₹11.12 billion" but the on-screen text showed only
    # "₹11.12" -- the unit word got silently dropped during fact-extraction.
    billion_script = dict(
        VALID_SCRIPT,
        description="Trailing twelve-month earnings of ₹49 million.",
        scenes=[
            {
                "narration": "The company has a market capitalization of ₹11.12 billion.",
                "visual_query": "stock chart",
                "on_screen_text": "₹11.12 billion",
            },
            {"narration": "That's a strong result overall.", "visual_query": "office", "on_screen_text": ""},
        ],
    )
    monkeypatch.setattr(prompt_script.llm, "chat_json", lambda *a, **k: _json_response(billion_script))
    result = prompt_script._write_freeform_script("Premier Polyfilm stock analysis")
    assert "billion" not in result["description"]
    assert "₹4.9 crore" in result["description"]
    assert "billion" not in result["scenes"][0]["narration"]
    assert "₹1,112 crore" in result["scenes"][0]["narration"]
    assert "billion" not in result["scenes"][0]["on_screen_text"]
    assert "1,112 crore" in result["scenes"][0]["on_screen_text"]


def test_write_freeform_script_does_not_mistake_subscribe_cta_for_company_name(monkeypatch):
    # Single-scene script: append_subscribe_cta overwrites the only (also
    # first) scene's on_screen_text with the CTA -- that must never leak out
    # as a "company name".
    one_scene_script = dict(VALID_SCRIPT, scenes=[
        {"narration": "Tata Motors at a glance.", "visual_query": "cars", "on_screen_text": "Tata Motors"},
    ])
    monkeypatch.setattr(prompt_script.llm, "chat_json", lambda *a, **k: _json_response(one_scene_script))
    result = prompt_script._write_freeform_script("Tata Motors stock analysis")
    assert result["company_name"] != "SUBSCRIBE FOR MORE"


def test_system_prompt_steers_visual_query_away_from_generic_finance_imagery():
    assert "stock market chart" in prompt_script.SYSTEM_PROMPT
    assert "sector" in prompt_script.SYSTEM_PROMPT


def test_system_prompt_prefers_diagram_over_stock_footage_for_numeric_scenes():
    assert "prefer setting this" in prompt_script.SYSTEM_PROMPT
    assert "ONLY set this" not in prompt_script.SYSTEM_PROMPT
