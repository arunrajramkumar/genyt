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

    def fake_write_segmented(prompt, segments, duration_sec, extra_tags=None):
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
    monkeypatch.setattr(prompt_script.research, "fetch_grounding_facts", lambda theme: "- Revenue grew 12% (source: example.com)")
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
    monkeypatch.setattr(prompt_script.research, "fetch_grounding_facts", lambda theme: "")
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
    monkeypatch.setattr(prompt_script.research, "fetch_grounding_facts", lambda theme: called.append(theme))
    monkeypatch.setattr(prompt_script.llm, "chat_json", lambda *a, **k: _json_response(VALID_SCRIPT))
    prompt_script._write_freeform_script("octopuses")
    assert called == []


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
