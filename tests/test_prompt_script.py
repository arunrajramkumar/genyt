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

    def fake_write_segmented(prompt, segments, duration_sec):
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
    monkeypatch.setattr(prompt_script, "_write_freeform_script", lambda prompt, duration_sec: VALID_SCRIPT)
    result = prompt_script.write_script_from_prompt("Tell me about octopuses")
    assert result == VALID_SCRIPT


def test_write_freeform_script_succeeds_on_valid_response(monkeypatch):
    monkeypatch.setattr(prompt_script.llm, "chat_json", lambda *a, **k: _json_response(VALID_SCRIPT))
    result = prompt_script._write_freeform_script("octopuses")
    assert result == VALID_SCRIPT


def test_write_freeform_script_retries_then_succeeds(monkeypatch):
    responses = iter(["not json", _json_response(VALID_SCRIPT)])
    monkeypatch.setattr(prompt_script.llm, "chat_json", lambda *a, **k: next(responses))
    result = prompt_script._write_freeform_script("octopuses")
    assert result == VALID_SCRIPT


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
