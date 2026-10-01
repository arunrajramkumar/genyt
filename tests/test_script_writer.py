import json

import pytest

from pipeline import script_writer

VALID_SCRIPT = {
    "title": "History of Rome",
    "description": "A journey through ancient Rome.",
    "tags": ["history", "rome"],
    "scenes": [{"narration": "Rome was founded in 753 BC.", "visual_query": "ancient rome ruins"}],
}


def test_write_script_succeeds_on_valid_response(monkeypatch):
    monkeypatch.setattr(script_writer.llm, "chat_json", lambda *a, **k: json.dumps(VALID_SCRIPT))
    result = script_writer.write_script("history of Rome")
    assert result == VALID_SCRIPT


def test_write_script_raises_after_repeated_invalid_json(monkeypatch):
    monkeypatch.setattr(script_writer.llm, "chat_json", lambda *a, **k: "garbage, not json")
    with pytest.raises(RuntimeError, match="3 times in a row"):
        script_writer.write_script("history of Rome")


def test_write_script_rejects_zero_scenes(monkeypatch):
    bad_script = dict(VALID_SCRIPT, scenes=[])
    monkeypatch.setattr(script_writer.llm, "chat_json", lambda *a, **k: json.dumps(bad_script))
    with pytest.raises(RuntimeError):
        script_writer.write_script("history of Rome")
