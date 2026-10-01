from pathlib import Path

import run
from pipeline import config


def test_load_themes_returns_empty_list_when_file_missing(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "THEMES_FILE", tmp_path / "missing.yaml")
    assert run.load_themes() == []


def test_save_then_load_themes_round_trips(monkeypatch, tmp_path):
    themes_file = tmp_path / "themes.yaml"
    monkeypatch.setattr(config, "THEMES_FILE", themes_file)

    themes = [
        {"theme": "History of Rome", "done": False},
        {"theme": "Octopus facts", "done": True, "duration_sec": 45},
    ]
    run.save_themes(themes)
    assert themes_file.exists()
    loaded = run.load_themes()
    assert loaded == themes


def test_save_themes_preserves_non_ascii_text(monkeypatch, tmp_path):
    themes_file = tmp_path / "themes.yaml"
    monkeypatch.setattr(config, "THEMES_FILE", themes_file)
    themes = [{"theme": "गोपाल की कहानी", "done": False}]
    run.save_themes(themes)
    loaded = run.load_themes()
    assert loaded[0]["theme"] == "गोपाल की कहानी"


def test_produce_video_writes_script_then_produces(monkeypatch):
    calls = {}

    def fake_produce(script, orientation, on_progress):
        calls["orientation"] = orientation
        return {"video": Path("v.mp4"), "srt": Path("v.srt"), "metadata": Path("v.meta"), "slug": "t"}

    monkeypatch.setattr(run.script_writer, "write_script", lambda theme, style, duration: {"title": "T", "scenes": []})
    monkeypatch.setattr(run.producer, "produce_from_script", fake_produce)

    result = run.produce_video("a theme", shorts=True)
    assert calls["orientation"] == config.SHORTS
    assert result == Path("v.mp4")
