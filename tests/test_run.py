from pathlib import Path

import run
from pipeline import config


def test_load_themes_returns_empty_list_when_file_missing(tmp_path):
    assert run.load_themes(tmp_path / "missing.yaml") == []


def test_save_then_load_themes_round_trips(tmp_path):
    themes_file = tmp_path / "themes.yaml"

    themes = [
        {"theme": "History of Rome", "done": False},
        {"theme": "Octopus facts", "done": True, "duration_sec": 45},
    ]
    run.save_themes(themes_file, themes)
    assert themes_file.exists()
    loaded = run.load_themes(themes_file)
    assert loaded == themes


def test_save_themes_preserves_non_ascii_text(tmp_path):
    themes_file = tmp_path / "themes.yaml"
    themes = [{"theme": "गोपाल की कहानी", "done": False}]
    run.save_themes(themes_file, themes)
    loaded = run.load_themes(themes_file)
    assert loaded[0]["theme"] == "गोपाल की कहानी"


def test_produce_video_writes_script_then_produces(monkeypatch):
    calls = {}

    def fake_produce(script, orientation, on_progress):
        calls["orientation"] = orientation
        return {"video": Path("v.mp4"), "srt": Path("v.srt"), "metadata": Path("v.meta"), "slug": "t"}

    monkeypatch.setattr(run.prompt_script, "write_script_from_prompt", lambda prompt, duration_sec: {"title": "T", "scenes": []})
    monkeypatch.setattr(run.producer, "produce_from_script", fake_produce)

    result = run.produce_video("a theme", shorts=True)
    assert calls["orientation"] == config.SHORTS
    assert result == Path("v.mp4")


def test_produce_video_folds_style_into_the_prompt(monkeypatch):
    captured = {}
    monkeypatch.setattr(
        run.prompt_script, "write_script_from_prompt",
        lambda prompt, duration_sec: captured.setdefault("prompt", prompt) or {"title": "T", "scenes": []},
    )
    monkeypatch.setattr(run.producer, "produce_from_script", lambda script, orientation, on_progress: {
        "video": Path("v.mp4"), "srt": Path("v.srt"), "metadata": Path("v.meta"), "slug": "t",
    })

    run.produce_video("a theme", style="curious and upbeat")
    assert "a theme" in captured["prompt"]
    assert "curious and upbeat" in captured["prompt"]


def test_main_processes_the_given_themes_file_not_the_default(monkeypatch, tmp_path):
    """--themes-file lets a second channel's queue run through the same script
    without touching themes.yaml, so two channels can share one codebase."""
    themes_file = tmp_path / "themes_other_channel.yaml"
    run.save_themes(themes_file, [{"theme": "Other channel theme", "done": False}])

    produced = []
    monkeypatch.setattr(run, "produce_video", lambda theme, style, duration, shorts: produced.append(theme))
    monkeypatch.setattr("sys.argv", ["run.py", "--themes-file", str(themes_file)])

    run.main()

    assert produced == ["Other channel theme"]
    assert run.load_themes(themes_file)[0]["done"] is True
