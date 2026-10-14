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

    def fake_produce(script, orientation, on_progress, thumbnail_badge=None):
        calls["orientation"] = orientation
        return {"video": Path("v.mp4"), "srt": Path("v.srt"), "metadata": Path("v.meta"), "slug": "t"}

    monkeypatch.setattr(run.prompt_script, "write_script_from_prompt", lambda prompt, duration_sec, extra_tags=None, style=None: {"title": "T", "scenes": []})
    monkeypatch.setattr(run.producer, "produce_from_script", fake_produce)

    result = run.produce_video("a theme", shorts=True)
    assert calls["orientation"] == config.SHORTS
    assert result == Path("v.mp4")


def test_produce_video_passes_style_through_to_write_script_from_prompt(monkeypatch):
    captured = {}
    monkeypatch.setattr(
        run.prompt_script, "write_script_from_prompt",
        lambda prompt, duration_sec, extra_tags=None, style=None: captured.setdefault("style", style) or {"title": "T", "scenes": []},
    )
    monkeypatch.setattr(run.producer, "produce_from_script", lambda script, orientation, on_progress, thumbnail_badge=None: {
        "video": Path("v.mp4"), "srt": Path("v.srt"), "metadata": Path("v.meta"), "slug": "t",
    })

    run.produce_video("a theme", style="curious and upbeat")
    assert captured["style"] == "curious and upbeat"


def test_produce_video_passes_thumbnail_badge_to_producer(monkeypatch):
    captured = {}
    monkeypatch.setattr(run.prompt_script, "write_script_from_prompt", lambda prompt, duration_sec, extra_tags=None, style=None: {"title": "T", "scenes": []})

    def fake_produce(script, orientation, on_progress, thumbnail_badge=None):
        captured["thumbnail_badge"] = thumbnail_badge
        return {"video": Path("v.mp4"), "srt": Path("v.srt"), "metadata": Path("v.meta"), "slug": "t"}

    monkeypatch.setattr(run.producer, "produce_from_script", fake_produce)

    run.produce_video("a theme", thumbnail_badge="CBSE | Grade 6 Maths")
    assert captured["thumbnail_badge"] == "CBSE | Grade 6 Maths"


def test_main_processes_the_given_themes_file_not_the_default(monkeypatch, tmp_path):
    """--themes-file lets a second channel's queue run through the same script
    without touching themes.yaml, so two channels can share one codebase."""
    themes_file = tmp_path / "themes_other_channel.yaml"
    run.save_themes(themes_file, [{"theme": "Other channel theme", "done": False}])

    produced = []
    monkeypatch.setattr(run, "produce_video", lambda theme, style, duration, shorts, extra_tags=None, thumbnail_badge=None: produced.append(theme))
    monkeypatch.setattr("sys.argv", ["run.py", "--themes-file", str(themes_file)])

    run.main()

    assert produced == ["Other channel theme"]
    assert run.load_themes(themes_file)[0]["done"] is True


def test_main_passes_each_themes_extra_tags_through(monkeypatch, tmp_path):
    themes_file = tmp_path / "themes_education.yaml"
    run.save_themes(themes_file, [{"theme": "Ch 1", "done": False, "extra_tags": ["NCERT", "CBSE"]}])

    captured = {}
    monkeypatch.setattr(
        run, "produce_video",
        lambda theme, style, duration, shorts, extra_tags=None, thumbnail_badge=None: captured.setdefault("extra_tags", extra_tags),
    )
    monkeypatch.setattr("sys.argv", ["run.py", "--themes-file", str(themes_file)])

    run.main()

    assert captured["extra_tags"] == ["NCERT", "CBSE"]


def test_main_passes_each_themes_thumbnail_badge_through(monkeypatch, tmp_path):
    themes_file = tmp_path / "themes_education.yaml"
    run.save_themes(themes_file, [{"theme": "Ch 1", "done": False, "thumbnail_badge": "CBSE | Grade 6 Maths"}])

    captured = {}
    monkeypatch.setattr(
        run, "produce_video",
        lambda theme, style, duration, shorts, extra_tags=None, thumbnail_badge=None: captured.setdefault("thumbnail_badge", thumbnail_badge),
    )
    monkeypatch.setattr("sys.argv", ["run.py", "--themes-file", str(themes_file)])

    run.main()

    assert captured["thumbnail_badge"] == "CBSE | Grade 6 Maths"


# -- TTS quota guardrail: stop cleanly instead of crashing the scheduled job ----

def test_main_stops_cleanly_and_leaves_theme_pending_when_tts_quota_exceeded(monkeypatch, tmp_path):
    themes_file = tmp_path / "themes.yaml"
    run.save_themes(themes_file, [
        {"theme": "Theme A", "done": False},
        {"theme": "Theme B", "done": False},
    ])

    def fake_produce_video(theme, style, duration, shorts, extra_tags=None, thumbnail_badge=None):
        raise run.TTSQuotaExceededError("budget exhausted")

    monkeypatch.setattr(run, "produce_video", fake_produce_video)
    monkeypatch.setattr("sys.argv", ["run.py", "--themes-file", str(themes_file), "--all"])

    run.main()  # must not raise

    themes = run.load_themes(themes_file)
    assert themes[0]["done"] is False
    assert themes[1]["done"] is False


def test_main_theme_flag_exits_cleanly_when_tts_quota_exceeded(monkeypatch):
    def fake_produce_video(theme, style, duration, shorts):
        raise run.TTSQuotaExceededError("budget exhausted")

    monkeypatch.setattr(run, "produce_video", fake_produce_video)
    monkeypatch.setattr("sys.argv", ["run.py", "--theme", "a theme"])

    try:
        run.main()
    except SystemExit as e:
        assert e.code == 0
    else:
        raise AssertionError("expected SystemExit(0)")
