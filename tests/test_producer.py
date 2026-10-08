from pathlib import Path
from unittest import mock

import pytest

from pipeline import producer


@pytest.mark.parametrize("title,expected", [
    ("5 Facts About Octopuses", "5-facts-about-octopuses"),
    ("XYZ Industries Ltd.: What You Should Know?!", "xyz-industries-ltd-what-you-should-know"),
    ("   leading/trailing---dashes   ", "leading-trailing-dashes"),
])
def test_slugify_normalizes_titles(title, expected):
    assert producer.slugify(title) == expected


def test_slugify_falls_back_to_video_for_non_latin_title():
    # Devanagari/Tamil/etc. titles have no a-z0-9 chars to keep — must not
    # produce an empty slug (which would break output file naming).
    assert producer.slugify("गोपाल की कहानी") == "video"
    assert producer.slugify("ஒரு குறுகிய கதை") == "video"


def test_slugify_truncates_to_60_chars():
    long_title = "word " * 30
    assert len(producer.slugify(long_title)) <= 60


def test_slugify_empty_string_falls_back_to_video():
    assert producer.slugify("") == "video"


# -- produce_from_script orchestration (all external services mocked) ---------

def _stub_script():
    return {
        "title": "Octopus Facts",
        "description": "desc",
        "tags": ["a", "b"],
        "scenes": [
            {"narration": "Octopuses have three hearts.", "visual_query": "octopus", "on_screen_text": ""},
            {"narration": "They can change color.", "visual_query": "octopus color", "on_screen_text": "3 hearts"},
        ],
    }


@pytest.fixture
def mocked_pipeline(monkeypatch, isolated_dirs, tmp_path):
    monkeypatch.setattr(producer.tts, "synthesize", lambda text, path, voice=None: (path.write_bytes(b"a"), 3.0)[1])
    monkeypatch.setattr(
        producer.visuals, "fetch_visual",
        lambda query, orientation, target_width, exclude=None: (tmp_path / "visual.jpg", "image"),
    )
    (tmp_path / "visual.jpg").write_bytes(b"img")
    monkeypatch.setattr(producer.textcard, "render_hook_card", lambda *a, **k: a[4])
    monkeypatch.setattr(
        producer.assemble, "assemble_video",
        lambda scenes, narration_paths, visual_paths, out_path, work_dir, width, height, hook_card_path: out_path,
    )
    return {}


def test_produce_from_script_excludes_already_used_visuals_for_later_scenes(mocked_pipeline, monkeypatch):
    """Regression test: scenes sharing a generic query used to be able to land
    on the exact same cached clip via independent random picks — visuals
    should avoid repeating within a single video when alternatives exist."""
    seen_excludes = []
    responses = [(Path(f"/tmp/visual_{i}.jpg"), "image") for i in range(2)]

    def fake_fetch_visual(query, orientation, target_width, exclude=None):
        seen_excludes.append(set(exclude or set()))
        return responses[len(seen_excludes) - 1]

    monkeypatch.setattr(producer.visuals, "fetch_visual", fake_fetch_visual)
    producer.produce_from_script(_stub_script(), on_progress=lambda m: None)

    assert seen_excludes[0] == set()
    assert seen_excludes[1] == {responses[0][0]}


def test_produce_from_script_fills_missing_first_scene_on_screen_text_with_title(mocked_pipeline):
    script = _stub_script()
    producer.produce_from_script(script, on_progress=lambda m: None)
    assert script["scenes"][0]["on_screen_text"] == "Octopus Facts"


def test_produce_from_script_returns_expected_output_paths(mocked_pipeline):
    from pipeline import config
    script = _stub_script()
    result = producer.produce_from_script(script, config.SHORTS, on_progress=lambda m: None)

    assert result["slug"] == "octopus-facts"
    assert result["video"].name == "octopus-facts.mp4"
    assert result["srt"].exists()
    assert result["metadata"].exists()


def test_produce_from_script_passes_voice_override_to_tts(mocked_pipeline, monkeypatch):
    captured = {}
    monkeypatch.setattr(
        producer.tts, "synthesize",
        lambda text, path, voice=None: (captured.setdefault("voice", voice), path.write_bytes(b"a"), 3.0)[2],
    )
    producer.produce_from_script(_stub_script(), on_progress=lambda m: None, voice="hi-IN-SwaraNeural")
    assert captured["voice"] == "hi-IN-SwaraNeural"


def test_produce_from_script_metadata_file_contains_title_description_tags(mocked_pipeline):
    script = _stub_script()
    result = producer.produce_from_script(script, on_progress=lambda m: None)
    text = result["metadata"].read_text(encoding="utf-8")
    assert "Octopus Facts" in text
    assert "desc" in text
    assert "a, b" in text


# -- hashtag appending: every upload's description should end with #hashtags --

def test_append_hashtags_appends_top_tags_as_hashtags():
    result = producer._append_hashtags("A description.", ["stock analysis", "nifty50", "long term investing"])
    assert result == "A description.\n\n#stockanalysis #nifty50 #longterminvesting"


def test_append_hashtags_caps_at_three_tags():
    result = producer._append_hashtags("Desc.", ["a", "b", "c", "d", "e"])
    assert result == "Desc.\n\n#a #b #c"


def test_append_hashtags_skips_tags_that_become_empty_after_cleaning():
    result = producer._append_hashtags("Desc.", ["!!!", "real tag"])
    assert result == "Desc.\n\n#realtag"


def test_append_hashtags_leaves_description_untouched_when_no_tags():
    assert producer._append_hashtags("Desc.", []) == "Desc."


def test_produce_from_script_metadata_file_includes_hashtags_from_tags(mocked_pipeline):
    script = _stub_script()
    result = producer.produce_from_script(script, on_progress=lambda m: None)
    text = result["metadata"].read_text(encoding="utf-8")
    assert "#a #b" in text


def test_produce_from_script_upload_description_includes_hashtags(mocked_pipeline, monkeypatch):
    captured = {}

    def fake_upload(video_path, title, description, tags, **kwargs):
        captured["description"] = description
        return {"video_id": "abc", "url": "https://youtu.be/abc"}

    monkeypatch.setattr(producer.youtube_upload, "upload_video", fake_upload)
    producer.produce_from_script(_stub_script(), on_progress=lambda m: None)
    assert captured["description"].endswith("#a #b")


def test_produce_from_script_picks_first_numeric_on_screen_text_as_stat_for_hook_card(mocked_pipeline, monkeypatch):
    captured = {}
    def fake_render_hook_card(title, stat_text, width, height, out_path, background_path=None, badge_text=None):
        captured["stat_text"] = stat_text
        return out_path
    monkeypatch.setattr(producer.textcard, "render_hook_card", fake_render_hook_card)

    producer.produce_from_script(_stub_script(), on_progress=lambda m: None)
    assert captured["stat_text"] == "3 hearts"


def test_produce_from_script_clears_stale_work_dir(mocked_pipeline):
    from pipeline import config
    script = _stub_script()
    work_dir = config.CACHE_DIR / "octopus-facts"
    work_dir.mkdir(parents=True)
    (work_dir / "stale_file.txt").write_text("leftover from a previous run")

    producer.produce_from_script(script, on_progress=lambda m: None)
    assert not (work_dir / "stale_file.txt").exists()


# -- YouTube upload wiring ------------------------------------------------------

def test_produce_from_script_skips_upload_silently_when_not_configured(mocked_pipeline):
    # No cache/youtube_token.json exists in the isolated test cache dir, so this
    # must behave as "not configured yet" rather than crash the whole job.
    result = producer.produce_from_script(_stub_script(), on_progress=lambda m: None)
    assert result["youtube_url"] is None


def test_produce_from_script_sets_youtube_url_on_successful_upload(mocked_pipeline, monkeypatch):
    monkeypatch.setattr(
        producer.youtube_upload, "upload_video",
        lambda *a, **k: {"video_id": "abc", "url": "https://youtu.be/abc"},
    )
    result = producer.produce_from_script(_stub_script(), on_progress=lambda m: None)
    assert result["youtube_url"] == "https://youtu.be/abc"


def test_produce_from_script_upload_failure_does_not_fail_the_job(mocked_pipeline, monkeypatch):
    logs = []
    monkeypatch.setattr(
        producer.youtube_upload, "upload_video",
        mock.Mock(side_effect=RuntimeError("quota exceeded")),
    )
    result = producer.produce_from_script(_stub_script(), on_progress=logs.append)
    assert result["youtube_url"] is None
    assert result["video"].exists() or result["video"].name  # job still completed
    assert any("quota exceeded" in msg for msg in logs)


def test_produce_from_script_respects_youtube_auto_upload_flag(mocked_pipeline, monkeypatch):
    called = []
    monkeypatch.setattr(producer.config, "YOUTUBE_AUTO_UPLOAD", False)
    monkeypatch.setattr(producer.youtube_upload, "upload_video", lambda *a, **k: called.append(1))
    result = producer.produce_from_script(_stub_script(), on_progress=lambda m: None)
    assert called == []
    assert result["youtube_url"] is None


def test_produce_from_script_sets_thumbnail_after_successful_upload(mocked_pipeline, monkeypatch):
    captured = {}
    monkeypatch.setattr(
        producer.youtube_upload, "upload_video",
        lambda *a, **k: {"video_id": "abc", "url": "https://youtu.be/abc"},
    )
    monkeypatch.setattr(
        producer.youtube_upload, "set_thumbnail",
        lambda video_id, thumbnail_path: captured.update(video_id=video_id, thumbnail_path=thumbnail_path),
    )
    result = producer.produce_from_script(_stub_script(), on_progress=lambda m: None)
    assert captured["video_id"] == "abc"
    assert captured["thumbnail_path"] == result["thumbnail"]


def test_produce_from_script_thumbnail_upload_failure_does_not_fail_the_job(mocked_pipeline, monkeypatch):
    logs = []
    monkeypatch.setattr(
        producer.youtube_upload, "upload_video",
        lambda *a, **k: {"video_id": "abc", "url": "https://youtu.be/abc"},
    )
    monkeypatch.setattr(
        producer.youtube_upload, "set_thumbnail",
        mock.Mock(side_effect=RuntimeError("channel not verified")),
    )
    result = producer.produce_from_script(_stub_script(), on_progress=logs.append)
    assert result["youtube_url"] == "https://youtu.be/abc"
    assert any("channel not verified" in msg for msg in logs)


def test_produce_from_script_passes_thumbnail_badge_to_hook_card(mocked_pipeline, monkeypatch):
    captured = {}

    def fake_render_hook_card(title, stat_text, width, height, out_path, background_path=None, badge_text=None):
        captured["badge_text"] = badge_text
        return out_path

    monkeypatch.setattr(producer.textcard, "render_hook_card", fake_render_hook_card)
    producer.produce_from_script(_stub_script(), on_progress=lambda m: None, thumbnail_badge="CBSE | Grade 6 Maths")
    assert captured["badge_text"] == "CBSE | Grade 6 Maths"
