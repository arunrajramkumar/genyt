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


def test_produce_from_script_adds_shorts_hashtag_for_portrait_orientation(mocked_pipeline):
    from pipeline import config
    script = _stub_script()
    result = producer.produce_from_script(script, config.SHORTS, on_progress=lambda m: None)
    text = result["metadata"].read_text(encoding="utf-8")
    assert "#Shorts" in text


def test_produce_from_script_omits_shorts_hashtag_for_landscape_orientation(mocked_pipeline):
    from pipeline import config
    script = _stub_script()
    result = producer.produce_from_script(script, config.LANDSCAPE, on_progress=lambda m: None)
    text = result["metadata"].read_text(encoding="utf-8")
    assert "#Shorts" not in text


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


def test_append_hashtags_prefers_tags_that_appear_in_the_title():
    # "NCERT"/"CBSE" are generic required tags present on every video;
    # "Photosynthesis" is this video's actual topic and appears in the title
    # -- it should win a hashtag slot over the generic ones.
    result = producer._append_hashtags(
        "Desc.", ["NCERT", "CBSE", "Photosynthesis"],
        title="Photosynthesis Explained for Grade 6 Students",
    )
    assert result == "Desc.\n\n#Photosynthesis #NCERT #CBSE"


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
    def fake_render_hook_card(title, stat_text, width, height, out_path, background_path=None, badge_text=None, style="alert"):
        captured["stat_text"] = stat_text
        return out_path
    monkeypatch.setattr(producer.textcard, "render_hook_card", fake_render_hook_card)

    producer.produce_from_script(_stub_script(), on_progress=lambda m: None)
    assert captured["stat_text"] == "3 hearts"


# -- _pick_hook_stat: prefer a percentage, then the largest number --------------

def test_pick_hook_stat_prefers_percentage_over_larger_plain_number():
    scenes = [
        {"on_screen_text": "Revenue: Rs 4120 Cr"},
        {"on_screen_text": "Profit grew 340%"},
    ]
    assert producer._pick_hook_stat(scenes) == "Profit grew 340%"


def test_pick_hook_stat_picks_largest_number_when_no_percentage_present():
    scenes = [
        {"on_screen_text": "FY23"},
        {"on_screen_text": "Revenue: Rs 4120 Cr"},
        {"on_screen_text": "P/E: 22.5"},
    ]
    assert producer._pick_hook_stat(scenes) == "Revenue: Rs 4120 Cr"


def test_pick_hook_stat_returns_empty_string_when_no_scene_has_a_number():
    scenes = [{"on_screen_text": "Intro"}, {"on_screen_text": ""}]
    assert producer._pick_hook_stat(scenes) == ""


def test_pick_hook_stat_skips_scenes_with_no_on_screen_text():
    scenes = [{}, {"on_screen_text": "22% ROE"}]
    assert producer._pick_hook_stat(scenes) == "22% ROE"


def test_pick_hook_stat_falls_back_to_narration_percentage_when_no_on_screen_text_has_a_number():
    scenes = [
        {"narration": "Titan Company Ltd posted a 24.3% year-on-year revenue jump.", "on_screen_text": ""},
        {"narration": "The stock looks promising.", "on_screen_text": "Intro"},
    ]
    assert producer._pick_hook_stat(scenes) == "24.3% Revenue"


def test_pick_hook_stat_still_returns_empty_string_when_neither_source_has_a_number():
    scenes = [{"narration": "Intro", "on_screen_text": "Intro"}]
    assert producer._pick_hook_stat(scenes) == ""


# -- _pick_hook_style: calm (SIP/wealth) vs. default alert (breaking news) ---

def test_pick_hook_style_defaults_to_alert_for_a_plain_stock_title():
    assert producer._pick_hook_style("Titan Company Ltd: 24% Revenue Jump") == "alert"


@pytest.mark.parametrize("title", [
    "Start a ₹5,000 SIP Today, Retire With ₹2 Crore",
    "How to Become a Crorepati With Mutual Funds",
    "The Power of Compounding Explained",
    "Achieve Financial Freedom in 10 Years",
])
def test_pick_hook_style_picks_calm_for_wealth_building_titles(title):
    assert producer._pick_hook_style(title) == "calm"


def test_pick_hook_style_is_case_insensitive():
    assert producer._pick_hook_style("BUILD YOUR WEALTH WITH SIP") == "calm"


# -- _pick_hook_background: use a real visual instead of a blank card --------

def test_pick_hook_background_uses_first_scene_image_directly():
    path = Path("/tmp/visual.jpg")
    assert producer._pick_hook_background([(path, "image")], Path("/tmp/work")) == path


def test_pick_hook_background_uses_first_scene_diagram_directly():
    path = Path("/tmp/diagram.png")
    assert producer._pick_hook_background([(path, "diagram")], Path("/tmp/work")) == path


def test_pick_hook_background_extracts_frame_when_first_scene_is_video(monkeypatch, tmp_path):
    """Regression test: a stock-footage video (the common case for finance
    queries, since Pexels usually returns a video before a photo) was
    previously skipped entirely, leaving the thumbnail with no background."""
    video_path = Path("/tmp/clip.mp4")
    captured = {}

    def fake_extract_frame(path, out_path):
        captured["path"] = path
        captured["out_path"] = out_path
        return out_path

    monkeypatch.setattr(producer.assemble, "extract_frame", fake_extract_frame)

    result = producer._pick_hook_background([(video_path, "video")], tmp_path)

    assert result == tmp_path / "hook_bg.jpg"
    assert captured["path"] == video_path


def test_pick_hook_background_falls_back_to_none_when_frame_extraction_fails(monkeypatch, tmp_path):
    monkeypatch.setattr(
        producer.assemble, "extract_frame",
        mock.Mock(side_effect=RuntimeError("ffmpeg failed")),
    )
    result = producer._pick_hook_background([(Path("/tmp/clip.mp4"), "video")], tmp_path, on_progress=lambda m: None)
    assert result is None


def test_pick_hook_background_returns_none_when_no_visuals():
    assert producer._pick_hook_background([], Path("/tmp/work")) is None


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
    logs = []
    result = producer.produce_from_script(_stub_script(), on_progress=logs.append)
    assert captured["video_id"] == "abc"
    assert captured["thumbnail_path"] == result["thumbnail"]
    assert any("Custom thumbnail set" in msg and "abc" in msg for msg in logs)


def test_produce_from_script_records_performance_log_entry_for_finance_video(mocked_pipeline, monkeypatch):
    monkeypatch.setattr(
        producer.youtube_upload, "upload_video",
        lambda *a, **k: {"video_id": "abc", "url": "https://youtu.be/abc"},
    )
    recorded = {}
    monkeypatch.setattr(
        producer.performance_log, "record",
        lambda video_id, symbol, company_name: recorded.update(
            video_id=video_id, symbol=symbol, company_name=company_name,
        ),
    )
    script = _stub_script()
    script["symbol"] = "CDSL"
    script["company_name"] = "CDSL Ltd"
    producer.produce_from_script(script, on_progress=lambda m: None)
    assert recorded == {"video_id": "abc", "symbol": "CDSL", "company_name": "CDSL Ltd"}


def test_produce_from_script_skips_performance_log_when_no_symbol(mocked_pipeline, monkeypatch):
    monkeypatch.setattr(
        producer.youtube_upload, "upload_video",
        lambda *a, **k: {"video_id": "abc", "url": "https://youtu.be/abc"},
    )
    called = []
    monkeypatch.setattr(producer.performance_log, "record", lambda *a, **k: called.append(1))
    producer.produce_from_script(_stub_script(), on_progress=lambda m: None)
    assert called == []


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


def test_produce_from_script_skips_playlist_add_when_not_configured(mocked_pipeline, monkeypatch):
    monkeypatch.setattr(producer.config, "YOUTUBE_PLAYLIST_ID", "")
    monkeypatch.setattr(
        producer.youtube_upload, "upload_video",
        lambda *a, **k: {"video_id": "abc", "url": "https://youtu.be/abc"},
    )
    called = []
    monkeypatch.setattr(producer.youtube_upload, "add_to_playlist", lambda *a, **k: called.append(1))
    producer.produce_from_script(_stub_script(), on_progress=lambda m: None)
    assert called == []


def test_produce_from_script_adds_to_playlist_when_configured(mocked_pipeline, monkeypatch):
    monkeypatch.setattr(producer.config, "YOUTUBE_PLAYLIST_ID", "PL123")
    monkeypatch.setattr(
        producer.youtube_upload, "upload_video",
        lambda *a, **k: {"video_id": "abc", "url": "https://youtu.be/abc"},
    )
    captured = {}
    monkeypatch.setattr(
        producer.youtube_upload, "add_to_playlist",
        lambda video_id, playlist_id: captured.update(video_id=video_id, playlist_id=playlist_id),
    )
    producer.produce_from_script(_stub_script(), on_progress=lambda m: None)
    assert captured == {"video_id": "abc", "playlist_id": "PL123"}


def test_produce_from_script_playlist_add_failure_does_not_fail_the_job(mocked_pipeline, monkeypatch):
    monkeypatch.setattr(producer.config, "YOUTUBE_PLAYLIST_ID", "PL123")
    logs = []
    monkeypatch.setattr(
        producer.youtube_upload, "upload_video",
        lambda *a, **k: {"video_id": "abc", "url": "https://youtu.be/abc"},
    )
    monkeypatch.setattr(
        producer.youtube_upload, "add_to_playlist",
        mock.Mock(side_effect=RuntimeError("playlist not found")),
    )
    result = producer.produce_from_script(_stub_script(), on_progress=logs.append)
    assert result["youtube_url"] == "https://youtu.be/abc"
    assert any("playlist not found" in msg for msg in logs)


def test_produce_from_script_passes_thumbnail_badge_to_hook_card(mocked_pipeline, monkeypatch):
    captured = {}

    def fake_render_hook_card(title, stat_text, width, height, out_path, background_path=None, badge_text=None, style="alert"):
        captured["badge_text"] = badge_text
        return out_path

    monkeypatch.setattr(producer.textcard, "render_hook_card", fake_render_hook_card)
    producer.produce_from_script(_stub_script(), on_progress=lambda m: None, thumbnail_badge="CBSE | Grade 6 Maths")
    assert captured["badge_text"] == "CBSE | Grade 6 Maths"


def test_produce_from_script_defaults_badge_to_company_name_when_no_explicit_badge(mocked_pipeline, monkeypatch):
    captured = {}

    def fake_render_hook_card(title, stat_text, width, height, out_path, background_path=None, badge_text=None, style="alert"):
        captured["badge_text"] = badge_text
        return out_path

    monkeypatch.setattr(producer.textcard, "render_hook_card", fake_render_hook_card)
    script = _stub_script()
    script["company_name"] = "Titan Company Ltd"
    producer.produce_from_script(script, on_progress=lambda m: None)
    assert captured["badge_text"] == "Titan Company Ltd"


def test_produce_from_script_leaves_badge_empty_when_no_badge_and_no_company_name(mocked_pipeline, monkeypatch):
    captured = {}

    def fake_render_hook_card(title, stat_text, width, height, out_path, background_path=None, badge_text=None, style="alert"):
        captured["badge_text"] = badge_text
        return out_path

    monkeypatch.setattr(producer.textcard, "render_hook_card", fake_render_hook_card)
    producer.produce_from_script(_stub_script(), on_progress=lambda m: None)
    assert captured["badge_text"] is None


# -- Google TTS whole-video budget preflight -------------------------------------

def test_produce_from_script_checks_google_tts_budget_before_any_synthesis(mocked_pipeline, monkeypatch):
    """The whole-video character total must be checked BEFORE the per-scene
    synthesize() loop starts, so a video can't partially narrate before
    hitting the monthly cap mid-way through."""
    captured = {}
    monkeypatch.setattr(producer.tts, "ensure_google_tts_budget", lambda total: captured.setdefault("total", total))
    script = _stub_script()
    producer.produce_from_script(script, on_progress=lambda m: None)
    expected_total = sum(len(s["narration"]) for s in script["scenes"])
    assert captured["total"] == expected_total


def test_produce_from_script_raises_before_synthesizing_when_budget_exhausted(mocked_pipeline, monkeypatch):
    def raise_quota_exceeded(total):
        raise producer.tts.TTSQuotaExceededError("budget exhausted")

    monkeypatch.setattr(producer.tts, "ensure_google_tts_budget", raise_quota_exceeded)
    synthesize_calls = []
    monkeypatch.setattr(producer.tts, "synthesize", lambda *a, **k: synthesize_calls.append(1))

    with pytest.raises(producer.tts.TTSQuotaExceededError):
        producer.produce_from_script(_stub_script(), on_progress=lambda m: None)
    assert synthesize_calls == []


# -- AI-generated (Pollinations.ai, free/keyless) hook-card background --------

def test_produce_from_script_uses_ai_background_when_generated(mocked_pipeline, monkeypatch):
    monkeypatch.setattr(producer.ai_thumbnail, "generate_hook_background", lambda *a, **k: Path("/tmp/ai_bg.png"))
    background_calls = []

    def fake_pick_hook_background(*a, **k):
        background_calls.append(1)
        return None

    monkeypatch.setattr(producer, "_pick_hook_background", fake_pick_hook_background)

    captured = {}
    def fake_render_hook_card(title, stat_text, width, height, out_path, background_path=None, badge_text=None, style="alert"):
        captured["background_path"] = background_path
        return out_path
    monkeypatch.setattr(producer.textcard, "render_hook_card", fake_render_hook_card)

    producer.produce_from_script(_stub_script(), on_progress=lambda m: None)

    assert captured["background_path"] == Path("/tmp/ai_bg.png")
    assert background_calls == []  # the Pexels fallback was never needed


def test_produce_from_script_falls_back_to_stock_background_when_ai_background_unavailable(mocked_pipeline, monkeypatch, tmp_path):
    monkeypatch.setattr(producer.ai_thumbnail, "generate_hook_background", lambda *a, **k: None)

    captured = {}
    def fake_render_hook_card(title, stat_text, width, height, out_path, background_path=None, badge_text=None, style="alert"):
        captured["background_path"] = background_path
        return out_path
    monkeypatch.setattr(producer.textcard, "render_hook_card", fake_render_hook_card)

    producer.produce_from_script(_stub_script(), on_progress=lambda m: None)

    assert captured["background_path"] == tmp_path / "visual.jpg"  # falls back to the scene's own visual


def test_produce_from_script_passes_company_name_to_ai_background(mocked_pipeline, monkeypatch):
    captured = {}
    monkeypatch.setattr(
        producer.ai_thumbnail, "generate_hook_background",
        lambda company_name, out_path, on_progress=print, width=None, height=None: captured.setdefault("company_name", company_name),
    )
    script = _stub_script()
    script["company_name"] = "Arrow Greentech"
    producer.produce_from_script(script, on_progress=lambda m: None)
    assert captured["company_name"] == "Arrow Greentech"


def test_produce_from_script_reuses_ai_background_as_hook_scene_visual(mocked_pipeline, monkeypatch, tmp_path):
    # A unique, company-specific AI image is a stronger differentiator on the
    # most-watched scene than generic Pexels stock footage -- reuse it as
    # scene 0's own in-video visual, not just the separate thumbnail PNG.
    ai_bg_path = tmp_path / "ai_bg.png"
    monkeypatch.setattr(producer.ai_thumbnail, "generate_hook_background", lambda *a, **k: ai_bg_path)
    fetch_calls = []
    monkeypatch.setattr(
        producer.visuals, "fetch_visual",
        lambda query, orientation, target_width, exclude=None: (fetch_calls.append(query) or (tmp_path / "visual.jpg", "image")),
    )

    captured_visual_paths = {}
    def fake_assemble_video(scenes, narration_paths, visual_paths, out_path, work_dir, width, height, hook_card_path):
        captured_visual_paths["value"] = visual_paths
        return out_path
    monkeypatch.setattr(producer.assemble, "assemble_video", fake_assemble_video)

    script = _stub_script()
    script["company_name"] = "Arrow Greentech"
    producer.produce_from_script(script, on_progress=lambda m: None)

    assert captured_visual_paths["value"][0] == (ai_bg_path, "image")
    assert fetch_calls == ["octopus color"]  # only scene 1 hit Pexels, not scene 0


def test_produce_from_script_renders_diagram_instead_of_fetching_stock_visual(mocked_pipeline, monkeypatch):
    script = _stub_script()
    script["scenes"][0]["diagram"] = {"kind": "number_line", "min": 1, "max": 5, "highlight": [3]}

    fetch_calls = []
    monkeypatch.setattr(producer.visuals, "fetch_visual", lambda *a, **k: fetch_calls.append(1) or (Path("/tmp/x.jpg"), "image"))

    rendered = []
    def fake_render_diagram(spec, width, height, out_path):
        rendered.append(spec)
        out_path.write_bytes(b"png")
        return out_path
    monkeypatch.setattr(producer.diagrams, "render_diagram", fake_render_diagram)

    producer.produce_from_script(script, on_progress=lambda m: None)

    assert rendered == [script["scenes"][0]["diagram"]]
    assert len(fetch_calls) == 1  # only scene 1 (no diagram) hits Pexels


def test_produce_from_script_falls_back_to_stock_visual_when_diagram_render_fails(mocked_pipeline, monkeypatch):
    script = _stub_script()
    script["scenes"][0]["diagram"] = {"kind": "number_line", "min": 1, "max": 5}

    def failing_render(spec, width, height, out_path):
        raise RuntimeError("boom")
    monkeypatch.setattr(producer.diagrams, "render_diagram", failing_render)

    fetch_calls = []
    def fake_fetch_visual(query, orientation, target_width, exclude=None):
        fetch_calls.append(query)
        return (Path("/tmp/x.jpg"), "image")
    monkeypatch.setattr(producer.visuals, "fetch_visual", fake_fetch_visual)

    logs = []
    producer.produce_from_script(script, on_progress=logs.append)

    assert len(fetch_calls) == 2  # both scenes fell through to Pexels
    assert any("diagram render failed" in msg for msg in logs)
