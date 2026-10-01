from pathlib import Path
from unittest import mock

import pytest

from pipeline import assemble


def _record_calls(monkeypatch):
    calls = []
    monkeypatch.setattr(assemble, "_run", lambda cmd: calls.append(cmd))
    return calls


def test_build_hook_clip_has_no_audio_stream(monkeypatch, tmp_path):
    calls = _record_calls(monkeypatch)
    assemble._build_hook_clip(tmp_path / "card.png", 2.0, tmp_path / "hook.mp4", 720, 1280)
    assert len(calls) == 1
    assert "-an" in calls[0]


def test_build_scene_clip_image_uses_zoompan(monkeypatch, tmp_path):
    calls = _record_calls(monkeypatch)
    assemble._build_scene_clip(tmp_path / "img.jpg", "image", 5.0, tmp_path / "out.mp4", 720, 1280)
    cmd = calls[0]
    assert any("zoompan" in arg for arg in cmd)


def test_build_scene_clip_video_longer_than_duration_trims_without_looping(monkeypatch, tmp_path):
    calls = _record_calls(monkeypatch)
    monkeypatch.setattr(assemble, "_probe_duration", lambda path: 10.0)
    assemble._build_scene_clip(tmp_path / "vid.mp4", "video", 5.0, tmp_path / "out.mp4", 1280, 720)
    cmd = calls[0]
    assert "-stream_loop" not in cmd
    assert "-t" in cmd and "5.0" in cmd


def test_build_scene_clip_video_shorter_than_duration_loops(monkeypatch, tmp_path):
    calls = _record_calls(monkeypatch)
    monkeypatch.setattr(assemble, "_probe_duration", lambda path: 2.0)
    assemble._build_scene_clip(tmp_path / "vid.mp4", "video", 5.0, tmp_path / "out.mp4", 1280, 720)
    cmd = calls[0]
    assert "-stream_loop" in cmd
    loop_count = int(cmd[cmd.index("-stream_loop") + 1])
    assert loop_count >= 2  # ceil(5/2) + 1 margin, must fully cover target duration


def test_all_ffmpeg_calls_apply_the_bitrate_cap(monkeypatch, tmp_path):
    """Regression guard for the Telegram 50MB-upload-limit fix: every encode
    path must include the bitrate cap, not just some of them."""
    calls = _record_calls(monkeypatch)
    assemble._build_hook_clip(tmp_path / "card.png", 2.0, tmp_path / "hook.mp4", 720, 1280)
    monkeypatch.setattr(assemble, "_probe_duration", lambda path: 10.0)
    assemble._build_scene_clip(tmp_path / "img.jpg", "image", 5.0, tmp_path / "out1.mp4", 720, 1280)
    assemble._build_scene_clip(tmp_path / "vid.mp4", "video", 5.0, tmp_path / "out2.mp4", 720, 1280)

    for cmd in calls:
        assert "-b:v" in cmd and "1200k" in cmd


def test_overlay_text_renders_card_then_composites(monkeypatch, tmp_path):
    calls = _record_calls(monkeypatch)
    rendered = {}
    monkeypatch.setattr(
        assemble.textcard, "render_text_card",
        lambda text, w, h, out: (rendered.setdefault("text", text), out.touch())[1] or out,
    )
    assemble._overlay_text(tmp_path / "clip.mp4", "Revenue: Rs 412 Cr", 720, 1280, tmp_path / "out.mp4", tmp_path)
    assert rendered["text"] == "Revenue: Rs 412 Cr"
    assert len(calls) == 1
    assert "overlay" in calls[0][calls[0].index("-filter_complex") + 1]


def test_assemble_video_skips_hook_clip_when_not_provided(monkeypatch, tmp_path):
    calls = _record_calls(monkeypatch)
    monkeypatch.setattr(assemble, "_probe_duration", lambda path: 3.0)
    narration = tmp_path / "n0.mp3"
    narration.write_bytes(b"x")
    visual = tmp_path / "v0.jpg"
    visual.write_bytes(b"x")

    out = assemble.assemble_video(
        scenes=[{"narration": "hi", "on_screen_text": ""}],
        narration_paths=[narration],
        visual_paths=[(visual, "image")],
        out_path=tmp_path / "final.mp4",
        work_dir=tmp_path / "work",
        width=720, height=1280,
        hook_card_path=None,
    )
    assert out == tmp_path / "final.mp4"
    assert not (tmp_path / "work" / "hook_intro.mp4").exists()


def test_assemble_video_overlays_text_only_for_scenes_with_on_screen_text(monkeypatch, tmp_path):
    monkeypatch.setattr(assemble, "_run", lambda cmd: None)
    monkeypatch.setattr(assemble, "_probe_duration", lambda path: 3.0)
    overlaid = []
    monkeypatch.setattr(
        assemble, "_overlay_text",
        lambda clip_path, text, w, h, out_path, work_dir: overlaid.append(text) or out_path.touch(),
    )
    narration = tmp_path / "n0.mp3"
    narration.write_bytes(b"x")
    visual = tmp_path / "v0.jpg"
    visual.write_bytes(b"x")

    assemble.assemble_video(
        scenes=[
            {"narration": "hi", "on_screen_text": ""},
            {"narration": "bye", "on_screen_text": "Fact: 42"},
        ],
        narration_paths=[narration, narration],
        visual_paths=[(visual, "image"), (visual, "image")],
        out_path=tmp_path / "final.mp4",
        work_dir=tmp_path / "work",
        width=720, height=1280,
    )
    assert overlaid == ["Fact: 42"]


def test_assemble_video_raises_when_ffmpeg_missing(monkeypatch, tmp_path):
    monkeypatch.setattr(assemble, "FFMPEG_BIN", None)
    with pytest.raises(RuntimeError, match="ffmpeg/ffprobe not found"):
        assemble.assemble_video([], [], [], tmp_path / "out.mp4", tmp_path / "work")
