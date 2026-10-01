"""Assembles per-scene visuals + narration into a single finished MP4 using ffmpeg."""
import shutil
import subprocess
from pathlib import Path

from . import config, textcard

FFMPEG_BIN = shutil.which("ffmpeg")
FFPROBE_BIN = shutil.which("ffprobe")

# Caps the video bitrate so a full video stays comfortably under Telegram's
# 50MB bot-upload limit even at the longest supported duration (180s) — ffmpeg's
# default CRF with no bitrate cap produced files ranging 10-48MB unpredictably,
# occasionally tipping over the limit and silently failing the Telegram send.
_VIDEO_BITRATE_ARGS = ["-b:v", "1200k", "-maxrate", "1500k", "-bufsize", "3000k"]


def _run(cmd: list[str]) -> None:
    subprocess.run(cmd, check=True, capture_output=True)


def _probe_duration(path: Path) -> float:
    result = subprocess.run(
        [FFPROBE_BIN, "-v", "error", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", str(path)],
        check=True, capture_output=True, text=True,
    )
    return float(result.stdout.strip())


def _build_scene_clip(visual_path: Path, kind: str, duration: float, out_path: Path, width: int, height: int) -> None:
    """Render one scene's visual (image or video) to exactly `duration` seconds,
    scaled/cropped to the target resolution."""
    w, h = width, height
    scale_crop = f"scale={w}:{h}:force_original_aspect_ratio=increase,crop={w}:{h}"

    if kind == "image":
        # Slow pan/zoom (Ken Burns) over a still image.
        zoom_frames = int(duration * config.VIDEO_FPS)
        vf = (
            f"{scale_crop},"
            f"zoompan=z='min(zoom+0.0006,1.15)':d={zoom_frames}:"
            f"x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)':s={w}x{h}:fps={config.VIDEO_FPS}"
        )
        cmd = [
            FFMPEG_BIN, "-y", "-loop", "1", "-i", str(visual_path),
            "-t", str(duration), "-vf", vf,
            "-c:v", "libx264", "-preset", "ultrafast", "-threads", "1", *_VIDEO_BITRATE_ARGS,
            "-pix_fmt", "yuv420p", str(out_path),
        ]
    else:
        # Pexels source clips vary in native frame rate (25/30/59.94fps). The
        # scenes are concatenated later with stream copy (-c copy), which
        # requires identical codec parameters across segments — mismatched
        # frame rates there corrupt the container's duration metadata and
        # desync audio from video. Force a consistent fps here so every scene
        # clip is uniform before concatenation.
        source_duration = _probe_duration(visual_path)
        if source_duration >= duration:
            cmd = [
                FFMPEG_BIN, "-y", "-i", str(visual_path),
                "-t", str(duration), "-vf", scale_crop, "-r", str(config.VIDEO_FPS),
                "-an", "-c:v", "libx264", "-preset", "ultrafast", "-threads", "1", *_VIDEO_BITRATE_ARGS,
                "-pix_fmt", "yuv420p", str(out_path),
            ]
        else:
            loops = int(duration // source_duration) + 1
            cmd = [
                FFMPEG_BIN, "-y", "-stream_loop", str(loops), "-i", str(visual_path),
                "-t", str(duration), "-vf", scale_crop, "-r", str(config.VIDEO_FPS),
                "-an", "-c:v", "libx264", "-preset", "ultrafast", "-threads", "1", *_VIDEO_BITRATE_ARGS,
                "-pix_fmt", "yuv420p", str(out_path),
            ]
    _run(cmd)


def _overlay_text(clip_path: Path, text: str, width: int, height: int, out_path: Path, work_dir: Path) -> None:
    """Burns a pre-rendered text-card PNG onto `clip_path` via the plain
    `overlay` filter (works without libass/freetype support)."""
    card_path = work_dir / f"{out_path.stem}_card.png"
    textcard.render_text_card(text, width, height, card_path)
    _run([
        FFMPEG_BIN, "-y", "-i", str(clip_path), "-i", str(card_path),
        "-filter_complex", "[0:v][1:v]overlay=0:0",
        "-c:v", "libx264", "-preset", "ultrafast", "-threads", "1", *_VIDEO_BITRATE_ARGS,
        "-pix_fmt", "yuv420p", str(out_path),
    ])


def _build_hook_clip(hook_card_path: Path, duration: float, out_path: Path, width: int, height: int) -> None:
    """Builds a silent video-only clip from the hook-card thumbnail image, so it
    can be prepended to the scene clips before the final stream-copy concat.
    Silent (no audio stream) because narration audio is overlaid separately,
    starting at t=0, so it plays under this card instead of waiting for it."""
    cmd = [
        FFMPEG_BIN, "-y",
        "-loop", "1", "-i", str(hook_card_path),
        "-t", str(duration),
        "-vf", f"scale={width}:{height}:force_original_aspect_ratio=increase,crop={width}:{height}",
        "-r", str(config.VIDEO_FPS),
        "-an", "-c:v", "libx264", "-preset", "ultrafast", "-threads", "1", *_VIDEO_BITRATE_ARGS,
        "-pix_fmt", "yuv420p",
        str(out_path),
    ]
    _run(cmd)


def assemble_video(
    scenes: list[dict],
    narration_paths: list[Path],
    visual_paths: list[tuple[Path, str]],
    out_path: Path,
    work_dir: Path,
    width: int = None,
    height: int = None,
    hook_card_path: Path = None,
) -> Path:
    """scenes: script scenes; narration_paths/visual_paths align by index.

    Captions are produced separately as an .srt file (see pipeline.captions) and
    uploaded to YouTube as closed captions rather than burned in, since this
    ffmpeg build has no libass/subtitle filter support.

    `hook_card_path` (optional): a pre-rendered thumbnail image prepended as a
    ~2s silent-video intro clip, to hook viewers before the first scene's visual
    appears. Narration audio starts at t=0 regardless (playing under the hook
    card) rather than waiting for the intro to finish.
    """
    if not FFMPEG_BIN or not FFPROBE_BIN:
        raise RuntimeError("ffmpeg/ffprobe not found on PATH. Install ffmpeg first.")

    width = width or config.VIDEO_WIDTH
    height = height or config.VIDEO_HEIGHT

    work_dir.mkdir(parents=True, exist_ok=True)
    clip_paths = []

    if hook_card_path:
        hook_clip_path = work_dir / "hook_intro.mp4"
        _build_hook_clip(hook_card_path, 2.0, hook_clip_path, width, height)
        clip_paths.append(hook_clip_path)

    for i, (scene, narration_path, (visual_path, kind)) in enumerate(
        zip(scenes, narration_paths, visual_paths)
    ):
        duration = _probe_duration(narration_path)
        clip_path = work_dir / f"scene_{i:02d}.mp4"
        _build_scene_clip(visual_path, kind, duration, clip_path, width, height)

        on_screen_text = (scene.get("on_screen_text") or "").strip()
        if on_screen_text:
            text_clip_path = work_dir / f"scene_{i:02d}_text.mp4"
            _overlay_text(clip_path, on_screen_text, width, height, text_clip_path, work_dir)
            clip_path = text_clip_path

        clip_paths.append(clip_path)

    concat_list = work_dir / "concat.txt"
    concat_list.write_text(
        "\n".join(f"file '{p.resolve()}'" for p in clip_paths), encoding="utf-8"
    )
    video_only_path = work_dir / "video_only.mp4"
    _run([
        FFMPEG_BIN, "-y", "-f", "concat", "-safe", "0", "-i", str(concat_list),
        "-c", "copy", str(video_only_path),
    ])

    # Concatenate all narration clips into one continuous audio track — re-encoded
    # (not stream-copied) since the source mp3s may not share identical codec
    # params, which -c copy concat requires.
    audio_concat_list = work_dir / "audio_concat.txt"
    audio_concat_list.write_text(
        "\n".join(f"file '{p.resolve()}'" for p in narration_paths), encoding="utf-8"
    )
    narration_concat_path = work_dir / "narration_concat.m4a"
    _run([
        FFMPEG_BIN, "-y", "-f", "concat", "-safe", "0", "-i", str(audio_concat_list),
        "-c:a", "aac", "-ar", "44100", "-ac", "2", str(narration_concat_path),
    ])

    # Mux the narration (starting at t=0) onto the full video. No -shortest: the
    # video (hook + scenes) is always >= the audio (scenes only), so the video
    # plays out in full with the final ~2s silent rather than cutting it short.
    out_path.parent.mkdir(parents=True, exist_ok=True)
    _run([
        FFMPEG_BIN, "-y", "-i", str(video_only_path), "-i", str(narration_concat_path),
        "-map", "0:v", "-map", "1:a", "-c:v", "copy", "-c:a", "aac", "-ar", "44100", "-ac", "2",
        str(out_path),
    ])
    return out_path
