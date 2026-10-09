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

HOOK_CARD_DURATION = 2.0


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
    elif kind == "diagram":
        # Static diagram PNG: no Ken-Burns zoompan -- a slow crop would eventually
        # clip diagram boxes/dots/labels off-frame, unlike photographic stock
        # where it's harmless.
        cmd = [
            FFMPEG_BIN, "-y", "-loop", "1", "-i", str(visual_path),
            "-t", str(duration), "-vf", scale_crop,
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


def _overlay_hook_card(clip_path: Path, hook_card_path: Path, duration: float, width: int, height: int, out_path: Path) -> None:
    """Overlays the branded hook card onto the first `duration` seconds of
    `clip_path` (scene 0's own clip) instead of prepending a separate clip to
    the video track. Prepending a clip made the video track longer than the
    audio track (narration starts at t=0 with no matching silence), which
    permanently shifted every scene's visual/on-screen text ~`duration`
    seconds later than its narration for the rest of the video. Overlaying
    keeps the video and audio tracks the same total length throughout.

    Builds the overlaid first-`duration`-seconds segment, then concatenates
    the untouched remainder of the clip after it — both inputs here are
    explicitly duration-bounded (rather than an `overlay=...:enable=...` over
    the whole clip with an infinitely-looped `-loop 1` image input), since
    that combination never signals EOF and makes ffmpeg hang indefinitely
    even with `-shortest` set."""
    clip_duration = _probe_duration(clip_path)
    hook_duration = min(duration, clip_duration)
    work_dir = out_path.parent
    stem = out_path.stem

    hook_segment = work_dir / f"{stem}_hookseg.mp4"
    _run([
        FFMPEG_BIN, "-y",
        "-t", str(hook_duration), "-i", str(clip_path),
        "-loop", "1", "-t", str(hook_duration), "-i", str(hook_card_path),
        "-filter_complex", f"[1:v]scale={width}:{height}[card];[0:v][card]overlay=0:0",
        "-c:v", "libx264", "-preset", "ultrafast", "-threads", "1", *_VIDEO_BITRATE_ARGS,
        "-pix_fmt", "yuv420p", str(hook_segment),
    ])

    if hook_duration >= clip_duration:
        hook_segment.replace(out_path)
        return

    remainder_segment = work_dir / f"{stem}_rest.mp4"
    _run([
        FFMPEG_BIN, "-y", "-ss", str(hook_duration), "-i", str(clip_path),
        "-c:v", "libx264", "-preset", "ultrafast", "-threads", "1", *_VIDEO_BITRATE_ARGS,
        "-pix_fmt", "yuv420p", str(remainder_segment),
    ])

    concat_list = work_dir / f"{stem}_concat.txt"
    concat_list.write_text(
        "\n".join(f"file '{p.resolve()}'" for p in (hook_segment, remainder_segment)), encoding="utf-8"
    )
    _run([
        FFMPEG_BIN, "-y", "-f", "concat", "-safe", "0", "-i", str(concat_list),
        "-c", "copy", str(out_path),
    ])


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

    `hook_card_path` (optional): a pre-rendered thumbnail image overlaid on top
    of scene 0's own clip for the first `HOOK_CARD_DURATION` seconds, to hook
    viewers before the first scene's visual appears underneath. It is overlaid
    rather than prepended as a separate clip, so the video track's total
    duration always matches the narration's — keeping every scene's on-screen
    text/visual in sync with its narration throughout the video.
    """
    if not FFMPEG_BIN or not FFPROBE_BIN:
        raise RuntimeError("ffmpeg/ffprobe not found on PATH. Install ffmpeg first.")

    width = width or config.VIDEO_WIDTH
    height = height or config.VIDEO_HEIGHT

    work_dir.mkdir(parents=True, exist_ok=True)
    clip_paths = []

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

        if i == 0 and hook_card_path:
            hook_clip_path = work_dir / f"scene_{i:02d}_hook.mp4"
            _overlay_hook_card(clip_path, hook_card_path, HOOK_CARD_DURATION, width, height, hook_clip_path)
            clip_path = hook_clip_path

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

    # Mux the narration onto the video. Both tracks total the same duration
    # (scene durations == narration durations, since each scene clip is built
    # to its matching narration's length) so no -shortest/padding is needed.
    out_path.parent.mkdir(parents=True, exist_ok=True)
    _run([
        FFMPEG_BIN, "-y", "-i", str(video_only_path), "-i", str(narration_concat_path),
        "-map", "0:v", "-map", "1:a", "-c:v", "copy", "-c:a", "aac", "-ar", "44100", "-ac", "2",
        str(out_path),
    ])
    return out_path
