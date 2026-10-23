"""Assembles per-scene visuals + narration into a single finished MP4 using ffmpeg."""
import random
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

# The opener card is now a transparent overlay (text over moving footage) that
# fades out -- see _overlay_hook_card. It used to be the opaque thumbnail over a
# still image, i.e. a frozen text slide for the first 3s of every Short, and
# ~70% of plays ended in the first few seconds (engagedViews/views ~30% on
# every video, 2026-10-23 analytics), so the Shorts feed never widened its test.
HOOK_OVERLAY_DURATION = 2.0
HOOK_OVERLAY_FADE = 0.35
# Scene 0's still image: a visible continuous push-in rather than the slow
# 0.0006/frame Ken Burns, so the first frames already move.
HOOK_SCENE_ZOOM = 0.3

# Slight contrast/saturation lift on real photo/video stock footage (not the
# flat-colored diagram PNGs, where boosting saturation would just wash out
# text/box contrast) -- stock footage straight off Pexels tends to look flat
# compared to professionally color-graded finance-channel content, and this
# is a free one-line ffmpeg filter rather than a new asset/dependency.
COLOR_GRADE_FILTER = "eq=contrast=1.08:saturation=1.18:brightness=0.01"

# Shorts retention falls off when a single visual holds the screen too long --
# a scene whose narration runs longer than this gets a hard cut partway
# through to a different excerpt of the same downloaded Pexels clip instead
# of one continuous shot, without needing to fetch/cache a second visual.
VIDEO_SCENE_CUT_THRESHOLD = 7.0

# YouTube re-normalizes playback loudness to roughly -14 LUFS anyway, but it
# does so by turning the whole track down to match the loudest video on the
# platform -- a narration track recorded quieter than that (which varies scene
# to scene with Google TTS depending on voice/punctuation) just plays back
# quieter than louder channels instead of getting boosted. Normalizing to the
# same -14 LUFS target here means viewers don't have to reach for the volume
# dial between this channel's videos and everyone else's.
LOUDNORM_FILTER = "loudnorm=I=-14:TP=-1.5:LRA=11"


def _run(cmd: list[str]) -> None:
    subprocess.run(cmd, check=True, capture_output=True)


def _probe_duration(path: Path) -> float:
    result = subprocess.run(
        [FFPROBE_BIN, "-v", "error", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", str(path)],
        check=True, capture_output=True, text=True,
    )
    return float(result.stdout.strip())


def extract_frame(video_path: Path, out_path: Path, timestamp: float = None) -> Path:
    """Grabs a single still frame from a video clip, for use as the thumbnail/
    hook-card background when a scene's visual is a video rather than a photo.
    Defaults to the clip's midpoint rather than frame 0 -- stock-footage clips
    often open on a black or fade-in frame, which made the hook card blank
    whenever the first scene's visual happened to be a video (the common case
    for finance-channel videos, since Pexels usually has video results before
    falling back to photos)."""
    if timestamp is None:
        timestamp = _probe_duration(video_path) / 2
    _run([FFMPEG_BIN, "-y", "-ss", str(timestamp), "-i", str(video_path), "-frames:v", "1", str(out_path)])
    return out_path


def _build_scene_clip(visual_path: Path, kind: str, duration: float, out_path: Path, width: int, height: int,
                      punch_in: bool = False) -> None:
    """Render one scene's visual (image or video) to exactly `duration` seconds,
    scaled/cropped to the target resolution. punch_in: faster, continuous zoom
    on a still image (the opening scene)."""
    w, h = width, height
    scale_crop = f"scale={w}:{h}:force_original_aspect_ratio=increase,crop={w}:{h}"

    if kind == "image":
        # Slow pan/zoom (Ken Burns) over a still image.
        zoom_frames = int(duration * config.VIDEO_FPS)
        zoom = (f"1+{HOOK_SCENE_ZOOM}*on/{max(1, zoom_frames)}" if punch_in else "min(zoom+0.0006,1.15)")
        vf = (
            f"{scale_crop},"
            f"zoompan=z='{zoom}':d={zoom_frames}:"
            f"x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)':s={w}x{h}:fps={config.VIDEO_FPS},"
            f"{COLOR_GRADE_FILTER}"
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
        video_vf = f"{scale_crop},{COLOR_GRADE_FILTER}"
        source_duration = _probe_duration(visual_path)
        if source_duration >= duration and duration > VIDEO_SCENE_CUT_THRESHOLD and source_duration >= duration * 1.5:
            # Hard-cut this scene into two excerpts of the same source clip
            # instead of one continuous shot -- the second excerpt starts
            # partway through the remaining footage so it's visibly a
            # different moment, not just a seamless continuation.
            half = duration / 2
            second_start = min(source_duration - (duration - half), source_duration / 2)
            filter_complex = (
                f"[0:v]trim=start=0:duration={half},setpts=PTS-STARTPTS,{video_vf}[a];"
                f"[0:v]trim=start={second_start}:duration={duration - half},setpts=PTS-STARTPTS,{video_vf}[b];"
                f"[a][b]concat=n=2:v=1:a=0[outv]"
            )
            cmd = [
                FFMPEG_BIN, "-y", "-i", str(visual_path),
                "-filter_complex", filter_complex, "-map", "[outv]",
                "-t", str(duration), "-r", str(config.VIDEO_FPS),
                "-an", "-c:v", "libx264", "-preset", "ultrafast", "-threads", "1", *_VIDEO_BITRATE_ARGS,
                "-pix_fmt", "yuv420p", str(out_path),
            ]
        elif source_duration >= duration:
            cmd = [
                FFMPEG_BIN, "-y", "-i", str(visual_path),
                "-t", str(duration), "-vf", video_vf, "-r", str(config.VIDEO_FPS),
                "-an", "-c:v", "libx264", "-preset", "ultrafast", "-threads", "1", *_VIDEO_BITRATE_ARGS,
                "-pix_fmt", "yuv420p", str(out_path),
            ]
        else:
            loops = int(duration // source_duration) + 1
            cmd = [
                FFMPEG_BIN, "-y", "-stream_loop", str(loops), "-i", str(visual_path),
                "-t", str(duration), "-vf", video_vf, "-r", str(config.VIDEO_FPS),
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
        "-filter_complex",
        f"[1:v]scale={width}:{height},format=rgba,"
        f"fade=t=out:st={max(0.0, hook_duration - HOOK_OVERLAY_FADE)}:d={HOOK_OVERLAY_FADE}:alpha=1[card];"
        f"[0:v][card]overlay=0:0:format=auto",
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


def _overlay_watermark(video_path: Path, watermark_path: Path, out_path: Path) -> None:
    """Overlays a static watermark PNG for the whole duration of `video_path`
    in one pass over the already-concatenated video, rather than per-scene --
    cheaper and the watermark position never needs to vary by scene."""
    _run([
        FFMPEG_BIN, "-y", "-i", str(video_path), "-loop", "1", "-i", str(watermark_path),
        "-filter_complex", "[0:v][1:v]overlay=0:0:shortest=1",
        "-c:v", "libx264", "-preset", "ultrafast", "-threads", "1", *_VIDEO_BITRATE_ARGS,
        "-pix_fmt", "yuv420p", str(out_path),
    ])


def _pick_music_track(music_dir: Path) -> Path:
    """Picks a random track from `music_dir` so repeated videos don't all use
    the exact same loop. Returns None if the directory is missing/empty --
    callers must treat that as "skip the music bed", not an error, since the
    user may not have supplied any tracks yet even with MUSIC_BED_DIR set."""
    if not music_dir or not music_dir.is_dir():
        return None
    tracks = sorted(p for p in music_dir.iterdir() if p.suffix.lower() in (".mp3", ".wav", ".m4a"))
    if not tracks:
        return None
    return random.choice(tracks)


def _mix_background_music(narration_path: Path, music_path: Path, volume: float, out_path: Path) -> None:
    """Loops `music_path` for the full length of `narration_path` and mixes it
    in quietly underneath. `duration=first` on amix ends the mix exactly when
    the narration (first input) ends, regardless of how long the looped music
    track runs -- keeping the mixed audio the same length as the narration it
    replaces, which the rest of assemble_video's duration invariant depends on."""
    duration = _probe_duration(narration_path)
    _run([
        FFMPEG_BIN, "-y",
        "-i", str(narration_path),
        "-stream_loop", "-1", "-i", str(music_path),
        "-filter_complex",
        f"[1:a]volume={volume},atrim=0:{duration}[bg];"
        f"[0:a][bg]amix=inputs=2:duration=first:dropout_transition=0[aout]",
        "-map", "[aout]", "-t", str(duration),
        "-c:a", "aac", "-ar", "44100", "-ac", "2", str(out_path),
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
    of scene 0's own clip for the first `HOOK_OVERLAY_DURATION` seconds, to hook
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
        _build_scene_clip(visual_path, kind, duration, clip_path, width, height, punch_in=(i == 0))

        on_screen_text = (scene.get("on_screen_text") or "").strip()
        if on_screen_text:
            text_clip_path = work_dir / f"scene_{i:02d}_text.mp4"
            _overlay_text(clip_path, on_screen_text, width, height, text_clip_path, work_dir)
            clip_path = text_clip_path

        if i == 0 and hook_card_path:
            hook_clip_path = work_dir / f"scene_{i:02d}_hook.mp4"
            _overlay_hook_card(clip_path, hook_card_path, HOOK_OVERLAY_DURATION, width, height, hook_clip_path)
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

    if config.CHANNEL_WATERMARK_TEXT:
        watermark_path = work_dir / "watermark.png"
        textcard.render_watermark(config.CHANNEL_WATERMARK_TEXT, width, height, watermark_path)
        watermarked_path = work_dir / "video_watermarked.mp4"
        _overlay_watermark(video_only_path, watermark_path, watermarked_path)
        video_only_path = watermarked_path

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
        "-af", LOUDNORM_FILTER, "-c:a", "aac", "-ar", "44100", "-ac", "2", str(narration_concat_path),
    ])

    if config.MUSIC_BED_DIR:
        track = _pick_music_track(config.MUSIC_BED_DIR)
        if track:
            narration_with_music_path = work_dir / "narration_with_music.m4a"
            _mix_background_music(narration_concat_path, track, config.MUSIC_BED_VOLUME, narration_with_music_path)
            narration_concat_path = narration_with_music_path

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
