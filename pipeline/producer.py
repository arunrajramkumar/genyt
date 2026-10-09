"""Shared logic: turn an already-written script dict into a finished video.

Used by both run.py (CLI/theme pipeline) and webapp.py (form-driven pipeline).
"""
import re
import shutil
import time
from pathlib import Path

from . import config, diagrams, tts, visuals, assemble, captions, textcard, youtube_upload


def slugify(title: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", title.lower()).strip("-")
    return slug[:60] or "video"


def _append_hashtags(description: str, tags: list, limit: int = 3) -> str:
    """YouTube highlights up to 3 hashtags found in a video's description above
    its title — appending the top tags there gets that for free on every upload."""
    hashtags = []
    for tag in tags:
        cleaned = re.sub(r"[^A-Za-z0-9]", "", tag)
        if cleaned:
            hashtags.append(f"#{cleaned}")
        if len(hashtags) == limit:
            break
    if not hashtags:
        return description
    return f"{description}\n\n{' '.join(hashtags)}"


def produce_from_script(script: dict, orientation: dict = None, on_progress=print, voice: str = None,
                         thumbnail_badge: str = None) -> dict:
    """orientation: config.LANDSCAPE or config.SHORTS (defaults to landscape).
    voice: edge-tts voice ID overriding config.TTS_VOICE for this video (e.g. a
    per-chat language preference) — defaults to the configured voice.
    thumbnail_badge: short corner label stamped on the thumbnail/hook card,
    e.g. "CBSE | Grade 6 Maths" for a syllabus-specific channel.
    Returns {"video": Path, "srt": Path, "metadata": Path, "slug": str}.
    """
    orientation = orientation or config.LANDSCAPE
    slug = slugify(script["title"])
    work_dir = config.CACHE_DIR / slug
    if work_dir.exists():
        shutil.rmtree(work_dir)
    work_dir.mkdir(parents=True)

    on_progress(f"Title: {script['title']} ({len(script['scenes'])} scenes)")

    if script["scenes"] and not (script["scenes"][0].get("on_screen_text") or "").strip():
        script["scenes"][0]["on_screen_text"] = script["title"][:40]

    on_progress("Synthesizing narration")
    narration_paths = []
    for i, scene in enumerate(script["scenes"]):
        if i > 0:
            # edge-tts's unofficial endpoint applies a short burst-rate limit on
            # cloud/datacenter IPs (GitHub Actions) -- firing scenes back-to-back
            # with no gap reliably tripped it after ~5-6 calls. A small gap keeps
            # us under it without meaningfully slowing down generation.
            time.sleep(1.5)
        wav_path = work_dir / f"narration_{i:02d}.mp3"
        on_progress(f"  scene {i}: {scene['narration']!r}")
        duration = tts.synthesize(scene["narration"], wav_path, voice=voice)
        scene["_duration"] = duration
        narration_paths.append(wav_path)

    on_progress("Fetching stock visuals")
    visual_paths = []
    used_visuals = set()
    for i, scene in enumerate(script["scenes"]):
        diagram_spec = scene.get("diagram")
        if diagram_spec:
            try:
                diagram_path = work_dir / f"diagram_{i:02d}.png"
                diagrams.render_diagram(diagram_spec, orientation["width"], orientation["height"], diagram_path)
                visual_paths.append((diagram_path, "diagram"))
                continue
            except Exception as e:
                on_progress(f"  scene {i}: diagram render failed ({e}), falling back to stock")
        visual_path, kind = visuals.fetch_visual(
            scene.get("visual_query", "").strip() or "abstract background",
            orientation=orientation["orientation"],
            target_width=orientation["width"],
            exclude=used_visuals,
        )
        used_visuals.add(visual_path)
        visual_paths.append((visual_path, kind))

    on_progress("Rendering hook card / thumbnail")
    stat_text = next(
        (
            (scene.get("on_screen_text") or "").strip()
            for scene in script["scenes"]
            if any(ch.isdigit() for ch in (scene.get("on_screen_text") or ""))
        ),
        "",
    )
    hook_bg_path = visual_paths[0][0] if visual_paths and visual_paths[0][1] in ("image", "diagram") else None
    thumb_path = config.OUTPUT_DIR / f"{slug}.thumbnail.png"
    textcard.render_hook_card(
        script["title"], stat_text, orientation["width"], orientation["height"],
        thumb_path, background_path=hook_bg_path, badge_text=thumbnail_badge,
    )

    on_progress("Building captions")
    t = 0.0
    timed_scenes = []
    for scene in script["scenes"]:
        timed_scenes.append({"narration": scene["narration"], "start": t, "end": t + scene["_duration"]})
        t += scene["_duration"]
    srt_path = captions.write_srt(timed_scenes, config.OUTPUT_DIR / f"{slug}.srt")

    on_progress("Assembling final video with ffmpeg")
    out_path = config.OUTPUT_DIR / f"{slug}.mp4"
    assemble.assemble_video(
        script["scenes"], narration_paths, visual_paths, out_path, work_dir,
        width=orientation["width"], height=orientation["height"],
        hook_card_path=thumb_path,
    )

    description = _append_hashtags(script["description"], script["tags"])

    meta_path = config.OUTPUT_DIR / f"{slug}.metadata.txt"
    meta_path.write_text(
        f"TITLE:\n{script['title']}\n\n"
        f"DESCRIPTION:\n{description}\n\n"
        f"TAGS:\n{', '.join(script['tags'])}\n",
        encoding="utf-8",
    )

    youtube_url = None
    if config.YOUTUBE_AUTO_UPLOAD:
        try:
            on_progress("Uploading to YouTube")
            upload = youtube_upload.upload_video(
                out_path, script["title"], description, script["tags"],
                captions_path=srt_path,
                captions_language=(voice or config.TTS_VOICE).split("-")[0],
            )
            youtube_url = upload["url"]
            on_progress(f"Uploaded: {youtube_url}")
            try:
                youtube_upload.set_thumbnail(upload["video_id"], thumb_path)
            except Exception as e:
                on_progress(f"Thumbnail upload failed (video still published): {e}")
        except youtube_upload.NotConfiguredError:
            on_progress("YouTube upload skipped (not configured — see README)")
        except Exception as e:
            on_progress(f"YouTube upload failed: {e}")

    on_progress(f"Done: {out_path}")
    return {
        "video": out_path, "srt": srt_path, "metadata": meta_path, "thumbnail": thumb_path,
        "slug": slug, "youtube_url": youtube_url,
    }
