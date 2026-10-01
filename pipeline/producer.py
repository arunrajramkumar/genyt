"""Shared logic: turn an already-written script dict into a finished video.

Used by both run.py (CLI/theme pipeline) and webapp.py (form-driven pipeline).
"""
import re
import shutil
from pathlib import Path

from . import config, tts, visuals, assemble, captions, textcard


def slugify(title: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", title.lower()).strip("-")
    return slug[:60] or "video"


def produce_from_script(script: dict, orientation: dict = None, on_progress=print, voice: str = None) -> dict:
    """orientation: config.LANDSCAPE or config.SHORTS (defaults to landscape).
    voice: edge-tts voice ID overriding config.TTS_VOICE for this video (e.g. a
    per-chat language preference) — defaults to the configured voice.
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
        wav_path = work_dir / f"narration_{i:02d}.mp3"
        duration = tts.synthesize(scene["narration"], wav_path, voice=voice)
        scene["_duration"] = duration
        narration_paths.append(wav_path)

    on_progress("Fetching stock visuals")
    visual_paths = [
        visuals.fetch_visual(
            scene.get("visual_query", "").strip() or "abstract background",
            orientation=orientation["orientation"],
            target_width=orientation["width"],
        )
        for scene in script["scenes"]
    ]

    on_progress("Rendering hook card / thumbnail")
    stat_text = next(
        (
            (scene.get("on_screen_text") or "").strip()
            for scene in script["scenes"]
            if any(ch.isdigit() for ch in (scene.get("on_screen_text") or ""))
        ),
        "",
    )
    hook_bg_path = visual_paths[0][0] if visual_paths and visual_paths[0][1] == "image" else None
    thumb_path = config.OUTPUT_DIR / f"{slug}.thumbnail.png"
    textcard.render_hook_card(
        script["title"], stat_text, orientation["width"], orientation["height"],
        thumb_path, background_path=hook_bg_path,
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

    meta_path = config.OUTPUT_DIR / f"{slug}.metadata.txt"
    meta_path.write_text(
        f"TITLE:\n{script['title']}\n\n"
        f"DESCRIPTION:\n{script['description']}\n\n"
        f"TAGS:\n{', '.join(script['tags'])}\n",
        encoding="utf-8",
    )

    on_progress(f"Done: {out_path}")
    return {"video": out_path, "srt": srt_path, "metadata": meta_path, "thumbnail": thumb_path, "slug": slug}
