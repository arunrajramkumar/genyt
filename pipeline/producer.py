"""Shared logic: turn an already-written script dict into a finished video.

Used by both run.py (CLI/theme pipeline) and webapp.py (form-driven pipeline).
"""
import re
import shutil
import time
from pathlib import Path

from . import ai_thumbnail, config, diagrams, performance_log, tts, visuals, assemble, captions, textcard, youtube_upload


def slugify(title: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", title.lower()).strip("-")
    return slug[:60] or "video"


def _append_hashtags(description: str, tags: list, title: str = "", limit: int = 3) -> str:
    """YouTube highlights up to 3 hashtags found in a video's description above
    its title — appending the top tags there gets that for free on every upload.

    Prefers tags that actually appear in this video's own title over ones that
    don't: `tags` is required-channel-tags-first (see segmented_script.merge_tags),
    so without this, every video on a channel showed the same generic branding
    hashtags (e.g. "#NCERT #CBSE #Grade6") instead of the one thing that's
    actually specific to this video (e.g. the chapter topic or company name)."""
    title_lower = title.lower()
    specific, generic = [], []
    for tag in tags:
        cleaned = re.sub(r"[^A-Za-z0-9]", "", tag)
        if not cleaned:
            continue
        bucket = specific if tag.lower() in title_lower else generic
        bucket.append(f"#{cleaned}")

    hashtags = (specific + generic)[:limit]
    if not hashtags:
        return description
    return f"{description}\n\n{' '.join(hashtags)}"


def _extract_max_number(text: str) -> float:
    numbers = re.findall(r"\d[\d,]*\.?\d*", text)
    if not numbers:
        return 0.0
    return max(float(n.replace(",", "")) for n in numbers)


def _extract_stat_phrase(narration: str) -> str:
    """Pulls a short "NN% <word>" hook phrase out of a sentence, e.g. from
    "posted a 24.3% year-on-year revenue jump" -> "24.3% Revenue". Used as a
    fallback when no scene's on_screen_text carries a percentage (see
    _pick_hook_stat) so the thumbnail's stat callout isn't silently empty for
    a video whose only numeric hook lives in the narration itself."""
    match = re.search(
        r"(\d[\d,]*\.?\d*)\s*%\s*(?:year-on-year|quarter-on-quarter|yoy|qoq)?\s*([a-zA-Z]+)?",
        narration,
    )
    if not match:
        return ""
    number, word = match.group(1), match.group(2)
    return f"{number}% {word.capitalize()}" if word else f"{number}%"


def _pick_hook_stat(scenes: list) -> str:
    """Picks the on_screen_text to headline the thumbnail/hook card.

    Previously this just took the first scene (in narration order) whose
    on_screen_text contained any digit -- for a finance video that's often
    whichever stat happens to come first (e.g. a plain year like "FY23"),
    not the most attention-grabbing one. Instead, score every numeric
    candidate across all scenes and prefer a percentage figure (growth/
    return rates read as the most compelling stat for a thumbnail) and,
    among those, the largest number -- falling back to the largest plain
    number when no candidate has a percentage.
    """
    candidates = [(scene.get("on_screen_text") or "").strip() for scene in scenes]
    candidates = [c for c in candidates if any(ch.isdigit() for ch in c)]
    if candidates:
        return max(candidates, key=lambda text: ("%" in text, _extract_max_number(text)))
    for scene in scenes:
        stat = _extract_stat_phrase(scene.get("narration") or "")
        if stat:
            return stat
    return ""


_CALM_HOOK_STYLE_KEYWORDS = (
    "sip", "crore", "lakh", "wealth", "retire", "retirement", "compound",
    "compounding", "mutual fund", "pension", "financial freedom", "crorepati",
    "nest egg", "long-term invest", "long term invest",
)


def _pick_hook_style(title: str) -> str:
    """Picks the hook-card/thumbnail visual treatment from the video's title.

    The default "alert" style (red badge, angled banner, glow-green stat) is
    built for breaking-news/earnings-surprise urgency -- it reads as tonally
    wrong for patient, long-horizon topics like SIP/wealth-building, which a
    real AI-generated reference thumbnail for that content type instead used
    a calm gold-on-dark-green treatment with no urgency badge at all. There's
    no structured content-type field on a theme/script (see produce_from_script),
    so this infers style from a keyword match on the title rather than adding
    one -- cheap, zero-config, and the title is written by the same prompt
    that would otherwise have to carry a separate explicit field anyway.
    """
    title_lower = title.lower()
    if any(keyword in title_lower for keyword in _CALM_HOOK_STYLE_KEYWORDS):
        return "calm"
    return "alert"


def _pick_hook_background(visual_paths: list, work_dir, on_progress=print):
    """Picks a still image to sit behind the hook card/thumbnail text.

    Previously this only used the first scene's visual, and only if it was
    already a still image or diagram -- a video clip (the common case for
    stock-footage queries, since Pexels usually returns videos before photos)
    was silently skipped, leaving the thumbnail with no background at all.
    Grabbing a frame from the clip instead means the thumbnail almost always
    has a real visual behind it."""
    if not visual_paths:
        return None
    path, kind = visual_paths[0]
    if kind in ("image", "diagram"):
        return path
    if kind == "video":
        try:
            return assemble.extract_frame(path, work_dir / "hook_bg.jpg")
        except Exception as e:
            on_progress(f"  thumbnail background frame extraction failed ({e}), using plain background")
    return None


def produce_from_script(script: dict, orientation: dict = None, on_progress=print, voice: str = None,
                         thumbnail_badge: str = None, category_id: str = None) -> dict:
    """orientation: config.LANDSCAPE or config.SHORTS (defaults to landscape).
    voice: edge-tts voice ID overriding config.TTS_VOICE for this video (e.g. a
    per-chat language preference) — defaults to the configured voice.
    thumbnail_badge: short corner label stamped on the thumbnail/hook card,
    e.g. "CBSE | Grade 6 Maths" for a syllabus-specific channel.
    category_id: YouTube category ID for this video (defaults to "22", People
    & Blogs) -- lets a channel-specific queue builder pick a more accurate
    category (e.g. "27" Education, "25" News & Politics) instead of every
    video on every channel landing in the same generic bucket.
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
    total_narration_chars = sum(len(scene["narration"]) for scene in script["scenes"])
    tts.ensure_google_tts_budget(total_narration_chars)
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

    # Generated before the per-scene fetch loop so a successful result can
    # also become the hook SCENE's own in-video visual (not just the YouTube
    # thumbnail) -- a unique, company-specific image no competitor's generic
    # stock footage can match, on the single most-watched/highest-leverage
    # scene of the video. Falls back to Pexels for that scene like any other
    # when generation fails, same as before this was reused this way.
    ai_hook_bg_path = ai_thumbnail.generate_hook_background(
        script.get("company_name"), work_dir / "ai_hook_bg.png", on_progress,
        width=orientation["width"], height=orientation["height"],
    )

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
        if i == 0 and ai_hook_bg_path is not None:
            visual_paths.append((ai_hook_bg_path, "image"))
            continue
        visual_path, kind = visuals.fetch_visual(
            scene.get("visual_query", "").strip() or "abstract background",
            orientation=orientation["orientation"],
            target_width=orientation["width"],
            exclude=used_visuals,
        )
        used_visuals.add(visual_path)
        visual_paths.append((visual_path, kind))

    on_progress("Rendering hook card / thumbnail")
    stat_text = _pick_hook_stat(script["scenes"])
    hook_bg_path = ai_hook_bg_path
    if hook_bg_path is None:
        hook_bg_path = _pick_hook_background(visual_paths, work_dir, on_progress)
    thumb_path = config.OUTPUT_DIR / f"{slug}.thumbnail.png"
    # The title already gets the company name prepended when missing (see
    # segmented_script._ensure_company_name_in_title), but long titles wrap
    # and the company name can end up on a line that's cut off visually in a
    # small feed thumbnail -- the corner badge is a second, always-intact
    # placement for it, same slot a channel-specific badge would otherwise use.
    badge_text = thumbnail_badge or script.get("company_name")
    textcard.render_hook_card(
        script["title"], stat_text, orientation["width"], orientation["height"],
        thumb_path, background_path=hook_bg_path, badge_text=badge_text,
        style=_pick_hook_style(script["title"]),
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

    description = _append_hashtags(script["description"], script["tags"], script["title"])
    if orientation["orientation"] == "portrait" and "#shorts" not in description.lower():
        # Nearly every video this pipeline produces is a Shorts-eligible
        # vertical clip, but nothing ever told YouTube that explicitly --
        # #Shorts in the description is the standard signal that gets a video
        # considered for the dedicated Shorts feed/shelf.
        description = f"{description}\n#Shorts"

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
            language = (voice or config.TTS_VOICE).split("-")[0]
            upload = youtube_upload.upload_video(
                out_path, script["title"], description, script["tags"],
                category_id=category_id or "22",
                captions_path=srt_path,
                captions_language=language,
                default_language=language,
            )
            youtube_url = upload["url"]
            on_progress(f"Uploaded: {youtube_url}")
            if script.get("symbol"):
                performance_log.record(upload["video_id"], script["symbol"], script.get("company_name"))
            try:
                youtube_upload.set_thumbnail(upload["video_id"], thumb_path)
                on_progress(f"Custom thumbnail set for {upload['video_id']}")
            except Exception as e:
                on_progress(f"Thumbnail upload failed (video still published): {e}")
            if config.YOUTUBE_PLAYLIST_ID:
                try:
                    youtube_upload.add_to_playlist(upload["video_id"], config.YOUTUBE_PLAYLIST_ID)
                except Exception as e:
                    on_progress(f"Playlist add failed (video still published): {e}")
        except youtube_upload.NotConfiguredError:
            on_progress("YouTube upload skipped (not configured — see README)")
        except Exception as e:
            on_progress(f"YouTube upload failed: {e}")

    on_progress(f"Done: {out_path}")
    return {
        "video": out_path, "srt": srt_path, "metadata": meta_path, "thumbnail": thumb_path,
        "slug": slug, "youtube_url": youtube_url,
    }
