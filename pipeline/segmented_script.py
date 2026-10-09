"""Handles structured prompts that already lay out a timed scene-by-scene
script (e.g. "0-5 seconds - Hook", "5-15 seconds - Company overview", ...).

Rather than asking the local LLM to compress a whole multi-part brief into a
handful of scenes in one shot (which is where audio/visual/on-screen-text
correlation breaks down for a small model), each labelled section becomes its
own scene, generated with a narrow, single-purpose LLM call. This keeps the
narration, the visual search query, and the on-screen fact tightly bound to
the same source content, and keeps each scene's spoken duration aligned with
the section the user asked for.
"""
import json
import random
import re

from . import diagrams, llm, youtube_insights

HEADER_RE = re.compile(
    r"(\d+)\s*[-–—]\s*(\d+)\s*sec(?:ond)?s?\s*[-–—]\s*(.+)",
    re.IGNORECASE,
)
END_SCREEN_RE = re.compile(r"^\s*end\s*screen\s*:?\s*$", re.IGNORECASE)

QUOTE_RE = re.compile(r'(?:^|\n)[ \t]*[“”"]([^“”"]{15,})[“”"]')
HASHTAG_RE = re.compile(r"#\w+")


def clean_title(title: str) -> str:
    """Strips hashtags (they belong in tags/description, not the title) and
    collapses any double-spacing the model leaves behind after removing them."""
    title = HASHTAG_RE.sub("", title)
    return re.sub(r"\s+", " ", title).strip()

FACT_PATTERNS = [
    r"₹\s?[\d,]+(?:\.\d+)?\s*(?:crore|cr\b|lakh)",
    r"\d+(?:\.\d+)?\s?%\s*(?:YoY|QoQ|year[- ]on[- ]year)?",
    r"P/?E[:\s]*[\d.]+",
    r"ROE[:\s]*[\d.]+%?",
    r"ROCE[:\s]*[\d.]+%?",
    r"NSE:\s*\w+",
    r"BSE:\s*\d+",
]

QUALITATIVE_FALLBACKS = [
    (r"debt[- ]free", "Debt-Free Balance Sheet"),
    (r"export", "Global Export Exposure"),
    (r"cyclical", "Cyclical Industry Risk"),
]

VISUAL_FALLBACKS = [
    ("hook", ["industrial factory establishing shot", "city skyline aerial view", "busy trading floor"]),
    ("overview", ["steel factory workers", "office team meeting", "business professional presentation"]),
    ("revenue", ["factory production line", "cash counting machine", "warehouse logistics"]),
    ("financial", ["stock market chart growth", "stock ticker screen", "financial analyst at desk"]),
    ("balance sheet", ["financial documents calculator", "accountant reviewing ledger", "spreadsheet on laptop screen"]),
    ("valuation", ["financial documents calculator", "investor analyzing charts", "stock exchange board"]),
    ("growth", ["construction site mining equipment", "upward arrow graph animation", "factory expansion"]),
    ("risk", ["warning sign caution", "stormy stock market graph", "red downward chart"]),
    ("conclusion", ["investor thinking research", "handshake business deal", "person reading newspaper finance"]),
    ("disclaimer", ["warning text screen", "fine print document", "legal disclaimer text"]),
    ("end screen", ["warning text screen", "subscribe button animation", "fine print document"]),
]

SYSTEM_PROMPT_SEGMENT = """You write ONE scene of a YouTube Shorts video, based only on the
facts given below. Output ONLY valid JSON (no markdown fences, no commentary):

{"narration": "...", "visual_query": "...", "on_screen_text": "...", "diagram": null}

Rules:
- Use ONLY the facts given. Never invent, estimate, or add any number, date, or claim
  that is not present in the facts below.
- narration: natural, TTS-friendly spoken sentences (no headers, no markdown, no bullet
  points), summarizing the given facts in about {target_words} words. If the facts
  already contain a fully-written narration passage, reproduce it exactly, word for word.
- visual_query: 2-4 word English search term for stock video/photo footage that visually
  matches this scene's topic ("{label}"). Must describe a concrete, filmable subject.
  Leave this as a reasonable generic fallback (e.g. "student writing notebook") if you
  set "diagram" below instead.
- on_screen_text: the single most important number, name, or fact from the given
  content, copied exactly as given, under 8 words (e.g. "Revenue: Rs 423 Cr FY26").
  Empty string only if the content truly has no concrete fact to show.
- diagram: ONLY set this (instead of leaving it null) when this scene's concept is a
  number pattern, sequence, shape progression, or numeric comparison that has NO real
  filmable stock-footage equivalent. Use ONLY numbers/words that literally appear in
  the facts given above -- never invent new ones. Must be one of:
    {"kind": "number_line", "min": <int>, "max": <int>, "highlight": [<int>, ...]}
    {"kind": "triangular_dots", "rows": <int>, "label": "<short string>"}
    {"kind": "sequence_boxes", "items": ["<str>", ...], "highlight_index": <int or null>}
    {"kind": "bar_compare", "labels": ["<str>", ...], "values": [<number>, ...]}
    {"kind": "shape_progression", "shape": "<str>", "counts": [<int>, ...]}
  If none of these fit, or the scene has no literal facts to draw a diagram from
  (e.g. a Hook or Recap scene), leave diagram as null.
"""

def _segment_system_prompt(target_words: int, label: str) -> str:
    return SYSTEM_PROMPT_SEGMENT.replace("{target_words}", str(target_words)).replace("{label}", label)


def _extract_json(text: str) -> dict:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(json)?", "", text).rstrip("`").strip()
    if not text.startswith("{"):
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            text = match.group(0)
    return json.loads(text)


def parse_segments(prompt: str) -> list:
    lines = prompt.splitlines()
    headers = []
    for idx, line in enumerate(lines):
        m = HEADER_RE.search(line)
        if m:
            headers.append((idx, int(m.group(1)), int(m.group(2)), m.group(3).strip(" *:")))
            continue
        if END_SCREEN_RE.match(line):
            headers.append((idx, None, None, "End screen"))

    if len(headers) < 3:
        return []

    segments = []
    for i, (idx, start, end, label) in enumerate(headers):
        content_start = idx + 1
        content_end = headers[i + 1][0] if i + 1 < len(headers) else len(lines)
        content = "\n".join(lines[content_start:content_end]).strip()
        duration = (end - start) if (start is not None and end is not None) else None
        if content:
            segments.append({"label": label, "content": content, "duration": duration})

    known_total = sum(s["duration"] for s in segments if s["duration"])
    for s in segments:
        if s["duration"] is None:
            s["duration"] = max(2, 4)

    return segments


def _extract_verbatim(content: str) -> str:
    matches = QUOTE_RE.findall(content)
    if matches:
        return " ".join(m.strip() for m in matches)
    return ""


def _extract_fact(content: str) -> str:
    for pat in FACT_PATTERNS:
        m = re.search(pat, content, re.IGNORECASE)
        if m:
            return m.group(0).strip()
    for pat, label in QUALITATIVE_FALLBACKS:
        if re.search(pat, content, re.IGNORECASE):
            return label
    return ""


def _is_grounded(text: str, content: str) -> bool:
    """A number the model puts on screen must actually appear in this
    segment's own source content — otherwise it's fabricated, not extracted."""
    if "?" in text:
        return False
    digit_runs = re.findall(r"\d{2,}", text)
    if not digit_runs:
        return True
    return all(run in content for run in digit_runs)


def _fix_double_escaped_unicode(text: str) -> str:
    """Small local models occasionally double-escape unicode in JSON output
    (literal backslash-u-XXXX instead of the actual character) — decode it back."""
    return re.sub(r"\\u([0-9a-fA-F]{4})", lambda m: chr(int(m.group(1), 16)), text)


def _fallback_visual(label: str) -> str:
    label_lower = label.lower()
    for key, queries in VISUAL_FALLBACKS:
        if key in label_lower:
            return random.choice(queries)
    return "business professional presentation"


def _write_segment_scene(label: str, content: str, duration_sec: float, global_instructions: str = None) -> dict:
    target_words = max(6, int(duration_sec / 60 * 140))
    system_prompt = _segment_system_prompt(target_words, label)
    user_prompt = f"Scene topic: {label}\n\nFacts:\n{content}\n\nProduce the JSON now."
    if global_instructions:
        user_prompt += f"\n\nOverall tone/style for this video: {global_instructions}"

    verbatim = _extract_verbatim(content)
    fact_fallback = _extract_fact(content)
    visual_fallback = _fallback_visual(label)

    last_error = None
    for attempt in range(3):
        raw = llm.chat_json(system_prompt, user_prompt, max_tokens=512, timeout=60)
        try:
            scene = _extract_json(raw)
            if not (scene.get("narration") or "").strip():
                raise ValueError("empty narration")
            break
        except (ValueError, KeyError, json.JSONDecodeError) as e:
            last_error = e
            scene = None
            continue
    if scene is None:
        raise RuntimeError(
            f"Model failed to write the '{label}' scene 3 times in a row ({last_error})."
        )

    if verbatim:
        scene["narration"] = verbatim
    if not (scene.get("visual_query") or "").strip():
        scene["visual_query"] = visual_fallback

    on_screen_text = (scene.get("on_screen_text") or "").strip()
    if on_screen_text and not _is_grounded(on_screen_text, content):
        on_screen_text = ""
    if not on_screen_text and fact_fallback:
        on_screen_text = fact_fallback

    diagram = scene.get("diagram")
    if diagram:
        ok, _reason = diagrams.validate_diagram_spec(diagram, source_text=content)
        if not ok:
            diagram = None

    return {
        "narration": _fix_double_escaped_unicode(scene["narration"].strip()),
        "visual_query": _fix_double_escaped_unicode(scene["visual_query"].strip()),
        "on_screen_text": _fix_double_escaped_unicode(on_screen_text),
        "diagram": diagram,
    }


SUBSCRIBE_CTA_NARRATION = " If this helped, subscribe for more videos like it."
SUBSCRIBE_CTA_ON_SCREEN_TEXT = "SUBSCRIBE FOR MORE"


def merge_tags(required: list, generated: list, char_budget: int = 460) -> list:
    """Combines a channel/theme's must-have tags with the LLM's generated ones.
    `required` goes first so it survives YouTube's ~500-character total tag
    limit even if the LLM's own list is long; `generated` fills the rest.
    Dedupes case-insensitively."""
    merged, seen, total = [], set(), 0
    for tag in list(required) + list(generated):
        tag = tag.strip()
        key = tag.lower()
        if not tag or key in seen:
            continue
        cost = len(tag) + (2 if merged else 0)
        if total + cost > char_budget:
            continue
        seen.add(key)
        merged.append(tag)
        total += cost
    return merged


def append_subscribe_cta(scenes: list) -> None:
    """Appends a spoken + on-screen subscribe ask to the final scene. Most Shorts
    viewers watch muted at first, so the ask needs to be both heard and seen —
    real channel analytics showed near-zero subscriber conversion despite solid
    retention on some videos, and no video ever actually asked viewers to
    subscribe."""
    if not scenes:
        return
    last = scenes[-1]
    last["narration"] = last["narration"].rstrip() + SUBSCRIBE_CTA_NARRATION
    if not (last.get("on_screen_text") or "").strip():
        last["on_screen_text"] = SUBSCRIBE_CTA_ON_SCREEN_TEXT


SYSTEM_PROMPT_METADATA = """You write YouTube metadata for a short video, based only on its
narration below. Output ONLY valid JSON (no markdown fences, no commentary):

{"title": "...", "description": "...", "tags": ["...", "..."]}

Rules:
- title: punchy, clickable, under 70 characters. Must name the specific subject
  (company, person, place, event, topic) the narration is actually about — never
  a generic placeholder like "Stock Analysis" or "Interesting Facts". Use correct
  English grammar and punctuation, including apostrophes in contractions and
  possessives (e.g. "What's", "India's"). Never put hashtags in the title —
  put them in tags instead.
- description: 2-3 sentences summarizing what the video covers.
- tags: 8-15 relevant search tags.
- Base this ONLY on the narration given — do not invent facts not present there.
"""


def _write_metadata(scenes: list, style_guidance: str = "") -> dict:
    narration = "\n".join(scene["narration"] for scene in scenes)
    user_prompt = f"Narration:\n{narration}"
    if style_guidance:
        user_prompt += (
            "\n\nBelow is guidance on what's currently drawing viewers — based on this "
            "channel's own data and/or the wider niche — apply it to the title (and tags "
            f"where relevant), without contradicting the narration above:\n{style_guidance}"
        )
    user_prompt += "\n\nProduce the JSON metadata now."

    last_error = None
    for attempt in range(3):
        raw = llm.chat_json(SYSTEM_PROMPT_METADATA, user_prompt, max_tokens=512, timeout=60)
        try:
            data = _extract_json(raw)
            if not (data.get("title") or "").strip():
                raise ValueError("empty title")
            return {
                "title": clean_title(data["title"])[:70],
                "description": (data.get("description") or "").strip(),
                "tags": [t for t in (data.get("tags") or []) if t],
            }
        except (ValueError, KeyError, json.JSONDecodeError) as e:
            last_error = e
            continue
    raise RuntimeError(f"Model failed to write video metadata 3 times in a row ({last_error}).")


def write_segmented_script(prompt: str, segments: list, duration_sec: int = 60, extra_tags: list = None,
                            global_instructions: str = None) -> dict:
    scenes = []
    for seg in segments:
        scene = _write_segment_scene(seg["label"], seg["content"], seg["duration"], global_instructions)
        scenes.append(scene)

    style_guidance = youtube_insights.get_combined_guidance()
    meta = _write_metadata(scenes, style_guidance)
    tags = merge_tags(extra_tags or [], meta["tags"] or ["short video"])
    append_subscribe_cta(scenes)

    return {"title": meta["title"], "description": meta["description"], "tags": tags, "scenes": scenes}
