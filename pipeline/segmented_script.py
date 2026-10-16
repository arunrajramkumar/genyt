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

# build_finance_queue.py always writes the subject into the theme text as
# "Company Name (NSE: SYMBOL)" -- a reliable anchor to pull the exact company
# name from, so the title can be forced to include it even if the metadata
# LLM call drifts onto a vaguer phrasing (e.g. "This Multibagger Stock").
NSE_COMPANY_RE = re.compile(r"([A-Za-z0-9&.'\-, ]+?)\s*\(NSE:\s*([A-Z0-9&\-]+)\)")


def _extract_company_name(prompt: str):
    m = NSE_COMPANY_RE.search(prompt)
    return m.group(1).strip() if m else None


def _extract_company_symbol(prompt: str):
    """Pulls the NSE ticker out of the same "(NSE: SYMBOL)" anchor used by
    _extract_company_name -- this is the one reliable, non-fuzzy join key back
    to the exact company a finance video was about, used by the performance
    log (see pipeline.performance_log) to track which company's video actually
    got uploaded, keyed off the real YouTube video_id rather than re-parsing
    the LLM-generated title."""
    m = NSE_COMPANY_RE.search(prompt)
    return m.group(2).strip().upper() if m else None


# Indian retail-investor audiences read rupee amounts in lakh/crore, never
# million/billion -- but grounding facts come from Tavily web-search snippets
# (international finance sites) and the LLM just echoes whatever unit the
# source text used. Reported bug: narration said "a market capitalization of
# ₹11.12 billion" and the on-screen text showed only "₹11.12" -- the "billion"
# got silently dropped during fact-extraction, making the number read as
# roughly 100 crore too small. Converting the unit (not inventing a new
# number) fixes this at the root: once narration itself says "₹1,112 crore",
# both the on-screen extraction (FACT_PATTERNS' crore/lakh pattern) and the
# narration/captions stay correct and in sync.
#
# Covers spelled-out units plus the abbreviations ("M"/"Mn", "B"/"Bn", "K")
# and "trillion" that international finance snippets also commonly use --
# any of these left unconverted falls through to FACT_PATTERNS' plain-price
# pattern, which silently drops the unit exactly like the original bug.
_CURRENCY_UNIT_RE = re.compile(
    r"₹\s?([\d,]+(?:\.\d+)?)\s*(trillion|billion|million|thousand|bn\b|mn\b|[bmk]\b)",
    re.IGNORECASE,
)
_CRORE_PER_UNIT = {
    "trillion": 100_000, "billion": 100, "million": 0.1, "thousand": 0.0001,
    "bn": 100, "mn": 0.1, "b": 100, "m": 0.1, "k": 0.0001,
}


def _format_indian_number(value: float) -> str:
    formatted = f"{value:,.2f}"
    if formatted.endswith(".00"):
        return formatted[:-3]
    return formatted.rstrip("0").rstrip(".")


def _convert_currency_unit(match: re.Match) -> str:
    number = float(match.group(1).replace(",", ""))
    unit = match.group(2).lower()
    crore_value = number * _CRORE_PER_UNIT[unit]
    if crore_value >= 100_000:  # >= 1 lakh crore -- the standard way large caps are quoted
        return f"₹{_format_indian_number(crore_value / 100_000)} lakh crore"
    if crore_value >= 1:
        return f"₹{_format_indian_number(crore_value)} crore"
    lakh_value = crore_value * 100
    if lakh_value >= 1:
        return f"₹{_format_indian_number(lakh_value)} lakh"
    return f"₹{_format_indian_number(lakh_value * 100_000)}"  # sub-lakh: plain rupees reads clearer than "0.x lakh"


def normalize_indian_currency_units(text: str) -> str:
    """Rewrites "₹<N> million/billion/thousand/trillion" (and common
    abbreviations like M/Mn/B/Bn/K) into the lakh/crore convention Indian
    audiences actually use. Only touches amounts with a literal ₹ prefix, so
    a USD ($) figure is never misinterpreted as rupees."""
    return _CURRENCY_UNIT_RE.sub(_convert_currency_unit, text)


def clean_title(title: str) -> str:
    """Strips hashtags (they belong in tags/description, not the title) and
    collapses any double-spacing the model leaves behind after removing them."""
    title = HASHTAG_RE.sub("", title)
    return re.sub(r"\s+", " ", title).strip()

FACT_PATTERNS = [
    # "lakh crore" must be matched as a single unit before the plain
    # "lakh"/"crore" pattern below -- otherwise large-cap market caps (e.g.
    # "₹15 lakh crore", the standard way Indian sources quote values over
    # ₹1 trillion) get truncated to "₹15 lakh", a 100,000x magnitude error.
    r"₹\s?[\d,]+(?:\.\d+)?\s*lakh\s*crore",
    r"₹\s?[\d,]+(?:\.\d+)?\s*(?:crore|cr\b|lakh)",
    # Allow a short run of words between the label and its number (e.g. "P/E
    # ratio stands at 20.09", "Price-to-Book at 5.51") -- real narration/source
    # content rarely puts the number immediately after the label.
    r"(?:P/?E|Price[- ]to[- ]Earnings)\b.{0,25}?[\d.]+",
    r"(?:P/?B|Price[- ]to[- ]Book)\b.{0,25}?[\d.]+",
    r"EV/?EBITDA\b.{0,25}?[\d.]+",
    r"ROE\b.{0,20}?[\d.]+%?",
    r"ROCE\b.{0,20}?[\d.]+%?",
    # Leading "-?" so a decline (e.g. fundamentals.py's "grew -12.3% YoY") is
    # never displayed as a positive figure -- same unit/sign-fidelity goal as
    # normalize_indian_currency_units above.
    r"-?\d+(?:\.\d+)?\s?%\s*(?:YoY|QoQ|year[- ]on[- ]year)?",
    r"NSE:\s*\w+",
    r"BSE:\s*\d+",
    r"₹\s?[\d,]+(?:\.\d+)?",  # plain price (e.g. "₹887.80"), lowest priority
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
- The facts may include open "Figure it Out" / exercise-style questions (e.g. "Which
  sequence do you get when you add the All 1's sequence up and down?") that are NOT
  answered anywhere in the facts given. Never invent a resolution to these -- if the
  facts pose a question without also stating its answer, narrate it AS a question for
  the viewer to think about, do not assert a specific answer.
- narration: natural, TTS-friendly spoken sentences (no headers, no markdown, no bullet
  points), summarizing the given facts in about {target_words} words. If the facts
  already contain a fully-written narration passage, reproduce it exactly, word for word.
- visual_query: 2-4 word English search term for stock video/photo footage that visually
  matches this scene's topic ("{label}"). Must describe a concrete, filmable subject.
  If the facts name a specific company/industry, prefer a term tied to that company's
  actual business or sector (e.g. "gold jewellery workshop" for a jewellery company,
  "cement factory kiln" for a cement company, "data center server racks" for an IT
  company) over generic finance stock imagery ("stock market chart", "money growth",
  "businessman handshake", "calculator documents") -- those generic terms are what
  every finance channel's footage already looks like, so only fall back to them when
  the facts genuinely give no sector/subject detail to search for instead.
  Leave this as a reasonable generic fallback (e.g. "student writing notebook") if you
  set "diagram" below instead.
- on_screen_text: the single most important number, name, or fact from the given
  content, copied exactly as given, under 8 words (e.g. "Revenue: Rs 423 Cr FY26").
  Empty string only if the content truly has no concrete fact to show.
- diagram: prefer setting this whenever this scene's concept is a number pattern,
  sequence, shape progression, or numeric comparison -- even when a filmable stock
  visual also exists for the topic. A diagram built from this company's own real
  numbers (e.g. its actual revenue trend) is unique to this video; generic finance
  stock footage of a "stock chart" or "office building" looks the same on every
  finance channel. Use ONLY numbers/words that literally appear in the facts given
  above -- never invent new ones. Must be one of:
    {"kind": "number_line", "min": <int>, "max": <int>, "highlight": [<int>, ...]}
    {"kind": "triangular_dots", "rows": <int>, "label": "<short string>"}
    {"kind": "sequence_boxes", "items": ["<str>", ...], "highlight_index": <int or null>}
    {"kind": "bar_compare", "labels": ["<str>", ...], "values": [<number>, ...], "unit": "<str>"}
    {"kind": "shape_progression", "shape": "<str>", "counts": [<int>, ...]}
    {"kind": "trend_line", "labels": ["<str>", ...], "values": [<number>, ...], "unit": "<str>"}
    {"kind": "donut_share", "labels": ["<str>", ...], "values": [<number>, ...], "unit": "<str>"}
  Prefer trend_line over bar_compare for a value changing across an ordered
  sequence (e.g. revenue/profit by year) -- reserve bar_compare for comparing
  a handful of unordered items (e.g. this company's P/E vs. a peer's). Use
  donut_share only when the values are real parts of a single whole (e.g.
  revenue split by segment, holding composition) -- never for an unrelated
  comparison, which belongs in bar_compare instead.
  For bar_compare/trend_line/donut_share, "unit" must state what the values
  are in (e.g. "Cr", "%", "x") exactly as the facts state it -- a number
  shown on a diagram with no unit is as misleading as one with the wrong
  unit. Use "" only if the values are genuinely unitless (e.g. a count) --
  for donut_share, leave "unit" as "" when the values are plain percentages.
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

    # Headerless sections (e.g. "End screen:") have no explicit time range, so
    # their spoken duration is estimated from their own word count at the same
    # ~140 words/minute pacing convention _write_segment_scene uses to size
    # narration -- a fixed 4s fallback either clipped longer disclaimers mid-
    # sentence or left short ones sitting in dead air.
    for s in segments:
        if s["duration"] is None:
            word_count = len(s["content"].split())
            s["duration"] = max(2, round(word_count / 140 * 60))

    return segments


def _extract_verbatim(content: str) -> str:
    matches = QUOTE_RE.findall(content)
    if matches:
        return " ".join(m.strip() for m in matches)
    return ""


def _extract_facts(content: str, limit: int = 3) -> list:
    """Collects every distinct metric matching FACT_PATTERNS in this segment's
    source content, not just the first -- a scene whose narration covers
    several numbers (e.g. "P/E 20.09, Price-to-Book 5.51, EV/EBITDA 13.48,
    ROE 30.2%") previously only ever got ONE of them picked as on-screen text,
    so a viewer watching muted never saw the others even though they were
    spoken. Longer/more-specific matches (e.g. "ROE 30.2%") win over shorter
    ones they fully contain (e.g. a bare "30.2%" from the generic % pattern)."""
    spans = []
    for pat in FACT_PATTERNS:
        for m in re.finditer(pat, content, re.IGNORECASE):
            raw = m.group(0)
            text = raw.strip()
            if not text:
                continue
            start = m.start() + raw.index(text)
            spans.append((start, start + len(text), text))
    spans.sort(key=lambda s: (s[0], s[0] - s[1]))  # earliest first; longer match wins at same start

    facts, covered = [], []
    for start, end, text in spans:
        if any(start >= cs and end <= ce for cs, ce in covered):
            continue
        if text in facts:
            continue
        facts.append(text)
        covered.append((start, end))
        if len(facts) >= limit:
            break
    if facts:
        return facts
    for pat, label in QUALITATIVE_FALLBACKS:
        if re.search(pat, content, re.IGNORECASE):
            return [label]
    return []


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
    content = normalize_indian_currency_units(content)
    target_words = max(6, int(duration_sec / 60 * 140))
    system_prompt = _segment_system_prompt(target_words, label)
    user_prompt = f"Scene topic: {label}\n\nFacts:\n{content}\n\nProduce the JSON now."
    if global_instructions:
        user_prompt += f"\n\nOverall tone/style for this video: {global_instructions}"

    verbatim = _extract_verbatim(content)
    facts = _extract_facts(content)
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
    scene["narration"] = normalize_indian_currency_units(scene["narration"])
    if not (scene.get("visual_query") or "").strip():
        scene["visual_query"] = visual_fallback

    on_screen_text = normalize_indian_currency_units((scene.get("on_screen_text") or "").strip())
    if on_screen_text and not _is_grounded(on_screen_text, content):
        on_screen_text = ""
    if len(facts) > 1:
        # This segment's content names several metrics -- show all of them
        # rather than trusting the model to have picked a representative one.
        on_screen_text = " | ".join(facts)
    elif not on_screen_text and facts:
        on_screen_text = facts[0]

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
- If the subject is a company/stock, never phrase the title as a verdict or
  recommendation (e.g. avoid "Buy or Sell", "Apply or Avoid", "Cheap or
  Overvalued") -- this channel stays neutral, per the narration's own stance.
  Prefer a data-driven, curiosity-led framing instead (e.g. "<Company>: 3
  Numbers Investors Must Know", "<Company> at ₹X: What the Numbers Show").
- description: 2-3 sentences summarizing what the video covers.
- tags: 8-15 relevant search tags.
- Base this ONLY on the narration given — do not invent facts not present there.
"""


def _ensure_company_name_in_title(title: str, company_name: str) -> str:
    """The metadata prompt already instructs the model to name the specific
    subject, but a small model can still drift onto a vaguer hook phrase (e.g.
    "This Multibagger Stock Nobody's Talking About") that reads well but loses
    the one thing a stock-analysis title/thumbnail must have for someone
    scanning a feed: which company it's actually about. Force it in
    deterministically rather than trusting the sample to always include it."""
    if not company_name or company_name.lower() in title.lower():
        return title
    return f"{company_name}: {title}"


def _truncate_title(title: str, limit: int = 70) -> str:
    """A hard `title[:limit]` cut mid-word (observed in production: "...Is It
    a Long-Term Wea" instead of "...Wealth Builder?") reads as broken on
    YouTube -- back off to the last word boundary within the limit instead."""
    if len(title) <= limit:
        return title
    truncated = title[:limit]
    cut = truncated.rfind(" ")
    return (truncated[:cut] if cut > 0 else truncated).rstrip(" :-–—")


def _write_metadata(scenes: list, style_guidance: str = "", company_name: str = None) -> dict:
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
            title = _truncate_title(_ensure_company_name_in_title(clean_title(data["title"]), company_name))
            return {
                "title": title,
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
    company_name = _extract_company_name(prompt)
    meta = _write_metadata(scenes, style_guidance, company_name=company_name)
    tags = merge_tags(extra_tags or [], meta["tags"] or ["short video"])
    append_subscribe_cta(scenes)

    return {
        "title": meta["title"],
        "description": meta["description"],
        "tags": tags,
        "scenes": scenes,
        "company_name": company_name,
        "symbol": _extract_company_symbol(prompt),
    }
