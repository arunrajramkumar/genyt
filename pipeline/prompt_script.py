"""Turns a single free-form user prompt into a structured video script via a
hosted LLM (Groq). Generic — works for any topic (stock analysis, history,
science, etc.), not just finance.

If the prompt contains specific facts/figures (e.g. revenue numbers), the model
is instructed to use only those and never invent additional specific claims.
For open-ended topics without hard facts, it can write normally/engagingly.
"""
import json
import re

from . import diagrams, llm, research, segmented_script, youtube_insights

SYSTEM_PROMPT = """You write scripts for short-form YouTube videos (including Shorts) on
whatever topic the user gives you. Output ONLY valid JSON (no markdown fences, no
commentary) matching this schema:

{
  "title": "punchy, clickable YouTube title, under 70 characters",
  "description": "2-3 sentence YouTube description",
  "tags": ["8-15 relevant search tags"],
  "scenes": [
    {
      "narration": "1-3 sentences of spoken narration for this scene",
      "visual_query": "2-4 word English search query for stock footage/photos that visually matches this scene",
      "on_screen_text": "SHORT text overlay for this scene, under 6 words, or empty string if none needed",
      "diagram": null
    }
  ]
}

Rules:
- The title must use correct English grammar and punctuation, including
  apostrophes in contractions and possessives (e.g. "What's", "India's").
  Never put hashtags in the title — put them in tags instead.
- If the user's prompt includes specific facts, figures, names, or claims, use ONLY
  those — never invent, estimate, or add extra numbers, dates, or claims that were
  not given to you. If the prompt is an analysis of something (e.g. a stock, a
  product, a place), stay neutral and do not add a recommendation/verdict unless
  the user explicitly asked for one. This applies to the title too -- never phrase
  it as a verdict (e.g. avoid "Buy or Sell", "Apply or Avoid", "Cheap or
  Overvalued"); prefer a data-driven, curiosity-led framing instead (e.g.
  "<Subject>: 3 Numbers to Know", "<Subject> at ₹X: What the Numbers Show").
- If the prompt poses an open question (e.g. an exercise/"figure it out" style
  question) without also stating its answer, never invent a resolution — narrate it
  as a question for the viewer to think about instead of asserting a specific answer.
- If the user's prompt is a general/creative topic with no specific facts to
  protect, write naturally and engagingly using your own knowledge.
- Break the narration into scenes so each scene is a single visual beat (aim for
  6-12 scenes for longer videos, 4-6 for short ~30-60s videos).
- Narration must sound natural when read aloud by text-to-speech: no headers, no
  emojis, no markdown.
- Keep total narration length close to the requested duration at ~140 words/minute.
- visual_query must describe concrete, filmable subjects (avoid abstract concepts).
- on_screen_text should surface the key number/name/fact from that scene's narration
  as short punchy text (e.g. "Revenue: Rs 412 Cr", "P/E: 22.5", "Roman Aqueducts, 312 BC"),
  so a muted viewer can still follow along. Use it whenever the scene has a concrete
  fact, figure, name, or date. Leave it as an empty string only for pure intro/outro/
  transition scenes with nothing concrete to show.
- The very first scene's on_screen_text MUST be the name of the main subject of the
  video (e.g. the company name, the person's name, the place/topic name) if the user's
  prompt names one — always show it, even though scene 1 is the hook/intro.
- diagram: ONLY set this (instead of leaving it null) when a scene's concept is a
  numeric comparison, trend, or sequence that has NO real filmable stock-footage
  equivalent (e.g. comparing two companies' P/E ratios, a revenue trend over time).
  Use ONLY numbers that literally appear in this scene's own narration — never invent
  new ones. Must be one of:
    {"kind": "number_line", "min": <int>, "max": <int>, "highlight": [<int>, ...]}
    {"kind": "triangular_dots", "rows": <int>, "label": "<short string>"}
    {"kind": "sequence_boxes", "items": ["<str>", ...], "highlight_index": <int or null>}
    {"kind": "bar_compare", "labels": ["<str>", ...], "values": [<number>, ...], "unit": "<str>"}
    {"kind": "shape_progression", "shape": "<str>", "counts": [<int>, ...]}
    {"kind": "trend_line", "labels": ["<str>", ...], "values": [<number>, ...], "unit": "<str>"}
  Prefer trend_line over bar_compare for a value changing across an ordered
  sequence (e.g. revenue/profit by year, a stock's price over time, compound
  growth) -- reserve bar_compare for comparing a handful of unordered items
  (e.g. two companies' P/E ratios side by side).
  For bar_compare/trend_line, "unit" must state what the values are in (e.g.
  "Cr", "%", "x") exactly as the narration states it -- a number shown on a
  diagram with no unit is as misleading as one with the wrong unit. Use ""
  only if the values are genuinely unitless (e.g. a count).
  Leave diagram as null for anything else, including every hook/intro/outro scene.
"""


def _extract_json(text: str) -> dict:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(json)?", "", text).rstrip("`").strip()
    if not text.startswith("{"):
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            text = match.group(0)
    return json.loads(text)


def write_script_from_prompt(prompt: str, duration_sec: int = 60, extra_tags: list = None, style: str = None) -> dict:
    """If the prompt already lays out a timed, section-by-section script (e.g.
    "0-5 seconds - Hook", "5-15 seconds - Overview", ...), each section becomes
    its own scene via a narrow per-section LLM call, keeping narration, visual
    and on-screen text tightly bound to the same source facts. Otherwise falls
    back to a single freeform generation call for the whole prompt.

    extra_tags: tags that must appear regardless of what the LLM comes up with
    (e.g. "NCERT", "CBSE" for a school-syllabus channel) — see
    segmented_script.merge_tags.

    style: overall tone/style guidance for the video. Threaded into both paths
    so a segmented (structured) prompt gets the same style guidance a freeform
    one does, instead of silently dropping it.
    """
    segments = segmented_script.parse_segments(prompt)
    if segments:
        return segmented_script.write_segmented_script(prompt, segments, duration_sec, extra_tags, global_instructions=style)
    freeform_prompt = f"{prompt}\n\nStyle/tone: {style}" if style else prompt
    return _write_freeform_script(freeform_prompt, duration_sec, extra_tags)


def _write_freeform_script(prompt: str, duration_sec: int = 60, extra_tags: list = None) -> dict:
    target_words = int(duration_sec / 60 * 140)
    is_finance_theme = research.needs_grounding(prompt)

    user_prompt = f"{prompt.strip()}\n\n"

    if is_finance_theme:
        facts = research.fetch_grounding_facts(prompt)
        if facts:
            user_prompt += (
                "Below are real, sourced facts from a live web search -- use ONLY "
                "these specific numbers, names, and claims; do not invent or add any "
                "other specific figures:\n"
                f"{facts}\n\n"
            )
        else:
            user_prompt += (
                "No verified real-world data could be retrieved for this topic. Do NOT "
                "invent specific numbers, prices, percentages, or dates -- keep the "
                "narration general and educational instead.\n\n"
            )

    user_prompt += (
        f"Target narration length: about {target_words} words "
        f"(~{duration_sec} seconds spoken).\n\n"
    )
    style_guidance = youtube_insights.get_combined_guidance()
    if style_guidance:
        user_prompt += (
            "Below is guidance on what's currently drawing viewers — based on this "
            "channel's own data and/or the wider niche — apply it to the title and the "
            "first scene's hook, without contradicting the facts above:\n"
            f"{style_guidance}\n\n"
        )
    user_prompt += "Produce the JSON script now."

    last_error = None
    for attempt in range(3):
        raw = llm.chat_json(SYSTEM_PROMPT, user_prompt, max_tokens=4096, timeout=120)
        try:
            script = _extract_json(raw)
            for key in ("title", "description", "tags", "scenes"):
                if key not in script:
                    raise ValueError(f"Script missing required field: {key}")
            if not script["scenes"]:
                raise ValueError("Script has zero scenes")
            script["title"] = segmented_script.clean_title(script["title"])[:70]
            script["description"] = segmented_script.normalize_indian_currency_units(script["description"])
            script["tags"] = segmented_script.merge_tags(extra_tags or [], script["tags"] or ["short video"])
            for scene in script["scenes"]:
                scene["narration"] = segmented_script.normalize_indian_currency_units(scene.get("narration") or "")
                diagram = scene.get("diagram")
                if diagram:
                    ok, _reason = diagrams.validate_diagram_spec(diagram, source_text=scene.get("narration", ""))
                    if not ok:
                        scene["diagram"] = None
            segmented_script.append_subscribe_cta(script["scenes"])
            for i, scene in enumerate(script["scenes"]):
                _ground_scene_on_screen_text(scene, is_first_scene=(i == 0))
            script["company_name"] = _extract_freeform_company_name(script, is_finance_theme)
            return script
        except (ValueError, KeyError) as e:
            last_error = e
            continue
    raise RuntimeError(
        f"Model returned an invalid script 3 times in a row ({last_error}). Try again."
    )


def _extract_freeform_company_name(script: dict, is_finance_theme: bool):
    """The freeform path has no per-section "content" field to run
    segmented_script._extract_company_name's "(NSE: SYMBOL)" regex against --
    most freeform finance themes are a plain one-line string (e.g. "National
    Aluminium Company - Stock Analysis For Long Term Growth") with no such
    marker. Without this, producer.py never had a company_name to pass to
    ai_thumbnail.generate_hook_background(), so the AI thumbnail silently
    never activated for ANY freeform-path video (confirmed via the
    "[ai_thumbnail] no company name available" log line on a real run).

    SYSTEM_PROMPT already requires the first scene's on_screen_text to be the
    video's subject name for exactly this reason -- reuse it instead of
    re-deriving a name from the raw theme string with a second regex. Only
    trusted for themes research.needs_grounding() already flagged as
    finance/named-entity topics, so a generic creative theme never forces an
    irrelevant "stock market" thumbnail background."""
    if not is_finance_theme:
        return None
    first_text = (script["scenes"][0].get("on_screen_text") or "").strip()
    if not first_text or first_text == segmented_script.SUBSCRIBE_CTA_ON_SCREEN_TEXT:
        return None
    return first_text


def _ground_scene_on_screen_text(scene: dict, is_first_scene: bool = False) -> None:
    """The freeform path writes a whole multi-scene script in a single LLM
    call -- unlike the segmented path (segmented_script._write_segment_scene),
    nothing previously checked that a scene's on_screen_text actually came
    from THAT scene's own narration. In practice the model sometimes drifts:
    showing a metric spoken in a different scene, or one it never said at all
    (reported: a real published video where on-screen metrics had "completely
    no correlation" with the narration). Reuses the same grounding/fact-
    extraction helpers the segmented path already relies on, applied against
    this scene's own narration (there's no separate source "content" field
    in the freeform path -- the narration itself is the only per-scene source
    of truth).

    is_first_scene: SYSTEM_PROMPT requires the very first scene's on_screen_text
    to be the video's subject name, not a metric -- unlike every other scene,
    where on_screen_text IS meant to surface a metric. So once that name has
    survived the grounding check, leave it alone: don't let a scene that
    happens to also narrate 2+ numbers (e.g. "current price of X and a market
    cap of Y") have its subject name clobbered by those numbers."""
    narration = scene.get("narration") or ""
    on_screen_text = segmented_script.normalize_indian_currency_units((scene.get("on_screen_text") or "").strip())
    if on_screen_text and not segmented_script._is_grounded(on_screen_text, narration):
        on_screen_text = ""
    if is_first_scene and on_screen_text:
        scene["on_screen_text"] = on_screen_text
        return
    facts = segmented_script._extract_facts(narration)
    if len(facts) > 1:
        on_screen_text = " | ".join(facts)
    elif not on_screen_text and facts:
        on_screen_text = facts[0]
    scene["on_screen_text"] = on_screen_text
