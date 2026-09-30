"""Turns a single free-form user prompt into a structured video script via a
hosted LLM (Groq). Generic — works for any topic (stock analysis, history,
science, etc.), not just finance.

If the prompt contains specific facts/figures (e.g. revenue numbers), the model
is instructed to use only those and never invent additional specific claims.
For open-ended topics without hard facts, it can write normally/engagingly.
"""
import json
import re

from . import llm, segmented_script

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
      "on_screen_text": "SHORT text overlay for this scene, under 6 words, or empty string if none needed"
    }
  ]
}

Rules:
- If the user's prompt includes specific facts, figures, names, or claims, use ONLY
  those — never invent, estimate, or add extra numbers, dates, or claims that were
  not given to you. If the prompt is an analysis of something (e.g. a stock, a
  product, a place), stay neutral and do not add a recommendation/verdict unless
  the user explicitly asked for one.
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


def write_script_from_prompt(prompt: str, duration_sec: int = 60) -> dict:
    """If the prompt already lays out a timed, section-by-section script (e.g.
    "0-5 seconds - Hook", "5-15 seconds - Overview", ...), each section becomes
    its own scene via a narrow per-section LLM call, keeping narration, visual
    and on-screen text tightly bound to the same source facts. Otherwise falls
    back to a single freeform generation call for the whole prompt.
    """
    segments = segmented_script.parse_segments(prompt)
    if segments:
        return segmented_script.write_segmented_script(prompt, segments, duration_sec)
    return _write_freeform_script(prompt, duration_sec)


def _write_freeform_script(prompt: str, duration_sec: int = 60) -> dict:
    target_words = int(duration_sec / 60 * 140)

    user_prompt = (
        f"{prompt.strip()}\n\n"
        f"Target narration length: about {target_words} words "
        f"(~{duration_sec} seconds spoken).\n\n"
        "Produce the JSON script now."
    )

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
            return script
        except (ValueError, KeyError) as e:
            last_error = e
            continue
    raise RuntimeError(
        f"Model returned an invalid script 3 times in a row ({last_error}). Try again."
    )
