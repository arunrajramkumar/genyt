"""Uses a local Ollama model to turn a theme into a structured video script."""
import json
import re

import requests

from . import config

SYSTEM_PROMPT = """You write scripts for short-form educational/entertainment YouTube videos.
Output ONLY valid JSON (no markdown fences, no commentary) matching this schema:

{
  "title": "punchy, clickable YouTube title, under 70 characters",
  "description": "2-3 sentence YouTube description, include a soft call to action",
  "tags": ["8-15 relevant search tags"],
  "scenes": [
    {
      "narration": "1-3 sentences of spoken narration for this scene",
      "visual_query": "2-4 word English search query for stock footage/photos that visually matches this scene"
    }
  ]
}

Rules:
- Break the narration into 6-12 scenes so each scene is a single visual beat.
- Narration must sound natural when read aloud by text-to-speech: no headers, no emojis, no markdown.
- Keep total narration length close to the requested duration at ~140 words/minute.
- visual_query must describe concrete, filmable subjects (avoid abstract concepts).
"""


def _extract_json(text: str) -> dict:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(json)?", "", text).rstrip("`").strip()
    # Ollama's json mode usually returns clean JSON, but models sometimes wrap it
    # in prose — grab the outermost {...} block as a fallback.
    if not text.startswith("{"):
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            text = match.group(0)
    return json.loads(text)


def write_script(theme: str, style: str = "", duration_sec: int = 180) -> dict:
    target_words = int(duration_sec / 60 * 140)

    user_prompt = (
        f"Theme: {theme}\n"
        f"Style/tone: {style or 'engaging, clear, conversational'}\n"
        f"Target narration length: about {target_words} words "
        f"(~{duration_sec} seconds spoken).\n\n"
        "Produce the JSON script now."
    )

    last_error = None
    for attempt in range(3):
        try:
            resp = requests.post(
                f"{config.OLLAMA_HOST}/api/chat",
                json={
                    "model": config.OLLAMA_MODEL,
                    "messages": [
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": user_prompt},
                    ],
                    "format": "json",
                    "stream": False,
                    "options": {"num_predict": 4096},
                },
                timeout=300,
            )
        except requests.ConnectionError as e:
            raise RuntimeError(
                f"Could not reach Ollama at {config.OLLAMA_HOST}. "
                "Is it running? Start it with `ollama serve` "
                f"(or the Ollama app), and make sure `{config.OLLAMA_MODEL}` is pulled: "
                f"`ollama pull {config.OLLAMA_MODEL}`."
            ) from e
        resp.raise_for_status()
        raw = resp.json()["message"]["content"]
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
        f"Local model returned an invalid script 3 times in a row ({last_error}). "
        "Try again, or use a larger OLLAMA_MODEL for more reliable JSON output."
    )
