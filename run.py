#!/usr/bin/env python3
"""Fully-automated pipeline: theme -> script -> narration -> visuals -> finished video.

Usage:
    python3 run.py                 # process the next un-done theme in themes.yaml
    python3 run.py --all           # process every un-done theme
    python3 run.py --theme "..."   # run a one-off theme, ignoring themes.yaml
    python3 run.py --theme "..." --shorts   # vertical 9:16 output for YouTube Shorts
    python3 run.py --trending      # produce one video about a currently-trending NSE
                                    # stock (see pipeline/trending_stocks.py) instead of
                                    # pulling the next entry from themes.yaml; falls back
                                    # to the themes.yaml queue if no trending candidate
                                    # can be found/verified right now
"""
import argparse
import sys
from pathlib import Path

import yaml

from pipeline import config, prompt_script, producer, trending_stocks
from pipeline.tts import TTSQuotaExceededError
from scripts import build_finance_queue


def load_themes(themes_file: Path) -> list[dict]:
    if not themes_file.exists():
        return []
    data = yaml.safe_load(themes_file.read_text()) or {}
    return data.get("themes", [])


def save_themes(themes_file: Path, themes: list[dict]) -> None:
    themes_file.write_text(
        yaml.safe_dump({"themes": themes}, sort_keys=False, allow_unicode=True)
    )


def produce_video(theme: str, style: str = "", duration_sec: int = 180, shorts: bool = False, extra_tags: list = None,
                   thumbnail_badge: str = None) -> Path:
    print(f"[1/2] Writing script for: {theme!r}")
    # prompt_script (not the older script_writer) pulls in this channel's own
    # performance guidance plus broader niche trend guidance (see
    # pipeline/youtube_insights.py / market_trends.py) before writing the
    # title/hook, so scheduled runs benefit from the same dynamic trend
    # feedback the web UI already gets.
    script = prompt_script.write_script_from_prompt(theme, duration_sec, extra_tags, style=style)
    orientation = config.SHORTS if shorts else config.LANDSCAPE

    print("[2/2] Producing video")
    result = producer.produce_from_script(script, orientation, on_progress=print, thumbnail_badge=thumbnail_badge)
    print(f"Captions (upload separately as closed captions): {result['srt']}")
    print(f"Metadata: {result['metadata']}")
    return result["video"]


def next_trending_entry(themes_file: Path) -> dict | None:
    """Best-effort themes.yaml-style entry for one currently-trending,
    fundamentals-verified NSE stock not already covered in themes_file.
    Returns None (never raises) if no trending candidate can be found or
    verified right now, so callers fall back to the static queue instead of
    producing a video with no real candidate behind it."""
    exclude = build_finance_queue.existing_symbols(str(themes_file))
    candidates = trending_stocks.select_trending_symbols(exclude_symbols=exclude, limit=1)
    if not candidates:
        return None
    candidate = candidates[0]
    return build_finance_queue.build_theme_entry(candidate["symbol"], candidate["name"])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--theme", help="Run a single one-off theme, ignoring themes.yaml")
    parser.add_argument("--style", default="", help="Style/tone hint for --theme")
    parser.add_argument("--duration", type=int, default=180, help="Target seconds for --theme")
    parser.add_argument("--shorts", action="store_true", help="Produce vertical 9:16 output for YouTube Shorts")
    parser.add_argument("--all", action="store_true", help="Process every un-done theme in themes.yaml")
    parser.add_argument(
        "--themes-file", default=str(config.THEMES_FILE),
        help="Path to the theme queue YAML (default: themes.yaml) -- use a different file to run a separate channel's queue",
    )
    parser.add_argument(
        "--trending", action="store_true",
        help="Produce one video about a currently-trending, verified-real NSE stock (see "
             "pipeline/trending_stocks.py) instead of pulling the next entry from --themes-file. "
             "Ignored together with --theme/--all. Falls back to the themes.yaml queue if no "
             "trending candidate can be found/verified this run.",
    )
    args = parser.parse_args()

    if args.theme:
        try:
            produce_video(args.theme, args.style, args.duration, args.shorts)
        except TTSQuotaExceededError as e:
            # Exit cleanly (code 0) instead of crashing the scheduled job with
            # a traceback -- this condition recurs daily until the monthly
            # quota resets, so a hard failure would just be noise every run.
            print(f"Skipping: {e}")
            sys.exit(0)
        return

    themes_file = Path(args.themes_file)

    if args.trending and not args.all:
        trending_entry = next_trending_entry(themes_file)
        if trending_entry is None:
            print("No trending candidate found/verified right now -- falling back to the themes.yaml queue")
        else:
            try:
                produce_video(
                    trending_entry["theme"],
                    trending_entry.get("style", ""),
                    trending_entry.get("duration_sec", 180),
                    trending_entry.get("shorts", False),
                    trending_entry.get("extra_tags"),
                    trending_entry.get("thumbnail_badge"),
                )
            except TTSQuotaExceededError as e:
                print(f"Skipping: {e}")
                sys.exit(0)
            # Record it in themes_file (done:true) so future --trending runs
            # don't re-pick the same company, and so it shows up in the same
            # audit trail as every hand-written theme.
            trending_entry["done"] = True
            themes = load_themes(themes_file)
            themes.append(trending_entry)
            save_themes(themes_file, themes)
            return

    themes = load_themes(themes_file)
    pending = [t for t in themes if not t.get("done")]
    if not pending:
        print(f"No pending themes in {themes_file}. Add one or use --theme.")
        return

    to_run = pending if args.all else pending[:1]
    for entry in to_run:
        try:
            produce_video(
                entry["theme"],
                entry.get("style", ""),
                entry.get("duration_sec", 180),
                entry.get("shorts", False),
                entry.get("extra_tags"),
                entry.get("thumbnail_badge"),
            )
        except TTSQuotaExceededError as e:
            # Leave this theme pending (not done) so it's retried once next
            # month's quota resets, and stop processing the rest of this run's
            # queue -- the same budget is shared across all of them, so every
            # other entry would hit the identical guardrail.
            print(f"Skipping: {e}")
            break
        entry["done"] = True
        save_themes(themes_file, themes)


if __name__ == "__main__":
    main()
