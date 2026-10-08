#!/usr/bin/env python3
"""Fully-automated pipeline: theme -> script -> narration -> visuals -> finished video.

Usage:
    python3 run.py                 # process the next un-done theme in themes.yaml
    python3 run.py --all           # process every un-done theme
    python3 run.py --theme "..."   # run a one-off theme, ignoring themes.yaml
    python3 run.py --theme "..." --shorts   # vertical 9:16 output for YouTube Shorts
"""
import argparse
from pathlib import Path

import yaml

from pipeline import config, prompt_script, producer


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
    prompt = f"{theme}\n\nStyle/tone: {style}" if style else theme
    script = prompt_script.write_script_from_prompt(prompt, duration_sec, extra_tags)
    orientation = config.SHORTS if shorts else config.LANDSCAPE

    print("[2/2] Producing video")
    result = producer.produce_from_script(script, orientation, on_progress=print, thumbnail_badge=thumbnail_badge)
    print(f"Captions (upload separately as closed captions): {result['srt']}")
    print(f"Metadata: {result['metadata']}")
    return result["video"]


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
    args = parser.parse_args()

    if args.theme:
        produce_video(args.theme, args.style, args.duration, args.shorts)
        return

    themes_file = Path(args.themes_file)
    themes = load_themes(themes_file)
    pending = [t for t in themes if not t.get("done")]
    if not pending:
        print(f"No pending themes in {themes_file}. Add one or use --theme.")
        return

    to_run = pending if args.all else pending[:1]
    for entry in to_run:
        produce_video(
            entry["theme"],
            entry.get("style", ""),
            entry.get("duration_sec", 180),
            entry.get("shorts", False),
            entry.get("extra_tags"),
            entry.get("thumbnail_badge"),
        )
        entry["done"] = True
        save_themes(themes_file, themes)


if __name__ == "__main__":
    main()
