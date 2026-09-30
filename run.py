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

from pipeline import config, script_writer, producer


def load_themes() -> list[dict]:
    if not config.THEMES_FILE.exists():
        return []
    data = yaml.safe_load(config.THEMES_FILE.read_text()) or {}
    return data.get("themes", [])


def save_themes(themes: list[dict]) -> None:
    config.THEMES_FILE.write_text(
        yaml.safe_dump({"themes": themes}, sort_keys=False, allow_unicode=True)
    )


def produce_video(theme: str, style: str = "", duration_sec: int = 180, shorts: bool = False) -> Path:
    print(f"[1/2] Writing script for: {theme!r}")
    script = script_writer.write_script(theme, style, duration_sec)
    orientation = config.SHORTS if shorts else config.LANDSCAPE

    print("[2/2] Producing video")
    result = producer.produce_from_script(script, orientation, on_progress=print)
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
    args = parser.parse_args()

    if args.theme:
        produce_video(args.theme, args.style, args.duration, args.shorts)
        return

    themes = load_themes()
    pending = [t for t in themes if not t.get("done")]
    if not pending:
        print("No pending themes in themes.yaml. Add one or use --theme.")
        return

    to_run = pending if args.all else pending[:1]
    for entry in to_run:
        produce_video(
            entry["theme"],
            entry.get("style", ""),
            entry.get("duration_sec", 180),
            entry.get("shorts", False),
        )
        entry["done"] = True
        save_themes(themes)


if __name__ == "__main__":
    main()
