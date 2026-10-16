#!/usr/bin/env python3
"""Reports which hour-of-day (UTC) this channel's own uploads have performed
best at, using real past-upload data (see pipeline/youtube_insights.py) --
no LLM, no guessing, just this channel's own history.

Run this by hand occasionally and use the result to decide whether to shift
the cron schedule in .github/workflows/auto-video*.yml. It only reports --
it never changes the schedule itself, since that's a one-line YAML edit a
human should glance at and commit deliberately.

Usage:
    python3 scripts/analyze_posting_times.py
    python3 scripts/analyze_posting_times.py --max-videos 50
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from pipeline import youtube_insights  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-videos", type=int, default=20, help="How many recent uploads to analyze")
    args = parser.parse_args()

    videos = youtube_insights.fetch_channel_insights(args.max_videos)
    if not videos:
        print("Not enough uploaded videos with data yet to analyze posting times.")
        return

    report = youtube_insights.posting_hour_report(videos)
    if not report:
        print(
            f"Analyzed {len(videos)} videos, but no UTC hour has been posted to more than once yet -- "
            "need more same-hour uploads before a posting-time pattern means anything."
        )
        return

    print(f"Analyzed {len(videos)} videos. Average views/day by posting hour (UTC), best first:\n")
    for row in report:
        print(f"  {row['hour']:02d}:00 UTC -- {row['avg_views_per_day']} views/day avg ({row['sample_size']} uploads)")
    best = report[0]
    print(f"\nBest-performing hour so far: {best['hour']:02d}:00 UTC.")
    print("If this holds up over more uploads, consider updating the cron schedule in .github/workflows/.")


if __name__ == "__main__":
    main()
