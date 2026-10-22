#!/usr/bin/env python3
"""Posts each recent upload's closing yes/no question as a channel comment once
the video has been made public (YouTube refuses comments on private videos,
and uploads start private). Safe to run any time: videos already carrying a
comment from the channel are skipped. Run: python3 scripts/post_engagement_comments.py"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pipeline import youtube_upload  # noqa: E402


def main():
    try:
        posted = youtube_upload.post_pending_engagement_comments()
    except youtube_upload.NotConfiguredError as e:
        print(f"Skipped: {e}")
        return
    print(f"Posted {posted} engagement comment(s)")


if __name__ == "__main__":
    main()
