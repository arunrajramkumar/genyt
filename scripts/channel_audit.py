#!/usr/bin/env python3
"""Read-only channel audit: per-video views/CTR/retention/subs-gained plus
channel-wide traffic sources. Run: python3 scripts/channel_audit.py"""
import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from googleapiclient.discovery import build

from pipeline import youtube_insights as yi
from pipeline import youtube_upload


def main():
    creds = youtube_upload._load_credentials()
    youtube = build("youtube", "v3", credentials=creds, cache_discovery=False)
    analytics = build("youtubeAnalytics", "v2", credentials=creds, cache_discovery=False)

    ids = yi._uploaded_video_ids(youtube, 200)
    snippets = yi._video_snippets(youtube, ids)
    details = {}
    for i in range(0, len(ids), 50):
        resp = youtube.videos().list(part="contentDetails,status,statistics", id=",".join(ids[i:i + 50])).execute()
        for item in resp.get("items", []):
            details[item["id"]] = item
    ch = youtube.channels().list(part="statistics", mine=True).execute()["items"][0]["statistics"]
    print(f"CHANNEL: {ch.get('subscriberCount')} subs, {ch.get('viewCount')} views, {ch.get('videoCount')} videos\n")

    end = date.today().isoformat()
    metrics = "views,estimatedMinutesWatched,averageViewDuration,averageViewPercentage,subscribersGained,likes,comments,shares"
    rows = {}
    try:
        resp = analytics.reports().query(
            ids="channel==MINE", startDate="2000-01-01", endDate=end,
            metrics=metrics, dimensions="video", filters="video==" + ",".join(ids[:200]),
            maxResults=200,
        ).execute()
        cols = [h["name"] for h in resp["columnHeaders"]]
        rows = {r[0]: dict(zip(cols, r)) for r in resp.get("rows", [])}
    except Exception as e:
        print("per-video analytics failed:", e)

    print("PUBLISHED   PRIV     VIEWS AVGPCT  AVGSEC SUBS  LIKES  TITLE")
    for vid in sorted(ids, key=lambda v: snippets.get(v, {}).get("publishedAt", "")):
        sn, d, r = snippets.get(vid, {}), details.get(vid, {}), rows.get(vid, {})
        print(
            f"{sn.get('publishedAt', '')[:10]}  {d.get('status', {}).get('privacyStatus', '?'):8} "
            f"{r.get('views', 0):5} {r.get('averageViewPercentage', 0):6.1f} {r.get('averageViewDuration', 0):7.1f} "
            f"{r.get('subscribersGained', 0):4} {r.get('likes', 0):5}  {sn.get('title', '')[:70]}"
        )

    for dim in ("insightTrafficSourceType", "country", "ageGroup"):
        try:
            resp = analytics.reports().query(
                ids="channel==MINE", startDate="2000-01-01", endDate=end,
                metrics="views" if dim != "ageGroup" else "viewerPercentage",
                dimensions=dim, sort="-" + ("views" if dim != "ageGroup" else "viewerPercentage"), maxResults=10,
            ).execute()
            print(f"\nBY {dim}:")
            for row in resp.get("rows", []):
                print("  ", row)
        except Exception as e:
            print(f"\nBY {dim}: failed ({e})")


if __name__ == "__main__":
    main()
