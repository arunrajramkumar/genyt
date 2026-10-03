"""Analyzes currently popular videos from the broader finance / stock-analysis /
personal-wealth niche — not just this channel — via the public YouTube Data
API v3, and summarizes what's drawing views right now into short guidance fed
into title/hook generation alongside this channel's own performance guidance
(see youtube_insights.py).

Uses the same OAuth credentials as the rest of the YouTube integration — no
extra scope needed, search/videos.list on public data works with any
authenticated client. Each search.list call costs 100 quota units against the
YouTube Data API's 10,000/day free quota, so the query list is kept short.
Best-effort throughout: a missing token, exhausted quota, or API error just
means no guidance, never a blocked video.
"""
import json
import re
from datetime import datetime, timedelta, timezone

from googleapiclient.discovery import build

from . import llm, youtube_upload

NICHE_QUERIES = ["stock analysis", "stock market investing", "personal finance tips"]
LOOKBACK_DAYS = 30
RESULTS_PER_QUERY = 10
MAX_VIDEOS_FOR_SUMMARY = 20


def _extract_json(text: str) -> dict:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(json)?", "", text).rstrip("`").strip()
    if not text.startswith("{"):
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            text = match.group(0)
    return json.loads(text)


def _client():
    return build("youtube", "v3", credentials=youtube_upload._load_credentials(), cache_discovery=False)


def _search_videos(youtube, query: str, published_after: str) -> list:
    resp = youtube.search().list(
        part="snippet", q=query, type="video", order="viewCount",
        publishedAfter=published_after, maxResults=RESULTS_PER_QUERY,
        relevanceLanguage="en",
    ).execute()
    return [
        {
            "video_id": item["id"]["videoId"],
            "title": item["snippet"]["title"],
            "channel_title": item["snippet"]["channelTitle"],
        }
        for item in resp.get("items", [])
    ]


def _video_view_counts(youtube, video_ids: list) -> dict:
    counts = {}
    for i in range(0, len(video_ids), 50):
        batch = video_ids[i:i + 50]
        resp = youtube.videos().list(part="statistics", id=",".join(batch)).execute()
        for item in resp.get("items", []):
            counts[item["id"]] = int(item.get("statistics", {}).get("viewCount", 0))
    return counts


def fetch_trending_niche_videos(queries: list = None, max_videos: int = MAX_VIDEOS_FOR_SUMMARY) -> list:
    """Returns the channel-agnostic set of currently popular niche videos,
    newest-trend-first by view count: [{title, channel_title, views}, ...]."""
    youtube = _client()
    published_after = (datetime.now(timezone.utc) - timedelta(days=LOOKBACK_DAYS)).strftime("%Y-%m-%dT%H:%M:%SZ")

    seen = {}
    for query in (queries or NICHE_QUERIES):
        for v in _search_videos(youtube, query, published_after):
            seen.setdefault(v["video_id"], v)

    video_ids = list(seen.keys())[:max_videos]
    if not video_ids:
        return []
    view_counts = _video_view_counts(youtube, video_ids)

    videos = [dict(seen[vid], views=view_counts.get(vid, 0)) for vid in video_ids]
    videos.sort(key=lambda v: v["views"], reverse=True)
    return videos


SYSTEM_PROMPT_MARKET_TRENDS = """You are a YouTube growth strategist specializing in the
finance / stock-analysis / personal-wealth niche. Given a list of currently popular videos
from OTHER channels in this niche (title, channel, view count), identify concrete patterns
in what's drawing views right now — hot subtopics, title phrasing, angle/format.

Output ONLY valid JSON (no markdown fences, no commentary):

{"guidance": "3-6 short bullet points, each a concrete, specific instruction for the NEXT
video's topic framing, title, and hook, e.g. '- Videos naming a specific stock plus a bold
number in the title are outperforming generic how-to-invest videos right now' — never
vague advice like 'make it interesting'."}

Stay neutral and factual — don't invent a trend that isn't visible in the data given.
"""


def summarize_market_trends(videos: list) -> str:
    if not videos:
        return ""
    lines = [f"- \"{v['title']}\" ({v['channel_title']}) — {v['views']:,} views" for v in videos]
    user_prompt = "Currently popular niche videos:\n" + "\n".join(lines) + "\n\nProduce the JSON guidance now."

    for attempt in range(2):
        raw = llm.chat_json(SYSTEM_PROMPT_MARKET_TRENDS, user_prompt, max_tokens=400, timeout=60)
        try:
            data = _extract_json(raw)
            guidance = (data.get("guidance") or "").strip()
            if guidance:
                return guidance
        except (ValueError, json.JSONDecodeError):
            continue
    return ""


def get_market_trend_guidance(queries: list = None) -> str:
    """Best-effort niche-wide trend guidance for the next video. Never raises —
    any failure (not configured, quota exhausted, API error) just means no
    guidance is available yet."""
    try:
        videos = fetch_trending_niche_videos(queries)
    except Exception:
        return ""
    try:
        return summarize_market_trends(videos)
    except Exception:
        return ""
