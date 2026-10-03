"""Analyzes the channel's own past video performance via the YouTube Analytics
API and turns it into a short, actionable style guide that's fed into title
and hook generation for the *next* video — aimed at improving click-through
and retention, not just raw view count.

Needs the yt-analytics.readonly scope (see youtube_upload.SCOPES) in addition
to the upload scopes already granted — re-run `scripts/youtube_auth.py` once
after this scope was added (existing tokens don't retroactively gain it).

Every call here is best-effort: if the channel isn't set up yet, has too few
videos, or the Analytics API rejects a query, this returns "" / [] rather than
raising, so a missing/incomplete channel history never blocks generation.
"""
import json
import re

from googleapiclient.discovery import build

from . import llm, market_trends, youtube_upload

ANALYTICS_METRICS_FULL = "views,estimatedMinutesWatched,averageViewDuration,impressions,impressionsClickThroughRate"
ANALYTICS_METRICS_BASIC = "views,estimatedMinutesWatched,averageViewDuration"

MIN_VIDEOS_FOR_INSIGHTS = 3


def _extract_json(text: str) -> dict:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(json)?", "", text).rstrip("`").strip()
    if not text.startswith("{"):
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            text = match.group(0)
    return json.loads(text)


def _uploaded_video_ids(youtube, max_videos: int) -> list:
    channels = youtube.channels().list(part="contentDetails", mine=True).execute()
    items = channels.get("items") or []
    if not items:
        return []
    uploads_playlist = items[0]["contentDetails"]["relatedPlaylists"]["uploads"]

    video_ids = []
    page_token = None
    while len(video_ids) < max_videos:
        resp = youtube.playlistItems().list(
            part="contentDetails",
            playlistId=uploads_playlist,
            maxResults=min(50, max_videos - len(video_ids)),
            pageToken=page_token,
        ).execute()
        video_ids += [item["contentDetails"]["videoId"] for item in resp.get("items", [])]
        page_token = resp.get("nextPageToken")
        if not page_token:
            break
    return video_ids


def _video_snippets(youtube, video_ids: list) -> dict:
    snippets = {}
    for i in range(0, len(video_ids), 50):
        batch = video_ids[i:i + 50]
        resp = youtube.videos().list(part="snippet", id=",".join(batch)).execute()
        for item in resp.get("items", []):
            snippets[item["id"]] = item["snippet"]
    return snippets


def _analytics_rows(analytics, video_ids: list) -> dict:
    """Returns {video_id: {metric_name: value}}. Falls back to a smaller
    metric set if the full one (impressions/CTR) isn't queryable for this
    channel's access tier."""
    filters = f"video=={','.join(video_ids)}"
    for metrics in (ANALYTICS_METRICS_FULL, ANALYTICS_METRICS_BASIC):
        try:
            resp = analytics.reports().query(
                ids="channel==MINE",
                startDate="2000-01-01",
                endDate="2100-01-01",
                metrics=metrics,
                dimensions="video",
                filters=filters,
            ).execute()
        except Exception:
            continue
        columns = [h["name"] for h in resp.get("columnHeaders", [])]
        return {
            dict(zip(columns, row))["video"]: dict(zip(columns, row))
            for row in resp.get("rows", [])
        }
    return {}


def fetch_channel_insights(max_videos: int = 20) -> list:
    """Returns [{title, tags, views, ctr, avg_view_duration_sec}, ...] for the
    channel's most recent uploads. Raises youtube_upload.NotConfiguredError if
    no YouTube token exists yet. Returns [] if there aren't enough videos for
    a pattern to be meaningful."""
    creds = youtube_upload._load_credentials()
    youtube = build("youtube", "v3", credentials=creds, cache_discovery=False)
    analytics = build("youtubeAnalytics", "v2", credentials=creds, cache_discovery=False)

    video_ids = _uploaded_video_ids(youtube, max_videos)
    if len(video_ids) < MIN_VIDEOS_FOR_INSIGHTS:
        return []

    snippets = _video_snippets(youtube, video_ids)
    analytics_rows = _analytics_rows(analytics, video_ids)

    videos = []
    for vid in video_ids:
        snippet = snippets.get(vid, {})
        row = analytics_rows.get(vid, {})
        videos.append({
            "title": snippet.get("title", ""),
            "tags": snippet.get("tags", []),
            "views": row.get("views", 0),
            "ctr": row.get("impressionsClickThroughRate"),
            "avg_view_duration_sec": row.get("averageViewDuration"),
        })
    return videos


SYSTEM_PROMPT_INSIGHTS = """You are a YouTube Shorts growth strategist. Given a list of a
channel's recent videos with their title, tags, view count, click-through rate (CTR, if
available) and average view duration, identify concrete, actionable patterns in what
hooks viewers — title phrasing, use of numbers/specifics, tag choices, hook length.

Output ONLY valid JSON (no markdown fences, no commentary):

{"guidance": "3-6 short bullet points, each a concrete, specific instruction for writing
the NEXT video's title and opening hook, e.g. '- Lead with a specific number in the
first 5 words' — never vague advice like 'be engaging'."}

If the data doesn't show a clear pattern (too few videos, no real variation), say so
honestly in one bullet rather than inventing a pattern that isn't there.
"""


def summarize_insights(videos: list) -> str:
    if not videos:
        return ""
    lines = []
    for v in videos:
        ctr = f"{v['ctr']:.1f}% CTR" if v.get("ctr") is not None else "CTR n/a"
        duration = f"{v['avg_view_duration_sec']:.0f}s avg view" if v.get("avg_view_duration_sec") is not None else "duration n/a"
        lines.append(f"- \"{v['title']}\" | tags: {', '.join(v['tags'][:8])} | {v['views']} views | {ctr} | {duration}")
    user_prompt = "Recent videos:\n" + "\n".join(lines) + "\n\nProduce the JSON guidance now."

    last_error = None
    for attempt in range(2):
        raw = llm.chat_json(SYSTEM_PROMPT_INSIGHTS, user_prompt, max_tokens=400, timeout=60)
        try:
            data = _extract_json(raw)
            guidance = (data.get("guidance") or "").strip()
            if guidance:
                return guidance
        except (ValueError, json.JSONDecodeError) as e:
            last_error = e
            continue
    return ""


def get_style_guidance(max_videos: int = 20) -> str:
    """Best-effort channel-performance guidance for the next video's title and
    hook. Never raises — any failure (not configured, too few videos, API
    error) just means no guidance is available yet."""
    try:
        videos = fetch_channel_insights(max_videos)
    except Exception:
        return ""
    try:
        return summarize_insights(videos)
    except Exception:
        return ""


def get_combined_guidance() -> str:
    """Merges this channel's own performance guidance with broader
    finance/stock-analysis niche trend guidance (see market_trends.py) into
    one blurb for the title/hook prompt. Either half may be empty (e.g. a
    brand-new channel with no own-channel data yet) — this never raises."""
    sections = []
    own = get_style_guidance()
    if own:
        sections.append(f"This channel's own recent-video performance:\n{own}")
    niche = market_trends.get_market_trend_guidance()
    if niche:
        sections.append(f"Broader finance/stock-analysis/wealth niche trends right now:\n{niche}")
    return "\n\n".join(sections)
