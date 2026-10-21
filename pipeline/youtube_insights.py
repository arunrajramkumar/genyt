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
from datetime import date, datetime

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


def fetch_channel_texts(max_videos: int = 500) -> list:
    """Title (first line) + description + tags for every video on the
    authenticated channel, private and unlisted included (the uploads
    playlist lists them all for the owner). Returns [] on any failure
    (unconfigured/expired token, API error) -- callers must treat that as
    "unknown", not "channel is empty"."""
    try:
        youtube = youtube_upload._client()
        ids = _uploaded_video_ids(youtube, max_videos)
        snippets = _video_snippets(youtube, ids)
    except Exception as e:
        print(f"[channel] could not list existing videos: {type(e).__name__}: {e}")
        return []
    return [
        "\n".join([sn.get("title", ""), sn.get("description", ""), " ".join(sn.get("tags", []))])
        for sn in snippets.values()
    ]


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
                endDate=date.today().isoformat(),
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


def _views_per_day(views: int, published_at: str) -> float:
    """Raw view count alone makes a month-old video look like a far bigger hit
    than a video uploaded yesterday purely because it's had more time to
    accumulate views, which skews the LLM's title/hook pattern-finding toward
    "this title is good" when the real signal is just "this video is older".
    Normalizing by days-live makes recent and older uploads comparable."""
    if not published_at:
        return 0.0
    try:
        published = datetime.fromisoformat(published_at.replace("Z", "+00:00")).date()
    except ValueError:
        return 0.0
    days_live = max(1, (date.today() - published).days)
    return round(views / days_live, 1)


def fetch_channel_insights(max_videos: int = 20) -> list:
    """Returns [{title, tags, views, views_per_day, ctr, avg_view_duration_sec}, ...]
    for the channel's most recent uploads. Raises youtube_upload.NotConfiguredError if
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
        views = row.get("views", 0)
        videos.append({
            "title": snippet.get("title", ""),
            "tags": snippet.get("tags", []),
            "views": views,
            "views_per_day": _views_per_day(views, snippet.get("publishedAt")),
            "published_at": snippet.get("publishedAt"),
            "ctr": row.get("impressionsClickThroughRate"),
            "avg_view_duration_sec": row.get("averageViewDuration"),
        })
    return videos


SYSTEM_PROMPT_INSIGHTS = """You are a YouTube Shorts growth strategist. Given a list of a
channel's recent videos with their title, tags, views-per-day-since-upload (use this, not
raw views, to judge performance -- a recently-uploaded video naturally has fewer total
views than an older one even if it's resonating just as well or better), click-through
rate (CTR, if available) and average view duration, identify concrete, actionable patterns
in what hooks viewers — title phrasing, use of numbers/specifics, tag choices, hook length.

Output ONLY valid JSON (no markdown fences, no commentary):

{"guidance": "a single string containing 3-6 short bullet points (newline-separated
within that one string, NOT a JSON array), each a concrete, specific instruction for
writing the NEXT video's title and opening hook, e.g. '- Lead with a specific number in
the first 5 words' — never vague advice like 'be engaging'."}

If the data doesn't show a clear pattern (too few videos, no real variation), say so
honestly in one bullet rather than inventing a pattern that isn't there.
"""


def _coerce_guidance(value) -> str:
    """A production run showed the model return "guidance" as a JSON array of
    bullet strings instead of the single newline-separated string the prompt
    asks for -- `(value or "").strip()` then crashed with AttributeError
    ('list' object has no attribute 'strip'), which silently killed this
    whole guidance call (caught by the broader except below) every time it
    happened. Accept either shape instead of trusting the model's formatting."""
    if isinstance(value, list):
        return "\n".join(str(item).strip() for item in value if str(item).strip())
    return (value or "").strip()


def summarize_insights(videos: list) -> str:
    if not videos:
        return ""
    lines = []
    for v in videos:
        ctr = f"{v['ctr']:.1f}% CTR" if v.get("ctr") is not None else "CTR n/a"
        duration = f"{v['avg_view_duration_sec']:.0f}s avg view" if v.get("avg_view_duration_sec") is not None else "duration n/a"
        lines.append(f"- \"{v['title']}\" | tags: {', '.join(v['tags'][:8])} | {v.get('views_per_day', 0)} views/day ({v['views']} total) | {ctr} | {duration}")
    user_prompt = "Recent videos:\n" + "\n".join(lines) + "\n\nProduce the JSON guidance now."

    last_error = None
    for attempt in range(2):
        raw = llm.chat_json(SYSTEM_PROMPT_INSIGHTS, user_prompt, max_tokens=400, timeout=60)
        try:
            data = _extract_json(raw)
            guidance = _coerce_guidance(data.get("guidance"))
            if guidance:
                return guidance
        except (ValueError, json.JSONDecodeError) as e:
            last_error = e
            continue
    return ""


def get_style_guidance(max_videos: int = 20) -> str:
    """Best-effort channel-performance guidance for the next video's title and
    hook. Never raises — any failure (not configured, too few videos, API
    error) just means no guidance is available yet.

    Printing the reason on every empty-return path (rather than swallowing it
    entirely) is deliberate: this whole feedback loop ran silently for weeks
    with no way to tell, from the outside, whether it was actually
    contributing guidance or quietly no-op'ing (e.g. a token missing the
    yt-analytics.readonly scope) -- see the "Worth double-checking: is the
    self-learning feedback loop actually running?" follow-up."""
    try:
        videos = fetch_channel_insights(max_videos)
    except Exception as e:
        print(f"[youtube_insights] channel insights unavailable ({e}); skipping own-channel guidance")
        return ""
    if not videos:
        print(
            f"[youtube_insights] fewer than {MIN_VIDEOS_FOR_INSIGHTS} uploaded videos with data yet; "
            "skipping own-channel guidance"
        )
        return ""
    try:
        guidance = summarize_insights(videos)
    except Exception as e:
        print(f"[youtube_insights] failed to summarize channel insights ({e}); skipping own-channel guidance")
        return ""
    if not guidance:
        print("[youtube_insights] model returned no usable guidance from channel insights")
    else:
        print(f"[youtube_insights] got own-channel guidance from {len(videos)} videos")
    return guidance


def posting_hour_report(videos: list, min_videos_per_hour: int = 2) -> list:
    """Groups past uploads by their UTC publish hour and averages views_per_day
    within each hour, so a posting-time tool can recommend when to schedule
    the next upload based on this channel's own real history -- no LLM
    involved, purely an aggregation of real data already fetched for
    get_style_guidance. Hours with fewer than `min_videos_per_hour` samples
    are dropped since one lucky/unlucky video shouldn't look like a trend.
    Returns [{hour, avg_views_per_day, sample_size}, ...] sorted best first."""
    buckets = {}
    for v in videos:
        published_at = v.get("published_at")
        if not published_at:
            continue
        try:
            hour = datetime.fromisoformat(published_at.replace("Z", "+00:00")).hour
        except ValueError:
            continue
        buckets.setdefault(hour, []).append(v.get("views_per_day", 0))

    report = [
        {"hour": hour, "avg_views_per_day": round(sum(vals) / len(vals), 1), "sample_size": len(vals)}
        for hour, vals in buckets.items()
        if len(vals) >= min_videos_per_hour
    ]
    report.sort(key=lambda r: r["avg_views_per_day"], reverse=True)
    return report


def get_posting_time_guidance(max_videos: int = 20) -> str:
    """Best-effort "post around hour X UTC" recommendation string from this
    channel's own upload history, or "" if there isn't enough same-hour
    repetition yet to say anything meaningful. Never raises."""
    try:
        videos = fetch_channel_insights(max_videos)
    except Exception as e:
        print(f"[youtube_insights] posting-time analysis unavailable ({e})")
        return ""
    report = posting_hour_report(videos)
    if not report:
        print("[youtube_insights] not enough repeated-hour upload history yet for a posting-time recommendation")
        return ""
    best = report[0]
    return (
        f"Videos posted around {best['hour']:02d}:00 UTC average {best['avg_views_per_day']} views/day "
        f"(from {best['sample_size']} uploads) -- the best of the hours with enough history to compare."
    )


_BANNED_GUIDANCE_RE = re.compile(
    r"emoji|long[- ]term wealth|wealth (?:play|builder)|growth (?:play|outlook)|deep dive|"
    r"[\U0001F300-\U0001FAFF☀-➿]",
    re.IGNORECASE,
)


def filter_title_guidance(text: str) -> str:
    """Drops guidance bullets that contradict the title rules (emoji, filler
    like "Long-Term Wealth") -- the model-written guidance kept recommending
    exactly those, and a published title ended up with an emoji that the render
    font draws as a missing-glyph box. Section headers (lines ending in ":")
    are kept only if at least one bullet under them survives."""
    kept, header = [], None
    for line in text.splitlines():
        if not line.strip():
            if header is None:
                kept.append(line)
            continue
        if line.rstrip().endswith(":") and not line.lstrip().startswith(("-", "*", "•")):
            header = line
            continue
        if _BANNED_GUIDANCE_RE.search(line):
            continue
        if header is not None:
            kept.append(header)
            header = None
        kept.append(line)
    return re.sub(r"\n{3,}", "\n\n", "\n".join(kept)).strip()


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
    combined = filter_title_guidance("\n\n".join(sections))
    if not combined:
        print("[youtube_insights] no channel or niche guidance available -- this video's title/hook prompt gets none")
    else:
        print(f"[youtube_insights] guidance fed into this video's title/hook prompt:\n{combined}")
    return combined
