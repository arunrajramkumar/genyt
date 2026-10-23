"""Uploads a finished video to YouTube via the YouTube Data API v3.

One-time setup required before this works — see README's "YouTube upload"
section: create an OAuth client in Google Cloud Console, download it as
youtube_client_secret.json in the repo root, then run
`python3 scripts/youtube_auth.py` once to grant access and store a reusable
token in cache/youtube_token.json. Until that's done, uploads are skipped
(NotConfiguredError) rather than failing the whole video-generation job.
"""
import re

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload

from . import config

SCOPES = [
    "https://www.googleapis.com/auth/youtube.upload",
    "https://www.googleapis.com/auth/youtube.force-ssl",
    "https://www.googleapis.com/auth/yt-analytics.readonly",
]


class NotConfiguredError(RuntimeError):
    """Raised when no YouTube OAuth token has been set up yet."""


def _token_path():
    return config.CACHE_DIR / "youtube_token.json"


def _load_credentials() -> Credentials:
    token_path = _token_path()
    if not token_path.exists() and config.YOUTUBE_TOKEN_SEED_FILE and config.YOUTUBE_TOKEN_SEED_FILE.exists():
        token_path.write_text(config.YOUTUBE_TOKEN_SEED_FILE.read_text())
    if not token_path.exists():
        seed = config.YOUTUBE_TOKEN_SEED_FILE
        raise NotConfiguredError(
            "No YouTube credentials found. Run `python3 scripts/youtube_auth.py` once "
            "to authorize this app against your channel (see README). Checked: "
            f"{token_path} (exists={token_path.exists()}), "
            f"YOUTUBE_TOKEN_SEED_FILE={seed} (exists={seed.exists() if seed else 'unset'})."
        )
    creds = Credentials.from_authorized_user_file(str(token_path), SCOPES)
    if creds.expired and creds.refresh_token:
        creds.refresh(Request())
        token_path.write_text(creds.to_json())
    return creds


def _client():
    return build("youtube", "v3", credentials=_load_credentials(), cache_discovery=False)


def upload_video(
    video_path,
    title: str,
    description: str,
    tags: list,
    privacy_status: str = None,
    category_id: str = "22",
    captions_path=None,
    captions_language: str = "en",
    default_language: str = None,
) -> dict:
    """Uploads video_path to the authorized channel.

    privacy_status: "private" (default), "unlisted", or "public".
    category_id: YouTube category ID, 22 = "People & Blogs" (default for Shorts/vlogs).
    default_language: BCP-47 language code for defaultLanguage/defaultAudioLanguage
    (e.g. "en") -- these were never set at all before, which left YouTube to guess
    the video's spoken language from scratch for search/recommendation matching
    instead of being told outright. Omitted when None.
    Returns {"video_id": str, "url": str}.
    """
    youtube = _client()
    snippet = {
        "title": title[:100],
        "description": description[:5000],
        "tags": tags[:500],
        "categoryId": category_id,
    }
    if default_language:
        snippet["defaultLanguage"] = default_language
        snippet["defaultAudioLanguage"] = default_language
    body = {
        "snippet": snippet,
        "status": {
            "privacyStatus": privacy_status or config.YOUTUBE_DEFAULT_PRIVACY,
            "selfDeclaredMadeForKids": False,
        },
    }
    media = MediaFileUpload(str(video_path), mimetype="video/mp4", resumable=True)
    request = youtube.videos().insert(part="snippet,status", body=body, media_body=media)
    response = None
    while response is None:
        # A multi-minute resumable upload has plenty of chances to hit a
        # transient network blip or a 5xx from YouTube -- without retries,
        # that silently throws away an entire render (the slow, expensive
        # part of the pipeline) over a single bad chunk. num_retries>0 makes
        # next_chunk() retry a failed chunk with exponential backoff instead
        # of raising immediately.
        _, response = request.next_chunk(num_retries=5)
    video_id = response["id"]

    if captions_path:
        caption_media = MediaFileUpload(str(captions_path), mimetype="application/octet-stream", resumable=True)
        youtube.captions().insert(
            part="snippet",
            body={"snippet": {"videoId": video_id, "language": captions_language, "name": "", "isDraft": False}},
            media_body=caption_media,
        ).execute()

    return {"video_id": video_id, "url": f"https://youtu.be/{video_id}",
            "privacy_status": privacy_status or config.YOUTUBE_DEFAULT_PRIVACY}


def set_thumbnail(video_id: str, thumbnail_path) -> None:
    """Sets the custom thumbnail for an already-uploaded video. Requires the
    channel to be phone-verified (YouTube restricts custom thumbnails to
    verified channels) -- otherwise the API call raises, which callers should
    treat as non-fatal the same way a failed upload_video() call is."""
    youtube = _client()
    media = MediaFileUpload(str(thumbnail_path), mimetype="image/png")
    youtube.thumbnails().set(videoId=video_id, media_body=media).execute()


def post_engagement_comment(video_id: str, text: str) -> None:
    """Posts a top-level comment from the channel on an already-uploaded video,
    giving viewers a question to reply to (see producer's engagement question).
    NOTE: the YouTube Data API cannot pin comments, so this comment is never
    pinned automatically -- the channel owner has to pin it by hand in Studio.
    Callers should treat a failure here as non-fatal, same as set_thumbnail
    (e.g. commentsDisabled or a forbidden response on a private video)."""
    youtube = _client()
    youtube.commentThreads().insert(
        part="snippet",
        body={"snippet": {"videoId": video_id, "topLevelComment": {"snippet": {"textOriginal": text}}}},
    ).execute()


ENGAGEMENT_PREFIX = "💬 "
ENGAGEMENT_SUFFIX = " Reply YES or NO in the comments."
ENGAGEMENT_SUFFIX_OPEN = " Tell us your view in the comments."
_ENGAGEMENT_LINE_RE = re.compile(
    rf"^{re.escape(ENGAGEMENT_PREFIX)}(.+?\?)(?:{re.escape(ENGAGEMENT_SUFFIX)}|{re.escape(ENGAGEMENT_SUFFIX_OPEN)})\s*$",
    re.MULTILINE,
)


def _is_either_or(question: str) -> bool:
    """"Is this a buying opportunity or the start of a deeper correction?"
    can't be answered YES/NO ("... or not?" still can)."""
    return bool(re.search(r"\bor\b(?!\s+not\b)", question, re.IGNORECASE))


def engagement_line(question: str) -> str:
    """The description line carrying the video's closing question."""
    return f"{ENGAGEMENT_PREFIX}{question}{ENGAGEMENT_SUFFIX_OPEN if _is_either_or(question) else ENGAGEMENT_SUFFIX}"


def engagement_comment(question: str) -> str:
    return f"{question} {'Tell us below' if _is_either_or(question) else 'Reply YES or NO below'} 👇"


def question_from_description(description: str) -> str:
    m = _ENGAGEMENT_LINE_RE.search(description or "")
    return m.group(1).strip() if m else ""


def post_pending_engagement_comments(max_videos: int = 30, on_progress=print) -> int:
    """Posts the engagement comment on recent uploads that have since been made
    public and don't have one from the channel yet. Uploads start private (the
    owner reviews and flips them public by hand) and YouTube rejects comments
    on private videos with a 403 "forbidden" -- so the comment can't be posted
    at upload time; this catches up on every later run instead. The question
    comes from the description line written at upload (engagement_line), so
    no extra state is kept. Returns how many comments were posted; each
    video's failure is logged and skipped."""
    youtube = _client()
    channel = youtube.channels().list(part="id,contentDetails", mine=True).execute()["items"][0]
    channel_id = channel["id"]
    uploads = channel["contentDetails"]["relatedPlaylists"]["uploads"]
    items = youtube.playlistItems().list(part="contentDetails", playlistId=uploads, maxResults=min(max_videos, 50)).execute()
    ids = [i["contentDetails"]["videoId"] for i in items.get("items", [])]
    if not ids:
        return 0
    videos = youtube.videos().list(part="snippet,status", id=",".join(ids)).execute().get("items", [])

    posted = 0
    for video in videos:
        if video.get("status", {}).get("privacyStatus") != "public":
            continue
        question = question_from_description(video.get("snippet", {}).get("description", ""))
        if not question:
            continue
        try:
            threads = youtube.commentThreads().list(part="snippet", videoId=video["id"], maxResults=100).execute()
            if any(t["snippet"]["topLevelComment"]["snippet"].get("authorChannelId", {}).get("value") == channel_id
                   for t in threads.get("items", [])):
                continue
            post_engagement_comment(video["id"], engagement_comment(question))
            posted += 1
            on_progress(f"Engagement comment posted on {video['id']}")
        except Exception as e:
            on_progress(f"Engagement comment skipped for {video['id']}: {e}")
    return posted


def add_to_playlist(video_id: str, playlist_id: str) -> None:
    """Adds an already-uploaded video to a playlist. A populated playlist gives
    YouTube's autoplay/"up next" another video from this channel to chain into,
    and gives a viewer who likes one video an easy next click -- but this
    requires a real playlist ID the user must create and supply (see
    config.YOUTUBE_PLAYLIST_ID), so it's a no-op unless that's configured.
    Callers should treat a failure here as non-fatal, same as set_thumbnail."""
    youtube = _client()
    youtube.playlistItems().insert(
        part="snippet",
        body={"snippet": {"playlistId": playlist_id, "resourceId": {"kind": "youtube#video", "videoId": video_id}}},
    ).execute()
