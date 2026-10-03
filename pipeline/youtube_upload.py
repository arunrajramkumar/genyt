"""Uploads a finished video to YouTube via the YouTube Data API v3.

One-time setup required before this works — see README's "YouTube upload"
section: create an OAuth client in Google Cloud Console, download it as
youtube_client_secret.json in the repo root, then run
`python3 scripts/youtube_auth.py` once to grant access and store a reusable
token in cache/youtube_token.json. Until that's done, uploads are skipped
(NotConfiguredError) rather than failing the whole video-generation job.
"""
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload

from . import config

SCOPES = [
    "https://www.googleapis.com/auth/youtube.upload",
    "https://www.googleapis.com/auth/youtube.force-ssl",
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
) -> dict:
    """Uploads video_path to the authorized channel.

    privacy_status: "private" (default), "unlisted", or "public".
    category_id: YouTube category ID, 22 = "People & Blogs" (default for Shorts/vlogs).
    Returns {"video_id": str, "url": str}.
    """
    youtube = _client()
    body = {
        "snippet": {
            "title": title[:100],
            "description": description[:5000],
            "tags": tags[:500],
            "categoryId": category_id,
        },
        "status": {
            "privacyStatus": privacy_status or config.YOUTUBE_DEFAULT_PRIVACY,
            "selfDeclaredMadeForKids": False,
        },
    }
    media = MediaFileUpload(str(video_path), mimetype="video/mp4", resumable=True)
    request = youtube.videos().insert(part="snippet,status", body=body, media_body=media)
    response = None
    while response is None:
        _, response = request.next_chunk()
    video_id = response["id"]

    if captions_path:
        caption_media = MediaFileUpload(str(captions_path), mimetype="application/octet-stream", resumable=True)
        youtube.captions().insert(
            part="snippet",
            body={"snippet": {"videoId": video_id, "language": captions_language, "name": "", "isDraft": False}},
            media_body=caption_media,
        ).execute()

    return {"video_id": video_id, "url": f"https://youtu.be/{video_id}"}
