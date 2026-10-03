#!/usr/bin/env python3
"""One-time interactive OAuth grant for YouTube uploads.

Opens a browser for you to sign in to the Google account that owns/manages
the target YouTube channel, then saves a reusable token so the pipeline can
upload videos without further prompting. Re-run this if cache/ is ever wiped
or you need to switch channels.

Usage: python3 scripts/youtube_auth.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from google_auth_oauthlib.flow import InstalledAppFlow

from pipeline import config
from pipeline.youtube_upload import SCOPES


def main():
    if not config.YOUTUBE_CLIENT_SECRETS_FILE.exists():
        print(
            f"Missing {config.YOUTUBE_CLIENT_SECRETS_FILE}.\n\n"
            "Create an OAuth client (type: Desktop app) in Google Cloud Console "
            "for a project with the YouTube Data API v3 AND YouTube Analytics API "
            "both enabled, download its JSON, and save it at that path (or point "
            "YOUTUBE_CLIENT_SECRETS_FILE at it). See README's \"YouTube upload\" section."
        )
        sys.exit(1)

    flow = InstalledAppFlow.from_client_secrets_file(str(config.YOUTUBE_CLIENT_SECRETS_FILE), SCOPES)
    creds = flow.run_local_server(port=0)

    token_path = config.CACHE_DIR / "youtube_token.json"
    token_path.write_text(creds.to_json())
    print(f"Saved YouTube credentials to {token_path}. Uploads will now work automatically.")


if __name__ == "__main__":
    main()
