#!/usr/bin/env python3
"""One-time interactive OAuth grant for YouTube uploads.

Opens a browser for you to sign in to the Google account that owns/manages
the target YouTube channel, then saves a reusable token so the pipeline can
upload videos without further prompting. Re-run this if cache/ is ever wiped,
you need to switch channels, or (while the OAuth consent screen is still in
"Testing" status) the refresh token hits its hard 7-day expiry.

Usage:
    python3 scripts/youtube_auth.py
    python3 scripts/youtube_auth.py --push-to-github common      # also pushes
    python3 scripts/youtube_auth.py --push-to-github education   # the refreshed
                                                                  # token straight
                                                                  # to that GitHub
                                                                  # Environment's
                                                                  # YOUTUBE_TOKEN_JSON
                                                                  # secret via the
                                                                  # GitHub API,
                                                                  # instead of you
                                                                  # copy-pasting it
                                                                  # into the GitHub
                                                                  # UI by hand.

--push-to-github still requires the one manual browser sign-in/consent click --
that step can't be automated away (and shouldn't be: scripting it would mean
storing your Google password and driving a fake browser login, which breaks
Google's ToS and trips 2FA/bot detection). It only removes the copy-paste step
after that.

It talks to the GitHub REST API directly (not the `gh` CLI), since `gh` may be
authenticated as an account without admin access to this repo's secrets. It
will prompt you to paste a GitHub Personal Access Token at that point -- create
one at https://github.com/settings/tokens with the "repo" scope (classic) or
"Secrets: write" repository permission (fine-grained), scoped to this repo.
The token is only read into memory for this one API call and is never stored.
"""
import argparse
import base64
import getpass
import sys
from pathlib import Path

import requests
from nacl import encoding, public

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from google_auth_oauthlib.flow import InstalledAppFlow

from pipeline import config
from pipeline.youtube_upload import SCOPES

GITHUB_REPO = "arunrajramkumar/genyt"
GITHUB_API = "https://api.github.com"


def _encrypt_secret(public_key_b64: str, secret_value: str) -> str:
    public_key = public.PublicKey(public_key_b64.encode("utf-8"), encoding.Base64Encoder())
    sealed_box = public.SealedBox(public_key)
    encrypted = sealed_box.encrypt(secret_value.encode("utf-8"))
    return base64.b64encode(encrypted).decode("utf-8")


def _push_to_github_secret(environment: str, token_json: str) -> None:
    token = getpass.getpass(
        "Paste a GitHub Personal Access Token with access to set secrets on "
        f"'{GITHUB_REPO}' (create one at https://github.com/settings/tokens, "
        "\"repo\" scope is enough) -- input is hidden and the token is not stored: "
    )
    headers = {"Authorization": f"Bearer {token}", "Accept": "application/vnd.github+json"}

    repo_resp = requests.get(f"{GITHUB_API}/repos/{GITHUB_REPO}", headers=headers)
    repo_resp.raise_for_status()
    repo_id = repo_resp.json()["id"]

    key_resp = requests.get(
        f"{GITHUB_API}/repositories/{repo_id}/environments/{environment}/secrets/public-key",
        headers=headers,
    )
    key_resp.raise_for_status()
    key_data = key_resp.json()

    print(f"Pushing refreshed token to the '{environment}' GitHub Environment's YOUTUBE_TOKEN_JSON secret...")
    put_resp = requests.put(
        f"{GITHUB_API}/repositories/{repo_id}/environments/{environment}/secrets/YOUTUBE_TOKEN_JSON",
        headers=headers,
        json={
            "encrypted_value": _encrypt_secret(key_data["key"], token_json),
            "key_id": key_data["key_id"],
        },
    )
    put_resp.raise_for_status()
    print("Done -- the next scheduled workflow run will use the refreshed token.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--push-to-github", metavar="ENVIRONMENT", choices=["common", "education"],
        help="After the browser consent, also push the token straight to this GitHub Environment's "
             "YOUTUBE_TOKEN_JSON secret via the GitHub API (you'll be prompted to paste a Personal "
             "Access Token with access to this repo's secrets) instead of pasting the token in by hand.",
    )
    args = parser.parse_args()

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
    token_json = creds.to_json()

    token_path = config.CACHE_DIR / "youtube_token.json"
    token_path.write_text(token_json)
    print(f"Saved YouTube credentials to {token_path}. Uploads will now work automatically.")

    if args.push_to_github:
        _push_to_github_secret(args.push_to_github, token_json)


if __name__ == "__main__":
    main()
