"""Shared configuration and paths for the YT automation pipeline."""
import os
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")

PEXELS_API_KEY = os.environ.get("PEXELS_API_KEY", "")

GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
GROQ_MODEL = os.environ.get("GROQ_MODEL", "openai/gpt-oss-120b")

THEMES_FILE = ROOT / "themes.yaml"
OUTPUT_DIR = ROOT / "output"
CACHE_DIR = ROOT / "cache"

# Narration voice — Microsoft Edge neural TTS (free, no API key).
# en-IN-PrabhatNeural / en-IN-NeerjaNeural give an Indian-English accent;
# run `edge-tts --list-voices` for the full catalog.
TTS_VOICE = os.environ.get("TTS_VOICE", "en-IN-PrabhatNeural")

# Video output settings — kept modest (720p/24fps) so ffmpeg's encode stays
# under the ~512MB RAM ceiling of the free hosting tier this runs on.
VIDEO_WIDTH = 1280
VIDEO_HEIGHT = 720
VIDEO_FPS = 24

# Orientation presets
LANDSCAPE = {"width": 1280, "height": 720, "orientation": "landscape"}
SHORTS = {"width": 720, "height": 1280, "orientation": "portrait"}

# YouTube upload — see README's "YouTube upload" section for one-time OAuth setup.
YOUTUBE_CLIENT_SECRETS_FILE = Path(
    os.environ.get("YOUTUBE_CLIENT_SECRETS_FILE", str(ROOT / "youtube_client_secret.json"))
)
YOUTUBE_DEFAULT_PRIVACY = os.environ.get("YOUTUBE_DEFAULT_PRIVACY", "private")
YOUTUBE_AUTO_UPLOAD = os.environ.get("YOUTUBE_AUTO_UPLOAD", "true").strip().lower() not in ("0", "false", "no")
# On hosts with an ephemeral/read-only filesystem (e.g. Render secret files),
# the live token can't be refreshed in place. Point this at that read-only
# seed copy and the pipeline will bootstrap a writable copy into CACHE_DIR
# the first time it's needed, re-seeding on every fresh instance boot.
YOUTUBE_TOKEN_SEED_FILE = (
    Path(os.environ["YOUTUBE_TOKEN_SEED_FILE"]) if os.environ.get("YOUTUBE_TOKEN_SEED_FILE") else None
)

for d in (OUTPUT_DIR, CACHE_DIR):
    d.mkdir(parents=True, exist_ok=True)
