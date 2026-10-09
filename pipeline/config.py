"""Shared configuration and paths for the YT automation pipeline."""
import os
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")

PEXELS_API_KEY = os.environ.get("PEXELS_API_KEY", "")

GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
GROQ_MODEL = os.environ.get("GROQ_MODEL", "openai/gpt-oss-120b")

# Tavily search API — free tier, used to ground fact-dependent freeform themes
# (e.g. named-company stock analysis) in real web data before Groq writes the
# script. Optional: if unset, those themes fall back to a no-fabrication
# instruction instead of a hard failure. See pipeline/research.py.
TAVILY_API_KEY = os.environ.get("TAVILY_API_KEY", "")

# Hard guardrail so this never burns past Tavily's free monthly credit quota
# (~1,500-2,000/month on the $0 plan). Tracked in a git-committed file (not
# cache/, which is gitignored/ephemeral in CI) so the count survives across
# scheduled GitHub Actions runs. Kept deliberately far below the free quota —
# see pipeline/research.py for the enforcement logic.
TAVILY_MONTHLY_CALL_CAP = int(os.environ.get("TAVILY_MONTHLY_CALL_CAP", "40"))
TAVILY_USAGE_FILE = ROOT / "tavily_usage.json"

# Primary guardrail: stop calling Tavily once the account's real remaining
# credits (per Tavily's own GET /usage) drop below this buffer, so a month
# never runs the account to zero (or into paid overage) even if other tools
# share the same key. The call-count cap above only kicks in if this live
# check itself is unreachable.
TAVILY_SAFETY_BUFFER_CREDITS = int(os.environ.get("TAVILY_SAFETY_BUFFER_CREDITS", "50"))

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
