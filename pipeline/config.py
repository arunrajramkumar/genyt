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

# Hard guardrail so this never burns past Tavily's real monthly credit quota
# (confirmed 1,500 credits/month on the current plan). Tracked in a
# git-committed file (not cache/, which is gitignored/ephemeral in CI) so the
# count survives across scheduled GitHub Actions runs. Kept well below the
# real quota — see pipeline/research.py for the enforcement logic.
TAVILY_MONTHLY_CALL_CAP = int(os.environ.get("TAVILY_MONTHLY_CALL_CAP", "400"))
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

# Git-committed (not cache/, which is gitignored/ephemeral in CI -- every
# scheduled run starts from a fresh VM with an empty cache) record of
# recently-used Pexels asset IDs, so a new video's visual search can actively
# avoid clips shown in recent prior videos instead of only avoiding repeats
# within its own scenes (see pipeline/visuals.py). Configurable per-channel
# via env var, same reasoning as GOOGLE_TTS_USAGE_FILE below: auto-video.yml
# and auto-video-education.yml each independently commit+push their own
# usage file, so a shared file risks a real git conflict if both runs overlap.
VISUAL_USAGE_FILE = Path(os.environ.get("VISUAL_USAGE_FILE", str(ROOT / "visual_usage.json")))

# Git-committed (not cache/, which is gitignored/ephemeral in CI) join table
# of real uploaded video_id -> NSE symbol, keyed off the exact video_id
# returned by a successful YouTube upload -- see pipeline/performance_log.py.
COMPANY_PERFORMANCE_LOG_FILE = ROOT / "company_performance_log.json"

# Optional small "@handle"-style corner watermark burned into every video
# (see pipeline/assemble.py / pipeline/textcard.render_watermark) -- off by
# default (no-op) since we don't want to fabricate a channel handle; set this
# to the channel's real @handle to turn it on.
CHANNEL_WATERMARK_TEXT = os.environ.get("CHANNEL_WATERMARK_TEXT", "")

# Narration voice — Microsoft Edge neural TTS (free, no API key).
# en-IN-PrabhatNeural / en-IN-NeerjaNeural give an Indian-English accent;
# run `edge-tts --list-voices` for the full catalog.
TTS_VOICE = os.environ.get("TTS_VOICE", "en-IN-PrabhatNeural")

# TTS engine selection. "edge" (default) is free/no-key but talks to an
# unofficial Microsoft endpoint that blocks GitHub Actions' published IP
# ranges outright -- "google" uses Google Cloud Text-to-Speech's officially
# authenticated REST API instead (set TTS_ENGINE=google + GOOGLE_TTS_API_KEY
# in CI only; Render/local keep using edge-tts unchanged since this env var
# is unset there).
TTS_ENGINE = os.environ.get("TTS_ENGINE", "edge")
GOOGLE_TTS_API_KEY = os.environ.get("GOOGLE_TTS_API_KEY", "")
GOOGLE_TTS_VOICE = os.environ.get("GOOGLE_TTS_VOICE", "en-IN-Neural2-B")  # male en-IN voice

# Google bills WaveNet/Neural2 voices (like GOOGLE_TTS_VOICE above) past a
# 1,000,000-characters/month free tier. Each channel's workflow tracks its own
# usage file (set via GOOGLE_TTS_USAGE_FILE) rather than sharing one, since
# auto-video.yml and auto-video-education.yml each independently commit+push
# their own usage updates and a shared file risks a real git conflict if both
# runs overlap. Each channel's cap is kept well under half the free tier so
# the combined total across both files never tips into billed usage even if
# a run retries a few times -- see pipeline/tts.py's usage-counter guardrail.
GOOGLE_TTS_USAGE_FILE = Path(os.environ.get("GOOGLE_TTS_USAGE_FILE", str(ROOT / "google_tts_usage.json")))
GOOGLE_TTS_MONTHLY_CHAR_CAP = int(os.environ.get("GOOGLE_TTS_MONTHLY_CHAR_CAP", "400000"))

# Extra headroom kept below GOOGLE_TTS_MONTHLY_CHAR_CAP itself, so the
# whole-video preflight check (pipeline/tts.ensure_google_tts_budget) stops a
# scheduled run a little before the cap rather than exactly at it -- avoids
# flip-flopping between "just under" and "just over" across retried runs.
GOOGLE_TTS_SAFETY_BUFFER_CHARS = int(os.environ.get("GOOGLE_TTS_SAFETY_BUFFER_CHARS", "5000"))

# Gemini 2.5 Flash Image ("Nano Banana") — Google AI Studio's free-tier
# image-generation model, used as the primary AI thumbnail/hook-background
# generator (see pipeline/ai_thumbnail.py) since it renders legible on-screen
# text/numbers far more reliably than Pollinations.ai. Get a free key at
# https://aistudio.google.com/apikey. Optional: if unset, ai_thumbnail.py
# skips straight to the Pollinations.ai fallback, same as before this existed.
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
GEMINI_IMAGE_MODEL = os.environ.get("GEMINI_IMAGE_MODEL", "gemini-2.5-flash-image")

# Google AI Studio's free tier is rate-limited per day, not metered by
# cost -- tracked the same way as GOOGLE_TTS_USAGE_FILE above (a git-committed
# counter, since cache/ is gitignored/ephemeral in CI), so a preflight check
# can proactively fall back to Pollinations.ai BEFORE actually hitting
# Gemini's real daily limit, rather than only reacting to a 429 error.
# Per-channel file (see VISUAL_USAGE_FILE comment above for why: each of
# auto-video.yml / auto-video-education.yml commits+pushes its own file
# independently, so sharing one risks a real git conflict).
GEMINI_IMAGE_USAGE_FILE = Path(os.environ.get("GEMINI_IMAGE_USAGE_FILE", str(ROOT / "gemini_image_usage.json")))

# Kept comfortably under the real free-tier daily request limit so a run that
# retries a few times (or the other channel's workflow sharing the same key)
# never risks tipping into billed usage.
GEMINI_IMAGE_DAILY_CAP = int(os.environ.get("GEMINI_IMAGE_DAILY_CAP", "40"))
GEMINI_IMAGE_SAFETY_BUFFER = int(os.environ.get("GEMINI_IMAGE_SAFETY_BUFFER", "5"))

# Cloudflare Workers AI — second-tier free image generator (see
# pipeline/ai_thumbnail.py), tried when Gemini is unconfigured, over its
# daily cap, or fails outright, before finally falling back to Pollinations.ai.
# No credit card required to start. Get CLOUDFLARE_ACCOUNT_ID from the
# Cloudflare dashboard's Workers & Pages overview page (right-hand sidebar),
# and CLOUDFLARE_API_TOKEN from My Profile -> API Tokens -> Create Token
# (the "Workers AI" template, or a custom token scoped to "Account - Workers
# AI - Edit"). Optional: if either is unset, ai_thumbnail.py skips straight
# from Gemini to Pollinations, same as before this existed.
CLOUDFLARE_ACCOUNT_ID = os.environ.get("CLOUDFLARE_ACCOUNT_ID", "")
CLOUDFLARE_API_TOKEN = os.environ.get("CLOUDFLARE_API_TOKEN", "")
CLOUDFLARE_IMAGE_MODEL = os.environ.get("CLOUDFLARE_IMAGE_MODEL", "@cf/black-forest-labs/flux-1-schnell")

# Same git-committed daily-counter pattern as GEMINI_IMAGE_USAGE_FILE above.
# Workers AI's free plan grants a real daily Neurons allowance rather than
# metering by cost -- kept conservative here since each model's
# Neurons-per-image cost differs; check the Cloudflare dashboard's AI
# Analytics tab and raise this if there's clear headroom.
CLOUDFLARE_IMAGE_USAGE_FILE = Path(os.environ.get("CLOUDFLARE_IMAGE_USAGE_FILE", str(ROOT / "cloudflare_image_usage.json")))
CLOUDFLARE_IMAGE_DAILY_CAP = int(os.environ.get("CLOUDFLARE_IMAGE_DAILY_CAP", "40"))
CLOUDFLARE_IMAGE_SAFETY_BUFFER = int(os.environ.get("CLOUDFLARE_IMAGE_SAFETY_BUFFER", "5"))


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
# Optional: auto-add every uploaded video to this playlist ID (see
# pipeline/youtube_upload.add_to_playlist) -- off by default since we don't
# want to guess/fabricate a playlist ID; set this to a real playlist ID
# (visible in its YouTube Studio URL) to turn it on.
YOUTUBE_PLAYLIST_ID = os.environ.get("YOUTUBE_PLAYLIST_ID", "")

# Optional background-music bed mixed in quietly under the narration -- off by
# default (no-op) since this requires the user to supply their own
# royalty-free/licensed audio loops (we won't fabricate/bundle audio assets);
# point this at a directory of .mp3/.wav/.m4a loops to turn it on. A track is
# picked at random per video so a run of videos doesn't all sound identical.
MUSIC_BED_DIR = Path(os.environ["MUSIC_BED_DIR"]) if os.environ.get("MUSIC_BED_DIR") else None
MUSIC_BED_VOLUME = float(os.environ.get("MUSIC_BED_VOLUME", "0.12"))

for d in (OUTPUT_DIR, CACHE_DIR):
    d.mkdir(parents=True, exist_ok=True)
