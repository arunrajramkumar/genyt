# Docker image for running the ytagent pipeline on Render's free web service tier.
# Script generation is delegated to Groq's hosted API (see pipeline/llm.py) rather
# than a local LLM, so the image doesn't need to bundle/run Ollama — this keeps it
# small enough to fit free hosting tiers with limited RAM (Render's free tier is
# ~512MB).
#
# Render sets $PORT at runtime; webapp.py already binds to that if present
# (falls back to 7860 for local use).
FROM python:3.11-slim

# fonts-noto-extra adds Devanagari/Tamil/Telugu/Kannada/Malayalam/Bengali/
# Gujarati/Arabic coverage for on-screen text (narration in those languages
# already works via edge-tts regardless of fonts — this is just so the text
# overlay doesn't render as tofu boxes when the script is non-Latin). CJK
# (Chinese/Japanese) on-screen text isn't covered — fonts-noto-cjk is 150MB+,
# too heavy for this free-tier build.
#
# libraqm-dev (+ build-essential/pkg-config to compile against it) is needed
# because Tamil/Kannada/Telugu/Malayalam/Devanagari reorder vowel signs before
# their base consonant — plain freetype rendering (what PyPI's prebuilt Pillow
# wheel uses) draws those as a disconnected dotted-circle + glyph instead of
# the correct shaped letter. Only raqm-enabled Pillow (built from source below)
# does this shaping correctly.
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    fonts-noto-core \
    fonts-noto-extra \
    ca-certificates \
    build-essential \
    pkg-config \
    libraqm-dev \
    libjpeg62-turbo-dev \
    zlib1g-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
# Pillow is rebuilt from source (below) so it links against libraqm for correct
# non-Latin glyph shaping — the prebuilt PyPI wheel lacks raqm support.
#
# --no-binary=Pillow (NOT :all:) matters: :all: forces pip to also build
# Pillow's own build-time dependency pybind11 from source, whose build
# backend (scikit-build-core) then needs to build CMake from source too —
# a multi-minute compile that OOM-crashes on Render's free-tier build box.
# Scoping --no-binary to just Pillow lets pybind11/setuptools/wheel install
# from their normal prebuilt wheels.
#
# --no-build-isolation reuses the setuptools/wheel/pybind11 already
# installed above instead of pip fetching a fresh isolated build env for
# every deploy.
RUN pip install --no-cache-dir -r requirements.txt \
    && pip install --no-cache-dir --upgrade setuptools wheel pybind11 \
    && pip install --no-cache-dir --force-reinstall --no-binary=Pillow --no-build-isolation Pillow

COPY . .

EXPOSE 7860

CMD ["python3", "webapp.py"]
