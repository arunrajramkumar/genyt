# Docker image for running the ytagent pipeline on Render's free web service tier.
# Script generation is delegated to Groq's hosted API (see pipeline/llm.py) rather
# than a local LLM, so the image doesn't need to bundle/run Ollama — this keeps it
# small enough to fit free hosting tiers with limited RAM (Render's free tier is
# ~512MB).
#
# Render sets $PORT at runtime; webapp.py already binds to that if present
# (falls back to 7860 for local use).
FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    fonts-noto-core \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 7860

CMD ["python3", "webapp.py"]
