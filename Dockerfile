# Docker image for running the ytagent pipeline on Hugging Face Spaces (Docker SDK).
# Bakes the Ollama model into the image at build time so a Space restart doesn't
# have to re-download ~2GB every time it wakes from sleep.
FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    fonts-noto-core \
    curl \
    ca-certificates \
    zstd \
    procps \
    && rm -rf /var/lib/apt/lists/*

RUN curl -fsSL https://ollama.com/install.sh | sh

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ARG OLLAMA_MODEL=llama3.2
RUN (ollama serve &) && \
    sleep 5 && \
    ollama pull ${OLLAMA_MODEL} && \
    pkill ollama && \
    sleep 1

RUN chmod +x start.sh

ENV OLLAMA_MODEL=${OLLAMA_MODEL}
EXPOSE 7860

CMD ["./start.sh"]
