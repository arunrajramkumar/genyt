#!/bin/bash
set -e

ollama serve &

until curl -s http://localhost:11434 >/dev/null 2>&1; do
  echo "Waiting for Ollama to start..."
  sleep 1
done
echo "Ollama is up."

exec python3 webapp.py
