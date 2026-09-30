"""Thin wrapper around Groq's free-tier hosted chat API (OpenAI-compatible).

Used instead of a locally-run Ollama model so the app fits inside free
hosting tiers that don't have the RAM to run a local LLM alongside ffmpeg.
"""
import requests

from . import config

GROQ_CHAT_URL = "https://api.groq.com/openai/v1/chat/completions"


def chat_json(system_prompt: str, user_prompt: str, max_tokens: int, timeout: int) -> str:
    """Returns the raw JSON-formatted string content of the model's reply."""
    if not config.GROQ_API_KEY:
        raise RuntimeError(
            "GROQ_API_KEY is not set. Get a free key at https://console.groq.com/keys "
            "and set it as an environment variable (or a Space secret)."
        )
    try:
        resp = requests.post(
            GROQ_CHAT_URL,
            headers={"Authorization": f"Bearer {config.GROQ_API_KEY}"},
            json={
                "model": config.GROQ_MODEL,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                "response_format": {"type": "json_object"},
                "max_tokens": max_tokens,
            },
            timeout=timeout,
        )
    except requests.ConnectionError as e:
        raise RuntimeError(f"Could not reach the Groq API: {e}") from e
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"]
