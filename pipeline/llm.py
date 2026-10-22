"""Thin wrapper around Groq's free-tier hosted chat API (OpenAI-compatible).

Used instead of a locally-run Ollama model so the app fits inside free
hosting tiers that don't have the RAM to run a local LLM alongside ffmpeg.
"""
import re
import time

import requests

from . import config

GROQ_CHAT_URL = "https://api.groq.com/openai/v1/chat/completions"
MAX_TOKENS_CAP = 4096
MAX_RATE_LIMIT_RETRIES = 3
MAX_JSON_VALIDATE_RETRIES = 3
MAX_SERVER_ERROR_RETRIES = 2

_RETRY_AFTER_MESSAGE_RE = re.compile(r"try again in ([\d.]+)s")


class LLMResponseError(ValueError):
    """The model produced no usable reply (JSON validation kept failing, or the
    content came back empty). A ValueError subclass so every caller's existing
    "malformed response -> resample" retry loop treats it like any other bad
    generation instead of crashing the run."""


def _is_json_validate_error(resp) -> bool:
    """openai/gpt-oss-120b is a reasoning model: even with reasoning_effort
    'low', it can occasionally still burn the whole max_tokens budget on
    hidden chain-of-thought for a fact-dense prompt and return this specific
    400 instead of the JSON body (observed with an empty "failed_generation"
    even at 2048 tokens) -- retryable with a bigger token budget. It can also
    (observed in production, e.g. a "tags" array collapsed into one string
    with mis-escaped quotes) return the exact same 400 for a generation that
    completed fine but just isn't valid JSON -- not a budget problem, but
    usually resolved by a plain resample."""
    try:
        return resp.json().get("error", {}).get("code") == "json_validate_failed"
    except (ValueError, AttributeError):
        return False


def _post(system_prompt: str, user_prompt: str, max_tokens: int, timeout: int, json_mode: bool = True):
    body = {
        "model": config.GROQ_MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "max_tokens": max_tokens,
        "reasoning_effort": "low",
    }
    if json_mode:
        body["response_format"] = {"type": "json_object"}
    try:
        return requests.post(
            GROQ_CHAT_URL,
            headers={"Authorization": f"Bearer {config.GROQ_API_KEY}"},
            json=body,
            timeout=timeout,
        )
    except (requests.ConnectionError, requests.Timeout) as e:
        raise RuntimeError(f"Could not reach the Groq API: {e}") from e


def _rate_limit_delay_seconds(resp, attempt: int) -> float:
    """Groq's 429 body tells you almost exactly how long to wait -- prefer
    that over a blind guess so a transient per-minute token-budget trip (e.g.
    "Used 2578, Requested 5617... try again in 1.4625s") doesn't fail the
    whole run when it would have succeeded a couple seconds later."""
    retry_after = resp.headers.get("Retry-After") if hasattr(resp, "headers") else None
    if retry_after:
        try:
            return float(retry_after) + 0.5
        except (TypeError, ValueError):
            pass
    try:
        message = resp.json().get("error", {}).get("message", "")
        match = _RETRY_AFTER_MESSAGE_RE.search(message)
        if match:
            return float(match.group(1)) + 0.5
    except (ValueError, AttributeError):
        pass
    return 2 ** attempt


def chat_json(system_prompt: str, user_prompt: str, max_tokens: int, timeout: int) -> str:
    """Returns the raw JSON-formatted string content of the model's reply.

    Handles every retryable failure in one loop, in whatever order they
    arrive: json_validate_failed (bigger budget, and a last try without
    Groq's JSON mode -- callers already extract the JSON object from free
    text), 429 rate limits (wait as told), and 5xx/over-capacity (back off).
    Raises LLMResponseError (a ValueError) when no usable reply comes back,
    RuntimeError for configuration/HTTP errors that resampling can't fix."""
    if not config.GROQ_API_KEY:
        raise RuntimeError(
            "GROQ_API_KEY is not set. Get a free key at https://console.groq.com/keys "
            "and set it as an environment variable (or a Space secret)."
        )
    json_mode = True
    json_retries = rate_retries = server_retries = 0
    while True:
        resp = _post(system_prompt, user_prompt, max_tokens, timeout, json_mode=json_mode)
        if resp.status_code == 400 and _is_json_validate_error(resp):
            if json_retries >= MAX_JSON_VALIDATE_RETRIES:
                raise LLMResponseError(f"Groq JSON validation kept failing: {resp.text[:300]}")
            json_retries += 1
            max_tokens = max(max_tokens, min(max_tokens * 2, MAX_TOKENS_CAP))
            if json_retries == MAX_JSON_VALIDATE_RETRIES:
                json_mode = False
            continue
        if resp.status_code == 429 and rate_retries < MAX_RATE_LIMIT_RETRIES:
            time.sleep(_rate_limit_delay_seconds(resp, rate_retries))
            rate_retries += 1
            continue
        if resp.status_code >= 500 and server_retries < MAX_SERVER_ERROR_RETRIES:
            time.sleep(2 ** server_retries)
            server_retries += 1
            continue
        break
    try:
        resp.raise_for_status()
    except requests.HTTPError as e:
        raise RuntimeError(f"Groq API error {resp.status_code}: {resp.text}") from e
    content = resp.json()["choices"][0]["message"].get("content") or ""
    if not content.strip():
        raise LLMResponseError("Groq returned an empty reply")
    return content
