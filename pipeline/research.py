"""Fetches real, sourced facts for freeform themes that plausibly need current
real-world data (e.g. named-company stock analysis) via Tavily's search API,
so the LLM script writer is grounded on real information instead of inventing
figures. Groq still writes every script -- this only supplies it real facts
to work from when the theme needs them.

Guarded two ways against running past Tavily's free credit quota:
  1. A live check against Tavily's own GET /usage endpoint -- the authoritative
     source of truth for actual remaining credits this billing cycle.
  2. A local, git-committed call counter (config.TAVILY_MONTHLY_CALL_CAP) that
     still applies if the live check itself is unreachable/fails, so a network
     hiccup in the guardrail can never turn into unbounded spend.
Either guard tripping skips the call (not charged) and the caller falls back
to the "don't invent numbers" instruction.
"""
import json
from datetime import datetime, timezone

import requests

from . import config

TAVILY_SEARCH_URL = "https://api.tavily.com/search"
TAVILY_USAGE_URL = "https://api.tavily.com/usage"

# Keeps the search call opt-in to themes that plausibly reference verifiable
# real-world facts -- generic/creative topics skip this and go straight to
# the existing freeform path unchanged.
GROUNDING_KEYWORDS = (
    "stock", "share price", "shares", "ipo", "nse:", "bse:", "nse ", "bse ",
    "dividend", "quarterly result", "earnings", "market cap", "revenue",
    " ltd", "limited", "nifty", "sensex", "mutual fund",
)


def needs_grounding(theme: str) -> bool:
    """True if the theme plausibly makes verifiable real-world claims (named
    companies, stock/financial data) that must not be left to the model's own
    (fabrication-prone) "knowledge"."""
    lowered = theme.lower()
    return any(keyword in lowered for keyword in GROUNDING_KEYWORDS)


def _current_month() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m")


def _load_usage() -> dict:
    try:
        data = json.loads(config.TAVILY_USAGE_FILE.read_text())
    except (FileNotFoundError, ValueError):
        data = {}
    if data.get("month") != _current_month():
        return {"month": _current_month(), "calls": 0}
    return data


def _record_call() -> None:
    usage = _load_usage()
    usage["calls"] += 1
    config.TAVILY_USAGE_FILE.write_text(json.dumps(usage))


def _under_monthly_cap() -> bool:
    return _load_usage()["calls"] < config.TAVILY_MONTHLY_CALL_CAP


def _fetch_remaining_credits():
    """Queries Tavily's own GET /usage for the real remaining credit balance
    this billing cycle. Returns None if the check itself fails (auth error,
    network issue, rate limit, or the account reports no fixed plan_limit) --
    callers must treat None as "unknown" and fall back to the local call-count
    guardrail rather than block indefinitely on an unreachable check."""
    try:
        resp = requests.get(
            TAVILY_USAGE_URL,
            headers={"Authorization": f"Bearer {config.TAVILY_API_KEY}"},
            timeout=10,
        )
        resp.raise_for_status()
        data = resp.json()
    except (requests.RequestException, ValueError):
        return None

    account = data.get("account", {})
    if account.get("paygo_usage"):
        return 0  # any pay-as-you-go overage usage is a hard stop signal
    plan_limit = account.get("plan_limit")
    if plan_limit is None:
        return None
    return max(0, plan_limit - (account.get("plan_usage") or 0))


def fetch_grounding_facts(topic: str) -> str:
    """Returns a block of sourced facts/snippets about topic from a live web
    search, or "" if Tavily isn't configured, either credit guardrail has
    tripped, or the search fails -- callers must treat "" as "no verified data
    available" and must not fabricate specifics in its place."""
    if not config.TAVILY_API_KEY or not _under_monthly_cap():
        return ""

    remaining = _fetch_remaining_credits()
    if remaining is not None and remaining < config.TAVILY_SAFETY_BUFFER_CREDITS:
        return ""

    _record_call()  # count the attempt before the request -- Tavily bills on
    # receipt, so an undercount here would risk the cap never actually biting.
    try:
        resp = requests.post(
            TAVILY_SEARCH_URL,
            json={
                "api_key": config.TAVILY_API_KEY,
                "query": topic,
                "search_depth": "basic",  # 1 credit/call instead of "advanced"'s 2
                "max_results": 5,
                "include_answer": False,
            },
            timeout=30,
        )
        resp.raise_for_status()
        results = resp.json().get("results", [])
    except (requests.RequestException, ValueError):
        return ""

    lines = [
        f"- {r.get('title', '').strip()}: {r.get('content', '').strip()} (source: {r.get('url', '').strip()})"
        for r in results if r.get("content", "").strip()
    ]
    return "\n".join(lines)

