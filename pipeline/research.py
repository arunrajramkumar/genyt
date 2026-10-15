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
import re
from datetime import datetime, timezone

import requests

from . import config, fundamentals

TAVILY_SEARCH_URL = "https://api.tavily.com/search"
TAVILY_USAGE_URL = "https://api.tavily.com/usage"

NSE_SYMBOL_RE = re.compile(r"NSE:\s*([A-Z0-9&\-]+)")

# Keeps the search call opt-in to themes that plausibly reference verifiable
# real-world facts -- generic/creative topics skip this and go straight to
# the existing freeform path unchanged.
GROUNDING_KEYWORDS = (
    "stock", "share price", "shares", "ipo", "nse:", "bse:", "nse ", "bse ",
    "dividend", "quarterly result", "earnings", "market cap", "revenue",
    " ltd", "limited", "nifty", "sensex", "mutual fund",
)

# Common topic-framing words that show up in generic/creative themes ("5 Facts
# About Octopuses", "History of the Roman Empire") but never in a bare company
# name -- used to tell the two apart below.
_TOPIC_FRAMING_WORDS = {
    "a", "an", "the", "of", "in", "on", "to", "for", "with", "and", "or",
    "about", "how", "why", "what", "when", "where", "who", "which",
    "is", "are", "was", "were", "explain", "describe", "learn",
    "history", "historical", "fact", "facts", "guide", "tips",
    "way", "ways", "thing", "things", "reason", "reasons",
    "secret", "secrets", "trick", "tricks", "basics", "introduction", "intro",
}


def _looks_like_bare_company_name(theme: str) -> bool:
    """True for short, keyword-free, Title-Case phrases like "KPI Green
    Energy" -- an ad-hoc prompt that's just a company name, with none of the
    "Ltd"/"stock"/"NSE:" markers GROUNDING_KEYWORDS looks for, but still a
    named real-world entity whose figures must not be left to the model's own
    guesswork."""
    words = theme.strip().split()
    if not (1 < len(words) <= 5):
        return False
    lowered_words = [w.strip(".,!?").lower() for w in words]
    if any(w in _TOPIC_FRAMING_WORDS for w in lowered_words):
        return False
    return all(w[0].isupper() for w in words if w[:1].isalpha())


def needs_grounding(theme: str) -> bool:
    """True if the theme plausibly makes verifiable real-world claims (named
    companies, stock/financial data) that must not be left to the model's own
    (fabrication-prone) "knowledge"."""
    lowered = theme.lower()
    if any(keyword in lowered for keyword in GROUNDING_KEYWORDS):
        return True
    return _looks_like_bare_company_name(theme)


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


def _extract_nse_symbol(theme: str):
    """Pulls a literal "NSE: SYMBOL" ticker out of a theme, when the theme
    names one explicitly (see themes.yaml's company-specific entries)."""
    m = NSE_SYMBOL_RE.search(theme.upper())
    return m.group(1) if m else None


def _fetch_tavily_facts(topic: str) -> str:
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


def fetch_grounding_facts(topic: str) -> str:
    """Returns a block of sourced facts about topic, or "" if nothing could be
    verified -- callers must treat "" as "no verified data available" and must
    not fabricate specifics in its place.

    When the topic names an NSE-listed company (e.g. "NSE: CDSL"), real
    scraped fundamentals (P/E, ROE, ROCE, revenue/profit growth -- see
    pipeline/fundamentals.py) are fetched first and take priority, since
    they're precise numbers rather than search-snippet prose. Tavily's
    web-search snippets are appended after for qualitative/news context that
    fundamentals data alone doesn't cover (no API key needed for the
    fundamentals half, so this works even without Tavily configured)."""
    parts = []

    symbol = _extract_nse_symbol(topic)
    if symbol:
        facts = fundamentals.format_fundamentals_facts(fundamentals.fetch_fundamentals(symbol))
        if facts:
            parts.append(facts)

    tavily_facts = _fetch_tavily_facts(topic)
    if tavily_facts:
        parts.append(tavily_facts)

    return "\n".join(parts)

