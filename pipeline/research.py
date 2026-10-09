"""Fetches real, sourced facts for freeform themes that plausibly need current
real-world data (e.g. named-company stock analysis) via Tavily's search API,
so the LLM script writer is grounded on real information instead of inventing
figures. Groq still writes every script -- this only supplies it real facts
to work from when the theme needs them.
"""
import requests

from . import config

TAVILY_SEARCH_URL = "https://api.tavily.com/search"

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


def fetch_grounding_facts(topic: str) -> str:
    """Returns a block of sourced facts/snippets about topic from a live web
    search, or "" if Tavily isn't configured or the search fails -- callers
    must treat "" as "no verified data available" and must not fabricate
    specifics in its place."""
    if not config.TAVILY_API_KEY:
        return ""
    try:
        resp = requests.post(
            TAVILY_SEARCH_URL,
            json={
                "api_key": config.TAVILY_API_KEY,
                "query": topic,
                "search_depth": "advanced",
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
