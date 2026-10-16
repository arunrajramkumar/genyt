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
Either guard tripping skips the call (not charged) and search_web() falls back
to scraping DuckDuckGo's plain HTML results page instead -- free, no API key,
no credit limit, so a dry Tavily account for the rest of the month doesn't
turn into "no grounding at all" for every theme after that. Only if both come
back empty does the caller fall back to the "don't invent numbers" instruction.
"""
import json
import re
from datetime import datetime, timezone
from urllib.parse import parse_qs, unquote, urlparse

import requests

from . import config, fundamentals, segmented_script

TAVILY_SEARCH_URL = "https://api.tavily.com/search"
TAVILY_USAGE_URL = "https://api.tavily.com/usage"

# DuckDuckGo's plain (JS-free) HTML results page -- undocumented but stable
# and widely used for exactly this (no official API, no key, no credit cap).
# Used only as a fallback once Tavily's guardrails trip, not as the primary
# source, since Tavily's results are cleaner and the DDG HTML layout could
# change without notice.
DUCKDUCKGO_HTML_URL = "https://html.duckduckgo.com/html/"
DUCKDUCKGO_USER_AGENT = "Mozilla/5.0 (compatible; ytagent-research/1.0)"
_DDG_RESULT_RE = re.compile(
    r'<a rel="nofollow" class="result__a" href="(.*?)">(.*?)</a>.*?'
    r'<a class="result__snippet"[^>]*>(.*?)</a>',
    re.DOTALL,
)

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


# Generic finance/video-title words that often sit right next to a company
# name in a freeform theme ("Pidilite Industries - Stock Analysis...", "Is
# Reliance Industries Ltd a good dividend stock?") but are never themselves
# part of the name -- used to trim a candidate name down to just the company,
# so the Screener.in name search below isn't polluted by trailing noise.
_GENERIC_FINANCE_WORDS = {
    "stock", "stocks", "shares", "share", "analysis", "review", "ltd", "limited",
    "ipo", "dividend", "quarterly", "results", "earnings", "price", "target",
    "buy", "sell", "hold", "growth", "wealth", "long", "term", "creation",
    "builder", "compounder", "valuation", "report", "update", "news", "outlook",
    "forecast", "crash", "rally", "rating", "rated", "investment", "investor",
    "investors", "market", "markets", "cap", "capitalisation", "capitalization",
    "fundamental", "fundamentals", "breakdown", "today", "week", "month", "year",
}


def _extract_company_name_candidate(theme: str):
    """Finds the longest run of consecutive capitalized, non-generic words in
    a freeform theme -- e.g. pulls "Pidilite Industries" out of "Pidilite
    Industries - Stock Analysis for Long Term Wealth Creation" or "Reliance
    Industries" out of "Is Reliance Industries Ltd a good dividend stock?".
    Returns "" when nothing plausible is found. This is only ever a *search
    query* for _resolve_company_symbol below, never trusted as a symbol by
    itself -- a wrong/too-broad guess here just means Screener's search comes
    back empty or with an unrelated company, which fetch_fundamentals then
    either confirms or silently drops."""
    best, current = [], []
    for word in re.findall(r"[A-Za-z0-9&]+", theme):
        lowered = word.lower()
        qualifies = word[0].isupper() and lowered not in _TOPIC_FRAMING_WORDS and lowered not in _GENERIC_FINANCE_WORDS
        if qualifies:
            current.append(word)
            if len(current) > len(best):
                best = current
        else:
            current = []
    candidate = " ".join(best)
    return candidate if len(candidate) >= 3 else ""


def _resolve_company_symbol(theme: str):
    """When a theme names a real company by plain name rather than a literal
    "NSE: SYMBOL" marker, resolves it to a real ticker via Screener.in's free
    company-search endpoint (fundamentals.search_company_symbol), verified by
    requiring fetch_fundamentals on the resolved symbol to return real data
    before it's trusted -- the same verify-then-trust pattern
    trending_stocks.select_trending_symbols uses for LLM-suggested tickers.
    Returns None when no company name can be extracted, Screener has no
    match, or the resolved symbol doesn't check out -- callers must fall back
    to web-search snippets rather than guess."""
    candidate = _extract_company_name_candidate(theme)
    if not candidate:
        return None
    symbol = fundamentals.search_company_symbol(candidate)
    return symbol or None


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
    guardrail rather than block indefinitely on an unreachable check.

    Also syncs the locally committed usage file to Tavily's own plan_usage
    whenever this succeeds. The local file is otherwise hand-incremented one
    call at a time (see _record_call), which silently drifts from the real
    number -- it never counts this very GET request, any call made during
    local testing whose updated file didn't get committed, or any call made
    by a concurrent run. Tavily's own account is the authoritative source of
    truth, so overwrite the local figure with it rather than let the two
    diverge indefinitely."""
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
    plan_usage = account.get("plan_usage")
    if plan_usage is not None:
        config.TAVILY_USAGE_FILE.write_text(json.dumps({"month": _current_month(), "calls": plan_usage}))
    if account.get("paygo_usage"):
        return 0  # any pay-as-you-go overage usage is a hard stop signal
    plan_limit = account.get("plan_limit")
    if plan_limit is None:
        return None
    return max(0, plan_limit - (plan_usage or 0))


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


def _clean_ddg_fragment(html: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"<[^>]+>", "", html)).strip()


def _resolve_ddg_url(href: str) -> str:
    """DDG's HTML results link through its own redirect
    (`//duckduckgo.com/l/?uddg=<encoded-real-url>&...`) rather than the real
    URL directly -- unwrap it so the cited source is the actual article."""
    if href.startswith("//"):
        href = "https:" + href
    uddg = parse_qs(urlparse(href).query).get("uddg")
    return unquote(uddg[0]) if uddg else href


def _fetch_duckduckgo_facts(topic: str) -> str:
    """Free, no-API-key, no-credit-limit fallback for when Tavily's quota/
    credit guardrail has tripped (or no TAVILY_API_KEY is configured at all)
    -- scrapes DuckDuckGo's plain HTML results page the same way
    pipeline/fundamentals.py scrapes Screener.in, so grounding doesn't go
    dark for the rest of the month just because one paid-tier search
    provider ran dry. Best-effort: any failure returns "", same contract as
    _fetch_tavily_facts, so callers still fall back to "don't invent
    numbers" if even this fails."""
    try:
        resp = requests.post(
            DUCKDUCKGO_HTML_URL,
            data={"q": topic, "kl": "in-en"},
            headers={"User-Agent": DUCKDUCKGO_USER_AGENT},
            timeout=15,
        )
        resp.raise_for_status()
    except requests.RequestException:
        return ""

    lines = []
    for href, title_html, snippet_html in _DDG_RESULT_RE.findall(resp.text)[:5]:
        title = _clean_ddg_fragment(title_html)
        snippet = _clean_ddg_fragment(snippet_html)
        if title and snippet:
            lines.append(f"- {title}: {snippet} (source: {_resolve_ddg_url(href)})")
    return "\n".join(lines)


def search_web(topic: str) -> str:
    """Sourced web snippets about topic, preferring Tavily and falling back to
    free DuckDuckGo scraping once Tavily's quota/credit guardrail trips (or
    if no TAVILY_API_KEY is configured at all) -- so a dry Tavily account
    doesn't silently turn every subsequent theme this month into
    no-grounding-available. Returns "" if both come back empty; callers must
    treat that as "no verified data available" and not fabricate in its
    place."""
    return _fetch_tavily_facts(topic) or _fetch_duckduckgo_facts(topic)


def fetch_grounding_facts(topic: str) -> str:
    """Returns a block of sourced facts about topic, or "" if nothing could be
    verified -- callers must treat "" as "no verified data available" and must
    not fabricate specifics in its place.

    When the topic names an NSE-listed company -- either explicitly (e.g.
    "NSE: CDSL") or just by plain company name (e.g. "Pidilite Industries -
    Stock Analysis...", resolved via _resolve_company_symbol) -- real scraped
    fundamentals (P/E, ROE, ROCE, revenue/profit growth -- see
    pipeline/fundamentals.py) are fetched first and take priority, since
    they're live, current numbers rather than search-snippet prose that can
    be stale (e.g. pre-stock-split prices/ratios a web search might still
    surface long after the split). Web search snippets (Tavily, falling back
    to free DuckDuckGo scraping -- see search_web()) are appended after for
    qualitative/news context that fundamentals data alone doesn't cover (no
    API key needed for the fundamentals half, so this works even without
    Tavily configured)."""
    parts = []

    symbol = _extract_nse_symbol(topic) or _resolve_company_symbol(topic)
    if symbol:
        facts = fundamentals.format_fundamentals_facts(fundamentals.fetch_fundamentals(symbol))
        if facts:
            parts.append(facts)

    web_facts = search_web(topic)
    if web_facts:
        parts.append(web_facts)

    return segmented_script.normalize_indian_currency_units("\n".join(parts))

