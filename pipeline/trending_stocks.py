"""Discovers which NSE-listed companies are currently trending in financial
news, so scripts/build_finance_queue.py can fill the finance queue
automatically instead of requiring a hand-curated --symbols list every time.

Reuses pipeline/research.py's already quota-guarded Tavily search call (no
new API budget/key needed) to pull real, dated snippets about the Indian
stock market, then asks the LLM to name only companies explicitly present in
those snippets -- the same no-fabrication discipline as the rest of this
pipeline. Every surviving candidate is then independently verified against
pipeline/fundamentals.py's live Screener.in fetch before it's ever queued: a
symbol that doesn't resolve to a real, scrapable company page is silently
dropped rather than queued with made-up data.
"""
import json
import re

from . import fundamentals, llm, research

# Two generic queries alone tend to surface the same handful of large,
# already-everywhere names every day, which then dominate the market-cap
# ranking in select_trending_symbols() by default -- adding a couple of
# sector-focused queries widens the candidate pool so that ranking actually
# has a real choice to make, instead of re-picking yesterday's headline name.
# Tavily usage has ample headroom (see tavily_usage.json / the real dashboard
# quota) to absorb 2 extra calls per trending run.
TRENDING_QUERIES = [
    "NSE stocks in the news India today",
    "most talked about Indian stocks this week",
    "NSE banking and IT stocks news today",
    "NSE auto and pharma stocks news today",
]

SYSTEM_PROMPT_TRENDING = """You are a financial news analyst. You will be given real
search-result snippets about the Indian stock market. Identify NSE-listed companies that
are EXPLICITLY named in the snippets below -- never a company you merely happen to know
about from elsewhere.

Output ONLY valid JSON (no markdown fences, no commentary):

{"candidates": [{"symbol": "NSE_TICKER", "name": "Company name exactly as it appears in the text"}]}

Rules:
- Only include a company whose name literally appears in the snippets below.
- "symbol" is the company's standard NSE ticker (e.g. "TCS", "RELIANCE") -- a wrong guess
  here is fine, it gets independently verified afterwards, but never invent a company that
  isn't named in the text.
- If no real companies are named in the snippets, return {"candidates": []}.
"""


def _extract_json(text: str) -> dict:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(json)?", "", text).rstrip("`").strip()
    if not text.startswith("{"):
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            text = match.group(0)
    return json.loads(text)


def _search_trending_stock_news(queries: list = None) -> str:
    parts = [research._fetch_tavily_facts(q) for q in (queries or TRENDING_QUERIES)]
    return "\n".join(p for p in parts if p)


def _extract_candidates(facts_text: str) -> list:
    """Returns [{symbol, name}, ...]. Grounded independently of what the LLM
    claims: a candidate only survives if its name literally appears in
    facts_text, so a hallucinated company can't sneak through even if the
    model ignores the system prompt."""
    if not facts_text.strip():
        return []

    user_prompt = "Search snippets:\n" + facts_text + "\n\nProduce the JSON now."
    candidates = []
    for _attempt in range(2):
        raw = llm.chat_json(SYSTEM_PROMPT_TRENDING, user_prompt, max_tokens=400, timeout=60)
        try:
            candidates = _extract_json(raw).get("candidates") or []
            break
        except (ValueError, json.JSONDecodeError):
            continue
    else:
        return []

    lowered_text = facts_text.lower()
    grounded = []
    for c in candidates:
        name = (c.get("name") or "").strip()
        symbol = (c.get("symbol") or "").strip().upper()
        if name and symbol and name.lower() in lowered_text:
            grounded.append({"symbol": symbol, "name": name})
    return grounded


def select_trending_symbols(exclude_symbols=(), limit: int = 5, queries: list = None) -> list:
    """Best-effort list of up to `limit` currently-trending, verified-real NSE
    symbols not already in exclude_symbols: [{symbol, name}, ...]. Returns []
    on any failure (Tavily not configured/quota exhausted, LLM failure, no
    companies found) -- callers must treat [] as "no trending candidates
    available right now" and fall back to requiring a manual --symbols list,
    never queue a theme with no real candidate behind it.

    Ranked by market cap (largest first) among all verified candidates rather
    than taking the first `limit` in whatever order the LLM happened to list
    them -- a bigger, more widely-recognized company is a reasonable free
    proxy for viewer search/click interest, and the figure is already scraped
    as part of the existing fundamentals verification step (no new API call)."""
    exclude = {s.strip().upper() for s in exclude_symbols}
    try:
        facts_text = _search_trending_stock_news(queries)
        candidates = _extract_candidates(facts_text)
    except Exception:
        return []

    verified, seen = [], set()
    for c in candidates:
        if c["symbol"] in exclude or c["symbol"] in seen:
            continue
        seen.add(c["symbol"])
        data = fundamentals.fetch_fundamentals(c["symbol"])
        if data:
            verified.append((c, fundamentals.market_cap_cr(data)))

    verified.sort(key=lambda pair: pair[1], reverse=True)
    return [c for c, _market_cap in verified[:limit]]
