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
import random
import re

from . import fundamentals, llm, research

# Two generic queries alone tend to surface the same handful of large,
# already-everywhere names every day, which then dominate the market-cap
# ranking in select_trending_symbols() by default -- adding a couple of
# sector-focused queries widens the candidate pool so that ranking actually
# has a real choice to make, instead of re-picking yesterday's headline name.
# research.search_web() falls back to free DuckDuckGo scraping once Tavily's
# quota runs dry, so these extra queries never go unanswered even on a
# maxed-out Tavily month.
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
    parts = [research.search_web(q) for q in (queries or TRENDING_QUERIES)]
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


def _weighted_pick_without_replacement(pairs: list, k: int) -> list:
    """Picks up to k items from pairs=[(candidate, market_cap_cr), ...] via
    weighted random sampling (weight = market cap, so a bigger company is
    still more likely to be picked) without replacement, instead of always
    taking the strict top-k by market cap. Always taking the single largest
    company converges this channel onto the same handful of mega-caps every
    other finance channel already covers, day after day -- a real company a
    retail audience still recognizes but hasn't seen covered everywhere else
    is a better find than the n-th video about the same index heavyweight."""
    pool = list(pairs)
    chosen = []
    for _ in range(min(k, len(pool))):
        weights = [cap + 1 for _, cap in pool]  # +1 so an unparseable/zero market cap still has a nonzero chance
        pick = random.choices(pool, weights=weights, k=1)[0]
        chosen.append(pick)
        pool.remove(pick)
    return chosen


def select_trending_symbols(exclude_symbols=(), limit: int = 5, queries: list = None) -> list:
    """Best-effort list of up to `limit` currently-trending, verified-real NSE
    symbols not already in exclude_symbols: [{symbol, name}, ...]. Returns []
    on any failure (Tavily not configured/quota exhausted, LLM failure, no
    companies found) -- callers must treat [] as "no trending candidates
    available right now" and fall back to requiring a manual --symbols list,
    never queue a theme with no real candidate behind it.

    Picked via market-cap-weighted random sampling (without replacement) from
    among all verified candidates, rather than strictly the top `limit` by
    market cap -- a bigger, more widely-recognized company is a reasonable
    free proxy for viewer search/click interest (the figure is already
    scraped as part of the existing fundamentals verification step, no new
    API call), but always picking the single largest name every run means
    this channel never covers anything but the same mega-caps every other
    finance channel already covers."""
    exclude = {s.strip().upper() for s in exclude_symbols}
    try:
        facts_text = _search_trending_stock_news(queries)
        candidates = _extract_candidates(facts_text)
    except Exception as e:
        print(f"[trending] search/extraction failed: {type(e).__name__}: {e}")
        return []
    print(f"[trending] {len(facts_text)} chars of news snippets, {len(candidates)} grounded candidate(s)")

    verified, seen = [], set()
    excluded = unverified = 0
    for c in candidates:
        if c["symbol"] in exclude or c["symbol"] in seen:
            excluded += 1
            continue
        seen.add(c["symbol"])
        data = fundamentals.fetch_fundamentals(c["symbol"])
        if data:
            verified.append((c, fundamentals.market_cap_cr(data)))
        else:
            unverified += 1
    print(f"[trending] {excluded} already covered/duplicate, {unverified} failed Screener verification, {len(verified)} verified")

    chosen = _weighted_pick_without_replacement(verified, limit)
    chosen.sort(key=lambda pair: pair[1], reverse=True)
    return [c for c, _market_cap in chosen]
