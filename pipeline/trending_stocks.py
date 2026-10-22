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
    "Indian companies quarterly results announced today NSE",
    "NSE stocks in the news India today",
    "IPO opening this week India GMP subscription status",
    "NSE stock order win guidance announcement today",
    "most talked about Indian stocks this week",
    "Nifty midcap smallcap stocks results news today",
]

# Tried in order, only when every candidate from the primary queries is already
# covered (themes file or the real channel) -- so a day with only
# already-covered headline names widens the search instead of giving up.
EXTRA_QUERY_BATCHES = [
    ["NSE midcap stocks news today", "NSE FMCG and consumer stocks news today"],
    ["NSE energy and metals stocks news today", "NSE smallcap stocks in focus India"],
    ["NSE infrastructure and realty stocks news today", "NSE stocks hitting 52 week high India"],
]

_NAME_SUFFIX_RE = re.compile(r"\b(ltd|limited|inc|corp|corporation)\b\.?", re.IGNORECASE)

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
- Prefer companies with a fresh event in the snippets (quarterly results, IPO opening or
  listing, guidance, order win) over companies merely mentioned in passing, and list those
  first. Include IPO companies too (use the best-guess NSE ticker; unlisted ones get dropped
  by verification).
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


# Groq's free tier caps a request at 8000 tokens per minute (prompt + reply); six
# queries' worth of snippets blew past it (13970 requested), so extraction runs
# over chunks of at most this many characters (~3k tokens).
MAX_EXTRACT_CHARS = 9000


def _chunk_text(text: str, limit: int) -> list:
    """Splits on line boundaries into pieces of at most `limit` characters (a
    single longer line is hard-cut)."""
    chunks, current = [], ""
    for line in text.split("\n"):
        while len(line) > limit:
            if current:
                chunks.append(current)
                current = ""
            chunks.append(line[:limit])
            line = line[limit:]
        if current and len(current) + len(line) + 1 > limit:
            chunks.append(current)
            current = ""
        current = f"{current}\n{line}" if current else line
    if current.strip():
        chunks.append(current)
    return chunks


def _extract_candidates(facts_text: str) -> list:
    """Returns [{symbol, name}, ...]. Grounded independently of what the LLM
    claims: a candidate only survives if its name literally appears in
    facts_text, so a hallucinated company can't sneak through even if the
    model ignores the system prompt."""
    if not facts_text.strip():
        return []

    candidates = []
    for chunk in _chunk_text(facts_text, MAX_EXTRACT_CHARS):
        user_prompt = "Search snippets:\n" + chunk + "\n\nProduce the JSON now."
        for _attempt in range(2):
            raw = llm.chat_json(SYSTEM_PROMPT_TRENDING, user_prompt, max_tokens=400, timeout=60)
            try:
                candidates += _extract_json(raw).get("candidates") or []
                break
            except (ValueError, json.JSONDecodeError):
                continue

    lowered_text = facts_text.lower()
    grounded, seen = [], set()
    for c in candidates:
        name = (c.get("name") or "").strip()
        symbol = (c.get("symbol") or "").strip().upper()
        if name and symbol and name.lower() in lowered_text and (symbol, name.lower()) not in seen:
            seen.add((symbol, name.lower()))
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
        weights = [cap ** 0.5 + 1 for _, cap in pool]  # sqrt flattens mega-cap dominance; +1 keeps zero-cap pickable
        pick = random.choices(pool, weights=weights, k=1)[0]
        chosen.append(pick)
        pool.remove(pick)
    return chosen


def _normalize_name(name: str) -> str:
    stripped = _NAME_SUFFIX_RE.sub("", name.lower())
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9& ]", " ", stripped)).strip()


def _already_on_channel(candidate: dict, channel_symbols: set, channel_titles: list) -> bool:
    """True if the candidate's symbol is tagged (NSE: SYMBOL) anywhere on the
    channel, or its company name appears in an existing video title -- titles
    always carry the company name (producer._ensure_company_name_in_title)
    but not necessarily the ticker."""
    if candidate["symbol"] in channel_symbols:
        return True
    name = _normalize_name(candidate["name"])
    return len(name) >= 3 and any(name in title for title in channel_titles)


def _candidates_from(queries) -> list:
    facts_text = _search_trending_stock_news(queries)
    candidates = _extract_candidates(facts_text)
    print(f"[trending] {len(facts_text)} chars of news snippets, {len(candidates)} grounded candidate(s)")
    return candidates


def select_trending_symbols(exclude_symbols=(), limit: int = 5, queries: list = None, channel_texts=()) -> list:
    """Best-effort list of up to `limit` currently-trending, verified-real NSE
    symbols not already covered: [{symbol, name}, ...]. "Covered" means in
    exclude_symbols (the themes file) OR present on the real YouTube channel,
    given as channel_texts (video title first line + description + tags,
    private and public). If every candidate from the primary queries is
    already covered, further query batches (EXTRA_QUERY_BATCHES) are tried
    before giving up.

    Returns [] on failure or when nothing new can be found -- callers must
    treat [] as "no new trending candidate available right now" and never
    queue a theme with no real candidate behind it.

    Picked via market-cap-weighted random sampling (without replacement) from
    among all verified candidates, rather than strictly the top `limit` by
    market cap -- a bigger, more widely-recognized company is a reasonable
    free proxy for viewer search/click interest (the figure is already
    scraped as part of the existing fundamentals verification step, no new
    API call), but always picking the single largest name every run means
    this channel never covers anything but the same mega-caps every other
    finance channel already covers."""
    exclude = {s.strip().upper() for s in exclude_symbols}
    channel_symbols = set()
    for text in channel_texts:
        channel_symbols.update(research.NSE_SYMBOL_RE.findall(text.upper()))
    channel_titles = [_normalize_name(t.split("\n")[0]) for t in channel_texts]

    batches = [queries] if queries else [None] + EXTRA_QUERY_BATCHES
    verified, seen = [], set()
    for index, batch in enumerate(batches):
        try:
            candidates = _candidates_from(batch)
        except Exception as e:
            print(f"[trending] search/extraction failed: {type(e).__name__}: {e}")
            if index == 0:
                return []
            continue

        excluded = unverified = 0
        for c in candidates:
            if (c["symbol"] in exclude or c["symbol"] in seen
                    or _already_on_channel(c, channel_symbols, channel_titles)):
                excluded += 1
                continue
            seen.add(c["symbol"])
            data = fundamentals.fetch_fundamentals(c["symbol"])
            if data:
                verified.append((c, fundamentals.market_cap_cr(data)))
            else:
                unverified += 1
        print(f"[trending] {excluded} already covered/duplicate, {unverified} failed Screener verification, {len(verified)} verified")
        if verified:
            break

    chosen = _weighted_pick_without_replacement(verified, limit)
    chosen.sort(key=lambda pair: pair[1], reverse=True)
    return [c for c, _market_cap in chosen]
