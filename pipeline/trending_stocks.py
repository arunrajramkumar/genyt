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
import math
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

# Fresh news events that drive search demand (viewers search for a company when
# its IPO opens, its results land, or it wins an order). "other" is a company
# merely named in passing. Real YouTube data: timely IPOs and results-day
# surprises were the best performers; evergreen mega-cap analyses the worst.
EVENT_TYPES = ("ipo", "results", "guidance", "order_win")

# An LLM-claimed event only counts if the snippet text itself backs it up: each
# pattern list must match within EVENT_WINDOW_CHARS of a literal mention of the
# company's name. Anything unverifiable is downgraded to "other".
EVENT_WINDOW_CHARS = 200
EVENT_PATTERNS = {
    "ipo": [re.compile(r"\bipo\b|initial public offering", re.IGNORECASE)],
    "results": [re.compile(r"\bresults?\b|\bquarter(ly)?\b|\bq[1-4]\b|\bprofit\b|\bearnings\b", re.IGNORECASE)],
    "guidance": [re.compile(r"\bguidance\b|\boutlook\b|\bforecast\b", re.IGNORECASE)],
    "order_win": [
        re.compile(r"\b(orders?|contracts?)\b", re.IGNORECASE),
        re.compile(r"\b(bags?|bagged|wins?|won|secures?|secured|receives?|received|grabs?|lands?|landed)\b", re.IGNORECASE),
    ],
}

# Market caps at or above this (Rs 5 lakh crore) are mega-caps that every
# finance channel already covers; without a fresh event they're a last resort.
MEGA_CAP_CR = 500_000

SYSTEM_PROMPT_TRENDING = """You are a financial news analyst. You will be given real
search-result snippets about the Indian stock market. Identify NSE-listed companies that
are EXPLICITLY named in the snippets below -- never a company you merely happen to know
about from elsewhere.

Output ONLY valid JSON (no markdown fences, no commentary):

{"candidates": [{"symbol": "NSE_TICKER", "name": "Company name exactly as it appears in the text", "event": "ipo|results|guidance|order_win|other", "event_quote": "short phrase copied word-for-word from the snippet, or empty"}]}

Rules:
- Only include a company whose name literally appears in the snippets below.
- "symbol" is the company's standard NSE ticker (e.g. "TCS", "RELIANCE") -- a wrong guess
  here is fine, it gets independently verified afterwards, but never invent a company that
  isn't named in the text.
- "event" is the fresh news event the snippets report for that company: "ipo" (IPO opening,
  subscription or listing), "results" (quarterly/annual results or profit), "guidance"
  (outlook or guidance), "order_win" (order or contract win), or "other" (merely mentioned).
  Only use an event the snippets actually state; otherwise use "other".
- "event_quote" must be copied exactly from the snippets and must mention that event. Leave
  it empty when the event is "other".
- Prefer companies with a fresh event in the snippets over companies merely mentioned in
  passing, and list those first. Include IPO companies too (use the best-guess NSE ticker;
  unlisted ones get dropped by verification).
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


def _event_is_verified(name: str, event: str, text: str) -> bool:
    """True if `text` contains an event-pattern match (for `event`) within
    EVENT_WINDOW_CHARS of some literal mention of `name`. "other" needs no
    evidence; an unknown event never verifies."""
    if event == "other":
        return True
    patterns = EVENT_PATTERNS.get(event)
    if not patterns:
        return False
    lowered_text, needle = text.lower(), name.lower()
    start = lowered_text.find(needle)
    while start != -1:
        window = lowered_text[max(0, start - EVENT_WINDOW_CHARS):start + len(needle) + EVENT_WINDOW_CHARS]
        if all(p.search(window) for p in patterns):
            return True
        start = lowered_text.find(needle, start + 1)
    return False


def _extract_candidates(facts_text: str) -> list:
    """Returns [{symbol, name, event, event_quote}, ...]. Grounded independently
    of what the LLM claims: a candidate only survives if its name literally
    appears in facts_text, so a hallucinated company can't sneak through even if
    the model ignores the system prompt. The LLM's event is kept only if
    _event_is_verified() finds it in the text (else "other"), and event_quote
    only if it is a literal substring of the text."""
    if not facts_text.strip():
        return []

    candidates = []
    for chunk in _chunk_text(facts_text, MAX_EXTRACT_CHARS):
        user_prompt = "Search snippets:\n" + chunk + "\n\nProduce the JSON now."
        for _attempt in range(2):
            try:
                raw = llm.chat_json(SYSTEM_PROMPT_TRENDING, user_prompt, max_tokens=600, timeout=60)
                data = _extract_json(raw)
                if not isinstance(data, dict):
                    raise ValueError("candidates response must be a JSON object")
                candidates += [c for c in data.get("candidates") or [] if isinstance(c, dict)]
                break
            except (ValueError, json.JSONDecodeError):
                continue
            except RuntimeError as e:
                # A hard API error on one chunk shouldn't throw away the
                # candidates already found in the others.
                print(f"[trending] extraction failed for one chunk: {e}")
                break

    lowered_text = facts_text.lower()
    best = {}  # keyed by (symbol, name); a verified event beats "other" for the same company
    for c in candidates:
        name = (c.get("name") or "").strip()
        symbol = (c.get("symbol") or "").strip().upper()
        if not (name and symbol and name.lower() in lowered_text):
            continue
        event = (c.get("event") or "other").strip().lower()
        if event not in EVENT_TYPES or not _event_is_verified(name, event, facts_text):
            event = "other"
        quote = (c.get("event_quote") or "").strip()[:200]
        if event == "other" or not quote or quote.lower() not in lowered_text:
            quote = ""
        entry = {"symbol": symbol, "name": name, "event": event, "event_quote": quote}
        key = (symbol, name.lower())
        if key not in best or (best[key]["event"] == "other" and event != "other"):
            best[key] = entry
    return list(best.values())


def _rank_tier(candidate: dict, cap_cr: float) -> int:
    """0 = fresh news event (ipo/results/guidance/order_win), 1 = other, 2 = other
    mega-cap (last resort). Lower tiers are always drawn from first."""
    if candidate.get("event", "other") in EVENT_TYPES:
        return 0
    return 2 if cap_cr >= MEGA_CAP_CR else 1


def _weighted_pick_without_replacement(pairs: list, k: int) -> list:
    """Picks up to k items from pairs=[(candidate, market_cap_cr), ...]:
    strictly by _rank_tier (event-driven candidates first, mega-caps without
    an event last), then by weighted random sampling without replacement
    within the tier. Within a tier the weight is log-flattened
    (1 + log10(1 + cap)), so a bigger company is still somewhat more likely to
    be picked but a Rs 50,000 cr name is no longer ~10x a Rs 500 cr one. Always
    taking the single largest company converges this channel onto the same
    mega-caps every other finance channel already covers, day after day. The
    randomness keeps the same name from always winning."""
    pool = list(pairs)
    chosen = []
    while pool and len(chosen) < k:
        best_tier = min(_rank_tier(c, cap) for c, cap in pool)
        tier_pool = [(c, cap) for c, cap in pool if _rank_tier(c, cap) == best_tier]
        weights = [1 + math.log10(1 + max(cap, 0.0)) for _, cap in tier_pool]  # >= 1, so zero-cap stays pickable
        pick = random.choices(tier_pool, weights=weights, k=1)[0]
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

    Picked from among all verified candidates by _weighted_pick_without_replacement:
    candidates with a verified fresh news event (IPO, results, guidance, order
    win) first, then other names, with multi-lakh-crore mega-caps last. Within
    a tier the pick is random, weighted by a log-flattened market cap, so the
    same name doesn't always win while bigger companies still get a modest
    edge. Returned dicts carry the verified "event" and "event_quote" when the
    LLM supplied them."""
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
        with_event = sum(1 for c, _cap in verified if c.get("event", "other") in EVENT_TYPES)
        print(f"[trending] {excluded} already covered/duplicate, {unverified} failed Screener verification, {len(verified)} verified, {with_event} with a fresh news event")
        if verified:
            break

    chosen = _weighted_pick_without_replacement(verified, limit)
    chosen.sort(key=lambda pair: pair[1], reverse=True)
    return [c for c, _market_cap in chosen]
