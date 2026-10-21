"""Two "did you know" beats that go beyond reciting a company's numbers:

- find_story_fact: one surprising, sourced fact about the company's history or
  business. Source text comes from the company's Wikipedia article first
  (free, no key, reliable from CI), then a live web search. Grounded the same
  way as trending_stocks: the LLM must quote the source verbatim, and the fact
  is dropped unless that quote really appears in the source and every number
  in the fact is inside the quote.
- peer_comparison_fact: a comparison against listed peers computed in plain
  code from Screener's peer table, so it can't be hallucinated.

Everything returns "" / [] when nothing verifiable is found -- callers must
then simply skip the beat, never invent one.
"""
import json
import re

import requests

from . import fundamentals, llm, research

MIN_PEERS = 3
WIKIPEDIA_API = "https://en.wikipedia.org/w/api.php"
WIKIPEDIA_HEADERS = {"User-Agent": "ytagent/1.0 (stock explainer videos; contact via GitHub)"}
WIKIPEDIA_MAX_CHARS = 4000

_NUMBER_RE = re.compile(r"\d[\d,]*\.?\d*")
_NUMBER_WORDS_RE = re.compile(
    r"\b(two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|twenty|thirty|forty|fifty|"
    r"hundred|thousand|million|billion|lakh|crore|decade|century)\b"
)
_ADVICE_RE = re.compile(r"\b(buy|sell|target price|recommend\w*|undervalued|overvalued|multibagger)\b", re.IGNORECASE)
_GENERIC_NAME_WORDS = {"ltd", "limited", "pvt", "private", "inc", "corp", "corporation", "co", "company"}
_COMPANY_WORDS_RE = re.compile(
    r"\b(?:company|corporation|manufactur|bank|limited|conglomerate|firm|brand|producer|provider|"
    r"enterprise|exchange|depository|group|retailer|jeweller|steel\b|cement|pharma)", re.IGNORECASE,
)

SYSTEM_PROMPT_STORY = """You pick ONE surprising, specific fact about a company's history,
origins, products or business from the source text below -- the kind of thing a
curious viewer would not already know (e.g. an unusual founding story, a surprising
first product, a record, a milestone, an unexpected business line). NOT stock price,
analyst views, quarterly results, or a dry description like "X is an Indian company
that makes Y".

Output ONLY valid JSON (no markdown fences, no commentary):

{"fact": "one self-contained sentence naming the company, under 30 words", "quote": "one passage of 1-2 sentences copied EXACTLY, word for word, from the source text that supports the fact"}

Rules:
- Use ONLY information literally present in the source text. Every number in "fact"
  must appear in "quote". Never add knowledge from elsewhere.
- Never use words like buy, sell, undervalued, or any recommendation.
- If the source text contains nothing genuinely surprising and specific about this
  company, return {"fact": "", "quote": ""}.
"""

_TYPOGRAPHY = str.maketrans({
    "‘": "'", "’": "'", "“": '"', "”": '"',
    "–": "-", "—": "-", "‑": "-", " ": " ", " ": " ",
})


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", text.translate(_TYPOGRAPHY).lower()).strip()


def _numbers(text: str) -> set:
    return {n.rstrip(".,").replace(",", "") for n in _NUMBER_RE.findall(text)}


def _core_name(name: str) -> str:
    """Lowercased name without parentheticals and trailing legal suffixes:
    "Pidilite Industries Ltd" -> "pidilite industries"."""
    tokens = re.findall(r"[a-z0-9&]+", re.sub(r"\(.*?\)", " ", name.lower()))
    while tokens and tokens[-1] in _GENERIC_NAME_WORDS:
        tokens.pop()
    return " ".join(tokens)


def _wikipedia_extract(company_name: str) -> str:
    """Plain-text extract of the company's own Wikipedia article ("" if there
    isn't a confidently matching one). The article title must equal the
    company's core name, so a same-named person/place/other company is never
    used."""
    core = _core_name(company_name)
    if len(core) < 3:
        return ""
    try:
        resp = requests.get(
            WIKIPEDIA_API,
            params={"action": "query", "list": "search", "srsearch": f"{company_name} India company",
                    "srlimit": 5, "format": "json"},
            headers=WIKIPEDIA_HEADERS, timeout=10,
        )
        resp.raise_for_status()
        hits = resp.json()["query"]["search"]
        title = next(
            (h["title"] for h in hits
             if _core_name(h["title"]) == core or _core_name(h["title"]).startswith(core + " ")),
            None,
        )
        if not title:
            return ""
        resp = requests.get(
            WIKIPEDIA_API,
            params={"action": "query", "prop": "extracts", "explaintext": 1, "exchars": WIKIPEDIA_MAX_CHARS,
                    "titles": title, "redirects": 1, "format": "json"},
            headers=WIKIPEDIA_HEADERS, timeout=10,
        )
        resp.raise_for_status()
        pages = resp.json()["query"]["pages"]
        extract = next(iter(pages.values())).get("extract", "") or ""
    except (requests.RequestException, ValueError, KeyError, StopIteration):
        return ""
    if not _COMPANY_WORDS_RE.search(extract[:600]):
        return ""
    return f"Wikipedia article '{title}':\n{extract.strip()}"


def _extract_story_fact(company_name: str, snippets: str) -> str:
    """Returns a verified one-sentence fact, or "" when none passes grounding."""
    if not snippets.strip():
        return ""
    user_prompt = f"Company: {company_name}\n\nSource text:\n{snippets}\n\nProduce the JSON now."
    data = None
    for _attempt in range(2):
        try:
            raw = llm.chat_json(SYSTEM_PROMPT_STORY, user_prompt, max_tokens=300, timeout=60)
            match = re.search(r"\{.*\}", raw, re.DOTALL)
            data = json.loads(match.group(0)) if match else None
            if isinstance(data, dict):
                break
        except (ValueError, json.JSONDecodeError):
            continue
    if not isinstance(data, dict):
        return ""

    fact = str(data.get("fact") or "").strip()
    quote = str(data.get("quote") or "").strip()
    if not fact or len(quote) < 25 or _ADVICE_RE.search(fact):
        return ""
    haystack = _norm(snippets)
    norm_quote = _norm(quote)
    if norm_quote not in haystack:
        return ""
    if not _numbers(fact) <= _numbers(quote):
        return ""
    if not set(_NUMBER_WORDS_RE.findall(fact.lower())) <= set(_NUMBER_WORDS_RE.findall(norm_quote)):
        return ""
    core = _core_name(company_name)
    if len(core) < 3 or core not in haystack:
        return ""
    return fact


def find_story_fact(company_name: str) -> str:
    """Best-effort: "" on any failure or when nothing verifiable is found.
    Wikipedia first (no search credit spent); web search only if that yields
    no verifiable fact."""
    try:
        wiki = _wikipedia_extract(company_name)
        fact = _extract_story_fact(company_name, wiki) if wiki else ""
        if not fact:
            web = research.search_web(f"{company_name} company history founded interesting facts")
            fact = _extract_story_fact(company_name, web)
        print(f"[insights] story fact for {company_name}: {fact or 'none verified'}")
        return fact
    except Exception as e:
        print(f"[insights] story lookup failed for {company_name}: {type(e).__name__}: {e}")
        return ""


def _rank(own, others, higher_is_better, positive_only=False):
    """(count beaten, total) comparing own to others, or None if not comparable."""
    usable = (lambda v: v is not None and v > 0) if positive_only else (lambda v: v is not None)
    others = [o for o in others if usable(o)]
    if not usable(own) or len(others) < MIN_PEERS:
        return None
    beaten = sum(1 for o in others if (own > o if higher_is_better else own < o))
    return beaten, len(others)


def peer_comparison_fact(peers: list) -> str:
    """One sentence comparing the company with its listed peers, using
    whichever of ROCE / P/E is most favourable-or-extreme. "" when the table
    has too few peers, the company's own row is missing, or neither metric
    puts it in the upper half of the group."""
    own = next((p for p in peers or [] if p.get("is_self")), None)
    others = [p for p in peers or [] if not p.get("is_self")]
    if not own:
        return ""

    best = None
    for name, key, higher_better, fmt, positive_only in (
        ("return on capital employed, or ROCE,", "roce", True, "{:g}%", False),
        ("P/E ratio", "pe", False, "{:g}", True),
    ):
        rank = _rank(own.get(key), [o.get(key) for o in others], higher_better, positive_only)
        if not rank:
            continue
        beaten, total = rank
        if beaten * 2 < total:
            continue
        if best is None or beaten / total > best[0]:
            best = (beaten / total, name, key, higher_better, fmt, beaten, total)
    if not best:
        return ""

    _frac, name, key, higher_better, fmt, beaten, total = best
    value = fmt.format(own[key])
    if beaten == total:
        if higher_better:
            return f"Its {name} of {value} is the highest among its {total} listed peers."
        return (f"Its {name} of {value} is the lowest among its {total} listed peers, "
                "making it the cheapest of the group on that measure.")
    comparison = "higher" if higher_better else "lower"
    return f"Its {name} of {value} is {comparison} than {beaten} of its {total} listed peers."


MAX_FACTS = 2  # all that fits the 7-10 second "Did You Know" beat
DAYS_PER_QUARTER = 91


def _num(text):
    m = _NUMBER_RE.search(str(text or ""))
    try:
        return float(m.group(0).rstrip(".,").replace(",", "")) if m else None
    except ValueError:
        return None


def _latest(values, offset=1):
    """float(values[-offset]) or None."""
    try:
        return float(str(values[-offset]).replace(",", ""))
    except (IndexError, ValueError, TypeError):
        return None


def _crore(x: float) -> str:
    return f"{x:,.0f}" if x >= 100 else f"{x:g}"


def payback_fact(data: dict) -> str:
    """For a rich valuation: how many years of today's profit the price equals
    (that is just the P/E, restated)."""
    pe = _num(((data or {}).get("ratios") or {}).get("Stock P/E"))
    if pe is None or pe < 40 or pe > 2000:
        return ""
    years = round(pe)
    return (f"At its P/E ratio of {years}, it would take about {years} years of its current profit "
            "to repay today's share price, if profit stayed flat.")


def profit_per_day_fact(data: dict) -> str:
    profit = _latest(((data or {}).get("quarterly") or {}).get("net_profit_cr", []))
    if not profit or profit <= 0:
        return ""
    per_day = profit / DAYS_PER_QUARTER
    if per_day >= 1:
        amount = f"Rs {per_day:.1f} crore" if per_day < 10 else f"Rs {per_day:.0f} crore"
    elif per_day >= 0.1:
        amount = f"Rs {per_day * 100:.0f} lakh"
    else:
        return ""
    return (f"Its latest quarterly net profit of Rs {_crore(profit)} crore works out to roughly "
            f"{amount} of profit every single day.")


def margin_fact(data: dict) -> str:
    q = (data or {}).get("quarterly") or {}
    sales, profit = q.get("sales_cr", []), q.get("net_profit_cr", [])
    s_now, p_now, s_ago, p_ago = _latest(sales), _latest(profit), _latest(sales, 5), _latest(profit, 5)
    if None in (s_now, p_now, s_ago, p_ago) or s_now <= 0 or s_ago <= 0:
        return ""
    now, ago = p_now / s_now * 100, p_ago / s_ago * 100
    if round(now) == round(ago) or abs(now - ago) < 1 or now <= 0 or ago <= 0:
        return ""
    return (f"Out of every Rs 100 it sold in the latest quarter, Rs {round(now)} became profit, "
            f"{'up' if now > ago else 'down'} from Rs {round(ago)} a year earlier.")


def _high_low_fact(data: dict) -> str:
    ratios = (data or {}).get("ratios") or {}
    price = _num(ratios.get("Current Price"))
    parts = _NUMBER_RE.findall(str(ratios.get("High / Low") or ""))
    if price is None or len(parts) != 2:
        return ""
    high, low = (float(p.rstrip(".,").replace(",", "")) for p in parts)
    if not (0 < low <= price <= high):
        return ""
    below_high = (high - price) / high * 100
    above_low = (price - low) / low * 100
    if below_high <= 3:
        return f"The stock is trading within 3% of its 52-week high of Rs {_crore(high)}."
    if below_high >= 10:
        return f"The stock is trading about {round(below_high)}% below its 52-week high of Rs {_crore(high)}."
    if above_low >= 50:
        return f"The stock is up about {round(above_low)}% from its 52-week low of Rs {_crore(low)}."
    return ""


def computed_facts(data: dict) -> list:
    """Facts derived by plain arithmetic from the already-scraped fundamentals
    (so they can't be hallucinated), most striking first. Never raises."""
    facts = []
    for fn in (payback_fact, profit_per_day_fact, margin_fact, _high_low_fact):
        try:
            fact = fn(data)
        except Exception as e:
            print(f"[insights] {fn.__name__} failed: {type(e).__name__}: {e}")
            continue
        if fact:
            facts.append(fact)
    return facts


def did_you_know_facts(company_name: str, data: dict) -> list:
    """The verified "did you know" facts available for this company right now
    (at most MAX_FACTS): a sourced story, a peer comparison, then facts computed
    from the company's own numbers -- any may be missing. `data` is the dict
    from fetch_fundamentals (also carries what's needed to fetch the peer
    table). Never raises."""
    story = find_story_fact(company_name)
    try:
        peer = peer_comparison_fact(fundamentals.fetch_peers(data))
    except Exception as e:
        print(f"[insights] peer comparison failed for {company_name}: {type(e).__name__}: {e}")
        peer = ""
    print(f"[insights] peer comparison for {company_name}: {peer or 'unavailable'}")
    facts = [f for f in (story, peer) if f] + computed_facts(data)
    print(f"[insights] {len(facts)} did-you-know candidate(s) for {company_name}; using {min(len(facts), MAX_FACTS)}")
    return facts[:MAX_FACTS]
