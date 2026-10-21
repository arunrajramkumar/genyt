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


def did_you_know_facts(company_name: str, data: dict) -> list:
    """The verified "did you know" facts available for this company right now:
    a sourced story and/or a peer comparison (either may be missing). `data` is
    the dict from fundamentals.fetch_fundamentals (carries what's needed to
    fetch the peer table). Never raises."""
    story = find_story_fact(company_name)
    try:
        peer = peer_comparison_fact(fundamentals.fetch_peers(data))
    except Exception as e:
        print(f"[insights] peer comparison failed for {company_name}: {type(e).__name__}: {e}")
        peer = ""
    print(f"[insights] peer comparison for {company_name}: {peer or 'unavailable'}")
    return [f for f in (story, peer) if f]
