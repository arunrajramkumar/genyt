"""Two "did you know" beats that go beyond reciting a company's numbers:

- find_story_fact: one surprising, sourced fact about the company's history or
  business, pulled from live web snippets. Grounded the same way as
  trending_stocks: the LLM must quote the snippets verbatim, and the fact is
  dropped unless that quote really appears in them and every number in the
  fact is inside the quote.
- peer_comparison_fact: a comparison against listed peers computed in plain
  code from Screener's peer table, so it can't be hallucinated.

Both return "" / None when nothing verifiable is found -- callers must then
simply skip the beat, never invent one.
"""
import json
import re

from . import llm, research

MIN_PEERS = 3
_NUMBER_RE = re.compile(r"\d[\d,]*\.?\d*")
_ADVICE_RE = re.compile(r"\b(buy|sell|target price|recommend|undervalued|overvalued|multibagger)\b", re.IGNORECASE)

SYSTEM_PROMPT_STORY = """You pick ONE surprising, specific fact about a company's history,
origins, products or business from the search snippets below -- the kind of thing a
curious viewer would not already know (e.g. an unusual founding story, a surprising
product, a record or milestone). Not stock price, analyst views, quarterly results or
generic descriptions.

Output ONLY valid JSON (no markdown fences, no commentary):

{"fact": "one sentence, under 30 words", "quote": "the exact words copied verbatim from the snippets that support the fact"}

Rules:
- Use ONLY information literally present in the snippets. Every number in "fact" must
  appear in "quote". Never add knowledge from elsewhere.
- If the snippets contain no such fact about this company, return {"fact": "", "quote": ""}.
"""


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower()).strip()


def _numbers(text: str) -> set:
    return {n.rstrip(".,").replace(",", "") for n in _NUMBER_RE.findall(text)}


def _extract_story_fact(company_name: str, snippets: str) -> str:
    """Returns a verified one-sentence fact, or "" when none passes grounding."""
    if not snippets.strip():
        return ""
    user_prompt = f"Company: {company_name}\n\nSearch snippets:\n{snippets}\n\nProduce the JSON now."
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

    fact = (data.get("fact") or "").strip()
    quote = (data.get("quote") or "").strip()
    if not fact or len(quote) < 25 or _ADVICE_RE.search(fact):
        return ""
    haystack = _norm(snippets)
    if _norm(quote) not in haystack:
        return ""
    if not _numbers(fact) <= _numbers(quote):
        return ""
    core = _norm(re.sub(r"\b(ltd|limited)\b\.?", "", company_name, flags=re.IGNORECASE))
    if len(core) < 3 or core not in haystack:
        return ""
    return fact


def find_story_fact(company_name: str) -> str:
    """Best-effort: "" on any failure or when nothing verifiable is found."""
    try:
        snippets = research.search_web(f"{company_name} company history founded interesting facts")
        return _extract_story_fact(company_name, snippets)
    except Exception as e:
        print(f"[insights] story lookup failed for {company_name}: {type(e).__name__}: {e}")
        return ""


def _rank_phrase(own, others, higher_is_better):
    """(count beaten, total) comparing own to others, or None if not comparable."""
    others = [o for o in others if o is not None]
    if own is None or len(others) < MIN_PEERS:
        return None
    beaten = sum(1 for o in others if (own > o if higher_is_better else own < o))
    return beaten, len(others)


def peer_comparison_fact(peers: list) -> str:
    """One sentence comparing the company with its listed peers, picking
    whichever of ROCE / P/E is most extreme. "" when the table has too few
    peers or no comparable values."""
    own = next((p for p in peers if p.get("is_self")), None)
    others = [p for p in peers if not p.get("is_self")]
    if not own:
        return ""

    best = None
    for label, key, higher_better, fmt in (
        ("ROCE", "roce", True, "{:g}%"),
        ("P/E", "pe", False, "{:g}"),
    ):
        rank = _rank_phrase(own.get(key), [o.get(key) for o in others], higher_better)
        if not rank:
            continue
        beaten, total = rank
        extremity = beaten / total
        if beaten * 2 < total:
            continue
        if best is None or extremity > best[0]:
            best = (extremity, label, key, higher_better, fmt, beaten, total)
    if not best:
        return ""

    _extremity, label, key, higher_better, fmt, beaten, total = best
    value = fmt.format(own[key])
    if higher_better:
        if beaten == total:
            return f"Among its {total} listed peers on Screener, its {label} of {value} is the highest."
        return f"Its {label} of {value} is higher than {beaten} of its {total} listed peers on Screener."
    if beaten == total:
        return f"Among its {total} listed peers on Screener, its {label} of {value} is the lowest, so it is the cheapest of the group on that measure."
    return f"Its {label} of {value} is lower than {beaten} of its {total} listed peers on Screener."
