#!/usr/bin/env python3
"""Turns a list of NSE-listed companies into a themes.yaml-style finance queue:
one segmented Short per company (Hook -> Key Metrics -> Revenue & Profit
Growth -> Risk & Verdict -> Recap; Risk & Verdict folds into Recap when short),
grounded in real scraped fundamentals
(pipeline/fundamentals.py) instead of a human hand-researching and
hand-writing the numbers into themes.yaml, the way CDSL/Kalyan Jewellers were.

Each labelled section becomes its own scene via segmented_script.py's narrow
per-section LLM calls (same mechanism the education queue uses), so the
numbers, the narration, and the on-screen text all stay tightly bound to the
same real facts instead of a single call having to compress everything (and
risk inventing a figure) at once.

Usage:
    python3 scripts/build_finance_queue.py --symbols CDSL,KALYANKJIL:Kalyan Jewellers \
        --output finance_queue.yaml

    # Or auto-discover currently trending NSE companies via live news search
    # instead of hand-picking symbols (see pipeline/trending_stocks.py):
    python3 scripts/build_finance_queue.py --trending --limit 5 --output finance_queue.yaml
"""
import argparse
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from pipeline import company_insights, config, fundamentals, market_trends, research, trending_stocks  # noqa: E402

# Shorts viewers swipe away after ~25-35s of a 90-120s video (real YouTube
# Analytics), so these run short: the hook lands in the first few seconds and
# the whole thing finishes before attention drops off.
MIN_DURATION = 35
MAX_DURATION = 50
DEFAULT_DURATION = 45
# Floor for Recap; below this the Risk & Verdict question is folded into it.
RECAP_MIN_SEC = 6
# Hook + Key Metrics + Revenue & Profit Growth + Recap minimums: the body must
# fit these even at MIN_DURATION, so a long Did You Know beat can't squeeze it.
MIN_BODY_SEC = 6 + 8 + 8 + RECAP_MIN_SEC

# Ratios worth leading with/keeping -- a retail viewer finds growth and
# capital-efficiency numbers far more striking than, say, Face Value or Book
# Value, but Screener.in's top-ratios dict is in its own scrape order, not a
# "most interesting first" order. Used to both pick the hook's headline fact
# and to cap the Key Metrics section instead of dumping every scraped ratio
# into one scene (YouTube Analytics feedback on a real video: viewers dropped
# off mid-way through a back-to-back recital of 8+ numbers with no pause).
_STRIKING_RATIO_PRIORITY = ["ROCE", "ROE", "Stock P/E", "Price to Book Value", "Dividend Yield", "Market Cap"]


def _select_striking_ratios(ratios: dict, limit: int = 3) -> dict:
    ordered_keys = [k for k in _STRIKING_RATIO_PRIORITY if k in ratios]
    ordered_keys += [k for k in ratios if k not in ordered_keys]
    return {k: ratios[k] for k in ordered_keys[:limit]}


def _pick_headline_fact(data: dict, ratios: dict) -> tuple:
    """A profit/revenue growth % (e.g. "89% Profit CAGR") is a far stronger
    hook than whatever ratio happens to be scraped first -- prefer it, then
    fall back to the most striking available ratio."""
    quarterly = data.get("quarterly", {})
    # Only a meaningful % leads (fundamentals.plain_growth_pct): a "56800%"
    # profit jump off a one-off quarter makes a misleading hook.
    profit_pct = fundamentals.plain_growth_pct(quarterly.get("net_profit_cr", []))
    if profit_pct is not None and profit_pct > 0 and not fundamentals.profit_exceeds_revenue(data):
        return "Profit growth", f"{profit_pct:g}% year-on-year"
    revenue_pct = fundamentals.plain_growth_pct(quarterly.get("sales_cr", []))
    if revenue_pct is not None and revenue_pct > 0:
        return "Revenue growth", f"{revenue_pct:g}% year-on-year"
    striking = _select_striking_ratios(ratios, limit=1)
    if striking:
        return next(iter(striking.items()))
    return "", ""


def build_theme_entry(symbol: str, company_name: str, duration_sec: int = DEFAULT_DURATION, competitor_tags: list = None) -> dict:
    """Returns None if fundamentals couldn't be fetched for this symbol --
    callers must skip the company rather than queue a theme with no real
    facts to ground it in."""
    data = fundamentals.fetch_fundamentals(symbol)
    if not data:
        return None
    # Screener names end in "Ltd." -- the prompts below follow the name with
    # their own punctuation ("...Industries Ltd.." otherwise).
    company_name = company_name.strip().rstrip(".")

    duration_sec = max(MIN_DURATION, min(MAX_DURATION, duration_sec))
    ratios = data["ratios"]
    quarterly = data.get("quarterly", {})

    striking_ratios = _select_striking_ratios(ratios, limit=3)
    ratio_lines = "\n".join(f"{k}: {v}" for k, v in striking_ratios.items())
    headline_label, headline_fact = _pick_headline_fact(data, ratios)

    growth_lines = []
    for key, noun in (("sales_cr", "Revenue"), ("net_profit_cr", "Net profit")):
        sentence = fundamentals.growth_sentence(quarterly.get(key, []), noun)
        if sentence:
            growth_lines.append(sentence)
    if quarterly.get("sales_cr"):
        growth_lines.append(f"Latest quarterly revenue: Rs {quarterly['sales_cr'][-1]} crore")
    if quarterly.get("net_profit_cr"):
        growth_lines.append(f"Latest quarterly net profit: Rs {quarterly['net_profit_cr'][-1]} crore")
    if fundamentals.profit_exceeds_revenue(data):
        growth_lines.append(fundamentals.ONE_OFF_PROFIT_NOTE)
    growth_text = "\n".join(growth_lines) or "No recent quarterly trend data is available."

    beat = company_insights.did_you_know_beat(company_name, data)
    did_you_know, dyk_narration = beat["facts"], beat["narration"]
    if not did_you_know:
        dyk_dur = 0
    elif dyk_narration:
        dyk_dur = 7 if len(dyk_narration.split()) <= 18 else 10
    else:
        dyk_dur = 7 if len(did_you_know) == 1 else 10
    if dyk_dur > 7 and duration_sec - dyk_dur < MIN_BODY_SEC:
        dyk_dur = 7
    body_sec = duration_sec - dyk_dur

    hook_dur = max(6, round(body_sec * 0.15))
    metrics_dur = max(8, round(body_sec * 0.30))
    growth_dur = max(8, round(body_sec * 0.25))
    risk_dur = max(7, round(body_sec * 0.15))
    remaining = body_sec - hook_dur - metrics_dur - growth_dur
    # Risk & Verdict only gets its own scene when there's room for it plus a
    # real Recap; in a short video its open question moves into the Recap.
    merge_risk = remaining - risk_dur < RECAP_MIN_SEC
    recap_dur = remaining if merge_risk else remaining - risk_dur

    # Text before the first timed header isn't a scene (parse_segments skips
    # it); this line is the clean "Name (NSE: SYMBOL)" anchor the title,
    # thumbnail and ticker handling read the company from.
    lines, cursor = [f"{company_name} (NSE: {symbol})", ""], 0

    lines.append(f"{cursor}-{cursor + hook_dur} seconds - Hook")
    lines.append(
        f"Write an exciting, curiosity-sparking opening question for stock-market viewers that leads "
        f"immediately with this headline number -- {headline_label}: {headline_fact} -- in the first 6-8 "
        f"words, before anything else; never open with the company name, a date, or other background "
        f"framing first. "
        f"Mention {company_name} (NSE: {symbol}) only after that number has hooked the viewer."
    )
    lines.append("")
    cursor += hook_dur

    if did_you_know:
        lines.append(f"{cursor}-{cursor + dyk_dur} seconds - Did You Know")
        if dyk_narration:
            lines.append(f'"Did you know? {dyk_narration}"')
        else:
            facts = "\n".join(f"- {f}" for f in did_you_know)
            lines.append(
                f"State this as one surprising, specific 'did you know' fact about {company_name} in 1-2 "
                "punchy sentences, using ONLY the fact(s) below and no new numbers or claims. Never state it "
                f"as a recommendation. Facts:\n{facts}"
            )
        lines.append("")
        cursor += dyk_dur

    lines.append(f"{cursor}-{cursor + metrics_dur} seconds - Key Metrics")
    lines.append(
        f"For {company_name} (NSE: {symbol}), pick ONLY the 2 or 3 most striking of these valuation "
        "metrics -- do not list all of them. For each one you use, state the number and then immediately "
        "translate it into one short plain-English clause a retail investor would understand (e.g. a "
        "high ROCE -- meaning it generates a lot of profit from the capital it employs), not just the "
        f"bare ratio name and figure. Use only the numbers below. Facts:\n{ratio_lines}"
    )
    lines.append("")
    cursor += metrics_dur

    lines.append(f"{cursor}-{cursor + growth_dur} seconds - Revenue & Profit Growth")
    lines.append(f"Explain what this quarterly growth trend means for {company_name}. Facts:\n{growth_text}")
    lines.append("")
    cursor += growth_dur

    risk_question = (
        f"Based ONLY on the valuation and growth facts already given above, raise one balanced, "
        f"neutral question a long-term investor should still answer for themselves about {company_name} "
        "(e.g. whether the current valuation already prices in the growth). Do not give a buy/sell "
        "recommendation, and do not add any new numbers beyond what's already been given."
    )
    if not merge_risk:
        lines.append(f"{cursor}-{cursor + risk_dur} seconds - Risk & Verdict")
        lines.append(risk_question)
        lines.append("")
        cursor += risk_dur

    lines.append(f"{cursor}-{cursor + recap_dur} seconds - Recap")
    if merge_risk:
        lines.append(
            f"{risk_question} Then briefly restate the key numbers about {company_name} so viewers remember them, "
            "and end with one short yes-or-no question inviting viewers to answer in the comments "
            "(a question only, never a recommendation)."
        )
    else:
        lines.append(
            f"Briefly restate the key numbers and the open question about {company_name} so viewers remember them, "
            "then end with one short yes-or-no question inviting viewers to answer in the comments "
            "(a question only, never a recommendation)."
        )
    cursor += recap_dur

    theme = "\n".join(lines)

    return {
        "theme": theme,
        "style": "confident, data-driven stock-analysis explainer, strictly neutral -- no investment advice",
        "duration_sec": cursor,
        "shorts": True,
        "done": False,
        # Tags that must appear regardless of what the LLM comes up with (see
        # segmented_script.merge_tags). Competitor tags (real tags mined from
        # other channels' currently-popular niche videos, see
        # market_trends.get_competitor_tags) come last so the company-specific
        # tags above always survive merge_tags' character budget first.
        "extra_tags": ["stock analysis", company_name, symbol, "NSE", "investing", *(competitor_tags or [])],
        # "25" = News & Politics -- these videos are about currently-trending
        # market news for a named company, not the generic "22" People & Blogs
        # bucket every upload defaulted to.
        "category_id": "25",
        # Not read by run.py -- just makes the generated queue file human-readable.
        "_label": f"{company_name} (NSE: {symbol})",
    }


def existing_symbols(path: str) -> set:
    """NSE symbols already present (done or not) in a themes.yaml-style queue
    file, so trending-driven discovery doesn't re-queue a company that's
    already covered. Returns an empty set if the file doesn't exist yet."""
    try:
        with open(path) as f:
            data = yaml.safe_load(f) or {}
    except FileNotFoundError:
        return set()

    symbols = set()
    for entry in data.get("themes", []):
        symbols.update(research.NSE_SYMBOL_RE.findall((entry.get("theme") or "").upper()))
    return symbols


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--symbols",
        help='Comma-separated NSE symbols, optionally "SYMBOL:Company Name" (e.g. "CDSL,KALYANKJIL:Kalyan Jewellers")',
    )
    parser.add_argument(
        "--trending", action="store_true",
        help="Auto-discover currently trending NSE companies via live news search instead of --symbols",
    )
    parser.add_argument("--limit", type=int, default=5, help="Max companies to queue when using --trending")
    parser.add_argument(
        "--avoid-existing", default=str(config.THEMES_FILE),
        help="Theme queue YAML to skip already-covered NSE symbols from when using --trending (default: themes.yaml)",
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--duration", type=int, default=DEFAULT_DURATION, help=f"Target seconds, clamped to [{MIN_DURATION}, {MAX_DURATION}]")
    args = parser.parse_args()

    if not args.symbols and not args.trending:
        parser.error("either --symbols or --trending is required")

    if args.trending:
        exclude = existing_symbols(args.avoid_existing)
        candidates = trending_stocks.select_trending_symbols(exclude_symbols=exclude, limit=args.limit)
        if not candidates:
            print("No trending candidates found (or none could be verified) -- nothing to queue")
            candidates = []
        items = [(c["symbol"], c["name"]) for c in candidates]
    else:
        items = []
        for item in args.symbols.split(","):
            item = item.strip()
            if not item:
                continue
            symbol, _, name = item.partition(":")
            items.append((symbol.strip(), name.strip() or symbol.strip()))

    entries = []
    competitor_tags = market_trends.get_competitor_tags(limit=5)
    if competitor_tags:
        print(f"Mined competitor tags: {', '.join(competitor_tags)}")
    for symbol, name in items:
        entry = build_theme_entry(symbol, name, args.duration, competitor_tags)
        if entry is None:
            print(f"Skipping {symbol} -- could not fetch fundamentals")
            continue
        entries.append(entry)
        print(f"Built entry for {entry['_label']} ({entry['duration_sec']}s)")

    with open(args.output, "w") as f:
        yaml.safe_dump({"themes": entries}, f, sort_keys=False, allow_unicode=True, width=100)
    print(f"Wrote {len(entries)} theme entries to {args.output}")


if __name__ == "__main__":
    main()
