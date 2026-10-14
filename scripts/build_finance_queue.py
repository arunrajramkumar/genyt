#!/usr/bin/env python3
"""Turns a list of NSE-listed companies into a themes.yaml-style finance queue:
one segmented Short per company (Hook -> Key Metrics -> Revenue & Profit
Growth -> Risk & Verdict -> Recap), grounded in real scraped fundamentals
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
from pipeline import config, fundamentals, research, trending_stocks  # noqa: E402

MIN_DURATION = 90
MAX_DURATION = 120


def build_theme_entry(symbol: str, company_name: str, duration_sec: int = 100) -> dict:
    """Returns None if fundamentals couldn't be fetched for this symbol --
    callers must skip the company rather than queue a theme with no real
    facts to ground it in."""
    data = fundamentals.fetch_fundamentals(symbol)
    if not data:
        return None

    duration_sec = max(MIN_DURATION, min(MAX_DURATION, duration_sec))
    ratios = data["ratios"]
    quarterly = data.get("quarterly", {})

    ratio_lines = "\n".join(f"{k}: {v}" for k, v in ratios.items())
    headline_fact = next(iter(ratios.values()), company_name)
    headline_label = next(iter(ratios.keys()), "")

    growth_lines = []
    if "revenue_growth_yoy_pct" in data:
        growth_lines.append(f"Revenue grew {data['revenue_growth_yoy_pct']}% year-on-year")
    if "profit_growth_yoy_pct" in data:
        growth_lines.append(f"Net profit grew {data['profit_growth_yoy_pct']}% year-on-year")
    if quarterly.get("sales_cr"):
        growth_lines.append(f"Latest quarterly revenue: Rs {quarterly['sales_cr'][-1]} crore")
    if quarterly.get("net_profit_cr"):
        growth_lines.append(f"Latest quarterly net profit: Rs {quarterly['net_profit_cr'][-1]} crore")
    growth_text = "\n".join(growth_lines) or "No recent quarterly trend data is available."

    hook_dur = max(6, round(duration_sec * 0.12))
    metrics_dur = max(15, round(duration_sec * 0.28))
    growth_dur = max(15, round(duration_sec * 0.25))
    risk_dur = max(15, round(duration_sec * 0.22))
    remaining = duration_sec - hook_dur - metrics_dur - growth_dur - risk_dur
    recap_dur = max(6, remaining)

    lines, cursor = [], 0

    lines.append(f"{cursor}-{cursor + hook_dur} seconds - Hook")
    lines.append(
        f'{company_name} (NSE: {symbol}). Write an exciting, curiosity-sparking opening question '
        f"for stock-market viewers, using ONLY this headline fact: {headline_label}: {headline_fact}."
    )
    lines.append("")
    cursor += hook_dur

    lines.append(f"{cursor}-{cursor + metrics_dur} seconds - Key Metrics")
    lines.append(
        f"Explain what these valuation metrics for {company_name} (NSE: {symbol}) mean for a retail "
        f"investor, in simple terms. Facts:\n{ratio_lines}"
    )
    lines.append("")
    cursor += metrics_dur

    lines.append(f"{cursor}-{cursor + growth_dur} seconds - Revenue & Profit Growth")
    lines.append(f"Explain what this quarterly growth trend means for {company_name}. Facts:\n{growth_text}")
    lines.append("")
    cursor += growth_dur

    lines.append(f"{cursor}-{cursor + risk_dur} seconds - Risk & Verdict")
    lines.append(
        f"Based ONLY on the valuation and growth facts already given above, raise one balanced, "
        f"neutral question a long-term investor should still answer for themselves about {company_name} "
        "(e.g. whether the current valuation already prices in the growth). Do not give a buy/sell "
        "recommendation, and do not add any new numbers beyond what's already been given."
    )
    lines.append("")
    cursor += risk_dur

    lines.append(f"{cursor}-{cursor + recap_dur} seconds - Recap")
    lines.append(f"Briefly restate the key numbers and the open question about {company_name} so viewers remember them.")
    cursor += recap_dur

    theme = "\n".join(lines)

    return {
        "theme": theme,
        "style": "confident, data-driven stock-analysis explainer, strictly neutral -- no investment advice",
        "duration_sec": cursor,
        "shorts": True,
        "done": False,
        # Tags that must appear regardless of what the LLM comes up with (see
        # segmented_script.merge_tags).
        "extra_tags": ["stock analysis", company_name, symbol, "NSE", "investing"],
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
    parser.add_argument("--duration", type=int, default=100, help=f"Target seconds, clamped to [{MIN_DURATION}, {MAX_DURATION}]")
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
    for symbol, name in items:
        entry = build_theme_entry(symbol, name, args.duration)
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
