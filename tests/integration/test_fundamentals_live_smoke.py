"""Live, un-mocked sweep across ~90 real NSE-listed companies spanning sectors,
market caps, and ratio shapes (banks with no "ROCE", loss-making names with
negative growth, newly-listed names with thin quarterly history, split-heavy
names, etc).

Why this exists: the project's default suite mocks Screener.in/Tavily with
hand-written payloads, which can only prove the code handles inputs *I*
imagined -- it structurally cannot catch a real response shaped differently
than my mock assumed (see memory: mocked tests miss live-data bugs). Two
production incidents this project hit were exactly that class of bug. This
test hits the real site for a broad, diverse sample and asserts nothing
*crashes* while building a theme entry from whatever real data comes back --
it does not assert the content is good (that still needs a human watching a
real generated video), only that the pipeline doesn't throw.

Excluded from the default run (see pytest.ini's live_network marker) since
it's slow, hits a real third party, and a transient Screener outage/layout
change shouldn't fail every CI run -- run explicitly with
`pytest -m live_network` (e.g. after touching pipeline/fundamentals.py or
scripts/build_finance_queue.py's data-handling logic).
"""
import time

import pytest

from pipeline import fundamentals
from pipeline.segmented_script import parse_segments
from scripts.build_finance_queue import build_theme_entry

pytestmark = pytest.mark.live_network

# Deliberately diverse: mega/mid/small cap, banks/NBFCs (no ROCE), PSUs,
# loss-making or cyclical names (negative growth), recent IPOs (thin
# history), old-economy vs new-age tech, and a few less-common tickers whose
# Screener page layout may differ from the mega-caps every other test uses.
SYMBOLS = [
    "RELIANCE", "TCS", "INFY", "HDFCBANK", "ICICIBANK", "SBIN", "ITC", "HINDUNILVR",
    "LT", "BAJFINANCE", "MARUTI", "SUNPHARMA", "TATAMOTORS", "WIPRO", "ULTRACEMCO",
    "ADANIENT", "ADANIPORTS", "AXISBANK", "KOTAKBANK", "HCLTECH", "ASIANPAINT",
    "NESTLEIND", "TITAN", "BAJAJFINSV", "M&M", "POWERGRID", "NTPC", "ONGC",
    "COALINDIA", "JSWSTEEL", "TATASTEEL", "GRASIM", "HINDALCO", "DRREDDY", "CIPLA",
    "DIVISLAB", "BRITANNIA", "DABUR", "GODREJCP", "PIDILITIND", "BERGEPAINT",
    "HAVELLS", "VOLTAS", "DIXON", "TRENT", "DMART", "ZOMATO", "NYKAA", "PAYTM",
    "POLICYBZR", "IRCTC", "IEX", "CDSL", "KALYANKJIL", "CAMS", "MCX", "BSE",
    "ANGELONE", "CGPOWER", "PERSISTENT", "LTIM", "COFORGE", "MPHASIS", "TATAELXSI",
    "BEL", "HAL", "RVNL", "IRFC", "RAILTEL", "SUZLON", "ADANIGREEN", "TATAPOWER",
    "NHPC", "SJVN", "GAIL", "IOC", "BPCL", "HPCL", "PFC", "RECLTD", "LICI",
    "SBILIFE", "HDFCLIFE", "ICICIPRULI", "BAJAJ-AUTO", "HEROMOTOCO", "EICHERMOT",
    "ASHOKLEY", "TVSMOTOR", "APOLLOHOSP", "LUPIN", "AUROPHARMA", "TORNTPHARM",
    "ZYDUSLIFE", "BIOCON",
]


def test_build_theme_entry_survives_a_broad_live_sweep_of_real_nse_symbols():
    failures = []  # (symbol, exception) -- an actual crash, always a bug
    unavailable = []  # symbol had no fetchable fundamentals -- not a bug, just skip
    built = 0

    for symbol in SYMBOLS:
        try:
            data = fundamentals.fetch_fundamentals(symbol)
            if not data:
                unavailable.append(symbol)
                continue
            entry = build_theme_entry(symbol, symbol)
            if entry is None:
                unavailable.append(symbol)
                continue
            # Must actually be usable downstream, not just "didn't raise".
            segments = parse_segments(entry["theme"])
            assert len(segments) == 5
            assert sum(s["duration"] for s in segments) == entry["duration_sec"]
            built += 1
        except Exception as e:  # noqa: BLE001 -- any crash here is exactly the bug class this test exists to catch
            failures.append((symbol, repr(e)))
        time.sleep(0.5)  # be a reasonable citizen toward Screener.in's free, unauthenticated endpoint

    print(f"\n[live smoke] {built} built, {len(unavailable)} unavailable, {len(failures)} crashed "
          f"(of {len(SYMBOLS)} symbols)")
    if unavailable:
        print(f"[live smoke] unavailable: {', '.join(unavailable)}")
    if failures:
        for symbol, err in failures:
            print(f"[live smoke] CRASH on {symbol}: {err}")

    assert not failures, f"{len(failures)} symbol(s) crashed building a theme entry: {failures}"
    # Screener's own availability/layout can wobble, but a near-total failure
    # rate means something systemic broke (e.g. a layout change fundamentals.py
    # no longer parses) rather than a handful of genuinely unlisted/renamed tickers.
    assert built >= len(SYMBOLS) * 0.7, (
        f"Only {built}/{len(SYMBOLS)} symbols produced a usable entry -- "
        "investigate whether Screener.in's page layout changed."
    )
