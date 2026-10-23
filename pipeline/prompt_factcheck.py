"""Cross-checks the figures a hand-written video prompt quotes against the
company's live Screener fundamentals, before the video is made.

Quoted lines in a prompt are spoken word for word, so a stale or mistyped
figure goes straight into the video. Seen live: a Cummins India prompt
presented year-old Q2 FY26 results as "Q2 FY2027" (profit +38.5%) while the
latest reported quarter, Q1 FY27, had profit -8%; a Lloyds Metals prompt
quoted a ₹572 crore quarter when the latest was ₹1,726 crore; a Bajaj Finance
prompt said "P/E about 27" while Screener showed 29.5 -- and the thumbnail,
built from Screener, contradicted the narration.

Warnings only: the video still gets made (uploads are private and reviewed),
but the warnings land in the run log, the metadata file and the issue comment.
"""
import re

from . import fundamentals, thumbnail_spec

GROWTH_TOLERANCE_PTS = 5.0
RATIO_TOLERANCE = 0.10

_QUARTER_RE = re.compile(r"\bQ([1-4])\s*FY\s*'?(?:20)?(\d{2})\b", re.IGNORECASE)
# Within the same sentence (a "." only ends it when followed by a space, so
# "26.4%" and "₹3,170" don't), skipping amounts on the way to the first %.
_PCT_NEAR_RE = r"(?:(?!\.\s)[^%]){0,80}?(?<![\d.,])(-?\d+(?:\.\d+)?)\s*%"
_PROFIT_GROWTH_RE = re.compile(r"\b(?:net\s+)?profit\b" + _PCT_NEAR_RE, re.IGNORECASE)
_REVENUE_GROWTH_RE = re.compile(r"\b(?:revenue|sales)\b" + _PCT_NEAR_RE, re.IGNORECASE)
_AMOUNT_NEAR_RE = r"(?:(?!\.\s)[^₹%]){0,50}?(?:₹|Rs\.?)\s?(\d[\d,]*(?:\.\d+)?)\s*(lakh crore|crore)"
_PROFIT_AMOUNT_RE = re.compile(r"\b(?:net\s+)?profit\b" + _AMOUNT_NEAR_RE, re.IGNORECASE)
_REVENUE_AMOUNT_RE = re.compile(r"\b(?:revenue|sales)\b" + _AMOUNT_NEAR_RE, re.IGNORECASE)
_FALL_WORDS_RE = re.compile(r"\b(fell|fall|falls|declin\w*|drop\w*|down|slipp?\w*|lower|shrank)\b", re.IGNORECASE)
_PE_RE = re.compile(r"(?:\bP/?E\b(?:\s+ratio)?|price[- ]to[- ]earnings)[^.\d]{0,25}?(\d+(?:\.\d+)?)"
                    r"|(\d+(?:\.\d+)?)\s*(?:times|x)\s+(?:its\s+)?(?:trailing\s+)?earnings", re.IGNORECASE)
_MCAP_RE = re.compile(r"market\s+(?:cap(?:italisation|italization)?|value)[^.₹\d]{0,30}?(?:₹|Rs\.?)\s?"
                      r"(\d[\d,]*(?:\.\d+)?)\s*(lakh crore|crore)", re.IGNORECASE)
_PRICE_RE = re.compile(r"(?:trad\w+\s+at|share\s+price\s+(?:of|is)?)[^.₹\d]{0,20}?(?:₹|Rs\.?)\s?(\d[\d,]*(?:\.\d+)?)",
                       re.IGNORECASE)


def _num(text):
    return float(str(text).replace(",", ""))


def _quarter_index(q: int, fy: int) -> int:
    return fy * 4 + q


def _signed_pct(match) -> float:
    """The quoted % with the sign the sentence gives it ("fell 8%" -> -8)."""
    value = float(match.group(1))
    if value > 0 and _FALL_WORDS_RE.search(match.group(0)):
        value = -value
    return value


def _check_quarter(prompt: str, data: dict) -> list:
    dates = (data.get("quarterly") or {}).get("quarter_dates") or []
    latest = thumbnail_spec.fiscal_quarter_label(dates[-1]) if dates else ""
    m_latest = _QUARTER_RE.search(latest)
    if not m_latest:
        return []
    latest_idx = _quarter_index(int(m_latest.group(1)), int(m_latest.group(2)))
    warnings = []
    for m in {(int(m.group(1)), int(m.group(2))) for m in _QUARTER_RE.finditer(prompt)}:
        q, fy = m
        idx = _quarter_index(q, fy)
        if idx > latest_idx:
            warnings.append(f"Prompt cites Q{q} FY{fy:02d}, but Screener's latest reported quarter is {latest} -- "
                            "those results may not be out yet; check the figures aren't from an older quarter.")
        elif idx < latest_idx - 1:
            warnings.append(f"Prompt cites Q{q} FY{fy:02d}, but {latest} results are already out -- "
                            "the prompt may be using stale numbers.")
    return warnings


def _check_growth(prompt: str, data: dict) -> list:
    quarterly = data.get("quarterly") or {}
    warnings = []
    for regex, key, noun in ((_PROFIT_GROWTH_RE, "net_profit_cr", "net profit"),
                             (_REVENUE_GROWTH_RE, "sales_cr", "revenue")):
        actual = fundamentals.plain_growth_pct(quarterly.get(key, []))
        m = regex.search(prompt)
        if actual is None or not m:
            continue
        quoted = _signed_pct(m)
        if abs(quoted - actual) > GROWTH_TOLERANCE_PTS:
            warnings.append(f"Prompt says {noun} growth of {quoted:+g}%, but Screener's latest quarter shows "
                            f"{actual:+g}% year-on-year.")
    return warnings


def _check_amounts(prompt: str, data: dict) -> list:
    """A quoted quarterly profit/revenue amount far from the latest quarter's
    (seen live: a stale ₹572 crore quarter quoted when the latest was ₹1,726 crore)."""
    quarterly = data.get("quarterly") or {}
    warnings = []
    for regex, key, noun in ((_PROFIT_AMOUNT_RE, "net_profit_cr", "net profit"),
                             (_REVENUE_AMOUNT_RE, "sales_cr", "revenue")):
        values = quarterly.get(key) or []
        m = regex.search(prompt)
        if not values or not m:
            continue
        try:
            actual = _num(values[-1])
        except ValueError:
            continue
        quoted = _num(m.group(1)) * (100000 if m.group(2).lower() == "lakh crore" else 1)
        if actual > 0 and abs(quoted - actual) / actual > RATIO_TOLERANCE:
            warnings.append(f"Prompt says {noun} of about ₹{quoted:,.0f} crore, but Screener's latest quarter "
                            f"shows ₹{actual:,.0f} crore.")
    return warnings


def _check_ratio(label: str, quoted: float, actual) -> list:
    if actual is None or actual <= 0 or quoted <= 0:
        return []
    if abs(quoted - actual) / actual > RATIO_TOLERANCE:
        return [f"Prompt says {label} about {quoted:g}, but Screener shows {actual:g}."]
    return []


def check(prompt: str, data: dict) -> list:
    """Human-readable warnings for figures in `prompt` that disagree with
    `data` (fundamentals.fetch_fundamentals output). [] when nothing to compare
    or everything matches. Never raises."""
    if not prompt or not data:
        return []
    try:
        ratios = data.get("ratios") or {}
        warnings = _check_quarter(prompt, data) + _check_growth(prompt, data) + _check_amounts(prompt, data)
        m = _PE_RE.search(prompt)
        if m:
            warnings += _check_ratio("a P/E of", float(m.group(1) or m.group(2)),
                                     thumbnail_spec._num(ratios.get("Stock P/E")))
        m = _MCAP_RE.search(prompt)
        actual_mcap = thumbnail_spec._num(ratios.get("Market Cap"))
        if m and actual_mcap:
            quoted = _num(m.group(1)) * (100000 if m.group(2).lower() == "lakh crore" else 1)
            if abs(quoted - actual_mcap) / actual_mcap > RATIO_TOLERANCE:
                warnings.append(f"Prompt says a market cap of about ₹{quoted:,.0f} crore, but Screener shows "
                                f"₹{actual_mcap:,.0f} crore.")
        m = _PRICE_RE.search(prompt)
        if m:
            warnings += _check_ratio("a share price of Rs", _num(m.group(1)),
                                     thumbnail_spec._num(ratios.get("Current Price")))
        return warnings
    except Exception as e:
        print(f"[factcheck] check failed: {type(e).__name__}: {e}")
        return []
