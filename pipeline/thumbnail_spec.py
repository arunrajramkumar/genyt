"""What goes on a stock video's thumbnail, built only from scraped fundamentals.

The old thumbnail scraped its stat out of the scenes' on-screen text and the
LLM-written title, which put junk on the most-seen image of the video (seen
live: "CRORE | 69.7% YEAR-ON-YEAR | ₹843" as a label, a "117%" ROCE copied
from a prompt example when the real figure was 17.5%, and a five-line title
with the company name twice). Here every number comes straight from
fundamentals.fetch_fundamentals and every label is fixed wording, so nothing
on the thumbnail can be invented.

Layout data only -- textcard.render_stock_thumbnail draws it.
"""
import re
from datetime import date

from . import fundamentals, segmented_script

MAX_CHIPS = 2
# A results tag only makes sense while the quarter is fresh news.
RESULTS_TAG_MAX_AGE_DAYS = 120


def _num(value):
    m = re.search(r"-?\d[\d,]*\.?\d*", str(value or ""))
    try:
        return float(m.group(0).replace(",", "")) if m else None
    except ValueError:
        return None


def _fmt_pct(x: float) -> str:
    """Same precision the narration/title use ("+140%", "-21.7%"), so the
    thumbnail never shows a differently-rounded figure."""
    return f"{round(x, 1):+g}%"


def fiscal_quarter_label(quarter_end: str) -> str:
    """"2026-09-30" -> "Q2 FY27" (Indian fiscal year: April-March)."""
    try:
        d = date.fromisoformat(quarter_end)
    except (TypeError, ValueError):
        return ""
    quarter = {6: 1, 9: 2, 12: 3, 3: 4}.get(d.month)
    if not quarter:
        return ""
    fy = d.year + 1 if d.month >= 4 else d.year
    return f"Q{quarter} FY{fy % 100:02d}"


def _tag(data: dict, title: str, today: date) -> str:
    if re.search(r"\bIPO\b", title or "", re.IGNORECASE):
        return "IPO"
    dates = (data.get("quarterly") or {}).get("quarter_dates") or []
    if not dates:
        return ""
    label = fiscal_quarter_label(dates[-1])
    try:
        age = (today - date.fromisoformat(dates[-1])).days
    except ValueError:
        return ""
    return f"{label} RESULTS" if label and 0 <= age <= RESULTS_TAG_MAX_AGE_DAYS else ""


def _hero(data: dict):
    """(value, label) for the big number: YoY profit growth when it's a clean
    figure, else YoY revenue growth, else ROCE/ROE. Declines are kept -- a
    falling profit is news too, drawn in red."""
    quarterly = data.get("quarterly") or {}
    profit = fundamentals.plain_growth_pct(quarterly.get("net_profit_cr", []))
    if profit is not None and not fundamentals.profit_exceeds_revenue(data):
        return _fmt_pct(profit), "NET PROFIT YoY"
    revenue = fundamentals.plain_growth_pct(quarterly.get("sales_cr", []))
    if revenue is not None:
        return _fmt_pct(revenue), "REVENUE YoY"
    ratios = data.get("ratios") or {}
    for key in _ratio_order(data):
        value = _num(ratios.get(key))
        if value is not None and value > 0:
            return f"{value:g}%", key
    return "", ""


def _ratio_order(data: dict) -> tuple:
    """ROE is the return measure that means something for a lender; ROCE for
    everyone else."""
    return ("ROE", "ROCE") if (data.get("quarterly") or {}).get("financial") else ("ROCE", "ROE")


def _chips(data: dict, hero_label: str) -> list:
    ratios = data.get("ratios") or {}
    chips = []
    pe = _num(ratios.get("Stock P/E"))
    if pe is not None and pe > 0:
        chips.append(f"P/E {pe:g}")
    for key in _ratio_order(data):
        value = _num(ratios.get(key))
        if key != hero_label and value is not None and value > 0:
            chips.append(f"{key} {value:g}%")
            break
    return chips[:MAX_CHIPS]


def _hook(title: str, hero_value: str) -> str:
    """A short question for the bottom line: the title's own closing question
    when it's short enough to read at thumbnail size, else a fixed neutral one
    that matches the direction of the hero number."""
    tail = re.split(r"[:|–—]\s*", title or "")[-1].strip()
    if tail.endswith("?") and len(tail.split()) <= 5 and not re.search(r"\d", tail):
        return tail.upper()
    if hero_value.startswith("-"):
        return "WHAT HAPPENED?"
    return "WHAT'S BEHIND IT?"


def build(data: dict, company_name: str, title: str = "", today: date = None) -> dict:
    """Thumbnail spec, or {} when there's nothing solid to headline (callers
    then fall back to the generic hook card)."""
    if not data or not company_name:
        return {}
    value, label = _hero(data)
    if not value:
        return {}
    return {
        "name": segmented_script._short_company_name(company_name).upper(),
        "tag": _tag(data, title, today or date.today()),
        "hero_value": value,
        "hero_label": label,
        # Arrows only for growth figures; a ratio hero (ROCE/ROE) gets none.
        "direction": ("down" if value.startswith("-") else "up") if label.endswith("YoY") else "",
        "chips": _chips(data, label),
        "hook": _hook(title, value),
    }


_RUPEE = r"(?:₹|Rs\.?|INR)\s?"
_AMT = r"(\d(?:[\d,]*\d)?)"  # "1,450" but never a trailing sentence comma
_PRICE_BAND_RE = re.compile(rf"price band[^₹\d]{{0,30}}{_RUPEE}{_AMT}\s*(?:to|-|–|—)\s*(?:{_RUPEE})?{_AMT}", re.IGNORECASE)
_GMP_RE = re.compile(r"(?:grey[- ]market premium|GMP)[^.%\d]{0,40}?(\d+(?:\.\d+)?)\s*%", re.IGNORECASE)
_LOT_RE = re.compile(r"lot (?:size )?of (\d+) shares", re.IGNORECASE)
_ISSUE_SIZE_RE = re.compile(rf"issue size[^₹\d]{{0,40}}{_RUPEE}{_AMT}\s*(?:crore|cr\b)", re.IGNORECASE)


def build_ipo(prompt: str, company_name: str, title: str = "") -> dict:
    """Thumbnail spec for an IPO video from the numbers in the prompt itself --
    an unlisted IPO has no Screener page, and the prompt is this video's
    grounding source. Price band is the hero; GMP / lot size / issue size fill
    the chips. {} when the prompt isn't about an IPO or has no price band."""
    if not company_name or not re.search(r"\bIPO\b", prompt or ""):
        return {}
    band = _PRICE_BAND_RE.search(prompt)
    if not band:
        return {}
    chips = []
    gmp, lot, size = _GMP_RE.search(prompt), _LOT_RE.search(prompt), _ISSUE_SIZE_RE.search(prompt)
    if gmp:
        chips.append(f"GMP ~{gmp.group(1)}%")
    if size:
        chips.append(f"₹{size.group(1)} CR ISSUE")
    if lot and len(chips) < MAX_CHIPS:
        chips.append(f"LOT {lot.group(1)} SHARES")
    tail = re.split(r"[:|–—]\s*", title or "")[-1].strip()
    hook = tail.upper() if tail.endswith("?") and len(tail.split()) <= 5 and not re.search(r"\d", tail) else "WHAT'S IT WORTH?"
    return {
        "name": segmented_script._short_company_name(company_name).upper(),
        "tag": "IPO",
        "hero_value": f"₹{band.group(1)}–{band.group(2)}",
        "hero_label": "PRICE BAND",
        "direction": "",
        "chips": chips[:MAX_CHIPS],
        "hook": hook,
    }
