"""Fetches real stock fundamentals (valuation ratios + quarterly revenue/profit
trend) for a given NSE symbol by scraping Screener.in's public company page --
no API key, no paid dependency. This is what lets finance themes get CDSL/
Kalyan-Jewellers-level depth (P/E, ROE, ROCE, revenue growth) automatically,
instead of relying on a human to hand-research and hand-write those numbers
into themes.yaml.

Best-effort throughout: any parsing failure (unknown symbol, site layout
change, network issue) returns {} / "" -- callers must treat that as "no
verified fundamentals available" and fall back to the existing
"don't invent numbers" instruction, never fabricate a number in its place.
"""
import re

import requests

SCREENER_URL = "https://www.screener.in/company/{symbol}/"
SCREENER_SEARCH_URL = "https://www.screener.in/api/company/search/"
SCREENER_PEERS_URL = "https://www.screener.in/api/company/{warehouse_id}/peers/"
USER_AGENT = "Mozilla/5.0 (compatible; ytagent-fundamentals/1.0)"

RATIO_ROW_RE = re.compile(
    r'<li class="flex flex-space-between"[^>]*>\s*<span class="name">\s*(.*?)\s*</span>(.*?)</li>',
    re.DOTALL,
)
NUMBER_RE = re.compile(r"[\d,]+\.?\d*")
SYMBOL_FROM_URL_RE = re.compile(r"/company/([A-Z0-9&\-]+)/?")


def _strip_tags(html: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", html)).strip()


def _parse_ratios(html: str) -> dict:
    match = re.search(r'<ul id="top-ratios">(.*?)</ul>', html, re.DOTALL)
    if not match:
        return {}
    ratios = {}
    for name_html, value_html in RATIO_ROW_RE.findall(match.group(1)):
        name = _strip_tags(name_html)
        value = _strip_tags(value_html)
        if name and value:
            ratios[name] = value
    return ratios


def _extract_quarterly_row(table_html: str, label: str) -> list:
    pattern = re.compile(
        r'<tr[^>]*>\s*<td class="text">.*?' + re.escape(label) + r'.*?</td>(.*?)</tr>',
        re.DOTALL,
    )
    m = pattern.search(table_html)
    if not m:
        return []
    return re.findall(r'<td[^>]*>\s*([\-\d,\.]+)\s*</td>', m.group(1))


def _parse_quarterly(html: str) -> dict:
    match = re.search(r'id="quarters".*?</table>\s*</div>', html, re.DOTALL)
    if not match:
        return {}
    table = match.group(0)
    dates = re.findall(r'data-date-key="(\d{4}-\d{2}-\d{2})"', table)
    sales = _extract_quarterly_row(table, "Sales")
    # Banks and NBFCs have a "Revenue" row instead of "Sales" -- requiring
    # "Sales" dropped the whole table (profit included) for every lender
    # (seen live: Bajaj Finance's thumbnail fell back to a ROCE figure).
    financial = False
    if not sales:
        sales = _extract_quarterly_row(table, "Revenue")
        financial = bool(sales)
    if not dates or not sales:
        return {}
    data = {"quarter_dates": dates, "sales_cr": sales}
    if financial:
        data["financial"] = True
    profit = _extract_quarterly_row(table, "Net Profit")
    if profit:
        data["net_profit_cr"] = profit
    return data


def _yoy_growth(values: list):
    """% growth of the latest quarter vs. the same quarter a year ago (the
    screener.in quarterly table lists one column per quarter, so 4 quarters
    back is the year-ago comparison)."""
    if len(values) < 5:
        return None
    try:
        latest = float(values[-1].replace(",", ""))
        year_ago = float(values[-5].replace(",", ""))
    except ValueError:
        return None
    if year_ago == 0:
        return None
    return round((latest - year_ago) / abs(year_ago) * 100, 1)


# A year-on-year % beyond this almost always comes off a tiny or one-off base
# (seen live: "Net profit grew 56800.0%" for a Rs 1 crore -> Rs 567 crore
# quarter) -- technically true but misleading as a headline, so it's stated
# as "from Rs X crore to Rs Y crore" instead.
MAX_PLAIN_GROWTH_PCT = 300


def yoy_pair(values: list):
    """(year_ago, latest) floats for the latest quarter vs. the same quarter a
    year earlier, or None."""
    if len(values or []) < 5:
        return None
    try:
        return float(str(values[-5]).replace(",", "")), float(str(values[-1]).replace(",", ""))
    except ValueError:
        return None


def plain_growth_pct(values: list):
    """The YoY % only when it's a meaningful headline number: both quarters
    positive (a loss on either side makes a % misleading) and no bigger than
    MAX_PLAIN_GROWTH_PCT. Otherwise None."""
    pair = yoy_pair(values)
    if not pair or pair[0] <= 0 or pair[1] <= 0:
        return None
    pct = round((pair[1] - pair[0]) / pair[0] * 100, 1)
    return pct if abs(pct) <= MAX_PLAIN_GROWTH_PCT else None


def _cr(x: float) -> str:
    return f"{x:,.0f}" if abs(x) >= 100 else f"{x:g}"


def growth_sentence(values: list, noun: str) -> str:
    """One plain, sign-correct sentence about the YoY change, e.g. "Net profit
    grew 37.3% year-on-year", "Revenue fell 8.2% year-on-year", "Net profit
    rose from Rs 1 crore to Rs 567 crore year-on-year", "Net profit went from
    a loss of Rs 12 crore to a profit of Rs 40 crore year-on-year". "" when
    there isn't a year-ago quarter to compare with."""
    pair = yoy_pair(values)
    if not pair:
        return ""
    ago, now = pair
    pct = plain_growth_pct(values)
    if pct is not None:
        return f"{noun} {'grew' if pct >= 0 else 'fell'} {abs(pct):g}% year-on-year"
    if ago > 0 and now > 0:
        return f"{noun} {'rose' if now > ago else 'fell'} from Rs {_cr(ago)} crore to Rs {_cr(now)} crore year-on-year"

    def amount(x):
        if x > 0:
            return f"a profit of Rs {_cr(x)} crore" if noun.lower().endswith("profit") else f"Rs {_cr(x)} crore"
        if x < 0:
            return f"a loss of Rs {_cr(-x)} crore"
        return "zero"
    return f"{noun} went from {amount(ago)} to {amount(now)} year-on-year"


def profit_exceeds_revenue(data: dict) -> bool:
    """Latest quarterly net profit bigger than that quarter's revenue -- only
    possible with other income or one-off items, so profit-based "facts"
    (margin, profit per day, profit growth headline) would mislead."""
    q = (data or {}).get("quarterly") or {}
    try:
        sales = float(str(q.get("sales_cr", [])[-1]).replace(",", ""))
        profit = float(str(q.get("net_profit_cr", [])[-1]).replace(",", ""))
    except (IndexError, ValueError):
        return False
    return profit > 0 and profit > sales


ONE_OFF_PROFIT_NOTE = (
    "Note: the latest quarterly net profit is larger than the quarter's revenue, so it must include "
    "other income or one-off items -- say so plainly rather than treating it as profit from operations."
)


def _parse_warehouse_id(html: str):
    m = re.search(r'data-warehouse-id="(\d+)"', html) or re.search(r"/api/company/(\d+)/peers/", html)
    return m.group(1) if m else None


def _parse_peers(html: str, own_symbol: str = "") -> list:
    """Parses Screener's peer-comparison table into [{name, symbol, pe,
    market_cap_cr, roce, is_self}, ...]. Finds the table whose header row has
    Name plus P/E or ROCE, and matches columns by header text (not position)
    so a column reorder doesn't silently misattribute values. Returns [] when
    no recognizable table is present."""
    for table in re.findall(r"<table[^>]*>.*?</table>", html, re.DOTALL):
        headers = [_strip_tags(h).lower() for h in re.findall(r"<th[^>]*>(.*?)</th>", table, re.DOTALL)]

        def col(*needles):
            for i, h in enumerate(headers):
                if any(n in h for n in needles):
                    return i
            return None

        name_i, pe_i, cap_i, roce_i = col("name"), col("p/e"), col("mar cap"), col("roce")
        if name_i is None or (pe_i is None and roce_i is None):
            continue
        peers = _parse_peer_rows(table, own_symbol, name_i, pe_i, cap_i, roce_i)
        if peers:
            return peers
    return []


def _parse_peer_rows(table: str, own_symbol: str, name_i, pe_i, cap_i, roce_i) -> list:
    def num(cells, i):
        if i is None or i >= len(cells):
            return None
        m = re.fullmatch(r"-?[\d,]+\.?\d*", _strip_tags(cells[i]))
        return float(m.group(0).replace(",", "")) if m else None

    peers = []
    for row in re.findall(r"<tr[^>]*>(.*?)</tr>", table, re.DOTALL):
        cells = re.findall(r"<td[^>]*>(.*?)</td>", row, re.DOTALL)
        if len(cells) <= name_i:
            continue
        name = _strip_tags(cells[name_i])
        if not name or "median" in name.lower():
            continue
        link = SYMBOL_FROM_URL_RE.search(cells[name_i])
        symbol = link.group(1) if link else ""
        peers.append({
            "name": name, "symbol": symbol,
            "pe": num(cells, pe_i), "market_cap_cr": num(cells, cap_i), "roce": num(cells, roce_i),
            "is_self": bool(own_symbol) and symbol.upper() == own_symbol.upper(),
        })
    return peers


def fetch_peers(data: dict) -> list:
    """Peer table for a company returned by fetch_fundamentals (best-effort:
    [] on any failure). Uses peers already found inline on the company page,
    else Screener's separate peers endpoint keyed by the page's warehouse id
    (one extra request)."""
    data = data or {}
    if data.get("peers"):
        return data["peers"]
    warehouse_id = data.get("warehouse_id")
    if not warehouse_id:
        return []
    try:
        resp = requests.get(
            SCREENER_PEERS_URL.format(warehouse_id=warehouse_id),
            headers={"User-Agent": USER_AGENT, "X-Requested-With": "XMLHttpRequest",
                     "Referer": SCREENER_URL.format(symbol=data.get("symbol", ""))},
            timeout=15,
        )
        resp.raise_for_status()
        return _parse_peers(resp.text, data.get("symbol", ""))
    except requests.RequestException:
        return []


def fetch_fundamentals(symbol: str) -> dict:
    """Returns {} on any failure. On success:
    {"symbol": str, "ratios": {name: value, ...},
     "quarterly": {"quarter_dates": [...], "sales_cr": [...], "net_profit_cr": [...]},
     "revenue_growth_yoy_pct": float, "profit_growth_yoy_pct": float,
     "warehouse_id": str}
    (everything after "ratios" is omitted when it can't be computed/found)."""
    try:
        resp = requests.get(
            SCREENER_URL.format(symbol=symbol.upper()),
            headers={"User-Agent": USER_AGENT}, timeout=15,
        )
        resp.raise_for_status()
        html = resp.text
    except requests.RequestException:
        return {}

    ratios = _parse_ratios(html)
    if not ratios:
        return {}

    data = {"symbol": symbol.upper(), "ratios": ratios}
    warehouse_id = _parse_warehouse_id(html)
    if warehouse_id:
        data["warehouse_id"] = warehouse_id
    inline_peers = _parse_peers(html, symbol)
    if inline_peers:
        data["peers"] = inline_peers
    quarterly = _parse_quarterly(html)
    if quarterly:
        data["quarterly"] = quarterly
        revenue_growth = _yoy_growth(quarterly.get("sales_cr", []))
        profit_growth = _yoy_growth(quarterly.get("net_profit_cr", []))
        if revenue_growth is not None:
            data["revenue_growth_yoy_pct"] = revenue_growth
        if profit_growth is not None:
            data["profit_growth_yoy_pct"] = profit_growth
    return data


def search_company(company_name: str) -> dict:
    """Resolves a plain company name (e.g. "Semiconductor") to Screener.in's
    best-matching company via its free, no-API-key company-search endpoint --
    the same site this module already scrapes for fundamentals, just its
    search index instead of a company page. Returns {"symbol": str, "name":
    str} for the first match, or {} on no match or any request/parse
    failure. The "name" is Screener's own canonical company name (e.g. "RRP
    Semiconductor Ltd") -- often not identical to the caller's search query,
    since a generic/sector-wide query (e.g. "Semiconductor Stocks") can
    resolve to one specific, differently-named real company; callers must
    surface this real name wherever they name the company (title, narration)
    rather than echo their own guess. Callers must still verify the returned
    symbol by calling fetch_fundamentals on it before trusting it (a
    name-search hit only confirms a company with that name exists, not that
    the caller's inferred name was exactly right)."""
    try:
        resp = requests.get(
            SCREENER_SEARCH_URL,
            params={"q": company_name},
            headers={"User-Agent": USER_AGENT}, timeout=10,
        )
        resp.raise_for_status()
        results = resp.json()
    except (requests.RequestException, ValueError):
        return {}

    if not results:
        return {}
    m = SYMBOL_FROM_URL_RE.search(results[0].get("url", ""))
    if not m:
        return {}
    return {"symbol": m.group(1), "name": (results[0].get("name") or "").strip()}


def search_company_symbol(company_name: str) -> str:
    """Thin wrapper over search_company for callers that only need the
    ticker -- see search_company's docstring for match semantics."""
    return search_company(company_name).get("symbol", "")


def market_cap_cr(data: dict) -> float:
    """Parses the already-scraped "Market Cap" ratio (e.g. "₹ 26,334 Cr.") into
    a plain crore float, so callers can rank candidates by how widely-recognized
    a company is without a new API/data source. Returns 0.0 when the ratio is
    missing or unparseable -- never raises, so a scrape-format change degrades
    ranking instead of breaking topic selection."""
    value = (data.get("ratios") or {}).get("Market Cap", "")
    match = NUMBER_RE.search(value)
    if not match:
        return 0.0
    try:
        return float(match.group(0).replace(",", ""))
    except ValueError:
        return 0.0


def format_fundamentals_facts(data: dict) -> str:
    """Formats fetched fundamentals into the same "- fact" bullet-list shape
    research.fetch_grounding_facts() produces, so callers can drop it straight
    into a grounding-facts prompt block."""
    if not data:
        return ""
    lines = [f"- {name}: {value}" for name, value in data["ratios"].items()]

    quarterly = data.get("quarterly", {})
    dates = quarterly.get("quarter_dates", [])
    if quarterly.get("sales_cr"):
        suffix = f" (quarter ending {dates[-1]})" if dates else ""
        lines.append(f"- Latest quarterly revenue: Rs {quarterly['sales_cr'][-1]} crore{suffix}")
    if quarterly.get("net_profit_cr"):
        lines.append(f"- Latest quarterly net profit: Rs {quarterly['net_profit_cr'][-1]} crore")
    for key, noun in (("sales_cr", "Revenue"), ("net_profit_cr", "Net profit")):
        sentence = growth_sentence(quarterly.get(key, []), noun)
        if sentence:
            lines.append(f"- {sentence}")
    if profit_exceeds_revenue(data):
        lines.append(f"- {ONE_OFF_PROFIT_NOTE}")

    return "\n".join(lines)
