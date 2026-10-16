from unittest import mock

from pipeline import fundamentals


def _fake_html(include_quarterly=True):
    ratios = """
    <ul id="top-ratios">
      <li class="flex flex-space-between" data-source="default">
        <span class="name">Market Cap</span>
        <span class="nowrap value">₹ <span class="number">26,334</span> Cr.</span>
      </li>
      <li class="flex flex-space-between" data-source="default">
        <span class="name">Stock P/E</span>
        <span class="nowrap value"><span class="number">57.2</span></span>
      </li>
      <li class="flex flex-space-between" data-source="default">
        <span class="name">ROE</span>
        <span class="nowrap value"><span class="number">31.3</span> %</span>
      </li>
    </ul>
    """
    quarterly = ""
    if include_quarterly:
        quarterly = """
        <div id="quarters" class="card card-large">
        <table class="data-table">
          <thead><tr>
            <th data-date-key="2025-06-30">Jun 2025</th>
            <th data-date-key="2025-09-30">Sep 2025</th>
            <th data-date-key="2025-12-31">Dec 2025</th>
            <th data-date-key="2026-03-31">Mar 2026</th>
            <th data-date-key="2026-06-30">Jun 2026</th>
          </tr></thead>
          <tbody>
            <tr class="stripe">
              <td class="text"><button>Sales&nbsp;<span>+</span></button></td>
              <td>198</td><td>248</td><td>221</td><td>182</td><td>222</td>
            </tr>
            <tr class="stripe">
              <td class="text"><button>Net Profit&nbsp;<span>+</span></button></td>
              <td>105</td><td>171</td><td>105</td><td>81</td><td>152</td>
            </tr>
          </tbody>
        </table>
        </div>
        """
    return f"<html><body>{ratios}{quarterly}</body></html>"


def _mock_response(html, status_ok=True):
    resp = mock.Mock()
    resp.text = html
    if status_ok:
        resp.raise_for_status = lambda: None
    else:
        resp.raise_for_status = mock.Mock(side_effect=fundamentals.requests.RequestException("boom"))
    return resp


def test_fetch_fundamentals_parses_ratios_and_quarterly_growth(monkeypatch):
    monkeypatch.setattr(fundamentals.requests, "get", lambda *a, **k: _mock_response(_fake_html()))
    data = fundamentals.fetch_fundamentals("cdsl")

    assert data["symbol"] == "CDSL"
    assert data["ratios"]["Market Cap"] == "₹ 26,334 Cr."
    assert data["ratios"]["Stock P/E"] == "57.2"
    assert data["ratios"]["ROE"] == "31.3 %"
    assert data["quarterly"]["sales_cr"] == ["198", "248", "221", "182", "222"]
    # latest quarter (222) vs. the same quarter a year ago (198)
    assert data["revenue_growth_yoy_pct"] == round((222 - 198) / 198 * 100, 1)
    assert data["profit_growth_yoy_pct"] == round((152 - 105) / 105 * 100, 1)


def test_fetch_fundamentals_omits_growth_when_fewer_than_5_quarters(monkeypatch):
    html = _fake_html().replace('<td data-date-key="2025-06-30">Jun 2025</td>', "")
    # Simulate too few quarters by truncating the quarterly table entirely.
    html = _fake_html(include_quarterly=False)
    monkeypatch.setattr(fundamentals.requests, "get", lambda *a, **k: _mock_response(html))
    data = fundamentals.fetch_fundamentals("cdsl")

    assert "quarterly" not in data
    assert "revenue_growth_yoy_pct" not in data


def test_fetch_fundamentals_returns_empty_dict_when_ratios_missing(monkeypatch):
    monkeypatch.setattr(fundamentals.requests, "get", lambda *a, **k: _mock_response("<html>no ratios here</html>"))
    assert fundamentals.fetch_fundamentals("unknownco") == {}


def test_fetch_fundamentals_returns_empty_dict_on_request_failure(monkeypatch):
    def raise_error(*a, **k):
        raise fundamentals.requests.RequestException("network down")
    monkeypatch.setattr(fundamentals.requests, "get", raise_error)
    assert fundamentals.fetch_fundamentals("cdsl") == {}


# -- market_cap_cr: parses the scraped "Market Cap" ratio for ranking --------

def test_market_cap_cr_parses_rupee_crore_string():
    data = {"ratios": {"Market Cap": "₹ 26,334 Cr."}}
    assert fundamentals.market_cap_cr(data) == 26334.0


def test_market_cap_cr_returns_zero_when_ratio_missing():
    assert fundamentals.market_cap_cr({"ratios": {"Stock P/E": "57.2"}}) == 0.0


def test_market_cap_cr_returns_zero_when_ratios_missing_entirely():
    assert fundamentals.market_cap_cr({"symbol": "CDSL"}) == 0.0


def test_format_fundamentals_facts_returns_empty_string_for_empty_data():
    assert fundamentals.format_fundamentals_facts({}) == ""


def test_format_fundamentals_facts_includes_ratios_and_growth():
    data = {
        "symbol": "CDSL",
        "ratios": {"Stock P/E": "57.2", "ROE": "31.3 %"},
        "quarterly": {"quarter_dates": ["2026-06-30"], "sales_cr": ["222"], "net_profit_cr": ["152"]},
        "revenue_growth_yoy_pct": 12.1,
        "profit_growth_yoy_pct": 44.8,
    }
    text = fundamentals.format_fundamentals_facts(data)

    assert "- Stock P/E: 57.2" in text
    assert "- ROE: 31.3 %" in text
    assert "- Latest quarterly revenue: Rs 222 crore (quarter ending 2026-06-30)" in text
    assert "- Latest quarterly net profit: Rs 152 crore" in text
    assert "- Revenue grew 12.1% year-on-year" in text
    assert "- Net profit grew 44.8% year-on-year" in text
