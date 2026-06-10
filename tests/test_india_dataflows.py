"""Tests for India dataflow modules: news, flows, shareholding, corporate actions.

All tests are unit tests — no network calls. External calls are patched.
"""

import unittest
import xml.etree.ElementTree as ET
from datetime import date
from unittest.mock import MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# india_news tests
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestIndiaNewsRssParsing(unittest.TestCase):
    """Test RSS item parsing in india_news without network calls."""

    def _make_rss(self, items: list[dict]) -> ET.Element:
        """Build a minimal RSS 2.0 XML tree."""
        channel = ET.Element("channel")
        for item in items:
            el = ET.SubElement(channel, "item")
            for tag, text in item.items():
                sub = ET.SubElement(el, tag)
                sub.text = text
        root = ET.Element("rss")
        root.append(channel)
        return root

    def test_parses_title_and_description(self):
        from tradingagents.dataflows.india.india_news import _parse_items
        root = self._make_rss([
            {"title": "RBI hikes rate", "description": "RBI raised repo rate by 25bp", "pubDate": ""},
        ])
        items = _parse_items(root)
        self.assertEqual(len(items), 1)
        self.assertEqual(items[0]["title"], "RBI hikes rate")
        self.assertIn("repo rate", items[0]["desc"])

    def test_deduplicates_empty_title(self):
        from tradingagents.dataflows.india.india_news import _parse_items
        root = self._make_rss([
            {"title": "", "description": "something"},
        ])
        items = _parse_items(root)
        self.assertEqual(len(items), 0, "Items with empty title should be excluded")

    def test_html_stripped_from_description(self):
        from tradingagents.dataflows.india.india_news import _clean
        raw = "<p>RBI <b>hikes</b> rate &amp; 25bp</p>"
        result = _clean(raw)
        self.assertNotIn("<p>", result)
        self.assertNotIn("<b>", result)
        self.assertIn("hikes", result)
        self.assertIn("&", result)  # html.unescape applied

    def test_keyword_match_case_insensitive(self):
        from tradingagents.dataflows.india.india_news import _keyword_match
        self.assertTrue(_keyword_match("Reliance Q1 results", "", ["reliance"]))
        self.assertTrue(_keyword_match("INFY outperforms", "", ["Infy"]))
        self.assertFalse(_keyword_match("Wipro wins deal", "", ["tcs"]))

    def test_get_india_global_news_no_feeds_available(self):
        """When all feeds fail, returns a graceful 'no news found' string."""
        from tradingagents.dataflows.india.india_news import get_india_global_news
        with patch("tradingagents.dataflows.india.india_news._fetch_rss", return_value=None):
            result = get_india_global_news("2026-06-10")
        self.assertIsInstance(result, str)
        # Either "unavailable" or "No ... news found" is acceptable graceful degradation
        degraded = "unavailable" in result.lower() or "no " in result.lower()
        self.assertTrue(degraded, f"Expected graceful degradation message, got: {result[:200]}")

    def test_get_india_news_no_feeds(self):
        from tradingagents.dataflows.india.india_news import get_india_news
        with patch("tradingagents.dataflows.india.india_news._fetch_rss", return_value=None):
            with patch("tradingagents.dataflows.india.india_news._resolve_company_name", return_value="Reliance Industries"):
                result = get_india_news("RELIANCE.NS", "2026-06-01", "2026-06-10")
        self.assertIsInstance(result, str)
        self.assertIn("No Indian news found", result)


# ---------------------------------------------------------------------------
# flows tests
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestFiiDiiFlows(unittest.TestCase):
    def test_returns_string(self):
        from tradingagents.dataflows.india.flows import get_fii_dii_flows
        with patch("tradingagents.dataflows.india.flows._fetch_flows_raw", return_value=None):
            result = get_fii_dii_flows(as_of=date(2026, 6, 10), sessions=5)
        self.assertIsInstance(result, str)

    def test_fallback_contains_unavailable_per_row(self):
        from tradingagents.dataflows.india.flows import get_fii_dii_flows
        with patch("tradingagents.dataflows.india.flows._fetch_flows_raw", return_value=None):
            result = get_fii_dii_flows(as_of=date(2026, 6, 10), sessions=3)
        # Each row should say "data unavailable"
        self.assertGreater(result.count("data unavailable"), 1)

    def test_parses_real_data_rows(self):
        from tradingagents.dataflows.india.flows import get_fii_dii_flows
        fake_data = [
            {"date": "2026-06-10", "fii_netPurchase_sales": "1234.56", "dii_netPurchase_sales": "-567.89"},
            {"date": "2026-06-09", "fii_netPurchase_sales": "-200.00", "dii_netPurchase_sales": "300.00"},
        ]
        with patch("tradingagents.dataflows.india.flows._fetch_flows_raw", return_value=fake_data):
            result = get_fii_dii_flows(as_of=date(2026, 6, 10), sessions=5)
        self.assertIn("1234.56", result)
        self.assertIn("-200.00", result)

    def test_header_mentions_sessions(self):
        from tradingagents.dataflows.india.flows import get_fii_dii_flows
        with patch("tradingagents.dataflows.india.flows._fetch_flows_raw", return_value=None):
            result = get_fii_dii_flows(as_of=date(2026, 6, 10), sessions=7)
        self.assertIn("7", result)


# ---------------------------------------------------------------------------
# shareholding tests
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestShareholdingSummary(unittest.TestCase):
    def test_returns_string_on_empty_info(self):
        from tradingagents.dataflows.india.shareholding import get_shareholding_summary
        with patch("tradingagents.dataflows.india.shareholding.yf.Ticker") as mock:
            mock.return_value.info = {}
            mock.return_value.institutional_holders = None
            result = get_shareholding_summary("RELIANCE.NS")
        self.assertIsInstance(result, str)
        self.assertIn("unavailable", result.lower())

    def test_returns_holding_pct_when_available(self):
        from tradingagents.dataflows.india.shareholding import get_shareholding_summary
        with patch("tradingagents.dataflows.india.shareholding.yf.Ticker") as mock:
            mock.return_value.info = {
                "longName": "Reliance Industries Limited",
                "heldPercentInsiders": 0.4965,
                "heldPercentInstitutions": 0.2100,
                "sharesOutstanding": 6768000000,
            }
            mock.return_value.institutional_holders = None
            result = get_shareholding_summary("RELIANCE.NS")
        self.assertIn("49.6", result)   # 0.4965 * 100
        self.assertIn("21.0", result)   # 0.2100 * 100

    def test_pledge_always_marked_unavailable(self):
        from tradingagents.dataflows.india.shareholding import get_shareholding_summary
        with patch("tradingagents.dataflows.india.shareholding.yf.Ticker") as mock:
            mock.return_value.info = {"longName": "Test Corp"}
            mock.return_value.institutional_holders = None
            result = get_shareholding_summary("TEST.NS")
        self.assertIn("pledge", result.lower())
        self.assertIn("unavailable", result.lower())

    def test_exception_returns_graceful_string(self):
        from tradingagents.dataflows.india.shareholding import get_shareholding_summary
        with patch("tradingagents.dataflows.india.shareholding.yf.Ticker", side_effect=RuntimeError("timeout")):
            result = get_shareholding_summary("UNKNOWN.NS")
        self.assertIsInstance(result, str)
        self.assertIn("unavailable", result.lower())


# ---------------------------------------------------------------------------
# corporate_actions tests
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestCorporateActions(unittest.TestCase):
    def test_returns_string(self):
        from tradingagents.dataflows.india.corporate_actions import get_corporate_actions
        with patch("tradingagents.dataflows.india.corporate_actions.yf.Ticker") as mock:
            mock.return_value.calendar = {}
            mock.return_value.dividends = MagicMock(empty=True)
            mock.return_value.splits = MagicMock(empty=True)
            result = get_corporate_actions("RELIANCE.NS", as_of=date(2026, 6, 10))
        self.assertIsInstance(result, str)

    def test_flags_earnings_inside_window(self):
        import pandas as pd
        from tradingagents.dataflows.india.corporate_actions import get_corporate_actions
        with patch("tradingagents.dataflows.india.corporate_actions.yf.Ticker") as mock:
            # Earnings date 5 days from now (inside 20-day window)
            earn_date = date(2026, 6, 15)
            mock.return_value.calendar = {"Earnings Date": earn_date}
            mock.return_value.dividends = MagicMock(empty=True)
            mock.return_value.splits = MagicMock(empty=True)
            result = get_corporate_actions("RELIANCE.NS", as_of=date(2026, 6, 10))
        self.assertIn("INSIDE SWING WINDOW", result)

    def test_no_flag_for_earnings_outside_window(self):
        from tradingagents.dataflows.india.corporate_actions import get_corporate_actions
        with patch("tradingagents.dataflows.india.corporate_actions.yf.Ticker") as mock:
            # Earnings date 60 days from now (outside 20-day window)
            earn_date = date(2026, 8, 10)
            mock.return_value.calendar = {"Earnings Date": earn_date}
            mock.return_value.dividends = MagicMock(empty=True)
            mock.return_value.splits = MagicMock(empty=True)
            result = get_corporate_actions("RELIANCE.NS", as_of=date(2026, 6, 10))
        self.assertNotIn("INSIDE SWING WINDOW", result)

    def test_exception_returns_graceful_string(self):
        from tradingagents.dataflows.india.corporate_actions import get_corporate_actions
        with patch("tradingagents.dataflows.india.corporate_actions.yf.Ticker", side_effect=RuntimeError("error")):
            result = get_corporate_actions("BAD.NS", as_of=date(2026, 6, 10))
        self.assertIsInstance(result, str)
        self.assertIn("unavailable", result.lower())


# ---------------------------------------------------------------------------
# screener_fundamentals tests
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestIndiaFundamentals(unittest.TestCase):
    def _make_info(self, **overrides):
        base = {
            "longName": "Reliance Industries Limited",
            "sector": "Energy",
            "industry": "Oil & Gas Refining & Marketing",
            "exchange": "NSE",
            "currency": "INR",
            "marketCap": 2_000_000_000_000,
            "trailingPE": 25.5,
            "priceToBook": 2.1,
            "returnOnEquity": 0.15,
            "debtToEquity": 40.0,
            "currentRatio": 1.5,
        }
        base.update(overrides)
        return base

    def test_returns_formatted_string(self):
        from tradingagents.dataflows.india.screener_fundamentals import get_india_fundamentals
        with patch("tradingagents.dataflows.india.screener_fundamentals.yf.Ticker") as mock:
            mock.return_value.info = self._make_info()
            result = get_india_fundamentals("RELIANCE.NS")
        self.assertIsInstance(result, str)
        self.assertIn("Reliance Industries", result)

    def test_market_cap_in_crores(self):
        from tradingagents.dataflows.india.screener_fundamentals import get_india_fundamentals
        with patch("tradingagents.dataflows.india.screener_fundamentals.yf.Ticker") as mock:
            mock.return_value.info = self._make_info(marketCap=2_000_000_000_000)
            result = get_india_fundamentals("RELIANCE.NS")
        # 2_000_000_000_000 / 1e7 = 200000 Cr
        self.assertIn("Cr", result)

    def test_empty_info_returns_unavailable(self):
        from tradingagents.dataflows.india.screener_fundamentals import get_india_fundamentals
        with patch("tradingagents.dataflows.india.screener_fundamentals.yf.Ticker") as mock:
            mock.return_value.info = {}
            result = get_india_fundamentals("UNKNOWN.NS")
        self.assertIn("unavailable", result.lower())

    def test_roe_below_10_gets_note(self):
        from tradingagents.dataflows.india.screener_fundamentals import get_india_fundamentals
        with patch("tradingagents.dataflows.india.screener_fundamentals.yf.Ticker") as mock:
            mock.return_value.info = self._make_info(returnOnEquity=0.05)
            result = get_india_fundamentals("WEAK.NS")
        self.assertIn("NOTE", result)

    def test_high_debt_caution(self):
        from tradingagents.dataflows.india.screener_fundamentals import get_india_fundamentals
        with patch("tradingagents.dataflows.india.screener_fundamentals.yf.Ticker") as mock:
            mock.return_value.info = self._make_info(debtToEquity=200.0)
            result = get_india_fundamentals("DEBT.NS")
        self.assertIn("CAUTION", result)

    def test_promoter_pledge_note_present(self):
        from tradingagents.dataflows.india.screener_fundamentals import get_india_fundamentals
        with patch("tradingagents.dataflows.india.screener_fundamentals.yf.Ticker") as mock:
            mock.return_value.info = self._make_info()
            result = get_india_fundamentals("RELIANCE.NS")
        self.assertIn("pledge", result.lower())

    def test_currency_note_present(self):
        from tradingagents.dataflows.india.screener_fundamentals import get_india_fundamentals
        with patch("tradingagents.dataflows.india.screener_fundamentals.yf.Ticker") as mock:
            mock.return_value.info = self._make_info()
            result = get_india_fundamentals("RELIANCE.NS")
        self.assertIn("INR", result)


# ---------------------------------------------------------------------------
# NSE client tests (no network)
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestNseClientHelpers(unittest.TestCase):
    def test_nse_symbol_strips_ns(self):
        from tradingagents.dataflows.india.nse_client import _nse_symbol
        self.assertEqual(_nse_symbol("RELIANCE.NS"), "RELIANCE")

    def test_nse_symbol_strips_bo(self):
        from tradingagents.dataflows.india.nse_client import _nse_symbol
        self.assertEqual(_nse_symbol("INFY.BO"), "INFY")

    def test_nse_symbol_passthrough(self):
        from tradingagents.dataflows.india.nse_client import _nse_symbol
        self.assertEqual(_nse_symbol("RELIANCE"), "RELIANCE")

    def test_yf_canonical_appends_ns(self):
        from tradingagents.dataflows.india.nse_client import _yf_canonical
        self.assertEqual(_yf_canonical("RELIANCE"), "RELIANCE.NS")

    def test_yf_canonical_preserves_ns(self):
        from tradingagents.dataflows.india.nse_client import _yf_canonical
        self.assertEqual(_yf_canonical("RELIANCE.NS"), "RELIANCE.NS")

    def test_get_fno_ban_returns_unavailable_on_failure(self):
        from tradingagents.dataflows.india.nse_client import get_fno_ban_list
        with patch("tradingagents.dataflows.india.nse_client._fetch_nse", return_value=None):
            with patch("tradingagents.dataflows.india.nse_client._read_cache", return_value=None):
                result = get_fno_ban_list(as_of=date(2026, 6, 10))
        self.assertIsInstance(result, str)
        self.assertIn("unavailable", result.lower())

    def test_get_bulk_block_deals_returns_no_deals_on_empty(self):
        from tradingagents.dataflows.india.nse_client import get_bulk_block_deals
        with patch("tradingagents.dataflows.india.nse_client._fetch_nse", return_value={"data": []}):
            with patch("tradingagents.dataflows.india.nse_client._read_cache", return_value=None):
                with patch("tradingagents.dataflows.india.nse_client._write_cache"):
                    result = get_bulk_block_deals(as_of=date(2026, 6, 10))
        self.assertIsInstance(result, str)
        self.assertIn("No bulk", result)


# ---------------------------------------------------------------------------
# Vendor routing tests
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestVendorRoutingIndia(unittest.TestCase):
    def test_get_vendor_india_fundamental_data(self):
        from tradingagents.dataflows.interface import get_vendor
        self.assertEqual(
            get_vendor("fundamental_data", ticker="RELIANCE.NS"),
            "india_composite",
        )

    def test_get_vendor_us_fundamental_data(self):
        from tradingagents.dataflows.interface import get_vendor
        from tradingagents.dataflows.config import get_config, set_config
        # Ensure default config is active
        set_config({"market_profile": "auto"})
        result = get_vendor("fundamental_data", ticker="AAPL")
        self.assertEqual(result, "yfinance")

    def test_get_vendor_india_news_data(self):
        from tradingagents.dataflows.interface import get_vendor
        self.assertEqual(
            get_vendor("news_data", ticker="INFY.NS"),
            "india_news",
        )

    def test_get_vendor_us_news_data(self):
        from tradingagents.dataflows.interface import get_vendor
        from tradingagents.dataflows.config import set_config
        set_config({"market_profile": "auto"})
        result = get_vendor("news_data", ticker="AAPL")
        self.assertEqual(result, "yfinance")

    def test_get_vendor_explicit_india_profile_no_ticker(self):
        """market_profile='india' forces India routing even without a ticker."""
        from tradingagents.dataflows.interface import get_vendor
        from tradingagents.dataflows.config import set_config
        set_config({"market_profile": "india"})
        result = get_vendor("news_data")
        self.assertEqual(result, "india_news")

    def test_get_vendor_resets_to_us_after_profile_change(self):
        from tradingagents.dataflows.interface import get_vendor
        from tradingagents.dataflows.config import set_config
        set_config({"market_profile": "auto"})
        result = get_vendor("news_data", ticker="AAPL")
        self.assertEqual(result, "yfinance")


# ---------------------------------------------------------------------------
# Config tests
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestIndiaConfig(unittest.TestCase):
    def test_default_market_profile_is_auto(self):
        from tradingagents.default_config import DEFAULT_CONFIG
        self.assertEqual(DEFAULT_CONFIG.get("market_profile"), "auto")

    def test_india_news_queries_present(self):
        from tradingagents.default_config import DEFAULT_CONFIG
        queries = DEFAULT_CONFIG.get("global_news_queries_india")
        self.assertIsNotNone(queries)
        self.assertIsInstance(queries, list)
        self.assertGreater(len(queries), 0)
        # Should include RBI (most important Indian macro signal)
        self.assertTrue(any("RBI" in q for q in queries))

    def test_india_reddit_subreddits_present(self):
        from tradingagents.default_config import DEFAULT_CONFIG
        subs = DEFAULT_CONFIG.get("india_reddit_subreddits")
        self.assertIsNotNone(subs)
        self.assertIsInstance(subs, list)
        self.assertIn("IndianStockMarket", subs)

    def test_us_news_queries_still_present(self):
        from tradingagents.default_config import DEFAULT_CONFIG
        queries = DEFAULT_CONFIG.get("global_news_queries")
        self.assertIsNotNone(queries)
        self.assertTrue(any("Federal Reserve" in q for q in queries))


if __name__ == "__main__":
    unittest.main()
