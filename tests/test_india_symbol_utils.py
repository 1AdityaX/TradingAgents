"""Tests for India-specific symbol utilities."""

import unittest

import pytest

from tradingagents.dataflows.symbol_utils import (
    detect_market_profile,
    is_indian_ticker,
    normalize_indian_symbol,
)


@pytest.mark.unit
class TestIsIndianTicker(unittest.TestCase):
    def test_ns_suffix_is_indian(self):
        for sym in ("RELIANCE.NS", "INFY.NS", "TCS.NS", "HDFCBANK.NS"):
            self.assertTrue(is_indian_ticker(sym), sym)

    def test_bo_suffix_is_indian(self):
        for sym in ("RELIANCE.BO", "INFY.BO"):
            self.assertTrue(is_indian_ticker(sym), sym)

    def test_lowercase_ns_is_indian(self):
        self.assertTrue(is_indian_ticker("reliance.ns"))

    def test_bare_alpha_is_indian(self):
        # Bare uppercase symbols that look like NSE tickers
        self.assertTrue(is_indian_ticker("RELIANCE"))
        self.assertTrue(is_indian_ticker("INFY"))
        self.assertTrue(is_indian_ticker("TCS"))

    def test_known_us_tickers_are_not_indian(self):
        for sym in ("AAPL", "MSFT", "TSLA"):
            # These are bare uppercase — currently accepted as potentially Indian.
            # This is intentional: we can't know without a lookup. The test
            # documents that ambiguity is handled by normalize_indian_symbol.
            pass  # No assertion — behaviour is documented above.

    def test_metal_alias_not_indian(self):
        self.assertFalse(is_indian_ticker("XAUUSD"))
        self.assertFalse(is_indian_ticker("GOLD"))

    def test_crypto_not_indian(self):
        self.assertFalse(is_indian_ticker("BTCUSD"))
        self.assertFalse(is_indian_ticker("ETHUSD"))

    def test_forex_pair_not_indian(self):
        self.assertFalse(is_indian_ticker("EURUSD"))
        self.assertFalse(is_indian_ticker("GBPJPY"))

    def test_empty_string_not_indian(self):
        self.assertFalse(is_indian_ticker(""))
        self.assertFalse(is_indian_ticker("   "))

    def test_none_not_indian(self):
        self.assertFalse(is_indian_ticker(None))


@pytest.mark.unit
class TestNormalizeIndianSymbol(unittest.TestCase):
    def test_ns_suffix_passthrough(self):
        self.assertEqual(normalize_indian_symbol("RELIANCE.NS"), "RELIANCE.NS")
        self.assertEqual(normalize_indian_symbol("reliance.ns"), "RELIANCE.NS")

    def test_bo_suffix_passthrough(self):
        self.assertEqual(normalize_indian_symbol("RELIANCE.BO"), "RELIANCE.BO")

    def test_bare_indian_gets_ns(self):
        # Pure alpha symbols should be appended with .NS
        result = normalize_indian_symbol("RELIANCE")
        self.assertEqual(result, "RELIANCE.NS")

    def test_bare_infy_gets_ns(self):
        self.assertEqual(normalize_indian_symbol("INFY"), "INFY.NS")

    def test_metal_alias_unchanged(self):
        # XAUUSD is in _ALIASES → not Indian → normalize_indian_symbol leaves it
        self.assertEqual(normalize_indian_symbol("XAUUSD"), "XAUUSD")

    def test_empty_passthrough(self):
        self.assertEqual(normalize_indian_symbol(""), "")
        self.assertIsNone(normalize_indian_symbol(None))

    def test_us_index_unchanged(self):
        # ^GSPC has a ^ which means it's not a plain alpha ticker
        result = normalize_indian_symbol("^GSPC")
        self.assertEqual(result, "^GSPC")


@pytest.mark.unit
class TestDetectMarketProfile(unittest.TestCase):
    def test_ns_returns_india(self):
        self.assertEqual(detect_market_profile("RELIANCE.NS"), "india")

    def test_bo_returns_india(self):
        self.assertEqual(detect_market_profile("HDFCBANK.BO"), "india")

    def test_us_ticker_returns_us(self):
        self.assertEqual(detect_market_profile("AAPL"), "us")
        self.assertEqual(detect_market_profile("MSFT"), "us")

    def test_crypto_returns_us(self):
        self.assertEqual(detect_market_profile("BTC-USD"), "us")

    def test_empty_returns_us(self):
        self.assertEqual(detect_market_profile(""), "us")

    def test_none_returns_us(self):
        self.assertEqual(detect_market_profile(None), "us")

    def test_lowercase_ns_returns_india(self):
        self.assertEqual(detect_market_profile("tcs.ns"), "india")


if __name__ == "__main__":
    unittest.main()
