"""Tests for India instrument context injection in agent_utils."""

import unittest
from unittest.mock import MagicMock, patch

import pytest

from tradingagents.agents.utils.agent_utils import build_instrument_context

# The India context helper is imported lazily inside _build_india_context_block;
# patch it at the source module so the lazy import picks up the mock.
_PATCH_TARGET = "tradingagents.dataflows.interface.get_india_instrument_context"


@pytest.mark.unit
class TestBuildInstrumentContextIndia(unittest.TestCase):
    """India-specific instrument context block is injected for .NS/.BO tickers."""

    def _mock_india_ctx(self):
        return {
            "fno_ban": "No",
            "calendar_context": "Market hours: 09:15–15:30 IST. Settlement: T+1. Next F&O monthly expiry: 2026-06-26.",
            "next_expiry": "2026-06-26",
        }

    def test_ns_ticker_gets_india_block(self):
        with patch(_PATCH_TARGET, return_value=self._mock_india_ctx()):
            ctx = build_instrument_context("RELIANCE.NS")
        self.assertIn("INDIA", ctx)
        self.assertIn("INR", ctx)
        self.assertIn("09:15", ctx)
        self.assertIn("T+1", ctx)

    def test_bo_ticker_gets_india_block(self):
        with patch(_PATCH_TARGET, return_value=self._mock_india_ctx()):
            ctx = build_instrument_context("HDFCBANK.BO")
        self.assertIn("INDIA", ctx)
        self.assertIn("₹", ctx)

    def test_us_ticker_does_not_get_india_block(self):
        ctx = build_instrument_context("AAPL")
        self.assertNotIn("INDIA", ctx)
        self.assertNotIn("09:15–15:30 IST", ctx)
        self.assertNotIn("T+1", ctx)

    def test_india_block_warns_about_inr(self):
        with patch(_PATCH_TARGET, return_value=self._mock_india_ctx()):
            ctx = build_instrument_context("RELIANCE.NS")
        self.assertIn("₹ (not $)", ctx)

    def test_india_block_warns_about_fno_ban(self):
        with patch(_PATCH_TARGET, return_value=self._mock_india_ctx()):
            ctx = build_instrument_context("RELIANCE.NS")
        self.assertIn("F&O ban", ctx)

    def test_india_block_includes_fno_ban_status(self):
        with patch(
            _PATCH_TARGET,
            return_value={"fno_ban": "YES", "calendar_context": "Market hours: 09:15–15:30 IST.", "next_expiry": "2026-06-26"},
        ):
            ctx = build_instrument_context("XYZ.NS")
        self.assertIn("YES", ctx)

    def test_identity_injected_alongside_india_block(self):
        with patch(_PATCH_TARGET, return_value=self._mock_india_ctx()):
            ctx = build_instrument_context(
                "RELIANCE.NS",
                "stock",
                {"company_name": "Reliance Industries Ltd.", "sector": "Energy"},
            )
        self.assertIn("Reliance Industries Ltd.", ctx)
        self.assertIn("Energy", ctx)
        self.assertIn("INDIA", ctx)

    def test_india_block_circuit_bands_mentioned(self):
        with patch(_PATCH_TARGET, return_value=self._mock_india_ctx()):
            ctx = build_instrument_context("RELIANCE.NS")
        self.assertIn("Circuit", ctx)

    def test_india_context_fallback_on_exception(self):
        """When get_india_instrument_context raises, build_instrument_context must not raise."""
        with patch(_PATCH_TARGET, side_effect=RuntimeError("network error")):
            try:
                ctx = build_instrument_context("RELIANCE.NS")
            except RuntimeError:
                self.fail("build_instrument_context raised RuntimeError on India context failure")


if __name__ == "__main__":
    unittest.main()
