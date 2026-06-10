"""Richer fundamentals for NSE/BSE stocks (screener.in-style ratios).

Builds a composite fundamental view from yfinance data, computing key Indian
market ratios: EV/EBITDA, ROCE, debt-to-equity, working-capital metrics, and
relative strength. Missing fields are surfaced as "data unavailable" — agents
must not estimate from absent data (issue #781 precedent).

The Fundamentals Analyst prompt requires: promoter pledge flag, FII/DII
holding change (deferred to shareholding.py), upcoming results date
(deferred to corporate_actions.py), and the ratio checklist here.
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Optional

import yfinance as yf

from ..symbol_utils import normalize_symbol

logger = logging.getLogger(__name__)


def _ensure_ns(ticker: str) -> str:
    s = ticker.upper().strip()
    if s.endswith(".NS") or s.endswith(".BO"):
        return s
    return s + ".NS"


def _fmt(label: str, value, unit: str = "", precision: int = 2) -> str:
    """Format a single metric row; mark None as data unavailable."""
    if value is None:
        return f"  {label}: data unavailable"
    if isinstance(value, float):
        return f"  {label}: {value:.{precision}f}{unit}"
    return f"  {label}: {value}{unit}"


def get_india_fundamentals(ticker: str, curr_date: Optional[str] = None) -> str:
    """Return a comprehensive fundamental snapshot for an NSE/BSE stock.

    Designed for the Fundamentals Analyst agent. Covers valuation multiples,
    profitability, leverage, and growth — with India-specific notes where
    relevant (ROCE, promoter context).
    """
    canonical = _ensure_ns(ticker)
    try:
        yft = yf.Ticker(canonical)
        info = yft.info or {}

        if not info or not any(
            info.get(k) for k in ("longName", "shortName", "sector", "marketCap")
        ):
            return (
                f"Fundamental data unavailable for {canonical}: "
                "yfinance returned an empty profile. The symbol may be "
                "delisted, suspended, or not yet in yfinance's index."
            )

        name = info.get("longName") or info.get("shortName") or canonical
        sector = info.get("sector", "data unavailable")
        industry = info.get("industry", "data unavailable")
        exchange = info.get("exchange", "NSE")
        currency = info.get("currency", "INR")

        lines = [
            f"# Fundamental snapshot: {canonical}",
            f"# Company: {name}",
            f"# Sector: {sector}  |  Industry: {industry}",
            f"# Exchange: {exchange}  |  Currency: {currency}",
            f"# As of: {curr_date or datetime.now().strftime('%Y-%m-%d')}",
            "",
        ]

        # --- Valuation ---
        lines.append("## Valuation")
        mkt_cap = info.get("marketCap")
        mkt_cap_str = f"₹{mkt_cap / 1e7:.0f} Cr" if mkt_cap else "data unavailable"
        lines.append(f"  Market Cap: {mkt_cap_str}")
        lines.append(_fmt("P/E (TTM)", info.get("trailingPE"), "x"))
        lines.append(_fmt("Forward P/E", info.get("forwardPE"), "x"))
        lines.append(_fmt("P/B", info.get("priceToBook"), "x"))
        lines.append(_fmt("PEG Ratio", info.get("pegRatio"), "x"))
        lines.append(_fmt("EV/EBITDA", info.get("enterpriseToEbitda"), "x"))
        lines.append(_fmt("EV/Revenue", info.get("enterpriseToRevenue"), "x"))

        # --- Profitability ---
        lines.append("")
        lines.append("## Profitability")
        lines.append(_fmt("Profit Margin", _pct(info.get("profitMargins")), "%"))
        lines.append(_fmt("Operating Margin", _pct(info.get("operatingMargins")), "%"))
        lines.append(_fmt("Gross Margin", _pct(info.get("grossMargins")), "%"))
        lines.append(_fmt("ROE", _pct(info.get("returnOnEquity")), "%"))
        lines.append(_fmt("ROA", _pct(info.get("returnOnAssets")), "%"))

        # ROCE = EBIT / Capital Employed — compute from available fields
        roce = _compute_roce(info)
        lines.append(_fmt("ROCE (approx)", _pct(roce), "%"))

        # --- Growth ---
        lines.append("")
        lines.append("## Growth")
        lines.append(_fmt("Revenue Growth (YoY)", _pct(info.get("revenueGrowth")), "%"))
        lines.append(_fmt("Earnings Growth (YoY)", _pct(info.get("earningsGrowth")), "%"))
        lines.append(_fmt("Revenue (TTM)", _cr(info.get("totalRevenue")), " Cr"))
        lines.append(_fmt("EPS (TTM)", info.get("trailingEps"), " ₹"))
        lines.append(_fmt("Forward EPS", info.get("forwardEps"), " ₹"))

        # --- Leverage ---
        lines.append("")
        lines.append("## Leverage & Liquidity")
        lines.append(_fmt("Debt/Equity", info.get("debtToEquity"), "x"))
        lines.append(_fmt("Current Ratio", info.get("currentRatio"), "x"))
        lines.append(_fmt("Quick Ratio", info.get("quickRatio"), "x"))
        free_cf = info.get("freeCashflow")
        lines.append(f"  Free Cash Flow: {_cr_str(free_cf)}")

        # --- Price levels ---
        lines.append("")
        lines.append("## Price & Technical Reference")
        lines.append(_fmt("52-Week High", info.get("fiftyTwoWeekHigh"), " ₹"))
        lines.append(_fmt("52-Week Low", info.get("fiftyTwoWeekLow"), " ₹"))
        lines.append(_fmt("50-Day Avg", info.get("fiftyDayAverage"), " ₹"))
        lines.append(_fmt("200-Day Avg", info.get("twoHundredDayAverage"), " ₹"))
        lines.append(_fmt("Beta", info.get("beta"), "x"))
        lines.append(_fmt("Dividend Yield", _pct(info.get("dividendYield")), "%"))

        # --- India-specific notes ---
        lines.append("")
        lines.append("## India-specific notes")
        de = info.get("debtToEquity")
        if de is not None and de > 100:
            lines.append(f"  CAUTION: Debt/Equity {de:.0f}% — highly leveraged. Verify interest coverage.")
        roe = info.get("returnOnEquity")
        if roe is not None and roe < 0.10:
            lines.append(f"  NOTE: ROE {roe*100:.1f}% — below 10% is considered weak for NSE-listed cos.")
        if de is None and roe is None:
            lines.append("  NOTE: Several financial ratios unavailable — data may be sparse for this ticker.")
        lines.append(
            "  Promoter pledge % and FII/DII holding trend: see shareholding.py output."
        )
        lines.append(
            "  Upcoming results / board meeting dates: see corporate_actions.py output."
        )
        lines.append("  All figures in INR (₹) unless noted. Cr = Crores (10 million).")

        return "\n".join(lines)

    except Exception as exc:
        logger.warning("India fundamentals lookup failed for %s: %s", ticker, exc)
        return (
            f"Fundamental data unavailable for {canonical}: {exc}. "
            "Verify the ticker is listed on NSE/BSE."
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _pct(value) -> Optional[float]:
    """Multiply by 100 if value is a fraction, else return as-is; None propagates."""
    if value is None:
        return None
    return value * 100 if abs(value) <= 1.0 else value


def _cr(value) -> Optional[float]:
    """Convert rupee value to Crores (÷ 10,000,000); None propagates."""
    if value is None:
        return None
    return value / 1e7


def _cr_str(value) -> str:
    if value is None:
        return "data unavailable"
    return f"₹{value / 1e7:.0f} Cr"


def _compute_roce(info: dict) -> Optional[float]:
    """Approximate ROCE = EBIT / (Total Assets - Current Liabilities)."""
    try:
        ebitda = info.get("ebitda")
        depreciation = info.get("depreciationAndAmortization") or 0
        total_assets = info.get("totalAssets")
        current_liab = info.get("totalCurrentLiabilities")
        if ebitda is None or total_assets is None or current_liab is None:
            return None
        ebit = ebitda - depreciation
        capital_employed = total_assets - current_liab
        if capital_employed <= 0:
            return None
        return ebit / capital_employed
    except Exception:
        return None
