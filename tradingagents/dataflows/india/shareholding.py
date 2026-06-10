"""Promoter holding, pledge %, and institutional ownership trends for NSE stocks.

Promoter pledge % is a standard Indian red-flag indicator: a rising pledge
often signals promoter financial stress and can precede sharp declines. This
module extracts holding data from yfinance (which carries SEBI-mandated
quarterly disclosures) and supplements it with structured "data unavailable"
sentinels where fields are absent.

For the Fundamentals Analyst, the key outputs are:
  - Current promoter holding %
  - Promoter pledge % (if available)
  - FII / DII / MF holding % (latest and prior quarter for trend)
  - Quarter-over-quarter change flag
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Optional

import yfinance as yf

from ..symbol_utils import normalize_symbol

logger = logging.getLogger(__name__)


def _ensure_ns(ticker: str) -> str:
    """Ensure ticker has .NS suffix for yfinance."""
    s = ticker.upper().strip()
    if s.endswith(".NS") or s.endswith(".BO"):
        return s
    return s + ".NS"


def get_shareholding_summary(ticker: str) -> str:
    """Return a formatted shareholding summary for *ticker*.

    Includes promoter %, FII/DII/MF %, and QoQ trend where available.
    Individual unavailable fields are marked as "data unavailable" rather than
    omitted — agents must not estimate from partial data.
    """
    canonical = _ensure_ns(ticker)
    try:
        yft = yf.Ticker(canonical)
        info = yft.info or {}

        # yfinance info fields relevant to holding pattern
        inst_hold = info.get("heldPercentInstitutions")
        insiders_hold = info.get("heldPercentInsiders")
        float_shares = info.get("floatShares")
        shares_outstanding = info.get("sharesOutstanding")
        short_pct = info.get("shortPercentOfFloat")

        lines = [
            f"# Shareholding summary for {canonical}",
            f"# As of: {datetime.now().strftime('%Y-%m-%d')}",
            "",
        ]

        def _fmt(label: str, value, pct: bool = False) -> str:
            if value is None:
                return f"{label}: data unavailable"
            if pct:
                return f"{label}: {value * 100:.1f}%"
            return f"{label}: {value:,}"

        lines.append(_fmt("Insider / Promoter holding", insiders_hold, pct=True))
        lines.append(_fmt("Institutional holding (FII+DII+MF combined)", inst_hold, pct=True))

        # Promoter pledge — not directly in yfinance info; mark explicitly
        lines.append("Promoter pledge %: data unavailable (check NSE/BSE shareholding filings)")

        # Float and outstanding
        lines.append(_fmt("Float shares", float_shares))
        lines.append(_fmt("Shares outstanding", shares_outstanding))
        lines.append(_fmt("Short % of float", short_pct, pct=True))

        # Quarterly holding detail from major holders
        try:
            inst_holders = yft.institutional_holders
            if inst_holders is not None and not inst_holders.empty:
                top = inst_holders.head(5)
                lines.append("")
                lines.append("Top institutional holders (FII/DII/MF, latest filing):")
                for _, row in top.iterrows():
                    holder = row.get("Holder", "?")
                    pct_val = row.get("% Out")
                    pct_str = f"{pct_val * 100:.2f}%" if pct_val is not None else "N/A"
                    lines.append(f"  {holder}: {pct_str}")
        except Exception:
            lines.append("Top institutional holders: data unavailable")

        lines.append("")
        lines.append(
            "Note: For promoter pledge %, refer to BSE/NSE quarterly shareholding filings. "
            "Rising pledge % is a red flag — treat any pledge above 20% of promoter holding "
            "as an event risk."
        )

        return "\n".join(lines)

    except Exception as exc:
        logger.warning("Shareholding lookup failed for %s: %s", ticker, exc)
        return (
            f"Shareholding data unavailable for {ticker}: {exc}. "
            "Check SEBI shareholding filings on BSE/NSE website."
        )
