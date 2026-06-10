"""Daily FII/DII cash-market net buy/sell figures for NSE.

FII (Foreign Institutional Investors) and DII (Domestic Institutional
Investors) daily net flows are one of the strongest swing-trade context
signals in India. This module fetches the last N sessions of provisional
figures from NSE, falling back to a "data unavailable" sentinel rather than
an empty string (issue #781 precedent).

Data source: NSE FIIDII endpoint (public, requires browser-style cookies).
The fallback provides structured "data unavailable" per-session rows so agents
know the signal is missing rather than seeing blank context.
"""

from __future__ import annotations

import logging
from datetime import date, timedelta
from typing import Optional

from .nse_client import _fetch_nse, _read_cache, _write_cache
from .market_calendar import get_prev_trading_day, is_trading_day

logger = logging.getLogger(__name__)

_CACHE_KEY = "fii_dii_flows"
_ENDPOINT = "/api/fiidiiTradeReact"


def _fetch_flows_raw() -> Optional[list]:
    """Fetch raw FII/DII data from NSE; return list of dicts or None."""
    cached = _read_cache(_CACHE_KEY)
    if cached is not None:
        return cached
    raw = _fetch_nse(_ENDPOINT)
    if raw:
        data = raw if isinstance(raw, list) else raw.get("data") or raw.get("result") or []
        if data:
            _write_cache(_CACHE_KEY, data)
            return data
    return None


def _last_n_trading_dates(from_date: date, n: int) -> list[date]:
    """Return the last *n* trading dates ending at (and including) *from_date*."""
    dates: list[date] = []
    d = from_date if is_trading_day(from_date) else get_prev_trading_day(from_date)
    while len(dates) < n:
        dates.append(d)
        d -= timedelta(days=1)
        while not is_trading_day(d):
            d -= timedelta(days=1)
    return list(reversed(dates))


def get_fii_dii_flows(
    as_of: Optional[date] = None,
    sessions: int = 10,
) -> str:
    """Return a formatted table of FII/DII cash-market net buy/sell figures.

    Args:
        as_of:    Reference date (today if None).
        sessions: Number of recent trading sessions to include (default 10).

    Returns a formatted string for agent injection. Individual session rows
    that have no data show "data unavailable" for that session rather than
    being silently omitted.
    """
    ref = as_of if as_of is not None else date.today()
    rows = _fetch_flows_raw()

    header = f"FII/DII Cash Market Flows (last {sessions} sessions, as of {ref}):\n"

    if rows:
        try:
            # NSE returns a list; each entry typically has date, fii_net, dii_net keys.
            # Key names vary — try a few common patterns.
            lines = [
                f"{'Date':<14} {'FII Net (₹ Cr)':>16} {'DII Net (₹ Cr)':>16}"
            ]
            lines.append("-" * 50)
            shown = 0
            for row in rows[-sessions:]:
                d_str = (
                    row.get("date")
                    or row.get("tradingDate")
                    or row.get("tradeDate")
                    or "?"
                )
                fii = (
                    row.get("fii_netPurchase_sales")
                    or row.get("fiiNetPurchaseSales")
                    or row.get("fii_net")
                    or row.get("netPurchasesSalesFII")
                    or "N/A"
                )
                dii = (
                    row.get("dii_netPurchase_sales")
                    or row.get("diiNetPurchaseSales")
                    or row.get("dii_net")
                    or row.get("netPurchasesSalesDII")
                    or "N/A"
                )
                lines.append(f"{str(d_str):<14} {str(fii):>16} {str(dii):>16}")
                shown += 1
                if shown >= sessions:
                    break

            if shown > 0:
                return header + "\n".join(lines)
        except Exception as exc:
            logger.debug("FII/DII flow parse error: %s", exc)

    # Graceful fallback: structured "unavailable" rows so agents know what's missing
    trading_dates = _last_n_trading_dates(ref, sessions)
    lines = [
        f"{'Date':<14} {'FII Net (₹ Cr)':>16} {'DII Net (₹ Cr)':>16}",
        "-" * 50,
    ]
    for d in trading_dates:
        lines.append(f"{str(d):<14} {'data unavailable':>16} {'data unavailable':>16}")

    note = (
        "\nNote: FII/DII provisional data unavailable from NSE — "
        "check https://www.nseindia.com/reports-indices-derivatives/fii-dii manually."
    )
    return header + "\n".join(lines) + note
