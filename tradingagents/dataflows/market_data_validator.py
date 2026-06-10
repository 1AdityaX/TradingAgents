"""Deterministic market-data verification snapshot.

The market analyst is an LLM that can confabulate exact numbers — citing a
Bollinger band or a "historically validated bounce" that the underlying data
doesn't support (#830). This module computes a ground-truth snapshot (latest
OHLCV row on or before the analysis date, common indicators, recent closes)
the analyst is told to treat as the source of truth for any exact numeric
claim. Deterministic, no LLM involved.

Data freshness guard
--------------------
``build_verified_market_snapshot`` checks whether the latest available
trading row is current relative to ``curr_date``.  For Indian tickers
(.NS/.BO) it uses the NSE market calendar to find the expected most-recent
session; for all other tickers it falls back to a simple weekday-skipping
heuristic.  When the latest row is more than ``max_stale_sessions`` sessions
behind the expected session a STALE DATA WARNING block is prepended to the
snapshot — the Market Analyst is instructed to flag the discrepancy rather
than act on potentially stale price levels.
"""

from __future__ import annotations

import logging
from datetime import date, timedelta
from typing import Iterable, Optional

import pandas as pd
from stockstats import wrap

from tradingagents.dataflows.stockstats_utils import load_ohlcv

logger = logging.getLogger(__name__)

# A fixed, common indicator set so the snapshot is the same shape every run.
DEFAULT_SNAPSHOT_INDICATORS: tuple[str, ...] = (
    "close_10_ema", "close_50_sma", "close_200_sma",
    "rsi", "boll", "boll_ub", "boll_lb",
    "macd", "macds", "macdh", "atr",
)

# ---------------------------------------------------------------------------
# Data freshness helpers
# ---------------------------------------------------------------------------


def _is_india_ticker(symbol: str) -> bool:
    """Return True for NSE/BSE tickers."""
    s = symbol.upper().strip()
    return s.endswith(".NS") or s.endswith(".BO")


def _prev_weekday(d: date) -> date:
    """Return d if it is a weekday, else the last weekday before d."""
    while d.weekday() >= 5:  # 5=Sat, 6=Sun
        d -= timedelta(days=1)
    return d


def _expected_latest_session(curr_date: str, symbol: str) -> date:
    """Return the expected most-recent trading session on or before curr_date.

    For Indian tickers uses the NSE market calendar; for others falls back to
    the last weekday (ignores US/other holidays — good enough for staleness
    detection).
    """
    try:
        as_of = date.fromisoformat(curr_date)
    except ValueError:
        return date.today()

    if _is_india_ticker(symbol):
        try:
            from tradingagents.dataflows.india.market_calendar import get_prev_trading_day
            return get_prev_trading_day(as_of)
        except Exception:  # noqa: BLE001
            pass  # fall through to weekday heuristic

    return _prev_weekday(as_of)


def _staleness_warning(
    latest_date_str: str,
    curr_date: str,
    symbol: str,
    max_stale_sessions: int = 2,
) -> str:
    """Return a warning block if data is stale; empty string if fresh.

    Staleness is defined as: the latest available data row is more than
    ``max_stale_sessions`` trading days before the expected most-recent
    session for ``curr_date``.
    """
    try:
        latest = date.fromisoformat(latest_date_str)
    except ValueError:
        return ""

    expected = _expected_latest_session(curr_date, symbol)

    # Count business-day gap (rough: count weekdays between dates)
    gap_days = 0
    d = latest + timedelta(days=1)
    while d <= expected:
        if d.weekday() < 5:
            gap_days += 1
        d += timedelta(days=1)

    if gap_days <= max_stale_sessions:
        return ""

    logger.warning(
        "Stale market data for %s: latest row %s is %d sessions before expected %s",
        symbol, latest_date_str, gap_days, expected,
    )
    return "\n".join([
        "⚠️  STALE DATA WARNING",
        f"   Latest available row:   {latest_date_str}",
        f"   Expected latest session: {expected}",
        f"   Gap: {gap_days} trading session(s)",
        "   Price levels in this snapshot may not reflect the current market.",
        "   Do NOT use these levels for new entries without confirming current prices.",
        "   Treat all support/resistance levels as provisional until refreshed.",
        "",
    ])


def _verified_rows(symbol: str, curr_date: str) -> pd.DataFrame:
    """OHLCV on or before curr_date, date-sorted. Raises if nothing usable.

    ``load_ohlcv`` already normalizes the Date column and filters out
    look-ahead rows, but we re-apply the cutoff defensively — this is a
    verification path, so it must not trust its input to be pre-filtered.
    """
    data = load_ohlcv(symbol, curr_date)
    if data is None or data.empty:
        raise ValueError(f"No OHLCV data available for {symbol}.")

    df = data.copy()
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date"])
    df = df[df["Date"] <= pd.to_datetime(curr_date)].sort_values("Date")
    if df.empty:
        raise ValueError(f"No OHLCV rows on or before {curr_date} for {symbol}.")
    return df


def _fmt(value) -> str:
    if value is None or pd.isna(value):
        return "N/A"
    if isinstance(value, pd.Timestamp):
        return value.strftime("%Y-%m-%d")
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, (int,)):
        return str(value)
    if isinstance(value, float):
        return f"{value:.2f}"
    return str(value)


def build_verified_market_snapshot(
    symbol: str,
    curr_date: str,
    look_back_days: int = 30,
    indicators: Optional[Iterable[str]] = None,
    max_stale_sessions: int = 2,
) -> str:
    """Render a ground-truth snapshot: latest OHLCV row, indicators, recent closes.

    Args:
        symbol: Ticker symbol (e.g. 'RELIANCE.NS', 'AAPL').
        curr_date: Analysis date in YYYY-MM-DD format.
        look_back_days: How many recent close rows to include in the table (max 30).
        indicators: Override the default indicator set.
        max_stale_sessions: If the latest data row is more than this many trading
            sessions behind the expected most-recent session, a STALE DATA WARNING
            block is prepended to the snapshot.  Set to 0 to disable the guard.
    """
    # `df` keeps the original capitalized OHLCV columns (Open/High/Low/Close/
    # Volume); stockstats `wrap()` lowercases columns and adds indicator
    # columns, so read raw prices from `df` and indicators from `stock_df`.
    df = _verified_rows(symbol, curr_date)
    stock_df = wrap(df.copy())

    selected = tuple(indicators or DEFAULT_SNAPSHOT_INDICATORS)
    indicator_values: dict[str, str] = {}
    for name in selected:
        try:
            stock_df[name]  # triggers stockstats calculation
            indicator_values[name] = _fmt(stock_df.iloc[-1][name])
        except Exception as exc:  # noqa: BLE001 — one bad indicator shouldn't sink the snapshot
            indicator_values[name] = f"N/A ({type(exc).__name__})"

    latest = df.iloc[-1]
    latest_date = _fmt(latest["Date"])
    window = max(1, min(int(look_back_days), 30))
    recent = df.tail(window)

    # Freshness check — prepended so the LLM sees it before any price levels.
    stale_warn = ""
    if max_stale_sessions > 0:
        stale_warn = _staleness_warning(latest_date, curr_date, symbol, max_stale_sessions)

    lines = []
    if stale_warn:
        lines += [stale_warn]

    lines += [
        f"## Verified market data snapshot for {symbol.upper()}",
        "",
        f"- Requested analysis date: {curr_date}",
        f"- Latest trading row used: {latest_date}",
        "- Rows after the requested analysis date are excluded before verification.",
        "",
        "### Latest verified OHLCV row",
        "",
        "| Field | Value |",
        "|---|---:|",
    ]
    for field in ("Open", "High", "Low", "Close", "Volume"):
        lines.append(f"| {field} | {_fmt(latest.get(field))} |")

    lines += ["", "### Verified technical indicators (latest row)", "",
              "| Indicator | Value |", "|---|---:|"]
    for name, value in indicator_values.items():
        lines.append(f"| {name} | {value} |")

    lines += ["", f"### Recent verified closes (last {len(recent)} rows)", "",
              "| Date | Close |", "|---|---:|"]
    for _, row in recent.iterrows():
        lines.append(f"| {_fmt(row['Date'])} | {_fmt(row.get('Close'))} |")

    lines += [
        "",
        "Use this snapshot as the source of truth for exact OHLCV, price-level, "
        "and indicator-value claims. If another tool output conflicts with it, "
        "flag the discrepancy rather than inventing a reconciled number. Do not "
        "claim historical validation, support/resistance bounces, or exact "
        "percentage moves unless directly supported by tool output with concrete "
        "dates and prices.",
    ]
    return "\n".join(lines)
