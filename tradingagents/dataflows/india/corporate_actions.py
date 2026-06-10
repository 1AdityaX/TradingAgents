"""Corporate actions and results calendar for NSE/BSE stocks.

For swing trades the key actions are:
  - Upcoming ex-dividend dates  → income event, often priced in quickly
  - Board meeting / results dates → discrete risk inside the holding window
  - Stock splits / bonuses / buybacks → can distort price levels

yfinance exposes dividends, splits, and calendar events. We parse them and
flag any results/board-meeting dates that fall within the typical swing window
(configurable; default 20 trading days) as explicit event risks.
"""

from __future__ import annotations

import logging
from datetime import date, datetime, timedelta
from typing import Optional

import pandas as pd
import yfinance as yf

from ..symbol_utils import normalize_symbol

logger = logging.getLogger(__name__)

_DEFAULT_WINDOW_DAYS = 20


def _ensure_ns(ticker: str) -> str:
    s = ticker.upper().strip()
    if s.endswith(".NS") or s.endswith(".BO"):
        return s
    return s + ".NS"


def _date_within_window(d: date, from_date: date, window_days: int) -> bool:
    return from_date <= d <= from_date + timedelta(days=window_days)


def get_corporate_actions(
    ticker: str,
    as_of: Optional[date] = None,
    window_days: int = _DEFAULT_WINDOW_DAYS,
) -> str:
    """Return upcoming corporate actions and results dates for *ticker*.

    Events within the *window_days* swing window are flagged explicitly so the
    risk team can assess them as discrete holding risks.
    """
    ref = as_of if as_of is not None else date.today()
    canonical = _ensure_ns(ticker)

    lines = [
        f"# Corporate actions for {canonical}",
        f"# As of: {ref}  |  Swing window: {window_days} trading days",
        "",
    ]
    event_risks: list[str] = []

    try:
        yft = yf.Ticker(canonical)

        # --- Results / earnings calendar ---
        try:
            cal = yft.calendar
            if cal is not None and len(cal) > 0:
                # yfinance returns either a dict or a DataFrame
                if isinstance(cal, dict):
                    earn_date = cal.get("Earnings Date")
                    if earn_date:
                        if isinstance(earn_date, (list, tuple)):
                            earn_date = earn_date[0]
                        if hasattr(earn_date, "date"):
                            earn_date = earn_date.date()
                        elif isinstance(earn_date, str):
                            try:
                                earn_date = datetime.strptime(earn_date, "%Y-%m-%d").date()
                            except ValueError:
                                earn_date = None
                        if earn_date:
                            flag = ""
                            if _date_within_window(earn_date, ref, window_days):
                                flag = " *** INSIDE SWING WINDOW — EVENT RISK ***"
                                event_risks.append(f"Results/earnings on {earn_date}")
                            lines.append(f"Next earnings/results date: {earn_date}{flag}")
                elif hasattr(cal, "loc"):
                    # DataFrame-style
                    for col in ("Earnings Date", "Earnings High", "Earnings Low"):
                        if col in cal.columns or col in cal.index:
                            try:
                                val = cal.loc[col].iloc[0] if col in cal.index else cal[col].iloc[0]
                                if pd.notna(val):
                                    lines.append(f"  {col}: {val}")
                            except Exception:
                                pass
            else:
                lines.append("Earnings/results calendar: data unavailable")
        except Exception as exc:
            logger.debug("Calendar fetch failed for %s: %s", ticker, exc)
            lines.append("Earnings/results calendar: data unavailable")

        # --- Upcoming dividends ---
        try:
            divs = yft.dividends
            if divs is not None and not divs.empty:
                # Filter to future ex-dates
                now_ts = pd.Timestamp(ref)
                future_divs = divs[divs.index.tz_localize(None) >= now_ts] if divs.index.tz else divs[divs.index >= now_ts]
                if not future_divs.empty:
                    lines.append("")
                    lines.append("Upcoming dividends (ex-date):")
                    for ex_date, amount in future_divs.head(3).items():
                        ex_d = ex_date.date() if hasattr(ex_date, "date") else ex_date
                        flag = ""
                        if _date_within_window(ex_d, ref, window_days):
                            flag = " *** INSIDE SWING WINDOW ***"
                            event_risks.append(f"Ex-dividend ₹{amount:.2f} on {ex_d}")
                        lines.append(f"  Ex-date {ex_d}: ₹{amount:.2f} per share{flag}")
                else:
                    lines.append("Upcoming dividends: none in near term")
            else:
                lines.append("Dividend history: data unavailable")
        except Exception as exc:
            logger.debug("Dividend fetch failed for %s: %s", ticker, exc)
            lines.append("Dividend data: data unavailable")

        # --- Splits and bonuses ---
        try:
            splits = yft.splits
            if splits is not None and not splits.empty:
                now_ts = pd.Timestamp(ref)
                future_splits = splits[splits.index.tz_localize(None) >= now_ts] if splits.index.tz else splits[splits.index >= now_ts]
                if not future_splits.empty:
                    lines.append("")
                    lines.append("Upcoming stock splits/bonuses:")
                    for spl_date, ratio in future_splits.head(3).items():
                        sd = spl_date.date() if hasattr(spl_date, "date") else spl_date
                        flag = ""
                        if _date_within_window(sd, ref, window_days):
                            flag = " *** INSIDE SWING WINDOW ***"
                            event_risks.append(f"Stock split/bonus (ratio {ratio}) on {sd}")
                        lines.append(f"  Date {sd}: ratio {ratio}{flag}")
        except Exception as exc:
            logger.debug("Splits fetch failed for %s: %s", ticker, exc)

        # --- Event risk summary ---
        if event_risks:
            lines.append("")
            lines.append("SWING WINDOW EVENT RISKS (must be reviewed before trade entry):")
            for risk in event_risks:
                lines.append(f"  - {risk}")

        return "\n".join(lines)

    except Exception as exc:
        logger.warning("Corporate actions lookup failed for %s: %s", ticker, exc)
        return (
            f"Corporate actions data unavailable for {ticker}: {exc}. "
            "Check NSE/BSE for upcoming results and ex-dates."
        )
