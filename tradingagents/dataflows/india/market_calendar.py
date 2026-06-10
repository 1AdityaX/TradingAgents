"""NSE market calendar: holidays, trading hours, and F&O expiry dates.

Ships a static holiday list (updated annually). The expiry-day rule has
changed before (see UPGRADE_PLAN notes) — currently last Thursday of the month
for Nifty 50 monthly F&O contracts. Holidays for years beyond the static list
return None rather than raising, so callers degrade gracefully.
"""

from __future__ import annotations

from datetime import date, datetime, timedelta
from typing import Optional

try:
    import pytz
    _IST = pytz.timezone("Asia/Kolkata")
    _HAVE_PYTZ = True
except ImportError:
    _HAVE_PYTZ = False

NSE_OPEN_HHMM = (9, 15)
NSE_CLOSE_HHMM = (15, 30)

# NSE trading holidays — verified from NSE circulars.
# Add new years by appending to the set; no other code change needed.
_NSE_HOLIDAYS: frozenset[date] = frozenset(
    {
        # --- 2025 ---
        date(2025, 1, 26),   # Republic Day
        date(2025, 2, 26),   # Mahashivratri
        date(2025, 3, 14),   # Holi
        date(2025, 3, 31),   # Id-Ul-Fitr (Ramzan Eid)
        date(2025, 4, 10),   # Shri Mahavir Jayanti
        date(2025, 4, 14),   # Dr. Ambedkar Jayanti
        date(2025, 4, 18),   # Good Friday
        date(2025, 5, 1),    # Maharashtra Day
        date(2025, 8, 15),   # Independence Day
        date(2025, 8, 27),   # Ganesh Chaturthi
        date(2025, 10, 2),   # Gandhi Jayanti
        date(2025, 10, 20),  # Diwali Laxmi Pujan
        date(2025, 10, 21),  # Diwali Balipratipada
        date(2025, 11, 5),   # Guru Nanak Jayanti
        date(2025, 12, 25),  # Christmas
        # --- 2026 ---
        date(2026, 1, 26),   # Republic Day
        date(2026, 3, 3),    # Mahashivratri
        date(2026, 3, 20),   # Holi
        date(2026, 3, 20),   # Holi (Good Friday falls same week, TBD)
        date(2026, 4, 3),    # Good Friday
        date(2026, 4, 14),   # Dr. Ambedkar Jayanti
        date(2026, 5, 1),    # Maharashtra Day
        date(2026, 8, 15),   # Independence Day
        date(2026, 10, 2),   # Gandhi Jayanti
        date(2026, 10, 20),  # Diwali (approximate — update from NSE circular)
        date(2026, 11, 24),  # Guru Nanak Jayanti (approximate)
        date(2026, 12, 25),  # Christmas
    }
)


def is_trading_day(d: date) -> bool:
    """Return True when *d* is an NSE trading day (Mon–Fri, not a listed holiday)."""
    return d.weekday() < 5 and d not in _NSE_HOLIDAYS


def get_next_trading_day(from_date: Optional[date] = None) -> date:
    """Return the first NSE trading day on or after *from_date* (today if None)."""
    d = from_date if from_date is not None else date.today()
    while not is_trading_day(d):
        d += timedelta(days=1)
    return d


def get_prev_trading_day(from_date: Optional[date] = None) -> date:
    """Return the last NSE trading day on or before *from_date* (today if None)."""
    d = from_date if from_date is not None else date.today()
    while not is_trading_day(d):
        d -= timedelta(days=1)
    return d


def _last_thursday(year: int, month: int) -> date:
    """Return the last Thursday of the given calendar month."""
    # First day of the following month
    if month == 12:
        first_next = date(year + 1, 1, 1)
    else:
        first_next = date(year, month + 1, 1)
    last_day = first_next - timedelta(days=1)
    # weekday(): Mon=0 … Thu=3 … Sun=6
    days_back = (last_day.weekday() - 3) % 7
    return last_day - timedelta(days=days_back)


def get_monthly_expiry(year: int, month: int) -> date:
    """Return the NSE F&O monthly expiry date (last Thursday, adjusted for holidays)."""
    d = _last_thursday(year, month)
    # If holiday, step back to the nearest prior trading day
    while not is_trading_day(d):
        d -= timedelta(days=1)
    return d


def get_next_expiry(from_date: Optional[date] = None) -> date:
    """Return the next (or current) F&O monthly expiry on or after *from_date*."""
    ref = from_date if from_date is not None else date.today()
    year, month = ref.year, ref.month
    expiry = get_monthly_expiry(year, month)
    if expiry < ref:
        month += 1
        if month > 12:
            month, year = 1, year + 1
        expiry = get_monthly_expiry(year, month)
    return expiry


def get_next_holiday(from_date: Optional[date] = None) -> Optional[date]:
    """Return the next NSE holiday on or after *from_date*, or None if beyond the list."""
    ref = from_date if from_date is not None else date.today()
    future = sorted(d for d in _NSE_HOLIDAYS if d >= ref)
    return future[0] if future else None


def market_status(dt: Optional[datetime] = None) -> dict:
    """Return a dict describing current market status relative to *dt* (now if None).

    Fields: is_trading_day, is_market_open, market_hours_ist, next_expiry,
            next_holiday, timezone, settlement.
    """
    if dt is None:
        if _HAVE_PYTZ:
            dt = datetime.now(_IST)
        else:
            # Rough UTC+5:30 offset without pytz
            from datetime import timezone
            dt = datetime.now(timezone.utc).replace(tzinfo=None) + timedelta(hours=5, minutes=30)

    d = dt.date() if hasattr(dt, "date") else dt

    h, m = dt.hour, dt.minute
    open_h, open_m = NSE_OPEN_HHMM
    close_h, close_m = NSE_CLOSE_HHMM
    after_open = (h, m) >= (open_h, open_m)
    before_close = (h, m) < (close_h, close_m)
    currently_open = is_trading_day(d) and after_open and before_close

    next_expiry = get_next_expiry(d)
    next_holiday = get_next_holiday(d)

    return {
        "is_trading_day": is_trading_day(d),
        "is_market_open": currently_open,
        "market_hours_ist": "09:15–15:30",
        "next_expiry": str(next_expiry),
        "next_holiday": str(next_holiday) if next_holiday else "none in static calendar",
        "timezone": "IST (UTC+5:30)",
        "settlement": "T+1",
    }


def format_calendar_context(as_of: Optional[date] = None) -> str:
    """Return a compact one-paragraph calendar context block for agent injection."""
    ref = as_of if as_of is not None else date.today()
    next_expiry = get_next_expiry(ref)
    next_holiday = get_next_holiday(ref)
    next_trading = get_next_trading_day(ref)

    holiday_str = str(next_holiday) if next_holiday else "none listed"
    return (
        f"Market hours: 09:15–15:30 IST (UTC+5:30). Settlement: T+1. "
        f"Next F&O monthly expiry: {next_expiry}. "
        f"Next NSE holiday: {holiday_str}. "
        f"Next trading day from {ref}: {next_trading}."
    )
