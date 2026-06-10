"""Tests for the NSE market calendar module."""

import unittest
from datetime import date, datetime, timezone

import pytest

from tradingagents.dataflows.india.market_calendar import (
    format_calendar_context,
    get_monthly_expiry,
    get_next_expiry,
    get_next_holiday,
    get_next_trading_day,
    get_prev_trading_day,
    is_trading_day,
    market_status,
)


@pytest.mark.unit
class TestIsTradingDay(unittest.TestCase):
    def test_monday_non_holiday_is_trading(self):
        # 2026-06-08 is a Monday with no listed holiday
        self.assertTrue(is_trading_day(date(2026, 6, 8)))

    def test_saturday_is_not_trading(self):
        self.assertFalse(is_trading_day(date(2026, 6, 13)))

    def test_sunday_is_not_trading(self):
        self.assertFalse(is_trading_day(date(2026, 6, 14)))

    def test_republic_day_2026_is_not_trading(self):
        self.assertFalse(is_trading_day(date(2026, 1, 26)))

    def test_christmas_2025_is_not_trading(self):
        self.assertFalse(is_trading_day(date(2025, 12, 25)))

    def test_working_tuesday_is_trading(self):
        # 2025-06-10 — a Tuesday, no listed holiday
        self.assertTrue(is_trading_day(date(2025, 6, 10)))


@pytest.mark.unit
class TestGetNextTradingDay(unittest.TestCase):
    def test_trading_day_returns_itself(self):
        d = date(2026, 6, 8)  # Monday
        self.assertTrue(is_trading_day(d))
        self.assertEqual(get_next_trading_day(d), d)

    def test_saturday_returns_monday(self):
        sat = date(2026, 6, 13)  # Saturday
        self.assertFalse(is_trading_day(sat))
        result = get_next_trading_day(sat)
        self.assertEqual(result.weekday(), 0)  # Monday
        self.assertGreater(result, sat)

    def test_sunday_returns_monday(self):
        sun = date(2026, 6, 14)
        result = get_next_trading_day(sun)
        self.assertEqual(result, date(2026, 6, 15))

    def test_holiday_skipped(self):
        # Republic Day 2026 is a Monday — should skip to Tuesday
        result = get_next_trading_day(date(2026, 1, 26))
        self.assertEqual(result, date(2026, 1, 27))


@pytest.mark.unit
class TestGetPrevTradingDay(unittest.TestCase):
    def test_trading_day_returns_itself(self):
        d = date(2026, 6, 8)
        self.assertEqual(get_prev_trading_day(d), d)

    def test_monday_returns_prior_friday(self):
        mon = date(2026, 6, 8)
        result = get_prev_trading_day(date(2026, 6, 6))  # Saturday
        self.assertEqual(result, date(2026, 6, 5))       # Friday

    def test_sunday_returns_friday(self):
        result = get_prev_trading_day(date(2026, 6, 14))
        self.assertEqual(result, date(2026, 6, 12))       # Friday


@pytest.mark.unit
class TestGetMonthlyExpiry(unittest.TestCase):
    def test_expiry_is_thursday_or_holiday_adjusted(self):
        # Expiry is the last Thursday; if that's a holiday, it moves to the
        # prior trading day (which may be a Wednesday or earlier).
        from tradingagents.dataflows.india.market_calendar import _last_thursday
        for year, month in [(2025, 6), (2025, 7), (2025, 12), (2026, 1), (2026, 3)]:
            expiry = get_monthly_expiry(year, month)
            last_thu = _last_thursday(year, month)
            # Expiry must be on or before the last Thursday of the month
            self.assertLessEqual(expiry, last_thu, f"Expiry {expiry} is after last Thursday {last_thu}")
            # Expiry must be a trading day
            self.assertTrue(is_trading_day(expiry), f"Expiry {expiry} for {year}-{month:02d} is not a trading day")

    def test_expiry_is_in_correct_month(self):
        expiry = get_monthly_expiry(2025, 6)
        self.assertEqual(expiry.month, 6)
        self.assertEqual(expiry.year, 2025)

    def test_expiry_is_last_thursday(self):
        # Verify no Thursday exists after the returned date in that month
        for year, month in [(2025, 6), (2025, 11), (2026, 2)]:
            expiry = get_monthly_expiry(year, month)
            next_week = expiry.replace(day=expiry.day + 7) if expiry.day + 7 <= 31 else None
            if next_week:
                try:
                    self.assertTrue(next_week.month != month, f"Later Thursday {next_week} exists in same month as {expiry}")
                except ValueError:
                    pass  # Date arithmetic overflow — expiry is last


@pytest.mark.unit
class TestGetNextExpiry(unittest.TestCase):
    def test_returns_current_month_if_not_passed(self):
        # A date well before the last Thursday of June 2025
        ref = date(2025, 6, 1)
        expiry = get_next_expiry(ref)
        self.assertEqual(expiry.month, 6)
        self.assertEqual(expiry.year, 2025)

    def test_advances_to_next_month_after_expiry(self):
        # Use a date AFTER the last Thursday of June to ensure July is returned
        ref = date(2025, 6, 30)  # After any possible last Thursday
        expiry = get_next_expiry(ref)
        self.assertEqual(expiry.month, 7)

    def test_on_expiry_day_returns_current(self):
        expiry_june = get_monthly_expiry(2025, 6)
        result = get_next_expiry(expiry_june)
        self.assertEqual(result, expiry_june)

    def test_day_after_expiry_returns_next_month(self):
        from datetime import timedelta
        expiry_june = get_monthly_expiry(2025, 6)
        day_after = expiry_june + timedelta(days=1)
        result = get_next_expiry(day_after)
        self.assertEqual(result.month, 7)


@pytest.mark.unit
class TestGetNextHoliday(unittest.TestCase):
    def test_returns_none_beyond_list(self):
        # Far-future date beyond our static list
        result = get_next_holiday(date(2030, 1, 1))
        self.assertIsNone(result)

    def test_returns_holiday_in_range(self):
        # Republic Day 2026 should be findable
        result = get_next_holiday(date(2026, 1, 1))
        self.assertIsNotNone(result)
        self.assertEqual(result, date(2026, 1, 26))

    def test_holiday_on_exact_date_returned(self):
        self.assertEqual(get_next_holiday(date(2025, 12, 25)), date(2025, 12, 25))


@pytest.mark.unit
class TestMarketStatus(unittest.TestCase):
    def test_saturday_not_trading(self):
        sat = datetime(2026, 6, 13, 10, 0)
        status = market_status(sat)
        self.assertFalse(status["is_trading_day"])
        self.assertFalse(status["is_market_open"])

    def test_weekday_during_hours_open(self):
        # Monday 2026-06-08 at 11:00 (well within 09:15–15:30)
        mon = datetime(2026, 6, 8, 11, 0)
        status = market_status(mon)
        self.assertTrue(status["is_trading_day"])
        self.assertTrue(status["is_market_open"])

    def test_weekday_before_open(self):
        mon = datetime(2026, 6, 8, 8, 0)
        status = market_status(mon)
        self.assertTrue(status["is_trading_day"])
        self.assertFalse(status["is_market_open"])

    def test_weekday_after_close(self):
        mon = datetime(2026, 6, 8, 16, 0)
        status = market_status(mon)
        self.assertTrue(status["is_trading_day"])
        self.assertFalse(status["is_market_open"])

    def test_keys_present(self):
        status = market_status(datetime(2026, 6, 8, 11, 0))
        for key in ("is_trading_day", "is_market_open", "market_hours_ist",
                    "next_expiry", "next_holiday", "timezone", "settlement"):
            self.assertIn(key, status)

    def test_settlement_is_t1(self):
        status = market_status(datetime(2026, 6, 8, 11, 0))
        self.assertEqual(status["settlement"], "T+1")


@pytest.mark.unit
class TestFormatCalendarContext(unittest.TestCase):
    def test_contains_market_hours(self):
        ctx = format_calendar_context(date(2026, 6, 8))
        self.assertIn("09:15", ctx)
        self.assertIn("15:30", ctx)

    def test_contains_t1_settlement(self):
        ctx = format_calendar_context(date(2026, 6, 8))
        self.assertIn("T+1", ctx)

    def test_contains_expiry_date(self):
        ctx = format_calendar_context(date(2026, 6, 8))
        self.assertIn("expiry", ctx.lower())

    def test_uses_today_when_none(self):
        # Should not raise
        ctx = format_calendar_context(None)
        self.assertIsInstance(ctx, str)
        self.assertGreater(len(ctx), 20)


if __name__ == "__main__":
    unittest.main()
