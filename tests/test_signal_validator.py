"""Unit tests for tradingagents/agents/validator/signal_validator.py

Covers every validation rule: SL side, stale levels, affordability,
RR floor, retry/force-NO_TRADE flow, and the position-action validator
(SL-widen rejection, broken-thesis guard).
"""

from __future__ import annotations

from unittest.mock import patch

import pytest

from tradingagents.agents.validator.signal_validator import (
    _validate,
    _validate_position_action,
    create_signal_validator,
    create_position_action_validator,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _cfg(
    equity: float = 1_000_000,
    risk_pct: float = 1.0,
    min_rr: float = 1.8,
    txn_cost: float = 0.0,
    max_dev_pct: float = 15.0,
) -> dict:
    return {
        "account_equity_inr": equity,
        "risk_pct_per_trade": risk_pct,
        "max_open_risk_pct": 6.0,
        "max_position_pct": 50.0,
        "min_risk_reward": min_rr,
        "txn_cost_pct_round_trip": txn_cost,
        "max_entry_deviation_pct": max_dev_pct,
    }


def _long_signal_dict(
    entry_price: float = 1000.0,
    stop_loss: float = 940.0,
    tp1_price: float = 1200.0,
) -> dict:
    return {
        "direction": "LONG",
        "setup_type": "test",
        "timeframe": "swing",
        "entries": [
            {"label": "EP1", "price": entry_price, "allocation_pct": 100.0,
             "trigger": "limit", "rationale": "test"},
        ],
        "stop_loss": stop_loss,
        "stop_basis": "structural",
        "take_profits": [
            {"label": "TP1", "price": tp1_price, "exit_pct": 100.0, "basis": "prior high"}
        ],
        "invalidation": "test",
        "event_risks": [],
        "risk_reward_min": 2.0,
        "confidence": "high",
    }


def _short_signal_dict(
    entry_price: float = 1000.0,
    stop_loss: float = 1060.0,
    tp1_price: float = 850.0,
) -> dict:
    return {
        "direction": "SHORT",
        "setup_type": "test",
        "timeframe": "swing",
        "entries": [
            {"label": "EP1", "price": entry_price, "allocation_pct": 100.0,
             "trigger": "limit", "rationale": "test"},
        ],
        "stop_loss": stop_loss,
        "stop_basis": "structural",
        "take_profits": [
            {"label": "TP1", "price": tp1_price, "exit_pct": 100.0, "basis": "prior low"}
        ],
        "invalidation": "test",
        "event_risks": [],
        "risk_reward_min": 2.0,
        "confidence": "high",
    }


def _no_trade_dict() -> dict:
    return {
        "direction": "NO_TRADE",
        "setup_type": "no-setup",
        "timeframe": "",
        "entries": [],
        "stop_loss": None,
        "stop_basis": "",
        "take_profits": [],
        "invalidation": "no valid setup found",
        "event_risks": [],
        "risk_reward_min": 0.0,
        "confidence": "low",
    }


_NOOP_CLOSE = patch(
    "tradingagents.agents.validator.signal_validator._fetch_last_close",
    return_value=1000.0,
)

_NO_CLOSE = patch(
    "tradingagents.agents.validator.signal_validator._fetch_last_close",
    return_value=None,
)


# ---------------------------------------------------------------------------
# _validate: happy path
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestValidateHappyPath:
    def test_valid_long_passes(self):
        with _NOOP_CLOSE:
            result = _validate(_long_signal_dict(1000.0, 940.0, 1200.0), "TEST.NS", _cfg())
        assert result["valid"] is True
        assert result["error"] == ""
        assert result["sizing"] is not None
        assert result["ticket"] is not None

    def test_no_trade_always_valid(self):
        with _NOOP_CLOSE:
            result = _validate(_no_trade_dict(), "TEST.NS", _cfg())
        assert result["valid"] is True
        assert result["sizing"] is None
        assert "NO_TRADE" in result["ticket"]

    def test_valid_short_passes(self):
        with _NOOP_CLOSE:
            result = _validate(_short_signal_dict(1000.0, 1060.0, 850.0), "TEST.NS", _cfg())
        assert result["valid"] is True

    def test_ticket_rendered_for_valid_signal(self):
        with _NOOP_CLOSE:
            result = _validate(_long_signal_dict(1000.0, 940.0, 1250.0), "TEST.NS", _cfg())
        assert "LONG" in result["ticket"]
        assert "TEST.NS" in result["ticket"]


# ---------------------------------------------------------------------------
# _validate: SL side checks
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestValidateSLSide:
    def test_long_sl_above_entry_fails(self):
        sig = _long_signal_dict(entry_price=1000.0, stop_loss=1050.0, tp1_price=1300.0)
        with _NOOP_CLOSE:
            result = _validate(sig, "TEST.NS", _cfg())
        assert result["valid"] is False
        assert "not below" in result["error"].lower()

    def test_long_sl_at_entry_fails(self):
        sig = _long_signal_dict(entry_price=1000.0, stop_loss=1000.0, tp1_price=1200.0)
        with _NOOP_CLOSE:
            result = _validate(sig, "TEST.NS", _cfg())
        assert result["valid"] is False

    def test_short_sl_below_entry_fails(self):
        sig = _short_signal_dict(entry_price=1000.0, stop_loss=940.0, tp1_price=800.0)
        with _NOOP_CLOSE:
            result = _validate(sig, "TEST.NS", _cfg())
        assert result["valid"] is False
        assert "not above" in result["error"].lower()

    def test_short_sl_at_entry_fails(self):
        sig = _short_signal_dict(entry_price=1000.0, stop_loss=1000.0, tp1_price=800.0)
        with _NOOP_CLOSE:
            result = _validate(sig, "TEST.NS", _cfg())
        assert result["valid"] is False


# ---------------------------------------------------------------------------
# _validate: stale-level guard
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestValidateStaleLevels:
    def test_entry_too_far_from_close_fails(self):
        """Entry 25% from close (1300 actual vs 1000 entry) > 15% threshold."""
        sig = _long_signal_dict(entry_price=1000.0, stop_loss=940.0, tp1_price=1200.0)
        with patch("tradingagents.agents.validator.signal_validator._fetch_last_close",
                   return_value=1300.0):
            result = _validate(sig, "TEST.NS", _cfg(max_dev_pct=15.0))
        assert result["valid"] is False
        assert "%" in result["error"] or "close" in result["error"].lower()

    def test_entry_just_within_threshold_passes(self):
        """Entry 5% from close — within 15% threshold."""
        sig = _long_signal_dict(entry_price=1000.0, stop_loss=940.0, tp1_price=1250.0)
        with patch("tradingagents.agents.validator.signal_validator._fetch_last_close",
                   return_value=1050.0):
            result = _validate(sig, "TEST.NS", _cfg(max_dev_pct=15.0))
        assert result["valid"] is True

    def test_no_close_available_skips_stale_check(self):
        """Graceful degradation: if yfinance fails, skip the stale check."""
        sig = _long_signal_dict(entry_price=1000.0, stop_loss=940.0, tp1_price=1250.0)
        with _NO_CLOSE:
            result = _validate(sig, "TEST.NS", _cfg())
        assert result["valid"] is True

    def test_custom_dev_threshold_respected(self):
        """A tighter threshold (5%) should catch a 7% deviation."""
        sig = _long_signal_dict(entry_price=1000.0, stop_loss=940.0, tp1_price=1200.0)
        with patch("tradingagents.agents.validator.signal_validator._fetch_last_close",
                   return_value=1070.0):  # 7% deviation
            result = _validate(sig, "TEST.NS", _cfg(max_dev_pct=5.0))
        assert result["valid"] is False


# ---------------------------------------------------------------------------
# _validate: sizing rejections
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestValidateSizing:
    def test_unaffordable_signal_fails(self):
        """A tiny account that can't afford even 1 share is rejected."""
        sig = _long_signal_dict(entry_price=5000.0, stop_loss=4000.0, tp1_price=7000.0)
        with patch("tradingagents.agents.validator.signal_validator._fetch_last_close",
                   return_value=5000.0):
            result = _validate(sig, "TEST.NS", _cfg(equity=10_000))
        assert result["valid"] is False
        assert "unaffordable" in result["error"].lower()

    def test_rr_below_floor_fails(self):
        """Net RR below floor is caught here as well as in position_sizing."""
        # entry=1000, SL=950, TP=1040 → gross_rr=0.8 — far below 1.8
        sig = _long_signal_dict(entry_price=1000.0, stop_loss=950.0, tp1_price=1040.0)
        with _NOOP_CLOSE:
            result = _validate(sig, "TEST.NS", _cfg(min_rr=1.8))
        assert result["valid"] is False

    def test_valid_sizing_produces_ticket(self):
        sig = _long_signal_dict(entry_price=1000.0, stop_loss=900.0, tp1_price=1300.0)
        with _NOOP_CLOSE:
            result = _validate(sig, "TEST.NS", _cfg())
        assert result["valid"] is True
        assert result["ticket"] is not None
        assert "Qty" in result["ticket"]


# ---------------------------------------------------------------------------
# Signal validator node
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestSignalValidatorNode:
    def _make_node(self):
        return create_signal_validator()

    def test_passes_through_when_no_signal(self):
        node = self._make_node()
        state = {"company_of_interest": "TEST.NS", "trade_signal": None, "trader_retry_count": 0}
        with patch("tradingagents.dataflows.config.get_config", return_value=_cfg()):
            out = node(state)
        assert out["signal_validation_result"]["valid"] is True

    def test_valid_signal_passes_forward(self):
        node = self._make_node()
        state = {
            "company_of_interest": "TEST.NS",
            "trade_signal": _long_signal_dict(1000.0, 900.0, 1300.0),
            "trader_retry_count": 0,
        }
        with patch("tradingagents.dataflows.config.get_config", return_value=_cfg()):
            with _NOOP_CLOSE:
                out = node(state)
        assert out["signal_validation_result"]["valid"] is True
        assert "trader_retry_count" not in out

    def test_first_failure_increments_retry_count(self):
        """On first validation failure, retry_count is incremented to 1."""
        node = self._make_node()
        # SL above entry → will fail
        state = {
            "company_of_interest": "TEST.NS",
            "trade_signal": _long_signal_dict(1000.0, stop_loss=1050.0, tp1_price=1200.0),
            "trader_retry_count": 0,
        }
        with patch("tradingagents.dataflows.config.get_config", return_value=_cfg()):
            with _NOOP_CLOSE:
                out = node(state)
        assert out.get("trader_retry_count") == 1
        assert out["signal_validation_result"]["valid"] is False
        assert out["signal_validation_result"]["forced_no_trade"] is False

    def test_second_failure_forces_no_trade(self):
        """After retry is exhausted, the signal is forced to NO_TRADE."""
        node = self._make_node()
        state = {
            "company_of_interest": "TEST.NS",
            "trade_signal": _long_signal_dict(1000.0, stop_loss=1050.0, tp1_price=1200.0),
            "trader_retry_count": 1,
        }
        with patch("tradingagents.dataflows.config.get_config", return_value=_cfg()):
            with _NOOP_CLOSE:
                out = node(state)
        forced = out.get("trade_signal", {})
        assert forced.get("direction") == "NO_TRADE"
        assert out["signal_validation_result"]["forced_no_trade"] is True

    def test_retry_count_not_incremented_for_valid_signal(self):
        node = self._make_node()
        state = {
            "company_of_interest": "TEST.NS",
            "trade_signal": _long_signal_dict(1000.0, 900.0, 1300.0),
            "trader_retry_count": 0,
        }
        with patch("tradingagents.dataflows.config.get_config", return_value=_cfg()):
            with _NOOP_CLOSE:
                out = node(state)
        # Valid signal: no retry_count change in output dict
        assert "trader_retry_count" not in out


# ---------------------------------------------------------------------------
# _validate_position_action: SL-widen rejection
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestPositionActionValidator:
    def _long_pos(self, stop_loss: float = 950.0, avg_entry: float = 1000.0) -> dict:
        return {
            "direction": "LONG",
            "stop_loss": stop_loss,
            "avg_entry": avg_entry,
            "qty_open": 10,
        }

    def _short_pos(self, stop_loss: float = 1050.0, avg_entry: float = 1000.0) -> dict:
        return {
            "direction": "SHORT",
            "stop_loss": stop_loss,
            "avg_entry": avg_entry,
            "qty_open": 10,
        }

    def _action(self, action: str, **kwargs) -> dict:
        base = {
            "action": action,
            "exit_pct": None,
            "new_stop_loss": None,
            "add_entry": None,
            "thesis_status": "intact",
            "reasoning": "test",
        }
        base.update(kwargs)
        return base

    # --- SL widen (LONG) ---

    def test_raise_sl_long_must_go_higher(self):
        """Moving SL lower on a LONG position is widening — must be rejected."""
        action = self._action("RAISE_SL", new_stop_loss=900.0)  # lower than current 950
        result = _validate_position_action(action, self._long_pos(stop_loss=950.0), _cfg())
        assert result["valid"] is False
        assert "higher" in result["error"].lower()

    def test_raise_sl_long_at_same_level_rejected(self):
        """Same SL is not a move in profit direction."""
        action = self._action("RAISE_SL", new_stop_loss=950.0)
        result = _validate_position_action(action, self._long_pos(stop_loss=950.0), _cfg())
        assert result["valid"] is False

    def test_raise_sl_long_higher_accepted(self):
        """Moving SL higher on LONG (tightening) is valid."""
        action = self._action("RAISE_SL", new_stop_loss=980.0)
        result = _validate_position_action(action, self._long_pos(stop_loss=950.0), _cfg())
        assert result["valid"] is True

    # --- SL widen (SHORT) ---

    def test_raise_sl_short_must_go_lower(self):
        """Moving SL higher on a SHORT position is widening — must be rejected."""
        action = self._action("RAISE_SL", new_stop_loss=1100.0)  # higher than current 1050
        result = _validate_position_action(action, self._short_pos(stop_loss=1050.0), _cfg())
        assert result["valid"] is False
        assert "lower" in result["error"].lower()

    def test_raise_sl_short_lower_accepted(self):
        """Moving SL lower on SHORT (tightening) is valid."""
        action = self._action("RAISE_SL", new_stop_loss=1020.0)
        result = _validate_position_action(action, self._short_pos(stop_loss=1050.0), _cfg())
        assert result["valid"] is True

    # --- Broken thesis guard ---

    def test_broken_thesis_with_hold_rejected(self):
        """thesis_status=broken + action=HOLD is never allowed."""
        action = self._action("HOLD", thesis_status="broken")
        result = _validate_position_action(action, {}, _cfg())
        assert result["valid"] is False
        assert "broken" in result["error"].lower()

    def test_broken_thesis_exit_full_accepted(self):
        action = self._action("EXIT_FULL", thesis_status="broken")
        result = _validate_position_action(action, {}, _cfg())
        assert result["valid"] is True

    def test_broken_thesis_exit_partial_accepted(self):
        action = self._action("EXIT_PARTIAL", thesis_status="broken", exit_pct=50.0)
        result = _validate_position_action(action, {}, _cfg())
        assert result["valid"] is True

    def test_intact_thesis_hold_accepted(self):
        action = self._action("HOLD", thesis_status="intact")
        result = _validate_position_action(action, {}, _cfg())
        assert result["valid"] is True

    def test_weakened_thesis_hold_accepted(self):
        """Weakened thesis still allows HOLD (only 'broken' forces an exit)."""
        action = self._action("HOLD", thesis_status="weakened")
        result = _validate_position_action(action, {}, _cfg())
        assert result["valid"] is True

    # --- Misc valid actions ---

    def test_hold_no_sl_change_valid(self):
        action = self._action("HOLD")
        result = _validate_position_action(action, self._long_pos(), _cfg())
        assert result["valid"] is True

    def test_exit_full_valid(self):
        action = self._action("EXIT_FULL")
        result = _validate_position_action(action, self._long_pos(), _cfg())
        assert result["valid"] is True

    def test_exit_partial_valid(self):
        action = self._action("EXIT_PARTIAL", exit_pct=50.0)
        result = _validate_position_action(action, self._long_pos(), _cfg())
        assert result["valid"] is True


# ---------------------------------------------------------------------------
# Position action validator node
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestPositionActionValidatorNode:
    def test_no_action_passes_through(self):
        node = create_position_action_validator()
        state = {"position_action": None, "open_position": {}}
        with patch("tradingagents.dataflows.config.get_config", return_value=_cfg()):
            out = node(state)
        assert out["signal_validation_result"]["valid"] is True

    def test_valid_action_passes(self):
        node = create_position_action_validator()
        state = {
            "position_action": {
                "action": "HOLD",
                "exit_pct": None,
                "new_stop_loss": None,
                "add_entry": None,
                "thesis_status": "intact",
                "reasoning": "test",
            },
            "open_position": {"direction": "LONG", "stop_loss": 950.0},
        }
        with patch("tradingagents.dataflows.config.get_config", return_value=_cfg()):
            out = node(state)
        assert out["signal_validation_result"]["valid"] is True

    def test_invalid_action_captured_in_result(self):
        node = create_position_action_validator()
        state = {
            "position_action": {
                "action": "RAISE_SL",
                "exit_pct": None,
                "new_stop_loss": 900.0,  # wrong direction for LONG
                "add_entry": None,
                "thesis_status": "intact",
                "reasoning": "test",
            },
            "open_position": {"direction": "LONG", "stop_loss": 950.0},
        }
        with patch("tradingagents.dataflows.config.get_config", return_value=_cfg()):
            out = node(state)
        assert out["signal_validation_result"]["valid"] is False
        assert out["signal_validation_result"]["error"] != ""
