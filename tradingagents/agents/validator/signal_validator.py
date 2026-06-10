"""Signal Validator: pure-code node between Trader and risk debators.

This node runs NO LLM. It validates the TradeSignal produced by the Trader,
runs deterministic position sizing, and either:

  - marks the signal valid and passes it forward to the risk debate, or
  - marks it invalid, increments trader_retry_count, and sends it back to the
    Trader with the exact rejection reason (one retry only), or
  - on a second failure, forces NO_TRADE and passes forward.

Validation checks (in order):
  1. trade_signal is present and parseable — if not, pass forward as-is (no
     structured output from provider; validator is a no-op).
  2. NO_TRADE: always valid; sizing is a zero-result; pass forward.
  3. SL is on the correct side of avg entry (LONG: SL < entry; SHORT: SL > entry).
  4. All entries are within max_entry_deviation_pct of the last verified close
     (stale-level guard).
  5. Position sizing: qty > 0 (affordable), within capital cap, within open
     risk cap, and net RR >= min_risk_reward.

The Signal Validator is the single strongest "fewer mistakes" win in the
pipeline — it is what stands between an LLM arithmetic slip and a bad order.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


def _fetch_last_close(ticker: str) -> float | None:
    """Fetch the most recent closing price for a ticker via yfinance.

    Returns None on any failure so the validator degrades gracefully rather
    than blocking the pipeline.
    """
    try:
        import yfinance as yf
        hist = yf.Ticker(ticker).history(period="5d")
        if hist.empty:
            return None
        return float(hist["Close"].iloc[-1])
    except Exception as exc:  # noqa: BLE001
        logger.debug("Signal Validator: could not fetch last close for %s: %s", ticker, exc)
        return None


def _validate(
    signal_dict: dict,
    ticker: str,
    cfg: dict,
    open_portfolio_risk_pct: float = 0.0,
) -> dict:
    """Run all validation checks and return a validation result dict.

    Returns:
        {
            "valid": bool,
            "forced_no_trade": bool,
            "error": str,        # empty when valid
            "sizing": dict|None, # PositionSizingResult.to_dict() or None
            "ticket": str|None,  # formatted signal ticket or None
        }
    """
    from tradingagents.agents.schemas import TradeSignal
    from tradingagents.risk.position_sizing import size_position

    account_equity = cfg.get("account_equity_inr", 1_000_000)
    risk_pct = cfg.get("risk_pct_per_trade", 1.0)
    max_open_risk = cfg.get("max_open_risk_pct", 6.0)
    max_pos_pct = cfg.get("max_position_pct", 15.0)
    min_rr = cfg.get("min_risk_reward", 1.8)
    txn_cost = cfg.get("txn_cost_pct_round_trip", 0.5)
    max_dev_pct = cfg.get("max_entry_deviation_pct", 15.0)

    try:
        signal = TradeSignal.model_validate(signal_dict)
    except Exception as exc:
        return {
            "valid": False,
            "forced_no_trade": False,
            "error": f"TradeSignal parse error: {exc}",
            "sizing": None,
            "ticket": None,
        }

    # NO_TRADE is always valid
    if signal.direction == "NO_TRADE":
        return {
            "valid": True,
            "forced_no_trade": False,
            "error": "",
            "sizing": None,
            "ticket": f"SIGNAL — {ticker} — NO_TRADE\nReason: {signal.invalidation}",
        }

    # --- SL direction check ---
    avg_entry: float
    total_alloc = sum(e.allocation_pct for e in signal.entries)
    if total_alloc > 0:
        avg_entry = sum(e.price * e.allocation_pct for e in signal.entries) / total_alloc
    else:
        avg_entry = signal.entries[0].price if signal.entries else 0.0

    sl = signal.stop_loss or 0.0
    if signal.direction == "LONG" and sl >= avg_entry:
        return {
            "valid": False,
            "forced_no_trade": False,
            "error": (
                f"LONG stop-loss ₹{sl:,.2f} is not below avg entry ₹{avg_entry:,.2f}. "
                "SL must be placed below a structural level beneath the entry."
            ),
            "sizing": None,
            "ticket": None,
        }
    if signal.direction == "SHORT" and sl <= avg_entry:
        return {
            "valid": False,
            "forced_no_trade": False,
            "error": (
                f"SHORT stop-loss ₹{sl:,.2f} is not above avg entry ₹{avg_entry:,.2f}. "
                "SL must be placed above a structural level above the entry."
            ),
            "sizing": None,
            "ticket": None,
        }

    # --- Stale-level guard: entries within max_dev_pct of last close ---
    last_close = _fetch_last_close(ticker)
    if last_close is not None and last_close > 0:
        for entry in signal.entries:
            deviation_pct = abs(entry.price - last_close) / last_close * 100.0
            if deviation_pct > max_dev_pct:
                return {
                    "valid": False,
                    "forced_no_trade": False,
                    "error": (
                        f"{entry.label} price ₹{entry.price:,.2f} is {deviation_pct:.1f}% "
                        f"from last close ₹{last_close:,.2f} (max allowed: {max_dev_pct:.0f}%). "
                        "Use price levels from the current Market Analyst report."
                    ),
                    "sizing": None,
                    "ticket": None,
                }

    # --- Position sizing (affordability + RR check) ---
    sizing = size_position(
        signal=signal,
        account_equity=account_equity,
        risk_pct_per_trade=risk_pct,
        open_portfolio_risk_pct=open_portfolio_risk_pct,
        max_open_risk_pct=max_open_risk,
        max_position_pct=max_pos_pct,
        min_risk_reward=min_rr,
        txn_cost_pct_round_trip=txn_cost,
    )

    if sizing.rejection_reason:
        return {
            "valid": False,
            "forced_no_trade": False,
            "error": sizing.rejection_reason,
            "sizing": sizing.to_dict(),
            "ticket": None,
        }

    ticket = sizing.format_ticket(ticker, signal, txn_cost_pct=txn_cost)
    return {
        "valid": True,
        "forced_no_trade": False,
        "error": "",
        "sizing": sizing.to_dict(),
        "ticket": ticket,
    }


def create_signal_validator():
    """Return a pure-code LangGraph node that validates the Trader's TradeSignal.

    Routing (via conditional edge in graph/conditional_logic.py):
      - "valid"   → proceed to Aggressive Analyst
      - "retry"   → send back to Trader (retry_count incremented to 1)
      - "forward" → forced NO_TRADE after exhausting retry; proceed forward
    """

    def signal_validator_node(state: Any) -> dict:
        from tradingagents.dataflows.config import get_config

        cfg = get_config()
        ticker = state.get("company_of_interest", "UNKNOWN")
        signal_dict = state.get("trade_signal")
        retry_count = state.get("trader_retry_count", 0)

        # No structured signal (provider didn't support it) — pass through.
        if signal_dict is None:
            logger.debug(
                "Signal Validator: no trade_signal in state (free-text fallback); "
                "passing through without validation."
            )
            return {
                "signal_validation_result": {
                    "valid": True,
                    "forced_no_trade": False,
                    "error": "",
                    "sizing": None,
                    "ticket": None,
                }
            }

        result = _validate(signal_dict, ticker, cfg)

        if result["valid"]:
            return {"signal_validation_result": result}

        # Invalid signal
        if retry_count >= 1:
            # Exhausted retry — force NO_TRADE and proceed
            logger.info(
                "Signal Validator: retry exhausted for %s; forcing NO_TRADE. Reason: %s",
                ticker, result["error"],
            )
            forced_signal = {
                "direction": "NO_TRADE",
                "setup_type": "forced-no-trade",
                "timeframe": "",
                "entries": [],
                "stop_loss": None,
                "stop_basis": "",
                "take_profits": [],
                "invalidation": f"Validator forced NO_TRADE after retry. Original error: {result['error']}",
                "event_risks": [],
                "risk_reward_min": 0.0,
                "confidence": "low",
            }
            ticket = f"SIGNAL — {ticker} — NO_TRADE\nReason (validator forced): {result['error']}"
            return {
                "trade_signal": forced_signal,
                "signal_validation_result": {
                    "valid": True,
                    "forced_no_trade": True,
                    "error": result["error"],
                    "sizing": None,
                    "ticket": ticket,
                },
            }

        # First failure — request a retry
        logger.info(
            "Signal Validator: signal rejected for %s (attempt 1). Reason: %s",
            ticker, result["error"],
        )
        return {
            "signal_validation_result": result,
            "trader_retry_count": retry_count + 1,
        }

    return signal_validator_node


# ---------------------------------------------------------------------------
# Phase 4 — Position Action Validator (manage_position mode)
# ---------------------------------------------------------------------------


def _validate_position_action(
    action_dict: dict,
    position: dict,
    cfg: dict,
) -> dict:
    """Validate a PositionAction against manage-mode hard rules.

    Returns:
        {
            "valid": bool,
            "error": str,  # empty when valid
        }
    """
    from tradingagents.agents.schemas import PositionAction

    try:
        action = PositionAction.model_validate(action_dict)
    except Exception as exc:
        return {"valid": False, "error": f"PositionAction parse error: {exc}"}

    # thesis_status=broken must not be HOLD (also enforced by schema, but hard-check here)
    if action.thesis_status == "broken" and action.action == "HOLD":
        return {
            "valid": False,
            "error": (
                "thesis_status='broken' but action='HOLD'. "
                "A broken thesis requires an exit action."
            ),
        }

    # RAISE_SL: new SL must tighten (move in profit direction only)
    if action.action == "RAISE_SL" and action.new_stop_loss is not None:
        current_sl = position.get("stop_loss")
        direction = position.get("direction", "LONG")
        if current_sl is not None:
            if direction == "LONG" and action.new_stop_loss <= current_sl:
                return {
                    "valid": False,
                    "error": (
                        f"RAISE_SL rejected: new_stop_loss ₹{action.new_stop_loss:,.2f} is not "
                        f"above current SL ₹{current_sl:,.2f}. "
                        "For a LONG position, raising the SL means moving it HIGHER."
                    ),
                }
            if direction == "SHORT" and action.new_stop_loss >= current_sl:
                return {
                    "valid": False,
                    "error": (
                        f"RAISE_SL rejected: new_stop_loss ₹{action.new_stop_loss:,.2f} is not "
                        f"below current SL ₹{current_sl:,.2f}. "
                        "For a SHORT position, raising the SL means moving it LOWER."
                    ),
                }

    # ADD: re-validate position sizing caps
    if action.action == "ADD" and action.add_entry is not None:
        import json as _json
        from tradingagents.agents.schemas import TakeProfit, TradeSignal
        from tradingagents.risk.position_sizing import size_position

        account_equity = cfg.get("account_equity_inr", 1_000_000)
        risk_pct = cfg.get("risk_pct_per_trade", 1.0)
        max_open_risk = cfg.get("max_open_risk_pct", 6.0)
        max_pos_pct = cfg.get("max_position_pct", 15.0)
        txn_cost = cfg.get("txn_cost_pct_round_trip", 0.5)

        sl = position.get("stop_loss")
        direction = position.get("direction", "LONG")
        avg_entry = position.get("avg_entry")
        qty_open = position.get("qty_open") or 0
        tps_raw = position.get("take_profits_json") or []

        if isinstance(tps_raw, str):
            try:
                tps_raw = _json.loads(tps_raw)
            except Exception:
                tps_raw = []

        # Compute existing open risk to check caps correctly
        existing_risk_pct = 0.0
        if avg_entry and sl and qty_open > 0:
            sign = 1 if direction == "LONG" else -1
            rps = sign * (avg_entry - sl)
            if rps > 0:
                existing_risk_pct = rps * qty_open / account_equity * 100.0

        if sl is not None:
            tps = []
            for tp_d in tps_raw:
                try:
                    tps.append(TakeProfit(**tp_d))
                except Exception:
                    pass
            if not tps:
                tps = [TakeProfit(
                    label="TP1",
                    price=float(action.add_entry.price) * 1.1,
                    exit_pct=100,
                    basis="placeholder",
                )]
            try:
                mini_signal = TradeSignal(
                    direction=direction,
                    setup_type="add",
                    timeframe="",
                    entries=[action.add_entry],
                    stop_loss=sl,
                    stop_basis="carry from original position",
                    take_profits=tps,
                    invalidation="add action",
                    event_risks=[],
                    risk_reward_min=0.0,
                    confidence="medium",
                )
                sizing = size_position(
                    signal=mini_signal,
                    account_equity=account_equity,
                    risk_pct_per_trade=risk_pct,
                    open_portfolio_risk_pct=existing_risk_pct,
                    max_open_risk_pct=max_open_risk,
                    max_position_pct=max_pos_pct,
                    min_risk_reward=0.0,
                    txn_cost_pct_round_trip=txn_cost,
                )
                if not sizing.affordable:
                    return {
                        "valid": False,
                        "error": (
                            f"ADD rejected: {sizing.rejection_reason}. "
                            "Cannot add to position within current risk caps."
                        ),
                    }
            except Exception as exc:
                logger.debug("Position Action Validator: ADD sizing check failed: %s", exc)

    return {"valid": True, "error": ""}


def create_position_action_validator():
    """Return a pure-code LangGraph node that validates the Position Manager's PositionAction.

    Passes through without validation penalty if no structured action is present.
    """

    def position_action_validator_node(state) -> dict:
        from tradingagents.dataflows.config import get_config

        cfg = get_config()
        action_dict = state.get("position_action")
        position = state.get("open_position") or {}

        if action_dict is None:
            logger.debug("Position Action Validator: no position_action; passing through.")
            return {
                "signal_validation_result": {
                    "valid": True,
                    "error": "",
                    "forced_no_trade": False,
                    "sizing": None,
                    "ticket": None,
                }
            }

        result = _validate_position_action(action_dict, position, cfg)
        logger.info(
            "Position Action Validator: %s — %s",
            "PASS" if result["valid"] else "FAIL",
            result.get("error") or "ok",
        )
        return {
            "signal_validation_result": {
                "valid": result["valid"],
                "error": result.get("error", ""),
                "forced_no_trade": False,
                "sizing": None,
                "ticket": None,
            }
        }

    return position_action_validator_node
