"""Pure-Python deterministic position sizing calculator.

The LLM proposes the trade structure (entries, SL, TPs). This module computes
the share quantity, capital allocation, and risk figures deterministically from
those levels and the account configuration. No LLM involved.

Affordability rule: if qty rounds to 0 shares within the risk caps the result
carries affordable=False and the Signal Validator downgrades to NO_TRADE.

Net-RR rule: risk-reward is checked after subtracting round-trip transaction
costs so small accounts see the true economics of a trade.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from tradingagents.agents.schemas import TradeSignal


@dataclass
class PositionSizingResult:
    """Output of size_position()."""

    qty: int                    # shares (floored to whole number)
    avg_entry_price: float      # weighted avg across all entry levels
    capital_inr: float          # total capital deployed = qty × avg_entry
    capital_pct: float          # capital as % of account equity
    risk_inr: float             # risk per share × qty
    risk_pct: float             # risk_inr / account_equity × 100
    risk_reward_gross: float    # (TP1 price − avg_entry) / risk_per_share  [long]
    risk_reward_net: float      # gross RR minus transaction-cost drag
    affordable: bool            # False when qty == 0
    cap_breached: bool          # capital_pct > max_position_pct
    open_risk_breached: bool    # adding this risk would exceed max_open_risk_pct
    rejection_reason: str       # empty string when sizing is valid

    def to_dict(self) -> dict:
        return {
            "qty": self.qty,
            "avg_entry_price": self.avg_entry_price,
            "capital_inr": self.capital_inr,
            "capital_pct": self.capital_pct,
            "risk_inr": self.risk_inr,
            "risk_pct": self.risk_pct,
            "risk_reward_gross": self.risk_reward_gross,
            "risk_reward_net": self.risk_reward_net,
            "affordable": self.affordable,
            "cap_breached": self.cap_breached,
            "open_risk_breached": self.open_risk_breached,
            "rejection_reason": self.rejection_reason,
        }

    @property
    def is_valid(self) -> bool:
        return bool(self.affordable and not self.cap_breached and not self.open_risk_breached and not self.rejection_reason)

    def format_ticket(self, ticker: str, signal: "TradeSignal") -> str:
        """Render the complete executable signal ticket."""
        if not self.affordable or signal.direction == "NO_TRADE":
            return f"SIGNAL — {ticker} — NO_TRADE\nReason: {self.rejection_reason or 'signal is NO_TRADE'}"

        entry_parts = []
        for e in signal.entries:
            entry_parts.append(f"{e.label} ₹{e.price:,.0f} ({e.allocation_pct:.0f}%) {e.trigger}")
        entries_str = " | ".join(entry_parts)

        tp_parts = []
        for t in signal.take_profits:
            tp_parts.append(f"{t.label} ₹{t.price:,.0f} ({t.exit_pct:.0f}% off)")
        tps_str = " | ".join(tp_parts)

        capital_l = self.capital_inr / 100_000  # convert to lakhs

        lines = [
            f"SIGNAL — {ticker} — {signal.direction} — {signal.timeframe}",
            entries_str,
            f"SL  ₹{signal.stop_loss:,.0f}  ({signal.stop_basis})",
            tps_str,
            (
                f"Qty {self.qty} | "
                f"Capital ₹{capital_l:.2f}L ({self.capital_pct:.1f}% of equity) | "
                f"Risk ₹{self.risk_inr:,.0f} ({self.risk_pct:.2f}%) | "
                f"RR {self.risk_reward_net:.2f} (net)"
            ),
        ]
        if signal.event_risks:
            lines.append(f"Event risks: {'; '.join(signal.event_risks)}")
        return "\n".join(lines)


def size_position(
    signal: "TradeSignal",
    account_equity: float,
    risk_pct_per_trade: float,
    open_portfolio_risk_pct: float = 0.0,
    max_open_risk_pct: float = 6.0,
    max_position_pct: float = 15.0,
    min_risk_reward: float = 1.8,
    txn_cost_pct_round_trip: float = 0.5,
) -> PositionSizingResult:
    """Compute position sizing from a TradeSignal and account parameters.

    Args:
        signal: Validated TradeSignal with entries, stop_loss, take_profits.
        account_equity: Total account equity in INR.
        risk_pct_per_trade: Max % of equity to risk between avg entry and SL.
        open_portfolio_risk_pct: Existing open risk as % of equity.
        max_open_risk_pct: Hard cap on total portfolio open risk %.
        max_position_pct: Hard cap on single position capital as % of equity.
        min_risk_reward: Minimum acceptable net RR after transaction costs.
        txn_cost_pct_round_trip: Estimated round-trip transaction cost %.
    """
    _no_trade = PositionSizingResult(
        qty=0, avg_entry_price=0.0, capital_inr=0.0, capital_pct=0.0,
        risk_inr=0.0, risk_pct=0.0, risk_reward_gross=0.0, risk_reward_net=0.0,
        affordable=False, cap_breached=False, open_risk_breached=False,
        rejection_reason="NO_TRADE signal — no sizing computed",
    )

    if signal.direction == "NO_TRADE" or not signal.entries or signal.stop_loss is None:
        return _no_trade

    # Weighted average entry price
    total_alloc = sum(e.allocation_pct for e in signal.entries)
    if total_alloc > 0:
        avg_entry = sum(e.price * e.allocation_pct for e in signal.entries) / total_alloc
    else:
        avg_entry = signal.entries[0].price

    stop = signal.stop_loss
    direction_sign = 1 if signal.direction == "LONG" else -1
    risk_per_share = direction_sign * (avg_entry - stop)  # positive when SL is correct

    if risk_per_share <= 0:
        return PositionSizingResult(
            qty=0, avg_entry_price=avg_entry, capital_inr=0.0, capital_pct=0.0,
            risk_inr=0.0, risk_pct=0.0, risk_reward_gross=0.0, risk_reward_net=0.0,
            affordable=False, cap_breached=False, open_risk_breached=False,
            rejection_reason=(
                f"SL on wrong side of entry: avg_entry={avg_entry:.2f}, "
                f"stop={stop:.2f}, direction={signal.direction}"
            ),
        )

    # Quantity from risk cap
    max_risk_inr = account_equity * risk_pct_per_trade / 100.0
    qty_from_risk = math.floor(max_risk_inr / risk_per_share)

    # Quantity from position cap
    max_capital = account_equity * max_position_pct / 100.0
    qty_from_capital = math.floor(max_capital / avg_entry)

    qty = min(qty_from_risk, qty_from_capital)

    # Gross RR using TP1
    if signal.take_profits:
        tp1_price = signal.take_profits[0].price
        gross_rr = direction_sign * (tp1_price - avg_entry) / risk_per_share
    else:
        gross_rr = 0.0

    # Net RR deducting round-trip transaction cost
    txn_cost_per_share = avg_entry * txn_cost_pct_round_trip / 100.0
    net_rr = gross_rr - (txn_cost_per_share / risk_per_share) if risk_per_share > 0 else 0.0

    capital_inr = qty * avg_entry
    capital_pct = (capital_inr / account_equity * 100.0) if account_equity > 0 else 0.0
    risk_inr = qty * risk_per_share
    risk_pct_actual = (risk_inr / account_equity * 100.0) if account_equity > 0 else 0.0

    affordable = qty > 0
    # Float tolerance: use a small epsilon to avoid false positives from rounding
    cap_breached = capital_pct > (max_position_pct + 0.01)
    open_risk_breached = (open_portfolio_risk_pct + risk_pct_actual) > (max_open_risk_pct + 0.01)

    rejection_reason = ""
    if not affordable:
        rejection_reason = "position unaffordable within risk caps (qty rounds to 0)"
    elif cap_breached:
        rejection_reason = (
            f"position capital {capital_pct:.1f}% exceeds max {max_position_pct:.1f}%"
        )
    elif open_risk_breached:
        rejection_reason = (
            f"adding {risk_pct_actual:.2f}% risk would bring total open risk to "
            f"{open_portfolio_risk_pct + risk_pct_actual:.2f}% "
            f"(cap: {max_open_risk_pct:.1f}%)"
        )
    elif net_rr < min_risk_reward:
        rejection_reason = (
            f"net RR {net_rr:.2f} below floor {min_risk_reward:.2f} "
            f"(gross {gross_rr:.2f} minus txn cost drag)"
        )

    return PositionSizingResult(
        qty=qty,
        avg_entry_price=avg_entry,
        capital_inr=capital_inr,
        capital_pct=capital_pct,
        risk_inr=risk_inr,
        risk_pct=risk_pct_actual,
        risk_reward_gross=gross_rr,
        risk_reward_net=net_rr,
        affordable=affordable,
        cap_breached=cap_breached,
        open_risk_breached=open_risk_breached,
        rejection_reason=rejection_reason,
    )
