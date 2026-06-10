"""Pydantic schemas used by agents that produce structured output.

The framework's primary artifact is still prose: each agent's natural-language
reasoning is what users read in the saved markdown reports and what the
downstream agents read as context.  Structured output is layered onto the
three decision-making agents (Research Manager, Trader, Portfolio Manager)
so that:

- Their outputs follow consistent section headers across runs and providers
- Each provider's native structured-output mode is used (json_schema for
  OpenAI/xAI, response_schema for Gemini, tool-use for Anthropic)
- Schema field descriptions become the model's output instructions, freeing
  the prompt body to focus on context and the rating-scale guidance
- A render helper turns the parsed Pydantic instance back into the same
  markdown shape the rest of the system already consumes, so display,
  memory log, and saved reports keep working unchanged
"""

from __future__ import annotations

from enum import Enum
from typing import Literal, Optional

from pydantic import BaseModel, Field, model_validator


# ---------------------------------------------------------------------------
# Shared rating types
# ---------------------------------------------------------------------------


class PortfolioRating(str, Enum):
    """5-tier rating used by the Research Manager and Portfolio Manager."""

    BUY = "Buy"
    OVERWEIGHT = "Overweight"
    HOLD = "Hold"
    UNDERWEIGHT = "Underweight"
    SELL = "Sell"


class TraderAction(str, Enum):
    """3-tier transaction direction used by the Trader.

    The Trader's job is to translate the Research Manager's investment plan
    into a concrete transaction proposal: should the desk execute a Buy, a
    Sell, or sit on Hold this round.  Position sizing and the nuanced
    Overweight / Underweight calls happen later at the Portfolio Manager.
    """

    BUY = "Buy"
    HOLD = "Hold"
    SELL = "Sell"


# ---------------------------------------------------------------------------
# Research Manager
# ---------------------------------------------------------------------------


class ResearchPlan(BaseModel):
    """Structured investment plan produced by the Research Manager.

    Hand-off to the Trader: the recommendation pins the directional view,
    the rationale captures which side of the bull/bear debate carried the
    argument, and the strategic actions translate that into concrete
    instructions the trader can execute against.
    """

    recommendation: PortfolioRating = Field(
        description=(
            "The investment recommendation. Exactly one of Buy / Overweight / "
            "Hold / Underweight / Sell. Reserve Hold for situations where the "
            "evidence on both sides is genuinely balanced; otherwise commit to "
            "the side with the stronger arguments."
        ),
    )
    rationale: str = Field(
        description=(
            "Conversational summary of the key points from both sides of the "
            "debate, ending with which arguments led to the recommendation. "
            "Speak naturally, as if to a teammate."
        ),
    )
    strategic_actions: str = Field(
        description=(
            "Concrete steps for the trader to implement the recommendation, "
            "including position sizing guidance consistent with the rating."
        ),
    )


def render_research_plan(plan: ResearchPlan) -> str:
    """Render a ResearchPlan to markdown for storage and the trader's prompt context."""
    return "\n".join([
        f"**Recommendation**: {plan.recommendation.value}",
        "",
        f"**Rationale**: {plan.rationale}",
        "",
        f"**Strategic Actions**: {plan.strategic_actions}",
    ])


# ---------------------------------------------------------------------------
# Trader
# ---------------------------------------------------------------------------


class TraderProposal(BaseModel):
    """Structured transaction proposal produced by the Trader.

    The trader reads the Research Manager's investment plan and the analyst
    reports, then turns them into a concrete transaction: what action to
    take, the reasoning that justifies it, and the practical levels for
    entry, stop-loss, and sizing.
    """

    action: TraderAction = Field(
        description="The transaction direction. Exactly one of Buy / Hold / Sell.",
    )
    reasoning: str = Field(
        description=(
            "The case for this action, anchored in the analysts' reports and "
            "the research plan. Two to four sentences."
        ),
    )
    entry_price: Optional[float] = Field(
        default=None,
        description="Optional entry price target in the instrument's quote currency.",
    )
    stop_loss: Optional[float] = Field(
        default=None,
        description="Optional stop-loss price in the instrument's quote currency.",
    )
    position_sizing: Optional[str] = Field(
        default=None,
        description="Optional sizing guidance, e.g. '5% of portfolio'.",
    )


def render_trader_proposal(proposal: TraderProposal) -> str:
    """Render a TraderProposal to markdown.

    The trailing ``FINAL TRANSACTION PROPOSAL: **BUY/HOLD/SELL**`` line is
    preserved for backward compatibility with the analyst stop-signal text
    and any external code that greps for it.
    """
    parts = [
        f"**Action**: {proposal.action.value}",
        "",
        f"**Reasoning**: {proposal.reasoning}",
    ]
    if proposal.entry_price is not None:
        parts.extend(["", f"**Entry Price**: {proposal.entry_price}"])
    if proposal.stop_loss is not None:
        parts.extend(["", f"**Stop Loss**: {proposal.stop_loss}"])
    if proposal.position_sizing:
        parts.extend(["", f"**Position Sizing**: {proposal.position_sizing}"])
    parts.extend([
        "",
        f"FINAL TRANSACTION PROPOSAL: **{proposal.action.value.upper()}**",
    ])
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Portfolio Manager
# ---------------------------------------------------------------------------


class PortfolioDecision(BaseModel):
    """Structured output produced by the Portfolio Manager.

    The model fills every field as part of its primary LLM call; no separate
    extraction pass is required. Field descriptions double as the model's
    output instructions, so the prompt body only needs to convey context and
    the rating-scale guidance.
    """

    rating: PortfolioRating = Field(
        description=(
            "The final position rating. Exactly one of Buy / Overweight / Hold / "
            "Underweight / Sell, picked based on the analysts' debate."
        ),
    )
    executive_summary: str = Field(
        description=(
            "A concise action plan covering entry strategy, position sizing, "
            "key risk levels, and time horizon. Two to four sentences."
        ),
    )
    investment_thesis: str = Field(
        description=(
            "Detailed reasoning anchored in specific evidence from the analysts' "
            "debate. If prior lessons are referenced in the prompt context, "
            "incorporate them; otherwise rely solely on the current analysis."
        ),
    )
    price_target: Optional[float] = Field(
        default=None,
        description="Optional target price in the instrument's quote currency.",
    )
    time_horizon: Optional[str] = Field(
        default=None,
        description="Optional recommended holding period, e.g. '3-6 months'.",
    )


def render_pm_decision(decision: PortfolioDecision) -> str:
    """Render a PortfolioDecision back to the markdown shape the rest of the system expects.

    Memory log, CLI display, and saved report files all read this markdown,
    so the rendered output preserves the exact section headers (``**Rating**``,
    ``**Executive Summary**``, ``**Investment Thesis**``) that downstream
    parsers and the report writers already handle.
    """
    parts = [
        f"**Rating**: {decision.rating.value}",
        "",
        f"**Executive Summary**: {decision.executive_summary}",
        "",
        f"**Investment Thesis**: {decision.investment_thesis}",
    ]
    if decision.price_target is not None:
        parts.extend(["", f"**Price Target**: {decision.price_target}"])
    if decision.time_horizon:
        parts.extend(["", f"**Time Horizon**: {decision.time_horizon}"])
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Sentiment Analyst
# ---------------------------------------------------------------------------


class SentimentBand(str, Enum):
    """Discrete sentiment direction produced by the Sentiment Analyst.

    Six tiers keep the signal granular enough to be actionable while remaining
    small enough for every provider to map reliably from its JSON output.
    """

    BULLISH = "Bullish"
    MILDLY_BULLISH = "Mildly Bullish"
    NEUTRAL = "Neutral"
    MIXED = "Mixed"
    MILDLY_BEARISH = "Mildly Bearish"
    BEARISH = "Bearish"


class SentimentReport(BaseModel):
    """Structured sentiment report produced by the Sentiment Analyst.

    Replaces the previous free-form prose output so downstream consumers
    (dashboards, audit logs, PDF renderers, other agents) can read
    ``overall_band`` and ``overall_score`` without maintaining fragile regex
    fallbacks that drift with every model release. ``narrative`` preserves the
    rich source-by-source analysis; ``render_sentiment_report`` prepends a
    deterministic header so the saved report stays human-readable.
    """

    overall_band: SentimentBand = Field(
        description=(
            "Overall sentiment direction. Exactly one of: "
            "Bullish / Mildly Bullish / Neutral / Mixed / Mildly Bearish / Bearish. "
            "Use Mixed when sources point in clearly different directions. "
            "Use Neutral only when all sources are genuinely silent or non-committal."
        ),
    )
    overall_score: float = Field(
        ge=0.0,
        le=10.0,
        description=(
            "Numeric sentiment intensity on a 0–10 scale. "
            "0 = maximally bearish, 5 = neutral, 10 = maximally bullish. "
            "Guideline for consistency with overall_band: "
            "Bullish ~6.5–10, Mildly Bullish ~5.5–6.4, Neutral/Mixed ~4.5–5.5, "
            "Mildly Bearish ~3.5–4.4, Bearish ~0–3.4. "
            "Only the 0–10 bounds are enforced."
        ),
    )
    confidence: Literal["low", "medium", "high"] = Field(
        description=(
            "Confidence in the assessment based on data quality and sample size. "
            "Use 'low' when one or more sources returned a placeholder or fewer "
            "than 5 data points; 'medium' when data is present but sparse; "
            "'high' when all three sources returned substantive data."
        ),
    )
    narrative: str = Field(
        description=(
            "Full sentiment report covering, in order: "
            "(1) source-by-source breakdown with specific evidence (cite message "
            "counts, ratios, notable posts); "
            "(2) cross-source divergences and alignments; "
            "(3) dominant narrative themes; "
            "(4) catalysts and risks surfaced by the data; "
            "(5) a markdown table summarising key sentiment signals, their "
            "direction, source, and supporting evidence."
        ),
    )


def render_sentiment_report(report: SentimentReport) -> str:
    """Render a SentimentReport to the markdown shape the rest of the system expects.

    The structured header (band + score + confidence) is prepended to the
    narrative so the saved report is both human-readable and machine-parseable
    without regex.
    """
    return "\n".join([
        f"**Overall Sentiment:** **{report.overall_band.value}** "
        f"(Score: {report.overall_score:.1f}/10)",
        f"**Confidence:** {report.confidence.capitalize()}",
        "",
        report.narrative,
    ])


# ---------------------------------------------------------------------------
# Phase 2 — Trade Signal engine
# ---------------------------------------------------------------------------


class EntryLevel(BaseModel):
    """A single entry point within a trade setup."""

    label: str = Field(
        description=(
            "Entry label, e.g. 'EP1', 'EP2'. Use sequential numbering "
            "where EP1 is the first (most conservative) entry."
        ),
    )
    price: float = Field(
        description=(
            "Entry price in the instrument's quote currency (₹ for Indian tickers). "
            "Must come from the Market Analyst's LEVELS table or a structural level "
            "visible in the chart data. Do not invent round numbers."
        ),
    )
    allocation_pct: float = Field(
        description=(
            "Percentage of the total position capital to deploy at this entry. "
            "All entry allocations must sum to 100."
        ),
    )
    trigger: str = Field(
        description=(
            "Order trigger description, e.g. 'limit at support retest 2,845–2,860' "
            "or 'stop-buy above 2,910 breakout close'. Be specific."
        ),
    )
    rationale: str = Field(
        description="One sentence explaining why this level is a valid entry point.",
    )


class TakeProfit(BaseModel):
    """A single take-profit target within a trade setup."""

    label: str = Field(
        description="TP label, e.g. 'TP1', 'TP2'. TP1 is always the nearest target.",
    )
    price: float = Field(
        description=(
            "Target price in the instrument's quote currency. "
            "Must reference a structural level (prior swing high, supply zone, "
            "key Fibonacci) or an R-multiple from the Market Analyst's LEVELS table."
        ),
    )
    exit_pct: float = Field(
        description=(
            "Percentage of the open position to close at this target. "
            "All TP exit percentages must sum to 100."
        ),
    )
    basis: str = Field(
        description=(
            "Structural basis for this target, e.g. 'prior swing high 3,050' "
            "or '1.5R from avg entry' or 'weekly supply zone 3,080–3,100'."
        ),
    )


class TradeSignal(BaseModel):
    """Structured swing-trade signal produced by the Trader.

    The LLM chooses the setup; deterministic code (position_sizing.py) computes
    all quantity and risk figures. All prices must trace to the Market Analyst's
    LEVELS table or the verified market snapshot — never invented round numbers.

    For NO_TRADE signals, set direction='NO_TRADE', populate setup_type and
    invalidation with the reason, and leave entries/stop_loss/take_profits empty.
    """

    direction: Literal["LONG", "SHORT", "NO_TRADE"] = Field(
        description=(
            "Trade direction. Use NO_TRADE when no setup meets the RR floor "
            "using real structural levels — NO_TRADE is a valid outcome, not a failure."
        ),
    )
    setup_type: str = Field(
        description=(
            "Short label for the pattern, e.g. 'breakout-retest', "
            "'pullback-to-50SMA', 'range-reversal', 'momentum-continuation'."
        ),
    )
    timeframe: str = Field(
        description="Expected holding period, e.g. 'swing 5–20 sessions'.",
    )
    entries: list[EntryLevel] = Field(
        default_factory=list,
        description=(
            "1–3 entry levels for LONG/SHORT. Empty list for NO_TRADE. "
            "Allocation percentages across all entries must sum to 100."
        ),
    )
    stop_loss: Optional[float] = Field(
        default=None,
        description=(
            "Stop-loss price. For LONG: must be below a structural level (swing low, "
            "key support). For SHORT: must be above a structural level. "
            "Never use a round number without a structural basis. "
            "Must be null for NO_TRADE."
        ),
    )
    stop_basis: str = Field(
        default="",
        description=(
            "Structural basis for the stop, e.g. "
            "'below swing low 2,790 minus 0.5×ATR(14)=22'. "
            "Empty string for NO_TRADE."
        ),
    )
    take_profits: list[TakeProfit] = Field(
        default_factory=list,
        description=(
            "1–3 take-profit targets for LONG/SHORT. Empty list for NO_TRADE. "
            "Exit percentages across all TPs must sum to 100."
        ),
    )
    invalidation: str = Field(
        description=(
            "Condition that voids this idea BEFORE any entry fills, "
            "e.g. 'daily close below 2,750 before EP1 fills'. "
            "For NO_TRADE: state the reason no valid setup was found."
        ),
    )
    event_risks: list[str] = Field(
        default_factory=list,
        description=(
            "Material events that fall inside the expected holding window, "
            "e.g. 'Q1 results on 2026-07-18', 'F&O expiry 2026-06-26'. "
            "Source from the instrument context and news reports."
        ),
    )
    risk_reward_min: float = Field(
        default=0.0,
        description=(
            "Your estimate of the minimum risk-reward ratio for this setup "
            "based on TP1 vs avg entry vs stop. The Signal Validator will "
            "recompute this deterministically net of transaction costs."
        ),
    )
    confidence: Literal["high", "medium", "low"] = Field(
        description=(
            "Your confidence in this setup based on signal alignment, "
            "data quality, and market conditions."
        ),
    )

    @model_validator(mode="after")
    def check_no_trade_fields(self) -> "TradeSignal":
        if self.direction == "NO_TRADE":
            return self
        if not self.entries:
            raise ValueError("entries must not be empty for LONG/SHORT signals")
        if self.stop_loss is None:
            raise ValueError("stop_loss must be set for LONG/SHORT signals")
        if not self.take_profits:
            raise ValueError("take_profits must not be empty for LONG/SHORT signals")
        return self


def render_trade_signal(signal: TradeSignal) -> str:
    """Render a TradeSignal to human-readable markdown.

    Keeps backward compatibility: the trailing ``FINAL TRANSACTION PROPOSAL``
    line is preserved so existing parsers that grep for it keep working.
    """
    if signal.direction == "NO_TRADE":
        lines = [
            "**Direction**: NO_TRADE",
            "",
            f"**Setup Type**: {signal.setup_type}",
            f"**Timeframe**: {signal.timeframe}",
            "",
            f"**Reason / Invalidation**: {signal.invalidation}",
            f"**Confidence**: {signal.confidence}",
            "",
            "FINAL TRANSACTION PROPOSAL: **NO_TRADE**",
        ]
        if signal.event_risks:
            lines.insert(-1, f"**Event Risks**: {'; '.join(signal.event_risks)}")
        return "\n".join(lines)

    entry_lines = []
    for e in signal.entries:
        entry_lines.append(
            f"  {e.label}: ₹{e.price:,.2f} ({e.allocation_pct:.0f}%) — {e.trigger}"
        )
    tp_lines = []
    for t in signal.take_profits:
        tp_lines.append(
            f"  {t.label}: ₹{t.price:,.2f} (exit {t.exit_pct:.0f}%) — {t.basis}"
        )

    lines = [
        f"**Direction**: {signal.direction}",
        f"**Setup**: {signal.setup_type} | **Timeframe**: {signal.timeframe}",
        f"**Confidence**: {signal.confidence}",
        "",
        "**Entries**:",
        *entry_lines,
        "",
        f"**Stop Loss**: ₹{signal.stop_loss:,.2f} — {signal.stop_basis}",
        "",
        "**Take Profits**:",
        *tp_lines,
        "",
        f"**Invalidation**: {signal.invalidation}",
        f"**Est. RR**: {signal.risk_reward_min:.2f}",
    ]
    if signal.event_risks:
        lines.append(f"**Event Risks**: {'; '.join(signal.event_risks)}")
    lines.extend([
        "",
        f"FINAL TRANSACTION PROPOSAL: **{signal.direction}**",
    ])
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Phase 4 — Position management action (manage_position mode)
# ---------------------------------------------------------------------------


class PositionAction(BaseModel):
    """Structured action produced by the Position Manager when reviewing an open position.

    Replaces TradeSignal in manage_position mode. The LLM decides the action;
    the Position Action Validator enforces the hard rules (SL can only tighten,
    broken thesis must force an exit action).
    """

    action: Literal["HOLD", "EXIT_FULL", "EXIT_PARTIAL", "RAISE_SL", "ADD", "TAKE_TP_EARLY"] = Field(
        description=(
            "Management action to take on the open position. "
            "HOLD: no change; EXIT_FULL: close the entire position; "
            "EXIT_PARTIAL: close a portion (set exit_pct); "
            "RAISE_SL: tighten the stop-loss (set new_stop_loss — only allowed to move in "
            "the profit direction, never widen); "
            "ADD: add to the position at a new entry (set add_entry — re-validated vs caps); "
            "TAKE_TP_EARLY: take a take-profit target before price reaches it."
        ),
    )
    exit_pct: Optional[float] = Field(
        default=None,
        description=(
            "Percentage of the remaining position to close. "
            "Required for EXIT_PARTIAL and TAKE_TP_EARLY. Null for all other actions."
        ),
    )
    new_stop_loss: Optional[float] = Field(
        default=None,
        description=(
            "New stop-loss price for RAISE_SL action. "
            "For a LONG position this must be HIGHER than the current stop-loss (tightening). "
            "For a SHORT position this must be LOWER than the current stop-loss. "
            "Never widen the stop-loss — that is the #1 discretionary-trader failure. "
            "Null for all other actions."
        ),
    )
    add_entry: Optional[EntryLevel] = Field(
        default=None,
        description=(
            "New entry level for an ADD action. "
            "Will be re-validated against position sizing caps (total risk must remain within limits). "
            "Null for all other actions."
        ),
    )
    thesis_status: Literal["intact", "weakened", "broken"] = Field(
        description=(
            "Current status of the original trade thesis based on the latest data. "
            "'intact': thesis is playing out as expected; "
            "'weakened': thesis has some cracks but is not invalidated; "
            "'broken': thesis is invalidated — must use EXIT_FULL, EXIT_PARTIAL, or TAKE_TP_EARLY, "
            "never HOLD."
        ),
    )
    reasoning: str = Field(
        description=(
            "2-4 sentences explaining the action decision. Cite specific evidence: "
            "current P&L in R-multiples, days held, key level tests, news catalysts, "
            "and which part of the original thesis held or failed."
        ),
    )

    @model_validator(mode="after")
    def check_action_fields(self) -> "PositionAction":
        if self.thesis_status == "broken" and self.action == "HOLD":
            raise ValueError(
                "thesis_status='broken' requires an exit action (EXIT_FULL, EXIT_PARTIAL, or "
                "TAKE_TP_EARLY). HOLD is not permitted when the thesis is broken."
            )
        if self.action in ("EXIT_PARTIAL", "TAKE_TP_EARLY") and self.exit_pct is None:
            raise ValueError(f"exit_pct is required for action={self.action}")
        if self.action == "RAISE_SL" and self.new_stop_loss is None:
            raise ValueError("new_stop_loss is required for action=RAISE_SL")
        if self.action == "ADD" and self.add_entry is None:
            raise ValueError("add_entry is required for action=ADD")
        return self


def render_position_action(action: PositionAction) -> str:
    """Render a PositionAction to markdown for display and the memory log."""
    thesis_emoji = {"intact": "✓", "weakened": "~", "broken": "✗"}.get(action.thesis_status, "?")
    lines = [
        f"**Action**: {action.action}",
        f"**Thesis Status**: {thesis_emoji} {action.thesis_status}",
        "",
        f"**Reasoning**: {action.reasoning}",
    ]
    if action.exit_pct is not None:
        lines.extend(["", f"**Exit %**: {action.exit_pct:.0f}% of remaining position"])
    if action.new_stop_loss is not None:
        lines.extend(["", f"**New Stop Loss**: ₹{action.new_stop_loss:,.2f}"])
    if action.add_entry is not None:
        e = action.add_entry
        lines.extend([
            "",
            f"**Add Entry**: {e.label} ₹{e.price:,.2f} ({e.allocation_pct:.0f}%) — {e.trigger}",
        ])
    lines.extend([
        "",
        f"FINAL POSITION ACTION: **{action.action}**",
    ])
    return "\n".join(lines)
