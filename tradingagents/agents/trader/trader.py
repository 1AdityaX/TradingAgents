"""Trader: turns the Research Manager's investment plan into a concrete TradeSignal.

Phase 2 upgrade: the Trader now produces a structured TradeSignal (direction,
1–3 entry levels with prices and allocations, stop-loss, 1–3 take-profit
targets, invalidation condition, event risks) instead of the old open-ended
TraderProposal.

The Signal Validator (a pure-code node downstream) recomputes risk-reward net
of transaction costs and runs position sizing. The Trader's job is only to
choose the setup and name real structural price levels — arithmetic happens
in code, not in the LLM's head.

When the validator rejects a signal and retries, the Trader's prompt includes
the rejection reason so it can correct the specific failure.
"""

from __future__ import annotations

import functools
import logging

from langchain_core.messages import AIMessage

from tradingagents.agents.schemas import TradeSignal, render_trade_signal
from tradingagents.agents.utils.agent_utils import (
    get_instrument_context_from_state,
    get_language_instruction,
)
from tradingagents.agents.utils.structured import bind_structured

logger = logging.getLogger(__name__)


_SYSTEM_PROMPT = """\
You are a swing-trade structuring agent. Your job is to read the Research \
Manager's investment plan and the analyst reports, then produce a precise \
TradeSignal: direction, 1-3 entry levels, stop-loss, 1-3 take-profit targets, \
and event risks for the expected holding window.

HARD RULES - violating any rule means you MUST output NO_TRADE instead:
1. All entry, stop-loss, and take-profit prices MUST come from the Market \
Analyst's LEVELS table (the table of support/resistance/swing levels with \
dates). You may adjust by a stated fraction of ATR(14) from that table. \
Do NOT invent round numbers.
2. Stop-loss for a LONG must be BELOW a structural level (swing low, key \
support). Stop-loss for a SHORT must be ABOVE a structural level. A round \
number alone is not a structural level.
3. Risk-reward (TP1 vs avg entry vs SL) must be at least {min_rr:.1f} using \
real levels. If you cannot build a valid setup with this RR floor, output \
NO_TRADE - that is a correct outcome, not a failure.
4. Entry allocation percentages must sum to 100. TP exit percentages must \
sum to 100.
5. For Indian tickers: prices in rupees (no dollar signs). Do not reference \
US market hours or assume after-hours trading. Check the instrument context \
for F&O ban and circuit bands.

---
WORKED EXAMPLES

## GOOD EXAMPLE — valid signal derived from LEVELS table

Market Analyst LEVELS table (excerpt):
| Type       | Price     | Date(s) Established | Strength | Notes              |
|------------|-----------|---------------------|----------|--------------------|
| Support    | ₹2,847    | 2026-05-22          | 3 touches | prior swing low    |
| Support    | ₹2,810    | 2026-04-18          | 2 touches | deeper demand zone |
| Resistance | ₹2,960    | 2026-05-08          | 2 touches | prior swing high   |
| Resistance | ₹3,080    | 2026-03-31          | 1 touch   | weekly supply      |
| ATR(14)    | ₹22       | 2026-06-09          | —        | 14-period ATR      |

Correct TradeSignal output:
- direction: LONG
- setup_type: pullback-to-support in uptrend
- entries: [EP1 ₹2,852 (60%) "limit at support retest 2,847 + 5pt buffer",
            EP2 ₹2,815 (40%) "deeper pullback to support 2,810"]
- stop_loss: ₹2,778  stop_basis: "below swing low 2,847 − 0.5×ATR(22) = 2,836; hard stop below 2,810"
- take_profits: [TP1 ₹2,960 (50%) "prior swing high", TP2 ₹3,080 (50%) "weekly supply"]
→ Why correct: every price traces to a row in the LEVELS table.
  Avg entry ≈ ₹2,837. Risk = 2,837 − 2,778 = ₹59. Reward to TP1 = 2,960 − 2,837 = ₹123. RR ≈ 2.1 ✓

## BAD EXAMPLE — invented levels, wrong SL direction, failed RR

LEVELS table has NO round-number entries; closest support is ₹2,847.

Incorrect TradeSignal output (DO NOT DO THIS):
- direction: LONG
- entries: [EP1 ₹2,900 (100%) "breakout above 2,900"] ← ₹2,900 is not in LEVELS table
- stop_loss: ₹2,900  ← SL equals entry price (must be BELOW for LONG)
- take_profits: [TP1 ₹3,000 (100%)] ← ₹3,000 is a round number not in LEVELS
→ Why wrong: entry invented, SL does not protect, round numbers fabricated.
  Rule 1 violated (not from LEVELS). Rule 2 violated (SL = entry). Output must be NO_TRADE.
---
"""

_USER_PROMPT = """\
{instrument_context}

Research Manager's investment plan:
{investment_plan}

Market Analyst's report (LEVELS table is in here - use ONLY these prices):
{market_report}

Fundamentals report:
{fundamentals_report}

News / macro report:
{news_report}

Sentiment report:
{sentiment_report}

Config: minimum risk-reward floor = {min_rr:.1f} (gross, before transaction costs).

Now produce the TradeSignal for {company_name}.{retry_section}{language_instruction}
"""

_RETRY_SECTION = """

WARNING - VALIDATOR REJECTION (retry #{retry_count}):
Your previous TradeSignal was rejected for this reason:
  {error}

Fix ONLY the issue above. If you cannot produce a valid signal after fixing \
it, output NO_TRADE.
"""


def create_trader(llm):
    from tradingagents.dataflows.config import get_config

    structured_llm = bind_structured(llm, TradeSignal, "Trader")

    def trader_node(state, name):
        cfg = get_config()
        min_rr = cfg.get("min_risk_reward", 1.8)

        company_name = state["company_of_interest"]
        instrument_context = get_instrument_context_from_state(state)
        investment_plan = state.get("investment_plan", "")
        market_report = state.get("market_report", "")
        fundamentals_report = state.get("fundamentals_report", "")
        news_report = state.get("news_report", "")
        sentiment_report = state.get("sentiment_report", "")

        retry_count = state.get("trader_retry_count", 0)
        validation_result = state.get("signal_validation_result") or {}

        retry_section = ""
        if retry_count > 0 and validation_result.get("error"):
            retry_section = _RETRY_SECTION.format(
                retry_count=retry_count,
                error=validation_result["error"],
            )

        messages = [
            {
                "role": "system",
                "content": _SYSTEM_PROMPT.format(min_rr=min_rr),
            },
            {
                "role": "user",
                "content": _USER_PROMPT.format(
                    instrument_context=instrument_context,
                    investment_plan=investment_plan,
                    market_report=market_report,
                    fundamentals_report=fundamentals_report,
                    news_report=news_report,
                    sentiment_report=sentiment_report,
                    min_rr=min_rr,
                    company_name=company_name,
                    retry_section=retry_section,
                    language_instruction=get_language_instruction(),
                ),
            },
        ]

        trade_signal_dict: dict | None = None
        trader_plan: str

        if structured_llm is not None:
            try:
                signal_obj = structured_llm.invoke(messages)
                if isinstance(signal_obj, TradeSignal):
                    trader_plan = render_trade_signal(signal_obj)
                    trade_signal_dict = signal_obj.model_dump()
                else:
                    # Fallback: provider returned something unexpected
                    trader_plan = str(signal_obj)
            except Exception as exc:
                logger.warning(
                    "Trader: structured-output invocation failed (%s); "
                    "retrying as free text",
                    exc,
                )
                response = llm.invoke(messages)
                trader_plan = response.content
        else:
            response = llm.invoke(messages)
            trader_plan = response.content

        return {
            "messages": [AIMessage(content=trader_plan)],
            "trader_investment_plan": trader_plan,
            "trade_signal": trade_signal_dict,
            "sender": name,
        }

    return functools.partial(trader_node, name="Trader")
