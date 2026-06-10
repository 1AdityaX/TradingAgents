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
