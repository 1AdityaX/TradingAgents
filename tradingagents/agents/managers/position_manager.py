"""Position Manager: reviews an open position and emits a PositionAction.

Used only in `manage_position` mode (Phase 4). Replaces the Trader node.

The Position Manager reads:
  - The open position context block (entry, P&L, days held, SL/TPs, thesis)
  - Fresh market analyst levels (to judge if structure has changed)
  - News report (recent catalysts that may affect the thesis)
  - The bull/bear debate (reframed as "thesis intact vs degraded")
  - The Research Manager's synthesised view

Hard rules enforced by the Position Action Validator downstream:
  - SL can only tighten (RAISE_SL must move SL in the profit direction)
  - thesis_status='broken' forbids HOLD
  - ADD action re-validates position sizing caps
"""

from __future__ import annotations

import functools
import logging

from langchain_core.messages import AIMessage

from tradingagents.agents.schemas import PositionAction, render_position_action
from tradingagents.agents.utils.agent_utils import (
    get_instrument_context_from_state,
    get_language_instruction,
)
from tradingagents.agents.utils.structured import bind_structured

logger = logging.getLogger(__name__)


_SYSTEM_PROMPT = """\
You are a position management agent reviewing an OPEN swing-trade position.
Your job is to assess whether the original trade thesis is intact and decide the
appropriate management action.

HARD RULES:
1. thesis_status='broken' FORBIDS 'HOLD'. Use EXIT_FULL, EXIT_PARTIAL, or TAKE_TP_EARLY.
2. RAISE_SL must ONLY tighten the stop — moving SL in the PROFIT direction.
   For LONG: new_stop_loss > current stop_loss.
   For SHORT: new_stop_loss < current stop_loss.
   NEVER widen the stop-loss — that is the #1 discretionary-trader failure.
3. ADD action must reference a structural level from the Market Analyst's LEVELS table.
   The Position Action Validator will re-check that total risk stays within caps.
4. Base your assessment only on the data provided. Do not use memorised knowledge
   about the company — it is stale. Cite evidence from the reports.
5. The P&L in R-multiples is the primary risk metric: if position is at -1R or worse,
   thesis_status must be at least 'weakened'.
"""

_USER_PROMPT = """\
{instrument_context}

{position_context_block}

Research Manager's view (from bull/bear debate):
{investment_plan}

Market Analyst's fresh report (LEVELS table — use only these prices for RAISE_SL or ADD):
{market_report}

News & macro report (recent catalysts):
{news_report}

Sentiment report:
{sentiment_report}

Now decide the PositionAction for this open position.{language_instruction}
"""


def create_position_manager(llm):
    """Return a LangGraph node that reviews an open position and emits PositionAction."""
    structured_llm = bind_structured(llm, PositionAction, "Position Manager")

    def position_manager_node(state, name):
        instrument_context = get_instrument_context_from_state(state)
        position_context_block = state.get("position_context_block", "")
        investment_plan = state.get("investment_plan", "")
        market_report = state.get("market_report", "")
        news_report = state.get("news_report", "")
        sentiment_report = state.get("sentiment_report", "")

        messages = [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {
                "role": "user",
                "content": _USER_PROMPT.format(
                    instrument_context=instrument_context,
                    position_context_block=position_context_block,
                    investment_plan=investment_plan,
                    market_report=market_report,
                    news_report=news_report,
                    sentiment_report=sentiment_report,
                    language_instruction=get_language_instruction(),
                ),
            },
        ]

        position_action_dict: dict | None = None
        action_text: str

        if structured_llm is not None:
            try:
                action_obj = structured_llm.invoke(messages)
                if isinstance(action_obj, PositionAction):
                    action_text = render_position_action(action_obj)
                    position_action_dict = action_obj.model_dump()
                else:
                    action_text = str(action_obj)
            except Exception as exc:
                logger.warning(
                    "Position Manager: structured-output invocation failed (%s); "
                    "falling back to free text",
                    exc,
                )
                response = llm.invoke(messages)
                action_text = response.content
        else:
            response = llm.invoke(messages)
            action_text = response.content

        return {
            "messages": [AIMessage(content=action_text)],
            "trader_investment_plan": action_text,
            "position_action": position_action_dict,
            "sender": name,
        }

    return functools.partial(position_manager_node, name="Position Manager")
