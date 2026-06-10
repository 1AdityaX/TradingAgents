"""Portfolio Manager: synthesises the risk-analyst debate into the final decision.

Phase 2 upgrade: the PM now receives the validated TradeSignal and its
deterministic sizing ticket (from the Signal Validator). Its checklist is:
  1. Did the Signal Validator pass? (always yes at this stage — validator
     already forced NO_TRADE if not)
  2. Are the event risks acceptable within the expected holding window?
  3. Do the portfolio caps remain respected?
  4. Approve / modify (adjust confidence) / reject with the specific failed
     check named.

The executable signal ticket (entries, SL, TPs, qty, capital, risk, RR) is
appended to the final_trade_decision regardless of the PM's narrative rating.
"""

from __future__ import annotations

from tradingagents.agents.schemas import PortfolioDecision, render_pm_decision
from tradingagents.agents.utils.agent_utils import (
    get_instrument_context_from_state,
    get_language_instruction,
)
from tradingagents.agents.utils.structured import (
    bind_structured,
    invoke_structured_or_freetext,
)


def create_portfolio_manager(llm):
    structured_llm = bind_structured(llm, PortfolioDecision, "Portfolio Manager")

    def portfolio_manager_node(state) -> dict:
        instrument_context = get_instrument_context_from_state(state)

        history = state["risk_debate_state"]["history"]
        risk_debate_state = state["risk_debate_state"]
        research_plan = state["investment_plan"]
        trader_plan = state["trader_investment_plan"]

        past_context = state.get("past_context", "")
        lessons_line = (
            f"- Lessons from prior decisions and outcomes:\n{past_context}\n"
            if past_context
            else ""
        )

        # Pull signal ticket from the validator result
        validation_result = state.get("signal_validation_result") or {}
        signal_ticket = validation_result.get("ticket") or ""
        sizing = validation_result.get("sizing") or {}
        forced_no_trade = validation_result.get("forced_no_trade", False)

        signal_section = ""
        if signal_ticket:
            signal_section = (
                f"\n**Validated Signal Ticket** (from Signal Validator — do not modify prices):\n"
                f"```\n{signal_ticket}\n```\n"
            )
        if forced_no_trade:
            signal_section += (
                "\n**Note**: Signal Validator forced NO_TRADE after retry exhaustion. "
                "Recommend Hold/Underweight/Sell accordingly.\n"
            )

        # PM checklist prompt
        checklist = (
            "Decision checklist — work through in order, stop at the first failed check:\n"
            "1. Signal Validator passed? (Yes — validator already forced NO_TRADE if not. "
            "If forced NO_TRADE, your decision must be Hold/Underweight/Sell.)\n"
            "2. Are event risks listed in the signal ticket acceptable within the holding window? "
            "Name each event and state whether it is tolerable (e.g., 'results on day 12 — "
            "acceptable, will exit 50% before if TP1 unhit') or disqualifying.\n"
            "3. Do portfolio caps remain respected? State the current open risk % after adding "
            "this trade and confirm it is within the configured max_open_risk_pct. State the "
            "capital % this position would represent.\n"
            "4. Final decision: Approve → **Buy/Overweight** | Concern but proceed → "
            "**Hold/Overweight** | Failed a check → **Hold/Underweight/Sell**. "
            "You MUST name the specific check number that failed when downgrading. "
            "Do not use vague phrases like 'some risk exists' — name the exact check."
        )

        prompt = f"""As the Portfolio Manager, synthesize the risk analysts' debate and deliver the final trading decision.

{instrument_context}

---

**Rating Scale** (use exactly one):
- **Buy**: Strong conviction to enter or add to position
- **Overweight**: Favorable outlook, gradually increase exposure
- **Hold**: Maintain current position, no action needed
- **Underweight**: Reduce exposure, take partial profits
- **Sell**: Exit position or avoid entry

**Context:**
- Research Manager's investment plan: **{research_plan}**
- Trader's transaction proposal: **{trader_plan}**
{signal_section}{lessons_line}
{checklist}

**Risk Analysts Debate History:**
{history}

---

Be decisive and ground every conclusion in specific evidence from the analysts.{get_language_instruction()}"""

        final_trade_decision = invoke_structured_or_freetext(
            structured_llm,
            llm,
            prompt,
            render_pm_decision,
            "Portfolio Manager",
        )

        # Append the executable signal ticket to the decision so it persists
        # in the memory log and CLI output regardless of PM narrative.
        if signal_ticket:
            final_trade_decision = final_trade_decision + "\n\n---\n\n" + signal_ticket

        new_risk_debate_state = {
            "judge_decision": final_trade_decision,
            "history": risk_debate_state["history"],
            "aggressive_history": risk_debate_state["aggressive_history"],
            "conservative_history": risk_debate_state["conservative_history"],
            "neutral_history": risk_debate_state["neutral_history"],
            "latest_speaker": "Judge",
            "current_aggressive_response": risk_debate_state["current_aggressive_response"],
            "current_conservative_response": risk_debate_state["current_conservative_response"],
            "current_neutral_response": risk_debate_state["current_neutral_response"],
            "count": risk_debate_state["count"],
        }

        return {
            "risk_debate_state": new_risk_debate_state,
            "final_trade_decision": final_trade_decision,
        }

    return portfolio_manager_node
