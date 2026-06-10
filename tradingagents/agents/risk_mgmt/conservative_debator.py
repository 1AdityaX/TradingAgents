from tradingagents.agents.utils.agent_utils import (
    get_instrument_context_from_state,
    get_language_instruction,
)


def create_conservative_debator(llm):
    def conservative_node(state) -> dict:
        risk_debate_state = state["risk_debate_state"]
        history = risk_debate_state.get("history", "")
        conservative_history = risk_debate_state.get("conservative_history", "")

        current_aggressive_response = risk_debate_state.get("current_aggressive_response", "")
        current_neutral_response = risk_debate_state.get("current_neutral_response", "")

        market_research_report = state["market_report"]
        sentiment_report = state["sentiment_report"]
        news_report = state["news_report"]
        fundamentals_report = state["fundamentals_report"]
        instrument_context = get_instrument_context_from_state(state)

        trader_decision = state["trader_investment_plan"]

        prompt = f"""You are the Conservative Risk Analyst. Your specific job in this one-round debate is to stress-test three concrete risk categories for the proposed trade. Work through each point and state your finding explicitly.

**CHECKLIST — answer every item:**

1. **Event risk inside the holding window**: Check the fundamentals report and instrument context for scheduled results announcements, board meetings, F&O expiry dates, or RBI/SEBI policy dates that fall within the expected 5–20 session holding window. Name each event and its date. If any material event is inside the window, state whether it changes the risk profile (e.g., "Q1 results on 2026-07-18 fall on day 8 of the expected hold — earnings gap risk is unquantifiable").

2. **Gap risk through the stop-loss (Indian market specific)**: Indian stocks regularly gap 3–8% on results, index rebalances, or global macro shocks. Circuit bands (2/5/10/20%) can lock a position for multiple sessions, preventing exit at the stated SL. Assess: (a) Does the stock have a circuit band that could trap the position? (b) If the stock gaps through the SL on open, what is the realistic worst-case loss vs. the stated 1R? (c) Is the liquidity (daily traded value) large enough to absorb the stated position size at the SL price without significant slippage?

3. **Liquidity vs. position quantity**: Compare the proposed position size (₹ capital and share quantity) against the stock's typical daily traded value. A position that exceeds 1–2% of average daily volume is difficult to exit quickly. Flag if this is a concern.

4. **Bear thesis**: What is the single scenario that would most quickly invalidate the trade thesis? Name the price level, event, or condition that triggers it.

5. **Rebuttal** (if prior aggressive/neutral arguments exist): Accept any valid pro-entry point from other analysts, but specifically address any risk they understated.

Here is the trader's decision:
{trader_decision}

{instrument_context}
Market Research Report: {market_research_report}
Sentiment Report: {sentiment_report}
News Report: {news_report}
Fundamentals Report: {fundamentals_report}
Debate history so far: {history}
Aggressive analyst's last argument: {current_aggressive_response}
Neutral analyst's last argument: {current_neutral_response}

Be specific: cite actual dates, prices, and quantities from the reports. Avoid vague caution. Output conversationally without special formatting.""" + get_language_instruction()

        response = llm.invoke(prompt)

        argument = f"Conservative Analyst: {response.content}"

        new_risk_debate_state = {
            "history": history + "\n" + argument,
            "aggressive_history": risk_debate_state.get("aggressive_history", ""),
            "conservative_history": conservative_history + "\n" + argument,
            "neutral_history": risk_debate_state.get("neutral_history", ""),
            "latest_speaker": "Conservative",
            "current_aggressive_response": risk_debate_state.get(
                "current_aggressive_response", ""
            ),
            "current_conservative_response": argument,
            "current_neutral_response": risk_debate_state.get(
                "current_neutral_response", ""
            ),
            "count": risk_debate_state["count"] + 1,
        }

        return {"risk_debate_state": new_risk_debate_state}

    return conservative_node
