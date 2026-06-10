from tradingagents.agents.utils.agent_utils import (
    get_instrument_context_from_state,
    get_language_instruction,
)


def create_neutral_debator(llm):
    def neutral_node(state) -> dict:
        risk_debate_state = state["risk_debate_state"]
        history = risk_debate_state.get("history", "")
        neutral_history = risk_debate_state.get("neutral_history", "")

        current_aggressive_response = risk_debate_state.get("current_aggressive_response", "")
        current_conservative_response = risk_debate_state.get("current_conservative_response", "")

        market_research_report = state["market_report"]
        sentiment_report = state["sentiment_report"]
        news_report = state["news_report"]
        fundamentals_report = state["fundamentals_report"]
        instrument_context = get_instrument_context_from_state(state)

        trader_decision = state["trader_investment_plan"]

        prompt = f"""You are the Neutral Risk Analyst. Your specific job in this one-round debate is to build a **scenario tree** for the proposed trade and assign rough probabilities to each branch. This gives the Portfolio Manager a probability-weighted view rather than an optimism vs. pessimism binary.

**SCENARIO TREE — construct and fill in each branch:**

**Base case (most likely — state probability %):**
- What needs to happen for the trade to work as designed (entry fills, thesis holds, targets reached)?
- Key assumption: (e.g., "Nifty stays above 200-SMA, no negative news catalyst")
- Expected outcome: (e.g., "TP1 hit in ~8 sessions, partial exit; trail SL to breakeven")
- Estimated probability: X%

**Bull case (better than expected — state probability %):**
- What accelerates the move? (e.g., "Positive quarterly result, sector rotation inflow")
- Expected outcome: (e.g., "Both TP1 and TP2 hit within 12 sessions")
- Estimated probability: X%

**Bear case (trade fails — state probability %):**
- What triggers the stop-loss? (e.g., "Broader market correction, stock-specific bad news")
- Expected outcome: (e.g., "SL hit at ₹2,778, loss = 1R")
- Estimated probability: X%

**Tail risk (low probability, high impact — state probability %):**
- What is the worst realistic scenario? (e.g., "Gap down on results / circuit lock / systemic market shock")
- Expected outcome: (e.g., "Exit at open prices, loss could be 2–3R")
- Estimated probability: X%

**Synthesis**: Given these probabilities, does the expected value of the trade justify execution? State your conclusion clearly and note which argument from the aggressive or conservative analyst you agree with and which you reject.

Here is the trader's decision:
{trader_decision}

{instrument_context}
Market Research Report: {market_research_report}
Sentiment Report: {sentiment_report}
News Report: {news_report}
Fundamentals Report: {fundamentals_report}
Debate history so far: {history}
Aggressive analyst's last argument: {current_aggressive_response}
Conservative analyst's last argument: {current_conservative_response}

Probabilities must sum to 100%. Be specific: use actual prices and dates from the reports. Output conversationally without special formatting.""" + get_language_instruction()

        response = llm.invoke(prompt)

        argument = f"Neutral Analyst: {response.content}"

        new_risk_debate_state = {
            "history": history + "\n" + argument,
            "aggressive_history": risk_debate_state.get("aggressive_history", ""),
            "conservative_history": risk_debate_state.get("conservative_history", ""),
            "neutral_history": neutral_history + "\n" + argument,
            "latest_speaker": "Neutral",
            "current_aggressive_response": risk_debate_state.get(
                "current_aggressive_response", ""
            ),
            "current_conservative_response": risk_debate_state.get("current_conservative_response", ""),
            "current_neutral_response": argument,
            "count": risk_debate_state["count"] + 1,
        }

        return {"risk_debate_state": new_risk_debate_state}

    return neutral_node
