from tradingagents.agents.utils.agent_utils import (
    get_instrument_context_from_state,
    get_language_instruction,
)


def create_aggressive_debator(llm):
    def aggressive_node(state) -> dict:
        risk_debate_state = state["risk_debate_state"]
        history = risk_debate_state.get("history", "")
        aggressive_history = risk_debate_state.get("aggressive_history", "")

        current_conservative_response = risk_debate_state.get("current_conservative_response", "")
        current_neutral_response = risk_debate_state.get("current_neutral_response", "")

        market_research_report = state["market_report"]
        sentiment_report = state["sentiment_report"]
        news_report = state["news_report"]
        fundamentals_report = state["fundamentals_report"]
        instrument_context = get_instrument_context_from_state(state)

        trader_decision = state["trader_investment_plan"]

        prompt = f"""You are the Aggressive Risk Analyst. Your specific job in this one-round debate is to answer ONE question: **Is the proposed position size leaving expectancy on the table?**

Work through this checklist in order and state your finding for each point:

1. **Conviction vs. sizing**: Given the setup quality (trend strength, R:R ratio, number of confirming factors), does the proposed position size reflect appropriate conviction? If the signal is high-confidence (multiple analysts aligned, clean structure, low event risk), a conservative size under-deploys capital.
2. **Expected value**: Estimate the expected value of this trade. EV = (win_probability × avg_win_R) − (loss_probability × 1R). Even a modest win-rate makes the trade worthwhile if RR > 2. State your estimated win probability and why.
3. **Opportunity cost**: What is lost by sizing down or passing? If market regime favours this setup type (momentum / pullback / breakout) and there are limited concurrent opportunities, opportunity cost is real.
4. **Bull thesis integrity**: Is the bull thesis still intact based on the data? State the single strongest piece of evidence supporting entry now.
5. **Rebuttal** (if prior conservative/neutral arguments exist): Address any specific concern from the other analysts that would justify reducing size or passing — accept valid points but rebut any that overstate risk without evidence.

Here is the trader's decision:
{trader_decision}

{instrument_context}
Market Research Report: {market_research_report}
Sentiment Report: {sentiment_report}
News Report: {news_report}
Fundamentals Report: {fundamentals_report}
Debate history so far: {history}
Conservative analyst's last argument: {current_conservative_response}
Neutral analyst's last argument: {current_neutral_response}

Be specific: cite actual numbers from the reports (prices, R:R, analyst scores). Avoid generic optimism. Output conversationally without special formatting.""" + get_language_instruction()

        response = llm.invoke(prompt)

        argument = f"Aggressive Analyst: {response.content}"

        new_risk_debate_state = {
            "history": history + "\n" + argument,
            "aggressive_history": aggressive_history + "\n" + argument,
            "conservative_history": risk_debate_state.get("conservative_history", ""),
            "neutral_history": risk_debate_state.get("neutral_history", ""),
            "latest_speaker": "Aggressive",
            "current_aggressive_response": argument,
            "current_conservative_response": risk_debate_state.get("current_conservative_response", ""),
            "current_neutral_response": risk_debate_state.get(
                "current_neutral_response", ""
            ),
            "count": risk_debate_state["count"] + 1,
        }

        return {"risk_debate_state": new_risk_debate_state}

    return aggressive_node
