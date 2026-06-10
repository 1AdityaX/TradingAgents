from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from tradingagents.agents.utils.agent_utils import (
    get_instrument_context_from_state,
    get_balance_sheet,
    get_cashflow,
    get_fundamentals,
    get_income_statement,
    get_insider_transactions,
    get_language_instruction,
)
from tradingagents.dataflows.config import get_config


def create_fundamentals_analyst(llm):
    def fundamentals_analyst_node(state):
        current_date = state["trade_date"]
        instrument_context = get_instrument_context_from_state(state)

        tools = [
            get_fundamentals,
            get_balance_sheet,
            get_cashflow,
            get_income_statement,
        ]

        is_indian = (
            str(state.get("company_of_interest", "")).upper().endswith(".NS")
            or str(state.get("company_of_interest", "")).upper().endswith(".BO")
        )
        india_checklist = (
            """

**INDIA-SPECIFIC CHECKLIST** (mandatory for .NS / .BO tickers — work through each item):
1. **Promoter pledge %**: Report current promoter pledge as a % of promoter holding. A rising pledge trend is a red flag — note the direction (rising / stable / falling / data unavailable).
2. **FII/DII holding change**: Report FII holding % and DII holding % for the last two available quarters. State the quarter-over-quarter direction for each (increasing / decreasing / flat / data unavailable).
3. **Upcoming results / board meeting date**: State the next scheduled results announcement or board meeting date. If this date falls inside a typical 5–20 session swing-trade window, flag it explicitly as **EVENT RISK**.
4. **Related-party / auditor red flags**: Note any related-party transaction concerns, auditor qualifications, or going-concern remarks visible in the available data. If none, state "none identified."
5. **Dividend / bonus / split events**: Note any upcoming or recent corporate actions that could affect price or liquidity.

**DATA AVAILABILITY RULE**: If any of the above data points are not present in the tool output, write exactly **"data unavailable"** for that item. Do NOT estimate, infer from industry averages, or leave a blank."""
            if is_indian else ""
        )
        system_message = (
            "You are a researcher tasked with analyzing fundamental information over the past week about a company. Please write a comprehensive report of the company's fundamental information such as financial documents, company profile, basic company financials, and company financial history to gain a full view of the company's fundamental information to inform traders. Make sure to include as much detail as possible. Provide specific, actionable insights with supporting evidence to help traders make informed decisions."
            + " Make sure to append a Markdown table at the end of the report to organize key points in the report, organized and easy to read."
            + " Use the available tools: `get_fundamentals` for comprehensive company analysis, `get_balance_sheet`, `get_cashflow`, and `get_income_statement` for specific financial statements."
            + india_checklist
            + get_language_instruction(),
        )

        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You are a helpful AI assistant, collaborating with other assistants."
                    " Use the provided tools to progress towards answering the question."
                    " If you are unable to fully answer, that's OK; another assistant with different tools"
                    " will help where you left off. Execute what you can to make progress."
                    " If you or any other assistant has the FINAL TRANSACTION PROPOSAL: **BUY/HOLD/SELL** or deliverable,"
                    " prefix your response with FINAL TRANSACTION PROPOSAL: **BUY/HOLD/SELL** so the team knows to stop."
                    " You have access to the following tools: {tool_names}.\n{system_message}"
                    "For your reference, the current date is {current_date}. {instrument_context}",
                ),
                MessagesPlaceholder(variable_name="messages"),
            ]
        )

        prompt = prompt.partial(system_message=system_message)
        prompt = prompt.partial(tool_names=", ".join([tool.name for tool in tools]))
        prompt = prompt.partial(current_date=current_date)
        prompt = prompt.partial(instrument_context=instrument_context)

        chain = prompt | llm.bind_tools(tools)

        result = chain.invoke(state["messages"])

        report = ""

        if len(result.tool_calls) == 0:
            report = result.content

        return {
            "messages": [result],
            "fundamentals_report": report,
        }

    return fundamentals_analyst_node
