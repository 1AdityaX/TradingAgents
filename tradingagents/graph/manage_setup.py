"""Graph setup for manage_position mode (Phase 4).

Topology mirrors the new-trade pipeline but with two key changes:

  1. The Position Manager node (produces PositionAction) replaces the Trader.
  2. The Position Action Validator replaces the Signal Validator.
  3. Bull/Bear debate prompts are contextualised for position management.

The same analyst nodes (Market, News, optionally Sentiment/Fundamentals) run
first so the debate has fresh price levels and news. The risk debaters and
Portfolio Manager then review the proposed action.

The `open_position` block and `position_context_block` in AgentState make the
position details visible to every agent without modifying their base prompts —
agents that call get_instrument_context_from_state() already render it.
"""

from __future__ import annotations

from typing import Any, Dict

from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode

from tradingagents.agents import *
from tradingagents.agents.managers.position_manager import create_position_manager
from tradingagents.agents.utils.agent_states import AgentState
from tradingagents.agents.validator.signal_validator import create_position_action_validator

from .analyst_execution import build_analyst_execution_plan
from .conditional_logic import ConditionalLogic


class ManageGraphSetup:
    """Sets up the LangGraph workflow for manage_position mode."""

    def __init__(
        self,
        quick_thinking_llm: Any,
        deep_thinking_llm: Any,
        tool_nodes: Dict[str, ToolNode],
        conditional_logic: ConditionalLogic,
        analyst_concurrency_limit: int = 1,
    ):
        self.quick_thinking_llm = quick_thinking_llm
        self.deep_thinking_llm = deep_thinking_llm
        self.tool_nodes = tool_nodes
        self.conditional_logic = conditional_logic
        self.analyst_concurrency_limit = analyst_concurrency_limit

    def setup_graph(
        self,
        selected_analysts=("market", "news"),
    ):
        """Build the manage_position graph.

        Default analysts for review mode: market (fresh levels) + news (catalysts).
        Fundamentals and sentiment are optional — callers can pass them in.
        """
        plan = build_analyst_execution_plan(
            list(selected_analysts),
            concurrency_limit=self.analyst_concurrency_limit,
        )

        analyst_factories = {
            "market": lambda: create_market_analyst(self.quick_thinking_llm),
            "social": lambda: create_sentiment_analyst(self.quick_thinking_llm),
            "news": lambda: create_news_analyst(self.quick_thinking_llm),
            "fundamentals": lambda: create_fundamentals_analyst(self.quick_thinking_llm),
        }

        bull_researcher_node = create_bull_researcher(self.quick_thinking_llm)
        bear_researcher_node = create_bear_researcher(self.quick_thinking_llm)
        research_manager_node = create_research_manager(self.deep_thinking_llm)
        position_manager_node = create_position_manager(self.quick_thinking_llm)
        position_action_validator_node = create_position_action_validator()

        aggressive_analyst = create_aggressive_debator(self.quick_thinking_llm)
        neutral_analyst = create_neutral_debator(self.quick_thinking_llm)
        conservative_analyst = create_conservative_debator(self.quick_thinking_llm)
        portfolio_manager_node = create_portfolio_manager(self.deep_thinking_llm)

        workflow = StateGraph(AgentState)

        # Analyst nodes
        for spec in plan.specs:
            workflow.add_node(spec.agent_node, analyst_factories[spec.key]())
            workflow.add_node(spec.clear_node, create_msg_delete())
            workflow.add_node(spec.tool_node, self.tool_nodes[spec.key])

        # Core nodes
        workflow.add_node("Bull Researcher", bull_researcher_node)
        workflow.add_node("Bear Researcher", bear_researcher_node)
        workflow.add_node("Research Manager", research_manager_node)
        workflow.add_node("Position Manager", position_manager_node)
        workflow.add_node("Position Action Validator", position_action_validator_node)
        workflow.add_node("Aggressive Analyst", aggressive_analyst)
        workflow.add_node("Neutral Analyst", neutral_analyst)
        workflow.add_node("Conservative Analyst", conservative_analyst)
        workflow.add_node("Portfolio Manager", portfolio_manager_node)

        # Analyst chain
        workflow.add_edge(START, plan.specs[0].agent_node)
        for i, spec in enumerate(plan.specs):
            workflow.add_conditional_edges(
                spec.agent_node,
                getattr(self.conditional_logic, f"should_continue_{spec.key}"),
                [spec.tool_node, spec.clear_node],
            )
            workflow.add_edge(spec.tool_node, spec.agent_node)
            if i < len(plan.specs) - 1:
                workflow.add_edge(spec.clear_node, plan.specs[i + 1].agent_node)
            else:
                workflow.add_edge(spec.clear_node, "Bull Researcher")

        # Debate
        workflow.add_conditional_edges(
            "Bull Researcher",
            self.conditional_logic.should_continue_debate,
            {
                "Bear Researcher": "Bear Researcher",
                "Research Manager": "Research Manager",
            },
        )
        workflow.add_conditional_edges(
            "Bear Researcher",
            self.conditional_logic.should_continue_debate,
            {
                "Bull Researcher": "Bull Researcher",
                "Research Manager": "Research Manager",
            },
        )

        workflow.add_edge("Research Manager", "Position Manager")
        workflow.add_edge("Position Manager", "Position Action Validator")

        # After validation always proceed (no retry in manage mode — one review pass)
        workflow.add_edge("Position Action Validator", "Aggressive Analyst")

        # Risk debate
        workflow.add_conditional_edges(
            "Aggressive Analyst",
            self.conditional_logic.should_continue_risk_analysis,
            {
                "Conservative Analyst": "Conservative Analyst",
                "Portfolio Manager": "Portfolio Manager",
            },
        )
        workflow.add_conditional_edges(
            "Conservative Analyst",
            self.conditional_logic.should_continue_risk_analysis,
            {
                "Neutral Analyst": "Neutral Analyst",
                "Portfolio Manager": "Portfolio Manager",
            },
        )
        workflow.add_conditional_edges(
            "Neutral Analyst",
            self.conditional_logic.should_continue_risk_analysis,
            {
                "Aggressive Analyst": "Aggressive Analyst",
                "Portfolio Manager": "Portfolio Manager",
            },
        )

        workflow.add_edge("Portfolio Manager", END)

        return workflow
