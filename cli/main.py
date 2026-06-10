from typing import Optional
import os
import datetime
import typer
import questionary
from pathlib import Path
from functools import wraps
from rich.console import Console
from rich.panel import Panel
from rich.spinner import Spinner
from rich.live import Live
from rich.columns import Columns
from rich.markdown import Markdown
from rich.layout import Layout
from rich.text import Text
from rich.table import Table
from collections import deque
import time
from rich.tree import Tree
from rich import box
from rich.align import Align
from rich.rule import Rule

from tradingagents.graph.trading_graph import TradingAgentsGraph
from tradingagents.graph.analyst_execution import (
    AnalystWallTimeTracker,
    build_analyst_execution_plan,
    get_initial_analyst_node,
    sync_analyst_tracker_from_chunk,
)
from tradingagents.default_config import DEFAULT_CONFIG
from cli.models import AnalystType
from cli.utils import *
from cli.announcements import fetch_announcements, display_announcements
from cli.stats_handler import StatsCallbackHandler

console = Console()

app = typer.Typer(
    name="TradingAgents",
    help="TradingAgents CLI: Multi-Agents LLM Financial Trading Framework",
    add_completion=True,  # Enable shell completion
)


# Create a deque to store recent messages with a maximum length
class MessageBuffer:
    # Fixed teams that always run (not user-selectable)
    FIXED_AGENTS = {
        "Research Team": ["Bull Researcher", "Bear Researcher", "Research Manager"],
        "Trading Team": ["Trader"],
        "Risk Management": ["Aggressive Analyst", "Neutral Analyst", "Conservative Analyst"],
        "Portfolio Management": ["Portfolio Manager"],
    }

    # Analyst name mapping
    ANALYST_MAPPING = {
        "market": "Market Analyst",
        "social": "Sentiment Analyst",
        "news": "News Analyst",
        "fundamentals": "Fundamentals Analyst",
    }

    # Report section mapping: section -> (analyst_key for filtering, finalizing_agent)
    # analyst_key: which analyst selection controls this section (None = always included)
    # finalizing_agent: which agent must be "completed" for this report to count as done
    REPORT_SECTIONS = {
        "market_report": ("market", "Market Analyst"),
        "sentiment_report": ("social", "Sentiment Analyst"),
        "news_report": ("news", "News Analyst"),
        "fundamentals_report": ("fundamentals", "Fundamentals Analyst"),
        "investment_plan": (None, "Research Manager"),
        "trader_investment_plan": (None, "Trader"),
        "final_trade_decision": (None, "Portfolio Manager"),
    }

    def __init__(self, max_length=100):
        self.messages = deque(maxlen=max_length)
        self.tool_calls = deque(maxlen=max_length)
        self.current_report = None
        self.final_report = None  # Store the complete final report
        self.agent_status = {}
        self.current_agent = None
        self.report_sections = {}
        self.selected_analysts = []
        self._processed_message_ids = set()

    def init_for_analysis(self, selected_analysts):
        """Initialize agent status and report sections based on selected analysts.

        Args:
            selected_analysts: List of analyst type strings (e.g., ["market", "news"])
        """
        self.selected_analysts = [a.lower() for a in selected_analysts]

        # Build agent_status dynamically
        self.agent_status = {}

        # Add selected analysts
        for analyst_key in self.selected_analysts:
            if analyst_key in self.ANALYST_MAPPING:
                self.agent_status[self.ANALYST_MAPPING[analyst_key]] = "pending"

        # Add fixed teams
        for team_agents in self.FIXED_AGENTS.values():
            for agent in team_agents:
                self.agent_status[agent] = "pending"

        # Build report_sections dynamically
        self.report_sections = {}
        for section, (analyst_key, _) in self.REPORT_SECTIONS.items():
            if analyst_key is None or analyst_key in self.selected_analysts:
                self.report_sections[section] = None

        # Reset other state
        self.current_report = None
        self.final_report = None
        self.current_agent = None
        self.messages.clear()
        self.tool_calls.clear()
        self._processed_message_ids.clear()

    def get_completed_reports_count(self):
        """Count reports that are finalized (their finalizing agent is completed).

        A report is considered complete when:
        1. The report section has content (not None), AND
        2. The agent responsible for finalizing that report has status "completed"

        This prevents interim updates (like debate rounds) from counting as completed.
        """
        count = 0
        for section in self.report_sections:
            if section not in self.REPORT_SECTIONS:
                continue
            _, finalizing_agent = self.REPORT_SECTIONS[section]
            # Report is complete if it has content AND its finalizing agent is done
            has_content = self.report_sections.get(section) is not None
            agent_done = self.agent_status.get(finalizing_agent) == "completed"
            if has_content and agent_done:
                count += 1
        return count

    def add_message(self, message_type, content):
        timestamp = datetime.datetime.now().strftime("%H:%M:%S")
        self.messages.append((timestamp, message_type, content))

    def add_tool_call(self, tool_name, args):
        timestamp = datetime.datetime.now().strftime("%H:%M:%S")
        self.tool_calls.append((timestamp, tool_name, args))

    def update_agent_status(self, agent, status):
        if agent in self.agent_status:
            self.agent_status[agent] = status
            self.current_agent = agent

    def update_report_section(self, section_name, content):
        if section_name in self.report_sections:
            self.report_sections[section_name] = content
            self._update_current_report()

    def _update_current_report(self):
        # For the panel display, only show the most recently updated section
        latest_section = None
        latest_content = None

        # Find the most recently updated section
        for section, content in self.report_sections.items():
            if content is not None:
                latest_section = section
                latest_content = content
               
        if latest_section and latest_content:
            # Format the current section for display
            section_titles = {
                "market_report": "Market Analysis",
                "sentiment_report": "Social Sentiment",
                "news_report": "News Analysis",
                "fundamentals_report": "Fundamentals Analysis",
                "investment_plan": "Research Team Decision",
                "trader_investment_plan": "Trading Team Plan",
                "final_trade_decision": "Portfolio Management Decision",
            }
            self.current_report = (
                f"### {section_titles[latest_section]}\n{latest_content}"
            )

        # Update the final complete report
        self._update_final_report()

    def _update_final_report(self):
        report_parts = []

        # Analyst Team Reports - use .get() to handle missing sections
        analyst_sections = ["market_report", "sentiment_report", "news_report", "fundamentals_report"]
        if any(self.report_sections.get(section) for section in analyst_sections):
            report_parts.append("## Analyst Team Reports")
            if self.report_sections.get("market_report"):
                report_parts.append(
                    f"### Market Analysis\n{self.report_sections['market_report']}"
                )
            if self.report_sections.get("sentiment_report"):
                report_parts.append(
                    f"### Social Sentiment\n{self.report_sections['sentiment_report']}"
                )
            if self.report_sections.get("news_report"):
                report_parts.append(
                    f"### News Analysis\n{self.report_sections['news_report']}"
                )
            if self.report_sections.get("fundamentals_report"):
                report_parts.append(
                    f"### Fundamentals Analysis\n{self.report_sections['fundamentals_report']}"
                )

        # Research Team Reports
        if self.report_sections.get("investment_plan"):
            report_parts.append("## Research Team Decision")
            report_parts.append(f"{self.report_sections['investment_plan']}")

        # Trading Team Reports
        if self.report_sections.get("trader_investment_plan"):
            report_parts.append("## Trading Team Plan")
            report_parts.append(f"{self.report_sections['trader_investment_plan']}")

        # Portfolio Management Decision
        if self.report_sections.get("final_trade_decision"):
            report_parts.append("## Portfolio Management Decision")
            report_parts.append(f"{self.report_sections['final_trade_decision']}")

        self.final_report = "\n\n".join(report_parts) if report_parts else None


message_buffer = MessageBuffer()


def create_layout():
    layout = Layout()
    layout.split_column(
        Layout(name="header", size=3),
        Layout(name="main"),
        Layout(name="footer", size=3),
    )
    layout["main"].split_column(
        Layout(name="upper", ratio=3), Layout(name="analysis", ratio=5)
    )
    layout["upper"].split_row(
        Layout(name="progress", ratio=2), Layout(name="messages", ratio=3)
    )
    return layout


def format_tokens(n):
    """Format token count for display."""
    if n >= 1000:
        return f"{n/1000:.1f}k"
    return str(n)


def update_display(layout, spinner_text=None, stats_handler=None, start_time=None):
    # Header with welcome message
    layout["header"].update(
        Panel(
            "[bold green]Welcome to TradingAgents CLI[/bold green]\n"
            "[dim]© [Tauric Research](https://github.com/TauricResearch)[/dim]",
            title="Welcome to TradingAgents",
            border_style="green",
            padding=(1, 2),
            expand=True,
        )
    )

    # Progress panel showing agent status
    progress_table = Table(
        show_header=True,
        header_style="bold magenta",
        show_footer=False,
        box=box.SIMPLE_HEAD,  # Use simple header with horizontal lines
        title=None,  # Remove the redundant Progress title
        padding=(0, 2),  # Add horizontal padding
        expand=True,  # Make table expand to fill available space
    )
    progress_table.add_column("Team", style="cyan", justify="center", width=20)
    progress_table.add_column("Agent", style="green", justify="center", width=20)
    progress_table.add_column("Status", style="yellow", justify="center", width=20)

    # Group agents by team - filter to only include agents in agent_status
    all_teams = {
        "Analyst Team": [
            "Market Analyst",
            "Sentiment Analyst",
            "News Analyst",
            "Fundamentals Analyst",
        ],
        "Research Team": ["Bull Researcher", "Bear Researcher", "Research Manager"],
        "Trading Team": ["Trader"],
        "Risk Management": ["Aggressive Analyst", "Neutral Analyst", "Conservative Analyst"],
        "Portfolio Management": ["Portfolio Manager"],
    }

    # Filter teams to only include agents that are in agent_status
    teams = {}
    for team, agents in all_teams.items():
        active_agents = [a for a in agents if a in message_buffer.agent_status]
        if active_agents:
            teams[team] = active_agents

    for team, agents in teams.items():
        # Add first agent with team name
        first_agent = agents[0]
        status = message_buffer.agent_status.get(first_agent, "pending")
        if status == "in_progress":
            spinner = Spinner(
                "dots", text="[blue]in_progress[/blue]", style="bold cyan"
            )
            status_cell = spinner
        else:
            status_color = {
                "pending": "yellow",
                "completed": "green",
                "error": "red",
            }.get(status, "white")
            status_cell = f"[{status_color}]{status}[/{status_color}]"
        progress_table.add_row(team, first_agent, status_cell)

        # Add remaining agents in team
        for agent in agents[1:]:
            status = message_buffer.agent_status.get(agent, "pending")
            if status == "in_progress":
                spinner = Spinner(
                    "dots", text="[blue]in_progress[/blue]", style="bold cyan"
                )
                status_cell = spinner
            else:
                status_color = {
                    "pending": "yellow",
                    "completed": "green",
                    "error": "red",
                }.get(status, "white")
                status_cell = f"[{status_color}]{status}[/{status_color}]"
            progress_table.add_row("", agent, status_cell)

        # Add horizontal line after each team
        progress_table.add_row("─" * 20, "─" * 20, "─" * 20, style="dim")

    layout["progress"].update(
        Panel(progress_table, title="Progress", border_style="cyan", padding=(1, 2))
    )

    # Messages panel showing recent messages and tool calls
    messages_table = Table(
        show_header=True,
        header_style="bold magenta",
        show_footer=False,
        expand=True,  # Make table expand to fill available space
        box=box.MINIMAL,  # Use minimal box style for a lighter look
        show_lines=True,  # Keep horizontal lines
        padding=(0, 1),  # Add some padding between columns
    )
    messages_table.add_column("Time", style="cyan", width=8, justify="center")
    messages_table.add_column("Type", style="green", width=10, justify="center")
    messages_table.add_column(
        "Content", style="white", no_wrap=False, ratio=1
    )  # Make content column expand

    # Combine tool calls and messages
    all_messages = []

    # Add tool calls
    for timestamp, tool_name, args in message_buffer.tool_calls:
        formatted_args = format_tool_args(args)
        all_messages.append((timestamp, "Tool", f"{tool_name}: {formatted_args}"))

    # Add regular messages
    for timestamp, msg_type, content in message_buffer.messages:
        content_str = str(content) if content else ""
        if len(content_str) > 200:
            content_str = content_str[:197] + "..."
        all_messages.append((timestamp, msg_type, content_str))

    # Sort by timestamp descending (newest first)
    all_messages.sort(key=lambda x: x[0], reverse=True)

    # Calculate how many messages we can show based on available space
    max_messages = 12

    # Get the first N messages (newest ones)
    recent_messages = all_messages[:max_messages]

    # Add messages to table (already in newest-first order)
    for timestamp, msg_type, content in recent_messages:
        # Format content with word wrapping
        wrapped_content = Text(content, overflow="fold")
        messages_table.add_row(timestamp, msg_type, wrapped_content)

    layout["messages"].update(
        Panel(
            messages_table,
            title="Messages & Tools",
            border_style="blue",
            padding=(1, 2),
        )
    )

    # Analysis panel showing current report
    if message_buffer.current_report:
        layout["analysis"].update(
            Panel(
                Markdown(message_buffer.current_report),
                title="Current Report",
                border_style="green",
                padding=(1, 2),
            )
        )
    else:
        layout["analysis"].update(
            Panel(
                "[italic]Waiting for analysis report...[/italic]",
                title="Current Report",
                border_style="green",
                padding=(1, 2),
            )
        )

    # Footer with statistics
    # Agent progress - derived from agent_status dict
    agents_completed = sum(
        1 for status in message_buffer.agent_status.values() if status == "completed"
    )
    agents_total = len(message_buffer.agent_status)

    # Report progress - based on agent completion (not just content existence)
    reports_completed = message_buffer.get_completed_reports_count()
    reports_total = len(message_buffer.report_sections)

    # Build stats parts
    stats_parts = [f"Agents: {agents_completed}/{agents_total}"]

    # LLM and tool stats from callback handler
    if stats_handler:
        stats = stats_handler.get_stats()
        stats_parts.append(f"LLM: {stats['llm_calls']}")
        stats_parts.append(f"Tools: {stats['tool_calls']}")

        # Token display with graceful fallback
        if stats["tokens_in"] > 0 or stats["tokens_out"] > 0:
            tokens_str = f"Tokens: {format_tokens(stats['tokens_in'])}\u2191 {format_tokens(stats['tokens_out'])}\u2193"
        else:
            tokens_str = "Tokens: --"
        stats_parts.append(tokens_str)

    stats_parts.append(f"Reports: {reports_completed}/{reports_total}")

    # Elapsed time
    if start_time:
        elapsed = time.time() - start_time
        elapsed_str = f"\u23f1 {int(elapsed // 60):02d}:{int(elapsed % 60):02d}"
        stats_parts.append(elapsed_str)

    stats_table = Table(show_header=False, box=None, padding=(0, 2), expand=True)
    stats_table.add_column("Stats", justify="center")
    stats_table.add_row(" | ".join(stats_parts))

    layout["footer"].update(Panel(stats_table, border_style="grey50"))


def get_user_selections():
    """Get all user selections before starting the analysis display."""
    # Display ASCII art welcome message
    with open(Path(__file__).parent / "static" / "welcome.txt", "r", encoding="utf-8") as f:
        welcome_ascii = f.read()

    # Create welcome box content
    welcome_content = f"{welcome_ascii}\n"
    welcome_content += "[bold green]TradingAgents: Multi-Agents LLM Financial Trading Framework - CLI[/bold green]\n\n"
    welcome_content += "[bold]Workflow Steps:[/bold]\n"
    welcome_content += "I. Analyst Team → II. Research Team → III. Trader → IV. Risk Management → V. Portfolio Management\n\n"
    welcome_content += (
        "[dim]Built by [Tauric Research](https://github.com/TauricResearch)[/dim]"
    )

    # Create and center the welcome box
    welcome_box = Panel(
        welcome_content,
        border_style="green",
        padding=(1, 2),
        title="Welcome to TradingAgents",
        subtitle="Multi-Agents LLM Financial Trading Framework",
    )
    console.print(Align.center(welcome_box))
    console.print()
    console.print()  # Add vertical space before announcements

    # Fetch and display announcements (silent on failure)
    announcements = fetch_announcements()
    display_announcements(console, announcements)

    # Create a boxed questionnaire for each step
    def create_question_box(title, prompt, default=None):
        box_content = f"[bold]{title}[/bold]\n"
        box_content += f"[dim]{prompt}[/dim]"
        if default:
            box_content += f"\n[dim]Default: {default}[/dim]"
        return Panel(box_content, border_style="blue", padding=(1, 2))

    # Step 1: Ticker symbol
    console.print(
        create_question_box(
            "Step 1: Ticker Symbol",
            "Enter the ticker, with exchange suffix when needed (e.g. SPY, 0700.HK, BTC-USD)",
            "SPY",
        )
    )
    selected_ticker = get_ticker()
    asset_type = detect_asset_type(selected_ticker)
    # Only announce when it's not the default stock path, to avoid printing
    # "stock" on every run.
    if asset_type.value != "stock":
        console.print(
            f"[green]Detected asset type:[/green] {asset_type.value}"
        )

    # Step 2: Analysis date
    default_date = datetime.datetime.now().strftime("%Y-%m-%d")
    console.print(
        create_question_box(
            "Step 2: Analysis Date",
            "Enter the analysis date (YYYY-MM-DD)",
            default_date,
        )
    )
    analysis_date = get_analysis_date()

    # Step 3: Output language (skipped when set via TRADINGAGENTS_OUTPUT_LANGUAGE)
    if os.environ.get("TRADINGAGENTS_OUTPUT_LANGUAGE"):
        output_language = DEFAULT_CONFIG["output_language"]
        console.print(
            f"[green]✓ Output language from environment:[/green] {output_language}"
        )
    else:
        console.print(
            create_question_box(
                "Step 3: Output Language",
                "Select the language for analyst reports and final decision"
            )
        )
        output_language = ask_output_language()

    # Step 4: Select analysts
    console.print(
        create_question_box(
            "Step 4: Analysts Team", "Select your LLM analyst agents for the analysis"
        )
    )
    selected_analysts = select_analysts(asset_type)
    console.print(
        f"[green]Selected analysts:[/green] {', '.join(analyst.value for analyst in selected_analysts)}"
    )

    # Step 5: Research depth
    console.print(
        create_question_box(
            "Step 5: Research Depth", "Select your research depth level"
        )
    )
    selected_research_depth = select_research_depth()

    # Step 6: LLM Provider (skipped when set via TRADINGAGENTS_LLM_PROVIDER).
    # The backend URL comes from TRADINGAGENTS_LLM_BACKEND_URL when set,
    # otherwise the provider's default endpoint — the same value the menu
    # would have picked.
    provider_from_env = bool(os.environ.get("TRADINGAGENTS_LLM_PROVIDER"))
    if provider_from_env:
        selected_llm_provider = DEFAULT_CONFIG["llm_provider"].lower()
        backend_url = DEFAULT_CONFIG["backend_url"] or provider_default_url(selected_llm_provider)
        console.print(f"[green]✓ LLM provider from environment:[/green] {selected_llm_provider}")
        console.print(f"[green]✓ Backend URL:[/green] {backend_url}")
        # Still confirm/persist the API key so the run doesn't fail later.
        ensure_api_key(selected_llm_provider)
    else:
        console.print(
            create_question_box(
                "Step 6: LLM Provider", "Select your LLM provider"
            )
        )
        selected_llm_provider, backend_url = select_llm_provider()

        # Providers with regional endpoints prompt for the region as a secondary
        # step so the main dropdown stays clean (mainland China and international
        # accounts cannot share API keys).
        if selected_llm_provider == "qwen":
            selected_llm_provider, backend_url = ask_qwen_region()
        elif selected_llm_provider == "minimax":
            selected_llm_provider, backend_url = ask_minimax_region()
        elif selected_llm_provider == "glm":
            selected_llm_provider, backend_url = ask_glm_region()

        # For Ollama, surface the resolved endpoint (OLLAMA_BASE_URL vs default)
        # before model selection so it's obvious where we're connecting.
        if selected_llm_provider == "ollama":
            confirm_ollama_endpoint(backend_url)

        # Confirm the provider's API key is present; prompt the user to paste
        # one and persist it to .env if it's missing, so the analysis run
        # doesn't fail later at the first API call.
        ensure_api_key(selected_llm_provider)

    # Step 7: Thinking agents (skipped when either model is set via environment)
    if os.environ.get("TRADINGAGENTS_QUICK_THINK_LLM") or os.environ.get("TRADINGAGENTS_DEEP_THINK_LLM"):
        selected_shallow_thinker = DEFAULT_CONFIG["quick_think_llm"]
        selected_deep_thinker = DEFAULT_CONFIG["deep_think_llm"]
        console.print(
            f"[green]✓ Thinking agents from environment:[/green] "
            f"quick={selected_shallow_thinker}, deep={selected_deep_thinker}"
        )
    else:
        console.print(
            create_question_box(
                "Step 7: Thinking Agents", "Select your thinking agents for analysis"
            )
        )
        selected_shallow_thinker = select_shallow_thinking_agent(selected_llm_provider)
        selected_deep_thinker = select_deep_thinking_agent(selected_llm_provider)

    # Step 8: Provider-specific thinking configuration
    thinking_level = None
    reasoning_effort = None
    anthropic_effort = None

    provider_lower = selected_llm_provider.lower()
    # When the provider is configured via environment we keep the run fully
    # non-interactive and use the config defaults (None = each provider's own
    # default reasoning/thinking behavior) instead of prompting.
    if provider_from_env:
        thinking_level = DEFAULT_CONFIG["google_thinking_level"]
        reasoning_effort = DEFAULT_CONFIG["openai_reasoning_effort"]
        anthropic_effort = DEFAULT_CONFIG["anthropic_effort"]
    elif provider_lower == "google":
        console.print(
            create_question_box(
                "Step 8: Thinking Mode",
                "Configure Gemini thinking mode"
            )
        )
        thinking_level = ask_gemini_thinking_config()
    elif provider_lower == "openai":
        console.print(
            create_question_box(
                "Step 8: Reasoning Effort",
                "Configure OpenAI reasoning effort level"
            )
        )
        reasoning_effort = ask_openai_reasoning_effort()
    elif provider_lower == "anthropic":
        console.print(
            create_question_box(
                "Step 8: Effort Level",
                "Configure Claude effort level"
            )
        )
        anthropic_effort = ask_anthropic_effort()

    return {
        "ticker": selected_ticker,
        "asset_type": asset_type.value,
        "analysis_date": analysis_date,
        "analysts": selected_analysts,
        "research_depth": selected_research_depth,
        "llm_provider": selected_llm_provider.lower(),
        "backend_url": backend_url,
        "shallow_thinker": selected_shallow_thinker,
        "deep_thinker": selected_deep_thinker,
        "google_thinking_level": thinking_level,
        "openai_reasoning_effort": reasoning_effort,
        "anthropic_effort": anthropic_effort,
        "output_language": output_language,
    }


def get_analysis_date():
    """Get the analysis date from user input."""
    while True:
        date_str = typer.prompt(
            "", default=datetime.datetime.now().strftime("%Y-%m-%d")
        )
        try:
            # Validate date format and ensure it's not in the future
            analysis_date = datetime.datetime.strptime(date_str, "%Y-%m-%d")
            if analysis_date.date() > datetime.datetime.now().date():
                console.print("[red]Error: Analysis date cannot be in the future[/red]")
                continue
            return date_str
        except ValueError:
            console.print(
                "[red]Error: Invalid date format. Please use YYYY-MM-DD[/red]"
            )


def save_report_to_disk(final_state, ticker: str, save_path: Path):
    """Save complete analysis report to disk with organized subfolders."""
    save_path.mkdir(parents=True, exist_ok=True)
    sections = []

    # 1. Analysts
    analysts_dir = save_path / "1_analysts"
    analyst_parts = []
    if final_state.get("market_report"):
        analysts_dir.mkdir(exist_ok=True)
        (analysts_dir / "market.md").write_text(final_state["market_report"], encoding="utf-8")
        analyst_parts.append(("Market Analyst", final_state["market_report"]))
    if final_state.get("sentiment_report"):
        analysts_dir.mkdir(exist_ok=True)
        (analysts_dir / "sentiment.md").write_text(final_state["sentiment_report"], encoding="utf-8")
        analyst_parts.append(("Sentiment Analyst", final_state["sentiment_report"]))
    if final_state.get("news_report"):
        analysts_dir.mkdir(exist_ok=True)
        (analysts_dir / "news.md").write_text(final_state["news_report"], encoding="utf-8")
        analyst_parts.append(("News Analyst", final_state["news_report"]))
    if final_state.get("fundamentals_report"):
        analysts_dir.mkdir(exist_ok=True)
        (analysts_dir / "fundamentals.md").write_text(final_state["fundamentals_report"], encoding="utf-8")
        analyst_parts.append(("Fundamentals Analyst", final_state["fundamentals_report"]))
    if analyst_parts:
        content = "\n\n".join(f"### {name}\n{text}" for name, text in analyst_parts)
        sections.append(f"## I. Analyst Team Reports\n\n{content}")

    # 2. Research
    if final_state.get("investment_debate_state"):
        research_dir = save_path / "2_research"
        debate = final_state["investment_debate_state"]
        research_parts = []
        if debate.get("bull_history"):
            research_dir.mkdir(exist_ok=True)
            (research_dir / "bull.md").write_text(debate["bull_history"], encoding="utf-8")
            research_parts.append(("Bull Researcher", debate["bull_history"]))
        if debate.get("bear_history"):
            research_dir.mkdir(exist_ok=True)
            (research_dir / "bear.md").write_text(debate["bear_history"], encoding="utf-8")
            research_parts.append(("Bear Researcher", debate["bear_history"]))
        if debate.get("judge_decision"):
            research_dir.mkdir(exist_ok=True)
            (research_dir / "manager.md").write_text(debate["judge_decision"], encoding="utf-8")
            research_parts.append(("Research Manager", debate["judge_decision"]))
        if research_parts:
            content = "\n\n".join(f"### {name}\n{text}" for name, text in research_parts)
            sections.append(f"## II. Research Team Decision\n\n{content}")

    # 3. Trading
    if final_state.get("trader_investment_plan"):
        trading_dir = save_path / "3_trading"
        trading_dir.mkdir(exist_ok=True)
        (trading_dir / "trader.md").write_text(final_state["trader_investment_plan"], encoding="utf-8")
        sections.append(f"## III. Trading Team Plan\n\n### Trader\n{final_state['trader_investment_plan']}")

    # 4. Risk Management
    if final_state.get("risk_debate_state"):
        risk_dir = save_path / "4_risk"
        risk = final_state["risk_debate_state"]
        risk_parts = []
        if risk.get("aggressive_history"):
            risk_dir.mkdir(exist_ok=True)
            (risk_dir / "aggressive.md").write_text(risk["aggressive_history"], encoding="utf-8")
            risk_parts.append(("Aggressive Analyst", risk["aggressive_history"]))
        if risk.get("conservative_history"):
            risk_dir.mkdir(exist_ok=True)
            (risk_dir / "conservative.md").write_text(risk["conservative_history"], encoding="utf-8")
            risk_parts.append(("Conservative Analyst", risk["conservative_history"]))
        if risk.get("neutral_history"):
            risk_dir.mkdir(exist_ok=True)
            (risk_dir / "neutral.md").write_text(risk["neutral_history"], encoding="utf-8")
            risk_parts.append(("Neutral Analyst", risk["neutral_history"]))
        if risk_parts:
            content = "\n\n".join(f"### {name}\n{text}" for name, text in risk_parts)
            sections.append(f"## IV. Risk Management Team Decision\n\n{content}")

        # 5. Portfolio Manager
        if risk.get("judge_decision"):
            portfolio_dir = save_path / "5_portfolio"
            portfolio_dir.mkdir(exist_ok=True)
            (portfolio_dir / "decision.md").write_text(risk["judge_decision"], encoding="utf-8")
            sections.append(f"## V. Portfolio Manager Decision\n\n### Portfolio Manager\n{risk['judge_decision']}")

    # Write consolidated report
    header = f"# Trading Analysis Report: {ticker}\n\nGenerated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
    (save_path / "complete_report.md").write_text(header + "\n\n".join(sections), encoding="utf-8")
    return save_path / "complete_report.md"


def display_complete_report(final_state):
    """Display the complete analysis report sequentially (avoids truncation)."""
    console.print()
    console.print(Rule("Complete Analysis Report", style="bold green"))

    # I. Analyst Team Reports
    analysts = []
    if final_state.get("market_report"):
        analysts.append(("Market Analyst", final_state["market_report"]))
    if final_state.get("sentiment_report"):
        analysts.append(("Sentiment Analyst", final_state["sentiment_report"]))
    if final_state.get("news_report"):
        analysts.append(("News Analyst", final_state["news_report"]))
    if final_state.get("fundamentals_report"):
        analysts.append(("Fundamentals Analyst", final_state["fundamentals_report"]))
    if analysts:
        console.print(Panel("[bold]I. Analyst Team Reports[/bold]", border_style="cyan"))
        for title, content in analysts:
            console.print(Panel(Markdown(content), title=title, border_style="blue", padding=(1, 2)))

    # II. Research Team Reports
    if final_state.get("investment_debate_state"):
        debate = final_state["investment_debate_state"]
        research = []
        if debate.get("bull_history"):
            research.append(("Bull Researcher", debate["bull_history"]))
        if debate.get("bear_history"):
            research.append(("Bear Researcher", debate["bear_history"]))
        if debate.get("judge_decision"):
            research.append(("Research Manager", debate["judge_decision"]))
        if research:
            console.print(Panel("[bold]II. Research Team Decision[/bold]", border_style="magenta"))
            for title, content in research:
                console.print(Panel(Markdown(content), title=title, border_style="blue", padding=(1, 2)))

    # III. Trading Team
    if final_state.get("trader_investment_plan"):
        console.print(Panel("[bold]III. Trading Team Plan[/bold]", border_style="yellow"))
        console.print(Panel(Markdown(final_state["trader_investment_plan"]), title="Trader", border_style="blue", padding=(1, 2)))

    # IV. Risk Management Team
    if final_state.get("risk_debate_state"):
        risk = final_state["risk_debate_state"]
        risk_reports = []
        if risk.get("aggressive_history"):
            risk_reports.append(("Aggressive Analyst", risk["aggressive_history"]))
        if risk.get("conservative_history"):
            risk_reports.append(("Conservative Analyst", risk["conservative_history"]))
        if risk.get("neutral_history"):
            risk_reports.append(("Neutral Analyst", risk["neutral_history"]))
        if risk_reports:
            console.print(Panel("[bold]IV. Risk Management Team Decision[/bold]", border_style="red"))
            for title, content in risk_reports:
                console.print(Panel(Markdown(content), title=title, border_style="blue", padding=(1, 2)))

        # V. Portfolio Manager Decision
        if risk.get("judge_decision"):
            console.print(Panel("[bold]V. Portfolio Manager Decision[/bold]", border_style="green"))
            console.print(Panel(Markdown(risk["judge_decision"]), title="Portfolio Manager", border_style="blue", padding=(1, 2)))


def update_research_team_status(status):
    """Update status for research team members (not Trader)."""
    research_team = ["Bull Researcher", "Bear Researcher", "Research Manager"]
    for agent in research_team:
        message_buffer.update_agent_status(agent, status)


# Ordered list of analysts for status transitions
ANALYST_ORDER = ["market", "social", "news", "fundamentals"]
ANALYST_AGENT_NAMES = {
    "market": "Market Analyst",
    "social": "Sentiment Analyst",
    "news": "News Analyst",
    "fundamentals": "Fundamentals Analyst",
}
ANALYST_REPORT_MAP = {
    "market": "market_report",
    "social": "sentiment_report",
    "news": "news_report",
    "fundamentals": "fundamentals_report",
}


def update_analyst_statuses(message_buffer, chunk, wall_time_tracker=None):
    """Update analyst statuses based on accumulated report state.

    Logic:
    - Store new report content from the current chunk if present
    - Check accumulated report_sections (not just current chunk) for status
    - Analysts with reports = completed
    - First analyst without report = in_progress
    - Remaining analysts without reports = pending
    - When all analysts done, set Bull Researcher to in_progress
    """
    selected = message_buffer.selected_analysts
    found_active = False

    if wall_time_tracker is not None:
        sync_analyst_tracker_from_chunk(wall_time_tracker, chunk)

    for analyst_key in ANALYST_ORDER:
        if analyst_key not in selected:
            continue

        agent_name = ANALYST_AGENT_NAMES[analyst_key]
        report_key = ANALYST_REPORT_MAP[analyst_key]

        # Capture new report content from current chunk
        if chunk.get(report_key):
            message_buffer.update_report_section(report_key, chunk[report_key])

        # Determine status from accumulated sections, not just current chunk
        has_report = bool(message_buffer.report_sections.get(report_key))

        if has_report:
            message_buffer.update_agent_status(agent_name, "completed")
        elif not found_active:
            message_buffer.update_agent_status(agent_name, "in_progress")
            found_active = True
        else:
            message_buffer.update_agent_status(agent_name, "pending")

    # When all analysts complete, transition research team to in_progress
    if not found_active and selected:
        if message_buffer.agent_status.get("Bull Researcher") == "pending":
            message_buffer.update_agent_status("Bull Researcher", "in_progress")

def extract_content_string(content):
    """Extract string content from various message formats.
    Returns None if no meaningful text content is found.
    """
    import ast

    def is_empty(val):
        """Check if value is empty using Python's truthiness."""
        if val is None or val == '':
            return True
        if isinstance(val, str):
            s = val.strip()
            if not s:
                return True
            try:
                return not bool(ast.literal_eval(s))
            except (ValueError, SyntaxError):
                return False  # Can't parse = real text
        return not bool(val)

    if is_empty(content):
        return None

    if isinstance(content, str):
        return content.strip()

    if isinstance(content, dict):
        text = content.get('text', '')
        return text.strip() if not is_empty(text) else None

    if isinstance(content, list):
        text_parts = [
            item.get('text', '').strip() if isinstance(item, dict) and item.get('type') == 'text'
            else (item.strip() if isinstance(item, str) else '')
            for item in content
        ]
        result = ' '.join(t for t in text_parts if t and not is_empty(t))
        return result if result else None

    return str(content).strip() if not is_empty(content) else None


def classify_message_type(message) -> tuple[str, str | None]:
    """Classify LangChain message into display type and extract content.

    Returns:
        (type, content) - type is one of: User, Agent, Data, Control
                        - content is extracted string or None
    """
    from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

    content = extract_content_string(getattr(message, 'content', None))

    if isinstance(message, HumanMessage):
        if content and content.strip() == "Continue":
            return ("Control", content)
        return ("User", content)

    if isinstance(message, ToolMessage):
        return ("Data", content)

    if isinstance(message, AIMessage):
        return ("Agent", content)

    # Fallback for unknown types
    return ("System", content)


def format_tool_args(args, max_length=80) -> str:
    """Format tool arguments for terminal display."""
    result = str(args)
    if len(result) > max_length:
        return result[:max_length - 3] + "..."
    return result

def run_analysis(checkpoint: bool = False):
    # First get all user selections
    selections = get_user_selections()

    # Create config with selected research depth
    config = DEFAULT_CONFIG.copy()
    config["max_debate_rounds"] = selections["research_depth"]
    config["max_risk_discuss_rounds"] = selections["research_depth"]
    config["quick_think_llm"] = selections["shallow_thinker"]
    config["deep_think_llm"] = selections["deep_thinker"]
    config["backend_url"] = selections["backend_url"]
    config["llm_provider"] = selections["llm_provider"].lower()
    # Provider-specific thinking configuration
    config["google_thinking_level"] = selections.get("google_thinking_level")
    config["openai_reasoning_effort"] = selections.get("openai_reasoning_effort")
    config["anthropic_effort"] = selections.get("anthropic_effort")
    config["output_language"] = selections.get("output_language", "English")
    config["checkpoint_enabled"] = checkpoint

    # Create stats callback handler for tracking LLM/tool calls
    stats_handler = StatsCallbackHandler()

    # Normalize analyst selection to predefined order (selection is a 'set', order is fixed)
    selected_set = {analyst.value for analyst in selections["analysts"]}
    selected_analyst_keys = [a for a in ANALYST_ORDER if a in selected_set]
    analyst_execution_plan = build_analyst_execution_plan(
        selected_analyst_keys,
        concurrency_limit=config["analyst_concurrency_limit"],
    )
    analyst_wall_time_tracker = AnalystWallTimeTracker(analyst_execution_plan)

    # Initialize the graph with callbacks bound to LLMs
    graph = TradingAgentsGraph(
        selected_analyst_keys,
        config=config,
        debug=True,
        callbacks=[stats_handler],
    )

    # Initialize message buffer with selected analysts
    message_buffer.init_for_analysis(selected_analyst_keys)

    # Track start time for elapsed display
    start_time = time.time()

    # Create result directory
    results_dir = Path(config["results_dir"]) / selections["ticker"] / selections["analysis_date"]
    results_dir.mkdir(parents=True, exist_ok=True)
    report_dir = results_dir / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    log_file = results_dir / "message_tool.log"
    log_file.touch(exist_ok=True)

    def save_message_decorator(obj, func_name):
        func = getattr(obj, func_name)
        @wraps(func)
        def wrapper(*args, **kwargs):
            func(*args, **kwargs)
            timestamp, message_type, content = obj.messages[-1]
            content = content.replace("\n", " ")  # Replace newlines with spaces
            with open(log_file, "a", encoding="utf-8") as f:
                f.write(f"{timestamp} [{message_type}] {content}\n")
        return wrapper
    
    def save_tool_call_decorator(obj, func_name):
        func = getattr(obj, func_name)
        @wraps(func)
        def wrapper(*args, **kwargs):
            func(*args, **kwargs)
            timestamp, tool_name, args = obj.tool_calls[-1]
            args_str = ", ".join(f"{k}={v}" for k, v in args.items())
            with open(log_file, "a", encoding="utf-8") as f:
                f.write(f"{timestamp} [Tool Call] {tool_name}({args_str})\n")
        return wrapper

    def save_report_section_decorator(obj, func_name):
        func = getattr(obj, func_name)
        @wraps(func)
        def wrapper(section_name, content):
            func(section_name, content)
            if section_name in obj.report_sections and obj.report_sections[section_name] is not None:
                content = obj.report_sections[section_name]
                if content:
                    file_name = f"{section_name}.md"
                    text = "\n".join(str(item) for item in content) if isinstance(content, list) else content
                    with open(report_dir / file_name, "w", encoding="utf-8") as f:
                        f.write(text)
        return wrapper

    message_buffer.add_message = save_message_decorator(message_buffer, "add_message")
    message_buffer.add_tool_call = save_tool_call_decorator(message_buffer, "add_tool_call")
    message_buffer.update_report_section = save_report_section_decorator(message_buffer, "update_report_section")

    # Now start the display layout
    layout = create_layout()

    with Live(layout, refresh_per_second=4) as live:
        # Initial display
        update_display(layout, stats_handler=stats_handler, start_time=start_time)

        # Add initial messages
        message_buffer.add_message("System", f"Selected ticker: {selections['ticker']}")
        if selections["asset_type"] != "stock":
            message_buffer.add_message("System", f"Detected asset type: {selections['asset_type']}")
        message_buffer.add_message(
            "System", f"Analysis date: {selections['analysis_date']}"
        )
        message_buffer.add_message(
            "System",
            f"Selected analysts: {', '.join(analyst.value for analyst in selections['analysts'])}",
        )
        update_display(layout, stats_handler=stats_handler, start_time=start_time)

        # Update agent status to in_progress for the first analyst
        first_analyst = get_initial_analyst_node(analyst_execution_plan)
        message_buffer.update_agent_status(first_analyst, "in_progress")
        analyst_wall_time_tracker.mark_started(selected_analyst_keys[0])
        update_display(layout, stats_handler=stats_handler, start_time=start_time)

        # Create spinner text
        spinner_text = (
            f"Analyzing {selections['ticker']} on {selections['analysis_date']}..."
        )
        update_display(layout, spinner_text, stats_handler=stats_handler, start_time=start_time)

        # Initialize state and get graph args with callbacks.
        # Resolve the instrument identity once here so all agents anchor to
        # the real company (#814); the CLI builds state directly rather than
        # going through propagate(), so this must happen on the CLI path too.
        instrument_context = graph.resolve_instrument_context(
            selections["ticker"], selections["asset_type"]
        )
        init_agent_state = graph.propagator.create_initial_state(
            selections["ticker"],
            selections["analysis_date"],
            asset_type=selections["asset_type"],
            instrument_context=instrument_context,
        )
        # Pass callbacks to graph config for tool execution tracking
        # (LLM tracking is handled separately via LLM constructor)
        args = graph.propagator.get_graph_args(callbacks=[stats_handler])

        # Stream the analysis
        trace = []
        for chunk in graph.graph.stream(init_agent_state, **args):
            # Process all messages in chunk, deduplicating by message ID
            for message in chunk.get("messages", []):
                msg_id = getattr(message, "id", None)
                if msg_id is not None:
                    if msg_id in message_buffer._processed_message_ids:
                        continue
                    message_buffer._processed_message_ids.add(msg_id)

                msg_type, content = classify_message_type(message)
                if content and content.strip():
                    message_buffer.add_message(msg_type, content)

                if hasattr(message, "tool_calls") and message.tool_calls:
                    for tool_call in message.tool_calls:
                        if isinstance(tool_call, dict):
                            message_buffer.add_tool_call(tool_call["name"], tool_call["args"])
                        else:
                            message_buffer.add_tool_call(tool_call.name, tool_call.args)

            # Update analyst statuses based on report state (runs on every chunk)
            update_analyst_statuses(
                message_buffer,
                chunk,
                wall_time_tracker=analyst_wall_time_tracker,
            )

            # Research Team - Handle Investment Debate State
            if chunk.get("investment_debate_state"):
                debate_state = chunk["investment_debate_state"]
                bull_hist = debate_state.get("bull_history", "").strip()
                bear_hist = debate_state.get("bear_history", "").strip()
                judge = debate_state.get("judge_decision", "").strip()

                # Only update status when there's actual content
                if bull_hist or bear_hist:
                    update_research_team_status("in_progress")
                if bull_hist:
                    message_buffer.update_report_section(
                        "investment_plan", f"### Bull Researcher Analysis\n{bull_hist}"
                    )
                if bear_hist:
                    message_buffer.update_report_section(
                        "investment_plan", f"### Bear Researcher Analysis\n{bear_hist}"
                    )
                if judge:
                    message_buffer.update_report_section(
                        "investment_plan", f"### Research Manager Decision\n{judge}"
                    )
                    update_research_team_status("completed")
                    message_buffer.update_agent_status("Trader", "in_progress")

            # Trading Team
            if chunk.get("trader_investment_plan"):
                message_buffer.update_report_section(
                    "trader_investment_plan", chunk["trader_investment_plan"]
                )
                if message_buffer.agent_status.get("Trader") != "completed":
                    message_buffer.update_agent_status("Trader", "completed")
                    message_buffer.update_agent_status("Aggressive Analyst", "in_progress")

            # Risk Management Team - Handle Risk Debate State
            if chunk.get("risk_debate_state"):
                risk_state = chunk["risk_debate_state"]
                agg_hist = risk_state.get("aggressive_history", "").strip()
                con_hist = risk_state.get("conservative_history", "").strip()
                neu_hist = risk_state.get("neutral_history", "").strip()
                judge = risk_state.get("judge_decision", "").strip()

                if agg_hist:
                    if message_buffer.agent_status.get("Aggressive Analyst") != "completed":
                        message_buffer.update_agent_status("Aggressive Analyst", "in_progress")
                    message_buffer.update_report_section(
                        "final_trade_decision", f"### Aggressive Analyst Analysis\n{agg_hist}"
                    )
                if con_hist:
                    if message_buffer.agent_status.get("Conservative Analyst") != "completed":
                        message_buffer.update_agent_status("Conservative Analyst", "in_progress")
                    message_buffer.update_report_section(
                        "final_trade_decision", f"### Conservative Analyst Analysis\n{con_hist}"
                    )
                if neu_hist:
                    if message_buffer.agent_status.get("Neutral Analyst") != "completed":
                        message_buffer.update_agent_status("Neutral Analyst", "in_progress")
                    message_buffer.update_report_section(
                        "final_trade_decision", f"### Neutral Analyst Analysis\n{neu_hist}"
                    )
                if judge:
                    if message_buffer.agent_status.get("Portfolio Manager") != "completed":
                        message_buffer.update_agent_status("Portfolio Manager", "in_progress")
                        message_buffer.update_report_section(
                            "final_trade_decision", f"### Portfolio Manager Decision\n{judge}"
                        )
                        message_buffer.update_agent_status("Aggressive Analyst", "completed")
                        message_buffer.update_agent_status("Conservative Analyst", "completed")
                        message_buffer.update_agent_status("Neutral Analyst", "completed")
                        message_buffer.update_agent_status("Portfolio Manager", "completed")

            # Update the display
            update_display(layout, stats_handler=stats_handler, start_time=start_time)

            trace.append(chunk)

        # Streamed chunks are per-node deltas, not full state. Merge them
        # so every report field populated across the run is present.
        final_state = {}
        for chunk in trace:
            final_state.update(chunk)
        decision = graph.process_signal(final_state["final_trade_decision"])

        # Update all agent statuses to completed
        for agent in message_buffer.agent_status:
            message_buffer.update_agent_status(agent, "completed")

        message_buffer.add_message(
            "System", f"Completed analysis for {selections['analysis_date']}"
        )
        message_buffer.add_message("System", analyst_wall_time_tracker.format_summary())

        # Update final report sections
        for section in message_buffer.report_sections.keys():
            if section in final_state:
                message_buffer.update_report_section(section, final_state[section])

        update_display(layout, stats_handler=stats_handler, start_time=start_time)

    # Post-analysis prompts (outside Live context for clean interaction)
    console.print("\n[bold cyan]Analysis Complete![/bold cyan]\n")
    console.print(f"[dim]{analyst_wall_time_tracker.format_summary()}[/dim]")

    # Prompt to save report
    save_choice = typer.prompt("Save report?", default="Y").strip().upper()
    if save_choice in ("Y", "YES", ""):
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        default_path = Path.cwd() / "reports" / f"{selections['ticker']}_{timestamp}"
        save_path_str = typer.prompt(
            "Save path (press Enter for default)",
            default=str(default_path)
        ).strip()
        save_path = Path(save_path_str)
        try:
            report_file = save_report_to_disk(final_state, selections["ticker"], save_path)
            console.print(f"\n[green]✓ Report saved to:[/green] {save_path.resolve()}")
            console.print(f"  [dim]Complete report:[/dim] {report_file.name}")
        except Exception as e:
            console.print(f"[red]Error saving report: {e}[/red]")

    # Prompt to display full report
    display_choice = typer.prompt("\nDisplay full report on screen?", default="Y").strip().upper()
    if display_choice in ("Y", "YES", ""):
        display_complete_report(final_state)


def _run_scan(
    universe: str,
    top: int,
    run_analysis_flag: bool,
    min_liquidity_cr: float,
    llm_provider: Optional[str],
    backend_url: Optional[str],
):
    """Core scan logic: screen → rank → optionally run full analysis."""
    from tradingagents.picker.screener import run_screen
    from tradingagents.picker.candidate import build_all_cards
    from tradingagents.picker.picker_agent import run_picker, render_picker_output
    from tradingagents.picker.universe import list_universes

    # ── 1. Validate universe ─────────────────────────────────────────────
    valid = list_universes()
    if universe not in valid:
        console.print(f"[red]Unknown universe '{universe}'. Valid: {', '.join(valid)}[/red]")
        raise typer.Exit(code=1)

    # ── 2. Quantitative screen ────────────────────────────────────────────
    universe_label = "DYNAMIC (NSE EQ-series)" if universe == "dynamic" else universe.upper()
    console.print(f"\n[bold cyan]Running quantitative screen — {universe_label}...[/bold cyan]")
    console.print(f"[dim]Filters: price ≥ ₹{DEFAULT_CONFIG['min_stock_price']}, "
                  f"liquidity ≥ ₹{min_liquidity_cr:.0f}Cr, not in F&O ban[/dim]")
    if universe == "dynamic":
        console.print(
            "[dim]Dynamic universe: auto-rebuilds from NSE when cache is older than "
            f"{DEFAULT_CONFIG.get('universe_max_age_days', 7)} days.[/dim]"
        )

    status_msg = (
        "[cyan]Loading dynamic universe + scoring (batched yfinance — may take 1–2 min on first run)...[/cyan]"
        if universe == "dynamic"
        else "[cyan]Downloading universe data (batched yfinance — may take ~30s)...[/cyan]"
    )
    with console.status(status_msg):
        scores = run_screen(
            universe=universe,
            max_candidates=25,
            min_liquidity_cr=min_liquidity_cr,
        )

    if not scores:
        console.print("[yellow]No stocks passed the screen. Try relaxing filters or a larger universe.[/yellow]")
        raise typer.Exit(code=0)

    # ── 3. Display screen results table ─────────────────────────────────
    screen_table = Table(
        title=f"Screen Results — {universe.upper()} — top {len(scores)} of universe",
        box=box.SIMPLE_HEAD,
        show_footer=False,
        header_style="bold magenta",
    )
    screen_table.add_column("Rank", width=4, justify="right")
    screen_table.add_column("Symbol", width=14)
    screen_table.add_column("Name", width=24)
    screen_table.add_column("Sector", width=18)
    screen_table.add_column("Price ₹", width=9, justify="right")
    screen_table.add_column("RSI", width=5, justify="right")
    screen_table.add_column("RS 1m%", width=8, justify="right")
    screen_table.add_column("VolSurge", width=9, justify="right")
    screen_table.add_column("DistHi%", width=8, justify="right")
    screen_table.add_column("Score", width=7, justify="right")

    for i, sc in enumerate(scores, 1):
        rs_color = "green" if sc.rel_strength_1m > 0 else "red"
        screen_table.add_row(
            str(i),
            sc.symbol,
            sc.name[:22],
            sc.sector[:16],
            f"{sc.price:,.0f}",
            f"{sc.rsi14:.0f}",
            f"[{rs_color}]{sc.rel_strength_1m:+.1f}%[/{rs_color}]",
            f"{sc.vol_surge_ratio:.1f}x",
            f"{sc.dist_from_high_pct:.1f}%",
            f"{sc.composite_score:.3f}",
        )

    console.print(screen_table)

    # ── 4. Build candidate cards ─────────────────────────────────────────
    console.print(f"\n[cyan]Building candidate cards (results dates, pledge data, headlines)...[/cyan]")
    with console.status("[cyan]Enriching candidate data...[/cyan]"):
        cards = build_all_cards(scores)

    # ── 5. LLM ranking ──────────────────────────────────────────────────
    console.print(f"\n[bold cyan]Running LLM picker (1 call — {DEFAULT_CONFIG['quick_think_llm']})...[/bold cyan]")

    llm = None
    if llm_provider:
        try:
            from tradingagents.llm_clients import create_llm_client
            llm = create_llm_client(
                provider=llm_provider,
                model=DEFAULT_CONFIG["quick_think_llm"],
                backend_url=backend_url,
                temperature=DEFAULT_CONFIG.get("temperature"),
            )
        except Exception as exc:
            console.print(f"[yellow]LLM init warning: {exc} — using default config[/yellow]")

    with console.status("[cyan]Picker agent ranking candidates...[/cyan]"):
        picker_result = run_picker(cards=cards, top_n=top, llm=llm)

    # ── 6. Display picker results ────────────────────────────────────────
    console.print()
    console.print(Rule("Stock Picker Results", style="bold green"))
    console.print(Panel(
        Markdown(render_picker_output(picker_result)),
        title=f"Top {top} Picks from {universe.upper()}",
        border_style="green",
        padding=(1, 2),
    ))

    # Print a compact action table
    picks_table = Table(
        title="Action Table",
        box=box.SIMPLE_HEAD,
        header_style="bold cyan",
    )
    picks_table.add_column("Rank", width=5, justify="right")
    picks_table.add_column("Ticker", width=14)
    picks_table.add_column("Setup Hypothesis", width=40)
    picks_table.add_column("Confidence", width=10)
    picks_table.add_column("Run Analysis?", width=13, justify="center")

    for sel in picker_result.top_picks:
        conf_color = {"high": "green", "medium": "yellow", "low": "dim"}.get(sel.confidence, "white")
        picks_table.add_row(
            str(sel.rank),
            sel.ticker,
            sel.setup_hypothesis[:38],
            f"[{conf_color}]{sel.confidence}[/{conf_color}]",
            "✓" if run_analysis_flag else "-",
        )
    console.print(picks_table)

    # ── 7. Optional full pipeline run ────────────────────────────────────
    if not run_analysis_flag:
        console.print(
            "\n[dim]Tip: add --run-analysis to pipe each pick through the full analysis pipeline.[/dim]"
        )
        return

    # Scan-gate check: warn if open risk is saturated (best-effort)
    console.print()
    console.print(Rule("Running Full Analysis on Each Pick", style="bold yellow"))

    tickers_to_run = [sel.ticker for sel in picker_result.top_picks]
    analysis_date = datetime.datetime.now().strftime("%Y-%m-%d")

    for ticker in tickers_to_run:
        console.print(f"\n[bold cyan]━━ Analyzing {ticker} ━━[/bold cyan]")

        # Reuse the full run_analysis flow with a minimal config
        config = DEFAULT_CONFIG.copy()
        if llm_provider:
            config["llm_provider"] = llm_provider
        if backend_url:
            config["backend_url"] = backend_url

        try:
            from tradingagents.graph.trading_graph import TradingAgentsGraph
            from tradingagents.graph.analyst_execution import (
                build_analyst_execution_plan,
                get_initial_analyst_node,
            )
            from tradingagents.dataflows.symbol_utils import detect_asset_type as _detect

            analyst_keys = ["market", "news", "fundamentals"]
            plan = build_analyst_execution_plan(analyst_keys, concurrency_limit=1)
            graph = TradingAgentsGraph(analyst_keys, config=config, debug=False)

            instrument_context = graph.resolve_instrument_context(ticker, "stock")
            init_state = graph.propagator.create_initial_state(
                ticker, analysis_date,
                asset_type="stock",
                instrument_context=instrument_context,
            )
            args = graph.propagator.get_graph_args()

            final_state: dict = {}
            with console.status(f"[cyan]Analysis running for {ticker}...[/cyan]"):
                for chunk in graph.graph.stream(init_state, **args):
                    final_state.update(chunk)

            decision = graph.process_signal(final_state.get("final_trade_decision", ""))
            console.print(Panel(
                Markdown(final_state.get("final_trade_decision", "No decision")),
                title=f"{ticker} — Final Decision",
                border_style="blue",
                padding=(1, 2),
            ))
        except Exception as exc:
            console.print(f"[red]Analysis failed for {ticker}: {exc}[/red]")


@app.command()
def scan(
    universe: str = typer.Option(
        "dynamic",
        "--universe", "-u",
        help=(
            "Stock universe to scan. "
            "'dynamic' (default) auto-builds from NSE EQUITY_L.csv (~400–600 liquid EQ stocks). "
            "Static overrides: nifty50, nifty_next50, nifty200, midcap150, nifty500."
        ),
    ),
    top: int = typer.Option(
        5,
        "--top", "-n",
        help="Number of top picks to select (1–5).",
    ),
    run_analysis: bool = typer.Option(
        False,
        "--run-analysis",
        help="After picking, run the full analysis pipeline on each pick sequentially.",
    ),
    min_liquidity: float = typer.Option(
        10.0,
        "--min-liquidity",
        help="Minimum 20-day median traded value in crores (₹ Cr). Default 10.",
    ),
    llm_provider: Optional[str] = typer.Option(
        None,
        "--llm-provider",
        help="LLM provider for the picker agent (openai, anthropic, google, etc.). "
             "Defaults to TRADINGAGENTS_LLM_PROVIDER env var or config.",
    ),
    backend_url: Optional[str] = typer.Option(
        None,
        "--backend-url",
        help="Optional backend URL override for the LLM provider.",
    ),
):
    """Scan a stock universe, rank with AI, and optionally run full analysis on top picks.

    Examples:
        python -m cli.main scan                            # dynamic universe, top 5
        python -m cli.main scan --universe nifty200 --top 3
        python -m cli.main scan --top 3 --run-analysis
    """
    # Display scan header
    console.print(Panel(
        "[bold green]TradingAgents — Stock Universe Scanner[/bold green]\n"
        "[dim]Phase 3: Quantitative screen → LLM ranking → top-N picks[/dim]",
        border_style="green",
        padding=(1, 2),
    ))

    # Resolve LLM provider from arg or environment
    effective_provider = llm_provider or DEFAULT_CONFIG.get("llm_provider", "openai")
    effective_backend = backend_url or DEFAULT_CONFIG.get("backend_url")

    # Ensure API key is present when using an interactive session
    if not os.environ.get("TRADINGAGENTS_LLM_PROVIDER"):
        try:
            from cli.utils import ensure_api_key
            ensure_api_key(effective_provider)
        except Exception:
            pass  # non-interactive mode; key assumed to be in env

    # Clamp top to [1, 5]
    top = max(1, min(5, top))

    _run_scan(
        universe=universe,
        top=top,
        run_analysis_flag=run_analysis,
        min_liquidity_cr=min_liquidity,
        llm_provider=effective_provider,
        backend_url=effective_backend,
    )


@app.command()
def universe(
    action: str = typer.Argument(
        "show",
        help="Action: 'refresh' to force-rebuild the dynamic universe, 'show' to inspect it.",
    ),
):
    """Manage the dynamic stock universe cache.

    Examples:
        python -m cli.main universe show      # show current cached universe metadata
        python -m cli.main universe refresh   # force-rebuild from NSE EQUITY_L.csv
    """
    from tradingagents.picker.universe import (
        universe_info,
        rebuild_universe,
        NSEUnavailableError,
        _load_cache,
    )

    if action == "show":
        info = universe_info()
        status_color = "green" if info["status"] == "fresh" else ("yellow" if info["status"] == "stale" else "red")
        console.print(Panel(
            f"[bold]Dynamic Universe Status[/bold]\n"
            f"Status:    [{status_color}]{info['status']}[/{status_color}]\n"
            f"Count:     {info['count']} stocks\n"
            f"Built on:  {info['built_on'] or 'not built yet'}\n"
            f"Age:       {info.get('age_days', 'N/A')} days (max {info.get('max_age_days', 7)})\n"
            f"Cache:     {info['cache_path']}",
            title="Universe Cache",
            border_style="cyan",
        ))
        cache = _load_cache()
        if cache and cache.stocks:
            tbl = Table(box=box.SIMPLE_HEAD, header_style="bold cyan", show_footer=False)
            tbl.add_column("Symbol", width=16)
            tbl.add_column("Name", width=30)
            tbl.add_column("Sector", width=20)
            for s in cache.stocks[:30]:
                tbl.add_row(s.symbol, s.name[:28], s.sector)
            if len(cache.stocks) > 30:
                tbl.add_row("...", f"...and {len(cache.stocks) - 30} more", "")
            console.print(tbl)

    elif action == "refresh":
        console.print("[bold cyan]Rebuilding dynamic universe from NSE EQUITY_L.csv...[/bold cyan]")
        console.print("[dim]This downloads ~2,000 NSE-listed stocks and applies eligibility filters.")
        console.print("[dim]May take 3–5 minutes on first run (batched yfinance downloads).[/dim]")
        try:
            with console.status("[cyan]Downloading and filtering...[/cyan]"):
                cache = rebuild_universe(DEFAULT_CONFIG)
            console.print(
                f"[green]Done — {len(cache.stocks)} stocks cached (built {cache.built_on})[/green]"
            )
        except NSEUnavailableError as exc:
            console.print(f"[red]NSE unavailable: {exc}[/red]")
            console.print("[yellow]Try again later or use --universe nifty200 as a fallback.[/yellow]")
            raise typer.Exit(code=1)
    else:
        console.print(f"[red]Unknown action '{action}'. Use 'show' or 'refresh'.[/red]")
        raise typer.Exit(code=1)


@app.command()
def analyze(
    checkpoint: bool = typer.Option(
        False,
        "--checkpoint",
        help="Enable checkpoint/resume: save state after each node so a crashed run can resume.",
    ),
    clear_checkpoints: bool = typer.Option(
        False,
        "--clear-checkpoints",
        help="Delete all saved checkpoints before running (force fresh start).",
    ),
):
    if clear_checkpoints:
        from tradingagents.graph.checkpointer import clear_all_checkpoints
        n = clear_all_checkpoints(DEFAULT_CONFIG["data_cache_dir"])
        console.print(f"[yellow]Cleared {n} checkpoint(s).[/yellow]")
    run_analysis(checkpoint=checkpoint)


# ---------------------------------------------------------------------------
# positions sub-commands (Phase 4 — position store & management mode)
# ---------------------------------------------------------------------------

positions_app = typer.Typer(
    name="positions",
    help="Manage open swing-trade positions and run position reviews.",
    no_args_is_help=True,
)
app.add_typer(positions_app, name="positions")


def _get_store():
    from tradingagents.portfolio.store import PositionStore
    return PositionStore()


def _print_positions_table(positions: list[dict], title: str = "Positions") -> None:
    """Render a compact positions table to the console."""
    tbl = Table(
        title=title,
        box=box.SIMPLE_HEAD,
        header_style="bold magenta",
        show_footer=False,
    )
    tbl.add_column("ID", width=4, justify="right")
    tbl.add_column("Ticker", width=12)
    tbl.add_column("Dir", width=6)
    tbl.add_column("Status", width=7)
    tbl.add_column("Opened", width=11)
    tbl.add_column("Avg Entry ₹", width=12, justify="right")
    tbl.add_column("Qty", width=6, justify="right")
    tbl.add_column("SL ₹", width=10, justify="right")
    tbl.add_column("Last Review", width=12)
    tbl.add_column("R (closed)", width=10, justify="right")

    for pos in positions:
        status = pos.get("status", "")
        status_color = "green" if status == "OPEN" else "dim"
        avg_entry = pos.get("avg_entry")
        sl = pos.get("stop_loss")
        r_val = pos.get("realized_r")

        tbl.add_row(
            str(pos.get("id", "")),
            pos.get("ticker", ""),
            pos.get("direction", ""),
            f"[{status_color}]{status}[/{status_color}]",
            pos.get("opened_date", "")[:10],
            f"{avg_entry:,.0f}" if avg_entry else "—",
            str(pos.get("qty_open") or "—"),
            f"{sl:,.0f}" if sl else "—",
            (pos.get("last_review_date") or "")[:10] or "—",
            f"{r_val:+.2f}R" if r_val is not None else "—",
        )
    console.print(tbl)


@positions_app.command("list")
def positions_list(
    status: str = typer.Option(
        "open",
        "--status", "-s",
        help="Filter by status: open, closed, all.",
    ),
):
    """List all positions."""
    store = _get_store()
    filter_status = None if status.lower() == "all" else status.upper()
    positions = store.list_positions(status=filter_status)
    if not positions:
        console.print(f"[yellow]No {status} positions found.[/yellow]")
        return
    _print_positions_table(positions, title=f"{status.upper()} Positions")


@positions_app.command("add")
def positions_add(
    ticker: Optional[str] = typer.Option(None, "--ticker", "-t", help="Ticker symbol (e.g. RELIANCE.NS)."),
    direction: Optional[str] = typer.Option(None, "--direction", "-d", help="LONG or SHORT."),
    from_last_signal: bool = typer.Option(False, "--from-last-signal", help="Log last signal from most recent run."),
):
    """Add a new position to the store.

    Use --from-last-signal to log the signal from the most recent analysis run,
    or provide --ticker and --direction for interactive entry.
    """
    store = _get_store()

    if from_last_signal:
        # Try to find the most recent signal file
        import json as _json
        results_dir = Path(DEFAULT_CONFIG["results_dir"])
        signal_files = sorted(results_dir.rglob("full_states_log_*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
        if not signal_files:
            console.print("[red]No recent analysis runs found. Run 'analyze' or 'scan --run-analysis' first.[/red]")
            raise typer.Exit(code=1)
        latest = signal_files[0]
        try:
            state = _json.loads(latest.read_text(encoding="utf-8"))
        except Exception as exc:
            console.print(f"[red]Could not read signal file: {exc}[/red]")
            raise typer.Exit(code=1)

        ticker_from_state = state.get("company_of_interest", "")
        trade_signal = state.get("trade_signal") or {}
        direction_from_signal = (trade_signal.get("direction") or "").upper()

        if direction_from_signal == "NO_TRADE" or not direction_from_signal:
            console.print(f"[yellow]Last signal for {ticker_from_state} was NO_TRADE. Nothing to add.[/yellow]")
            raise typer.Exit(code=0)

        console.print(Panel(
            f"[bold]Ticker:[/bold] {ticker_from_state}\n"
            f"[bold]Direction:[/bold] {direction_from_signal}\n"
            f"[bold]Setup:[/bold] {trade_signal.get('setup_type', '')}\n"
            f"[bold]Stop Loss:[/bold] ₹{trade_signal.get('stop_loss', 'N/A')}\n"
            f"[bold]Signal source:[/bold] {latest.name}",
            title="Last Signal",
            border_style="cyan",
        ))
        confirm = typer.confirm("Log this position?", default=True)
        if not confirm:
            raise typer.Exit(code=0)

        pos_id = store.add_position(
            ticker=ticker_from_state,
            direction=direction_from_signal,
            signal=trade_signal,
            notes="Added from last analysis run",
        )
        console.print(f"\n[green]✓ Position #{pos_id} created.[/green]")
        console.print("[dim]Run 'positions update --fill {pos_id}' once orders are filled.[/dim]")
        return

    # Interactive entry
    ticker = ticker or typer.prompt("Ticker (e.g. RELIANCE.NS)")
    direction = direction or typer.prompt("Direction (LONG/SHORT)").upper()
    if direction not in ("LONG", "SHORT"):
        console.print("[red]Direction must be LONG or SHORT.[/red]")
        raise typer.Exit(code=1)

    stop_loss_str = typer.prompt("Stop Loss price ₹ (optional, press Enter to skip)", default="")
    sl = float(stop_loss_str) if stop_loss_str.strip() else None
    setup_type = typer.prompt("Setup type (e.g. breakout-retest)", default="")
    notes = typer.prompt("Notes (optional)", default="")

    signal: dict = {"direction": direction, "setup_type": setup_type, "stop_loss": sl, "entries": [], "take_profits": []}
    pos_id = store.add_position(ticker=ticker, direction=direction, signal=signal, notes=notes)
    console.print(f"\n[green]✓ Position #{pos_id} created.[/green]")


@positions_app.command("update")
def positions_update(
    position_id: int = typer.Argument(..., help="Position ID to update."),
    fill: bool = typer.Option(False, "--fill", help="Record a fill (avg entry price and qty)."),
    exit_partial: bool = typer.Option(False, "--exit", help="Record a partial exit."),
    move_sl: bool = typer.Option(False, "--move-sl", help="Move the stop-loss (tighten only)."),
):
    """Record a fill, partial exit, or SL move on a position."""
    store = _get_store()
    pos = store.get_position(position_id)
    if pos is None:
        console.print(f"[red]Position {position_id} not found.[/red]")
        raise typer.Exit(code=1)

    if fill:
        avg_entry = float(typer.prompt("Avg entry price ₹"))
        qty = int(typer.prompt("Quantity filled (shares)"))
        store.record_fill(position_id, avg_entry=avg_entry, qty=qty)
        console.print(f"[green]✓ Fill recorded: {qty} shares @ ₹{avg_entry:,.2f}[/green]")

    elif exit_partial:
        qty_exited = int(typer.prompt("Qty to exit"))
        exit_price = float(typer.prompt("Exit price ₹"))
        label = typer.prompt("Label (e.g. TP1)", default="")
        store.record_partial_exit(position_id, qty_exited=qty_exited, exit_price=exit_price, label=label)
        console.print(f"[green]✓ Partial exit recorded: {qty_exited} shares @ ₹{exit_price:,.2f}[/green]")

    elif move_sl:
        current_sl = pos.get("stop_loss")
        direction = pos.get("direction", "LONG")
        console.print(f"[dim]Current SL: ₹{current_sl:,.2f} | Direction: {direction}[/dim]")
        new_sl = float(typer.prompt("New stop-loss price ₹"))

        # Enforce tighten-only rule here too
        if current_sl is not None:
            if direction == "LONG" and new_sl <= current_sl:
                console.print("[red]Error: For a LONG position, new SL must be ABOVE the current SL (tighten only).[/red]")
                raise typer.Exit(code=1)
            if direction == "SHORT" and new_sl >= current_sl:
                console.print("[red]Error: For a SHORT position, new SL must be BELOW the current SL (tighten only).[/red]")
                raise typer.Exit(code=1)

        store.move_stop_loss(position_id, new_sl)
        console.print(f"[green]✓ SL moved from ₹{current_sl:,.2f} → ₹{new_sl:,.2f}[/green]")

    else:
        console.print("[yellow]Specify one of --fill, --exit, or --move-sl.[/yellow]")


@positions_app.command("close")
def positions_close(
    position_id: int = typer.Argument(..., help="Position ID to close."),
    price: Optional[float] = typer.Option(None, "--price", help="Final exit price ₹."),
    date: Optional[str] = typer.Option(None, "--date", help="Close date (YYYY-MM-DD). Defaults to today."),
):
    """Close a position and compute realized R-multiple.

    Feeds the realized R into the reflection layer so the memory log learns
    from actual outcomes (Phase 4.3 outcome feedback loop).
    """
    store = _get_store()
    pos = store.get_position(position_id)
    if pos is None:
        console.print(f"[red]Position {position_id} not found.[/red]")
        raise typer.Exit(code=1)
    if pos.get("status") == "CLOSED":
        console.print(f"[yellow]Position {position_id} is already closed.[/yellow]")
        raise typer.Exit(code=0)

    exit_price = price or float(typer.prompt("Exit price ₹"))
    close_date = date or datetime.datetime.now().strftime("%Y-%m-%d")

    # Compute realized R
    avg_entry = pos.get("avg_entry")
    sl = pos.get("stop_loss")
    direction = pos.get("direction", "LONG")
    realized_r = 0.0
    if avg_entry and sl:
        sign = 1 if direction == "LONG" else -1
        risk_per_share = sign * (avg_entry - sl)
        if risk_per_share > 0:
            realized_r = sign * (exit_price - avg_entry) / risk_per_share

    console.print(f"\n[bold]Closing position #{position_id}:[/bold]")
    console.print(f"  {pos['ticker']} {direction}: avg entry ₹{avg_entry or '?'} → exit ₹{exit_price:,.2f}")
    console.print(f"  Realized R: [{'green' if realized_r >= 0 else 'red'}]{realized_r:+.2f}R[/]")
    confirm = typer.confirm("Confirm close?", default=True)
    if not confirm:
        raise typer.Exit(code=0)

    store.close_position(position_id, exit_price=exit_price, realized_r=realized_r, closed_date=close_date)
    console.print(f"[green]✓ Position #{position_id} closed. Realized R: {realized_r:+.2f}[/green]")

    # Feed realized R into the reflection layer (Phase 4.3)
    ticker = pos.get("ticker", "")
    final_decision_summary = (
        f"Position closed: {ticker} {direction} | "
        f"Avg entry ₹{avg_entry} | Exit ₹{exit_price:.2f} | "
        f"Realized {realized_r:+.2f}R"
    )
    console.print("[dim]Storing outcome in memory log for future reflection...[/dim]")
    try:
        from tradingagents.agents.utils.memory import TradingMemoryLog
        memory_log = TradingMemoryLog(DEFAULT_CONFIG)
        memory_log.store_decision(
            ticker=ticker,
            trade_date=close_date,
            final_trade_decision=final_decision_summary,
        )
        console.print("[dim]Memory log updated.[/dim]")
    except Exception as exc:
        console.print(f"[yellow]Memory log update failed (non-critical): {exc}[/yellow]")


@positions_app.command("review")
def positions_review(
    ticker: Optional[str] = typer.Argument(None, help="Ticker to review (e.g. RELIANCE.NS). Omit with --all."),
    all_positions: bool = typer.Option(False, "--all", "-a", help="Review all open positions."),
    llm_provider: Optional[str] = typer.Option(
        None, "--llm-provider", help="LLM provider (openai, anthropic, google, etc.)."
    ),
    backend_url: Optional[str] = typer.Option(None, "--backend-url", help="Backend URL override."),
    analysts: str = typer.Option(
        "market,news",
        "--analysts",
        help="Comma-separated analyst list to run (market,news,fundamentals,social).",
    ),
    save: bool = typer.Option(False, "--save", help="Save review to position_events automatically."),
):
    """Review open position(s) using the manage_position pipeline.

    This is your morning routine: run 'positions review --all' to get a
    one-line verdict table for every open position.

    Example:
        positions review RELIANCE.NS
        positions review --all --save
    """
    store = _get_store()

    if all_positions:
        open_positions = store.list_positions(status="OPEN")
        if not open_positions:
            console.print("[yellow]No open positions to review.[/yellow]")
            return
    elif ticker:
        pos = store.get_open_position(ticker.upper())
        if pos is None:
            console.print(f"[yellow]No open position found for {ticker.upper()}.[/yellow]")
            return
        open_positions = [pos]
    else:
        console.print("[red]Provide a TICKER or use --all.[/red]")
        raise typer.Exit(code=1)

    # Validate analyst selection
    valid_analysts = {"market", "news", "fundamentals", "social"}
    selected = [a.strip().lower() for a in analysts.split(",") if a.strip()]
    invalid = [a for a in selected if a not in valid_analysts]
    if invalid:
        console.print(f"[red]Unknown analysts: {', '.join(invalid)}. Valid: {', '.join(valid_analysts)}[/red]")
        raise typer.Exit(code=1)

    # Resolve LLM provider
    effective_provider = llm_provider or DEFAULT_CONFIG.get("llm_provider", "openai")
    effective_backend = backend_url or DEFAULT_CONFIG.get("backend_url")
    if not os.environ.get("TRADINGAGENTS_LLM_PROVIDER"):
        try:
            ensure_api_key(effective_provider)
        except Exception:
            pass

    # Build config
    config = DEFAULT_CONFIG.copy()
    config["llm_provider"] = effective_provider
    if effective_backend:
        config["backend_url"] = effective_backend
    config["max_debate_rounds"] = 1
    config["max_risk_discuss_rounds"] = 1

    # Create graph once (reused across all positions)
    from tradingagents.graph.trading_graph import TradingAgentsGraph
    graph = TradingAgentsGraph(
        selected_analysts=selected,
        config=config,
        debug=False,
    )

    analysis_date = datetime.datetime.now().strftime("%Y-%m-%d")
    verdict_rows = []

    for pos in open_positions:
        ticker_sym = pos.get("ticker", "UNKNOWN")
        pos_id = pos.get("id")
        console.print(f"\n[bold cyan]━━ Reviewing {ticker_sym} (position #{pos_id}) ━━[/bold cyan]")

        with console.status(f"[cyan]Running manage_position pipeline for {ticker_sym}...[/cyan]"):
            try:
                final_state, action_dict = graph.run_manage_position(
                    position=pos,
                    trade_date=analysis_date,
                    selected_analysts=tuple(selected),
                )
            except Exception as exc:
                console.print(f"[red]Review failed for {ticker_sym}: {exc}[/red]")
                verdict_rows.append((ticker_sym, "ERROR", "—", str(exc)[:60]))
                continue

        # Display the position action
        action = action_dict or {}
        action_str = action.get("action", "UNKNOWN")
        thesis = action.get("thesis_status", "—")
        reasoning = action.get("reasoning", "")

        color = {
            "HOLD": "yellow",
            "EXIT_FULL": "red",
            "EXIT_PARTIAL": "red",
            "RAISE_SL": "cyan",
            "ADD": "green",
            "TAKE_TP_EARLY": "magenta",
        }.get(action_str, "white")

        console.print(Panel(
            Markdown(final_state.get("final_trade_decision", "No decision")),
            title=f"{ticker_sym} — Review Decision",
            border_style="blue",
            padding=(1, 2),
        ))

        # Save review to position_events if requested
        if save and action_dict and pos_id:
            store.record_review(
                position_id=pos_id,
                review_date=analysis_date,
                action=action_str,
                reasoning=reasoning,
                thesis_status=thesis,
                position_action_dict=action_dict,
            )
            console.print(f"[dim]Review saved to position_events for #{pos_id}.[/dim]")

        verdict_rows.append((ticker_sym, action_str, thesis, reasoning[:60] + "…" if len(reasoning) > 60 else reasoning))

    # Print summary verdict table for --all
    if all_positions and verdict_rows:
        console.print()
        console.print(Rule("Morning Review Summary", style="bold green"))
        summary_tbl = Table(box=box.SIMPLE_HEAD, header_style="bold magenta")
        summary_tbl.add_column("Ticker", width=14)
        summary_tbl.add_column("Action", width=14)
        summary_tbl.add_column("Thesis", width=10)
        summary_tbl.add_column("Reasoning", width=60)
        for t_sym, act, thesis_s, reason_s in verdict_rows:
            act_color = {
                "HOLD": "yellow", "EXIT_FULL": "red", "EXIT_PARTIAL": "red",
                "RAISE_SL": "cyan", "ADD": "green", "TAKE_TP_EARLY": "magenta",
                "ERROR": "dim",
            }.get(act, "white")
            summary_tbl.add_row(t_sym, f"[{act_color}]{act}[/{act_color}]", thesis_s, reason_s)
        console.print(summary_tbl)
        if not save:
            console.print("[dim]Tip: add --save to persist reviews to the position store.[/dim]")


if __name__ == "__main__":
    app()
