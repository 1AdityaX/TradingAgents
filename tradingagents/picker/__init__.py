"""TradingAgents Stock Picker — Phase 3.

Upstream stage that runs *before* the per-ticker deep analysis pipeline:

    cheap quantitative screen → LLM ranking → top-N fed into full pipeline

Modules:
    universe   — load NSE index constituent lists from static CSVs
    screener   — quantitative pre-filter (no LLM; pandas + yfinance)
    candidate  — CandidateCard builder (compact per-stock fact sheet)
    picker_agent — one LLM call that ranks pre-filtered candidates

Cost control: screen is free; only 1 LLM call for ranking;
deep pipeline runs only on the top N.
"""

from .universe import (
    get_universe,
    load_universe,
    rebuild_universe,
    symbols_only,
    list_universes,
    universe_info,
    StockEntry,
    UniverseCache,
    NSEUnavailableError,
)
from .screener import run_screen, ScreenerScore
from .candidate import build_candidate_card, build_all_cards, format_card_for_prompt
from .picker_agent import run_picker, PickerOutput, PickerSelection, render_picker_output

__all__ = [
    # universe — dynamic
    "get_universe",
    "rebuild_universe",
    "universe_info",
    "NSEUnavailableError",
    "UniverseCache",
    # universe — static CSV
    "load_universe",
    "symbols_only",
    "list_universes",
    "StockEntry",
    # screener (Stages 1–2)
    "run_screen",
    "ScreenerScore",
    # candidate cards
    "build_candidate_card",
    "build_all_cards",
    "format_card_for_prompt",
    # picker agent (Stage 3)
    "run_picker",
    "PickerOutput",
    "PickerSelection",
    "render_picker_output",
]
