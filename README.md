<p align="center">
  <img src="assets/TauricResearch.png" style="width: 60%; height: auto;">
</p>

<div align="center" style="line-height: 1;">
  <a href="https://arxiv.org/abs/2412.20138" target="_blank"><img alt="arXiv" src="https://img.shields.io/badge/arXiv-2412.20138-B31B1B?logo=arxiv"/></a>
  <a href="https://discord.com/invite/hk9PGKShPK" target="_blank"><img alt="Discord" src="https://img.shields.io/badge/Discord-TradingResearch-7289da?logo=discord&logoColor=white&color=7289da"/></a>
  <a href="./assets/wechat.png" target="_blank"><img alt="WeChat" src="https://img.shields.io/badge/WeChat-TauricResearch-brightgreen?logo=wechat&logoColor=white"/></a>
  <a href="https://x.com/TauricResearch" target="_blank"><img alt="X Follow" src="https://img.shields.io/badge/X-TauricResearch-white?logo=x&logoColor=white"/></a>
  <br>
  <a href="https://github.com/TauricResearch/" target="_blank"><img alt="Community" src="https://img.shields.io/badge/Join_GitHub_Community-TauricResearch-14C290?logo=discourse"/></a>
</div>

---

# TradingAgents — India Edition

A swing-trading research system for NSE/BSE built on top of the TauricResearch multi-agent LangGraph pipeline. The base framework provides the agent graph (analysts → debate → trader → risk → portfolio manager), multi-provider LLM support, structured outputs, and memory/reflection. This fork adds eight layers on top of that foundation, tailored specifically to Indian markets.

> **Research tool only.** Signals require your judgment before execution. Nothing here is financial advice.

<div align="center">

[Installation](#installation) · [CLI Commands](#cli-commands) · [Configuration](#configuration) · [Python API](#python-api) · [Architecture](#architecture) · [Citation](#citation)

</div>

---

## What's in this fork

### Base pipeline (from TauricResearch v0.2.5)

The LangGraph graph runs five stages in sequence:

1. **Analyst Team** — Market (technical), Fundamentals, News, and Sentiment analysts run in parallel or series. Each produces a structured report.
2. **Research Team** — Bull Researcher and Bear Researcher debate the analysts' findings; the Research Manager adjudicates.
3. **Trader** — Reads all reports and produces a `TradeSignal` (LONG/SHORT/NO_TRADE) with entry levels, stop-loss, and take-profit targets.
4. **Risk Management** — Aggressive, Neutral, and Conservative risk analysts each apply a focused checklist to the Trader's proposal.
5. **Portfolio Manager** — Approves, modifies, or rejects the signal.

Multi-provider LLM support: OpenAI, Anthropic, Google Gemini, xAI Grok, DeepSeek, Qwen (international + China), GLM, MiniMax, OpenRouter, Ollama, Azure OpenAI.

### India-specific additions

| Phase | What was added |
|---|---|
| **1 — India data layer** | `dataflows/india/`: NSE client, FII/DII flows, shareholding/pledge, corporate actions, Indian news (RSS), market calendar. Auto-detected from `.NS`/`.BO` suffix. |
| **2 — Signal engine** | `TradeSignal` schema (entries EP1/EP2, SL with structural basis, TP ladder), `position_sizing.py` (deterministic, no LLM), Signal Validator node. |
| **3 — Stock picker** | Dynamic NSE universe (~400–600 EQ stocks), quantitative screener (Stage 1–2), LLM picker (Stage 3, 1 call), `scan` CLI command. |
| **4 — Position store** | SQLite at `~/.tradingagents/portfolio.db`, manage-position mode in the graph, `positions` CLI sub-commands, outcome feedback to the reflection layer. |
| **5 — Prompt precision** | India preamble injected into every agent, structured levels table from Market Analyst, Indian checklists for Fundamentals/News/Risk agents, few-shot anchors. |
| **6 — Hardening** | Signal Validator unit tests, data freshness guards, NSE client retry/cache, ATR-based SL computation. |
| **7 — Operating cadence** | `routine` CLI command, scan-gating (blocks scan when fully invested), review reminders. |
| **8 — Small-account mode** | `account_profile: "small"` preset, light pipeline, monthly LLM budget tracking (`budget/spend_tracker.py`), weekly scan cadence, first-run framing. |

---

## Installation

```bash
git clone <your-fork-url>
cd TradingAgents
conda create -n tradingagents python=3.13
conda activate tradingagents
pip install .
```

Or with Docker:
```bash
cp .env.example .env   # fill in your API keys
docker compose run --rm tradingagents
```

### API keys

Set the key for whichever LLM provider you use:

```bash
export OPENAI_API_KEY=...
export ANTHROPIC_API_KEY=...
export GOOGLE_API_KEY=...
export XAI_API_KEY=...
export DEEPSEEK_API_KEY=...
export DASHSCOPE_API_KEY=...        # Qwen — international
export DASHSCOPE_CN_API_KEY=...     # Qwen — China
export ZHIPU_API_KEY=...            # GLM — international
export ZHIPU_CN_API_KEY=...         # GLM — China
export MINIMAX_API_KEY=...          # MiniMax — global
export MINIMAX_CN_API_KEY=...       # MiniMax — China
export OPENROUTER_API_KEY=...
export ALPHA_VANTAGE_API_KEY=...
```

For enterprise providers (Azure OpenAI, AWS Bedrock), copy `.env.enterprise.example` to `.env.enterprise`.

---

## CLI Commands

All commands are available as `tradingagents <command>` (installed entry point) or `python -m cli.main <command>`.

### `analyze` — deep analysis on a single ticker

```bash
tradingagents analyze
tradingagents analyze --checkpoint          # enable checkpoint/resume
tradingagents analyze --clear-checkpoints   # reset checkpoints, then run
```

Interactive prompts walk you through: ticker, analysis date, output language, analyst selection, research depth, LLM provider, model selection, and provider-specific thinking configuration. All prompts are skipped when the corresponding `TRADINGAGENTS_*` environment variable is set, making it fully non-interactive in CI.

Supported tickers include anything yfinance covers: US stocks (`AAPL`, `SPY`), Indian (`RELIANCE.NS`, `TCS.NS`, `INFY.BO`), Hong Kong (`0700.HK`), Japan (`7203.T`), London (`AZN.L`), China A-shares (`600519.SS`), crypto (`BTC-USD`), and more.

Reports are saved under `~/.tradingagents/logs/<TICKER>/<DATE>/` with per-section markdown files and a consolidated `complete_report.md`.

### `scan` — stock universe scanner

```bash
tradingagents scan                              # dynamic universe, top 5 picks
tradingagents scan --universe nifty200 --top 3
tradingagents scan --top 3 --run-analysis       # pipe picks into full analysis
tradingagents scan --min-liquidity 25           # higher liquidity floor (₹25 Cr)
```

The scan runs a three-stage funnel:

1. **Quantitative screen** (`tradingagents/picker/screener.py`) — liquidity filter, price band, F&O ban check, composite technical score (RSI, relative strength vs Nifty, volume surge, distance from 52-week high).
2. **Candidate card enrichment** (`tradingagents/picker/candidate.py`) — attaches next results date, promoter pledge flag, and 3 latest headlines to each top-25 candidate.
3. **LLM ranking** (`tradingagents/picker/picker_agent.py`) — one LLM call ranks all 25 candidates and returns the top N with a setup hypothesis per pick.

Universe options: `dynamic` (default — auto-rebuilt from NSE `EQUITY_L.csv`, ~400–600 liquid EQ stocks), `nifty50`, `nifty_next50`, `nifty200`, `midcap150`, `nifty500` (static CSV fallbacks).

The scan is **blocked** automatically when open portfolio risk has reached the cap — the CLI prints a "fully invested" notice and exits.

### `universe` — manage the dynamic stock universe cache

```bash
tradingagents universe show       # show cache metadata and first 30 stocks
tradingagents universe refresh    # force-rebuild from NSE EQUITY_L.csv
```

The dynamic universe downloads NSE's official equity master list, applies eligibility rules (EQ series only, price band, liquidity floor, not in F&O ban list), and caches the result at `~/.tradingagents/universe.parquet`. It auto-rebuilds when older than `universe_max_age_days` (default 7) — no manual refresh needed during normal use.

### `routine` — operating cadence guide

```bash
tradingagents routine          # show cadence table + current portfolio status
tradingagents routine --run    # auto-run the right command for the current time
```

Shows the recommended IST-time-aware workflow, current market phase (pre-open / open / post-close / holiday), portfolio status (open risk %, position count, unreviewed positions), and time-contextual guidance. Use `--run` to automatically execute the appropriate command (morning review before open, scan after close).

### `positions` — position store

```bash
tradingagents positions list                       # list open positions
tradingagents positions list --status all          # all positions
tradingagents positions add                        # interactive entry
tradingagents positions add --from-last-signal     # log last analysis signal
tradingagents positions update <ID> --fill         # record a fill
tradingagents positions update <ID> --exit         # record a partial exit
tradingagents positions update <ID> --move-sl      # tighten the stop-loss
tradingagents positions close <ID>                 # close + compute realized R
tradingagents positions review RELIANCE.NS         # manage-position mode
tradingagents positions review --all               # morning review all open
tradingagents positions review --all --save        # review + persist to store
```

Positions are stored in SQLite at `~/.tradingagents/portfolio.db`. The schema tracks entries, avg entry, qty, SL, TPs, thesis, last review date, and a full event log (fills, partial exits, SL moves, reviews, closes).

**Hard rules enforced in the CLI:**
- Stop-loss can only tighten (never widen) — checked on `--move-sl` and in the Signal Validator.
- `positions close` computes the realized R-multiple and feeds it into the memory log for future reflection.

The `review` command runs the graph in `manage_position` mode. The Trader produces a `PositionAction` (HOLD / EXIT_FULL / EXIT_PARTIAL / RAISE_SL / ADD / TAKE_TP_EARLY) instead of a `TradeSignal`, and the prompt is reframed: bull argues thesis intact, bear argues thesis degraded.

### `profile` — account profile and LLM budget

```bash
tradingagents profile                  # show effective config with sources
tradingagents profile set small        # persist small-account profile
tradingagents profile set standard
tradingagents profile budget           # monthly LLM spend history
```

---

## Configuration

`tradingagents/default_config.py` is the single source of truth. Precedence (highest wins): **env vars → account-profile preset → saved `~/.tradingagents/profile.json` → base defaults**.

### Key settings

```python
# LLM
"llm_provider": "openai"          # openai, google, anthropic, xai, deepseek, qwen, glm, minimax, openrouter, ollama, azure
"deep_think_llm": "gpt-5.5"
"quick_think_llm": "gpt-5.4-mini"
"temperature": None               # set e.g. 0.2 for Trader/PM; None = provider default
"max_debate_rounds": 1
"pipeline": "full"                # "full" | "light" (skips debates, ~60% fewer tokens)

# Market profile
"market_profile": "auto"          # "auto" | "india" | "us" — auto-detects from ticker suffix

# Position sizing (pure Python, no LLM)
"account_equity_inr": 1_000_000
"risk_pct_per_trade": 1.0         # % of equity risked between avg entry and SL
"max_open_risk_pct": 6.0          # hard cap on total portfolio risk
"max_position_pct": 15.0          # single position cap as % of equity
"min_risk_reward": 1.8            # minimum net RR (after transaction costs)
"txn_cost_pct_round_trip": 0.5    # brokerage + STT + DP + slippage

# Stock universe (Phase 3)
"universe": "dynamic"             # "dynamic" | "nifty50" | "nifty200" | "midcap150" | "nifty500"
"universe_max_age_days": 7

# Small-account mode (Phase 8)
"account_profile": "standard"     # "standard" | "small"
"monthly_llm_budget_inr": None    # warn at 80%, alert at 100%
"scan_cadence": "daily"           # "daily" | "weekly"
```

### Environment variable overrides

Every key in the config can be set via a `TRADINGAGENTS_*` env var without code changes:

```bash
TRADINGAGENTS_LLM_PROVIDER=anthropic
TRADINGAGENTS_DEEP_THINK_LLM=claude-opus-4-8
TRADINGAGENTS_QUICK_THINK_LLM=claude-haiku-4-5-20251001
TRADINGAGENTS_MARKET_PROFILE=india
TRADINGAGENTS_ACCOUNT_EQUITY_INR=500000
TRADINGAGENTS_RISK_PCT_PER_TRADE=1.0
TRADINGAGENTS_MIN_RISK_REWARD=1.8
TRADINGAGENTS_ACCOUNT_PROFILE=small
TRADINGAGENTS_PIPELINE=light
TRADINGAGENTS_MONTHLY_LLM_BUDGET_INR=300
TRADINGAGENTS_TEMPERATURE=0.2
```

### Small-account profile

`tradingagents profile set small` (or `TRADINGAGENTS_ACCOUNT_PROFILE=small`) activates a preset designed for accounts ≤ ₹50k:

| Setting | Standard | Small |
|---|---|---|
| `max_stock_price` | no cap | ₹500 |
| `max_position_pct` | 15% | 100% |
| `max_concurrent_positions` | unlimited | 2 |
| `risk_pct_per_trade` | 1.0% | 2.0% |
| `max_open_risk_pct` | 6.0% | 4.0% |
| `pipeline` | full | light |
| `monthly_llm_budget_inr` | no cap | ₹300 |
| `scan_cadence` | daily | weekly |

A one-time framing message is shown on the first scan/analyze run with a small account.

---

## Architecture

### Directory layout

```
tradingagents/
├── agents/
│   ├── analysts/           # market, news, fundamentals, sentiment analysts
│   ├── managers/           # portfolio manager
│   ├── researchers/        # bull, bear, research manager
│   ├── risk_mgmt/          # aggressive, neutral, conservative risk analysts
│   ├── stock_picker/       # Phase 3 picker agent node
│   ├── trader/             # trader agent (produces TradeSignal or PositionAction)
│   ├── validator/          # signal_validator.py — deterministic post-trade check
│   ├── utils/              # memory log, structured output helpers, market data validation
│   └── schemas.py          # Pydantic schemas: TradeSignal, PositionAction, PortfolioDecision, ...
├── budget/
│   └── spend_tracker.py    # monthly LLM spend tracking (Phase 8)
├── dataflows/
│   ├── india/              # nse_client, flows, shareholding, corporate_actions, india_news, market_calendar
│   ├── interface.py        # market profile router (auto-detects .NS/.BO)
│   ├── y_finance.py        # yfinance OHLCV, fundamentals, indicators
│   ├── reddit.py           # Reddit sentiment (India subreddits for .NS/.BO)
│   └── symbol_utils.py     # ticker normalisation, asset-type detection
├── graph/
│   ├── trading_graph.py    # TradingAgentsGraph: main entry point
│   ├── manage_setup.py     # manage_position mode setup
│   ├── propagation.py      # state initialisation and streaming
│   ├── reflection.py       # realized-return reflection layer
│   └── checkpointer.py     # LangGraph SQLite checkpointing
├── picker/
│   ├── universe.py         # dynamic NSE universe builder + static CSV fallback
│   ├── screener.py         # Stage 1–2 quantitative screen (no LLM)
│   ├── candidate.py        # CandidateCard builder
│   └── picker_agent.py     # Stage 3 LLM ranking (1 call)
├── portfolio/
│   └── store.py            # SQLite position store, event log
├── risk/
│   └── position_sizing.py  # deterministic position sizing calculator
└── default_config.py       # all config defaults + env-var override table
cli/
├── main.py                 # analyze, scan, universe, routine, profile, positions sub-commands
└── utils.py                # LLM provider menus, API key prompts, ticker helpers
```

### Signal flow for a new trade

```
scan
  ↓
universe.py  → ~400–600 NSE EQ stocks (lazy cache, auto-refresh weekly)
screener.py  → top 25 by composite technical score (price/liquidity/RSI/RS/volume)
candidate.py → enrich with results dates, pledge flags, headlines
picker_agent → 1 LLM call → top 3–5 with setup hypotheses
  ↓ (--run-analysis or 'analyze' directly)
TradingAgentsGraph.stream()
  ├── Market Analyst    → LEVELS table, ATR(14), verified snapshot
  ├── News Analyst      → Indian news (RSS), FII/DII flows, source+date required
  ├── Fundamentals      → yfinance + India composite (pledge, FII holding, results date)
  └── Sentiment Analyst → r/IndianStockMarket + Google News headlines
  ↓
Bull/Bear debate → Research Manager → TradeSignal draft
  ↓
Trader → TradeSignal (LONG/SHORT/NO_TRADE with EP1/EP2, SL+basis, TP1/TP2)
  ↓
Signal Validator (code, no LLM):
  - SL on correct side of entry
  - entries within ±15% of last close
  - net RR ≥ min_risk_reward after txn costs
  - qty ≥ 1 share within risk caps
  - no circuit-band violation
  On failure: one retry to Trader, then downgrade to NO_TRADE
  ↓
position_sizing.py → qty, capital ₹, risk ₹, gross/net RR
  ↓
Risk debate (Aggressive/Neutral/Conservative with distinct checklists)
  ↓
Portfolio Manager → PortfolioDecision + approved signal ticket
```

### Key schema types (`agents/schemas.py`)

**`TradeSignal`** — produced by the Trader:
- `direction`: LONG / SHORT / NO_TRADE
- `entries`: list of `EntryLevel` (label, price in ₹, allocation %, trigger, rationale)
- `stop_loss` + `stop_basis`: price anchored to a structural level
- `take_profits`: list of `TakeProfit` (label, price, exit %, basis)
- `event_risks`: list of upcoming events inside the holding window
- `confidence`: high / medium / low

**`PositionAction`** — produced in manage-position mode:
- `action`: HOLD / EXIT_FULL / EXIT_PARTIAL / RAISE_SL / ADD / TAKE_TP_EARLY
- `thesis_status`: intact / weakened / broken (broken forces an exit action — validated)
- `new_stop_loss`: for RAISE_SL (validator enforces tighten-only)
- `reasoning`: 2–4 sentences citing current R, days held, level tests, news

**`PositionSizingResult`** — from `position_sizing.py`:
- `qty`, `avg_entry_price`, `capital_inr`, `capital_pct`, `risk_inr`, `risk_pct`
- `risk_reward_gross`, `risk_reward_net` (net of round-trip txn costs)
- `affordable`, `cap_breached`, `open_risk_breached`, `rejection_reason`

### India data layer (`dataflows/india/`)

| Module | What it provides |
|---|---|
| `nse_client.py` | NSE public endpoints: quote, F&O ban list, bulk/block deals, index snapshots. Browser-UA + retry + on-disk cache. Falls back to yfinance with explicit "unavailable" strings. |
| `flows.py` | FII/DII daily net cash-market figures for the last 10 sessions. |
| `shareholding.py` | Promoter holding %, pledge %, quarter-over-quarter FII/DII holding change. |
| `corporate_actions.py` | Upcoming ex-dates, board meetings, results dates. |
| `india_news.py` | RSS feeds (MoneyControl, ET Markets, Business Standard, LiveMint) + company-name keyword filter. |
| `market_calendar.py` | NSE holiday list, 09:15–15:30 IST session, monthly F&O expiry, `market_status()` for the `routine` command. |

The market profile resolver in `dataflows/interface.py` auto-upgrades `fundamental_data` → `india_composite` and `news_data` → `india_news` when the ticker ends in `.NS` or `.BO`.

Every agent receives an **India instrument context block** with: symbol, sector, index membership, currency (INR), F&O lot size, ban-list status, circuit band, market hours, next holiday, next results date, settlement cycle (T+1), and tax footnote (STCG, STT). This block prevents an entire class of LLM errors (dollar signs, US hours, fabricated levels, ignoring circuits).

---

## Python API

```python
from tradingagents.graph.trading_graph import TradingAgentsGraph
from tradingagents.default_config import DEFAULT_CONFIG

# Standard analysis
ta = TradingAgentsGraph(debug=True, config=DEFAULT_CONFIG.copy())
_, decision = ta.propagate("RELIANCE.NS", "2026-06-10")
print(decision)

# Custom config
config = DEFAULT_CONFIG.copy()
config["llm_provider"] = "anthropic"
config["deep_think_llm"] = "claude-opus-4-8"
config["quick_think_llm"] = "claude-haiku-4-5-20251001"
config["market_profile"] = "india"
config["temperature"] = 0.2
config["max_debate_rounds"] = 1
config["pipeline"] = "light"          # skip debates for faster/cheaper runs

ta = TradingAgentsGraph(
    selected_analysts=["market", "news", "fundamentals"],
    config=config,
    debug=False,
)
_, decision = ta.propagate("TCS.NS", "2026-06-10")
```

### Position sizing (standalone)

```python
from tradingagents.risk.position_sizing import size_position
from tradingagents.agents.schemas import TradeSignal, EntryLevel, TakeProfit

signal = TradeSignal(
    direction="LONG",
    setup_type="pullback-to-50SMA",
    timeframe="swing 5–20 sessions",
    entries=[EntryLevel(label="EP1", price=2852, allocation_pct=100, trigger="limit", rationale="support retest")],
    stop_loss=2778,
    stop_basis="below swing low",
    take_profits=[TakeProfit(label="TP1", price=2960, exit_pct=50, basis="prior swing high"),
                  TakeProfit(label="TP2", price=3080, exit_pct=50, basis="weekly supply")],
    invalidation="daily close below 2750",
    confidence="medium",
)

result = size_position(
    signal=signal,
    account_equity=1_000_000,
    risk_pct_per_trade=1.0,
    open_portfolio_risk_pct=2.0,
    max_open_risk_pct=6.0,
    max_position_pct=15.0,
    min_risk_reward=1.8,
    txn_cost_pct_round_trip=0.5,
)

print(result.format_ticket("RELIANCE.NS", signal))
# SIGNAL — RELIANCE.NS — LONG — swing 5–20 sessions
# EP1 ₹2,852 (100%) limit
# SL  ₹2,778  (below swing low)
# TP1 ₹2,960 (50% off) | TP2 ₹3,080 (50% off)
# Qty 135 | Capital ₹3.85L (38.5% of equity) | Risk ₹9,990 (1.00%) | RR 1.92 (net)
# Expected ₹ P&L at TP1: ₹14,580 gross / ₹5,355 net (after ~0.5% round-trip costs)
```

---

## Persistence

### Decision log / reflection

Always on. Each completed run appends its decision to `~/.tradingagents/memory/trading_memory.md`. On the next run for the same ticker, the system fetches realized return, computes alpha vs the regional benchmark (Nifty 50 for `.NS`, Sensex for `.BO`, SPY for US tickers), generates a reflection, and injects prior decisions into the Portfolio Manager prompt. `positions close` also writes realized R-multiples into this log.

Override with `TRADINGAGENTS_MEMORY_LOG_PATH`.

### Checkpoint resume

```bash
tradingagents analyze --checkpoint         # save state after each node
tradingagents analyze --clear-checkpoints  # reset all checkpoints first
```

Per-ticker SQLite checkpoints at `~/.tradingagents/cache/checkpoints/<TICKER>.db`. Override base with `TRADINGAGENTS_CACHE_DIR`.

### Position store

SQLite at `~/.tradingagents/portfolio.db`. Tables: `positions` (one row per trade) and `position_events` (fills, partial exits, SL moves, reviews, closes). The `positions` commands manage this store; the scan gate reads from it.

---

## Operating cadence

```
After market close (15:30 IST)    tradingagents scan --top 5 [--run-analysis]
After order fills                  tradingagents positions add / update
Before market open (09:15 IST)    tradingagents positions review --all
Weekly                            tradingagents positions close <finished>
Intraday                          nothing (swing system)
Holidays / weekends               nothing (no new data)
```

Run `tradingagents routine` to see the current recommendation based on IST time and your portfolio state.

---

## Reproducibility

LLM-driven analysis is non-deterministic by design. Two identical runs can differ. To reduce variation:

```python
config["temperature"] = 0.2   # or lowest non-zero value your provider accepts
config["deep_think_llm"] = "gpt-4.1"   # non-reasoning models honor temperature better
```

What is deterministic: ticker identity resolution, price/indicator data (pinned to `analysis_date`), position sizing calculations, signal validator checks. Live data sources (news, Reddit, NSE flows) reflect the moment of the call.

---

## Contributing

Bug fixes, documentation, and feature ideas are welcome. See [`CHANGELOG.md`](CHANGELOG.md) for prior contributions.

---

## Citation

```
@misc{xiao2025tradingagentsmultiagentsllmfinancial,
      title={TradingAgents: Multi-Agents LLM Financial Trading Framework}, 
      author={Yijia Xiao and Edward Sun and Di Luo and Wei Wang},
      year={2025},
      eprint={2412.20138},
      archivePrefix={arXiv},
      primaryClass={q-fin.TR},
      url={https://arxiv.org/abs/2412.20138}, 
}
```
