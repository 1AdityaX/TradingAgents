---
name: project-india-edition
description: TradingAgents India Edition upgrade — UPGRADE_PLAN.md phase status and key design decisions
metadata:
  type: project
---

Phase 1 (Indian market data layer) is **complete** as of 2026-06-10.

**What was built:**
- `tradingagents/dataflows/india/` — 8 files: `__init__`, `market_calendar`, `nse_client`, `flows`, `shareholding`, `corporate_actions`, `screener_fundamentals`, `india_news`
- NSE holiday list ships as static set in `market_calendar.py`; F&O monthly expiry = last Thursday (adjusted for holidays)
- NSE client wraps public JSON API with browser UA + retry + disk cache (4hr TTL); yfinance fallback on block
- FII/DII flows from NSE `/api/fiidiiTradeReact`; fallback rows say "data unavailable" per session
- Shareholding from yfinance; promoter pledge always "data unavailable" (not in yfinance — requires NSE filings)
- Corporate actions flag any earnings/ex-div dates inside the 20-day swing window as EVENT RISK
- screener_fundamentals computes ROCE approx from yfinance; shows in Crores (÷1e7)
- india_news uses RSS-only (no scraping): MoneyControl, ET Markets, Business Standard, LiveMint

**Routing wired in `interface.py`:**
- `market_profile: "auto"` detects `.NS`/`.BO` → India; explicit `"india"` forces India for all calls
- `fundamental_data` → `india_composite` and `news_data` → `india_news` for Indian tickers
- `get_global_news` also routes to India feeds when `market_profile == "india"` (no ticker needed)

**`default_config.py` additions:**
- `market_profile: "auto"` (env: `TRADINGAGENTS_MARKET_PROFILE`)
- `global_news_queries_india` — 7 RBI/Nifty/FII/SEBI/INR/monsoon queries
- `india_reddit_subreddits` — IndianStockMarket, IndiaInvestments, DalalStreetTalks

**`symbol_utils.py` additions:**
- `is_indian_ticker()`, `normalize_indian_symbol()` (bare RELIANCE → RELIANCE.NS), `detect_market_profile()`

**`agent_utils.py` — India instrument context block:**
- For any `.NS`/`.BO` ticker, `build_instrument_context()` appends an INDIA block:  currency ₹, market hours IST, F&O ban status, circuit bands, T+1 settlement, STCG note, explicit ₹-not-$ warning

**`sentiment_analyst.py`:**
- For `.NS`/`.BO` tickers, Reddit uses `india_reddit_subreddits` instead of wallstreetbets/stocks/investing

**Tests:** 108 unit tests in 4 files (`test_india_market_calendar`, `test_india_symbol_utils`, `test_india_dataflows`, `test_india_instrument_context`) — all passing, no regressions in existing suite.

**Why:** Phases 2+ (signal engine, stock picker, position store, prompt rewrites, backtest) depend on this data layer.

**How to apply:** When working on Phases 2–8 from UPGRADE_PLAN.md, assume Phase 1 modules exist at the paths above. Phase 2 (TradeSignal schema + position_sizing.py + Signal Validator node) is next.
