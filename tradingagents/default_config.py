import os

_TRADINGAGENTS_HOME = os.path.join(os.path.expanduser("~"), ".tradingagents")

# Single source of truth for env-var → config-key overrides. To expose
# a new config key for environment-based override, add a row here — no
# entry-point script changes required. Coercion is driven by the type
# of the existing default, so users can keep writing plain strings in
# their .env file.
_ENV_OVERRIDES = {
    "TRADINGAGENTS_LLM_PROVIDER":         "llm_provider",
    "TRADINGAGENTS_DEEP_THINK_LLM":       "deep_think_llm",
    "TRADINGAGENTS_QUICK_THINK_LLM":      "quick_think_llm",
    "TRADINGAGENTS_LLM_BACKEND_URL":      "backend_url",
    "TRADINGAGENTS_OUTPUT_LANGUAGE":      "output_language",
    "TRADINGAGENTS_MAX_DEBATE_ROUNDS":    "max_debate_rounds",
    "TRADINGAGENTS_MAX_RISK_ROUNDS":      "max_risk_discuss_rounds",
    "TRADINGAGENTS_CHECKPOINT_ENABLED":   "checkpoint_enabled",
    "TRADINGAGENTS_BENCHMARK_TICKER":     "benchmark_ticker",
    "TRADINGAGENTS_TEMPERATURE":          "temperature",
    "TRADINGAGENTS_MARKET_PROFILE":       "market_profile",
    # Phase 2 — risk / sizing overrides
    "TRADINGAGENTS_ACCOUNT_EQUITY_INR":       "account_equity_inr",
    "TRADINGAGENTS_RISK_PCT_PER_TRADE":       "risk_pct_per_trade",
    "TRADINGAGENTS_MAX_OPEN_RISK_PCT":        "max_open_risk_pct",
    "TRADINGAGENTS_MAX_POSITION_PCT":         "max_position_pct",
    "TRADINGAGENTS_MIN_RISK_REWARD":          "min_risk_reward",
    "TRADINGAGENTS_MIN_STOCK_PRICE":          "min_stock_price",
    "TRADINGAGENTS_TXN_COST_PCT_ROUND_TRIP":  "txn_cost_pct_round_trip",
    # Phase 3 — universe
    "TRADINGAGENTS_UNIVERSE":                 "universe",
    "TRADINGAGENTS_UNIVERSE_MAX_AGE_DAYS":    "universe_max_age_days",
}


def _coerce(value: str, reference):
    """Coerce env-var string to the type of the existing default value."""
    if isinstance(reference, bool):
        return value.strip().lower() in ("true", "1", "yes", "on")
    if isinstance(reference, int) and not isinstance(reference, bool):
        return int(value)
    if isinstance(reference, float):
        return float(value)
    return value


def _apply_env_overrides(config: dict) -> dict:
    """Apply TRADINGAGENTS_* env vars to the config dict in-place."""
    for env_var, key in _ENV_OVERRIDES.items():
        raw = os.environ.get(env_var)
        if raw is None or raw == "":
            continue
        config[key] = _coerce(raw, config.get(key))
    return config


DEFAULT_CONFIG = _apply_env_overrides({
    "project_dir": os.path.abspath(os.path.join(os.path.dirname(__file__), ".")),
    "results_dir": os.getenv("TRADINGAGENTS_RESULTS_DIR", os.path.join(_TRADINGAGENTS_HOME, "logs")),
    "data_cache_dir": os.getenv("TRADINGAGENTS_CACHE_DIR", os.path.join(_TRADINGAGENTS_HOME, "cache")),
    "memory_log_path": os.getenv("TRADINGAGENTS_MEMORY_LOG_PATH", os.path.join(_TRADINGAGENTS_HOME, "memory", "trading_memory.md")),
    # Optional cap on the number of resolved memory log entries. When set,
    # the oldest resolved entries are pruned once this limit is exceeded.
    # Pending entries are never pruned. None disables rotation entirely.
    "memory_log_max_entries": None,
    # LLM settings
    "llm_provider": "openai",
    "deep_think_llm": "gpt-5.5",
    "quick_think_llm": "gpt-5.4-mini",
    # When None, each provider's client falls back to its own default endpoint
    # (api.openai.com for OpenAI, generativelanguage.googleapis.com for Gemini, ...).
    # The CLI overrides this per provider when the user picks one. Keeping a
    # provider-specific URL here would leak (e.g. OpenAI's /v1 was previously
    # being forwarded to Gemini, producing malformed request URLs).
    "backend_url": None,
    # Provider-specific thinking configuration
    "google_thinking_level": None,      # "high", "minimal", etc.
    "openai_reasoning_effort": None,    # "medium", "high", "low"
    "anthropic_effort": None,           # "high", "medium", "low"
    # Sampling temperature, forwarded to every provider when set. None leaves
    # each provider at its own default. Lower values reduce run-to-run
    # variation on models that honor it; reasoning models largely ignore it
    # and no setting makes LLM output bit-identical across runs (see README).
    "temperature": None,
    # Checkpoint/resume: when True, LangGraph saves state after each node
    # so a crashed run can resume from the last successful step.
    "checkpoint_enabled": False,
    # Output language for analyst reports and final decision
    # Internal agent debate stays in English for reasoning quality
    "output_language": "English",
    # Debate and discussion settings
    "max_debate_rounds": 1,
    "max_risk_discuss_rounds": 1,
    "max_recur_limit": 100,
    "analyst_concurrency_limit": 1,
    # News / data fetching parameters
    # Increase for longer lookback strategies or to broaden macro coverage;
    # decrease to reduce token usage in agent prompts.
    "news_article_limit": 20,             # max articles per ticker (ticker-news)
    "global_news_article_limit": 10,      # max articles for global/macro news
    "global_news_lookback_days": 7,       # macro news lookback window
    # Search queries used by get_global_news for macro headlines. Extend or
    # replace to broaden geographic / sector coverage.
    "global_news_queries": [
        "Federal Reserve interest rates inflation",
        "S&P 500 earnings GDP economic outlook",
        "geopolitical risk trade war sanctions",
        "ECB Bank of England BOJ central bank policy",
        "oil commodities supply chain energy",
    ],
    # Indian macro queries — used when market_profile is "india" (or auto-detected).
    # These replace global_news_queries for Indian tickers.
    "global_news_queries_india": [
        "RBI monetary policy repo rate inflation India",
        "Nifty Sensex FII DII flows outlook",
        "India GDP IIP CPI WPI data",
        "SEBI regulation announcement",
        "INR USD rupee crude oil price impact India",
        "US Fed rates impact emerging markets India",
        "monsoon rural demand India",
    ],
    # Indian sentiment subreddits (used instead of wallstreetbets/stocks for .NS/.BO)
    "india_reddit_subreddits": [
        "IndianStockMarket",
        "IndiaInvestments",
        "DalalStreetTalks",
    ],
    # Market profile: "auto" detects from ticker suffix (.NS/.BO → india, else us).
    # Set to "india" or "us" to force a specific profile for all tickers in the run.
    "market_profile": "auto",
    # Data vendor configuration
    # Category-level configuration (default for all tools in category)
    "data_vendors": {
        "core_stock_apis": "yfinance",       # Options: alpha_vantage, yfinance
        "technical_indicators": "yfinance",  # Options: alpha_vantage, yfinance
        # For .NS/.BO tickers, "fundamental_data" is auto-upgraded to "india_composite"
        # by the market profile resolver in interface.py.
        "fundamental_data": "yfinance",      # Options: alpha_vantage, yfinance
        # For .NS/.BO tickers, "news_data" is auto-upgraded to "india_news".
        "news_data": "yfinance",             # Options: alpha_vantage, yfinance
    },
    # Tool-level configuration (takes precedence over category-level)
    "tool_vendors": {
        # Example: "get_stock_data": "alpha_vantage",  # Override category default
    },
    # Benchmark for alpha calculation in the reflection layer.
    # ``benchmark_ticker`` (when set) overrides the suffix map for all
    # tickers; leave it None to use ``benchmark_map`` for auto-detection
    # based on the ticker's exchange suffix. SPY remains the US default
    # so the reflection label keeps reading "Alpha vs SPY" for US tickers
    # while non-US tickers get their regional index automatically.
    # Phase 2 — Trade Signal engine risk parameters
    # These drive position_sizing.py (pure Python; no LLM involved).
    "account_equity_inr": 1_000_000,       # ₹10L default account size
    "risk_pct_per_trade": 1.0,             # % of equity risked between avg entry and SL
    "max_open_risk_pct": 6.0,              # hard cap on total portfolio open risk %
    "max_position_pct": 15.0,             # hard cap on single position as % of equity
    "min_risk_reward": 1.8,               # minimum acceptable net RR (after txn costs)
    "min_stock_price": 50,                # skip penny stocks (used by screener, Phase 3)
    "max_stock_price": None,              # set e.g. 500 for small accounts; None = no cap
    # Phase 3 — stock picker universe
    # "dynamic" (default): downloads NSE EQUITY_L.csv, applies eligibility rules,
    # caches ~400–600 liquid EQ-series stocks. Rebuilds automatically when older
    # than universe_max_age_days. Static names (nifty50, nifty200, midcap150, nifty500)
    # are offline fallback or explicit overrides via --universe flag.
    "universe": "dynamic",
    "universe_max_age_days": 7,
    # Round-trip transaction cost % (brokerage + STT + DP + slippage).
    # Used in net-RR check: a 1.8 gross RR that becomes 1.4 net on a small
    # position fails the floor. On tiny accounts costs are first-class risk.
    "txn_cost_pct_round_trip": 0.5,
    # Max entry price deviation from last close: signals with all entries
    # further than this % from last close are rejected as stale levels.
    "max_entry_deviation_pct": 15.0,
    "benchmark_ticker": None,
    "benchmark_map": {
        ".NS":  "^NSEI",       # NSE India (Nifty 50)
        ".BO":  "^BSESN",      # BSE India (Sensex)
        ".T":   "^N225",       # Tokyo (Nikkei 225)
        ".HK":  "^HSI",        # Hong Kong (Hang Seng)
        ".L":   "^FTSE",       # London (FTSE 100)
        ".TO":  "^GSPTSE",     # Toronto (TSX Composite)
        ".AX":  "^AXJO",       # Australia (ASX 200)
        ".SS":  "000001.SS",   # Shanghai (SSE Composite)
        ".SZ":  "399001.SZ",   # Shenzhen (SZSE Component)
        "":     "SPY",         # default for US-listed tickers (no suffix)
    },
})
