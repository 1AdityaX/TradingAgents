from typing import Annotated

# Import from vendor-specific modules
from .y_finance import (
    get_YFin_data_online,
    get_stock_stats_indicators_window,
    get_fundamentals as get_yfinance_fundamentals,
    get_balance_sheet as get_yfinance_balance_sheet,
    get_cashflow as get_yfinance_cashflow,
    get_income_statement as get_yfinance_income_statement,
    get_insider_transactions as get_yfinance_insider_transactions,
)
from .yfinance_news import get_news_yfinance, get_global_news_yfinance
from .alpha_vantage import (
    get_stock as get_alpha_vantage_stock,
    get_indicator as get_alpha_vantage_indicator,
    get_fundamentals as get_alpha_vantage_fundamentals,
    get_balance_sheet as get_alpha_vantage_balance_sheet,
    get_cashflow as get_alpha_vantage_cashflow,
    get_income_statement as get_alpha_vantage_income_statement,
    get_insider_transactions as get_alpha_vantage_insider_transactions,
    get_news as get_alpha_vantage_news,
    get_global_news as get_alpha_vantage_global_news,
)
from .alpha_vantage_common import AlphaVantageRateLimitError
from .symbol_utils import NoMarketDataError, detect_market_profile

# India-specific data modules
from .india.screener_fundamentals import get_india_fundamentals
from .india.india_news import get_india_news, get_india_global_news

# Configuration and routing logic
from .config import get_config

# Tools organized by category
TOOLS_CATEGORIES = {
    "core_stock_apis": {
        "description": "OHLCV stock price data",
        "tools": [
            "get_stock_data"
        ]
    },
    "technical_indicators": {
        "description": "Technical analysis indicators",
        "tools": [
            "get_indicators"
        ]
    },
    "fundamental_data": {
        "description": "Company fundamentals",
        "tools": [
            "get_fundamentals",
            "get_balance_sheet",
            "get_cashflow",
            "get_income_statement"
        ]
    },
    "news_data": {
        "description": "News and insider data",
        "tools": [
            "get_news",
            "get_global_news",
            "get_insider_transactions",
        ]
    }
}

VENDOR_LIST = [
    "yfinance",
    "alpha_vantage",
    "india_composite",
    "india_news",
]

# Mapping of methods to their vendor-specific implementations
VENDOR_METHODS = {
    # core_stock_apis
    "get_stock_data": {
        "alpha_vantage": get_alpha_vantage_stock,
        "yfinance": get_YFin_data_online,
    },
    # technical_indicators
    "get_indicators": {
        "alpha_vantage": get_alpha_vantage_indicator,
        "yfinance": get_stock_stats_indicators_window,
    },
    # fundamental_data
    "get_fundamentals": {
        "alpha_vantage": get_alpha_vantage_fundamentals,
        "yfinance": get_yfinance_fundamentals,
        "india_composite": get_india_fundamentals,
    },
    "get_balance_sheet": {
        "alpha_vantage": get_alpha_vantage_balance_sheet,
        "yfinance": get_yfinance_balance_sheet,
    },
    "get_cashflow": {
        "alpha_vantage": get_alpha_vantage_cashflow,
        "yfinance": get_yfinance_cashflow,
    },
    "get_income_statement": {
        "alpha_vantage": get_alpha_vantage_income_statement,
        "yfinance": get_yfinance_income_statement,
    },
    # news_data
    "get_news": {
        "alpha_vantage": get_alpha_vantage_news,
        "yfinance": get_news_yfinance,
        "india_news": get_india_news,
    },
    "get_global_news": {
        "yfinance": get_global_news_yfinance,
        "alpha_vantage": get_alpha_vantage_global_news,
        "india_news": get_india_global_news,
    },
    "get_insider_transactions": {
        "alpha_vantage": get_alpha_vantage_insider_transactions,
        "yfinance": get_yfinance_insider_transactions,
    },
}

def get_category_for_method(method: str) -> str:
    """Get the category that contains the specified method."""
    for category, info in TOOLS_CATEGORIES.items():
        if method in info["tools"]:
            return category
    raise ValueError(f"Method '{method}' not found in any category")


# Methods whose first positional arg is a ticker symbol — used for market
# profile auto-detection in route_to_vendor.
_TICKER_FIRST_METHODS = frozenset(
    {
        "get_stock_data",
        "get_indicators",
        "get_fundamentals",
        "get_balance_sheet",
        "get_cashflow",
        "get_income_statement",
        "get_news",
        "get_insider_transactions",
    }
)

# India vendor overrides applied when the ticker resolves to market_profile="india"
# and the configured profile is "auto" or "india".
_INDIA_VENDOR_OVERRIDES: dict[str, str] = {
    "fundamental_data": "india_composite",
    "news_data": "india_news",
}


def get_vendor(category: str, method: str = None, ticker: str = None) -> str:
    """Get the configured vendor for a data category or specific tool method.

    Tool-level configuration takes precedence over category-level.  When
    *ticker* is provided and the active market profile resolves to ``"india"``,
    the fundamental_data and news_data categories are transparently redirected
    to the India-specific vendors.
    """
    config = get_config()

    # Check tool-level configuration first (if method provided)
    if method:
        tool_vendors = config.get("tool_vendors", {})
        if method in tool_vendors:
            return tool_vendors[method]

    # Apply India overrides when the ticker is an NSE/BSE instrument or when the
    # market_profile is explicitly set to "india" (covers get_global_news which
    # has no ticker to auto-detect from).
    if category in _INDIA_VENDOR_OVERRIDES:
        profile = config.get("market_profile", "auto")
        if profile == "india":
            return _INDIA_VENDOR_OVERRIDES[category]
        if profile == "auto" and ticker is not None and detect_market_profile(ticker) == "india":
            return _INDIA_VENDOR_OVERRIDES[category]

    # Fall back to category-level configuration
    return config.get("data_vendors", {}).get(category, "default")


def route_to_vendor(method: str, *args, **kwargs):
    """Route method calls to appropriate vendor implementation with fallback support."""
    category = get_category_for_method(method)
    # Extract ticker from positional args for market profile detection
    ticker_hint = args[0] if args and method in _TICKER_FIRST_METHODS else None
    vendor_config = get_vendor(category, method, ticker=ticker_hint)
    primary_vendors = [v.strip() for v in vendor_config.split(',')]

    if method not in VENDOR_METHODS:
        raise ValueError(f"Method '{method}' not supported")

    # Build fallback chain: primary vendors first, then remaining available vendors
    all_available_vendors = list(VENDOR_METHODS[method].keys())
    fallback_vendors = primary_vendors.copy()
    for vendor in all_available_vendors:
        if vendor not in fallback_vendors:
            fallback_vendors.append(vendor)

    last_no_data: NoMarketDataError | None = None
    first_error: Exception | None = None
    for vendor in fallback_vendors:
        if vendor not in VENDOR_METHODS[method]:
            continue

        vendor_impl = VENDOR_METHODS[method][vendor]
        impl_func = vendor_impl[0] if isinstance(vendor_impl, list) else vendor_impl

        try:
            return impl_func(*args, **kwargs)
        except AlphaVantageRateLimitError:
            continue  # Rate limits: try the next vendor
        except NoMarketDataError as e:
            last_no_data = e  # No data here; another vendor may have it
            continue
        except Exception as e:
            # A fallback vendor failing for an incidental reason (e.g. no API
            # key configured) must not crash the call when another vendor
            # already determined the symbol simply has no data. Remember the
            # first error so a genuine primary-vendor failure still surfaces.
            if first_error is None:
                first_error = e
            continue

    # If any vendor reported "no data", the symbol is genuinely unavailable.
    # Return one explicit, instructive sentinel rather than a vendor-specific
    # empty string, so the agent reports "unavailable" instead of inventing a
    # value. This takes precedence over incidental fallback errors.
    if last_no_data is not None:
        sym = last_no_data.symbol
        canonical = last_no_data.canonical
        resolved = "" if canonical == sym else f" (resolved to '{canonical}')"
        return (
            f"NO_DATA_AVAILABLE: No market data found for '{sym}'{resolved} from "
            f"any configured vendor. The symbol may be invalid, delisted, or not "
            f"covered by Yahoo Finance / Alpha Vantage. Do not estimate or "
            f"fabricate values — report that data is unavailable for this symbol."
        )

    # No vendor returned data and none reported clean "no data" — surface the
    # first real error (e.g. the primary vendor's network failure).
    if first_error is not None:
        raise first_error

    raise RuntimeError(f"No available vendor for '{method}'")


# ---------------------------------------------------------------------------
# India-specific data access (called directly by agents when market_profile
# resolves to "india"; these supplement the standard vendor-routed calls).
# ---------------------------------------------------------------------------

def get_india_fii_dii_flows(ticker: str = "", sessions: int = 10) -> str:
    """Return the last *sessions* FII/DII cash-market flow sessions.

    *ticker* is accepted for interface consistency but not used — flows are
    market-wide, not per-stock.
    """
    from .india.flows import get_fii_dii_flows
    from datetime import date
    return get_fii_dii_flows(as_of=date.today(), sessions=sessions)


def get_india_shareholding(ticker: str) -> str:
    """Return promoter/institutional shareholding summary for an Indian stock."""
    from .india.shareholding import get_shareholding_summary
    return get_shareholding_summary(ticker)


def get_india_corporate_actions(ticker: str) -> str:
    """Return upcoming corporate actions and results dates for an Indian stock."""
    from .india.corporate_actions import get_corporate_actions
    return get_corporate_actions(ticker)


def get_india_instrument_context(ticker: str) -> dict:
    """Return a dict of India-specific instrument context fields.

    Used by ``build_instrument_context`` in agent_utils to enrich the
    India INSTRUMENT CONTEXT block injected into every agent prompt.
    """
    from .india.nse_client import get_fno_ban_list
    from .india.market_calendar import format_calendar_context, get_next_expiry
    from .india.corporate_actions import get_corporate_actions
    from datetime import date

    result: dict = {}

    # F&O ban list
    try:
        ban = get_fno_ban_list(date.today())
        nse_sym = ticker.upper().replace(".NS", "").replace(".BO", "")
        result["fno_ban"] = "YES" if nse_sym in ban else "No"
    except Exception:
        result["fno_ban"] = "data unavailable"

    # Calendar context
    try:
        result["calendar_context"] = format_calendar_context(date.today())
    except Exception:
        result["calendar_context"] = "data unavailable"

    # Next expiry date
    try:
        result["next_expiry"] = str(get_next_expiry(date.today()))
    except Exception:
        result["next_expiry"] = "data unavailable"

    return result