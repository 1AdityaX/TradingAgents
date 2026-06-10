"""Indian market data layer for TradingAgents.

Provides NSE/BSE-specific data: quotes, FII/DII flows, promoter/pledge data,
corporate actions, richer fundamentals, RSS news, and the NSE market calendar.

All public functions return formatted strings ready for agent injection.
On data unavailability (NSE blocks server-side access, yfinance gaps), each
function returns an explicit "data unavailable" sentinel rather than an empty
string — matching the pattern established in issue #781.
"""

from .market_calendar import (
    is_trading_day,
    get_next_trading_day,
    get_monthly_expiry,
    get_next_expiry,
    get_next_holiday,
    market_status,
    format_calendar_context,
)
from .nse_client import (
    get_quote,
    get_fno_ban_list,
    get_bulk_block_deals,
    get_index_snapshot,
)
from .flows import get_fii_dii_flows
from .shareholding import get_shareholding_summary
from .corporate_actions import get_corporate_actions
from .screener_fundamentals import get_india_fundamentals
from .india_news import get_india_news, get_india_global_news

__all__ = [
    # calendar
    "is_trading_day",
    "get_next_trading_day",
    "get_monthly_expiry",
    "get_next_expiry",
    "get_next_holiday",
    "market_status",
    "format_calendar_context",
    # nse client
    "get_quote",
    "get_fno_ban_list",
    "get_bulk_block_deals",
    "get_index_snapshot",
    # flows
    "get_fii_dii_flows",
    # shareholding
    "get_shareholding_summary",
    # corporate actions
    "get_corporate_actions",
    # fundamentals
    "get_india_fundamentals",
    # news
    "get_india_news",
    "get_india_global_news",
]
