"""Symbol normalization and market-data error types for vendor calls.

Yahoo Finance (the default vendor) uses specific ticker conventions that
differ from the broker / TradingView / MT5 style symbols users often type:

    user types        Yahoo wants       why
    ---------------   ---------------   -----------------------------------
    XAUUSD, XAUUSD+   GC=F              gold has no forex pair on Yahoo;
                                        it is quoted as a COMEX future
    EURUSD            EURUSD=X          spot forex pairs take a ``=X`` suffix
    BTCUSD            BTC-USD           crypto pairs use a ``-`` separator
    SPX500, US500     ^GSPC             index CFDs map to Yahoo index symbols

Passing the raw broker symbol to Yahoo returns an empty result, which the
agents previously received as free text and could hallucinate a price
around (see issue #781). Centralizing the mapping here means every yfinance
entry point resolves symbols the same way, and new instruments are added by
appending a table row rather than editing call sites.
"""

from __future__ import annotations

import logging
import re

logger = logging.getLogger(__name__)


class NoMarketDataError(Exception):
    """Raised when a vendor returns no rows/records for a symbol.

    Carries both the symbol the user requested and the canonical symbol the
    vendor was actually queried with, so callers can build a clear message
    instead of emitting a vendor-specific empty string into the data channel.
    """

    def __init__(self, symbol: str, canonical: str | None = None, detail: str = ""):
        self.symbol = symbol
        self.canonical = canonical or symbol
        self.detail = detail
        msg = f"No market data for {symbol!r}"
        if canonical and canonical != symbol:
            msg += f" (queried as {canonical!r})"
        if detail:
            msg += f": {detail}"
        super().__init__(msg)


# ISO-4217 codes common enough to appear in retail forex pairs. A bare
# six-letter symbol whose halves are BOTH in this set is treated as a spot
# forex pair and given Yahoo's ``=X`` suffix.
_FOREX_CURRENCIES = frozenset(
    {
        "USD", "EUR", "GBP", "JPY", "CHF", "CAD", "AUD", "NZD",
        "CNY", "CNH", "HKD", "SGD", "SEK", "NOK", "DKK", "PLN",
        "MXN", "ZAR", "TRY", "INR", "KRW", "BRL", "RUB", "THB",
    }
)

# Crypto bases that brokers quote against USD without a separator.
_CRYPTO_BASES = frozenset(
    {"BTC", "ETH", "SOL", "XRP", "ADA", "DOGE", "LTC", "BCH", "DOT", "AVAX", "LINK"}
)

# Explicit aliases for instruments whose broker symbol does not map to a
# Yahoo symbol by rule. Metals/energy resolve to their front-month future;
# index CFD names resolve to the underlying Yahoo index symbol. Extend by
# adding rows — no call site changes required.
_ALIASES = {
    # Precious metals (spot names -> COMEX/NYMEX futures)
    "XAUUSD": "GC=F", "XAU": "GC=F", "GOLD": "GC=F",
    "XAGUSD": "SI=F", "XAG": "SI=F", "SILVER": "SI=F",
    "XPTUSD": "PL=F", "XPDUSD": "PA=F",
    # Energy
    "WTICOUSD": "CL=F", "USOIL": "CL=F", "WTI": "CL=F",
    "BCOUSD": "BZ=F", "UKOIL": "BZ=F", "BRENT": "BZ=F",
    "NATGAS": "NG=F", "XNGUSD": "NG=F",
    "COPPER": "HG=F", "XCUUSD": "HG=F",
    # Index CFDs -> Yahoo index symbols
    "SPX500": "^GSPC", "US500": "^GSPC", "SPX": "^GSPC",
    "NAS100": "^NDX", "US100": "^NDX", "USTEC": "^NDX",
    "US30": "^DJI", "DJI30": "^DJI", "WS30": "^DJI",
    "GER40": "^GDAXI", "GER30": "^GDAXI", "DE40": "^GDAXI",
    "UK100": "^FTSE", "JP225": "^N225", "JPN225": "^N225",
    "FRA40": "^FCHI", "EU50": "^STOXX50E", "HK50": "^HSI",
}

# Yahoo symbols may contain letters, digits, and these structural characters.
_YAHOO_SAFE = re.compile(r"^[A-Za-z0-9._\-\^=]+$")


def normalize_symbol(raw: str) -> str:
    """Map a user/broker symbol to its canonical Yahoo Finance symbol.

    Resolution order (first match wins):
      1. Explicit alias table (metals, energy, index CFDs).
      2. Crypto rule: ``<BASE>USD`` where BASE is a known crypto -> ``BASE-USD``.
      3. Forex rule: six letters that are two ISO currency codes -> ``PAIR=X``.
      4. Otherwise the upper-cased symbol is returned unchanged (plain
         equities, ETFs, Yahoo-native symbols like ``GC=F`` or ``^GSPC``).

    A trailing ``+`` (broker CFD marker, e.g. ``XAUUSD+``) is stripped before
    matching. The function is purely syntactic — it performs no network
    calls — so it is safe to apply on every request.
    """
    if not isinstance(raw, str) or not raw.strip():
        return raw

    s = raw.strip().upper()
    # Broker CFD/qualifier suffixes Yahoo never uses.
    s = s.rstrip("+")

    if s in _ALIASES:
        canonical = _ALIASES[s]
    elif len(s) == 6 and s[:3] in _CRYPTO_BASES and s[3:] == "USD":
        canonical = f"{s[:3]}-USD"
    elif s[:-3] in _CRYPTO_BASES and s.endswith("USD") and "-" not in s:
        canonical = f"{s[:-3]}-USD"
    elif len(s) == 6 and s[:3] in _FOREX_CURRENCIES and s[3:] in _FOREX_CURRENCIES:
        canonical = f"{s}=X"
    else:
        canonical = s

    if canonical != raw.strip().upper():
        logger.info("Resolved symbol %r to Yahoo symbol %r", raw, canonical)
    return canonical


def is_yahoo_safe(symbol: str) -> bool:
    """True when ``symbol`` only contains characters Yahoo symbols use."""
    return bool(symbol) and _YAHOO_SAFE.fullmatch(symbol) is not None


# ---------------------------------------------------------------------------
# Indian market helpers
# ---------------------------------------------------------------------------

def is_indian_ticker(symbol: str) -> bool:
    """Return True when *symbol* is (or normalises to) an NSE or BSE ticker.

    Accepts:
      - Suffixed symbols: ``RELIANCE.NS``, ``INFY.BO``
      - Bare NSE symbols passed without suffix: ``RELIANCE``, ``INFY``
        (bare uppercase-only strings with no other exchange indicator)

    Does NOT treat bare symbols as Indian if they already map to a known
    non-Indian instrument (metals, index CFDs, forex pairs, crypto).
    """
    if not isinstance(symbol, str) or not symbol.strip():
        return False
    s = symbol.strip().upper().rstrip("+")
    if s.endswith(".NS") or s.endswith(".BO"):
        return True
    # Not already Indian-suffixed; check it isn't a known non-Indian instrument
    if s in _ALIASES:
        return False
    if len(s) == 6 and s[:3] in _CRYPTO_BASES and s[3:] == "USD":
        return False
    if len(s) == 6 and s[:3] in _FOREX_CURRENCIES and s[3:] in _FOREX_CURRENCIES:
        return False
    # Bare symbol that looks like an NSE stock (all-caps letters, optionally
    # digits, no structural Yahoo characters other than alphanumerics).
    # We conservatively accept bare symbols only when they contain only
    # uppercase letters/digits and no Yahoo-native structural chars (=, ^, -).
    return bool(re.fullmatch(r"[A-Z][A-Z0-9&]{0,19}", s))


def normalize_indian_symbol(symbol: str) -> str:
    """Normalise an Indian ticker to Yahoo Finance's ``.NS`` convention.

    Resolution order:
      1. Already has ``.NS`` or ``.BO`` suffix → return as-is (uppercased).
      2. Bare symbol that ``is_indian_ticker`` accepts → append ``.NS``.
      3. Everything else → return unchanged (let ``normalize_symbol`` handle it).

    Bare-symbol normalisation covers users who type ``RELIANCE`` instead of
    ``RELIANCE.NS``.  The ``.NS`` suffix is used as the canonical form because
    NSE is the primary exchange for most listed Indian equities.
    """
    if not isinstance(symbol, str) or not symbol.strip():
        return symbol
    s = symbol.strip().upper().rstrip("+")
    if s.endswith(".NS") or s.endswith(".BO"):
        return s
    if is_indian_ticker(symbol):
        logger.info("Bare Indian symbol %r normalised to %r", symbol, s + ".NS")
        return s + ".NS"
    return symbol


def detect_market_profile(symbol: str) -> str:
    """Return ``"india"`` for ``.NS``/``.BO`` tickers, ``"us"`` otherwise."""
    if not isinstance(symbol, str):
        return "us"
    s = symbol.strip().upper()
    if s.endswith(".NS") or s.endswith(".BO"):
        return "india"
    return "us"
