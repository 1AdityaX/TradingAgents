"""NSE India public API client.

Wraps NSE's JSON endpoints (quote, F&O ban list, bulk/block deals, index
snapshot). Every successful fetch is disk-cached (4-hour TTL) to avoid
hammering NSE's servers and to stay under their rate limits.

NSE's public API requires browser-style cookies and a descriptive User-Agent.
Access is frequently blocked from cloud/data-centre IPs, so every function
has a yfinance fallback and returns an explicit "data unavailable" string
rather than an empty result — agents must never fabricate a value (issue #781).
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import time
from datetime import date, datetime
from typing import Optional

import requests
import yfinance as yf

from ..config import get_config
from ..symbol_utils import NoMarketDataError, normalize_symbol

logger = logging.getLogger(__name__)

_NSE_BASE = "https://www.nseindia.com"
_UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
)
_HEADERS = {
    "User-Agent": _UA,
    "Accept": "application/json, text/plain, */*",
    "Accept-Language": "en-US,en;q=0.9",
    "Referer": "https://www.nseindia.com/",
    "X-Requested-With": "XMLHttpRequest",
}
_CACHE_TTL = 4 * 3600  # seconds; stale-within-session is fine for swing trades

# Module-level session; None until first use.
_session: Optional[requests.Session] = None

# yfinance ticker symbols for major Indian indices (fallback path)
_INDEX_YF_MAP = {
    "NIFTY 50": "^NSEI",
    "NIFTY50": "^NSEI",
    "NIFTY BANK": "^NSEBANK",
    "NIFTYBANK": "^NSEBANK",
    "NIFTY MIDCAP 150": "^NSEMDCP50",
    "SENSEX": "^BSESN",
    "BSE SENSEX": "^BSESN",
}


# ---------------------------------------------------------------------------
# Cache helpers
# ---------------------------------------------------------------------------

def _cache_dir() -> str:
    cfg = get_config()
    d = os.path.join(cfg["data_cache_dir"], "nse_cache")
    os.makedirs(d, exist_ok=True)
    return d


def _cache_path(key: str) -> str:
    safe = hashlib.md5(key.encode()).hexdigest()
    return os.path.join(_cache_dir(), f"{safe}.json")


def _read_cache(key: str) -> Optional[dict]:
    path = _cache_path(key)
    if not os.path.exists(path):
        return None
    try:
        with open(path) as f:
            entry = json.load(f)
        if time.time() - entry.get("_ts", 0) > _CACHE_TTL:
            return None
        return entry.get("data")
    except (json.JSONDecodeError, KeyError, OSError):
        return None


def _write_cache(key: str, data) -> None:
    try:
        with open(_cache_path(key), "w") as f:
            json.dump({"_ts": time.time(), "data": data}, f)
    except OSError:
        pass


# ---------------------------------------------------------------------------
# HTTP helpers
# ---------------------------------------------------------------------------

def _get_session() -> requests.Session:
    """Return a session pre-warmed with NSE homepage cookies."""
    global _session
    if _session is not None:
        return _session
    s = requests.Session()
    s.headers.update(_HEADERS)
    try:
        s.get(_NSE_BASE, timeout=10)
    except Exception as exc:
        logger.warning("NSE cookie warm-up failed: %s", exc)
    _session = s
    return _session


def _fetch_nse(endpoint: str, params: Optional[dict] = None) -> Optional[dict]:
    """GET a JSON endpoint on nseindia.com; return parsed dict or None on error."""
    url = f"{_NSE_BASE}{endpoint}"
    for attempt in range(3):
        try:
            resp = _get_session().get(url, params=params, timeout=15)
            if resp.status_code == 200:
                return resp.json()
            if resp.status_code in (403, 429):
                logger.warning("NSE blocked %s (HTTP %d)", endpoint, resp.status_code)
                return None
            time.sleep(1.5 ** attempt)
        except Exception as exc:
            logger.warning("NSE fetch attempt %d/%d for %s: %s", attempt + 1, 3, endpoint, exc)
            if attempt < 2:
                time.sleep(1.5 ** attempt)
    return None


# ---------------------------------------------------------------------------
# Symbol helpers
# ---------------------------------------------------------------------------

def _nse_symbol(ticker: str) -> str:
    """Strip exchange suffix — 'RELIANCE.NS' → 'RELIANCE'."""
    s = ticker.upper().strip()
    for suffix in (".NS", ".BO"):
        if s.endswith(suffix):
            return s[: -len(suffix)]
    return s


def _yf_canonical(ticker: str) -> str:
    """Ensure ticker has .NS suffix for yfinance lookups."""
    s = ticker.upper().strip()
    if s.endswith(".NS") or s.endswith(".BO"):
        return s
    return s + ".NS"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def get_quote(ticker: str) -> str:
    """Return a formatted NSE quote string for *ticker*.

    Falls back to yfinance when NSE API is unavailable.
    Returns an explicit "data unavailable" string on total failure.
    """
    nse_sym = _nse_symbol(ticker)
    cache_key = f"quote_{nse_sym}"
    cached = _read_cache(cache_key)

    if cached is None:
        raw = _fetch_nse(f"/api/quote-equity?symbol={nse_sym}")
        if raw:
            _write_cache(cache_key, raw)
            cached = raw

    if cached:
        try:
            pi = cached.get("priceInfo", {})
            meta = cached.get("info", {})
            ltp = pi.get("lastPrice", "N/A")
            open_ = pi.get("open", "N/A")
            day_hl = pi.get("intraDayHighLow", {})
            high = day_hl.get("max", "N/A")
            low = day_hl.get("min", "N/A")
            prev_close = pi.get("previousClose", "N/A")
            pct_chg = pi.get("pChange", "N/A")
            name = meta.get("companyName", nse_sym)
            lines = [
                f"NSE Quote: {nse_sym} ({name})",
                f"  LTP ₹{ltp}  |  Open ₹{open_}  |  Day H/L ₹{high}/₹{low}",
                f"  Prev Close ₹{prev_close}  |  Change {pct_chg}%",
                f"  As of {datetime.now().strftime('%Y-%m-%d %H:%M IST')}",
            ]
            return "\n".join(lines)
        except Exception as exc:
            logger.debug("NSE quote parse error for %s: %s", ticker, exc)

    # Fallback: yfinance
    canonical = _yf_canonical(ticker)
    try:
        info = yf.Ticker(canonical).info
        if info:
            ltp = info.get("currentPrice") or info.get("regularMarketPrice", "N/A")
            prev = info.get("previousClose", "N/A")
            if ltp != "N/A":
                return (
                    f"Quote (yfinance): {canonical}  LTP ₹{ltp}  |  Prev Close ₹{prev}"
                    f"  |  As of {datetime.now().strftime('%Y-%m-%d')}"
                )
    except Exception as exc:
        logger.debug("yfinance quote fallback failed for %s: %s", ticker, exc)

    return f"Quote data unavailable for {ticker} — data unavailable."


def get_fno_ban_list(as_of: Optional[date] = None) -> str:
    """Return today's F&O ban-period securities as a formatted string.

    Returns an explicit "data unavailable" string when NSE is inaccessible.
    Stocks in the ban list may not take fresh F&O positions.
    """
    ref = as_of if as_of is not None else date.today()
    cache_key = f"fno_ban_{ref}"
    cached = _read_cache(cache_key)

    if cached is None:
        raw = _fetch_nse("/api/fo-ban-status")
        if raw:
            _write_cache(cache_key, raw)
            cached = raw

    if cached:
        try:
            entries = cached.get("data") or []
            symbols = [e.get("symbol", "") for e in entries if e.get("symbol")]
            if not symbols:
                return f"F&O ban list for {ref}: no securities in ban period."
            return f"F&O ban list for {ref}: {', '.join(symbols)}."
        except Exception as exc:
            logger.debug("F&O ban list parse error: %s", exc)

    return f"F&O ban list data unavailable for {ref} — data unavailable."


def get_bulk_block_deals(as_of: Optional[date] = None) -> str:
    """Return bulk and block deals for *as_of* date as a formatted string.

    Returns an explicit "data unavailable" string when NSE is inaccessible.
    """
    ref = as_of if as_of is not None else date.today()
    cache_key = f"bulk_deals_{ref}"
    cached = _read_cache(cache_key)

    if cached is None:
        raw = _fetch_nse("/api/bulk-deals")
        if raw:
            _write_cache(cache_key, raw)
            cached = raw

    if cached:
        try:
            deals = cached.get("data") or []
            if not deals:
                return f"No bulk/block deals reported on {ref}."
            lines = [f"Bulk/Block deals on {ref}:"]
            for d in deals[:20]:
                sym = d.get("symbol", "?")
                client = d.get("clientName", "?")
                bs = d.get("buySell", "?")
                qty = d.get("quantityTraded", "?")
                price = d.get("tradePrice", "?")
                lines.append(f"  {sym}: {client} {bs}  qty {qty}  @ ₹{price}")
            return "\n".join(lines)
        except Exception as exc:
            logger.debug("Bulk deals parse error: %s", exc)

    return f"Bulk/block deal data unavailable for {ref} — data unavailable."


def get_index_snapshot(index_name: str = "NIFTY 50") -> str:
    """Return a snapshot of the requested NSE/BSE index.

    Falls back to yfinance for major indices when NSE API is blocked.
    """
    cache_key = f"index_{index_name.upper()}"
    cached = _read_cache(cache_key)

    if cached is None:
        raw = _fetch_nse("/api/allIndices")
        if raw:
            _write_cache(cache_key, raw)
            cached = raw

    if cached:
        try:
            target = index_name.upper()
            for entry in cached.get("data", []):
                idx_sym = (entry.get("indexSymbol") or entry.get("index") or "").upper()
                if idx_sym == target:
                    last = entry.get("last", "N/A")
                    pct = entry.get("percentChange", 0)
                    high = entry.get("high", "N/A")
                    low = entry.get("low", "N/A")
                    return (
                        f"{index_name}: {last} ({pct:+.2f}% today)"
                        f" | Day H/L: {high} / {low}"
                    )
        except Exception as exc:
            logger.debug("Index snapshot parse error: %s", exc)

    # yfinance fallback
    yf_sym = _INDEX_YF_MAP.get(index_name.upper())
    if yf_sym:
        try:
            info = yf.Ticker(yf_sym).info
            price = info.get("regularMarketPrice") or info.get("previousClose")
            if price:
                return (
                    f"{index_name} (yfinance): {price}"
                    f" | As of {datetime.now().strftime('%Y-%m-%d')}"
                )
        except Exception as exc:
            logger.debug("yfinance index fallback failed for %s: %s", index_name, exc)

    return f"Index data unavailable for '{index_name}' — data unavailable."
