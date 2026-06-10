"""NSE stock universe — dynamic builder (default) + static CSV fallback.

Default mode ("dynamic"):
  1. Download NSE's official equity master list (EQUITY_L.csv, ~2,000 stocks).
  2. Filter: EQ series only, price band, 20-day median traded value > ₹5 Cr.
  3. Cache the ~400–600 survivors to disk with a built_on date.
  4. Lazy auto-refresh: every `scan` call silently rebuilds when the cache
     is older than `universe_max_age_days` (default 7).

Static CSV fallback ("nifty50", "nifty200", etc.):
  - Used when NSE is unreachable and no cached dynamic universe exists.
  - Also used for explicit `--universe nifty50`-style overrides.

Config keys:
  "universe":              "dynamic" | "nifty50" | "nifty200" | ...
  "universe_max_age_days": 7    (default)

CLI:
  python -m cli.main universe refresh   # force rebuild
  python -m cli.main universe show      # print current set
  python -m tradingagents.picker.universe refresh   # same, raw
"""

from __future__ import annotations

import csv
import io
import json
import logging
import os
import time
import warnings
from dataclasses import dataclass
from datetime import date, datetime
from typing import Optional

logger = logging.getLogger(__name__)

_DATA_DIR = os.path.join(os.path.dirname(__file__), "data")

# NSE equity master list — publicly available, no cookie needed (file download)
_EQUITY_L_URL = "https://archives.nseindia.com/content/equities/EQUITY_L.csv"
_UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
)

# Lighter liquidity floor for universe building (screener re-applies ₹10 Cr on
# fresher daily-cached data; using ₹5 Cr here keeps more names in the cache
# so the screener has headroom if thresholds are relaxed).
_UNIVERSE_MIN_LIQUIDITY_CR = 5.0


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------

class NSEUnavailableError(Exception):
    """Raised when the NSE equity master list cannot be fetched and no cached
    universe exists to fall back on."""


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class StockEntry:
    symbol: str    # e.g. "RELIANCE.NS"
    name: str
    sector: str    # "Unknown" for dynamically built entries


@dataclass
class UniverseCache:
    built_on: date
    source: str          # "dynamic" | "static:<name>"
    stocks: list[StockEntry]


# ---------------------------------------------------------------------------
# Cache helpers
# ---------------------------------------------------------------------------

def _cache_dir() -> str:
    try:
        from tradingagents.dataflows.config import get_config
        base = get_config()["data_cache_dir"]
    except Exception:
        base = os.path.join(os.path.expanduser("~"), ".tradingagents", "cache")
    os.makedirs(base, exist_ok=True)
    return base


def _cache_json_path() -> str:
    return os.path.join(_cache_dir(), "universe_dynamic.json")


def _load_cache() -> Optional[UniverseCache]:
    path = _cache_json_path()
    if not os.path.exists(path):
        return None
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        built_on = date.fromisoformat(data["built_on"])
        stocks = [
            StockEntry(s["symbol"], s["name"], s.get("sector", "Unknown"))
            for s in data["stocks"]
        ]
        return UniverseCache(built_on=built_on, source=data.get("source", "dynamic"), stocks=stocks)
    except Exception as exc:
        logger.warning("Could not load universe cache: %s", exc)
        return None


def _save_cache(cache: UniverseCache) -> None:
    path = _cache_json_path()
    try:
        data = {
            "built_on": str(cache.built_on),
            "source": cache.source,
            "count": len(cache.stocks),
            "stocks": [
                {"symbol": s.symbol, "name": s.name, "sector": s.sector}
                for s in cache.stocks
            ],
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f)
        logger.info("Saved universe cache: %d stocks → %s", len(cache.stocks), path)
    except Exception as exc:
        logger.warning("Could not save universe cache: %s", exc)


# ---------------------------------------------------------------------------
# NSE equity master list
# ---------------------------------------------------------------------------

def _fetch_equity_master() -> list[tuple[str, str]]:
    """Download EQUITY_L.csv and return [(nse_symbol, company_name), ...] for EQ series."""
    import requests

    try:
        resp = requests.get(
            _EQUITY_L_URL,
            headers={"User-Agent": _UA},
            timeout=30,
        )
        resp.raise_for_status()
    except Exception as exc:
        raise NSEUnavailableError(f"Could not download EQUITY_L.csv: {exc}") from exc

    results: list[tuple[str, str]] = []
    try:
        # NSE sometimes serves with BOM; io.StringIO handles utf-8-sig
        text = resp.content.decode("utf-8-sig", errors="replace")
        reader = csv.DictReader(io.StringIO(text))
        for row in reader:
            series = row.get("SERIES", "").strip()
            if series != "EQ":
                continue
            sym = row.get("SYMBOL", "").strip()
            name = row.get("NAME OF COMPANY", sym).strip()
            if sym:
                results.append((sym, name))
    except Exception as exc:
        raise NSEUnavailableError(f"Could not parse EQUITY_L.csv: {exc}") from exc

    if not results:
        raise NSEUnavailableError("EQUITY_L.csv parsed but returned zero EQ entries")

    logger.info("EQUITY_L.csv: %d EQ-series stocks", len(results))
    return results


# ---------------------------------------------------------------------------
# Eligibility filter via batch yfinance download
# ---------------------------------------------------------------------------

def _batch_filter(
    symbols_names: list[tuple[str, str]],
    min_price: float,
    max_price: Optional[float],
    min_liquidity_cr: float,
    progress_cb=None,
) -> list[StockEntry]:
    """Apply price + liquidity filters via batched yfinance downloads.

    Downloads 1-month OHLCV in batches of 100. Stocks that cannot be
    downloaded or fail the filters are silently skipped.
    """
    import yfinance as yf
    import pandas as pd
    import warnings

    eligible: list[StockEntry] = []
    tickers = [sym + ".NS" for sym, _ in symbols_names]
    sym_map = {sym + ".NS": (sym, name) for sym, name in symbols_names}
    batch_size = 100
    total = len(tickers)

    for batch_start in range(0, total, batch_size):
        batch = tickers[batch_start: batch_start + batch_size]
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                raw = yf.download(
                    batch,
                    period="1mo",
                    interval="1d",
                    auto_adjust=True,
                    progress=False,
                    threads=True,
                    group_by="ticker",
                )

            for ticker in batch:
                sym, name = sym_map.get(ticker, (ticker, ticker))
                # Extract per-ticker DataFrame
                try:
                    if len(batch) == 1:
                        df = raw
                    else:
                        df = raw[ticker]
                    df = df.dropna(how="all")
                    if df is None or df.empty or len(df) < 5:
                        continue

                    close_col = df.get("Close") if hasattr(df, "get") else df["Close"]
                    vol_col = df.get("Volume") if hasattr(df, "get") else df["Volume"]

                    last_close = float(close_col.iloc[-1])
                    if not (last_close == last_close):   # NaN check
                        continue

                    # Price filter
                    if last_close < min_price:
                        continue
                    if max_price is not None and last_close > max_price:
                        continue

                    # Liquidity filter (20-day median traded value)
                    n = min(20, len(close_col))
                    traded_cr = float((close_col.tail(n) * vol_col.tail(n)).median()) / 1e7
                    if traded_cr < min_liquidity_cr:
                        continue

                    eligible.append(StockEntry(symbol=ticker, name=name, sector="Unknown"))
                except Exception:
                    pass

        except Exception as exc:
            logger.debug("Batch download failed for %s..%s: %s",
                         batch[0], batch[-1], exc)

        if progress_cb:
            progress_cb(min(batch_start + batch_size, total), total)

    return eligible


# ---------------------------------------------------------------------------
# Public API — dynamic universe
# ---------------------------------------------------------------------------

def rebuild_universe(config: Optional[dict] = None) -> UniverseCache:
    """Fetch EQUITY_L.csv, apply eligibility rules, persist cache.

    Raises NSEUnavailableError if the master list cannot be fetched.
    """
    if config is None:
        try:
            from tradingagents.default_config import DEFAULT_CONFIG
            config = DEFAULT_CONFIG
        except Exception:
            config = {}

    min_price = float(config.get("min_stock_price", 50))
    max_price = config.get("max_stock_price")
    max_price = float(max_price) if max_price is not None else None

    logger.info("Rebuilding dynamic universe from NSE EQUITY_L.csv...")
    symbols_names = _fetch_equity_master()

    logger.info("Applying eligibility filters to %d EQ stocks...", len(symbols_names))
    stocks = _batch_filter(
        symbols_names,
        min_price=min_price,
        max_price=max_price,
        min_liquidity_cr=_UNIVERSE_MIN_LIQUIDITY_CR,
    )

    logger.info("Dynamic universe: %d eligible stocks after filters", len(stocks))
    cache = UniverseCache(
        built_on=date.today(),
        source="dynamic",
        stocks=stocks,
    )
    _save_cache(cache)
    return cache


def get_universe(config: Optional[dict] = None) -> list[StockEntry]:
    """Return the eligible stock universe, auto-rebuilding when stale.

    Uses lazy refresh: if the cached universe is older than
    `universe_max_age_days` (default 7), it is silently rebuilt.
    Falls back to the stale cache (then static CSVs) on NSE failure.
    """
    if config is None:
        try:
            from tradingagents.default_config import DEFAULT_CONFIG
            config = DEFAULT_CONFIG
        except Exception:
            config = {}

    max_age = int(config.get("universe_max_age_days", 7))
    cache = _load_cache()
    today = date.today()

    needs_rebuild = (
        cache is None
        or (today - cache.built_on).days >= max_age
    )

    if needs_rebuild:
        try:
            cache = rebuild_universe(config)
        except NSEUnavailableError as exc:
            msg = str(exc)
            if cache is not None:
                warnings.warn(
                    f"NSE unreachable — using stale universe from {cache.built_on} "
                    f"({len(cache.stocks)} stocks). {msg}",
                    stacklevel=2,
                )
            else:
                warnings.warn(
                    f"NSE unreachable and no cached universe. Falling back to static CSVs. {msg}",
                    stacklevel=2,
                )
                return _static_fallback(config)

    return cache.stocks  # type: ignore[return-value]


def _static_fallback(config: Optional[dict] = None) -> list[StockEntry]:
    """Return the largest available static universe as an offline fallback."""
    for name in ("nifty500", "nifty200", "nifty50"):
        try:
            entries = load_universe(name)
            if entries:
                warnings.warn(
                    f"Using static fallback universe '{name}' ({len(entries)} stocks).",
                    stacklevel=3,
                )
                return entries
        except Exception:
            pass
    return []


# ---------------------------------------------------------------------------
# Public API — static CSV universes
# ---------------------------------------------------------------------------

_UNIVERSE_FILES = {
    "nifty50":      "nifty50.csv",
    "nifty_next50": "nifty_next50.csv",
    "midcap150":    "midcap150.csv",
}

_COMPOSITE_UNIVERSES = {
    "nifty200": ["nifty50", "nifty_next50", "midcap150"],
    "nifty500": ["nifty50", "nifty_next50", "midcap150"],
}


def _load_csv(filename: str) -> list[StockEntry]:
    path = os.path.join(_DATA_DIR, filename)
    entries: list[StockEntry] = []
    try:
        with open(path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                sym = row.get("symbol", "").strip()
                if not sym:
                    continue
                entries.append(StockEntry(
                    symbol=sym,
                    name=row.get("name", sym).strip(),
                    sector=row.get("sector", "Unknown").strip(),
                ))
    except FileNotFoundError:
        logger.error("Universe file not found: %s", path)
    return entries


def load_universe(name: str) -> list[StockEntry]:
    """Return static universe *name* (nifty50, nifty200, midcap150, nifty500).

    For the dynamic universe, call get_universe() instead.
    """
    name = name.lower().strip()

    if name in _UNIVERSE_FILES:
        return _load_csv(_UNIVERSE_FILES[name])

    if name in _COMPOSITE_UNIVERSES:
        seen: set[str] = set()
        result: list[StockEntry] = []
        for part in _COMPOSITE_UNIVERSES[name]:
            for entry in _load_csv(_UNIVERSE_FILES[part]):
                if entry.symbol not in seen:
                    seen.add(entry.symbol)
                    result.append(entry)
        return result

    raise ValueError(
        f"Unknown static universe '{name}'. "
        f"Supported: {', '.join(list(_UNIVERSE_FILES) + list(_COMPOSITE_UNIVERSES))}"
    )


def list_universes() -> list[str]:
    """All supported universe names (dynamic + static)."""
    return ["dynamic"] + list(_UNIVERSE_FILES) + list(_COMPOSITE_UNIVERSES)


def symbols_only(universe: str) -> list[str]:
    """Convenience: return just the ticker symbols for *universe*."""
    if universe == "dynamic":
        return [e.symbol for e in get_universe()]
    return [e.symbol for e in load_universe(universe)]


def universe_info() -> dict:
    """Return metadata about the currently cached dynamic universe."""
    cache = _load_cache()
    if cache is None:
        return {"status": "not built", "count": 0, "built_on": None}
    today = date.today()
    age = (today - cache.built_on).days
    try:
        from tradingagents.default_config import DEFAULT_CONFIG
        max_age = int(DEFAULT_CONFIG.get("universe_max_age_days", 7))
    except Exception:
        max_age = 7
    return {
        "status": "stale" if age >= max_age else "fresh",
        "count": len(cache.stocks),
        "built_on": str(cache.built_on),
        "age_days": age,
        "max_age_days": max_age,
        "cache_path": _cache_json_path(),
    }


# ---------------------------------------------------------------------------
# __main__ entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    cmd = sys.argv[1] if len(sys.argv) > 1 else "show"

    if cmd == "refresh":
        print("Rebuilding dynamic universe from NSE...")
        try:
            c = rebuild_universe()
            print(f"Done — {len(c.stocks)} stocks cached (built {c.built_on})")
        except NSEUnavailableError as e:
            print(f"NSE unavailable: {e}")
            sys.exit(1)

    elif cmd == "show":
        info = universe_info()
        print(f"Dynamic universe: {info}")
        cache = _load_cache()
        if cache:
            for s in cache.stocks[:20]:
                print(f"  {s.symbol:16} {s.name}")
            if len(cache.stocks) > 20:
                print(f"  ... and {len(cache.stocks) - 20} more")

    else:
        print("Usage: python -m tradingagents.picker.universe [refresh|show]")
