"""Quantitative stock screener — NO LLM involved.

Applies liquidity, price, and swing-momentum filters to a universe list,
scores remaining stocks on several factors, and returns up to *max_candidates*
ranked StockEntry objects ready for the picker agent.

All data comes from batched yfinance downloads (cached daily). The screener
never calls an LLM; cost is effectively zero beyond API rate limits.

Scoring factors (each normalised 0–1, weighted sum):
  - trend_align:   price position relative to SMA50 and SMA200
  - momentum:      RSI(14) in the constructive 40–65 band
  - rel_strength:  1-month return vs Nifty 50 (^NSEI)
  - vol_surge:     5-day avg volume / 50-day avg volume ratio
  - pullback_room: distance below 52-week high (some room left = better)
"""

from __future__ import annotations

import json
import logging
import math
import os
import time
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from typing import Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Scoring weights — must sum to 1.0
# ---------------------------------------------------------------------------
_WEIGHTS = {
    "trend_align":   0.25,
    "momentum":      0.20,
    "rel_strength":  0.25,
    "vol_surge":     0.15,
    "pullback_room": 0.15,
}

_CACHE_TTL = 24 * 3600  # daily


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class ScreenerScore:
    symbol: str
    name: str
    sector: str
    price: float
    sma50: float
    sma200: float
    rsi14: float
    atr_pct: float            # ATR(14) as % of price
    high_52w: float
    dist_from_high_pct: float # % below 52w high (positive = below)
    rel_strength_1m: float    # stock 1m return minus Nifty 1m return
    vol_surge_ratio: float    # 5d avg vol / 50d avg vol
    traded_value_20d_cr: float  # median 20-day ₹ traded value in crores
    composite_score: float
    factors: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Cache helpers
# ---------------------------------------------------------------------------

def _cache_dir() -> str:
    try:
        from tradingagents.dataflows.config import get_config
        base = get_config()["data_cache_dir"]
    except Exception:
        base = os.path.join(os.path.expanduser("~"), ".tradingagents", "cache")
    d = os.path.join(base, "screener_cache")
    os.makedirs(d, exist_ok=True)
    return d


def _cache_path(key: str) -> str:
    return os.path.join(_cache_dir(), f"{key}.json")


def _read_cache(key: str):
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
# Metric helpers
# ---------------------------------------------------------------------------

def _compute_rsi(close_series, period: int = 14) -> float:
    """Wilder RSI via EWM — no external TA library required."""
    try:
        import pandas as pd
        delta = close_series.diff().dropna()
        gain = delta.clip(lower=0)
        loss = (-delta).clip(lower=0)
        avg_gain = gain.ewm(com=period - 1, min_periods=period).mean()
        avg_loss = loss.ewm(com=period - 1, min_periods=period).mean()
        last_loss = float(avg_loss.iloc[-1])
        if last_loss == 0:
            return 100.0
        rs = float(avg_gain.iloc[-1]) / last_loss
        return 100.0 - 100.0 / (1.0 + rs)
    except Exception:
        return 50.0


def _compute_atr(high, low, close, period: int = 14) -> float:
    try:
        import pandas as pd
        prev_close = close.shift(1)
        tr = pd.concat([
            high - low,
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ], axis=1).max(axis=1)
        atr = tr.ewm(span=period, min_periods=period).mean()
        return float(atr.iloc[-1]) if not atr.empty else 0.0
    except Exception:
        return 0.0


def _normalise(value: float, low: float, high: float) -> float:
    """Normalise *value* to [0, 1] within [low, high]."""
    if high == low:
        return 0.5
    return max(0.0, min(1.0, (value - low) / (high - low)))


# ---------------------------------------------------------------------------
# Core download + scoring
# ---------------------------------------------------------------------------

def _download_batch(tickers: list[str], period: str = "1y") -> dict:
    """Download OHLCV for *tickers* via yfinance; return {ticker: DataFrame}."""
    import yfinance as yf
    import pandas as pd

    cache_key = f"batch_{'_'.join(sorted(tickers[:5]))}_{len(tickers)}_{period}"
    cached = _read_cache(cache_key)
    if cached is not None:
        # Reconstruct DataFrames from serialised dict
        result = {}
        for sym, rows in cached.items():
            try:
                df = pd.DataFrame(rows)
                df.index = pd.to_datetime(df.index)
                result[sym] = df
            except Exception:
                pass
        if result:
            return result

    try:
        raw = yf.download(
            tickers,
            period=period,
            interval="1d",
            group_by="ticker",
            auto_adjust=True,
            progress=False,
            threads=True,
        )
    except Exception as exc:
        logger.warning("yfinance batch download failed: %s", exc)
        return {}

    result = {}
    import pandas as pd
    if len(tickers) == 1:
        sym = tickers[0]
        if not raw.empty:
            result[sym] = raw
    else:
        for sym in tickers:
            try:
                df = raw[sym].dropna(how="all")
                if not df.empty:
                    result[sym] = df
            except (KeyError, TypeError):
                pass

    # Serialise to cache (convert datetime index → string)
    to_cache = {}
    for sym, df in result.items():
        try:
            serialised = df.copy()
            serialised.index = serialised.index.strftime("%Y-%m-%d")
            to_cache[sym] = serialised.to_dict()
        except Exception:
            pass
    _write_cache(cache_key, to_cache)

    return result


def _score_stock(
    symbol: str,
    name: str,
    sector: str,
    df,
    nifty_1m_return: float,
) -> Optional[ScreenerScore]:
    """Compute all metrics for one stock; return None if data is insufficient."""
    import pandas as pd

    if df is None or len(df) < 60:
        return None

    # Column name normalisation (yfinance sometimes uses title-case)
    cols = {c.lower(): c for c in df.columns}

    def col(name_lower: str):
        return df[cols[name_lower]] if name_lower in cols else None

    close = col("close")
    high = col("high")
    low = col("low")
    volume = col("volume")

    if close is None or close.dropna().empty:
        return None

    price = float(close.iloc[-1])
    if not math.isfinite(price) or price <= 0:
        return None

    # SMA 50 / 200
    sma50 = float(close.rolling(50).mean().iloc[-1]) if len(close) >= 50 else price
    sma200 = float(close.rolling(200).mean().iloc[-1]) if len(close) >= 200 else price

    # RSI
    rsi14 = _compute_rsi(close)

    # ATR %
    if high is not None and low is not None:
        atr_val = _compute_atr(high, low, close)
        atr_pct = (atr_val / price * 100.0) if price > 0 else 0.0
    else:
        atr_pct = 0.0

    # 52-week high
    high_52w = float(close.tail(252).max()) if len(close) >= 20 else price
    dist_from_high_pct = (high_52w - price) / high_52w * 100.0 if high_52w > 0 else 0.0

    # 1-month return (21 trading days)
    if len(close) >= 22:
        price_1m_ago = float(close.iloc[-22])
        stock_1m_return = (price - price_1m_ago) / price_1m_ago * 100.0 if price_1m_ago > 0 else 0.0
    else:
        stock_1m_return = 0.0

    rel_strength_1m = stock_1m_return - nifty_1m_return

    # 20-day median traded value in crores
    if volume is not None and len(volume) >= 20:
        import numpy as np
        last20_vol = volume.tail(20)
        last20_close = close.tail(20)
        traded_val = (last20_vol * last20_close).median()
        traded_value_20d_cr = float(traded_val) / 1e7  # convert to crores
    else:
        traded_value_20d_cr = 0.0

    # Volume surge: 5d avg / 50d avg
    if volume is not None and len(volume) >= 50:
        vol5 = float(volume.tail(5).mean())
        vol50 = float(volume.tail(50).mean())
        vol_surge = vol5 / vol50 if vol50 > 0 else 1.0
    else:
        vol_surge = 1.0

    # --- Scoring ---
    # trend_align: 0 = price below both SMAs, 0.5 = between, 1.0 = above both
    if price > sma50 and price > sma200:
        trend_score = 1.0
    elif price > sma50:
        trend_score = 0.6
    elif price > sma200:
        trend_score = 0.3
    else:
        trend_score = 0.0

    # momentum: RSI 45–65 is the sweet spot for swing continuation
    if 45 <= rsi14 <= 65:
        momentum_score = 1.0
    elif 35 <= rsi14 < 45 or 65 < rsi14 <= 75:
        momentum_score = 0.6
    elif rsi14 < 35:
        momentum_score = 0.2   # oversold — possible mean reversion
    else:
        momentum_score = 0.1   # overbought

    # rel_strength: normalise relative to ±10% range
    rel_str_score = _normalise(rel_strength_1m, -10.0, 10.0)

    # vol_surge: 0.8–3x is constructive; cap at 3x
    vol_surge_score = _normalise(min(vol_surge, 3.0), 0.8, 3.0)

    # pullback_room: 5–20% below 52w high is the sweet spot
    # (at high = no room; too far below = broken trend)
    if 5.0 <= dist_from_high_pct <= 20.0:
        pullback_score = 1.0
    elif dist_from_high_pct < 5.0:
        pullback_score = 0.5   # very close to high — may be extended
    elif dist_from_high_pct <= 35.0:
        pullback_score = _normalise(35.0 - dist_from_high_pct, 0, 15.0)
    else:
        pullback_score = 0.0   # too far from highs

    composite = (
        _WEIGHTS["trend_align"]   * trend_score
        + _WEIGHTS["momentum"]      * momentum_score
        + _WEIGHTS["rel_strength"]  * rel_str_score
        + _WEIGHTS["vol_surge"]     * vol_surge_score
        + _WEIGHTS["pullback_room"] * pullback_score
    )

    return ScreenerScore(
        symbol=symbol,
        name=name,
        sector=sector,
        price=price,
        sma50=sma50,
        sma200=sma200,
        rsi14=rsi14,
        atr_pct=atr_pct,
        high_52w=high_52w,
        dist_from_high_pct=dist_from_high_pct,
        rel_strength_1m=rel_strength_1m,
        vol_surge_ratio=vol_surge,
        traded_value_20d_cr=traded_value_20d_cr,
        composite_score=composite,
        factors={
            "trend_align":   round(trend_score, 3),
            "momentum":      round(momentum_score, 3),
            "rel_strength":  round(rel_str_score, 3),
            "vol_surge":     round(vol_surge_score, 3),
            "pullback_room": round(pullback_score, 3),
        },
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def run_screen(
    universe: str = "dynamic",
    max_candidates: int = 25,
    min_liquidity_cr: float = 10.0,
    as_of: Optional[date] = None,
) -> list[ScreenerScore]:
    """Run the quantitative screen (Stages 1–2) and return ranked candidates.

    Args:
        universe:         "dynamic" (default) or a static name like "nifty50",
                          "nifty200", "midcap150", "nifty500".
                          "dynamic" uses the lazy-refreshed NSE-sourced universe.
        max_candidates:   Maximum results to return (top by composite score).
        min_liquidity_cr: Minimum 20-day median traded value in crores (₹ Cr).
                          The dynamic universe builder pre-filters at ₹5 Cr;
                          this re-applies a stricter threshold on fresher data.
        as_of:            Reference date (today if None) — used for F&O ban filter.

    Returns:
        List of ScreenerScore objects sorted by composite_score descending.
    """
    from .universe import get_universe, load_universe

    try:
        from tradingagents.default_config import DEFAULT_CONFIG
        min_price = float(DEFAULT_CONFIG.get("min_stock_price", 50))
        max_price = DEFAULT_CONFIG.get("max_stock_price")
        max_price = float(max_price) if max_price is not None else None
    except Exception:
        min_price = 50.0
        max_price = None

    if universe == "dynamic":
        try:
            from tradingagents.default_config import DEFAULT_CONFIG
            entries = get_universe(DEFAULT_CONFIG)
        except Exception:
            entries = get_universe()
    else:
        entries = load_universe(universe)
    if not entries:
        logger.warning("Universe '%s' is empty", universe)
        return []

    # F&O ban list (best-effort; screen proceeds if unavailable)
    fno_banned: set[str] = set()
    try:
        from tradingagents.dataflows.india.nse_client import get_fno_ban_list
        ban_str = get_fno_ban_list(as_of)
        if "data unavailable" not in ban_str:
            for token in ban_str.split(":")[1:]:
                for s in token.split(","):
                    sym = s.strip().rstrip(".")
                    if sym:
                        fno_banned.add(sym)
    except Exception as exc:
        logger.debug("F&O ban list unavailable: %s", exc)

    all_symbols = [e.symbol for e in entries]
    symbol_meta = {e.symbol: (e.name, e.sector) for e in entries}

    # Download Nifty 50 for relative-strength baseline
    nifty_df_map = _download_batch(["^NSEI"], period="1y")
    nifty_df = nifty_df_map.get("^NSEI")
    nifty_1m_return = 0.0
    if nifty_df is not None and len(nifty_df) >= 22:
        try:
            nifty_close = nifty_df.get("Close") or nifty_df.get("close")
            if nifty_close is not None:
                p_now = float(nifty_close.iloc[-1])
                p_1m = float(nifty_close.iloc[-22])
                nifty_1m_return = (p_now - p_1m) / p_1m * 100.0 if p_1m > 0 else 0.0
        except Exception:
            pass

    # Download universe in batches of 50 (yfinance limit)
    batch_size = 50
    all_data: dict = {}
    for i in range(0, len(all_symbols), batch_size):
        batch = all_symbols[i: i + batch_size]
        data = _download_batch(batch, period="1y")
        all_data.update(data)

    scores: list[ScreenerScore] = []
    for entry in entries:
        sym = entry.symbol
        nse_bare = sym.replace(".NS", "").replace(".BO", "")

        # F&O ban filter
        if nse_bare in fno_banned:
            logger.debug("Skipping %s — in F&O ban list", sym)
            continue

        df = all_data.get(sym)
        name, sector = symbol_meta.get(sym, (sym, "Unknown"))

        sc = _score_stock(sym, name, sector, df, nifty_1m_return)
        if sc is None:
            continue

        # Price filter
        if sc.price < min_price:
            logger.debug("Skipping %s — price ₹%.0f < min ₹%.0f", sym, sc.price, min_price)
            continue
        if max_price is not None and sc.price > max_price:
            logger.debug("Skipping %s — price ₹%.0f > max ₹%.0f", sym, sc.price, max_price)
            continue

        # Liquidity filter
        if sc.traded_value_20d_cr < min_liquidity_cr:
            logger.debug("Skipping %s — liquidity ₹%.1fCr < min ₹%.0fCr", sym, sc.traded_value_20d_cr, min_liquidity_cr)
            continue

        # ATR sanity band: skip hyper-volatile (>6% daily ATR) or dead stocks (<0.3%)
        if sc.atr_pct > 6.0 or sc.atr_pct < 0.3:
            logger.debug("Skipping %s — ATR%% %.2f%% outside [0.3, 6.0]%%", sym, sc.atr_pct)
            continue

        scores.append(sc)

    scores.sort(key=lambda s: s.composite_score, reverse=True)
    return scores[:max_candidates]
