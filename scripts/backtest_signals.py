#!/usr/bin/env python3
"""Backtest harness — simulate TradeSignal EP/SL/TP outcomes on historical bars.

Usage
-----
# Simulate signals from a JSON file
python -m scripts.backtest_signals --signals signals.json

# Simulate signals from closed positions in the portfolio store
python -m scripts.backtest_signals --from-store

# Limit max holding period to 20 bars
python -m scripts.backtest_signals --signals signals.json --max-hold 20

# Compare against Nifty 50 benchmark
python -m scripts.backtest_signals --signals signals.json --benchmark ^NSEI

Signal JSON format
------------------
[
  {
    "ticker": "RELIANCE.NS",
    "signal_date": "2024-01-15",
    "direction": "LONG",
    "entries": [
      {"label": "EP1", "price": 2850.0, "allocation_pct": 60.0,
       "trigger": "limit at support", "rationale": "key support zone"},
      {"label": "EP2", "price": 2820.0, "allocation_pct": 40.0,
       "trigger": "limit at deeper support", "rationale": "second support"}
    ],
    "stop_loss": 2790.0,
    "take_profits": [
      {"label": "TP1", "price": 2950.0, "exit_pct": 50.0, "basis": "prior high"},
      {"label": "TP2", "price": 3050.0, "exit_pct": 50.0, "basis": "supply zone"}
    ]
  }
]

Simulation rules
----------------
- LONG entries fill when a bar's Low <= entry price (limit order)
- LONG SL hit when a bar's Low <= stop_loss
- LONG TP hit when a bar's High >= TP price
- SHORT entries fill when a bar's High >= entry price
- SHORT SL hit when a bar's High >= stop_loss
- SHORT TP hit when a bar's Low <= TP price
- On bars where both SL and TP would trigger, SL wins (conservative)
- Max holding period: --max-hold bars after first fill (default 30)
- If no entry fills in 10 bars, signal is marked NO_FILL

Output
------
Per-signal outcome table + summary statistics:
  Win rate, Average R (winners / losers / all), Max drawdown on R-curve,
  Comparison vs benchmark buy-and-hold over the same periods.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class SimSignal:
    """A single signal ready for backtesting."""
    ticker: str
    signal_date: str
    direction: str               # LONG | SHORT
    entries: list[dict]          # [{label, price, allocation_pct, ...}]
    stop_loss: float
    take_profits: list[dict]     # [{label, price, exit_pct, ...}]


@dataclass
class SimResult:
    """Outcome of simulating one signal on historical bars."""
    ticker: str
    signal_date: str
    direction: str
    avg_entry_price: Optional[float] = None
    exit_price: Optional[float] = None
    exit_date: Optional[str] = None
    exit_reason: str = "PENDING"       # WIN_TP1|WIN_TP2|...|LOSS_SL|EXPIRED|NO_FILL
    realized_r: Optional[float] = None
    # Benchmark
    benchmark_entry: Optional[float] = None
    benchmark_exit: Optional[float] = None
    benchmark_return_pct: Optional[float] = None
    signal_return_pct: Optional[float] = None
    bars_held: int = 0
    partial_exits: list[dict] = field(default_factory=list)


# ---------------------------------------------------------------------------
# OHLCV loader
# ---------------------------------------------------------------------------


def _fetch_ohlcv(ticker: str, start: str, end: str) -> list[dict]:
    """Return list of {date, open, high, low, close} dicts from yfinance.

    Returns an empty list on any failure.
    """
    try:
        import yfinance as yf
        import pandas as pd

        df = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=True)
        if df is None or df.empty:
            return []

        df = df.reset_index()
        # yfinance returns MultiIndex columns when only one ticker — flatten
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [c[0] if c[1] == "" else c[0] for c in df.columns]

        rows = []
        for _, row in df.iterrows():
            try:
                d = row["Date"]
                if hasattr(d, "date"):
                    d = d.date()
                rows.append({
                    "date": str(d),
                    "open":  float(row["Open"]),
                    "high":  float(row["High"]),
                    "low":   float(row["Low"]),
                    "close": float(row["Close"]),
                })
            except (KeyError, TypeError, ValueError):
                continue
        return rows
    except Exception as exc:
        logger.warning("OHLCV fetch failed for %s (%s–%s): %s", ticker, start, end, exc)
        return []


# ---------------------------------------------------------------------------
# Core simulation
# ---------------------------------------------------------------------------


def _date_plus_days(d: str, n: int) -> str:
    return str(date.fromisoformat(d) + timedelta(days=n))


def simulate_signal(
    sig: SimSignal,
    max_hold_bars: int = 30,
    max_fill_wait_bars: int = 10,
    benchmark: str = "^NSEI",
) -> SimResult:
    """Simulate one signal on daily OHLCV bars; return SimResult."""
    result = SimResult(
        ticker=sig.ticker,
        signal_date=sig.signal_date,
        direction=sig.direction,
    )

    if sig.direction not in ("LONG", "SHORT"):
        result.exit_reason = "NO_FILL"
        return result

    # Fetch price data: signal_date + 1 through signal_date + max_hold_bars + 20 calendar days
    fetch_end = _date_plus_days(sig.signal_date, max_hold_bars + 30)
    bars = _fetch_ohlcv(sig.ticker, _date_plus_days(sig.signal_date, 1), fetch_end)
    if not bars:
        result.exit_reason = "NO_DATA"
        return result

    # Benchmark bars for the same window
    bench_bars = _fetch_ohlcv(benchmark, _date_plus_days(sig.signal_date, 1), fetch_end)

    is_long = sig.direction == "LONG"

    # Weighted avg entry price
    total_alloc = sum(e["allocation_pct"] for e in sig.entries)
    if total_alloc > 0:
        avg_entry = sum(e["price"] * e["allocation_pct"] for e in sig.entries) / total_alloc
    else:
        avg_entry = sig.entries[0]["price"]

    risk_per_share = (avg_entry - sig.stop_loss) if is_long else (sig.stop_loss - avg_entry)
    if risk_per_share <= 0:
        result.exit_reason = "INVALID_SL"
        return result

    # Build TP ladder sorted by exit order
    tps = sorted(sig.take_profits, key=lambda t: t["price"] if is_long else -t["price"])

    # Simulate bar by bar
    filled = False
    fill_bar_idx = None
    tp_cursor = 0              # index into tps
    remaining_pct = 100.0     # position remaining
    weighted_exit_price = 0.0
    weighted_exit_pct = 0.0

    for bar_idx, bar in enumerate(bars):
        if not filled:
            # Check for any entry fill on this bar
            if bar_idx >= max_fill_wait_bars:
                result.exit_reason = "NO_FILL"
                return result

            for ep in sig.entries:
                if is_long and bar["low"] <= ep["price"]:
                    filled = True
                    fill_bar_idx = bar_idx
                    result.avg_entry_price = avg_entry
                    # Benchmark entry: open of same bar
                    if bench_bars and bar_idx < len(bench_bars):
                        result.benchmark_entry = bench_bars[bar_idx]["open"]
                    break
                if not is_long and bar["high"] >= ep["price"]:
                    filled = True
                    fill_bar_idx = bar_idx
                    result.avg_entry_price = avg_entry
                    if bench_bars and bar_idx < len(bench_bars):
                        result.benchmark_entry = bench_bars[bar_idx]["open"]
                    break
            continue

        # Position is open — check SL and TP
        bars_since_fill = bar_idx - fill_bar_idx
        result.bars_held = bars_since_fill

        # --- SL check ---
        sl_hit = (is_long and bar["low"] <= sig.stop_loss) or \
                 (not is_long and bar["high"] >= sig.stop_loss)

        # --- TP check (partial exits) ---
        tp_hit = False
        if tp_cursor < len(tps):
            tp = tps[tp_cursor]
            tp_hit = (is_long and bar["high"] >= tp["price"]) or \
                     (not is_long and bar["low"] <= tp["price"])

        # When both SL and TP trigger on the same bar, use SL (conservative)
        if sl_hit:
            exit_r = (sig.stop_loss - avg_entry) / risk_per_share if is_long else \
                     (avg_entry - sig.stop_loss) / risk_per_share
            result.exit_price = sig.stop_loss
            result.exit_date = bar["date"]
            result.exit_reason = "LOSS_SL"
            result.realized_r = exit_r
            _fill_benchmark(result, bench_bars, bar_idx)
            _compute_returns(result)
            return result

        if tp_hit:
            tp = tps[tp_cursor]
            exit_pct = tp["exit_pct"]
            weighted_exit_price += tp["price"] * exit_pct
            weighted_exit_pct += exit_pct
            remaining_pct -= exit_pct
            result.partial_exits.append({
                "label": tp.get("label", f"TP{tp_cursor+1}"),
                "price": tp["price"],
                "exit_pct": exit_pct,
                "date": bar["date"],
            })
            tp_cursor += 1

            if remaining_pct <= 0.01 or tp_cursor >= len(tps):
                # All TPs hit — position closed
                avg_tp_exit = weighted_exit_price / weighted_exit_pct
                exit_r = (avg_tp_exit - avg_entry) / risk_per_share if is_long else \
                         (avg_entry - avg_tp_exit) / risk_per_share
                result.exit_price = avg_tp_exit
                result.exit_date = bar["date"]
                result.exit_reason = f"WIN_TP{tp_cursor}"
                result.realized_r = exit_r
                _fill_benchmark(result, bench_bars, bar_idx)
                _compute_returns(result)
                return result

        # Max hold period check
        if bars_since_fill >= max_hold_bars:
            close_price = bar["close"]
            weighted_exit_price += close_price * remaining_pct
            weighted_exit_pct += remaining_pct
            avg_close_exit = weighted_exit_price / weighted_exit_pct if weighted_exit_pct > 0 else close_price
            exit_r = (avg_close_exit - avg_entry) / risk_per_share if is_long else \
                     (avg_entry - avg_close_exit) / risk_per_share
            result.exit_price = avg_close_exit
            result.exit_date = bar["date"]
            result.exit_reason = "EXPIRED"
            result.realized_r = exit_r
            _fill_benchmark(result, bench_bars, bar_idx)
            _compute_returns(result)
            return result

    # Ran out of bars with position still open (shouldn't happen if fetch window is big enough)
    if filled and bars:
        last = bars[-1]
        close_price = last["close"]
        result.exit_price = close_price
        result.exit_date = last["date"]
        result.exit_reason = "EXPIRED"
        result.realized_r = (close_price - avg_entry) / risk_per_share if is_long else \
                             (avg_entry - close_price) / risk_per_share
        _fill_benchmark(result, bench_bars, len(bars) - 1)
        _compute_returns(result)
    else:
        result.exit_reason = "NO_FILL"
    return result


def _fill_benchmark(result: SimResult, bench_bars: list, exit_bar_idx: int) -> None:
    if not bench_bars or result.benchmark_entry is None:
        return
    idx = min(exit_bar_idx, len(bench_bars) - 1)
    result.benchmark_exit = bench_bars[idx]["close"]


def _compute_returns(result: SimResult) -> None:
    if result.avg_entry_price and result.exit_price:
        if result.direction == "LONG":
            result.signal_return_pct = (result.exit_price - result.avg_entry_price) / result.avg_entry_price * 100
        else:
            result.signal_return_pct = (result.avg_entry_price - result.exit_price) / result.avg_entry_price * 100
    if result.benchmark_entry and result.benchmark_exit and result.benchmark_entry > 0:
        result.benchmark_return_pct = (result.benchmark_exit - result.benchmark_entry) / result.benchmark_entry * 100


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------


def _print_per_signal_table(results: list[SimResult]) -> None:
    header = f"{'Ticker':<18} {'Date':<12} {'Dir':<6} {'Result':<12} {'R':>6} {'Sig%':>7} {'Bench%':>8} {'Bars':>5}"
    print(header)
    print("-" * len(header))
    for r in results:
        r_str = f"{r.realized_r:+.2f}" if r.realized_r is not None else "  N/A"
        s_str = f"{r.signal_return_pct:+.1f}%" if r.signal_return_pct is not None else "  N/A"
        b_str = f"{r.benchmark_return_pct:+.1f}%" if r.benchmark_return_pct is not None else "  N/A"
        print(f"{r.ticker:<18} {r.signal_date:<12} {r.direction:<6} {r.exit_reason:<12} "
              f"{r_str:>6} {s_str:>7} {b_str:>8} {r.bars_held:>5}")


def _print_summary(results: list[SimResult]) -> None:
    completed = [r for r in results if r.realized_r is not None]
    if not completed:
        print("\nNo completed trades to summarise.")
        return

    total = len(completed)
    wins = [r for r in completed if r.realized_r is not None and r.realized_r > 0]
    losses = [r for r in completed if r.realized_r is not None and r.realized_r <= 0]
    no_fills = len([r for r in results if r.exit_reason in ("NO_FILL", "NO_DATA", "INVALID_SL")])

    win_rate = len(wins) / total * 100 if total else 0
    avg_r = sum(r.realized_r for r in completed) / total
    avg_r_wins = sum(r.realized_r for r in wins) / len(wins) if wins else 0.0
    avg_r_losses = sum(r.realized_r for r in losses) / len(losses) if losses else 0.0

    # R-curve drawdown
    r_curve: list[float] = []
    cumulative = 0.0
    peak = 0.0
    max_dd = 0.0
    for r in completed:
        cumulative += r.realized_r
        r_curve.append(cumulative)
        if cumulative > peak:
            peak = cumulative
        dd = peak - cumulative
        if dd > max_dd:
            max_dd = dd

    # Benchmark comparison
    bench_rets = [r.benchmark_return_pct for r in completed if r.benchmark_return_pct is not None]
    sig_rets = [r.signal_return_pct for r in completed if r.signal_return_pct is not None]
    avg_bench = sum(bench_rets) / len(bench_rets) if bench_rets else None
    avg_sig = sum(sig_rets) / len(sig_rets) if sig_rets else None

    print("\n" + "=" * 60)
    print("BACKTEST SUMMARY")
    print("=" * 60)
    print(f"  Signals analysed : {len(results)}")
    print(f"  Completed trades : {total}")
    print(f"  No-fills / errors: {no_fills}")
    print(f"  Wins             : {len(wins)}  ({win_rate:.1f}%)")
    print(f"  Losses           : {len(losses)}")
    print()
    print(f"  Avg R (all)      : {avg_r:+.2f}R")
    print(f"  Avg R (winners)  : {avg_r_wins:+.2f}R")
    print(f"  Avg R (losers)   : {avg_r_losses:+.2f}R")
    print(f"  Max R drawdown   : {max_dd:.2f}R")
    if avg_sig is not None:
        print()
        print(f"  Avg signal return: {avg_sig:+.1f}%")
    if avg_bench is not None:
        print(f"  Avg bench return : {avg_bench:+.1f}%")
        if avg_sig is not None:
            print(f"  Alpha vs bench   : {avg_sig - avg_bench:+.1f}%")
    print("=" * 60)

    # Expectancy (positive expectancy = edge)
    if total > 0:
        expectancy = (win_rate / 100 * avg_r_wins) + ((1 - win_rate / 100) * avg_r_losses)
        print(f"\n  Expectancy       : {expectancy:+.3f}R per trade")
        if expectancy > 0:
            print("  → Positive expectancy: system has a statistical edge.")
        else:
            print("  → Negative expectancy: review signal quality before sizing real capital.")
    print()


# ---------------------------------------------------------------------------
# Signal loaders
# ---------------------------------------------------------------------------


def _load_from_json(path: str) -> list[SimSignal]:
    with open(path) as f:
        data = json.load(f)
    signals = []
    for item in data:
        try:
            signals.append(SimSignal(
                ticker=item["ticker"],
                signal_date=item["signal_date"],
                direction=item["direction"].upper(),
                entries=item.get("entries", []),
                stop_loss=float(item["stop_loss"]),
                take_profits=item.get("take_profits", []),
            ))
        except (KeyError, TypeError, ValueError) as exc:
            logger.warning("Skipping malformed signal entry: %s — %s", item, exc)
    return signals


def _load_from_store() -> list[SimSignal]:
    """Load closed (and open) positions from the portfolio SQLite store."""
    try:
        from tradingagents.portfolio.store import PortfolioStore
        import json as _json

        store = PortfolioStore()
        positions = store.list_positions(status=None)  # all statuses
        signals = []
        for pos in positions:
            sig_json = pos.get("signal_json")
            if not sig_json:
                continue
            try:
                sig_dict = _json.loads(sig_json) if isinstance(sig_json, str) else sig_json
                entries = sig_dict.get("entries", [])
                tps = sig_dict.get("take_profits", [])
                sl = sig_dict.get("stop_loss")
                if not entries or sl is None:
                    continue
                signals.append(SimSignal(
                    ticker=pos["ticker"],
                    signal_date=pos["opened_date"],
                    direction=pos.get("direction", "LONG").upper(),
                    entries=entries,
                    stop_loss=float(sl),
                    take_profits=tps,
                ))
            except Exception as exc:
                logger.debug("Could not parse position %s: %s", pos.get("id"), exc)
        return signals
    except Exception as exc:
        logger.error("Could not load from portfolio store: %s", exc)
        return []


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    parser = argparse.ArgumentParser(
        description="Backtest TradeSignal EP/SL/TP outcomes on historical daily bars."
    )
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument(
        "--signals",
        metavar="FILE",
        help="JSON file with pre-generated signals (see module docstring for format).",
    )
    src.add_argument(
        "--from-store",
        action="store_true",
        help="Load signals from the portfolio SQLite store.",
    )
    parser.add_argument(
        "--max-hold",
        type=int,
        default=30,
        metavar="BARS",
        help="Maximum holding period in trading bars (default: 30).",
    )
    parser.add_argument(
        "--fill-wait",
        type=int,
        default=10,
        metavar="BARS",
        help="Max bars to wait for an entry fill before marking NO_FILL (default: 10).",
    )
    parser.add_argument(
        "--benchmark",
        default="^NSEI",
        metavar="TICKER",
        help="Benchmark ticker for buy-and-hold comparison (default: ^NSEI / Nifty 50).",
    )
    parser.add_argument(
        "--output-json",
        metavar="FILE",
        help="Optional: write per-signal results to a JSON file.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress per-signal table; show summary only.",
    )

    args = parser.parse_args()

    if args.signals:
        signals = _load_from_json(args.signals)
    else:
        signals = _load_from_store()

    if not signals:
        print("No signals to backtest.", file=sys.stderr)
        sys.exit(1)

    print(f"Backtesting {len(signals)} signal(s) | max_hold={args.max_hold} bars "
          f"| benchmark={args.benchmark}")
    print()

    results = []
    for i, sig in enumerate(signals, 1):
        print(f"  [{i}/{len(signals)}] {sig.ticker} {sig.signal_date} {sig.direction} ...",
              end="", flush=True)
        r = simulate_signal(
            sig,
            max_hold_bars=args.max_hold,
            max_fill_wait_bars=args.fill_wait,
            benchmark=args.benchmark,
        )
        results.append(r)
        r_str = f"{r.realized_r:+.2f}R" if r.realized_r is not None else r.exit_reason
        print(f" → {r.exit_reason} ({r_str})")

    print()
    if not args.quiet:
        _print_per_signal_table(results)
    _print_summary(results)

    if args.output_json:
        import dataclasses
        out = [dataclasses.asdict(r) for r in results]
        with open(args.output_json, "w") as f:
            json.dump(out, f, indent=2)
        print(f"Results written to {args.output_json}")


if __name__ == "__main__":
    main()
