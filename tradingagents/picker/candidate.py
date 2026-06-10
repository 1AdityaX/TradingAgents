"""CandidateCard builder: compact per-stock fact sheet for the picker agent.

Each card is a plain dict with deterministically fetched data — the picker
agent is explicitly instructed to rank only on the data in the cards, not
memorised knowledge about these companies.

Fields in a CandidateCard:
  symbol, name, sector, price, sma50, sma200, rsi14, atr_pct,
  dist_from_high_pct, rel_strength_1m, vol_surge_ratio,
  traded_value_20d_cr, composite_score, factor_breakdown,
  next_results_date, promoter_pledge_flag, headlines (list[str])
"""

from __future__ import annotations

import logging
from typing import Optional

from .screener import ScreenerScore

logger = logging.getLogger(__name__)


def build_candidate_card(score: ScreenerScore) -> dict:
    """Build a CandidateCard dict from a ScreenerScore plus India-specific data.

    India-specific enrichment (results date, pledge flag, news) is best-effort:
    each field carries "data unavailable" rather than a blank or fabricated value
    so the picker agent knows what's missing.
    """
    card: dict = {
        "symbol":              score.symbol,
        "name":                score.name,
        "sector":              score.sector,
        "price":               round(score.price, 2),
        "sma50":               round(score.sma50, 2),
        "sma200":              round(score.sma200, 2),
        "rsi14":               round(score.rsi14, 1),
        "atr_pct":             round(score.atr_pct, 2),
        "dist_from_high_pct":  round(score.dist_from_high_pct, 1),
        "rel_strength_1m":     round(score.rel_strength_1m, 2),
        "vol_surge_ratio":     round(score.vol_surge_ratio, 2),
        "traded_value_20d_cr": round(score.traded_value_20d_cr, 1),
        "composite_score":     round(score.composite_score, 3),
        "factor_breakdown":    score.factors,
        "trend":               _describe_trend(score),
        # Enriched fields — filled below
        "next_results_date":   "data unavailable",
        "promoter_pledge_flag": False,
        "promoter_pledge_note": "data unavailable",
        "headlines":           [],
    }

    # Results / board-meeting date
    try:
        from tradingagents.dataflows.india.corporate_actions import get_corporate_actions
        actions_str = get_corporate_actions(score.symbol)
        if "data unavailable" not in actions_str:
            card["next_results_date"] = _extract_results_date(actions_str)
    except Exception as exc:
        logger.debug("Corporate actions unavailable for %s: %s", score.symbol, exc)

    # Promoter pledge flag
    try:
        from tradingagents.dataflows.india.shareholding import get_shareholding_summary
        sh_str = get_shareholding_summary(score.symbol)
        if "data unavailable" not in sh_str:
            pledge_flag, pledge_note = _extract_pledge(sh_str)
            card["promoter_pledge_flag"] = pledge_flag
            card["promoter_pledge_note"] = pledge_note
    except Exception as exc:
        logger.debug("Shareholding unavailable for %s: %s", score.symbol, exc)

    # Latest 3 headlines
    try:
        from tradingagents.dataflows.india.india_news import get_india_news
        news_str = get_india_news(score.symbol)
        if "data unavailable" not in news_str:
            card["headlines"] = _extract_headlines(news_str, max_count=3)
    except Exception as exc:
        logger.debug("News unavailable for %s: %s", score.symbol, exc)

    return card


def _describe_trend(score: ScreenerScore) -> str:
    """One-line trend description for the card."""
    p = score.price
    if p > score.sma50 and p > score.sma200:
        return "uptrend (above SMA50 and SMA200)"
    elif p > score.sma50:
        return "recovering (above SMA50, below SMA200)"
    elif p > score.sma200:
        return "weakening (below SMA50, above SMA200)"
    else:
        return "downtrend (below both SMAs)"


def _extract_results_date(actions_str: str) -> str:
    """Parse the nearest results/board-meeting date from a corporate actions string."""
    import re
    from datetime import date

    today = date.today()
    pattern = re.compile(r"\b(\d{4}-\d{2}-\d{2})\b")
    found_dates = []
    for match in pattern.finditer(actions_str):
        try:
            d = date.fromisoformat(match.group(1))
            if d >= today:
                found_dates.append(d)
        except ValueError:
            pass

    keywords = ("result", "board", "quarterly", "Q1", "Q2", "Q3", "Q4")
    lower = actions_str.lower()
    if found_dates and any(k.lower() in lower for k in keywords):
        return str(min(found_dates))

    return "none found in next 90 days" if not found_dates else str(min(found_dates))


def _extract_pledge(sh_str: str) -> tuple[bool, str]:
    """Return (pledge_flag, note) from shareholding summary string."""
    import re

    lower = sh_str.lower()
    # Look for pledge percentage
    match = re.search(r"pledge[d\s]+(?:shares?[:\s]+)?(\d+\.?\d*)\s*%", lower)
    if match:
        pct = float(match.group(1))
        flagged = pct > 5.0
        return flagged, f"pledged ~{pct:.1f}% of promoter shares"

    if "pledge" in lower:
        return False, "pledge data present but unparseable"
    return False, "no pledge data found"


def _extract_headlines(news_str: str, max_count: int = 3) -> list[str]:
    """Extract the first *max_count* headline lines from a news string."""
    headlines = []
    for line in news_str.splitlines():
        line = line.strip()
        # Skip empty lines and section headers
        if not line or line.startswith("#") or len(line) < 20:
            continue
        # Skip lines that are purely metadata (dates, sources)
        if line.startswith("Source:") or line.startswith("Date:"):
            continue
        headlines.append(line)
        if len(headlines) >= max_count:
            break
    return headlines


def format_card_for_prompt(card: dict) -> str:
    """Format a CandidateCard as compact text for the picker agent prompt."""
    trend_indicator = "▲" if "uptrend" in card["trend"] else ("◆" if "recovering" in card["trend"] else "▼")
    pledge_note = f" [PLEDGE: {card['promoter_pledge_note']}]" if card.get("promoter_pledge_flag") else ""

    lines = [
        f"--- {card['symbol']} | {card['name']} | {card['sector']} ---",
        f"Price: ₹{card['price']:,.0f}  {trend_indicator} {card['trend']}",
        f"SMA50: ₹{card['sma50']:,.0f}  SMA200: ₹{card['sma200']:,.0f}",
        f"RSI14: {card['rsi14']:.0f}  ATR%: {card['atr_pct']:.1f}%  Vol Surge: {card['vol_surge_ratio']:.1f}x",
        f"Dist from 52w High: {card['dist_from_high_pct']:.1f}%  RS vs Nifty (1m): {card['rel_strength_1m']:+.1f}%",
        f"Liquidity: ₹{card['traded_value_20d_cr']:.0f}Cr/day  Score: {card['composite_score']:.3f}",
        f"Next results: {card['next_results_date']}{pledge_note}",
    ]
    if card.get("headlines"):
        lines.append("Recent news:")
        for h in card["headlines"]:
            lines.append(f"  • {h[:120]}")

    return "\n".join(lines)


def build_all_cards(scores: list[ScreenerScore]) -> list[dict]:
    """Build CandidateCards for all ScreenerScores (best-effort enrichment)."""
    cards = []
    for sc in scores:
        try:
            card = build_candidate_card(sc)
            cards.append(card)
        except Exception as exc:
            logger.warning("Card build failed for %s: %s", sc.symbol, exc)
    return cards
