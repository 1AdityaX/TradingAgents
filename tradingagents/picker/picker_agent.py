"""Picker agent: ONE LLM call that ranks pre-filtered candidates.

Cost control principle:
  - Screener: free (no LLM, pure pandas + yfinance)
  - Picker ranking: 1 LLM call using quick_think_llm
  - Deep pipeline: runs only on top N picks

The agent receives all candidate cards + market regime context and returns a
structured PickerOutput with top N tickers, setup hypotheses, what the deep
analysis should verify, and rejection rationale for major passes.

Prompt rule: "Rank only on the data in the cards. Do not use memorised
knowledge about these companies — it is stale and unverifiable."
"""

from __future__ import annotations

import logging
from datetime import date
from typing import Literal, Optional

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Structured output schemas
# ---------------------------------------------------------------------------

class PickerSelection(BaseModel):
    """One top-ranked stock from the picker agent."""

    ticker: str = Field(
        description="NSE ticker symbol, e.g. 'RELIANCE.NS'.",
    )
    company_name: str = Field(
        description="Company name as shown in the candidate card.",
    )
    rank: int = Field(
        description="Rank among the picks (1 = highest conviction).",
    )
    setup_hypothesis: str = Field(
        description=(
            "One concise sentence describing the suspected trade setup, "
            "e.g. 'pullback-to-support in established uptrend' or "
            "'volume-confirmed breakout above consolidation zone'. "
            "Must be derivable from the card data, not memorised knowledge."
        ),
    )
    verify_in_analysis: str = Field(
        description=(
            "Specific questions the deep analysis pipeline should answer "
            "to confirm or reject this hypothesis. "
            "E.g. 'Confirm support level holds on daily chart; check Q4 "
            "results date falls outside 20-session swing window'."
        ),
    )
    confidence: Literal["high", "medium", "low"] = Field(
        description=(
            "Confidence in this pick based solely on the card data. "
            "High = multiple factors align; Low = single factor, weak signal."
        ),
    )
    key_metrics: str = Field(
        description=(
            "Two or three bullet-point metrics from the card that most "
            "support this selection, e.g. 'RSI 52 (momentum), RS +4.2% vs "
            "Nifty (relative strength), vol surge 1.8x (accumulation)'."
        ),
    )


class PickerOutput(BaseModel):
    """Structured output from the picker agent ranking call."""

    top_picks: list[PickerSelection] = Field(
        description=(
            "Top N stocks to run through the full analysis pipeline. "
            "Ordered by rank (rank=1 first). Minimum 1, maximum 5."
        ),
    )
    rejected_highlights: str = Field(
        description=(
            "Brief rationale for why 2–3 notable candidates were not selected. "
            "Focus on the specific metrics that disqualified them "
            "(e.g. 'INFY: RSI 78 — overbought, no pullback room'). "
            "If all candidates were reasonable, note which factors separated the picks."
        ),
    )
    market_regime_assessment: str = Field(
        description=(
            "One short paragraph on the current market regime based on the "
            "context provided (Nifty trend, India VIX, FII/DII flows, sector "
            "rotation). State whether broad conditions favour momentum, "
            "mean-reversion, or are unclear/choppy."
        ),
    )
    scan_date: str = Field(
        description="ISO date of this scan, e.g. '2026-06-10'.",
    )


def render_picker_output(output: PickerOutput) -> str:
    """Render PickerOutput to human-readable markdown."""
    lines = [
        f"## Picker Output — {output.scan_date}",
        "",
        f"**Market Regime**: {output.market_regime_assessment}",
        "",
        "### Top Picks",
    ]
    for sel in output.top_picks:
        conf_map = {"high": "★★★", "medium": "★★☆", "low": "★☆☆"}
        lines += [
            f"",
            f"**{sel.rank}. {sel.ticker}** — {sel.company_name} {conf_map.get(sel.confidence, '')}",
            f"  *Setup*: {sel.setup_hypothesis}",
            f"  *Key metrics*: {sel.key_metrics}",
            f"  *Verify*: {sel.verify_in_analysis}",
        ]
    lines += [
        "",
        "### Rejected / Not Selected",
        output.rejected_highlights,
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Market regime context builder
# ---------------------------------------------------------------------------

def _build_market_regime_context(as_of: Optional[date] = None) -> str:
    """Assemble a compact market context block for the picker prompt."""
    ref = as_of or date.today()
    parts: list[str] = [f"MARKET REGIME CONTEXT — {ref}"]

    # Nifty 50 trend (last 50 days)
    try:
        import yfinance as yf
        nifty = yf.Ticker("^NSEI")
        hist = nifty.history(period="3mo", interval="1d")
        if not hist.empty:
            close = hist["Close"]
            last = float(close.iloc[-1])
            sma20 = float(close.tail(20).mean())
            sma50 = float(close.tail(50).mean())
            chg1m = (last - float(close.iloc[-22])) / float(close.iloc[-22]) * 100 if len(close) >= 22 else 0.0
            trend = "uptrend" if last > sma20 > sma50 else ("downtrend" if last < sma20 < sma50 else "sideways/mixed")
            parts.append(
                f"Nifty 50: {last:,.0f}  SMA20={sma20:,.0f}  SMA50={sma50:,.0f}  1m return={chg1m:+.1f}%  Trend={trend}"
            )
    except Exception as exc:
        parts.append(f"Nifty 50 data: unavailable ({exc})")

    # India VIX
    try:
        import yfinance as yf
        vix = yf.Ticker("^INDIAVIX")
        info = vix.history(period="5d", interval="1d")
        if not info.empty:
            vix_level = float(info["Close"].iloc[-1])
            vix_interp = "low fear" if vix_level < 14 else ("elevated fear" if vix_level > 20 else "normal")
            parts.append(f"India VIX: {vix_level:.1f} ({vix_interp})")
    except Exception:
        parts.append("India VIX: data unavailable")

    # FII/DII 10-day flows
    try:
        from tradingagents.dataflows.india.flows import get_fii_dii_flows
        flows_str = get_fii_dii_flows(as_of=ref, sessions=10)
        parts.append(flows_str[:600])  # cap length for prompt economy
    except Exception as exc:
        parts.append(f"FII/DII flows: data unavailable ({exc})")

    return "\n".join(parts)


# ---------------------------------------------------------------------------
# LLM ranking call
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = """\
You are the Stock Picker agent for an Indian equity swing-trading system.

Your only job is to select the best 3–5 stocks from the pre-filtered candidate \
list for further deep analysis. The quantitative screener has already removed low-liquidity, \
price-banned, and trend-less stocks.

RULES:
1. Rank ONLY on the data in the candidate cards. Do NOT use memorised knowledge about \
these companies — prices and fundamentals in your training data are stale and unverifiable.
2. Prefer stocks where multiple factors align (trend + momentum + relative strength + \
volume) over single-factor stories.
3. Flag any candidate with a promoter pledge > 5% or a results date inside a 20-session window as a risk.
4. NO_TRADE is a valid outcome if market regime is unfavourable — say so explicitly.
5. Keep setup_hypothesis and verify_in_analysis specific to the card data, not generic advice.
"""


def run_picker(
    cards: list[dict],
    top_n: int = 5,
    as_of: Optional[date] = None,
    llm=None,
) -> PickerOutput:
    """Run the picker agent: ONE LLM call → PickerOutput.

    Args:
        cards:  CandidateCard dicts from candidate.build_all_cards().
        top_n:  Maximum picks to select (1–5).
        as_of:  Reference date for market context (today if None).
        llm:    Pre-built LangChain LLM instance. If None, uses DEFAULT_CONFIG
                quick_think_llm with the configured provider.

    Returns:
        Structured PickerOutput. Falls back to a stub on total LLM failure.
    """
    from tradingagents.agents.utils.structured import bind_structured, invoke_structured_or_freetext
    from .candidate import format_card_for_prompt

    ref = as_of or date.today()

    # Build the LLM if not provided
    if llm is None:
        try:
            from tradingagents.llm_clients import create_llm_client
            from tradingagents.default_config import DEFAULT_CONFIG
            llm = create_llm_client(
                provider=DEFAULT_CONFIG["llm_provider"],
                model=DEFAULT_CONFIG["quick_think_llm"],
                backend_url=DEFAULT_CONFIG.get("backend_url"),
                temperature=DEFAULT_CONFIG.get("temperature"),
            )
        except Exception as exc:
            logger.error("Could not create LLM for picker: %s", exc)
            return _stub_output(cards, top_n, ref, reason=str(exc))

    # Build prompt
    market_ctx = _build_market_regime_context(as_of=ref)
    cards_text = "\n\n".join(format_card_for_prompt(c) for c in cards)

    prompt = f"""{_SYSTEM_PROMPT}

---
{market_ctx}

---
CANDIDATE CARDS ({len(cards)} stocks pre-filtered from screener):

{cards_text}

---
Select the top {min(top_n, len(cards))} stocks for deep analysis.
Return structured JSON matching the PickerOutput schema exactly.
scan_date must be '{ref}'.
"""

    # Structured output with free-text fallback
    structured_llm = bind_structured(llm, PickerOutput, "PickerAgent")
    try:
        result = invoke_structured_or_freetext(
            structured_llm=structured_llm,
            plain_llm=llm,
            prompt=prompt,
            render=lambda x: x,   # we want the Pydantic object, not rendered text
            agent_name="PickerAgent",
        )
        if isinstance(result, PickerOutput):
            return result
        # Free-text fallback returned a string — parse best effort
        return _parse_freetext_fallback(result, cards, top_n, ref)
    except Exception as exc:
        logger.error("Picker agent LLM call failed: %s", exc)
        return _stub_output(cards, top_n, ref, reason=str(exc))


# ---------------------------------------------------------------------------
# Fallbacks
# ---------------------------------------------------------------------------

def _stub_output(cards: list[dict], top_n: int, ref: date, reason: str = "") -> PickerOutput:
    """Return the top-N by composite_score when LLM is unavailable."""
    top = sorted(cards, key=lambda c: c.get("composite_score", 0), reverse=True)[:top_n]
    picks = []
    for i, c in enumerate(top, 1):
        picks.append(PickerSelection(
            ticker=c["symbol"],
            company_name=c["name"],
            rank=i,
            setup_hypothesis="Quantitative screener top pick — LLM ranking unavailable",
            verify_in_analysis="Run full analysis pipeline to evaluate setup",
            confidence="low",
            key_metrics=f"Score={c.get('composite_score', 0):.3f}  RSI={c.get('rsi14', 0):.0f}  RS={c.get('rel_strength_1m', 0):+.1f}%",
        ))
    return PickerOutput(
        top_picks=picks,
        rejected_highlights=f"LLM ranking unavailable ({reason}). Fallback to top-{top_n} by composite score.",
        market_regime_assessment="Market regime assessment unavailable — LLM call failed.",
        scan_date=str(ref),
    )


def _parse_freetext_fallback(text: str, cards: list[dict], top_n: int, ref: date) -> PickerOutput:
    """Best-effort parse of free-text LLM response; falls back to stub."""
    try:
        import json, re
        # Try to extract a JSON block
        match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
        if not match:
            match = re.search(r"(\{.*\})", text, re.DOTALL)
        if match:
            obj = json.loads(match.group(1))
            return PickerOutput(**obj)
    except Exception:
        pass
    return _stub_output(cards, top_n, ref, reason="freetext parse failed")
