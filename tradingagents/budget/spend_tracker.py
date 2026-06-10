"""Monthly LLM spend tracker for Phase 8 small-account mode.

Accumulates token-based cost estimates in ~/.tradingagents/llm_spend.json
keyed by "YYYY-MM". The CLI warns at 80% of budget and alerts at 100%.

Cost rates are rough INR approximations — update _MODEL_COST_INR_PER_1K_TOKENS
as provider pricing changes. The tracker's job is to flag runaway spend, not
provide accurate billing.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

_DEFAULT_SPEND_PATH = Path.home() / ".tradingagents" / "llm_spend.json"

# Approximate INR per 1,000 tokens (input + output combined).
# Exchange rate assumed: ≈ ₹84/USD.
_MODEL_COST_INR_PER_1K_TOKENS: dict[str, float] = {
    # OpenAI
    "gpt-5.5": 1.08,           # ~$0.013/1k combined (est.)
    "gpt-5.4-mini": 0.13,      # ~$0.0015/1k combined
    "gpt-4o": 0.84,
    "gpt-4o-mini": 0.13,
    # Anthropic Claude
    "claude-opus-4-8": 2.52,
    "claude-sonnet-4-6": 0.63,
    "claude-haiku-4-5": 0.10,
    # Google Gemini
    "gemini-2.5-pro": 0.63,
    "gemini-2.5-flash": 0.08,
    # Fallback
    "default": 0.63,
}


def _rate_for(model: str) -> float:
    """Return INR/1k-token rate, falling back to the default."""
    if model in _MODEL_COST_INR_PER_1K_TOKENS:
        return _MODEL_COST_INR_PER_1K_TOKENS[model]
    # Partial match (e.g. "gpt-4o-mini-2024" → "gpt-4o-mini")
    for key, rate in _MODEL_COST_INR_PER_1K_TOKENS.items():
        if key != "default" and key in model:
            return rate
    return _MODEL_COST_INR_PER_1K_TOKENS["default"]


class SpendTracker:
    """Persistent monthly LLM spend accumulator.

    Thread-safe enough for sequential CLI use (no concurrent writes expected).
    """

    def __init__(self, spend_path: Path | None = None):
        self.path = Path(spend_path) if spend_path else _DEFAULT_SPEND_PATH
        self.path.parent.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load(self) -> dict:
        if self.path.exists():
            try:
                return json.loads(self.path.read_text(encoding="utf-8"))
            except Exception:
                pass
        return {}

    def _save(self, data: dict) -> None:
        self.path.write_text(json.dumps(data, indent=2), encoding="utf-8")

    @staticmethod
    def _month_key() -> str:
        return datetime.now().strftime("%Y-%m")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def record_spend(self, amount_inr: float) -> None:
        """Add amount_inr to this month's running total."""
        data = self._load()
        key = self._month_key()
        data[key] = round(data.get(key, 0.0) + amount_inr, 4)
        self._save(data)

    def estimate_and_record(
        self,
        tokens_in: int,
        tokens_out: int,
        model: str = "default",
    ) -> float:
        """Estimate cost from token counts, record it, and return estimated INR."""
        rate = _rate_for(model)
        amount_inr = (tokens_in + tokens_out) / 1000.0 * rate
        if amount_inr > 0:
            self.record_spend(amount_inr)
        return amount_inr

    def get_monthly_spend(self) -> float:
        """Return this month's accumulated spend in INR."""
        return self._load().get(self._month_key(), 0.0)

    def get_all_months(self) -> dict[str, float]:
        """Return the full spend history keyed by YYYY-MM."""
        return self._load()

    def check_budget(self, monthly_budget_inr: float) -> dict:
        """Return budget-status dict.

        Keys:
            spend_inr    — this month's spend
            budget_inr   — configured monthly cap
            pct_used     — spend / budget * 100
            warning      — True when pct_used >= 80
            over_budget  — True when pct_used >= 100
            message      — human-readable status line
        """
        spend = self.get_monthly_spend()
        pct = (spend / monthly_budget_inr * 100.0) if monthly_budget_inr > 0 else 0.0
        over = pct >= 100.0
        warn = pct >= 80.0

        if over:
            msg = (
                f"LLM budget EXCEEDED: ₹{spend:.0f} spent of ₹{monthly_budget_inr:.0f} "
                f"({pct:.0f}% used). Consider pausing non-critical analysis."
            )
        elif warn:
            msg = (
                f"LLM budget at {pct:.0f}%: ₹{spend:.0f} of ₹{monthly_budget_inr:.0f} "
                "used this month."
            )
        else:
            msg = (
                f"LLM budget: ₹{spend:.0f} / ₹{monthly_budget_inr:.0f} "
                f"({pct:.0f}% used this month)."
            )

        return {
            "spend_inr": spend,
            "budget_inr": monthly_budget_inr,
            "pct_used": pct,
            "warning": warn,
            "over_budget": over,
            "message": msg,
        }
