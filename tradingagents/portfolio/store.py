"""SQLite-backed position store for open swing-trade positions.

Database path: ~/.tradingagents/portfolio.db

Schema
------
positions
    id               INTEGER PRIMARY KEY AUTOINCREMENT
    ticker           TEXT NOT NULL
    direction        TEXT NOT NULL    -- LONG | SHORT
    status           TEXT NOT NULL    -- OPEN | CLOSED
    opened_date      TEXT NOT NULL    -- ISO date (YYYY-MM-DD)
    entries_json     TEXT             -- JSON list of {label, price, allocation_pct, trigger, rationale}
    avg_entry        REAL             -- filled weighted-avg entry (set on first FILL event)
    qty_open         INTEGER          -- remaining open quantity after partial exits
    qty_total        INTEGER          -- original total quantity
    stop_loss        REAL
    take_profits_json TEXT            -- JSON list of {label, price, exit_pct, basis}
    thesis           TEXT             -- original investment thesis / setup_type
    setup_type       TEXT
    signal_json      TEXT             -- full TradeSignal.model_dump() JSON
    last_review_date TEXT             -- ISO date of most recent review
    closed_date      TEXT             -- ISO date when position was fully closed
    realized_r       REAL             -- realized R-multiple (set on close)
    notes            TEXT

position_events
    id           INTEGER PRIMARY KEY AUTOINCREMENT
    position_id  INTEGER NOT NULL REFERENCES positions(id)
    ts           TEXT NOT NULL        -- ISO datetime
    type         TEXT NOT NULL        -- FILL | PARTIAL_EXIT | SL_MOVE | REVIEW | CLOSE
    payload_json TEXT                 -- JSON with event-specific fields
"""

from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Generator, Optional

_DEFAULT_DB_PATH = Path.home() / ".tradingagents" / "portfolio.db"

_DDL = """
CREATE TABLE IF NOT EXISTS positions (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    ticker           TEXT NOT NULL,
    direction        TEXT NOT NULL,
    status           TEXT NOT NULL DEFAULT 'OPEN',
    opened_date      TEXT NOT NULL,
    entries_json     TEXT,
    avg_entry        REAL,
    qty_open         INTEGER,
    qty_total        INTEGER,
    stop_loss        REAL,
    take_profits_json TEXT,
    thesis           TEXT,
    setup_type       TEXT,
    signal_json      TEXT,
    last_review_date TEXT,
    closed_date      TEXT,
    realized_r       REAL,
    notes            TEXT
);

CREATE TABLE IF NOT EXISTS position_events (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    position_id  INTEGER NOT NULL REFERENCES positions(id),
    ts           TEXT NOT NULL,
    type         TEXT NOT NULL,
    payload_json TEXT
);
"""


def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


class PositionStore:
    """Thread-safe SQLite position store.

    All write operations commit immediately; reads return plain dicts so callers
    never depend on sqlite3 Row objects.
    """

    def __init__(self, db_path: Optional[Path] = None):
        self.db_path = Path(db_path) if db_path else _DEFAULT_DB_PATH
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @contextmanager
    def _conn(self) -> Generator[sqlite3.Connection, None, None]:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    def _init_db(self) -> None:
        with self._conn() as conn:
            conn.executescript(_DDL)

    @staticmethod
    def _row_to_dict(row: sqlite3.Row) -> dict:
        d = dict(row)
        for key in ("entries_json", "take_profits_json", "signal_json", "payload_json"):
            if key in d and d[key] is not None:
                try:
                    d[key] = json.loads(d[key])
                except (json.JSONDecodeError, TypeError):
                    pass
        return d

    def _add_event(
        self,
        conn: sqlite3.Connection,
        position_id: int,
        event_type: str,
        payload: dict,
    ) -> None:
        conn.execute(
            "INSERT INTO position_events (position_id, ts, type, payload_json) VALUES (?,?,?,?)",
            (position_id, _now_iso(), event_type, json.dumps(payload)),
        )

    # ------------------------------------------------------------------
    # Public write API
    # ------------------------------------------------------------------

    def add_position(
        self,
        ticker: str,
        direction: str,
        signal: dict,
        opened_date: Optional[str] = None,
        notes: str = "",
    ) -> int:
        """Create a new OPEN position from a TradeSignal dict.

        Returns the new position id.
        """
        if direction not in ("LONG", "SHORT"):
            raise ValueError(f"direction must be LONG or SHORT, got {direction!r}")

        date = opened_date or datetime.now().strftime("%Y-%m-%d")
        entries = signal.get("entries", [])
        tps = signal.get("take_profits", [])
        sl = signal.get("stop_loss")
        setup = signal.get("setup_type", "")
        thesis = signal.get("invalidation", "")  # best description of the trade idea pre-entry

        with self._conn() as conn:
            cursor = conn.execute(
                """INSERT INTO positions
                   (ticker, direction, status, opened_date,
                    entries_json, stop_loss, take_profits_json,
                    setup_type, thesis, signal_json, notes)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
                (
                    ticker.upper(),
                    direction,
                    "OPEN",
                    date,
                    json.dumps(entries),
                    sl,
                    json.dumps(tps),
                    setup,
                    thesis,
                    json.dumps(signal),
                    notes,
                ),
            )
            position_id = cursor.lastrowid
            self._add_event(conn, position_id, "CREATE", {"signal": signal, "opened_date": date})
        return position_id

    def record_fill(
        self,
        position_id: int,
        avg_entry: float,
        qty: int,
    ) -> None:
        """Record that entries were filled at avg_entry for qty shares."""
        with self._conn() as conn:
            conn.execute(
                "UPDATE positions SET avg_entry=?, qty_open=?, qty_total=? WHERE id=?",
                (avg_entry, qty, qty, position_id),
            )
            self._add_event(conn, position_id, "FILL", {"avg_entry": avg_entry, "qty": qty})

    def record_partial_exit(
        self,
        position_id: int,
        qty_exited: int,
        exit_price: float,
        label: str = "",
    ) -> None:
        """Record a partial exit (e.g. TP1 hit). Updates qty_open."""
        with self._conn() as conn:
            row = conn.execute("SELECT qty_open FROM positions WHERE id=?", (position_id,)).fetchone()
            if row is None:
                raise LookupError(f"Position {position_id} not found")
            new_qty = max(0, (row["qty_open"] or 0) - qty_exited)
            conn.execute("UPDATE positions SET qty_open=? WHERE id=?", (new_qty, position_id))
            self._add_event(
                conn,
                position_id,
                "PARTIAL_EXIT",
                {"qty_exited": qty_exited, "exit_price": exit_price, "label": label, "qty_remaining": new_qty},
            )

    def move_stop_loss(
        self,
        position_id: int,
        new_sl: float,
    ) -> None:
        """Move stop-loss. Caller is responsible for direction validation (see validator)."""
        with self._conn() as conn:
            old_row = conn.execute("SELECT stop_loss FROM positions WHERE id=?", (position_id,)).fetchone()
            old_sl = old_row["stop_loss"] if old_row else None
            conn.execute("UPDATE positions SET stop_loss=? WHERE id=?", (new_sl, position_id))
            self._add_event(conn, position_id, "SL_MOVE", {"old_sl": old_sl, "new_sl": new_sl})

    def record_review(
        self,
        position_id: int,
        review_date: str,
        action: str,
        reasoning: str,
        thesis_status: str,
        position_action_dict: Optional[dict] = None,
    ) -> None:
        """Append a management review event."""
        with self._conn() as conn:
            conn.execute(
                "UPDATE positions SET last_review_date=? WHERE id=?",
                (review_date, position_id),
            )
            self._add_event(
                conn,
                position_id,
                "REVIEW",
                {
                    "date": review_date,
                    "action": action,
                    "reasoning": reasoning,
                    "thesis_status": thesis_status,
                    "position_action": position_action_dict,
                },
            )

    def close_position(
        self,
        position_id: int,
        exit_price: float,
        realized_r: float,
        closed_date: Optional[str] = None,
    ) -> None:
        """Mark position CLOSED, record realized R-multiple."""
        date = closed_date or datetime.now().strftime("%Y-%m-%d")
        with self._conn() as conn:
            conn.execute(
                "UPDATE positions SET status='CLOSED', closed_date=?, realized_r=?, qty_open=0 WHERE id=?",
                (date, realized_r, position_id),
            )
            self._add_event(
                conn,
                position_id,
                "CLOSE",
                {"exit_price": exit_price, "realized_r": realized_r, "closed_date": date},
            )

    def update_notes(self, position_id: int, notes: str) -> None:
        with self._conn() as conn:
            conn.execute("UPDATE positions SET notes=? WHERE id=?", (notes, position_id))

    # ------------------------------------------------------------------
    # Public read API
    # ------------------------------------------------------------------

    def get_position(self, position_id: int) -> Optional[dict]:
        with self._conn() as conn:
            row = conn.execute("SELECT * FROM positions WHERE id=?", (position_id,)).fetchone()
        return self._row_to_dict(row) if row else None

    def get_open_position(self, ticker: str) -> Optional[dict]:
        """Return the most-recent OPEN position for a ticker, or None."""
        with self._conn() as conn:
            row = conn.execute(
                "SELECT * FROM positions WHERE ticker=? AND status='OPEN' ORDER BY id DESC LIMIT 1",
                (ticker.upper(),),
            ).fetchone()
        return self._row_to_dict(row) if row else None

    def list_positions(self, status: Optional[str] = None) -> list[dict]:
        """Return positions filtered by status. None = all."""
        with self._conn() as conn:
            if status:
                rows = conn.execute(
                    "SELECT * FROM positions WHERE status=? ORDER BY id DESC",
                    (status.upper(),),
                ).fetchall()
            else:
                rows = conn.execute("SELECT * FROM positions ORDER BY id DESC").fetchall()
        return [self._row_to_dict(r) for r in rows]

    def get_events(self, position_id: int) -> list[dict]:
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT * FROM position_events WHERE position_id=? ORDER BY id",
                (position_id,),
            ).fetchall()
        return [self._row_to_dict(r) for r in rows]

    def get_open_risk_pct(self, account_equity: float) -> float:
        """Compute total open risk across all OPEN positions as % of equity.

        Uses avg_entry, stop_loss, qty_open from each position.
        Returns 0.0 if equity is zero or no open positions.
        """
        if account_equity <= 0:
            return 0.0
        positions = self.list_positions(status="OPEN")
        total_risk = 0.0
        for pos in positions:
            avg_entry = pos.get("avg_entry")
            sl = pos.get("stop_loss")
            qty = pos.get("qty_open") or 0
            direction = pos.get("direction", "LONG")
            if avg_entry is None or sl is None or qty == 0:
                continue
            sign = 1 if direction == "LONG" else -1
            risk_per_share = sign * (avg_entry - sl)
            if risk_per_share > 0:
                total_risk += risk_per_share * qty
        return total_risk / account_equity * 100.0

    def compute_position_context_block(
        self,
        position: dict,
        current_price: Optional[float] = None,
    ) -> str:
        """Render a markdown block injected into every agent's context in manage mode.

        Includes: entry, current P&L in R-multiples, days held, original thesis,
        current SL/TPs, and upcoming event risks from the original signal.
        """
        ticker = position.get("ticker", "UNKNOWN")
        direction = position.get("direction", "LONG")
        avg_entry = position.get("avg_entry")
        qty_open = position.get("qty_open") or 0
        qty_total = position.get("qty_total") or qty_open
        sl = position.get("stop_loss")
        opened_date = position.get("opened_date", "")
        thesis = position.get("thesis", "")
        setup_type = position.get("setup_type", "")

        # Days held
        days_held = 0
        if opened_date:
            try:
                from datetime import date
                d0 = date.fromisoformat(opened_date)
                days_held = (date.today() - d0).days
            except (ValueError, TypeError):
                pass

        # R-multiple current P&L
        r_multiple_str = "unknown"
        if avg_entry and sl and current_price:
            sign = 1 if direction == "LONG" else -1
            risk_per_share = sign * (avg_entry - sl)
            if risk_per_share > 0:
                r_mult = sign * (current_price - avg_entry) / risk_per_share
                r_multiple_str = f"{r_mult:+.2f}R"

        # Take-profits
        tps = position.get("take_profits_json") or []
        if isinstance(tps, str):
            try:
                tps = json.loads(tps)
            except (json.JSONDecodeError, TypeError):
                tps = []

        tp_lines = []
        for tp in tps:
            tp_lines.append(
                f"  {tp.get('label','TP?')}: ₹{tp.get('price', 0):,.0f} "
                f"(exit {tp.get('exit_pct', 0):.0f}%) — {tp.get('basis','')}"
            )

        # Event risks from original signal
        signal = position.get("signal_json") or {}
        if isinstance(signal, str):
            try:
                signal = json.loads(signal)
            except (json.JSONDecodeError, TypeError):
                signal = {}
        event_risks = signal.get("event_risks", [])

        current_price_str = f"₹{current_price:,.2f}" if current_price else "unknown"
        avg_entry_str = f"₹{avg_entry:,.2f}" if avg_entry else "not filled"
        sl_str = f"₹{sl:,.2f}" if sl else "not set"
        qty_str = f"{qty_open}/{qty_total} shares remaining"

        lines = [
            "---",
            f"OPEN POSITION — {ticker} — {direction} — {setup_type}",
            f"  Opened: {opened_date} ({days_held}d held) | Qty: {qty_str}",
            f"  Avg Entry: {avg_entry_str} | Current Price: {current_price_str}",
            f"  Current P&L: {r_multiple_str}",
            f"  Stop Loss: {sl_str}",
        ]
        if tp_lines:
            lines.append("  Take Profits:")
            lines.extend(tp_lines)
        if thesis:
            lines.append(f"  Original Thesis: {thesis}")
        if event_risks:
            lines.append(f"  Event Risks: {'; '.join(event_risks)}")
        lines.append("---")
        return "\n".join(lines)
