"""SQLite storage (stdlib sqlite3, WAL). Small query methods, no ORM (docs/DECISIONS.md D14)."""

import json
import sqlite3
import threading
import uuid
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

from weather_risk.sources.open_meteo import DailyRow

SCHEMA = """
CREATE TABLE IF NOT EXISTS weather_daily (
    hub_id TEXT NOT NULL, date TEXT NOT NULL,
    snowfall_cm REAL, precip_mm REAL, tmax_c REAL, tmin_c REAL, gust_kmh REAL,
    PRIMARY KEY (hub_id, date)
);
CREATE TABLE IF NOT EXISTS http_cache (
    key TEXT PRIMARY KEY, source TEXT NOT NULL, fetched_at TEXT NOT NULL, payload TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS score_snapshots (
    id INTEGER PRIMARY KEY, hub_id TEXT NOT NULL, kind TEXT NOT NULL, computed_at TEXT NOT NULL,
    score REAL NOT NULL, band TEXT, breakdown TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS ix_snapshots ON score_snapshots (kind, hub_id, id);
CREATE TABLE IF NOT EXISTS alerts (
    id INTEGER PRIMARY KEY, hub_id TEXT NOT NULL, created_at TEXT NOT NULL,
    prev_band TEXT, new_band TEXT NOT NULL, prev_score REAL, new_score REAL NOT NULL,
    drivers TEXT NOT NULL, delivery TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS conversations (id TEXT PRIMARY KEY, user_email TEXT, created_at TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS turns (
    id INTEGER PRIMARY KEY, conversation_id TEXT NOT NULL, created_at TEXT NOT NULL,
    user_text TEXT NOT NULL, plan TEXT, answer TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS source_status (
    source TEXT PRIMARY KEY, last_success_at TEXT, last_error TEXT, last_error_at TEXT
);
"""

DAILY_COLS = ("snowfall_cm", "precip_mm", "tmax_c", "tmin_c", "gust_kmh")


def utcnow() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


class Store:
    def __init__(self, path: Path | str):
        if path != ":memory:":
            Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(path), check_same_thread=False, isolation_level=None)
        self.conn.row_factory = sqlite3.Row
        if path != ":memory:":
            self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.executescript(SCHEMA)
        self._lock = threading.RLock()

    def _exec(self, sql: str, params: tuple | list = ()) -> sqlite3.Cursor:
        with self._lock:
            return self.conn.execute(sql, params)

    # -- weather_daily ------------------------------------------------------
    def upsert_daily(self, hub_id: str, rows: list[DailyRow]) -> None:
        with self._lock:
            self.conn.execute("BEGIN")
            self.conn.executemany(
                "INSERT OR REPLACE INTO weather_daily VALUES (?, ?, ?, ?, ?, ?, ?)",
                [(hub_id, r.date.isoformat(), *(getattr(r, c) for c in DAILY_COLS)) for r in rows],
            )
            self.conn.execute("COMMIT")

    def daily(self, hub_id: str, start: date, end: date) -> list[DailyRow]:
        cur = self._exec(
            "SELECT * FROM weather_daily WHERE hub_id = ? AND date BETWEEN ? AND ? ORDER BY date",
            (hub_id, start.isoformat(), end.isoformat()),
        )
        return [DailyRow(date=date.fromisoformat(r["date"]), **{c: r[c] for c in DAILY_COLS}) for r in cur]

    def daily_range(self, hub_id: str) -> tuple[date, date] | None:
        row = self._exec("SELECT MIN(date) a, MAX(date) b FROM weather_daily WHERE hub_id = ?", (hub_id,)).fetchone()
        if row["a"] is None:
            return None
        return date.fromisoformat(row["a"]), date.fromisoformat(row["b"])

    def has_weather(self) -> bool:
        return self._exec("SELECT 1 FROM weather_daily LIMIT 1").fetchone() is not None

    # -- http_cache (parsed payloads read whole) ----------------------------
    def cache_put(self, key: str, source: str, payload: Any, fetched_at: str | None = None) -> None:
        self._exec(
            "INSERT OR REPLACE INTO http_cache VALUES (?, ?, ?, ?)",
            (key, source, fetched_at or utcnow(), json.dumps(payload, default=str)),
        )

    def cache_get(self, key: str) -> tuple[str, Any] | None:
        row = self._exec("SELECT fetched_at, payload FROM http_cache WHERE key = ?", (key,)).fetchone()
        return (row["fetched_at"], json.loads(row["payload"])) if row else None

    # -- score_snapshots ----------------------------------------------------
    def save_snapshot(self, hub_id: str, kind: str, score: float, band: str | None, breakdown: dict) -> None:
        self._exec(
            "INSERT INTO score_snapshots (hub_id, kind, computed_at, score, band, breakdown) VALUES (?, ?, ?, ?, ?, ?)",
            (hub_id, kind, utcnow(), score, band, json.dumps(breakdown, default=str)),
        )

    def latest_snapshots(self, kind: str) -> dict[str, dict]:
        cur = self._exec(
            """SELECT s.* FROM score_snapshots s
               JOIN (SELECT hub_id, MAX(id) id FROM score_snapshots WHERE kind = ? GROUP BY hub_id) m ON s.id = m.id""",
            (kind,),
        )
        return {r["hub_id"]: _snapshot(r) for r in cur}

    # -- alerts -------------------------------------------------------------
    def add_alert(self, hub_id: str, prev_band: str | None, new_band: str, prev_score: float | None,
                  new_score: float, drivers: list[str], delivery: str) -> int:
        cur = self._exec(
            "INSERT INTO alerts (hub_id, created_at, prev_band, new_band, prev_score, new_score, drivers, delivery)"
            " VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (hub_id, utcnow(), prev_band, new_band, prev_score, new_score, json.dumps(drivers), delivery),
        )
        return cur.lastrowid

    def set_alert_delivery(self, alert_id: int, delivery: str) -> None:
        self._exec("UPDATE alerts SET delivery = ? WHERE id = ?", (delivery, alert_id))

    def recent_alerts(self, limit: int = 20) -> list[dict]:
        cur = self._exec("SELECT * FROM alerts ORDER BY id DESC LIMIT ?", (limit,))
        return [dict(r) | {"drivers": json.loads(r["drivers"])} for r in cur]

    # -- conversations ------------------------------------------------------
    def create_conversation(self, user_email: str | None) -> str:
        cid = uuid.uuid4().hex
        self._exec("INSERT INTO conversations VALUES (?, ?, ?)", (cid, user_email, utcnow()))
        return cid

    def conversation_exists(self, cid: str) -> bool:
        return self._exec("SELECT 1 FROM conversations WHERE id = ?", (cid,)).fetchone() is not None

    def add_turn(self, cid: str, user_text: str, plan: dict | None, answer: dict) -> None:
        self._exec(
            "INSERT INTO turns (conversation_id, created_at, user_text, plan, answer) VALUES (?, ?, ?, ?, ?)",
            (cid, utcnow(), user_text, json.dumps(plan, default=str) if plan else None, json.dumps(answer, default=str)),
        )

    def recent_turns(self, cid: str, limit: int = 3) -> list[dict]:
        cur = self._exec(
            "SELECT * FROM turns WHERE conversation_id = ? ORDER BY id DESC LIMIT ?", (cid, limit)
        )
        rows = [
            {
                "user_text": r["user_text"],
                "plan": json.loads(r["plan"]) if r["plan"] else None,
                "answer": json.loads(r["answer"]),
            }
            for r in cur
        ]
        return list(reversed(rows))

    # -- source_status ------------------------------------------------------
    def source_ok(self, source: str, at: str | None = None) -> None:
        self._exec(
            "INSERT INTO source_status (source, last_success_at) VALUES (?, ?) "
            "ON CONFLICT(source) DO UPDATE SET last_success_at = excluded.last_success_at",
            (source, at or utcnow()),
        )

    def source_error(self, source: str, error: str) -> None:
        self._exec(
            "INSERT INTO source_status (source, last_error, last_error_at) VALUES (?, ?, ?) "
            "ON CONFLICT(source) DO UPDATE SET last_error = excluded.last_error, last_error_at = excluded.last_error_at",
            (source, error[:300], utcnow()),
        )

    def source_statuses(self) -> dict[str, dict]:
        return {r["source"]: dict(r) for r in self._exec("SELECT * FROM source_status")}


def _snapshot(row: sqlite3.Row) -> dict:
    return {
        "hub_id": row["hub_id"],
        "kind": row["kind"],
        "computed_at": row["computed_at"],
        "score": row["score"],
        "band": row["band"],
        "breakdown": json.loads(row["breakdown"]),
    }
