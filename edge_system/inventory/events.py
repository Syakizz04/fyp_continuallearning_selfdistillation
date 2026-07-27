"""
Durable event log for the inventory service.

Append-only SQLite, deliberately off the hot path: the escrow decisions happen
in memory and against Redis, and events are recorded afterwards. The log exists
so E1/E2 have a queryable record and so the dashboard can show live flow without
reaching into service internals.

Every row is one thing that happened to stock. `staleness_units` is written on
every reserve because it is the quantity E2 needs and it can only be observed at
the moment the decision was made - reconstructing it afterwards is not possible.
"""

from __future__ import annotations

import sqlite3
import threading
import time
from pathlib import Path
from typing import Dict, Iterable, List, Optional

SCHEMA = """
CREATE TABLE IF NOT EXISTS events (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    ts              REAL    NOT NULL,
    tick            INTEGER,
    sim_date        TEXT,
    kind            TEXT    NOT NULL,   -- reserve|commit|release|expire|refill|oversell|replenish
    node            TEXT,
    sku             TEXT,
    qty             INTEGER,
    reservation_id  TEXT,
    policy          TEXT,
    granted         INTEGER,            -- 1/0 for reserve outcomes
    latency_ms      REAL,
    true_available  INTEGER,            -- ground truth at decision time
    node_view       INTEGER,            -- what the node believed
    staleness_units INTEGER             -- true_available - node_view
);
CREATE INDEX IF NOT EXISTS idx_events_tick ON events(tick);
CREATE INDEX IF NOT EXISTS idx_events_kind ON events(kind);
CREATE INDEX IF NOT EXISTS idx_events_sku  ON events(sku);
"""

_FIELDS = (
    "ts", "tick", "sim_date", "kind", "node", "sku", "qty", "reservation_id",
    "policy", "granted", "latency_ms", "true_available", "node_view",
    "staleness_units",
)


class EventLog:
    """Append-only event store. Thread-safe; batches writes to keep the hot path cheap."""

    def __init__(self, path: str | Path, *, batch_size: int = 200) -> None:
        self.path = str(path)
        if self.path != ":memory:":
            Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(self.path, check_same_thread=False,
                                     isolation_level=None)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=NORMAL")
        self._conn.executescript(SCHEMA)
        self._lock = threading.Lock()
        self._buf: List[tuple] = []
        self._batch_size = batch_size

    def record(self, kind: str, **fields) -> None:
        fields.setdefault("ts", time.time())
        fields["kind"] = kind
        row = tuple(fields.get(f) for f in _FIELDS)
        with self._lock:
            self._buf.append(row)
            if len(self._buf) >= self._batch_size:
                self._flush_locked()

    def flush(self) -> None:
        with self._lock:
            self._flush_locked()

    def _flush_locked(self) -> None:
        if not self._buf:
            return
        placeholders = ",".join("?" * len(_FIELDS))
        self._conn.executemany(
            f"INSERT INTO events ({','.join(_FIELDS)}) VALUES ({placeholders})",
            self._buf,
        )
        self._buf.clear()

    # ── Queries used by the dashboard and the experiment scripts ─────────────

    def recent(self, limit: int = 100, kind: Optional[str] = None) -> List[Dict]:
        self.flush()
        sql = "SELECT * FROM events"
        args: tuple = ()
        if kind:
            sql += " WHERE kind = ?"
            args = (kind,)
        sql += " ORDER BY id DESC LIMIT ?"
        cur = self._conn.execute(sql, args + (limit,))
        cols = [d[0] for d in cur.description]
        return [dict(zip(cols, r)) for r in cur.fetchall()]

    def counts_by_kind(self) -> Dict[str, int]:
        self.flush()
        return dict(self._conn.execute(
            "SELECT kind, COUNT(*) FROM events GROUP BY kind").fetchall())

    def staleness_series(self, sku: Optional[str] = None) -> List[Dict]:
        """Per-tick mean staleness — E2's independent variable, as realised."""
        self.flush()
        sql = ("SELECT tick, node, AVG(staleness_units) AS mean_staleness, "
               "COUNT(*) AS n FROM events WHERE kind='reserve' "
               "AND staleness_units IS NOT NULL")
        args: tuple = ()
        if sku:
            sql += " AND sku = ?"
            args = (sku,)
        sql += " GROUP BY tick, node ORDER BY tick"
        cur = self._conn.execute(sql, args)
        cols = [d[0] for d in cur.description]
        return [dict(zip(cols, r)) for r in cur.fetchall()]

    def to_csv(self, path: str | Path) -> Path:
        """Dump for outputs/system/results/ in the long form the dashboard reads."""
        import pandas as pd
        self.flush()
        df = pd.read_sql_query("SELECT * FROM events", self._conn)
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(path, index=False)
        return path

    def close(self) -> None:
        self.flush()
        self._conn.close()
