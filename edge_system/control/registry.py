"""
Model registry and event log for the control plane.

The cloud side of the system. Deliberately thin: it records **metadata** about
what each node is serving and when it retrained - never training data, never
model weights, never customer history. That restraint is the architecture
carrying the thesis. A control plane that collected raw history to retrain
centrally would be the MLaaS design the project argues against, and would make
the privacy motivation for replay-free CL hollow.

SQLite because the write rate is a handful of rows per retrain. Redis is for the
inventory hot path; this is not one.
"""

from __future__ import annotations

import json
import sqlite3
import threading
import time
from pathlib import Path
from typing import Dict, List, Optional

SCHEMA = """
CREATE TABLE IF NOT EXISTS model_versions (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    ts          REAL NOT NULL,
    node        TEXT NOT NULL,
    strategy    TEXT,
    generation  INTEGER,
    kind        TEXT,            -- forecasting | rl
    sim_date    TEXT,
    tick        INTEGER,
    reason      TEXT,
    duration_s  REAL,
    ok          INTEGER,
    skipped     INTEGER,
    error       TEXT
);
CREATE INDEX IF NOT EXISTS idx_mv_node ON model_versions(node);

CREATE TABLE IF NOT EXISTS node_state (
    node        TEXT PRIMARY KEY,
    ts          REAL NOT NULL,
    payload     TEXT NOT NULL     -- JSON blob of the node's last /health
);
"""


class Registry:
    def __init__(self, path: str | Path) -> None:
        self.path = str(path)
        if self.path != ":memory:":
            Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(self.path, check_same_thread=False,
                                     isolation_level=None)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.executescript(SCHEMA)
        self._lock = threading.Lock()

    # ── Writes ──────────────────────────────────────────────────────────────

    def register(self, payload: Dict) -> int:
        """Record a retrain outcome reported by a node."""
        with self._lock:
            cur = self._conn.execute(
                "INSERT INTO model_versions "
                "(ts, node, strategy, generation, kind, sim_date, tick, reason, "
                " duration_s, ok, skipped, error) "
                "VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                (
                    time.time(), payload.get("node"), payload.get("strategy"),
                    payload.get("generation"), payload.get("kind"),
                    payload.get("sim_date"), payload.get("tick"),
                    payload.get("reason"), payload.get("duration_s"),
                    1 if payload.get("ok") else 0,
                    1 if payload.get("skipped") else 0,
                    payload.get("error"),
                ),
            )
            return int(cur.lastrowid)

    def heartbeat(self, node: str, payload: Dict) -> None:
        with self._lock:
            self._conn.execute(
                "INSERT INTO node_state (node, ts, payload) VALUES (?,?,?) "
                "ON CONFLICT(node) DO UPDATE SET ts=excluded.ts, payload=excluded.payload",
                (node, time.time(), json.dumps(payload)),
            )

    # ── Reads ───────────────────────────────────────────────────────────────

    def versions(self, node: Optional[str] = None, limit: int = 100) -> List[Dict]:
        sql = "SELECT * FROM model_versions"
        args: tuple = ()
        if node:
            sql += " WHERE node = ?"
            args = (node,)
        sql += " ORDER BY id DESC LIMIT ?"
        cur = self._conn.execute(sql, args + (limit,))
        cols = [d[0] for d in cur.description]
        return [dict(zip(cols, r)) for r in cur.fetchall()]

    def nodes(self) -> List[Dict]:
        out = []
        for node, ts, payload in self._conn.execute(
                "SELECT node, ts, payload FROM node_state ORDER BY node"):
            out.append({"node": node, "ts": ts, "age_s": time.time() - ts,
                        **json.loads(payload)})
        return out

    def summary(self) -> Dict:
        rows = self._conn.execute(
            "SELECT node, strategy, COUNT(*), SUM(ok), SUM(skipped), "
            "       SUM(COALESCE(duration_s,0)), MAX(generation) "
            "FROM model_versions GROUP BY node, strategy").fetchall()
        return {
            "nodes": [
                {"node": r[0], "strategy": r[1], "retrains": r[2],
                 "ok": r[3], "skipped": r[4], "total_retrain_s": r[5],
                 "generation": r[6]}
                for r in rows
            ],
            "total_retrains": sum(r[2] for r in rows),
        }

    def close(self) -> None:
        self._conn.close()
