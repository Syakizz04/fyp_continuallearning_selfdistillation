"""
Durable state for the decision-support gateway: price rules, recommendations,
and the operator's decisions on them. SQLite - a handful of writes per simulated
day, off every hot path.

Recommendation lifecycle:

    pending ──approve──> approved
       │    ──reject───> rejected
       └──(a newer day's recommendation for the same SKU x channel)──> superseded

Only `pending` can be decided. A decision is final - there is no un-approve -
because an approval log that can be rewritten is not an audit trail. Every rule
change is logged with its before/after for the same reason.
"""

from __future__ import annotations

import json
import sqlite3
import threading
import time
from dataclasses import asdict, fields
from pathlib import Path
from typing import Dict, Iterable, List, Optional

from .guardrails import GlobalRule, SkuRule, validate_margin

SCHEMA = """
CREATE TABLE IF NOT EXISTS rules_global (
    id               INTEGER PRIMARY KEY CHECK (id = 1),
    min_margin_pct   REAL,
    max_discount_pct REAL,
    max_markup_pct   REAL,
    unit_cost_ratio  REAL NOT NULL,
    updated_at       REAL NOT NULL,
    updated_by       TEXT
);
CREATE TABLE IF NOT EXISTS rules_sku (
    sku              TEXT PRIMARY KEY,
    min_price        REAL,
    max_price        REAL,
    min_margin_pct   REAL,
    unit_cost        REAL,
    updated_at       REAL NOT NULL,
    updated_by       TEXT
);
CREATE TABLE IF NOT EXISTS rule_changes (
    id      INTEGER PRIMARY KEY AUTOINCREMENT,
    ts      REAL NOT NULL,
    user    TEXT,
    scope   TEXT NOT NULL,          -- global | sku
    sku     TEXT,
    before  TEXT,
    after   TEXT
);
CREATE TABLE IF NOT EXISTS recommendations (
    id                 INTEGER PRIMARY KEY AUTOINCREMENT,
    sku                TEXT NOT NULL,
    channel            TEXT NOT NULL,
    sim_date           TEXT NOT NULL,
    created_at         REAL NOT NULL,
    model_generation   INTEGER,
    tier               REAL,
    raw_price          REAL NOT NULL,
    base_price         REAL NOT NULL,
    final_price        REAL NOT NULL,
    unit_cost          REAL NOT NULL,
    margin_pct         REAL,
    raw_margin_pct     REAL,
    floor              REAL,
    ceiling            REAL,
    bound_by           TEXT NOT NULL,   -- JSON list
    conflict           INTEGER NOT NULL DEFAULT 0,
    inventory_view     REAL,
    true_available     REAL,
    forecast_p50_total REAL,
    status             TEXT NOT NULL DEFAULT 'pending',
    live_price         REAL,            -- what actually went on sale that day
    live_source        TEXT,            -- approved | auto | rejected | unreviewed
    UNIQUE (sku, channel, sim_date)
);
CREATE INDEX IF NOT EXISTS idx_rec_status ON recommendations(status);
CREATE INDEX IF NOT EXISTS idx_rec_sku ON recommendations(sku);
CREATE TABLE IF NOT EXISTS decisions (
    id                INTEGER PRIMARY KEY AUTOINCREMENT,
    recommendation_id INTEGER NOT NULL UNIQUE REFERENCES recommendations(id),
    action            TEXT NOT NULL CHECK (action IN ('approve', 'reject')),
    user              TEXT NOT NULL,
    note              TEXT,
    ts                REAL NOT NULL,
    applied_price     REAL
);
"""

STATUSES = ("pending", "approved", "rejected", "superseded")

_REC_COLS = ("sku", "channel", "sim_date", "model_generation", "tier", "raw_price",
             "base_price", "final_price", "unit_cost", "margin_pct", "raw_margin_pct",
             "floor", "ceiling", "bound_by", "conflict", "inventory_view",
             "true_available", "forecast_p50_total")


class DecisionError(Exception):
    """A decision the state machine does not allow. Carries an HTTP status."""

    def __init__(self, status: int, message: str) -> None:
        super().__init__(message)
        self.status = status


class Store:
    def __init__(self, path: str | Path, *, default_unit_cost_ratio: float = 0.70) -> None:
        self.path = str(path)
        if self.path != ":memory:":
            Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(self.path, check_same_thread=False,
                                     isolation_level=None)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA foreign_keys=ON")
        self._conn.executescript(SCHEMA)
        # A DB from before live prices existed: add the columns in place.
        have = {r["name"] for r in self._conn.execute("PRAGMA table_info(recommendations)")}
        for col, typ in (("live_price", "REAL"), ("live_source", "TEXT")):
            if col not in have:
                self._conn.execute(f"ALTER TABLE recommendations ADD COLUMN {col} {typ}")
        self._lock = threading.Lock()
        if self._conn.execute("SELECT 1 FROM rules_global").fetchone() is None:
            g = GlobalRule(unit_cost_ratio=default_unit_cost_ratio)
            self._conn.execute(
                "INSERT INTO rules_global VALUES (1, ?, ?, ?, ?, ?, ?)",
                (g.min_margin_pct, g.max_discount_pct, g.max_markup_pct,
                 g.unit_cost_ratio, time.time(), "system"))

    def close(self) -> None:
        self._conn.close()

    def clear_run(self, archive_to: Optional[str | Path] = None) -> None:
        """Start a new simulated run: drop recommendations and decisions, keep
        the operator's rules and their change log. Decisions are signed, so they
        are copied to `archive_to` first rather than simply destroyed."""
        with self._lock:
            if archive_to is not None:
                Path(archive_to).parent.mkdir(parents=True, exist_ok=True)
                dest = sqlite3.connect(str(archive_to))
                try:
                    self._conn.backup(dest)
                finally:
                    dest.close()
            self._conn.execute("DELETE FROM decisions")
            self._conn.execute("DELETE FROM recommendations")

    # ── Rules ───────────────────────────────────────────────────────────────

    def global_rule(self) -> GlobalRule:
        r = self._conn.execute("SELECT * FROM rules_global WHERE id = 1").fetchone()
        return GlobalRule(min_margin_pct=r["min_margin_pct"],
                          max_discount_pct=r["max_discount_pct"],
                          max_markup_pct=r["max_markup_pct"],
                          unit_cost_ratio=r["unit_cost_ratio"])

    def global_rule_row(self) -> Dict:
        return dict(self._conn.execute("SELECT * FROM rules_global WHERE id = 1").fetchone())

    def set_global_rule(self, rule: GlobalRule, *, user: str) -> GlobalRule:
        validate_margin(rule.min_margin_pct)
        for name in ("max_discount_pct", "max_markup_pct"):
            v = getattr(rule, name)
            if v is not None and not (0.0 <= v <= 1.0):
                raise ValueError(f"{name} must be in [0, 1], got {v}")
        if not (0.0 < rule.unit_cost_ratio < 1.0):
            raise ValueError(f"unit_cost_ratio must be in (0, 1), got {rule.unit_cost_ratio}: "
                             f"at 1 or above every sale at base price loses money")
        with self._lock:
            before = asdict(self.global_rule())
            self._conn.execute(
                "UPDATE rules_global SET min_margin_pct=?, max_discount_pct=?, "
                "max_markup_pct=?, unit_cost_ratio=?, updated_at=?, updated_by=? WHERE id=1",
                (rule.min_margin_pct, rule.max_discount_pct, rule.max_markup_pct,
                 rule.unit_cost_ratio, time.time(), user))
            self._log_rule_change(user, "global", None, before, asdict(rule))
        return rule

    def sku_rules(self) -> Dict[str, SkuRule]:
        names = [f.name for f in fields(SkuRule)]
        return {r["sku"]: SkuRule(**{n: r[n] for n in names})
                for r in self._conn.execute("SELECT * FROM rules_sku")}

    def sku_rule_rows(self) -> List[Dict]:
        return [dict(r) for r in self._conn.execute("SELECT * FROM rules_sku ORDER BY sku")]

    def set_sku_rule(self, sku: str, rule: SkuRule, *, user: str) -> SkuRule:
        validate_margin(rule.min_margin_pct)
        for name in ("min_price", "max_price", "unit_cost"):
            v = getattr(rule, name)
            if v is not None and v <= 0:
                raise ValueError(f"{name} must be positive, got {v}")
        if (rule.min_price is not None and rule.max_price is not None
                and rule.min_price > rule.max_price):
            raise ValueError(f"min_price {rule.min_price} is above max_price {rule.max_price}")
        with self._lock:
            before = self.sku_rules().get(sku)
            self._conn.execute(
                "INSERT INTO rules_sku VALUES (?, ?, ?, ?, ?, ?, ?) "
                "ON CONFLICT(sku) DO UPDATE SET min_price=excluded.min_price, "
                "max_price=excluded.max_price, min_margin_pct=excluded.min_margin_pct, "
                "unit_cost=excluded.unit_cost, updated_at=excluded.updated_at, "
                "updated_by=excluded.updated_by",
                (sku, rule.min_price, rule.max_price, rule.min_margin_pct,
                 rule.unit_cost, time.time(), user))
            self._log_rule_change(user, "sku", sku,
                                  asdict(before) if before else None, asdict(rule))
        return rule

    def delete_sku_rule(self, sku: str, *, user: str) -> bool:
        with self._lock:
            before = self.sku_rules().get(sku)
            if before is None:
                return False
            self._conn.execute("DELETE FROM rules_sku WHERE sku = ?", (sku,))
            self._log_rule_change(user, "sku", sku, asdict(before), None)
        return True

    def _log_rule_change(self, user, scope, sku, before, after) -> None:
        self._conn.execute(
            "INSERT INTO rule_changes (ts, user, scope, sku, before, after) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (time.time(), user, scope, sku,
             json.dumps(before) if before is not None else None,
             json.dumps(after) if after is not None else None))

    def rule_changes(self, limit: int = 100) -> List[Dict]:
        out = []
        for r in self._conn.execute(
                "SELECT * FROM rule_changes ORDER BY id DESC LIMIT ?", (limit,)):
            d = dict(r)
            d["before"] = json.loads(d["before"]) if d["before"] else None
            d["after"] = json.loads(d["after"]) if d["after"] else None
            out.append(d)
        return out

    # ── Recommendations ─────────────────────────────────────────────────────

    def add_recommendation(self, rec: Dict) -> Optional[int]:
        """Insert one recommendation; supersede older pending ones for the same
        SKU x channel. Idempotent per (sku, channel, sim_date): a repeat returns
        None and changes nothing, so re-polling a day is safe."""
        row = {c: rec.get(c) for c in _REC_COLS}
        row["bound_by"] = json.dumps(list(row["bound_by"] or []))
        row["conflict"] = int(bool(row["conflict"]))
        with self._lock:
            cur = self._conn.execute(
                f"INSERT OR IGNORE INTO recommendations (created_at, {', '.join(_REC_COLS)}) "
                f"VALUES (?, {', '.join('?' for _ in _REC_COLS)})",
                (time.time(), *[row[c] for c in _REC_COLS]))
            if cur.rowcount == 0:
                return None
            new_id = cur.lastrowid
            self._conn.execute(
                "UPDATE recommendations SET status = 'superseded' "
                "WHERE sku = ? AND channel = ? AND status = 'pending' AND sim_date < ?",
                (row["sku"], row["channel"], row["sim_date"]))
        return new_id

    def set_live(self, rec_id: int, price: float, source: str) -> None:
        with self._lock:
            self._conn.execute(
                "UPDATE recommendations SET live_price = ?, live_source = ? WHERE id = ?",
                (price, source, rec_id))

    def recommendation(self, rec_id: int) -> Optional[Dict]:
        rows = self.recommendations(ids=[rec_id], limit=1)
        return rows[0] if rows else None

    @staticmethod
    def _rec_where(status, sku, channel, sim_date, ids, q, bound) -> tuple:
        where, args = [], []
        if status:
            if status not in STATUSES:
                raise ValueError(f"unknown status {status!r}; have {STATUSES}")
            where.append("r.status = ?"); args.append(status)
        for col, val in (("r.sku", sku), ("r.channel", channel), ("r.sim_date", sim_date)):
            if val:
                where.append(f"{col} = ?"); args.append(val)
        if ids is not None:
            ids = list(ids)
            where.append(f"r.id IN ({', '.join('?' for _ in ids)})"); args += ids
        if q:
            where.append("r.sku LIKE ? ESCAPE '\\'")
            args.append("%" + q.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_") + "%")
        if bound:
            where.append("(r.bound_by != '[]' OR r.conflict)")
        return (" WHERE " + " AND ".join(where)) if where else "", args

    def recommendations(self, *, status: Optional[str] = None, sku: Optional[str] = None,
                        channel: Optional[str] = None, sim_date: Optional[str] = None,
                        ids: Optional[Iterable[int]] = None, q: Optional[str] = None,
                        bound: bool = False, limit: int = 200, offset: int = 0) -> List[Dict]:
        where, args = self._rec_where(status, sku, channel, sim_date, ids, q, bound)
        sql = ("SELECT r.*, d.action AS decision, d.user AS decided_by, d.note AS decision_note, "
               "d.ts AS decided_at, d.applied_price FROM recommendations r "
               "LEFT JOIN decisions d ON d.recommendation_id = r.id" + where +
               " ORDER BY r.sim_date DESC, r.id DESC LIMIT ? OFFSET ?")
        args += [limit, max(offset, 0)]
        out = []
        for r in self._conn.execute(sql, args):
            d = dict(r)
            d["bound_by"] = json.loads(d["bound_by"])
            d["conflict"] = bool(d["conflict"])
            out.append(d)
        return out

    def count_recommendations(self, *, status: Optional[str] = None,
                              sku: Optional[str] = None, channel: Optional[str] = None,
                              sim_date: Optional[str] = None, q: Optional[str] = None,
                              bound: bool = False) -> int:
        where, args = self._rec_where(status, sku, channel, sim_date, None, q, bound)
        return int(self._conn.execute(
            "SELECT COUNT(*) FROM recommendations r" + where, args).fetchone()[0])

    def batches(self, limit: int = 100) -> List[Dict]:
        """One row per simulated day's generation run, for the audit trail -
        listing every SKU x channel row individually would drown it."""
        return [dict(r) for r in self._conn.execute(
            "SELECT sim_date, COUNT(*) AS count, MIN(created_at) AS ts, "
            "SUM(bound_by != '[]') AS bound FROM recommendations "
            "GROUP BY sim_date ORDER BY sim_date DESC LIMIT ?", (limit,))]

    def counts(self) -> Dict[str, int]:
        got = dict(self._conn.execute(
            "SELECT status, COUNT(*) FROM recommendations GROUP BY status").fetchall())
        return {s: int(got.get(s, 0)) for s in STATUSES}

    # ── Decisions ───────────────────────────────────────────────────────────

    def decide(self, rec_id: int, action: str, *, user: str,
               note: Optional[str] = None) -> Dict:
        if action not in ("approve", "reject"):
            raise DecisionError(400, f"action must be approve or reject, got {action!r}")
        if not user or not user.strip():
            raise DecisionError(400, "a decision needs the name of who made it")
        with self._lock:
            rec = self.recommendation(rec_id)
            if rec is None:
                raise DecisionError(404, f"no recommendation {rec_id}")
            if rec["status"] != "pending":
                raise DecisionError(409, f"recommendation {rec_id} is {rec['status']}, "
                                         f"only pending ones can be decided")
            applied = rec["final_price"] if action == "approve" else None
            self._conn.execute("BEGIN")
            try:
                self._conn.execute(
                    "INSERT INTO decisions (recommendation_id, action, user, note, ts, "
                    "applied_price) VALUES (?, ?, ?, ?, ?, ?)",
                    (rec_id, action, user.strip(), note, time.time(), applied))
                self._conn.execute(
                    "UPDATE recommendations SET status = ? WHERE id = ?",
                    ("approved" if action == "approve" else "rejected", rec_id))
                self._conn.execute("COMMIT")
            except Exception:
                self._conn.execute("ROLLBACK")
                raise
        return self.recommendation(rec_id)

    def decisions(self, *, user: Optional[str] = None, limit: int = 200) -> List[Dict]:
        sql = ("SELECT d.*, r.sku, r.channel, r.sim_date, r.raw_price, r.final_price, "
               "r.base_price, r.bound_by FROM decisions d "
               "JOIN recommendations r ON r.id = d.recommendation_id")
        args: list = []
        if user:
            sql += " WHERE d.user = ?"; args.append(user)
        sql += " ORDER BY d.id DESC LIMIT ?"; args.append(limit)
        out = []
        for r in self._conn.execute(sql, args):
            d = dict(r)
            d["bound_by"] = json.loads(d["bound_by"])
            out.append(d)
        return out
