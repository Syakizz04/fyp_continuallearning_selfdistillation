"""
Persistence for the central stock pool.

`PoolStore` is a narrow interface with two implementations:

- `RedisPoolStore`   - the real one. The conditional decrement runs as a Lua
                       script, so the atomicity is Redis's, not ours.
- `SqlitePoolStore`  - same semantics via an IMMEDIATE transaction. Keeps the
                       unit tests runnable with nothing installed, and is a
                       usable fallback for a single-process run.

Why the atomicity matters
-------------------------
`take(sku, qty)` is a compare-and-decrement: read the free pool, take at most
what is there, never go below zero. Done as separate GET and DECRBY calls it is
a race - two callers can both read 5, both take 5, and the pool goes to -5.

That race is the whole subject of E1, which means the *control arm must not
share the method's implementation*. If `strong_lock` were implemented with the
same hand-written locking as `escrow_quota`, comparing them would only compare
one author's code against itself. Delegating strong consistency to a Lua script
executed atomically by Redis makes it an independent, standard primitive.

Note honestly: with a single uvicorn worker, Python's GIL already serialises the
service's own handlers, so SQLite would be sufficient. Redis earns its place
when the service runs multiple workers, and as the independent primitive above.
"""

from __future__ import annotations

import sqlite3
import threading
from pathlib import Path
from typing import Dict, Optional, Protocol, runtime_checkable

# Take at most `qty` from the free pool, never below zero; return what was taken.
_TAKE_LUA = """
local free = tonumber(redis.call('GET', KEYS[1]) or '0')
local want = tonumber(ARGV[1])
if want <= 0 then return 0 end
local take = math.min(free, want)
if take > 0 then
    redis.call('DECRBY', KEYS[1], take)
end
return take
"""

# Unconditional take, allowed to drive the pool negative. Used ONLY by the
# `eventual` arm, whose defining property is that it can oversell.
_TAKE_UNSAFE_LUA = """
local want = tonumber(ARGV[1])
if want <= 0 then return 0 end
redis.call('DECRBY', KEYS[1], want)
return want
"""


@runtime_checkable
class PoolStore(Protocol):
    """Durable central pool. Implementations must make `take` atomic."""

    def init_sku(self, sku: str, total: int) -> None: ...
    def take(self, sku: str, qty: int, *, allow_negative: bool = False) -> int: ...
    def give_back(self, sku: str, qty: int) -> None: ...
    def record_commit(self, sku: str, qty: int) -> None: ...
    def free(self, sku: str) -> int: ...
    def snapshot(self, sku: str) -> Dict[str, int]: ...
    def skus(self) -> list: ...
    def reset(self) -> None: ...
    def close(self) -> None: ...


class RedisPoolStore:
    """Redis-backed pool. `take` is a Lua compare-and-decrement."""

    def __init__(self, url: str, *, prefix: str = "fyp:inv:") -> None:
        try:
            import redis  # noqa: PLC0415 - optional dependency, imported on use
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "redis-py is required for RedisPoolStore. "
                "pip install redis, and start the server with: "
                "docker compose up -d redis"
            ) from exc
        self._r = redis.Redis.from_url(url, decode_responses=True)
        self._prefix = prefix
        self._take = self._r.register_script(_TAKE_LUA)
        self._take_unsafe = self._r.register_script(_TAKE_UNSAFE_LUA)
        # Fail fast and loudly: a silent fallback to SQLite here would quietly
        # invalidate E1 by swapping out the control arm's primitive.
        self._r.ping()

    def _k(self, kind: str, sku: str) -> str:
        return f"{self._prefix}{kind}:{sku}"

    def init_sku(self, sku: str, total: int) -> None:
        pipe = self._r.pipeline()
        pipe.set(self._k("total", sku), total)
        pipe.set(self._k("free", sku), total)
        pipe.setnx(self._k("committed", sku), 0)
        pipe.sadd(f"{self._prefix}skus", sku)
        pipe.execute()

    def take(self, sku: str, qty: int, *, allow_negative: bool = False) -> int:
        script = self._take_unsafe if allow_negative else self._take
        return int(script(keys=[self._k("free", sku)], args=[int(qty)]))

    def give_back(self, sku: str, qty: int) -> None:
        if qty:
            self._r.incrby(self._k("free", sku), int(qty))

    def record_commit(self, sku: str, qty: int) -> None:
        if qty:
            self._r.incrby(self._k("committed", sku), int(qty))

    def free(self, sku: str) -> int:
        return int(self._r.get(self._k("free", sku)) or 0)

    def snapshot(self, sku: str) -> Dict[str, int]:
        vals = self._r.mget([self._k("total", sku), self._k("free", sku),
                             self._k("committed", sku)])
        total, free, committed = (int(v or 0) for v in vals)
        return {"total": total, "free": free, "committed": committed}

    def skus(self) -> list:
        return sorted(self._r.smembers(f"{self._prefix}skus"))

    def reset(self) -> None:
        keys = self._r.keys(f"{self._prefix}*")
        if keys:
            self._r.delete(*keys)

    def close(self) -> None:
        self._r.close()


class SqlitePoolStore:
    """
    SQLite pool with the same semantics.

    `take` runs inside a BEGIN IMMEDIATE transaction, which takes the database's
    write lock before reading, so the read-modify-write cannot interleave.
    """

    def __init__(self, path: str | Path = ":memory:") -> None:
        self._path = str(path)
        if self._path != ":memory:":
            Path(self._path).parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(self._path, check_same_thread=False,
                                     isolation_level=None)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS pool (
                sku       TEXT PRIMARY KEY,
                total     INTEGER NOT NULL,
                free      INTEGER NOT NULL,
                committed INTEGER NOT NULL DEFAULT 0
            )
        """)

    def init_sku(self, sku: str, total: int) -> None:
        with self._lock:
            self._conn.execute(
                "INSERT INTO pool (sku, total, free, committed) VALUES (?,?,?,0) "
                "ON CONFLICT(sku) DO UPDATE SET total=excluded.total, free=excluded.free",
                (sku, total, total),
            )

    def take(self, sku: str, qty: int, *, allow_negative: bool = False) -> int:
        qty = int(qty)
        if qty <= 0:
            return 0
        with self._lock:
            self._conn.execute("BEGIN IMMEDIATE")
            try:
                row = self._conn.execute(
                    "SELECT free FROM pool WHERE sku=?", (sku,)).fetchone()
                if row is None:
                    self._conn.execute("ROLLBACK")
                    return 0
                free = int(row[0])
                take = qty if allow_negative else max(0, min(free, qty))
                if take:
                    self._conn.execute(
                        "UPDATE pool SET free = free - ? WHERE sku=?", (take, sku))
                self._conn.execute("COMMIT")
                return take
            except Exception:
                self._conn.execute("ROLLBACK")
                raise

    def give_back(self, sku: str, qty: int) -> None:
        if qty:
            with self._lock:
                self._conn.execute(
                    "UPDATE pool SET free = free + ? WHERE sku=?", (int(qty), sku))

    def record_commit(self, sku: str, qty: int) -> None:
        if qty:
            with self._lock:
                self._conn.execute(
                    "UPDATE pool SET committed = committed + ? WHERE sku=?",
                    (int(qty), sku))

    def free(self, sku: str) -> int:
        row = self._conn.execute("SELECT free FROM pool WHERE sku=?", (sku,)).fetchone()
        return int(row[0]) if row else 0

    def snapshot(self, sku: str) -> Dict[str, int]:
        row = self._conn.execute(
            "SELECT total, free, committed FROM pool WHERE sku=?", (sku,)).fetchone()
        if row is None:
            return {"total": 0, "free": 0, "committed": 0}
        return {"total": int(row[0]), "free": int(row[1]), "committed": int(row[2])}

    def skus(self) -> list:
        return [r[0] for r in self._conn.execute("SELECT sku FROM pool ORDER BY sku")]

    def reset(self) -> None:
        with self._lock:
            self._conn.execute("DELETE FROM pool")

    def close(self) -> None:
        self._conn.close()


def make_pool_store(
    backend: str = "redis",
    *,
    url: Optional[str] = None,
    path: Optional[str] = None,
    prefix: str = "fyp:inv:",
) -> PoolStore:
    """
    Build a PoolStore. `backend` is "redis", "sqlite", or "auto".

    "auto" prefers Redis and falls back to SQLite with a warning. Do not use it
    for E1 runs - which backend served the control arm has to be recorded, not
    discovered, so the experiment scripts pass an explicit backend.
    """
    backend = backend.lower()
    if backend == "redis":
        from ..config import SYSTEM_CONFIG
        return RedisPoolStore(url or SYSTEM_CONFIG["redis"]["url"], prefix=prefix)
    if backend == "sqlite":
        return SqlitePoolStore(path or ":memory:")
    if backend == "auto":
        try:
            return make_pool_store("redis", url=url, prefix=prefix)
        except Exception as exc:
            import warnings
            warnings.warn(
                f"Redis unavailable ({exc}); falling back to SqlitePoolStore. "
                "Do not use 'auto' for E1 - pass an explicit backend so the "
                "result records which primitive was used.",
                RuntimeWarning, stacklevel=2,
            )
            return SqlitePoolStore(path or ":memory:")
    raise ValueError(f"unknown backend {backend!r}; use redis|sqlite|auto")
