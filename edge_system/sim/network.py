"""
Simulated node <-> centre network.

**This is the single most important design decision in the experimental setup,
so it is worth being explicit about why the latency here is fake.**

Network delay is the *independent variable* of experiment E1 and the mechanism
that generates staleness for E2. An experiment needs to set it, not observe it.
A real network - or a broker such as Kafka sitting in front of one - produces
latency as an emergent property of batching, buffering and scheduling, which
makes it a confounder rather than a treatment: two runs at "the same" delay are
not actually comparable, and a sweep over delay is not reproducible.

Injecting the delay makes it a dial. `delay_ms=200` means exactly 200 ms of
one-way delay on every coordination hop, on every run, on every machine. The
comparison between `strong_lock` (one hop per order) and `escrow_quota` (one hop
per refill) then measures the thing it claims to measure - the number of hops -
instead of measuring how the OS scheduler happened to behave that afternoon.

The cost is external validity, and that belongs in the limitations section: this
models delay and partition, not packet loss, reordering, congestion collapse or
TCP head-of-line blocking.

## Partition

A partitioned node cannot reach the centre at all. Each policy degrades
differently under partition, and that difference *is* the E1 result:

    strong_lock    cannot reserve anything - it needs the centre per order.
                   Correct, and completely unavailable.
    escrow_quota   keeps selling against the quota it already holds, and refuses
                   only once that quota is exhausted. Correct AND available for
                   as long as the escrow lasts.
    eventual       keeps selling regardless, and oversells silently.

That is the classic availability/consistency trade-off made concrete on a retail
stock pool, and it is the strongest argument for escrow that the system can
produce.
"""

from __future__ import annotations

import random
import threading
import time
from dataclasses import dataclass, field
from typing import Callable, Dict, Iterable, List, Optional, Set


class Partitioned(RuntimeError):
    """Raised by `SimNetwork.delay` when a node is cut off from the centre."""


@dataclass
class NetworkConditions:
    """One point in E1's sweep space."""

    delay_ms: float = 0.0
    jitter_ms: float = 0.0
    #: Channels currently unable to reach the inventory service.
    partition: Set[str] = field(default_factory=set)

    @classmethod
    def from_config(cls, cfg: Dict) -> "NetworkConditions":
        return cls(
            delay_ms=float(cfg.get("delay_ms", 0.0)),
            jitter_ms=float(cfg.get("jitter_ms", 0.0)),
            partition=set(cfg.get("partition", ()) or ()),
        )

    def as_dict(self) -> Dict:
        return {
            "delay_ms": self.delay_ms,
            "jitter_ms": self.jitter_ms,
            "partition": sorted(self.partition),
        }


class SimNetwork:
    """
    A stand-in for the wire between an edge node and the inventory service.

    Lives inside the inventory service process and is invoked by
    `SyncPolicy._hop`, so every policy pays the same simulated cost per
    coordination round trip and the policies differ only in *how many* they need.

    Thread-safe: uvicorn serves requests from a thread pool, so the counters and
    the RNG are guarded.
    """

    def __init__(self, conditions: Optional[NetworkConditions] = None, *,
                 seed: int = 42,
                 sleep: Callable[[float], None] = time.sleep) -> None:
        self.conditions = conditions or NetworkConditions()
        self._rng = random.Random(seed)
        # Injectable so tests can assert on the delay *requested* without
        # actually paying it - a 1000 ms sweep cell would otherwise make the
        # unit tests unusably slow.
        self._sleep = sleep
        self._lock = threading.Lock()
        self.hops: int = 0
        self.blocked: int = 0
        self.seconds_delayed: float = 0.0
        self.per_node: Dict[str, int] = {}

    # ── Configuration ───────────────────────────────────────────────────────

    def configure(self, *, delay_ms: Optional[float] = None,
                  jitter_ms: Optional[float] = None,
                  partition: Optional[Iterable[str]] = None) -> NetworkConditions:
        """Re-point mid-run. E1 sweeps by calling this between cells."""
        with self._lock:
            if delay_ms is not None:
                self.conditions.delay_ms = float(delay_ms)
            if jitter_ms is not None:
                self.conditions.jitter_ms = float(jitter_ms)
            if partition is not None:
                self.conditions.partition = set(partition)
            return self.conditions

    def is_partitioned(self, node: str) -> bool:
        return node in self.conditions.partition

    # ── The hot path ────────────────────────────────────────────────────────

    def hop(self, node: str) -> bool:
        """
        Pay one simulated round trip to the centre.

        Returns True if the hop got through, False if `node` is partitioned.
        A boolean rather than an exception because every caller has a *correct*
        degraded behaviour available to it - refusing the order, or spending
        existing quota - and unwinding through an exception would make those
        paths harder to read than they need to be.
        """
        with self._lock:
            self.hops += 1
            self.per_node[node] = self.per_node.get(node, 0) + 1
            if node in self.conditions.partition:
                self.blocked += 1
                return False
            base = self.conditions.delay_ms
            jitter = self.conditions.jitter_ms
            wait_ms = base + (self._rng.uniform(-jitter, jitter) if jitter else 0.0)
            wait_s = max(0.0, wait_ms) / 1000.0
            self.seconds_delayed += wait_s

        # Slept OUTSIDE the lock: holding it would serialise every node through
        # the delay and turn a 200 ms one-way latency into a global bottleneck,
        # which is the opposite of what is being modelled.
        if wait_s > 0:
            self._sleep(wait_s)
        return True

    def delay(self, node: str) -> None:
        """Exception-raising form, for callers that cannot handle a False."""
        if not self.hop(node):
            raise Partitioned(f"{node} is partitioned from the inventory service")

    # ── Reporting ───────────────────────────────────────────────────────────

    def stats(self) -> Dict:
        with self._lock:
            return {
                "hops": self.hops,
                "hops_blocked": self.blocked,
                "seconds_delayed": self.seconds_delayed,
                "per_node_hops": dict(self.per_node),
                **self.conditions.as_dict(),
            }

    def reset_stats(self) -> None:
        with self._lock:
            self.hops = 0
            self.blocked = 0
            self.seconds_delayed = 0.0
            self.per_node.clear()
