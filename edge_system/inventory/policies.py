"""
Sync policies - the independent variable in experiment E1.

Three ways for N nodes to draw down one shared pool, differing only in *when the
centre is consulted*:

    strong_lock    every reserve is a compare-and-decrement against the centre.
                   Correct, but pays a round trip per order.

    eventual       the node decrements its local view immediately and reconciles
                   later. One round trip amortised over many orders, but two
                   nodes can spend the same unit. Oversells by construction.

    escrow_quota   the node pre-acquires a quota and spends it locally with no
                   coordination, returning to the centre only to refill.
                   Correct AND mostly local - the proposed method.

The expected finding is that escrow_quota attains strong_lock's zero-oversell at
close to eventual's latency, with the cost showing up somewhere else entirely:
each node's *view* of stock is stale, which is what E2 then feeds back into the
pricing agent and the forecaster.

Every policy exposes the same interface, so `run_system.py --scenario` swaps
them without touching the edge nodes.
"""

from __future__ import annotations

import inspect
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from .escrow import BoundedCounter, Reservation
from .store import PoolStore


@dataclass
class PolicyMetrics:
    """Per-policy counters. Written out per tick as E1's raw record."""
    reserve_attempts: int = 0
    reserve_granted: int = 0
    reserve_refused: int = 0
    oversell_units: int = 0
    oversell_events: int = 0
    central_roundtrips: int = 0      # the coordination cost being traded away
    hops_blocked: int = 0            # round trips lost to a network partition
    quota_refills: int = 0
    latencies_ms: List[float] = field(default_factory=list)

    def record_latency(self, started: float) -> None:
        self.latencies_ms.append((time.perf_counter() - started) * 1000.0)

    def summary(self) -> Dict[str, float]:
        lat = sorted(self.latencies_ms)

        def pct(p: float) -> float:
            if not lat:
                return 0.0
            return lat[min(int(p * len(lat)), len(lat) - 1)]

        attempts = max(self.reserve_attempts, 1)
        return {
            "reserve_attempts": self.reserve_attempts,
            "reserve_granted": self.reserve_granted,
            "reserve_refused": self.reserve_refused,
            "rejection_rate": self.reserve_refused / attempts,
            "oversell_units": self.oversell_units,
            "oversell_events": self.oversell_events,
            "oversell_rate": self.oversell_units / attempts,
            "central_roundtrips": self.central_roundtrips,
            "roundtrips_per_reserve": self.central_roundtrips / attempts,
            "hops_blocked": self.hops_blocked,
            "quota_refills": self.quota_refills,
            "latency_p50_ms": pct(0.50),
            "latency_p99_ms": pct(0.99),
        }


class SyncPolicy(ABC):
    """Base class. Subclasses differ only in `reserve`."""

    name: str = "base"

    def __init__(self, store: PoolStore, *, network=None,
                 reservation_ttl_s: float = 300.0) -> None:
        self.store = store
        # `network` simulates node <-> centre delay. Injected rather than real:
        # latency is E1's independent variable and has to be a dial we set.
        self.network = network
        # Threaded through to every `reserve` rather than left to the escrow
        # core's own default. The two defaults used to coincide at 300 s, which
        # meant the configured value was dead and looked alive.
        self.reservation_ttl_s = float(reservation_ttl_s)
        self.metrics = PolicyMetrics()
        self.counter = BoundedCounter(allow_oversell=self.allows_oversell)

    @property
    def allows_oversell(self) -> bool:
        return False

    # ── Setup ───────────────────────────────────────────────────────────────

    def stock_sku(self, sku: str, total: int) -> None:
        self.store.init_sku(sku, total)
        self.counter.stock_sku(sku, total)

    # ── Hot path ────────────────────────────────────────────────────────────

    @abstractmethod
    def reserve(self, node: str, sku: str, qty: int) -> Optional[Reservation]: ...

    def commit(self, reservation_id: str) -> Reservation:
        res = self.counter.commit(reservation_id)
        self.store.record_commit(res.sku, res.qty)
        return res

    def release(self, reservation_id: str) -> Reservation:
        return self.counter.release(reservation_id)

    def sweep_expired(self, now: Optional[float] = None) -> List[Reservation]:
        return self.counter.sweep_expired(now)

    # ── Views ───────────────────────────────────────────────────────────────

    def node_view(self, node: str, sku: str) -> int:
        return self.counter.node_view(node, sku)

    def true_available(self, sku: str) -> int:
        return self.counter.true_available(sku)

    def staleness(self, node: str, sku: str) -> int:
        """Gap between truth and this node's SELLING RIGHTS. E1's cost metric."""
        return self.counter.staleness(node, sku)

    def stock_estimate(self, node: str, sku: str) -> int:
        """What this node BELIEVES global stock is. E2's treatment signal."""
        return self.counter.stock_estimate(node, sku)

    def estimate_error(self, node: str, sku: str) -> int:
        """How wrong that belief currently is, signed."""
        return self.counter.estimate_error(node, sku)

    # ── Internals ───────────────────────────────────────────────────────────

    def _hop(self, node: str) -> bool:
        """
        One simulated round trip to the centre. False if `node` is partitioned.

        Every caller has a correct degraded behaviour when this fails, and the
        three behaviours differ - which is exactly the availability result:
        strong_lock cannot sell at all, escrow_quota keeps selling until its
        escrow runs out, eventual keeps selling and oversells.
        """
        self.metrics.central_roundtrips += 1
        if self.network is None:
            return True
        reachable = self.network.hop(node)
        if not reachable:
            self.metrics.hops_blocked += 1
        return reachable

    def _sync_oversell(self) -> None:
        self.metrics.oversell_units = self.counter.oversell_units
        self.metrics.oversell_events = self.counter.oversell_events


class StrongLockPolicy(SyncPolicy):
    """
    Serialise every decrement through the centre.

    The control arm. Correctness comes from `store.take()`, which is a Redis Lua
    compare-and-decrement - an independent primitive, deliberately not our own
    escrow code, so E1 is not comparing an implementation against itself.
    """

    name = "strong_lock"

    def reserve(self, node: str, sku: str, qty: int) -> Optional[Reservation]:
        started = time.perf_counter()
        self.metrics.reserve_attempts += 1
        if not self._hop(node):               # every single order pays this
            # Partitioned: this policy has no local rights to fall back on, so
            # the only safe answer is no. Correct, and completely unavailable.
            self.metrics.reserve_refused += 1
            self.metrics.record_latency(started)
            return None

        taken = self.store.take(sku, qty)
        if taken < qty:
            if taken:
                self.store.give_back(sku, taken)   # all-or-nothing
            self.metrics.reserve_refused += 1
            self.metrics.record_latency(started)
            return None

        # Mirror the grant into the ledger so reservation bookkeeping and the
        # staleness view stay consistent across policies.
        self.counter.grant_quota(node, sku, qty)
        # This arm talks to the centre on every order, so its belief about stock
        # is never more than one order out of date. That accuracy is what it is
        # buying with the round trip, and E2's contrast depends on it.
        self.counter.refresh_view(node, sku)
        res = self.counter.reserve(node, sku, qty, ttl_s=self.reservation_ttl_s)
        self.metrics.reserve_granted += 1
        self.metrics.record_latency(started)
        return res


class EscrowQuotaPolicy(SyncPolicy):
    """
    The proposed method: spend a locally-held quota, refill only when it runs low.

    A reserve that fits inside the node's quota costs zero coordination. When the
    quota cannot cover the order, the node takes a larger block from the centre
    in one round trip (`refill_multiple` x the order), amortising the cost.
    """

    name = "escrow_quota"

    def __init__(self, store: PoolStore, *, network=None,
                 refill_multiple: float = 3.0, min_refill: int = 8,
                 low_watermark: float = 0.0,
                 reservation_ttl_s: float = 300.0) -> None:
        super().__init__(store, network=network,
                         reservation_ttl_s=reservation_ttl_s)
        self.refill_multiple = refill_multiple
        self.min_refill = min_refill
        # Fraction of the last grant below which the node tops up *before* it is
        # forced to. 0.0 = purely reactive, the demarcation protocol's basic
        # form.
        #
        # MEASURED: this is very nearly inert as the refill is currently written,
        # and it is worth knowing why before anyone tunes it. 300 orders of 2
        # units against a fixed block of 8 gives 75 refills at watermark 0.0 and
        # 76 at 0.9. The coordination rate is set by `block size / order size`,
        # not by when the top-up triggers: `_refill` takes a fixed block sized on
        # the *current order*, so moving the trigger earlier only shifts the same
        # refill within its cycle rather than changing how often one happens.
        #
        # Making the watermark a real lever means refilling *to a target level*
        # rather than by a fixed block. That is a change to the method under
        # test, so it is a deliberate Phase 5 decision, not a quiet fix.
        self.low_watermark = float(low_watermark)
        self._last_grant: Dict[tuple, int] = {}

    def reserve(self, node: str, sku: str, qty: int) -> Optional[Reservation]:
        started = time.perf_counter()
        self.metrics.reserve_attempts += 1

        held = self.counter.quota_of(node, sku)
        if held < qty or held < self._watermark(node, sku):
            self._refill(node, sku, qty)

        res = self.counter.reserve(node, sku, qty, ttl_s=self.reservation_ttl_s)
        if res is None:
            self.metrics.reserve_refused += 1
        else:
            self.metrics.reserve_granted += 1
        self.metrics.record_latency(started)
        return res

    def _watermark(self, node: str, sku: str) -> float:
        """The level below which a proactive top-up is triggered."""
        if self.low_watermark <= 0:
            return 0.0
        return self._last_grant.get((node, sku), 0) * self.low_watermark

    def _refill(self, node: str, sku: str, qty: int) -> None:
        """One round trip that buys rights for many future orders."""
        want = max(int(qty * self.refill_multiple), self.min_refill, qty)
        if not self._hop(node):
            # Partitioned: no refill, but the quota already in hand is still
            # valid to spend. The node stays available until it runs out, which
            # is the whole point of pre-acquiring rights.
            return
        granted = self.store.take(sku, want)
        if granted < qty:
            # Not enough for the order even after refilling. Keep whatever came
            # back as quota rather than returning it - the next, smaller order
            # may well fit, and churning it back costs another round trip.
            if granted:
                self.counter.grant_quota(node, sku, granted)
                self._last_grant[(node, sku)] = granted
                self.counter.refresh_view(node, sku)
            return
        self.counter.grant_quota(node, sku, granted)
        self._last_grant[(node, sku)] = granted
        # The refill is this arm's ONLY sight of the centre, so its belief about
        # stock is fresh here and then drifts for the whole block - it cannot see
        # other nodes selling in the meantime. That drift is E2's treatment, and
        # `refill_multiple` sets how long it lasts.
        self.counter.refresh_view(node, sku)
        self.metrics.quota_refills += 1


class EventualPolicy(SyncPolicy):
    """
    Optimistic local decrement, reconciled with the centre afterwards.

    The unsafe arm. A node spends against its own stale view without checking, so
    concurrent nodes can spend the same unit; reconciliation discovers the
    overdraft only after the fact, by which point the order is placed.

    Its oversells are the measurement, not a defect - they are what the other two
    policies are being credited with preventing.
    """

    name = "eventual"

    def __init__(self, store: PoolStore, *, network=None,
                 reconcile_every: int = 25) -> None:
        super().__init__(store, network=network)
        self.reconcile_every = reconcile_every
        self._since_reconcile = 0
        self._push_mark: Dict[str, int] = {}   # per-instance: sku -> units pushed

    @property
    def allows_oversell(self) -> bool:
        return True

    def reserve(self, node: str, sku: str, qty: int) -> Optional[Reservation]:
        started = time.perf_counter()
        self.metrics.reserve_attempts += 1

        # No check against the centre: spend now, find out later.
        res = self.counter.reserve(node, sku, qty, ttl_s=self.reservation_ttl_s)
        self.metrics.reserve_granted += 1
        self._since_reconcile += 1
        if self._since_reconcile >= self.reconcile_every:
            self._reconcile(node, sku)
        self._sync_oversell()
        self.metrics.record_latency(started)
        return res

    def _reconcile(self, node: str, sku: str) -> None:
        """Push accumulated local spend to the centre. Too late to prevent anything."""
        self._since_reconcile = 0
        if not self._hop(node):
            # Partitioned: the push is lost and the mark is NOT advanced, so the
            # spend is retried on the next reconcile. Meanwhile the node has gone
            # on selling - unavailability it avoids by being wrong.
            return
        led = self.counter.ledger(sku)
        spent = led.committed + led.reserved
        delta = max(0, spent - self._push_mark.get(sku, 0))
        if delta:
            # allow_negative: the centre must be able to record that it went
            # below zero, because that IS the oversell being measured.
            self.store.take(sku, delta, allow_negative=True)
        self._push_mark[sku] = spent
        # Reconciliation is bidirectional: the node pushes its spend and learns
        # the true figure. It happens once every `reconcile_every` orders, so
        # this arm carries the stalest belief of the three.
        self.counter.refresh_view(node, sku)


POLICIES = {
    StrongLockPolicy.name: StrongLockPolicy,
    EscrowQuotaPolicy.name: EscrowQuotaPolicy,
    EventualPolicy.name: EventualPolicy,
}


def make_policy(name: str, store: PoolStore, **kwargs) -> SyncPolicy:
    """
    Build a policy, dropping options the chosen policy does not take.

    Callers pass the whole sync config rather than picking out the subset each
    policy accepts, because that picking-out is precisely where a setting goes
    missing: the caller forgets one, the policy silently uses its own default,
    and the run reports a configuration it never used.
    """
    if name not in POLICIES:
        raise ValueError(f"unknown policy {name!r}; have {sorted(POLICIES)}")

    cls = POLICIES[name]
    accepted = inspect.signature(cls.__init__).parameters
    return cls(store, **{k: v for k, v in kwargs.items() if k in accepted})
