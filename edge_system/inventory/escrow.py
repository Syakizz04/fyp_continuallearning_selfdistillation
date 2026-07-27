"""
Escrow / bounded-counter core for shared inventory.

This is the algorithm the project contributes, kept deliberately free of I/O so
it can be property-tested on its own.

Background
----------
The problem is the classic one: N nodes decrement a shared quantity that must
never go below zero, without serialising every decrement through a central lock.

O'Neil's Escrow Method (ACM TODS 11(4), 1986) solves it by tracking, per record,
not just the current value but the *range* it could take once every in-flight
transaction resolves:

    inf  - the lowest value the field can reach (all pending decrements commit)
    val  - the value if nothing further happens
    sup  - the highest value it can reach (all pending increments commit)

A decrement of `q` is admissible iff `inf - q >= 0`. Because admission is decided
against `inf`, no combination of in-flight commits can drive the value negative,
and no lock is held across the transaction.

The Demarcation Protocol (Barbara-Milla & Garcia-Molina, VLDB Journal 1994)
distributes this: rather than consult the centre per operation, each node is
granted a **quota** - a private slice of the available amount it may spend with
no coordination at all. The centre only participates when a node's quota runs
dry. The safety invariant is arithmetic:

    sum(node quotas) + central_free + reserved + committed == total

Every node spends only from its own quota, so total consumption can never exceed
`total` regardless of message ordering, delay, or partition. The same structure
appears in modern form as the Bounded Counter CRDT (Balegas et al., EuroSys 2015),
whose motivating example is exactly distributed retail stock.

What this buys, and the trade-off being measured
------------------------------------------------
Correctness is unconditional, but a node's *view* of stock is deliberately
stale - it knows its own quota, not the global total. That staleness is the cost
of avoiding coordination, and in this project it is not merely an engineering
cost: it feeds the pricing agent a degraded `inventory_level` and lets
unanticipated stockouts censor observed demand. Quantifying it is the point.
"""

from __future__ import annotations

import itertools
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, Iterable, List, Optional, Tuple


class ReservationState(str, Enum):
    PENDING = "pending"
    COMMITTED = "committed"
    RELEASED = "released"
    EXPIRED = "expired"


@dataclass
class Reservation:
    """A claim on stock that is not yet a sale.

    Reservations exist so the reserve -> pay -> fulfil sequence can span time
    without holding a lock. An abandoned one must not leak stock, which is what
    the TTL and `release()` (Saga-style compensation) are for.
    """
    id: str
    node: str
    sku: str
    qty: int
    created_at: float
    ttl_s: float
    state: ReservationState = ReservationState.PENDING

    def is_expired(self, now: Optional[float] = None) -> bool:
        now = time.monotonic() if now is None else now
        return (self.state is ReservationState.PENDING
                and now - self.created_at > self.ttl_s)


@dataclass
class SkuLedger:
    """Escrow state for one SKU: O'Neil's (inf, val, sup) plus node quotas."""
    sku: str
    total: int
    committed: int = 0
    reserved: int = 0
    quotas: Dict[str, int] = field(default_factory=dict)

    # ── O'Neil's three quantities ───────────────────────────────────────────
    @property
    def val(self) -> int:
        """Stock if nothing further resolves: total minus what is already sold."""
        return self.total - self.committed

    @property
    def inf(self) -> int:
        """Worst case: every pending reservation and every granted quota spends."""
        return self.total - self.committed - self.reserved - self.outstanding_quota

    @property
    def sup(self) -> int:
        """Best case: every pending reservation is released."""
        return self.total - self.committed

    @property
    def outstanding_quota(self) -> int:
        return sum(self.quotas.values())

    @property
    def central_free(self) -> int:
        """Amount the centre may still hand out. This is the admission bound."""
        return self.inf

    def invariant_holds(self) -> bool:
        """The arithmetic identity that makes oversell impossible.

        Checked in tests and asserted by the service on every mutation, because
        an invariant that is never verified is only a comment.
        """
        accounted = (self.committed + self.reserved
                     + self.outstanding_quota + self.central_free)
        return accounted == self.total and self.central_free >= 0


class OversellError(RuntimeError):
    """Raised when an operation would break the stock >= 0 invariant.

    Under `strong_lock` and `escrow_quota` this should be unreachable. Under
    `eventual` it is reachable by design - that is the finding, not a bug.
    """


class BoundedCounter:
    """
    In-memory escrow ledger over many SKUs and many nodes.

    Pure and synchronous: no network, no database, no clock beyond an injectable
    `now`. The service layer wraps this with persistence and HTTP; the policies
    layer decides *when* the centre is consulted. Keeping the algorithm here
    means the safety property can be property-tested directly.
    """

    def __init__(self, *, allow_oversell: bool = False) -> None:
        # `allow_oversell` exists only to implement the `eventual` policy arm,
        # which must be able to violate the invariant so E1 can measure it.
        self._skus: Dict[str, SkuLedger] = {}
        self._reservations: Dict[str, Reservation] = {}
        self.allow_oversell = allow_oversell
        self.oversell_units = 0
        self.oversell_events = 0
        self._ids = itertools.count(1)
        # Each node's *belief* about global stock: the last figure it saw from
        # the centre, and what it has sold since. See `stock_estimate`.
        self._last_seen: Dict[tuple, int] = {}
        self._spend_since: Dict[tuple, int] = {}

    # ── Setup ───────────────────────────────────────────────────────────────

    def stock_sku(self, sku: str, total: int) -> SkuLedger:
        """Create or top up a SKU's total stock (a replenishment delivery)."""
        if total < 0:
            raise ValueError(f"total must be >= 0, got {total}")
        led = self._skus.get(sku)
        if led is None:
            led = self._skus[sku] = SkuLedger(sku=sku, total=total)
        else:
            led.total += total
        return led

    def ledger(self, sku: str) -> SkuLedger:
        if sku not in self._skus:
            raise KeyError(f"unknown sku {sku!r}")
        return self._skus[sku]

    @property
    def skus(self) -> List[str]:
        return list(self._skus)

    # ── Quota management (the demarcation half) ──────────────────────────────

    def grant_quota(self, node: str, sku: str, qty: int) -> int:
        """Hand `qty` spending rights to `node`, or as many as are free.

        Returns the amount actually granted, which may be less than requested
        (or zero). Granting less is how backpressure reaches a node without an
        error path - it simply gets fewer rights and asks again later.
        """
        led = self.ledger(sku)
        grant = max(0, min(qty, led.central_free))
        if grant:
            led.quotas[node] = led.quotas.get(node, 0) + grant
        return grant

    def return_quota(self, node: str, sku: str, qty: Optional[int] = None) -> int:
        """Give unspent rights back to the centre (node shutdown, rebalancing)."""
        led = self.ledger(sku)
        held = led.quotas.get(node, 0)
        give = held if qty is None else max(0, min(qty, held))
        led.quotas[node] = held - give
        if led.quotas[node] == 0:
            led.quotas.pop(node, None)
        return give

    def quota_of(self, node: str, sku: str) -> int:
        return self._skus[sku].quotas.get(node, 0) if sku in self._skus else 0

    # ── The hot path ────────────────────────────────────────────────────────

    def reserve(
        self,
        node: str,
        sku: str,
        qty: int,
        *,
        ttl_s: float = 300.0,
        from_quota: bool = True,
        now: Optional[float] = None,
    ) -> Optional[Reservation]:
        """
        Claim `qty` units for `node`. Returns the Reservation, or None if refused.

        `from_quota=True` spends the node's private quota - no coordination, the
        escrow_quota fast path. `from_quota=False` admits directly against the
        centre's `inf`, which is the strong_lock path.

        Refusal is the correct, safe outcome when stock is genuinely gone. The
        rejection *rate* is the price paid for zero oversell, and is what E1
        measures against the alternatives.
        """
        if qty <= 0:
            raise ValueError(f"qty must be positive, got {qty}")
        led = self.ledger(sku)
        now = time.monotonic() if now is None else now

        if from_quota:
            held = led.quotas.get(node, 0)
            if held < qty:
                if not self.allow_oversell:
                    return None          # caller should refill quota and retry
                # `eventual`: node spends rights it does not hold. This is the
                # only route by which stock can go negative, and it is recorded.
                self._record_oversell(qty - held)
                led.quotas[node] = qty
                held = qty
            led.quotas[node] = held - qty
            if led.quotas[node] == 0:
                led.quotas.pop(node, None)
        else:
            if led.central_free < qty:
                if not self.allow_oversell:
                    return None
                self._record_oversell(qty - led.central_free)

        led.reserved += qty
        # The node knows what it sold itself, so its estimate of global stock
        # drops by this even without talking to anyone.
        key = (node, sku)
        self._spend_since[key] = self._spend_since.get(key, 0) + qty
        res = Reservation(
            id=f"r{next(self._ids):08d}-{uuid.uuid4().hex[:8]}",
            node=node, sku=sku, qty=qty, created_at=now, ttl_s=ttl_s,
        )
        self._reservations[res.id] = res
        self._check(led)
        return res

    def commit(self, reservation_id: str) -> Reservation:
        """Turn a reservation into a sale. Idempotent for already-committed ids."""
        res = self._reservation(reservation_id)
        if res.state is ReservationState.COMMITTED:
            return res
        if res.state is not ReservationState.PENDING:
            raise ValueError(f"cannot commit {reservation_id}: state={res.state.value}")
        led = self.ledger(res.sku)
        led.reserved -= res.qty
        led.committed += res.qty
        res.state = ReservationState.COMMITTED
        self._check(led)
        return res

    def release(self, reservation_id: str,
                *, expired: bool = False) -> Reservation:
        """
        Return a reservation's units to the node that holds it.

        This is the Saga compensation step: if payment or a downstream step
        fails, the claim on stock has to be undone or the units leak. Units go
        back to the node's quota rather than the centre, so releasing costs no
        coordination either.
        """
        res = self._reservation(reservation_id)
        if res.state in (ReservationState.RELEASED, ReservationState.EXPIRED):
            return res
        if res.state is ReservationState.COMMITTED:
            raise ValueError(f"cannot release {reservation_id}: already committed")
        led = self.ledger(res.sku)
        led.reserved -= res.qty
        led.quotas[res.node] = led.quotas.get(res.node, 0) + res.qty
        # The sale did not happen, so the node's own belief recovers too - it
        # saw the abandonment locally and needs no round trip to learn of it.
        key = (res.node, res.sku)
        self._spend_since[key] = max(0, self._spend_since.get(key, 0) - res.qty)
        res.state = ReservationState.EXPIRED if expired else ReservationState.RELEASED
        self._check(led)
        return res

    def sweep_expired(self, now: Optional[float] = None) -> List[Reservation]:
        """Release every reservation past its TTL. Call this once per tick."""
        now = time.monotonic() if now is None else now
        out = []
        for res in list(self._reservations.values()):
            if res.is_expired(now):
                out.append(self.release(res.id, expired=True))
        return out

    # ── Views ───────────────────────────────────────────────────────────────

    def true_available(self, sku: str) -> int:
        """Ground-truth sellable stock. The centre knows this; nodes do not."""
        led = self.ledger(sku)
        return led.total - led.committed - led.reserved

    def node_view(self, node: str, sku: str) -> int:
        """What `node` believes it can sell right now: its own quota, nothing more."""
        return self.quota_of(node, sku)

    def refresh_view(self, node: str, sku: str) -> int:
        """
        Record that `node` has just seen the centre's true figure.

        Called by the policies on every successful round trip. How often that
        happens is the whole difference between the arms - strong_lock refreshes
        on every order, escrow_quota only when it refills, eventual almost never
        - so the estimate below goes stale at a rate the policy determines.
        """
        seen = self.true_available(sku)
        self._last_seen[(node, sku)] = seen
        self._spend_since[(node, sku)] = 0
        return seen

    def stock_estimate(self, node: str, sku: str) -> int:
        """
        What `node` believes global stock to be: last seen, minus its own sales.

        **This, not the quota, is what a deployed node would feed its pricing
        agent, and it is the signal E2 manipulates.** The distinction matters
        more than it looks. `node_view` is a quota - how many units this node may
        sell without coordinating - and it is a *different variable* from stock
        on hand, on a different scale: a node holding 10 units of quota against
        495 units of real stock is not working from stale information, it is
        working from another quantity entirely. Feeding that to an agent trained
        on true inventory would demolish it rather than degrade it, and the
        experiment would be measuring a unit mismatch instead of staleness.

        This estimate is the *same* variable as ground truth, merely out of date:
        correct at the last refresh, then drifting as other nodes sell units this
        one cannot see. That is what staleness means, and it degrades smoothly
        with the refresh interval, which is what makes it a usable treatment.
        """
        key = (node, sku)
        if key not in self._last_seen:
            # Never spoken to the centre about this SKU. Falling back to ground
            # truth would hand the node information it has no way to hold.
            return 0
        return max(0, self._last_seen[key] - self._spend_since[key])

    def estimate_error(self, node: str, sku: str) -> int:
        """
        How wrong `node`'s belief about stock currently is, in units.

        Signed: positive means the node under-estimates (the usual case, since
        it cannot see other nodes returning stock or the centre replenishing),
        negative means it over-estimates - the dangerous direction, because it
        prices and promises against units that are gone.
        """
        return self.true_available(sku) - self.stock_estimate(node, sku)

    def staleness(self, node: str, sku: str) -> int:
        """
        Gap between ground truth and a node's local view, in units.

        The core measurement of this phase. Under escrow_quota it is inherently
        positive (a node underestimates, because it cannot see other nodes'
        quotas) - which is why the system is *safe* but the pricing agent is
        working from a degraded signal.
        """
        return self.true_available(sku) - self.node_view(node, sku)

    def snapshot(self) -> Dict[str, Dict]:
        return {
            sku: {
                "total": l.total, "committed": l.committed, "reserved": l.reserved,
                "inf": l.inf, "val": l.val, "sup": l.sup,
                "central_free": l.central_free, "quotas": dict(l.quotas),
                "true_available": self.true_available(sku),
            }
            for sku, l in self._skus.items()
        }

    def pending(self, node: Optional[str] = None) -> Iterable[Reservation]:
        return [r for r in self._reservations.values()
                if r.state is ReservationState.PENDING
                and (node is None or r.node == node)]

    # ── Internals ───────────────────────────────────────────────────────────

    def _reservation(self, rid: str) -> Reservation:
        if rid not in self._reservations:
            raise KeyError(f"unknown reservation {rid!r}")
        return self._reservations[rid]

    def _record_oversell(self, units: int) -> None:
        self.oversell_units += units
        self.oversell_events += 1

    def _check(self, led: SkuLedger) -> None:
        if self.allow_oversell:
            return   # the eventual arm is expected to break it; measure, don't raise
        if not led.invariant_holds():
            raise OversellError(
                f"escrow invariant violated for {led.sku}: total={led.total} "
                f"committed={led.committed} reserved={led.reserved} "
                f"quotas={led.quotas} central_free={led.central_free}"
            )
