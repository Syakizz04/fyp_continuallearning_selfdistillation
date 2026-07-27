"""
Property tests for the escrow core.

The headline property is the one the whole design exists to guarantee:

    under ANY interleaving of reserve / commit / release / quota-refill across
    ANY number of nodes, committed stock never exceeds the stock that was put in.

`test_eventual_policy_can_oversell` asserts the *opposite* for the unsafe arm.
That is deliberate: `eventual` is the control condition in E1, and a run in
which it never oversells would mean the contention scenario is too weak to be
measuring anything.

Run:  pytest edge_system/inventory/ -v
"""

from __future__ import annotations

import random

import pytest

from .escrow import BoundedCounter, OversellError, ReservationState

NODES = ["pos", "web", "marketplace"]


def _drive(counter: BoundedCounter, *, sku: str, n_ops: int, seed: int,
           max_qty: int = 5, refill: int = 8) -> dict:
    """Hammer one SKU with randomised operations from several nodes."""
    rng = random.Random(seed)
    live: list = []
    stats = {"reserved": 0, "committed": 0, "released": 0, "refused": 0}

    for _ in range(n_ops):
        op = rng.random()
        node = rng.choice(NODES)

        if op < 0.15:
            counter.grant_quota(node, sku, refill)

        elif op < 0.70:
            qty = rng.randint(1, max_qty)
            if counter.quota_of(node, sku) < qty:
                counter.grant_quota(node, sku, max(refill, qty))
            res = counter.reserve(node, sku, qty)
            if res is None:
                stats["refused"] += 1
            else:
                live.append(res.id)
                stats["reserved"] += 1

        elif op < 0.90 and live:
            rid = live.pop(rng.randrange(len(live)))
            counter.commit(rid)
            stats["committed"] += 1

        elif live:
            rid = live.pop(rng.randrange(len(live)))
            counter.release(rid)
            stats["released"] += 1

    return stats


# ─── The safety property ─────────────────────────────────────────────────────

@pytest.mark.parametrize("seed", range(25))
def test_never_oversells(seed: int) -> None:
    """Committed units never exceed total stock, for any interleaving."""
    total = 120
    c = BoundedCounter()
    c.stock_sku("SKU", total)

    _drive(c, sku="SKU", n_ops=600, seed=seed)

    led = c.ledger("SKU")
    assert led.committed <= total, f"oversold: {led.committed} > {total}"
    assert c.oversell_units == 0
    assert led.invariant_holds()
    assert led.central_free >= 0
    assert c.true_available("SKU") >= 0


@pytest.mark.parametrize("seed", range(25))
def test_invariant_holds_after_every_op(seed: int) -> None:
    """The accounting identity survives every individual mutation.

    `_check` already raises on violation, so reaching the end without an
    OversellError is the assertion; this makes the intent explicit.
    """
    c = BoundedCounter()
    c.stock_sku("SKU", 80)
    try:
        _drive(c, sku="SKU", n_ops=400, seed=seed)
    except OversellError as exc:  # pragma: no cover - this is the failure we test for
        pytest.fail(f"invariant broke mid-run: {exc}")
    assert c.ledger("SKU").invariant_holds()


def test_exhaustion_refuses_rather_than_oversells() -> None:
    """When stock genuinely runs out, reservations are refused, not fulfilled."""
    c = BoundedCounter()
    c.stock_sku("SKU", 10)
    c.grant_quota("pos", "SKU", 10)

    for _ in range(10):
        assert c.reserve("pos", "SKU", 1) is not None
    assert c.reserve("pos", "SKU", 1) is None          # quota exhausted
    assert c.grant_quota("pos", "SKU", 5) == 0         # centre has nothing either
    assert c.reserve("pos", "SKU", 1) is None
    assert c.oversell_units == 0


def test_quota_isolation_between_nodes() -> None:
    """One node cannot spend another's rights - the demarcation guarantee."""
    c = BoundedCounter()
    c.stock_sku("SKU", 10)
    assert c.grant_quota("pos", "SKU", 10) == 10
    assert c.grant_quota("web", "SKU", 5) == 0, "centre gave away rights it lacked"
    assert c.reserve("web", "SKU", 1) is None
    assert c.reserve("pos", "SKU", 1) is not None


# ─── Reservation lifecycle ───────────────────────────────────────────────────

def test_release_returns_units_to_the_node() -> None:
    """Saga compensation: an abandoned reservation must not leak stock."""
    c = BoundedCounter()
    c.stock_sku("SKU", 10)
    c.grant_quota("pos", "SKU", 10)

    res = c.reserve("pos", "SKU", 4)
    assert c.quota_of("pos", "SKU") == 6
    c.release(res.id)
    assert c.quota_of("pos", "SKU") == 10
    assert c.ledger("SKU").reserved == 0
    assert c.ledger("SKU").committed == 0


def test_ttl_sweep_reclaims_abandoned_reservations() -> None:
    c = BoundedCounter()
    c.stock_sku("SKU", 10)
    c.grant_quota("pos", "SKU", 10)

    res = c.reserve("pos", "SKU", 3, ttl_s=60.0, now=1_000.0)
    assert c.sweep_expired(now=1_030.0) == []          # not yet expired
    swept = c.sweep_expired(now=1_100.0)
    assert [r.id for r in swept] == [res.id]
    assert swept[0].state is ReservationState.EXPIRED
    assert c.quota_of("pos", "SKU") == 10


def test_commit_is_idempotent_and_release_after_commit_raises() -> None:
    c = BoundedCounter()
    c.stock_sku("SKU", 5)
    c.grant_quota("pos", "SKU", 5)
    res = c.reserve("pos", "SKU", 2)

    assert c.commit(res.id).state is ReservationState.COMMITTED
    assert c.commit(res.id).state is ReservationState.COMMITTED   # idempotent
    assert c.ledger("SKU").committed == 2
    with pytest.raises(ValueError, match="already committed"):
        c.release(res.id)


# ─── Staleness: the measurement this phase exists to produce ─────────────────

def test_node_view_understates_true_stock() -> None:
    """A node sees only its own quota, so its view is conservative by design.

    This gap is what degrades the pricing agent's `inventory_level` input, and
    is the mechanism E2 tests.
    """
    c = BoundedCounter()
    c.stock_sku("SKU", 100)
    c.grant_quota("pos", "SKU", 20)
    c.grant_quota("web", "SKU", 30)

    assert c.true_available("SKU") == 100
    assert c.node_view("pos", "SKU") == 20
    assert c.staleness("pos", "SKU") == 80
    # Larger quotas mean fewer round trips but no less staleness - the trade-off
    # is round trips vs how wrong every node's view is, not one or the other.
    assert c.staleness("web", "SKU") == 70


# ─── The unsafe control arm ──────────────────────────────────────────────────

def test_eventual_policy_can_oversell() -> None:
    """`eventual` must be ABLE to oversell, or E1's comparison is vacuous."""
    c = BoundedCounter(allow_oversell=True)
    c.stock_sku("SKU", 10)

    for node in NODES:                    # each node thinks it has the full pool
        c.grant_quota(node, "SKU", 10)
    for node in NODES:
        for _ in range(10):
            res = c.reserve(node, "SKU", 1)
            if res:
                c.commit(res.id)

    assert c.ledger("SKU").committed > 10, "no oversell: contention too weak"
    assert c.oversell_units > 0


def test_safe_counter_raises_rather_than_silently_oversells() -> None:
    """If a bug ever did break the invariant, it must be loud."""
    c = BoundedCounter()
    c.stock_sku("SKU", 5)
    c.grant_quota("pos", "SKU", 5)
    led = c.ledger("SKU")
    led.total = 2                          # corrupt state behind the algorithm's back
    with pytest.raises(OversellError):
        c.reserve("pos", "SKU", 1)
