"""
Tests for a node's *belief* about global stock - E2's treatment signal.

The distinction under test is between two things a node knows, which are easy to
confuse and are not interchangeable:

    node_view       its QUOTA: how many units it may sell without coordinating.
                    A different variable from stock on hand, on a different
                    scale. E1's cost metric.

    stock_estimate  what it BELIEVES total stock to be: the last figure it saw
                    from the centre, less what it has sold since. The same
                    variable as ground truth, merely out of date. E2's signal.

E2 substitutes a degraded inventory reading for the true one and asks how the CL
methods cope. That only measures staleness if the substituted value is the same
*quantity* as the original. Feeding a quota of 10 to an agent trained on stock
levels of 495 measures a unit mismatch instead.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from edge_system.inventory.escrow import BoundedCounter
from edge_system.inventory.policies import (EscrowQuotaPolicy, EventualPolicy,
                                            StrongLockPolicy)
from edge_system.inventory.store import SqlitePoolStore


@pytest.fixture
def store():
    s = SqlitePoolStore(str(Path(tempfile.mkdtemp()) / "pool.db"))
    yield s
    s.close()


# ─── The core distinction ────────────────────────────────────────────────────

def test_estimate_is_on_the_same_scale_as_truth_but_quota_is_not():
    """
    The property the whole design rests on.

    A node holding a small quota against a large pool has a *correct* belief
    about stock and a small selling allowance. Confusing the two is what would
    have made E2 measure a scale change instead of staleness.
    """
    c = BoundedCounter()
    c.stock_sku("A", 500)
    c.grant_quota("pos", "A", 10)
    c.refresh_view("pos", "A")

    assert c.node_view("pos", "A") == 10        # rights: small
    assert c.stock_estimate("pos", "A") == 500  # belief: correct, and large
    assert c.estimate_error("pos", "A") == 0
    # The old signal would have handed the agent 10 where truth was 500.
    assert c.staleness("pos", "A") == 490


def test_a_node_that_has_never_reached_the_centre_believes_nothing():
    """Falling back to truth would hand it information it cannot hold."""
    c = BoundedCounter()
    c.stock_sku("A", 500)
    assert c.stock_estimate("pos", "A") == 0


def test_belief_drops_by_the_nodes_own_sales_without_any_round_trip():
    c = BoundedCounter()
    c.stock_sku("A", 500)
    c.grant_quota("pos", "A", 50)
    c.refresh_view("pos", "A")

    c.reserve("pos", "A", 12)
    assert c.stock_estimate("pos", "A") == 488
    assert c.estimate_error("pos", "A") == 0     # it sold them, so it knows


def test_an_abandoned_reservation_restores_the_belief():
    c = BoundedCounter()
    c.stock_sku("A", 100)
    c.grant_quota("pos", "A", 20)
    c.refresh_view("pos", "A")

    res = c.reserve("pos", "A", 5)
    assert c.stock_estimate("pos", "A") == 95
    c.release(res.id)
    # No round trip needed: the node watched the customer walk away.
    assert c.stock_estimate("pos", "A") == 100
    assert c.estimate_error("pos", "A") == 0


def test_belief_goes_stale_only_because_of_OTHER_nodes():
    """
    The mechanism E2 exploits, in isolation.

    A node's own sales it knows about. What it cannot see is anyone else's - so
    the error grows in the dangerous direction: it believes there is more stock
    than there is.
    """
    c = BoundedCounter()
    c.stock_sku("A", 100)
    for node in ("pos", "web"):
        c.grant_quota(node, "A", 40)
        c.refresh_view(node, "A")

    c.reserve("web", "A", 30)          # pos cannot see this

    assert c.stock_estimate("pos", "A") == 100   # unchanged belief
    assert c.true_available("A") == 70
    assert c.estimate_error("pos", "A") == -30   # negative: over-estimates
    # web sold them, so web's own belief is still correct.
    assert c.estimate_error("web", "A") == 0


# ─── How the policies differ ─────────────────────────────────────────────────

def test_strong_lock_belief_is_never_stale(store):
    """It buys freshness with a round trip per order. That is the trade."""
    policy = StrongLockPolicy(store)
    policy.stock_sku("A", 200)
    for _ in range(10):
        policy.reserve("pos", "A", 2)
        policy.reserve("web", "A", 2)
    # Each node refreshed on its own most recent order, so at worst it is behind
    # by what the other sold since - and here web went last.
    assert policy.estimate_error("web", "A") == 0


def test_escrow_belief_is_fresh_at_refill_and_drifts_between(store):
    policy = EscrowQuotaPolicy(store, refill_multiple=10.0, min_refill=40)
    policy.stock_sku("A", 500)

    policy.reserve("pos", "A", 1)          # triggers pos's first refill
    assert policy.estimate_error("pos", "A") == 0

    for _ in range(20):                    # web spends, pos is not told
        policy.reserve("web", "A", 1)
    assert policy.estimate_error("pos", "A") < 0

    # Draining pos's quota forces a refill, which resets its belief.
    for _ in range(60):
        policy.reserve("pos", "A", 1)
    assert policy.estimate_error("pos", "A") == 0


def test_eventual_carries_the_stalest_belief(store):
    policy = EventualPolicy(store, reconcile_every=1000)
    policy.stock_sku("A", 500)
    policy.counter.refresh_view("pos", "A")

    for _ in range(50):
        policy.reserve("web", "A", 2)
    # 50 orders in and it has still not spoken to anyone.
    assert policy.estimate_error("pos", "A") == -100


# ─── The E2 dial ─────────────────────────────────────────────────────────────

def test_refill_multiple_graduates_the_staleness():
    """
    `quota_refill_multiple` must move staleness smoothly, because E2 sweeps it.

    A bigger block means longer between refreshes, so a node's belief is out of
    date for longer and drifts further. This is the graduated treatment the
    experiment needs - not an on/off switch.
    """
    errors = {}
    for multiple in (2.0, 10.0, 40.0):
        store = SqlitePoolStore(str(Path(tempfile.mkdtemp()) / "p.db"))
        try:
            policy = EscrowQuotaPolicy(store, refill_multiple=multiple,
                                       min_refill=1)
            policy.stock_sku("A", 5000)
            total = 0
            for i in range(200):
                policy.reserve("pos", "A", 1)
                policy.reserve("web", "A", 1)
                total += abs(policy.estimate_error("pos", "A"))
            errors[multiple] = total / 200
        finally:
            store.close()

    assert errors[2.0] < errors[10.0] < errors[40.0], errors
