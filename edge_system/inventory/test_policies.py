"""
Policy-level tests, run against BOTH PoolStore backends.

This is E1 in miniature: drive the same contended workload through each policy
and assert the properties each one is supposed to have. If these fail, the E1
sweep is not worth spending GPU time on.

Redis tests skip automatically when no server is reachable, so the suite passes
on a clean checkout. Run `docker compose up -d redis` to exercise the real path.

Run:  pytest edge_system/inventory/ -v
"""

from __future__ import annotations

import random
from typing import List

import pytest

from .policies import EventualPolicy, make_policy
from .store import SqlitePoolStore, make_pool_store

NODES = ["pos", "web", "marketplace"]
SKU = "FOODS_3_090"


# ─── Backends ────────────────────────────────────────────────────────────────

def _redis_store():
    try:
        store = make_pool_store("redis", prefix="fyp:test:")
    except Exception as exc:
        pytest.skip(f"redis unavailable: {exc}")
    store.reset()
    return store


@pytest.fixture(params=["sqlite", "redis"])
def store(request):
    s = _redis_store() if request.param == "redis" else SqlitePoolStore(":memory:")
    yield s
    s.reset()
    s.close()


# ─── Shared workload ─────────────────────────────────────────────────────────

def _contended_run(policy, *, total: int, n_orders: int, seed: int) -> dict:
    """Every node chases the same SKU until the stock is gone."""
    rng = random.Random(seed)
    policy.stock_sku(SKU, total)
    granted: List[str] = []

    for _ in range(n_orders):
        node = rng.choice(NODES)
        qty = rng.randint(1, 3)
        res = policy.reserve(node, SKU, qty)
        if res is not None:
            granted.append(res.id)
            if rng.random() < 0.85:
                policy.commit(res.id)
            else:
                policy.release(res.id)     # abandoned cart

    return policy.metrics.summary()


# ─── The properties each policy must have ────────────────────────────────────

@pytest.mark.parametrize("policy_name", ["strong_lock", "escrow_quota"])
def test_safe_policies_never_oversell(store, policy_name: str) -> None:
    policy = make_policy(policy_name, store)
    summary = _contended_run(policy, total=200, n_orders=400, seed=7)

    assert summary["oversell_units"] == 0, f"{policy_name} oversold"
    assert policy.counter.ledger(SKU).committed <= 200
    assert policy.counter.ledger(SKU).invariant_holds()
    # The workload has to actually exhaust the stock, or "no oversell" is trivial.
    assert summary["reserve_refused"] > 0, "stock never ran out; test is vacuous"


def test_eventual_oversells_under_contention(store) -> None:
    """The control arm's defining failure. If this stops happening, E1 is broken."""
    policy = EventualPolicy(store)
    policy.stock_sku(SKU, 50)
    for node in NODES:
        policy.counter.grant_quota(node, SKU, 50)   # each node sees the whole pool

    for _ in range(60):
        for node in NODES:
            res = policy.reserve(node, SKU, 1)
            if res:
                policy.commit(res.id)

    assert policy.counter.ledger(SKU).committed > 50
    assert policy.metrics.summary()["oversell_units"] > 0


def test_escrow_uses_far_less_coordination_than_strong_lock(store) -> None:
    """The whole point: same safety, fewer round trips."""
    strong = make_policy("strong_lock", store)
    strong_summary = _contended_run(strong, total=500, n_orders=300, seed=11)

    store.reset()
    escrow = make_policy("escrow_quota", store, refill_multiple=4.0)
    escrow_summary = _contended_run(escrow, total=500, n_orders=300, seed=11)

    assert strong_summary["roundtrips_per_reserve"] == pytest.approx(1.0)
    assert escrow_summary["roundtrips_per_reserve"] < 0.5
    assert escrow_summary["oversell_units"] == 0


def test_escrow_view_is_stale_but_safe(store) -> None:
    """Safety is unconditional; the cost lands on the node's view of stock.

    This gap is the input E2 feeds to the pricing agent.
    """
    policy = make_policy("escrow_quota", store, refill_multiple=5.0)
    policy.stock_sku(SKU, 300)

    policy.reserve("pos", SKU, 2)
    assert policy.true_available(SKU) > policy.node_view("pos", SKU)
    assert policy.staleness("pos", SKU) > 0
    assert policy.counter.ledger(SKU).invariant_holds()


# ─── Store-level atomicity ───────────────────────────────────────────────────

def test_take_never_goes_negative_when_safe(store) -> None:
    store.init_sku(SKU, 10)
    assert store.take(SKU, 4) == 4
    assert store.take(SKU, 10) == 6      # partial: only what is left
    assert store.take(SKU, 1) == 0
    assert store.free(SKU) == 0


def test_take_allow_negative_is_opt_in(store) -> None:
    """Only the eventual arm may drive the pool below zero."""
    store.init_sku(SKU, 5)
    assert store.take(SKU, 8) == 5       # safe path clamps
    store.init_sku(SKU, 5)
    assert store.take(SKU, 8, allow_negative=True) == 8
    assert store.free(SKU) == -3


def test_give_back_restores_free_pool(store) -> None:
    store.init_sku(SKU, 10)
    store.take(SKU, 6)
    store.give_back(SKU, 6)
    assert store.free(SKU) == 10
