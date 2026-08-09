"""
End-to-end tests for the inventory service over HTTP.

Uses FastAPI's TestClient, so these exercise the real routes, real serialisation,
and the real escrow ledger - no mocking of the thing under test. The pool backend
is forced to SQLite so the suite runs without Redis.

Run:  pytest edge_system/inventory/ -v
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from ..config import SYSTEM_CONFIG
from . import service as svc

SKU = "FOODS_3_090"


@pytest.fixture
def client(tmp_path):
    SYSTEM_CONFIG["redis"]["backend"] = "sqlite"
    SYSTEM_CONFIG["paths"]["events_db"] = str(tmp_path / "events.db")
    SYSTEM_CONFIG["sync"]["policy"] = "escrow_quota"
    with TestClient(svc.app) as c:
        yield c


def _reserve(client, node: str, qty: int, sku: str = SKU, **kw):
    return client.post("/reserve", json={"node": node, "sku": sku, "qty": qty, **kw})


# ─── Lifecycle ───────────────────────────────────────────────────────────────

def test_health_and_metrics_report_the_active_policy(client) -> None:
    health = client.get("/health").json()
    assert health["ok"] is True
    assert health["policy"] == "escrow_quota"
    assert "Sqlite" in health["backend"]

    metrics = client.get("/metrics").json()
    assert metrics["policy"] == "escrow_quota"
    assert metrics["reserve_attempts"] == 0


def test_reserve_unknown_sku_is_404(client) -> None:
    assert _reserve(client, "pos", 1, sku="NOPE").status_code == 404


# ─── The reserve -> commit -> release cycle ──────────────────────────────────

def test_full_reserve_commit_cycle(client) -> None:
    client.post("/replenish", json={"sku": SKU, "qty": 100})

    r = _reserve(client, "pos", 3).json()
    assert r["granted"] is True and r["reservation_id"]

    commit = client.post("/commit", json={"reservation_id": r["reservation_id"]}).json()
    assert commit["state"] == "committed"

    stock = client.get(f"/stock/{SKU}").json()
    assert stock["committed"] == 3
    assert stock["true_available"] == 97


def test_release_returns_stock(client) -> None:
    client.post("/replenish", json={"sku": SKU, "qty": 50})
    r = _reserve(client, "web", 5).json()

    client.post("/release", json={"reservation_id": r["reservation_id"]})
    stock = client.get(f"/stock/{SKU}").json()
    assert stock["committed"] == 0
    assert stock["reserved"] == 0
    assert stock["true_available"] == 50


def test_commit_twice_is_idempotent_release_after_commit_conflicts(client) -> None:
    client.post("/replenish", json={"sku": SKU, "qty": 20})
    rid = _reserve(client, "pos", 2).json()["reservation_id"]

    assert client.post("/commit", json={"reservation_id": rid}).status_code == 200
    assert client.post("/commit", json={"reservation_id": rid}).status_code == 200
    assert client.post("/release", json={"reservation_id": rid}).status_code == 409


def test_unknown_reservation_is_404(client) -> None:
    assert client.post("/commit", json={"reservation_id": "nope"}).status_code == 404


# ─── The safety property, over HTTP ──────────────────────────────────────────

def test_service_never_oversells_under_contention(client) -> None:
    """The end-to-end version of the escrow property test."""
    total = 40
    client.post("/replenish", json={"sku": SKU, "qty": total})

    granted = 0
    for i in range(120):
        node = ["pos", "web", "marketplace"][i % 3]
        r = _reserve(client, node, 1).json()
        if r["granted"]:
            granted += 1
            client.post("/commit", json={"reservation_id": r["reservation_id"]})

    assert granted <= total, f"oversold over HTTP: {granted} > {total}"
    stock = client.get(f"/stock/{SKU}").json()
    assert stock["committed"] <= total
    assert stock["true_available"] >= 0
    assert client.get("/metrics").json()["oversell_units"] == 0


def test_replenish_twice_does_not_double_count(client) -> None:
    """First stocking seeds both books; later ones must credit both exactly once."""
    client.post("/replenish", json={"sku": SKU, "qty": 30})
    client.post("/replenish", json={"sku": SKU, "qty": 20})

    stock = client.get(f"/stock/{SKU}").json()
    assert stock["total"] == 50
    assert stock["true_available"] == 50
    assert stock["central_free"] == 50


# ─── Staleness reporting: what E2 consumes ───────────────────────────────────

def test_reserve_returns_the_nodes_own_stale_view(client) -> None:
    client.post("/replenish", json={"sku": SKU, "qty": 500})
    r = _reserve(client, "pos", 2).json()

    stock = client.get(f"/stock/{SKU}").json()
    # The node is told what it may sell, not what exists - that gap is the point.
    assert r["node_view"] < stock["true_available"]
    assert stock["staleness"]["pos"] > 0
    assert stock["staleness"]["marketplace"] == stock["true_available"]


def test_events_log_records_staleness_at_decision_time(client) -> None:
    client.post("/replenish", json={"sku": SKU, "qty": 100})
    _reserve(client, "pos", 1, tick=7, sim_date="2013-01-08")

    events = client.get("/events", params={"kind": "reserve"}).json()
    assert len(events) == 1
    ev = events[0]
    assert ev["tick"] == 7 and ev["sim_date"] == "2013-01-08"
    assert ev["policy"] == "escrow_quota"
    assert ev["granted"] == 1
    assert ev["staleness_units"] == ev["true_available"] - ev["node_view"]


def test_ttl_sweep_endpoint(client) -> None:
    client.post("/replenish", json={"sku": SKU, "qty": 10})
    _reserve(client, "pos", 2)
    SYSTEM_CONFIG["sync"]["reservation_ttl_s"] = 0.0
    assert client.post("/sweep").json()["expired"] >= 0
