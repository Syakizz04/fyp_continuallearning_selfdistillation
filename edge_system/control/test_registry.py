"""Control-plane tests. No models needed - this layer is metadata only."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from ..config import SYSTEM_CONFIG
from . import service as svc
from .registry import Registry


@pytest.fixture
def registry(tmp_path):
    reg = Registry(tmp_path / "registry.db")
    yield reg
    reg.close()


@pytest.fixture
def client(tmp_path):
    SYSTEM_CONFIG["paths"]["registry_db"] = str(tmp_path / "registry.db")
    with TestClient(svc.app) as c:
        yield c


def test_register_and_read_back(registry) -> None:
    registry.register({
        "node": "pos", "strategy": "sdft", "generation": 1, "kind": "forecasting",
        "sim_date": "2014-03-10", "tick": 42, "reason": "drift", "duration_s": 91.2,
        "ok": True, "skipped": False,
    })
    rows = registry.versions()
    assert len(rows) == 1
    assert rows[0]["node"] == "pos" and rows[0]["ok"] == 1
    assert rows[0]["duration_s"] == pytest.approx(91.2)


def test_summary_aggregates_per_node(registry) -> None:
    for i, node in enumerate(["pos", "pos", "web"]):
        registry.register({"node": node, "strategy": "sdft", "generation": i + 1,
                           "duration_s": 10.0, "ok": True, "skipped": False})
    summary = registry.summary()
    assert summary["total_retrains"] == 3
    by_node = {n["node"]: n for n in summary["nodes"]}
    assert by_node["pos"]["retrains"] == 2
    assert by_node["web"]["retrains"] == 1


def test_heartbeat_upserts(registry) -> None:
    registry.heartbeat("pos", {"model_generation": 0})
    registry.heartbeat("pos", {"model_generation": 3})
    nodes = registry.nodes()
    assert len(nodes) == 1
    assert nodes[0]["model_generation"] == 3


def test_service_endpoints(client) -> None:
    assert client.get("/health").json()["ok"] is True

    client.post("/models/register", json={
        "node": "web", "strategy": "replay", "generation": 1,
        "kind": "rl", "ok": True, "skipped": False, "duration_s": 5.0,
    })
    assert len(client.get("/models").json()) == 1
    assert client.get("/summary").json()["total_retrains"] == 1

    client.post("/heartbeat", json={"node": "web", "payload": {"ok": True}})
    assert client.get("/nodes").json()[0]["node"] == "web"


def test_failed_retrain_is_recorded_not_dropped(registry) -> None:
    """Failures have to be visible - a node that keeps serving after a failed
    retrain looks healthy otherwise."""
    registry.register({"node": "pos", "strategy": "sdft", "ok": False,
                       "skipped": False, "error": "CUDA out of memory"})
    row = registry.versions()[0]
    assert row["ok"] == 0
    assert "CUDA" in row["error"]
