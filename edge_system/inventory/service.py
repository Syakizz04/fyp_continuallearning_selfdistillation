"""
Inventory sync service (FastAPI).

The central plane for shared stock. Edge nodes call it to reserve, commit, and
release; it owns the escrow ledger, the durable pool, and the event log.

    POST /reserve        claim units (the hot path)
    POST /commit         turn a reservation into a sale
    POST /release        Saga compensation - undo a claim
    POST /replenish      a delivery arrives
    GET  /stock/{sku}    ground truth + per-node views + staleness
    GET  /metrics        E1's counters
    GET  /events         recent activity, for the dashboard
    POST /admin/reset    wipe state between experiment cells

Run:  uvicorn edge_system.inventory.service:app --port 8001
"""

from __future__ import annotations

import time
from contextlib import asynccontextmanager
from typing import Dict, List, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from ..config import APPLIED_OVERRIDES, SYSTEM_CONFIG, channel_names, ensure_dirs
from ..sim.network import NetworkConditions, SimNetwork
from .events import EventLog
from .policies import SyncPolicy, make_policy
from .store import make_pool_store


# ─── Wire format ─────────────────────────────────────────────────────────────

class ReserveRequest(BaseModel):
    node: str
    sku: str
    qty: int = Field(gt=0)
    tick: Optional[int] = None
    sim_date: Optional[str] = None


class ReserveResponse(BaseModel):
    granted: bool
    reservation_id: Optional[str] = None
    qty: int
    # Returned on every reserve so the caller can feed its *own* stale view to
    # the pricing agent. This is the degraded signal E2 studies - the node is
    # told what it may sell, not what exists.
    node_view: int
    latency_ms: float
    reason: Optional[str] = None


class ReservationRef(BaseModel):
    reservation_id: str


class ReplenishRequest(BaseModel):
    sku: str
    qty: int = Field(gt=0)
    tick: Optional[int] = None
    sim_date: Optional[str] = None


class StockResponse(BaseModel):
    sku: str
    true_available: int
    total: int
    committed: int
    reserved: int
    central_free: int
    #: Selling rights per node - what it may sell without coordinating (E1).
    node_views: Dict[str, int]
    #: Gap between truth and those rights (E1's cost metric).
    staleness: Dict[str, int]
    #: What each node BELIEVES global stock to be - the signal a deployed node
    #: would price against, and the one E2 manipulates. Same variable as
    #: `true_available`, merely out of date; unlike `node_views`, which is a
    #: different quantity on a different scale.
    stock_estimates: Dict[str, int]
    #: Signed error in that belief. Negative = the node thinks there is more
    #: stock than there is, which is the direction that causes mispricing.
    estimate_errors: Dict[str, int]


# ─── Service state ───────────────────────────────────────────────────────────

class InventoryService:
    """Holds the policy, the store, and the log. One instance per process."""

    def __init__(self) -> None:
        self.policy: Optional[SyncPolicy] = None
        self.events: Optional[EventLog] = None
        self.network: Optional[SimNetwork] = None
        self.backend: str = "unset"
        self.started_at: float = 0.0

    def start(self, *, policy_name: Optional[str] = None,
              backend: Optional[str] = None) -> None:
        ensure_dirs()
        cfg = SYSTEM_CONFIG
        policy_name = policy_name or cfg["sync"]["policy"]
        backend = backend or cfg["redis"].get("backend", "auto")

        store = make_pool_store(backend, prefix=cfg["redis"]["key_prefix"])
        self.backend = type(store).__name__
        self.events = EventLog(cfg["paths"]["events_db"])
        # The simulated wire lives here, on the centre's side, so every policy
        # pays the same cost per hop and they differ only in how many hops they
        # need. Carried across a policy swap (see `reset`) so a sweep can change
        # one variable at a time.
        self.network = self.network or SimNetwork(
            NetworkConditions.from_config(cfg["network"]), seed=cfg["sim"]["seed"])
        # The whole sync config goes in; `make_policy` drops what this policy
        # does not accept. Selecting the subset here is how `reservation_ttl_s`
        # came to be configured-but-dead.
        self.policy = make_policy(
            policy_name, store, network=self.network,
            refill_multiple=cfg["sync"]["quota_refill_multiple"],
            low_watermark=cfg["sync"]["quota_low_watermark"],
            reservation_ttl_s=cfg["sync"]["reservation_ttl_s"],
        )
        self.started_at = time.time()

    def require(self) -> SyncPolicy:
        if self.policy is None:
            raise HTTPException(503, "inventory service not started")
        return self.policy

    def reset(self, *, policy_name: Optional[str] = None) -> None:
        if self.policy is not None:
            self.policy.store.reset()
            self.policy.store.close()
        if self.events is not None:
            self.events.close()
        self.start(policy_name=policy_name)


STATE = InventoryService()


@asynccontextmanager
async def lifespan(app: FastAPI):
    STATE.start()
    yield
    if STATE.events is not None:
        STATE.events.close()
    if STATE.policy is not None:
        STATE.policy.store.close()


app = FastAPI(title="FYP inventory sync", version="1.0", lifespan=lifespan)


# ─── Hot path ────────────────────────────────────────────────────────────────

@app.post("/reserve", response_model=ReserveResponse)
def reserve(req: ReserveRequest) -> ReserveResponse:
    policy = STATE.require()
    started = time.perf_counter()

    try:
        # Ground truth is captured BEFORE the reserve mutates it, so the logged
        # staleness is what the node was actually deciding against.
        true_avail = policy.true_available(req.sku)
    except KeyError:
        raise HTTPException(404, f"unknown sku {req.sku!r}")

    node_view_before = policy.node_view(req.node, req.sku)
    res = policy.reserve(req.node, req.sku, req.qty)
    latency_ms = (time.perf_counter() - started) * 1000.0

    STATE.events.record(
        "reserve", tick=req.tick, sim_date=req.sim_date, node=req.node,
        sku=req.sku, qty=req.qty, reservation_id=res.id if res else None,
        policy=policy.name, granted=1 if res else 0, latency_ms=latency_ms,
        true_available=true_avail, node_view=node_view_before,
        staleness_units=true_avail - node_view_before,
    )

    return ReserveResponse(
        granted=res is not None,
        reservation_id=res.id if res else None,
        qty=req.qty,
        node_view=policy.node_view(req.node, req.sku),
        latency_ms=latency_ms,
        reason=None if res else "insufficient stock",
    )


@app.post("/commit")
def commit(ref: ReservationRef) -> Dict:
    policy = STATE.require()
    try:
        res = policy.commit(ref.reservation_id)
    except KeyError:
        raise HTTPException(404, f"unknown reservation {ref.reservation_id!r}")
    except ValueError as exc:
        raise HTTPException(409, str(exc))

    STATE.events.record("commit", node=res.node, sku=res.sku, qty=res.qty,
                        reservation_id=res.id, policy=policy.name)
    return {"reservation_id": res.id, "state": res.state.value, "qty": res.qty}


@app.post("/release")
def release(ref: ReservationRef) -> Dict:
    policy = STATE.require()
    try:
        res = policy.release(ref.reservation_id)
    except KeyError:
        raise HTTPException(404, f"unknown reservation {ref.reservation_id!r}")
    except ValueError as exc:
        raise HTTPException(409, str(exc))

    STATE.events.record("release", node=res.node, sku=res.sku, qty=res.qty,
                        reservation_id=res.id, policy=policy.name)
    return {"reservation_id": res.id, "state": res.state.value, "qty": res.qty}


@app.post("/sweep")
def sweep() -> Dict:
    """Release everything past its TTL. The simulation calls this once per tick."""
    policy = STATE.require()
    swept = policy.sweep_expired()
    for res in swept:
        STATE.events.record("expire", node=res.node, sku=res.sku, qty=res.qty,
                            reservation_id=res.id, policy=policy.name)
    return {"expired": len(swept)}


@app.post("/replenish")
def replenish(req: ReplenishRequest) -> Dict:
    """A delivery arrives: raise total stock and hand the units to the centre."""
    policy = STATE.require()
    if req.sku in policy.counter.skus:
        # Existing SKU: add to the ledger total AND credit the centre's free
        # pool, which are two separate books that must move together.
        policy.counter.stock_sku(req.sku, req.qty)
        policy.store.give_back(req.sku, req.qty)
    else:
        # First stocking: `stock_sku` seeds both books, so crediting the free
        # pool again here would double-count the delivery.
        policy.stock_sku(req.sku, req.qty)

    STATE.events.record("replenish", sku=req.sku, qty=req.qty, tick=req.tick,
                        sim_date=req.sim_date, policy=policy.name)
    return {"sku": req.sku, "total": policy.counter.ledger(req.sku).total}


# ─── Views ───────────────────────────────────────────────────────────────────

@app.get("/stock/{sku}", response_model=StockResponse)
def stock(sku: str) -> StockResponse:
    policy = STATE.require()
    try:
        led = policy.counter.ledger(sku)
    except KeyError:
        raise HTTPException(404, f"unknown sku {sku!r}")
    nodes = channel_names()
    return StockResponse(
        sku=sku,
        true_available=policy.true_available(sku),
        total=led.total, committed=led.committed, reserved=led.reserved,
        central_free=led.central_free,
        node_views={n: policy.node_view(n, sku) for n in nodes},
        staleness={n: policy.staleness(n, sku) for n in nodes},
        stock_estimates={n: policy.stock_estimate(n, sku) for n in nodes},
        estimate_errors={n: policy.estimate_error(n, sku) for n in nodes},
    )


@app.get("/metrics")
def metrics() -> Dict:
    policy = STATE.require()
    return {
        "policy": policy.name,
        "backend": STATE.backend,
        "uptime_s": time.time() - STATE.started_at,
        "skus": len(policy.counter.skus),
        **policy.metrics.summary(),
        "network": STATE.network.stats() if STATE.network else {},
    }


@app.get("/events")
def events(limit: int = 100, kind: Optional[str] = None) -> List[Dict]:
    if STATE.events is None:
        raise HTTPException(503, "not started")
    return STATE.events.recent(limit=limit, kind=kind)


@app.get("/health")
def health() -> Dict:
    """
    Report the **effective** configuration, read back off the live objects.

    Not what SYSTEM_CONFIG says - what the running policy actually holds. A
    setting that failed to cross the process boundary shows up here as a
    mismatch, which is what `run_system.verify_inventory_config` checks before
    letting a run proceed.
    """
    policy = STATE.policy
    effective = {}
    if policy is not None:
        effective = {
            "reservation_ttl_s": policy.reservation_ttl_s,
            "refill_multiple": getattr(policy, "refill_multiple", None),
            "low_watermark": getattr(policy, "low_watermark", None),
            "allows_oversell": policy.allows_oversell,
        }
    return {
        "ok": policy is not None,
        "policy": policy.name if policy else None,
        "backend": STATE.backend,
        "effective": effective,
        # Which settings arrived from the launch environment, from the manifest.
        "applied_overrides": APPLIED_OVERRIDES,
    }


@app.post("/admin/reset")
def admin_reset(policy: Optional[str] = None) -> Dict:
    """Wipe state between experiment cells."""
    STATE.reset(policy_name=policy)
    return {"ok": True, "policy": STATE.policy.name, "backend": STATE.backend}


class NetworkRequest(BaseModel):
    delay_ms: Optional[float] = None
    jitter_ms: Optional[float] = None
    partition: Optional[List[str]] = None


@app.post("/admin/network")
def admin_network(req: NetworkRequest) -> Dict:
    """
    Set the simulated node <-> centre conditions.

    This is E1's independent variable, exposed as an endpoint so a sweep can move
    one cell to the next on a live system - no restart, no model reload, and no
    chance of some other difference creeping in between cells.
    """
    if STATE.network is None:
        raise HTTPException(503, "inventory service not started")
    conditions = STATE.network.configure(
        delay_ms=req.delay_ms, jitter_ms=req.jitter_ms, partition=req.partition)
    return {"ok": True, **conditions.as_dict()}


@app.post("/admin/flush")
def admin_flush() -> Dict:
    """Force the batched event log to disk before a reader opens the DB."""
    if STATE.events is None:
        raise HTTPException(503, "not started")
    STATE.events.flush()
    return {"ok": True}
