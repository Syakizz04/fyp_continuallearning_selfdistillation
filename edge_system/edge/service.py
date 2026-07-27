"""
Edge node service (FastAPI).

One process per sales channel. Everything a request touches is local: the TFT
and PPO live in this process's memory, drift is detected here, and retraining
runs here. The only outbound calls are to the inventory service to reserve stock
and to the control plane to register a new model version.

    GET  /price      price for one SKU, using this node's own (stale) stock view
    GET  /forecast   local TFT forecast
    POST /check      feed one tick's metrics to the drift monitor
    GET  /health     model generation, load time, readiness
    GET  /model      what is being served and its retrain history

Run:  uvicorn edge_system.edge.service:app --port 8010
      FYP_NODE=pos FYP_STRATEGY=sdft uvicorn ... --port 8010
"""

from __future__ import annotations

import os
import time
from contextlib import asynccontextmanager
from typing import Dict, List, Optional

import httpx
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel

from ..config import SYSTEM_CONFIG, inventory_url, control_url
from .inference import LocalInference
from .monitor_local import LocalDriftMonitor
from .retrain_local import LocalRetrainer


class CheckRequest(BaseModel):
    tick: int
    sim_date: str
    mase: Optional[float] = None
    profit_index: Optional[float] = None
    # The experiment scripts set this so a retrain lands before the next tick is
    # evaluated; the live system leaves it False so serving is never blocked.
    blocking: bool = False


class EdgeNode:
    """Everything one channel owns."""

    def __init__(self) -> None:
        self.name: str = os.environ.get("FYP_NODE", "pos")
        self.strategy: str = os.environ.get("FYP_STRATEGY", "sdft")
        self.inference: Optional[LocalInference] = None
        self.monitor: Optional[LocalDriftMonitor] = None
        self.retrainer: Optional[LocalRetrainer] = None
        self.started_at: float = 0.0

    def start(self) -> None:
        self.inference = LocalInference(self.name)
        self.inference.load()
        self.monitor = LocalDriftMonitor(self.name, self.inference.calibration)
        self.retrainer = LocalRetrainer(
            self.name, self.inference, self.strategy,
            on_complete=self._register_version,
        )
        self.started_at = time.time()

    def require(self) -> LocalInference:
        if self.inference is None or not self.inference.ready:
            raise HTTPException(503, "node still loading models")
        return self.inference

    def _register_version(self, record) -> None:
        """Tell the control plane a new model generation is live. Best-effort:
        the node must keep serving even if the control plane is down."""
        try:
            httpx.post(f"{control_url()}/models/register", timeout=2.0, json={
                "node": self.name, "strategy": self.strategy,
                **record.as_dict(),
            })
        except Exception:  # noqa: BLE001, S110
            pass


NODE = EdgeNode()


@asynccontextmanager
async def lifespan(app: FastAPI):
    NODE.start()
    yield


app = FastAPI(title="FYP edge node", version="1.0", lifespan=lifespan)


# ─── Inference ───────────────────────────────────────────────────────────────

@app.get("/price")
def price(sku: str, sim_date: str,
          inventory_level: Optional[float] = Query(
              None, description="Override stock. Omit to ask the inventory "
                                "service for this node's own view.")) -> Dict:
    """
    Price one SKU.

    When `inventory_level` is omitted the node prices against **its own belief
    about global stock** - the last figure it saw from the centre, less what it
    has sold since. That is what a deployed node would hold, and it is the
    signal that carries sync staleness into the pricing decision.

    It deliberately does NOT price against the node's quota. A quota is a
    different variable on a different scale (10 units of selling rights against
    495 units of real stock), and feeding that to an agent trained on true
    inventory would demolish it rather than degrade it - E2 would be measuring a
    unit mismatch instead of staleness. The estimate is the same variable as
    ground truth, merely out of date, so it degrades smoothly with the refresh
    interval and is a usable treatment.
    """
    inf = NODE.require()
    stale_view = inventory_level
    source = "caller"

    if stale_view is None:
        source = "node_estimate"
        try:
            r = httpx.get(f"{inventory_url()}/stock/{sku}", timeout=2.0)
            r.raise_for_status()
            stale_view = float(r.json()["stock_estimates"].get(NODE.name, 0.0))
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(502, f"inventory service unreachable: {exc}")

    try:
        out = inf.price(sku, sim_date=sim_date, inventory_level=stale_view)
    except KeyError as exc:
        raise HTTPException(404, str(exc))
    out["node"] = NODE.name
    out["inventory_source"] = source
    return out


@app.get("/forecast")
def forecast(sku: str, sim_date: str) -> Dict:
    inf = NODE.require()
    try:
        out = inf.forecast(sku, sim_date=sim_date)
    except KeyError as exc:
        raise HTTPException(404, str(exc))
    out["node"] = NODE.name
    return out


# ─── Drift and retraining ────────────────────────────────────────────────────

@app.post("/check")
def check(req: CheckRequest) -> Dict:
    """Feed one tick's metrics to the local monitor; retrain if it fires."""
    if NODE.monitor is None or NODE.retrainer is None:
        raise HTTPException(503, "node not started")

    events = NODE.monitor.update(
        tick=req.tick, sim_date=req.sim_date,
        mase=req.mase, profit_index=req.profit_index,
    )
    triggered = [
        NODE.retrainer.on_drift(e, blocking=req.blocking).as_dict()
        for e in events
    ]
    return {
        "node": NODE.name,
        "events": [e.as_dict() for e in events],
        "retrains": triggered,
        "monitor": NODE.monitor.status(),
    }


@app.get("/model")
def model() -> Dict:
    inf = NODE.require()
    return {
        "node": NODE.name,
        "strategy": NODE.strategy,
        "version": inf.version.as_dict(),
        "load_seconds": inf.load_seconds,
        "retrainer": NODE.retrainer.status() if NODE.retrainer else None,
        "history": NODE.retrainer.recent() if NODE.retrainer else [],
    }


@app.get("/drift")
def drift(limit: int = 20) -> Dict:
    if NODE.monitor is None:
        raise HTTPException(503, "node not started")
    return {
        "status": NODE.monitor.status(),
        "events": NODE.monitor.recent_events(limit),
        "stream": NODE.monitor.stream[-limit:],
    }


@app.get("/health")
def health() -> Dict:
    ready = NODE.inference is not None and NODE.inference.ready
    return {
        "ok": ready,
        "node": NODE.name,
        "strategy": NODE.strategy,
        "uptime_s": time.time() - NODE.started_at if NODE.started_at else 0.0,
        "model_generation": NODE.inference.version.generation if ready else None,
        "load_seconds": NODE.inference.load_seconds if ready else None,
        "retrain_in_flight": NODE.retrainer.busy if NODE.retrainer else False,
    }
