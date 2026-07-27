"""
Control plane service (FastAPI).

    POST /models/register   a node reports a retrain outcome
    GET  /models            model version history
    GET  /nodes             last known state of every node
    GET  /summary           retrain counts per node, for the dashboard
    POST /heartbeat         a node reports its /health

Records metadata only - no training data, no weights. See registry.py.

Run:  uvicorn edge_system.control.service:app --port 8000
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Dict, List, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from ..config import SYSTEM_CONFIG, ensure_dirs
from .registry import Registry


class HeartbeatRequest(BaseModel):
    node: str
    payload: Dict


STATE: Dict[str, Optional[Registry]] = {"registry": None}


@asynccontextmanager
async def lifespan(app: FastAPI):
    ensure_dirs()
    STATE["registry"] = Registry(SYSTEM_CONFIG["paths"]["registry_db"])
    yield
    if STATE["registry"] is not None:
        STATE["registry"].close()


app = FastAPI(title="FYP control plane", version="1.0", lifespan=lifespan)


def _registry() -> Registry:
    reg = STATE["registry"]
    if reg is None:
        raise HTTPException(503, "control plane not started")
    return reg


@app.post("/models/register")
def register(payload: Dict) -> Dict:
    return {"id": _registry().register(payload)}


@app.get("/models")
def models(node: Optional[str] = None, limit: int = 100) -> List[Dict]:
    return _registry().versions(node=node, limit=limit)


@app.post("/heartbeat")
def heartbeat(req: HeartbeatRequest) -> Dict:
    _registry().heartbeat(req.node, req.payload)
    return {"ok": True}


@app.get("/nodes")
def nodes() -> List[Dict]:
    return _registry().nodes()


@app.get("/summary")
def summary() -> Dict:
    return _registry().summary()


@app.get("/health")
def health() -> Dict:
    return {"ok": STATE["registry"] is not None}
