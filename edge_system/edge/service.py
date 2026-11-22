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
import threading
import time
from contextlib import asynccontextmanager
from typing import Dict, List, Optional

import httpx
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel

from ..config import SYSTEM_CONFIG, channel, inventory_url, control_url
from .inference import LocalInference
from .monitor_local import LocalDriftMonitor
from .retrain_local import LocalRetrainer

# One pooled client for every call this node makes. A bare `httpx.get` builds a
# fresh client - SSL context, certificate bundle and all - per request, which
# costs ~1 s on Windows; /price called it per SKU, so a 10-SKU, 3-channel day
# took ~30 s instead of well under one. A shared client reuses the connection.
HTTP = httpx.Client()


class CheckRequest(BaseModel):
    tick: int
    sim_date: str
    mase: Optional[float] = None
    profit_index: Optional[float] = None
    # The day's revenue and its base-price reference, of which profit_index is
    # the ratio. Only the rolling RL detector reads them.
    revenue: Optional[float] = None
    reference: Optional[float] = None
    # Per SKU at this channel: {sku, price, requested, sold, refused}. With
    # models.rl_signal = "critic" the node judges its pricer on these instead
    # (LocalInference.pricing_outcome) and revenue/reference above are unused.
    pricing: Optional[List[Dict]] = None
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
        # The monitor is now fed from two threads (the request, and the weekly
        # forecast check), and DebouncedDetector streaks are read-modify-write.
        self.monitor_lock = threading.Lock()
        self._fc_check_thread: Optional[threading.Thread] = None
        self.last_fc_check: Optional[Dict] = None

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

    # ── Weekly forecast check ───────────────────────────────────────────────

    def fc_check_due(self, tick: int) -> bool:
        every = int(SYSTEM_CONFIG["models"]["fc_check_every_ticks"])
        return every > 0 and tick > 0 and tick % every == 0

    def run_fc_check(self, tick: int, sim_date: str) -> List[Dict]:
        """Score the forecaster on realised demand and feed it to the monitor.

        Returns the retrains it triggered. Synchronous; `schedule_fc_check` is
        the non-blocking wrapper the live system uses.
        """
        inf = self.require()
        started = time.time()
        if SYSTEM_CONFIG["models"].get("fc_signal", "dataset") == "realised":
            mase = self._realised_error(inf, sim_date)
        else:
            mase = inf.realised_mase(sim_date=sim_date)
        self.last_fc_check = {"tick": tick, "sim_date": sim_date, "mase": mase,
                              "seconds": time.time() - started, "ts": time.time(),
                              "realised_days": inf.realised_days}
        if mase is None:
            return []
        with self.monitor_lock:
            events = self.monitor.update(tick=tick, sim_date=sim_date, mase=mase)
        return [self.retrainer.on_drift(e).as_dict()
                for e in events if e.kind == "forecasting"]

    #: Realised days fetched per check: the encoder (60) + horizon (14), and the
    #: retrain's 8 recent weeks padded by an encoder - see retrain._recent_tft.
    REALISED_DAYS = 130

    def _realised_error(self, inf: LocalInference, sim_date: str) -> Optional[float]:
        """Refresh this node's view of what it has sold, then score the
        forecaster on it. None until a whole forecast horizon has traded - an
        earlier window would still be part dataset, part simulation."""
        from drift_pipeline.core_pipeline import CONFIG as DRIFT_CONFIG  # noqa: PLC0415
        # Channel weights are launch topology (like the ports every process
        # reads), and no scenario overrides them - so this child's copy is the
        # supervisor's, which is not true of the manifest sections.
        share = channel(self.name)["weight"]
        try:
            daily = HTTP.get(f"{inventory_url()}/daily", timeout=10.0,
                             params={"days": self.REALISED_DAYS, "node": self.name}).json()
        except Exception:  # noqa: BLE001 - no data is no signal, not a crash
            return None
        if inf.apply_realised(daily, share=share) < DRIFT_CONFIG["forecasting"]["prediction_length"]:
            return None
        return inf.realised_forecast_error(sim_date=sim_date)

    def schedule_fc_check(self, tick: int, sim_date: str) -> bool:
        """Run the check in the background; skip if the previous one is still
        going. A scoring pass takes seconds, and a backed-up queue of them would
        score weeks that are already stale by the time they run."""
        t = self._fc_check_thread
        if t is not None and t.is_alive():
            return False
        self._fc_check_thread = threading.Thread(
            target=self.run_fc_check, args=(tick, sim_date),
            name=f"fc-check-{self.name}", daemon=True)
        self._fc_check_thread.start()
        return True

    def _register_version(self, record) -> None:
        """Tell the control plane a new model generation is live. Best-effort:
        the node must keep serving even if the control plane is down."""
        if record.ok and self.monitor is not None:
            with self.monitor_lock:
                if record.kind == "rl":
                    self.monitor.recalibrate_rl()
                else:
                    self.monitor.recalibrate_fc()
        try:
            HTTP.post(f"{control_url()}/models/register", timeout=2.0, json={
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
            r = HTTP.get(f"{inventory_url()}/stock/{sku}", timeout=2.0)
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

    profit_index, revenue, reference = req.profit_index, req.revenue, req.reference
    # Rolling mode only: the fixed detector holds one day against a floor of
    # 1.0, which is calibrated to the revenue index and means nothing here.
    signal = SYSTEM_CONFIG["models"].get("rl_signal", "revenue")
    if (signal in ("critic", "markup")
            and NODE.monitor.rl_mode == "rolling"
            and req.pricing is not None and NODE.inference is not None):
        # Outside the monitor lock: a few ms normally, seconds once per pricer
        # (the critic's line / markup coefficient is refitted for a new one).
        judge = (NODE.inference.pricing_outcome if signal == "critic"
                 else NODE.inference.markup_outcome)
        got = judge(sim_date=req.sim_date, outcomes=req.pricing,
                    share=float(channel(NODE.name).get("weight", 1.0)))
        if got is None:
            profit_index = revenue = reference = None      # nothing traded: no reading
        else:
            revenue, reference = got
            profit_index = revenue / reference if reference > 0 else None

    with NODE.monitor_lock:
        events = NODE.monitor.update(
            tick=req.tick, sim_date=req.sim_date,
            mase=req.mase, profit_index=profit_index,
            revenue=revenue, reference=reference,
        )
    triggered = [
        NODE.retrainer.on_drift(e, blocking=req.blocking).as_dict()
        for e in events
    ]

    # The driver reports profit only, so without this the forecasting detector
    # never sees a value and the forecaster can never retrain. A caller that
    # supplies its own MASE (an experiment script) is taken at its word.
    fc_check = None
    if req.mase is None and NODE.fc_check_due(req.tick):
        if req.blocking:
            triggered += NODE.run_fc_check(req.tick, req.sim_date)
            fc_check = "ran"
        else:
            fc_check = ("scheduled" if NODE.schedule_fc_check(req.tick, req.sim_date)
                        else "skipped_busy")

    return {
        "node": NODE.name,
        "events": [e.as_dict() for e in events],
        "retrains": triggered,
        "fc_check": fc_check,
        "monitor": NODE.monitor.status(),
    }


@app.post("/admin/reset")
def admin_reset() -> Dict:
    """Back to the base checkpoints with an empty drift monitor - the interactive
    simulator's "reset run". Reloads from disk rather than keeping a pristine
    copy in memory: a fine-tune may share tensors with the model it started
    from, and a reset that quietly kept a retrained weight would be worse than
    none. Synchronous; takes as long as a cold start."""
    if NODE.retrainer is not None and NODE.retrainer.busy:
        raise HTTPException(409, "a retrain is in flight; try again when it finishes")
    with NODE.monitor_lock:
        # Drop the old models before loading new ones: two copies side by side
        # is ~1 GB extra per node, which a demo laptop running three does not have.
        NODE.inference = NODE.monitor = NODE.retrainer = None
        import gc  # noqa: PLC0415
        gc.collect()
        NODE.start()
        NODE.last_fc_check = None
    return {"ok": True, "node": NODE.name,
            "model_generation": NODE.inference.version.generation}


@app.get("/model")
def model() -> Dict:
    inf = NODE.require()
    return {
        "node": NODE.name,
        "strategy": NODE.strategy,
        "version": inf.version.as_dict(),
        "load_seconds": inf.load_seconds,
        "base_files": getattr(inf, "base_files", None),
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
        # Read back off the live object: proof the deployed trade-off arrived.
        "sdft_alpha": NODE.inference.sdft_alpha if ready else None,
        "forecaster": NODE.inference._forecaster_meta() if ready else None,
        "last_fc_check": NODE.last_fc_check,
    }
