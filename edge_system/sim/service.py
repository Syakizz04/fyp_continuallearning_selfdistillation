"""
The interactive simulator (FastAPI): the demo's clock, run by the operator.

The batch driver (`SimulationDriver.run`) replays every day back to back. Here
the same driver trades **one day per step**, and each day goes through the
cycle a deployed retailer would:

    open    the gateway prices every SKU on every channel for the new day
            (recommendations, through the operator's rules)
    decide  the operator approves or rejects - the clock waits while paused
    trade   the day's prices go live: approved -> the recommendation, rejected
            -> the shelf price stays, undecided -> per `require_approval`
            (see Gateway.live_prices); then orders, reservations, drift checks

    GET  /state            clock, totals, per-day readout, event log
    POST /play  /pause  /step  /speed
    POST /outage           cut a channel off from central stock, or restore it
    POST /latency          one-way node <-> centre delay
    POST /shock            demand multiplier for N days (all SKUs or one)
    POST /clear-shock
    POST /restock          a delivery for one SKU or all of them
    POST /settings         require_approval, speed
    POST /reset            a fresh run: stock, base models, history; new policy

A demo, not an experiment: nothing it produces is reported as a result. The
experiment path is untouched - `trade_day` is the batch loop's own body.

Run:  uvicorn edge_system.sim.service:app --port 8030
      (run_system --scenario demo starts it; the scenario's `sim` section
       arrives as JSON in FYP_SIM_CONFIG, since this process is a child)
"""

from __future__ import annotations

import json
import os
import threading
import time
from collections import deque
from contextlib import asynccontextmanager
from typing import Dict, List, Optional

import httpx
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from ..config import (SYSTEM_CONFIG, channel_names, control_url, gateway_url,
                      inventory_url, node_url)

# The whole `sim` section, as the supervisor resolved it (scenario + CLI). The
# manifest in config.py carries scalars; this section is one structured unit.
if os.environ.get("FYP_SIM_CONFIG"):
    SYSTEM_CONFIG["sim"].update(json.loads(os.environ["FYP_SIM_CONFIG"]))
# The operator moves the network from the page, not a schedule.
SYSTEM_CONFIG["network"]["schedule"] = []

from .clock import Tick  # noqa: E402
from .driver import SimulationDriver  # noqa: E402

POLICIES = ("escrow_quota", "strong_lock", "eventual")
SPEEDS = (5.0, 2.0, 1.0, 0.0)      # seconds per trading day; 0 = as fast as it goes
HTTP = httpx.Client(timeout=30.0)


def _label(ch: str) -> str:
    from ..config import channel as chan  # noqa: PLC0415
    try:
        named = chan(ch).get("label")
    except KeyError:
        named = None
    return named or {"pos": "In-store till", "web": "Web store",
                     "marketplace": "Marketplace"}.get(ch, ch)


class Runner:
    def __init__(self) -> None:
        self.lock = threading.RLock()          # state
        self.day_lock = threading.Lock()       # one day (or a reset) at a time
        self.wake = threading.Event()
        self.mode = "starting"   # starting | paused | playing | finished | resetting | error
        self.error: Optional[str] = None
        self.speed_s = 2.0
        self.require_approval = False
        self.driver: Optional[SimulationDriver] = None
        self.dates: List = []
        self.cursor = 0                        # index of the open (not yet traded) day
        self.policy = SYSTEM_CONFIG["sync"]["policy"]
        self.delay_ms = float(SYSTEM_CONFIG["network"]["delay_ms"])
        self.partition: set = set()
        self.shock: Optional[Dict] = None
        self.days: deque = deque(maxlen=400)
        self.log: deque = deque(maxlen=200)
        self.last_generation: Optional[Dict] = None
        self._step = False
        self._nodes: Dict[str, Dict] = {}
        self._oversold = 0
        self.waiting_for: List[str] = []       # channels whose retrain holds the clock
        self._hold_since: Optional[float] = None
        self._gave_up: set = set()             # held past the cap; not held again this fit
        self.thread = threading.Thread(target=self._loop, name="sim-clock", daemon=True)

    # ── small helpers ───────────────────────────────────────────────────────

    @property
    def sim_date(self) -> Optional[str]:
        return self.dates[self.cursor].strftime("%Y-%m-%d") if self.cursor < len(self.dates) else None

    def note(self, kind: str, text: str, sim_date: Optional[str] = None) -> None:
        self.log.appendleft({"ts": time.time(), "kind": kind, "text": text,
                             "sim_date": sim_date or self.sim_date, "tick": self.cursor})

    def _post(self, url: str, **kw) -> Dict:
        if "/api/internal/" in url and os.environ.get("FYP_INTERNAL_TOKEN"):
            kw.setdefault("headers", {})["X-Internal-Token"] = os.environ["FYP_INTERNAL_TOKEN"]
        r = HTTP.post(url, **kw)
        r.raise_for_status()
        return r.json()

    def _network(self) -> None:
        self._post(f"{inventory_url()}/admin/network",
                   json={"delay_ms": self.delay_ms, "partition": sorted(self.partition)})
        if self.driver is not None:
            self.driver.delay_ms = self.delay_ms

    # ── run lifecycle ───────────────────────────────────────────────────────

    def boot(self, *, policy: Optional[str] = None, reload_models: bool = False) -> None:
        """A fresh run. At startup the nodes are freshly loaded already; a reset
        from the page reloads them too."""
        with self.day_lock:
            try:
                with self.lock:
                    self.mode = "resetting" if reload_models else "starting"
                    self.error = None
                policy = policy or self.policy
                self._post(f"{inventory_url()}/admin/reset",
                           params={"policy": policy, "clear_events": True})
                self.policy, self.partition, self.shock = policy, set(), None
                self.delay_ms = float(SYSTEM_CONFIG["network"]["delay_ms"])
                self._network()
                if reload_models:
                    self._reload_nodes()
                self._post(f"{control_url()}/admin/reset")
                self._post(f"{gateway_url()}/api/internal/reset")

                if self.driver is not None:
                    self.driver.close()
                driver = SimulationDriver(run_id="demo_interactive", verbose=False)
                driver.demand_multiplier = self._multiplier
                driver.stock_up()
                with self.lock:
                    self.driver, self.dates, self.cursor = driver, list(driver.clock.dates), 0
                    self.days.clear(); self.log.clear()
                    self._oversold = 0
                    self._nodes = self._node_snapshot()
                self.note("run", f"New run · sync policy {policy} · {len(driver.skus)} products")
                self._open_day()
                with self.lock:
                    self.mode = "paused"
            except Exception as exc:  # noqa: BLE001 - surface it on the page
                with self.lock:
                    self.mode, self.error = "error", f"{type(exc).__name__}: {exc}"

    def _reload_nodes(self) -> None:
        errors: List[str] = []

        def one(ch: str) -> None:
            try:
                HTTP.post(f"{node_url(ch)}/admin/reset", timeout=300.0).raise_for_status()
            except Exception as exc:  # noqa: BLE001
                errors.append(f"{ch}: {exc}")

        threads = [threading.Thread(target=one, args=(ch,)) for ch in channel_names()]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        if errors:
            raise RuntimeError("node reset failed - " + "; ".join(errors))

    def _open_day(self) -> None:
        d = self.sim_date
        if d is None:
            return
        try:
            self.last_generation = self._post(f"{gateway_url()}/api/internal/open-day",
                                              json={"sim_date": d}, timeout=120.0)
        except httpx.HTTPError as exc:
            self.last_generation = None
            self.note("warn", f"Recommendations failed for this day ({exc}); "
                              f"channels will price on their own")

    # ── one trading day ─────────────────────────────────────────────────────

    def trade(self) -> None:
        with self.day_lock:
            drv = self.driver
            if drv is None or self.cursor >= len(self.dates):
                return
            i, date = self.cursor, self.dates[self.cursor]
            d = date.strftime("%Y-%m-%d")

            sources: Dict[str, int] = {}
            try:
                live = self._post(f"{gateway_url()}/api/internal/live-prices",
                                  json={"sim_date": d, "require_approval": self.require_approval})
                drv.price_book = {(p["sku"], p["channel"]): (float(p["price"]), float(p["tier"]),
                                                             int(p["generation"]))
                                  for p in live["prices"]}
                sources = live["counts"]
            except httpx.HTTPError as exc:
                drv.price_book = None
                self.note("warn", f"Gateway unreachable ({exc}); channels priced on their own", d)

            tick = Tick(index=i, date=date, is_first=i == 0, is_last=i == len(self.dates) - 1)
            started = time.time()
            per = drv.trade_day(tick)

            try:
                oversold = int(HTTP.get(f"{inventory_url()}/metrics").json().get("oversell_units", 0))
            except Exception:  # noqa: BLE001
                oversold = self._oversold
            rows = list(per.values())
            req = sum(r.units_requested for r in rows)
            day = {
                "tick": i, "sim_date": d, "seconds": round(time.time() - started, 2),
                "orders": sum(r.orders for r in rows),
                "units_requested": req,
                "sold": sum(r.units_committed for r in rows),
                "refused": sum(r.units_refused for r in rows),
                "oversold": oversold - self._oversold,
                "revenue": round(sum(r.revenue for r in rows), 2),
                "fill_rate": (sum(r.units_granted for r in rows) / req) if req else None,
                "shock": self.shock["multiplier"] if self.shock and self._shock_on(i) else None,
                "partition": sorted(self.partition),
                "prices": sources,
                "channels": [{"channel": r.channel, "sold": r.units_committed,
                              "refused": r.units_refused, "revenue": round(r.revenue, 2),
                              "mean_price": round(r.mean_price, 2)} for r in rows],
            }
            with self.lock:
                self._oversold = oversold
                self.days.append(day)
                if self.shock and i >= self.shock["until_tick"]:
                    self.note("shock", "Demand back to normal", d)
                    self.shock = None
                self.cursor += 1
            self._watch_nodes(d)
            if self.cursor >= len(self.dates):
                with self.lock:
                    self.mode = "finished"
                self.note("run", "Last trading day reached", d)
            else:
                self._open_day()

    # ── node watch: turn the monitors' state into readable events ──────────

    def _node_snapshot(self) -> Dict[str, Dict]:
        out = {}
        for ch in channel_names():
            try:
                h = HTTP.get(f"{node_url(ch)}/health", timeout=3.0).json()
                dr = HTTP.get(f"{node_url(ch)}/drift", params={"limit": 1}, timeout=3.0).json()
                out[ch] = {"generation": h.get("model_generation") or 0,
                           "retraining": bool(h.get("retrain_in_flight")),
                           "events": dr["status"].get("n_events", 0),
                           "last": (dr.get("events") or [None])[-1]}
            except Exception:  # noqa: BLE001
                out[ch] = self._nodes.get(ch, {"generation": 0, "retraining": False,
                                               "events": 0, "last": None})
        return out

    def _watch_nodes(self, d: str) -> None:
        from ..config import channel as chan  # noqa: PLC0415
        now = self._node_snapshot()
        for ch, cur in now.items():
            was = self._nodes.get(ch, cur)
            label = chan(ch).get("label") or {"pos": "In-store till", "web": "Web store",
                                              "marketplace": "Marketplace"}.get(ch, ch)
            if cur["events"] > was["events"] and cur["last"]:
                kind = "pricing model" if cur["last"].get("kind") == "rl" else "demand forecaster"
                self.note("drift", f"{label}: drift warning on the {kind}", d)
            if cur["retraining"] and not was["retraining"]:
                self.note("retrain", f"{label}: retraining (SDFT)", d)
            if cur["generation"] > was["generation"]:
                self.note("model", f"{label}: model v{cur['generation']} now serving", d)
        self._nodes = now

    # ── disruptions ─────────────────────────────────────────────────────────

    def _shock_on(self, tick: int) -> bool:
        s = self.shock
        return bool(s) and s["from_tick"] <= tick <= s["until_tick"]

    def _multiplier(self, sku: str, tick) -> float:
        s = self.shock
        if not s or not self._shock_on(tick.index):
            return 1.0
        return s["multiplier"] if s["sku"] in (None, sku) else 1.0

    def set_shock(self, multiplier: float, days: int, sku: Optional[str]) -> None:
        until = min(self.cursor + days - 1, len(self.dates) - 1)
        with self.lock:
            self.shock = {"multiplier": multiplier, "sku": sku, "days": days,
                          "from_tick": self.cursor, "until_tick": until,
                          "until_date": self.dates[until].strftime("%Y-%m-%d")}
        what = "surge" if multiplier > 1 else "slump"
        self.note("shock", f"Demand {what}: ×{multiplier:g} for {days} days on "
                           f"{sku or 'every product'}")

    def restock(self, sku: Optional[str], qty: Optional[int]) -> int:
        drv = self.driver
        if drv is None:
            raise HTTPException(409, "no run yet")
        skus = [sku] if sku else list(drv.skus)
        units = 0
        for s in skus:
            if s not in drv._targets:
                raise HTTPException(404, f"unknown product {s!r}")
            q = int(qty or drv._targets[s])
            self._post(f"{inventory_url()}/replenish", json={
                "sku": s, "qty": q, "tick": self.cursor, "sim_date": self.sim_date})
            units += q
        self.note("restock", f"Delivery: {units} units of {sku or 'every product'}")
        return units

    # ── the clock ───────────────────────────────────────────────────────────

    def _retraining(self) -> List[str]:
        """Channels with a retrain in flight. An unreachable node holds nothing:
        it cannot be waited for, and the clock must not hang on it."""
        out = []
        for ch in channel_names():
            try:
                if HTTP.get(f"{node_url(ch)}/health", timeout=3.0).json().get("retrain_in_flight"):
                    out.append(ch)
            except Exception:  # noqa: BLE001
                pass
        return out

    def _hold(self) -> bool:
        """True while the next day must wait for a model update (see
        `hold_for_retrain` in config). Called before each day opens."""
        if not SYSTEM_CONFIG["sim"].get("hold_for_retrain", True):
            return False
        busy = self._retraining()
        self._gave_up &= set(busy)          # a finished fit can hold the clock again
        busy = [c for c in busy if c not in self._gave_up]
        if not busy:
            if self.waiting_for:
                self.note("model", "Model update finished; trading resumes")
            with self.lock:
                self.waiting_for, self._hold_since = [], None
            return False
        now = time.time()
        if self._hold_since is None:
            self._hold_since = now
            self.note("retrain", "Clock holding: the next day opens once "
                                 f"{', '.join(_label(c) for c in busy)} finishes updating")
        cap = float(SYSTEM_CONFIG["sim"].get("retrain_hold_max_s", 900.0))
        if now - self._hold_since > cap:
            self.note("warn", f"Model update still running after {cap / 60:g} min; "
                              "trading on the old model until it lands")
            self._gave_up |= set(busy)
            with self.lock:
                self.waiting_for, self._hold_since = [], None
            return False
        with self.lock:
            self.waiting_for = busy
        return True

    def _loop(self) -> None:
        while True:
            self.wake.wait(timeout=0.5)
            self.wake.clear()
            with self.lock:
                go = self.mode == "playing" or (self._step and self.mode == "paused")
            if not go:
                continue
            # Keep a pending Next day while holding: it trades once the model lands.
            if self._hold():
                continue
            with self.lock:
                self._step = False
            started = time.time()
            try:
                self.trade()
            except Exception as exc:  # noqa: BLE001
                with self.lock:
                    self.mode, self.error = "paused", f"{type(exc).__name__}: {exc}"
                self.note("warn", f"Trading day failed: {exc}")
                continue
            with self.lock:
                playing = self.mode == "playing"
            if playing:
                slack = self.speed_s - (time.time() - started)
                if slack > 0:
                    # A pause during the wait is honoured at once, not next day.
                    self.wake.wait(timeout=slack)
                self.wake.set()

    def state(self, brief: bool = False) -> Dict:
        with self.lock:
            days = list(self.days)
            out = {
                "mode": self.mode, "error": self.error, "speed_s": self.speed_s,
                "require_approval": self.require_approval,
                "tick": self.cursor, "total": len(self.dates),
                "sim_date": self.sim_date,
                "last_traded": days[-1]["sim_date"] if days else None,
                "policy": self.policy, "delay_ms": self.delay_ms,
                "partition": sorted(self.partition), "shock": self.shock,
                "waiting_for": list(self.waiting_for),
            }
            if brief:
                return out
            sold = sum(x["sold"] for x in days)
            req = sum(x["units_requested"] for x in days)
            refused = sum(x["refused"] for x in days)
            out.update({
                "channels": channel_names(), "policies": list(POLICIES), "speeds": list(SPEEDS),
                "skus": list(self.driver.skus) if self.driver else [],
                "totals": {"days": len(days), "sold": sold, "refused": refused,
                           "oversold": sum(x["oversold"] for x in days),
                           "revenue": round(sum(x["revenue"] for x in days), 2),
                           "fill_rate": ((req - refused) / req) if req else None,
                           "prices": {k: sum(x["prices"].get(k, 0) for x in days)
                                      for k in ("approved", "rejected", "auto", "unreviewed")}},
                "days": days[-120:],
                "log": list(self.log)[:80],
                "nodes": self._nodes,
                "last_generation": self.last_generation,
            })
            return out


RUN = Runner()


@asynccontextmanager
async def lifespan(app: FastAPI):
    RUN.thread.start()
    threading.Thread(target=RUN.boot, name="sim-boot", daemon=True).start()
    yield


app = FastAPI(title="Shelfsense simulator", version="1.0", lifespan=lifespan)


def _ready() -> None:
    if RUN.mode in ("starting", "resetting"):
        raise HTTPException(409, "the simulator is still setting up a run")


class SpeedIn(BaseModel):
    speed_s: float = Field(..., ge=0, le=30)


class OutageIn(BaseModel):
    channel: str
    offline: bool


class LatencyIn(BaseModel):
    delay_ms: float = Field(..., ge=0, le=5000)


class ShockIn(BaseModel):
    multiplier: float = Field(..., gt=0, le=10)
    days: int = Field(..., ge=1, le=120)
    sku: Optional[str] = None


class RestockIn(BaseModel):
    sku: Optional[str] = None
    qty: Optional[int] = Field(None, gt=0)


class SettingsIn(BaseModel):
    require_approval: Optional[bool] = None
    speed_s: Optional[float] = Field(None, ge=0, le=30)


class ResetIn(BaseModel):
    policy: Optional[str] = None


@app.get("/health")
def health() -> Dict:
    return {"ok": True, "mode": RUN.mode}


@app.get("/state")
def state(brief: bool = False) -> Dict:
    return RUN.state(brief=brief)


@app.post("/play")
def play(body: Optional[SettingsIn] = None) -> Dict:
    _ready()
    with RUN.lock:
        if RUN.mode == "finished":
            raise HTTPException(409, "the run is over; reset to start another")
        if body and body.speed_s is not None:
            RUN.speed_s = body.speed_s
        RUN.mode = "playing"
    RUN.wake.set()
    return RUN.state()


@app.post("/pause")
def pause() -> Dict:
    with RUN.lock:
        if RUN.mode == "playing":
            RUN.mode = "paused"
    RUN.wake.set()
    return RUN.state()


@app.post("/step")
def step() -> Dict:
    _ready()
    with RUN.lock:
        if RUN.mode == "playing":
            RUN.mode = "paused"
        if RUN.mode != "paused":
            raise HTTPException(409, f"cannot step while {RUN.mode}")
        RUN._step = True
    RUN.wake.set()
    return RUN.state()


@app.post("/speed")
def speed(body: SpeedIn) -> Dict:
    with RUN.lock:
        RUN.speed_s = body.speed_s
    RUN.wake.set()
    return RUN.state()


@app.post("/settings")
def settings(body: SettingsIn) -> Dict:
    with RUN.lock:
        if body.require_approval is not None and body.require_approval != RUN.require_approval:
            RUN.require_approval = body.require_approval
            RUN.note("settings", "Only approved prices go live" if body.require_approval
                     else "Undecided prices go live at the recommended price")
        if body.speed_s is not None:
            RUN.speed_s = body.speed_s
    return RUN.state()


@app.post("/outage")
def outage(body: OutageIn) -> Dict:
    _ready()
    if body.channel not in channel_names():
        raise HTTPException(404, f"unknown channel {body.channel!r}")
    with RUN.lock:
        changed = (body.channel in RUN.partition) != body.offline
        (RUN.partition.add if body.offline else RUN.partition.discard)(body.channel)
    if changed:
        RUN._network()
        RUN.note("outage", f"{body.channel}: {'lost' if body.offline else 'restored'} "
                           f"its link to central stock")
    return RUN.state()


@app.post("/latency")
def latency(body: LatencyIn) -> Dict:
    _ready()
    with RUN.lock:
        RUN.delay_ms = body.delay_ms
    RUN._network()
    RUN.note("latency", f"Network delay set to {body.delay_ms:g} ms")
    return RUN.state()


@app.post("/shock")
def shock(body: ShockIn) -> Dict:
    _ready()
    if body.sku and RUN.driver and body.sku not in RUN.driver.skus:
        raise HTTPException(404, f"unknown product {body.sku!r}")
    RUN.set_shock(body.multiplier, body.days, body.sku)
    return RUN.state()


@app.post("/clear-shock")
def clear_shock() -> Dict:
    with RUN.lock:
        had, RUN.shock = RUN.shock, None
    if had:
        RUN.note("shock", "Demand back to normal")
    return RUN.state()


@app.post("/restock")
def restock(body: RestockIn) -> Dict:
    _ready()
    RUN.restock(body.sku, body.qty)
    return RUN.state()


@app.post("/reset")
def reset(body: ResetIn) -> Dict:
    if body.policy and body.policy not in POLICIES:
        raise HTTPException(422, f"unknown policy {body.policy!r}; have {POLICIES}")
    if any(n.get("retraining") for n in RUN._nodes.values()):
        raise HTTPException(409, "a model is retraining; reset when it finishes")
    with RUN.lock:
        if RUN.mode in ("starting", "resetting"):
            raise HTTPException(409, "already setting up a run")
        RUN.mode = "resetting"
    threading.Thread(target=RUN.boot, kwargs={"policy": body.policy, "reload_models": True},
                     name="sim-reset", daemon=True).start()
    return RUN.state()
