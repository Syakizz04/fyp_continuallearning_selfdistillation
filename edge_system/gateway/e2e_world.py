"""
A whole, believable system behind the REAL gateway, for the browser tests.

`test_service.FakeUpstream` answers each route's unit tests with the minimum a
route needs. The dashboard reads far more - a simulator with a history, nodes
with drift streams and retrains, per-SKU sales - so this extends it with that,
shaped like the live services' responses. Every value is plausible on purpose:
the browser tests fail on a NaN or "undefined" on screen, and a fake full of
zeros would hide exactly the bugs they look for.

`serve()` runs the gateway, with the built dashboard, on a free local port.
"""

from __future__ import annotations

import copy
import datetime as dt
import socket
import threading
import time
from typing import Dict, List

from .test_service import CHANNELS, SKUS, FakeUpstream
from .upstream import UpstreamError

DAYS = 60
START = dt.date(2013, 1, 1)


def day(i: int) -> str:
    return (START + dt.timedelta(days=i)).isoformat()


class WorldUpstream(FakeUpstream):
    """FakeUpstream plus a simulator, drift streams, retrains and sales."""

    def __init__(self, *, sim_up: bool = True) -> None:
        super().__init__()
        self.sim_date = day(DAYS)
        self.sim_up = sim_up
        self.mode = "paused"
        self.partition: List[str] = []
        self.waiting_for: List[str] = []
        self.shock = None
        self.posts: List[tuple] = []

    # ── simulator ───────────────────────────────────────────────────────────
    def _state(self) -> Dict:
        days = []
        for i in range(DAYS):
            sold = 90 + (i * 7) % 25
            days.append({
                "tick": i, "sim_date": day(i), "seconds": 0.4, "orders": 40 + i % 9,
                "units_requested": sold + 6, "sold": sold, "refused": 6, "oversold": 0,
                "revenue": round(sold * 3.1, 2), "fill_rate": sold / (sold + 6),
                "shock": 2.5 if 50 <= i < 55 else None,
                "partition": ["marketplace"] if 20 <= i < 25 else [],
                "prices": {"approved": 20, "rejected": 2, "auto": 8},
                "channels": [{"channel": c, "sold": sold // 3, "refused": 2,
                              "revenue": round(sold, 2), "mean_price": 3.1} for c in CHANNELS],
            })
        return {
            "mode": self.mode, "error": None, "speed_s": 1.0, "require_approval": False,
            "tick": DAYS, "total": 400, "sim_date": self.sim_date, "last_traded": day(DAYS - 1),
            "policy": "escrow_quota", "delay_ms": 50.0, "partition": list(self.partition),
            "waiting_for": list(self.waiting_for),
            "shock": copy.deepcopy(self.shock), "channels": CHANNELS,
            "policies": ["escrow_quota", "strong_lock", "eventual"], "speeds": [5.0, 2.0, 1.0, 0.0],
            "skus": SKUS,
            "totals": {"days": DAYS, "sold": sum(d["sold"] for d in days),
                       "refused": 6 * DAYS, "oversold": 0,
                       "revenue": round(sum(d["revenue"] for d in days), 2), "fill_rate": 0.94,
                       "prices": {"approved": 1200, "rejected": 120, "auto": 480, "unreviewed": 0}},
            "days": days,
            "log": [{"ts": 1.7e9 + i, "kind": k, "text": t, "sim_date": day(i), "tick": i}
                    for i, (k, t) in enumerate([("info", "run started"),
                                                ("outage", "marketplace offline"),
                                                ("shock", "demand x2.5 for 10 days")])],
            "nodes": {c: {"generation": 2, "retraining": c == "web", "events": 3} for c in CHANNELS},
            "last_generation": {"created": 30, "seconds": 1.2, "n_errors": 0},
        }

    def sim(self, path, params=None):
        if not self.sim_up:
            raise UpstreamError("simulator", "unreachable (ConnectError)")
        if path == "/state":
            return self._state()
        raise AssertionError(f"unexpected sim path {path}")

    def sim_post(self, path, body=None):
        if not self.sim_up:
            raise UpstreamError("simulator", "unreachable (ConnectError)")
        self.posts.append((path, body))
        if path == "/play":
            self.mode = "playing"
        elif path == "/pause":
            self.mode = "paused"
        elif path == "/outage":
            ch = body["channel"]
            self.partition = ([*self.partition, ch] if body["offline"]
                              else [c for c in self.partition if c != ch])
        elif path == "/shock":
            self.shock = {"multiplier": body["multiplier"], "sku": body.get("sku"),
                          "days": body["days"], "from_tick": DAYS,
                          "until_tick": DAYS + body["days"] - 1,
                          "until_date": day(DAYS + body["days"] - 1)}
        elif path == "/clear-shock":
            self.shock = None
        return self._state()

    # ── edge nodes ──────────────────────────────────────────────────────────
    def node(self, name, path, params=None, timeout=None):
        self._check(f"edge node {name}")
        if path == "/drift":
            limit = int((params or {}).get("limit", 20))
            stream = []
            # Like a real node: each check carries the line it was judged
            # against, and that line moves when a retrain recalibrates it - so
            # the warnings sit beyond their own day's line, not today's. After
            # a retrain there is no line while the node re-learns (None).
            for i in range(14, DAYS):
                pi = 0.66 if i in (29, 30) else 0.92 + 0.05 * ((i * 3) % 5) / 5
                stream.append({"tick": i, "date": day(i), "mase": None,
                               "profit_index": pi, "profit_index_daily": 0.9,
                               "fc_threshold": None,
                               "rl_threshold": 0.7 if i <= 32 else None if i <= 44 else 0.8})
                m = {50: 2.0, 51: 2.1, 52: 2.2}.get(i, 1.3 + 0.02 * (i % 7))
                stream.append({"tick": i, "date": day(i), "mase": m,
                               "profit_index": None, "profit_index_daily": None,
                               "fc_threshold": 1.9 if i <= 54 else None if i <= 56 else 1.6,
                               "rl_threshold": None})
            events = [{"tick": 52, "sim_date": day(52), "kind": "forecasting", "value": 2.2,
                       "threshold": 1.9, "node": name},
                      {"tick": 30, "sim_date": day(30), "kind": "rl", "value": 0.66,
                       "threshold": 0.7, "node": name}]
            return {"status": {
                        "node": name, "checks": len(stream), "fc_threshold": 1.6,
                        "rl_threshold": 0.8, "fc_streak": 0, "fc_signal": "realised",
                        "fc_calibration": {"mu": 1.35, "sigma": 0.08, "k": 3.0,
                                           "warmup_days": 28, "consecutive": 3},
                        "fc_warmup_left": 0, "rl_streak": 0, "rl_detector": "rolling",
                        "rl_calibration": {"mu": 0.95, "sigma": 0.05, "k": 2.5,
                                           "window_days": 7, "warmup_days": 28},
                        "rl_warmup_left": 0, "last_mase": 1.41, "last_profit_index": 0.95,
                        "n_events": len(events)},
                    "events": events[-limit:], "stream": stream[-limit:]}
        if path == "/health":
            return {"ok": True, "node": name, "model_generation": 2, "sdft_alpha": 0.99,
                    "retrain_in_flight": name == "web",
                    "last_fc_check": {"tick": DAYS, "sim_date": self.sim_date, "mase": 1.41,
                                      "seconds": 0.06, "ts": time.time() - 30,
                                      "realised_days": 46},
                    "forecaster": {"forecaster_generation": 1,
                                   "forecaster_updated_sim_date": day(53),
                                   "forecaster_updated_at": time.time() - 600}}
        if path == "/model":
            return {"node": name, "version": {"generation": 2}, "history": [
                {"tick": 52, "sim_date": day(52), "kind": "forecasting", "strategy": "sdft",
                 "reason": "forecasting drift: 2.2 vs 1.6", "ok": True, "generation": 2,
                 "duration_s": 51.0, "skipped": False}]}
        return super().node(name, path, params, timeout)

    # ── inventory ───────────────────────────────────────────────────────────
    def inventory(self, path, params=None):
        if path == "/daily":
            self._check("inventory service")
            return {s: [{"sim_date": day(i), "sold": 5 + i % 4, "refused": i % 2,
                         "ordered": 6 + i % 4, "stock": 40 - i % 10} for i in range(DAYS)]
                    for s in SKUS}
        if path == "/events":
            self._check("inventory service")
            return [{"ts": 1.7e9 + i, "kind": k, "node": c, "sku": SKUS[0], "qty": 3,
                     "sim_date": day(i)}
                    for i, (k, c) in enumerate([("refill", "web"), ("reserve", "pos"),
                                                ("reject", "marketplace")])]
        return super().inventory(path, params)

    # ── control plane ───────────────────────────────────────────────────────
    def control(self, path, params=None):
        if path == "/models":
            self._check("control plane")
            return [{"id": i, "ts": 1.7e9 + i, "node": c, "kind": k, "generation": g,
                     "sim_date": day(d), "reason": f"{k} drift", "ok": ok,
                     "duration_s": dur, "skipped": 0, "error": err}
                    for i, (c, k, g, d, ok, dur, err) in enumerate([
                        ("web", "rl", 1, 34, 1, 110.0, None),
                        ("marketplace", "forecasting", 1, 55, 1, 51.0, None),
                        ("pos", "forecasting", 2, 58, 0, 12.0, "RuntimeError: example failure")])]
        return super().control(path, params)


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class Served:
    """The real gateway app on a real socket, in a thread."""

    def __init__(self, up: WorldUpstream, db_dir) -> None:
        import uvicorn

        from . import service as svc
        from .auth import AuthStore
        from .store import Store

        self.up = up
        svc.GW["gw"] = svc.Gateway(Store(str(db_dir / "decisions.db")), up)
        svc.AUTH["auth"] = AuthStore(str(db_dir / "auth.db"))
        self.port = free_port()
        self.url = f"http://127.0.0.1:{self.port}"
        self.server = uvicorn.Server(uvicorn.Config(svc.app, host="127.0.0.1", port=self.port,
                                                    log_level="warning", lifespan="on"))
        self.thread = threading.Thread(target=self.server.run, daemon=True)
        self._svc = svc

    def __enter__(self):
        self.thread.start()
        deadline = time.time() + 20
        while not self.server.started:
            if time.time() > deadline:
                raise RuntimeError("gateway did not start")
            time.sleep(0.05)
        return self

    def __exit__(self, *exc):
        self.server.should_exit = True
        self.thread.join(10)
        self._svc.GW["gw"] = None
        self._svc.AUTH["auth"] = None
