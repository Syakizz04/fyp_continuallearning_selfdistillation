"""
On-node retraining.

When a node's local drift monitor fires, it retrains **on the node** - the claim
the whole project rests on. Two properties matter here and both are load-bearing:

1. **The node keeps serving.** Retraining runs on a background thread and builds
   a new model object; the old one answers requests until the new one is ready
   and swapped in atomically. A node that went offline to retrain would not be a
   credible deployment.

2. **The retraining is the same code the experiments used.** This wraps
   `drift_pipeline.retrain.RetrainController` rather than reimplementing SDFT /
   replay / EWC. If the deployed retrain diverged from the measured one, the
   FYP1 results would not transfer to the system, and the comparison in E2 would
   be between two different things.

Only one retrain runs at a time per node. A trigger arriving mid-retrain of the
SAME model is recorded and dropped, not queued: by the time the in-flight
retrain lands, the condition that fired the second trigger has usually been
addressed by it, and queueing would produce the "retrain churn" that EWC
already showed in the first real run.

A trigger for the OTHER model is deferred instead - the latest one per model,
started the moment the in-flight retrain lands. Retraining the pricer does
nothing for the forecaster, so dropping it would just lose it: live, a PPO
retrain takes ~2 minutes, and at full demo speed that is ~40 trading days in
which every forecaster alarm used to be discarded.

Why not run the two models concurrently - neither reads the other's model?
Measured (2026-11-15, x2.5 surge, demo stack): it is SLOWER. Both retrains are
threads in one node process and both are mostly Python loops (PPO rollouts,
Lightning's fit), so the GIL makes them take turns. Concurrent, a node took
400-470 s to land both (pricer 409-473 s, forecaster 195-390 s); in sequence it
took ~190 s (pricer 110-145 s, then forecaster 42-65 s), with machine CPU at
55% - the GIL, not the cores, was the limit. Real parallelism would need a
separate process per retrain, which costs another copy of the models' memory.

Experiments run with `blocking=True`, so they never reach either branch.
"""

from __future__ import annotations

import threading
import time
import traceback
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional


@dataclass
class RetrainRecord:
    tick: int
    sim_date: str
    kind: str                 # "forecasting" | "rl"
    strategy: str
    reason: str
    started_at: float
    finished_at: Optional[float] = None
    ok: bool = False
    generation: Optional[int] = None
    error: Optional[str] = None
    traceback: Optional[str] = None
    skipped: bool = False
    deferred: bool = False    # waiting for the other model's retrain to land

    @property
    def duration_s(self) -> Optional[float]:
        return None if self.finished_at is None else self.finished_at - self.started_at

    def as_dict(self) -> Dict:
        return {
            "tick": self.tick, "sim_date": self.sim_date, "kind": self.kind,
            "strategy": self.strategy, "reason": self.reason,
            "started_at": self.started_at, "finished_at": self.finished_at,
            "duration_s": self.duration_s, "ok": self.ok,
            "generation": self.generation, "error": self.error,
            "skipped": self.skipped, "deferred": self.deferred,
        }


class LocalRetrainer:
    """Runs drift-triggered retraining for one node, off the request path."""

    def __init__(self, node: str, inference, strategy: str = "sdft",
                 *, on_complete: Optional[Callable[[RetrainRecord], None]] = None) -> None:
        self.node = node
        self.inference = inference
        self.strategy = strategy
        self.history: List[RetrainRecord] = []
        self._lock = threading.Lock()
        self._thread: Optional[threading.Thread] = None
        self._kind: Optional[str] = None             # kind of the in-flight retrain
        self._deferred: Dict[str, RetrainRecord] = {}
        self._controller = None
        self._on_complete = on_complete

    # ── State ───────────────────────────────────────────────────────────────

    @property
    def busy(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def controller(self):
        """Lazily build the RetrainController; it seeds CL state on first use."""
        if self._controller is None:
            from drift_pipeline.retrain import RetrainController
            self._controller = RetrainController(
                self.strategy, self.inference.base_bundle, self.inference.data,
            )
        return self._controller

    # ── Trigger handling ────────────────────────────────────────────────────

    def on_drift(self, event, *, blocking: bool = False) -> RetrainRecord:
        """
        Handle one drift event. Returns immediately unless `blocking`.

        `blocking=True` is for the experiment scripts, which need the retrain to
        have landed before the next simulated tick is evaluated; the live system
        uses the default so serving is never held up.
        """
        rec = RetrainRecord(
            tick=event.tick, sim_date=event.sim_date, kind=event.kind,
            strategy=self.strategy,
            reason=f"{event.kind} drift: {event.value:.4f} vs {event.threshold:.4f}",
            started_at=time.time(),
        )

        with self._lock:
            if self.busy and self._kind == rec.kind:
                # Dropped, not queued - see the module docstring.
                rec.skipped = True
                rec.finished_at = time.time()
                rec.error = "a retrain was already in flight"
                self.history.append(rec)
                return rec
            if self.busy:
                # The other model is retraining: run this one after it.
                old = self._deferred.get(rec.kind)
                if old is not None:
                    old.skipped, old.finished_at = True, time.time()
                    old.error = "superseded by a later trigger"
                    old.deferred = False
                rec.deferred = True
                self._deferred[rec.kind] = rec
                self.history.append(rec)
                return rec

            self.history.append(rec)
            self._start(rec)

        if blocking:
            self.wait()
        return rec

    def _start(self, rec: RetrainRecord) -> None:
        """Start `rec` on a fresh thread. Caller holds `_lock`."""
        rec.started_at = time.time()
        rec.deferred = False
        self._kind = rec.kind
        self._thread = threading.Thread(
            target=self._run, args=(rec,), name=f"retrain-{self.node}", daemon=True,
        )
        self._thread.start()

    @staticmethod
    def _clone_forecaster(served, train_ds):
        """A weight-identical CLTFT, built the way the SDFT teacher is
        (drift_pipeline/retrain.py `_set_fc_teacher`): state_dict, not deepcopy,
        which breaks on tensors cached by a forward pass."""
        from drift_pipeline.core_pipeline import DEVICE  # noqa: PLC0415
        from drift_pipeline.trainers import build_cltft  # noqa: PLC0415
        model = build_cltft(train_ds, cl_method=getattr(served, "cl_method", "naive"))
        model.load_state_dict({k: v for k, v in served.state_dict().items()
                               if not k.startswith("teacher.")})
        return model.to(DEVICE)

    @staticmethod
    def _clone_pricer(served):
        """A clean copy of the served PPO through SB3's own save/load: the
        weights, optimiser state and timestep count, none of the tensors a
        previous fit left on the policy. The env is not saved; retrain_pricer
        sets one."""
        import io  # noqa: PLC0415

        from stable_baselines3 import PPO  # noqa: PLC0415
        buf = io.BytesIO()
        served.save(buf)
        buf.seek(0)
        return PPO.load(buf, device=served.device)

    # ── Retraining in a child process ───────────────────────────────────────

    def _use_child(self) -> bool:
        from ..config import SYSTEM_CONFIG  # noqa: PLC0415
        from .retrain_child import CHILD_STRATEGIES  # noqa: PLC0415
        return (SYSTEM_CONFIG["models"].get("retrain_process", "thread") == "subprocess"
                and self.strategy in CHILD_STRATEGIES)

    #: A child that has not finished by then is stuck, not slow.
    CHILD_TIMEOUT_S = 1800

    def _run_in_child(self, rec: RetrainRecord):
        """Retrain in `retrain_child`, then load its result here and swap it in.
        Returns the new ModelVersion. See retrain_child for why."""
        import json  # noqa: PLC0415
        import os  # noqa: PLC0415
        import shutil  # noqa: PLC0415
        import subprocess  # noqa: PLC0415
        import sys  # noqa: PLC0415
        import tempfile  # noqa: PLC0415
        from pathlib import Path  # noqa: PLC0415

        import torch  # noqa: PLC0415
        from filelock import FileLock  # noqa: PLC0415

        from ..config import PROJECT_ROOT, SYSTEM_CONFIG  # noqa: PLC0415
        from .retrain_child import strip_teacher  # noqa: PLC0415

        root = Path(SYSTEM_CONFIG["paths"]["state_dir"]) / "retrain_jobs"
        root.mkdir(parents=True, exist_ok=True)
        job = Path(tempfile.mkdtemp(prefix=f"{self.node}-{rec.kind}-", dir=root))
        inf = self.inference
        try:
            (job / "job.json").write_text(json.dumps({
                "kind": rec.kind, "origin": rec.sim_date, "reason": rec.reason,
                "node": self.node, "strategy": self.strategy}))
            if rec.kind == "forecasting":
                torch.save(strip_teacher(inf.forecaster().state_dict()), job / "forecaster.pt")
                realised = inf.data.get("tft_full_censored")
                if realised is not None:
                    realised.to_pickle(job / "realised.pkl")
            else:
                inf.pricer().save(job / "pricer.zip")

            # Machine-wide, one child at a time: each needs ~1.7 GB while it
            # runs, and a demand shock fires every node within a day or two -
            # three children at once is what this exists to avoid. A node
            # waiting here still serves; only its retrain queues.
            with FileLock(str(root / "child.lock")), open(job / "child.log", "w") as log:
                proc = subprocess.run(
                    [sys.executable, "-m", "edge_system.edge.retrain_child", str(job)],
                    cwd=str(PROJECT_ROOT), env=os.environ.copy(),
                    stdout=log, stderr=subprocess.STDOUT, timeout=self.CHILD_TIMEOUT_S)
            if proc.returncode != 0 or not (job / "result.json").exists():
                tail = (job / "child.log").read_text(errors="replace").strip().splitlines()[-3:]
                raise RuntimeError(f"retrain child exited {proc.returncode}: " + " | ".join(tail))

            if rec.kind == "forecasting":
                from drift_pipeline.core_pipeline import DEVICE  # noqa: PLC0415
                from drift_pipeline.trainers import build_cltft  # noqa: PLC0415
                served = inf.forecaster()
                model = build_cltft(inf._train_ds, cl_method=getattr(served, "cl_method", "naive"))
                model.load_state_dict(torch.load(job / "result.pt", map_location="cpu",
                                                 weights_only=True))
                model.to(DEVICE).eval()
                return inf.swap(forecaster=model, source=self.strategy, reason=rec.reason,
                                sim_date=rec.sim_date)
            from stable_baselines3 import PPO  # noqa: PLC0415
            pricer = PPO.load(job / "result.zip", device=inf.pricer().device)
            return inf.swap(pricer=pricer, source=self.strategy, reason=rec.reason,
                            sim_date=rec.sim_date)
        finally:
            shutil.rmtree(job, ignore_errors=True)

    def _run(self, rec: RetrainRecord) -> None:
        try:
            if self._use_child():
                version = self._run_in_child(rec)
                rec.ok = True
                rec.generation = version.generation
                return
            ctrl = self.controller()
            origin = rec.sim_date

            if rec.kind == "forecasting":
                # Fine-tune a private copy. The controller trains its forecaster
                # in place, and after a swap that object IS the served one - the
                # daily check and /forecast would put it in eval mode mid-fit
                # ("cudnn RNN backward can only be called in training mode",
                # or Lightning's "self.log twice"). Copy every time, not once.
                ctrl.forecaster = self._clone_forecaster(self.inference.forecaster(),
                                                         ctrl.train_ds)
                ctrl.retrain_forecaster(origin, rec.reason)
                version = self.inference.swap(
                    forecaster=ctrl.forecaster, source=self.strategy, reason=rec.reason,
                    sim_date=rec.sim_date)
            else:
                # Same reason, and a second symptom: after one SDFT fit the
                # policy holds a graph-attached tensor, so the next retrain's
                # teacher deepcopy (RLTeacherStore.store) raised "Only Tensors
                # created explicitly by the user ... support the deepcopy
                # protocol" - every pricer retrain after a node's first failed.
                ctrl.pricer = self._clone_pricer(self.inference.pricer())
                ctrl.retrain_pricer(origin, rec.reason)
                version = self.inference.swap(
                    pricer=ctrl.pricer, source=self.strategy, reason=rec.reason,
                    sim_date=rec.sim_date)

            rec.ok = True
            rec.generation = version.generation
        except Exception as exc:                      # noqa: BLE001
            # A failed retrain must not take the node down: it keeps serving the
            # previous model, and the failure is recorded for the run log.
            rec.ok = False
            rec.error = f"{type(exc).__name__}: {exc}"
            rec.traceback = traceback.format_exc()
        finally:
            rec.finished_at = time.time()
            if self._on_complete is not None:
                try:
                    self._on_complete(rec)
                except Exception:                     # noqa: BLE001, S110
                    pass                              # a bad callback must not mask the result
            with self._lock:
                nxt = next(iter(self._deferred.values()), None)
                if nxt is not None:
                    del self._deferred[nxt.kind]
                    self._start(nxt)
                else:
                    self._kind = None

    def wait(self, timeout: Optional[float] = None) -> bool:
        """Block until any in-flight retrain - and any deferred one it hands
        over to - finishes. Returns True if idle."""
        deadline = None if timeout is None else time.time() + timeout
        while True:
            t = self._thread
            if t is None:
                return True
            t.join(None if deadline is None else max(deadline - time.time(), 0))
            if t.is_alive():
                return False
            if self._thread is t:          # nothing was handed over
                return True

    # ── Views ───────────────────────────────────────────────────────────────

    def status(self) -> Dict:
        done = [r for r in self.history if r.ok]
        return {
            "node": self.node,
            "strategy": self.strategy,
            "busy": self.busy,
            "n_triggered": len(self.history),
            "n_completed": len(done),
            "n_skipped": sum(1 for r in self.history if r.skipped),
            "n_failed": sum(1 for r in self.history if not r.ok and not r.skipped
                            and r.finished_at is not None),
            "total_retrain_s": sum(r.duration_s or 0.0 for r in done),
            "generation": self.inference.version.generation,
        }

    def recent(self, limit: int = 20) -> List[Dict]:
        return [r.as_dict() for r in self.history[-limit:]]
