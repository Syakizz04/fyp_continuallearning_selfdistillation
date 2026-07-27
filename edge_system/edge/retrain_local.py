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

Only one retrain runs at a time per node. A trigger arriving mid-retrain is
recorded and dropped, not queued: by the time the in-flight retrain lands, the
condition that fired the second trigger has usually been addressed by it, and
queueing would produce the "retrain churn" that EWC already showed in the first
real run.
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
            "skipped": self.skipped,
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
            if self.busy:
                # Dropped, not queued - see the module docstring.
                rec.skipped = True
                rec.finished_at = time.time()
                rec.error = "a retrain was already in flight"
                self.history.append(rec)
                return rec

            self._thread = threading.Thread(
                target=self._run, args=(rec,), name=f"retrain-{self.node}", daemon=True,
            )
            self.history.append(rec)
            self._thread.start()

        if blocking:
            self._thread.join()
        return rec

    def _run(self, rec: RetrainRecord) -> None:
        try:
            ctrl = self.controller()
            origin = rec.sim_date

            if rec.kind == "forecasting":
                ctrl.retrain_forecaster(origin, rec.reason)
                version = self.inference.swap(
                    forecaster=ctrl.forecaster, source=self.strategy, reason=rec.reason)
            else:
                ctrl.retrain_pricer(origin, rec.reason)
                version = self.inference.swap(
                    pricer=ctrl.pricer, source=self.strategy, reason=rec.reason)

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

    def wait(self, timeout: Optional[float] = None) -> bool:
        """Block until any in-flight retrain finishes. Returns True if idle."""
        t = self._thread
        if t is None:
            return True
        t.join(timeout)
        return not t.is_alive()

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
