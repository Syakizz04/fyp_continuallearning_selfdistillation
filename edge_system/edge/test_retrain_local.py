"""Local retrain scheduling: a same-model trigger mid-retrain is dropped, an
other-model trigger is deferred and runs when the in-flight one lands; and the
forecaster is fine-tuned as a copy, never the served object."""

import threading
import time
from types import SimpleNamespace

from edge_system.edge.retrain_local import LocalRetrainer


class FakeController:
    def __init__(self):
        self.gate = threading.Event()
        self.order = []
        self.forecaster = self.pricer = None
        self.train_ds = None

    def retrain_forecaster(self, origin, reason):
        self.order.append(("fc", origin))

    def retrain_pricer(self, origin, reason):
        self.gate.wait(5)
        self.order.append(("rl", origin))


class FakeInference:
    def __init__(self):
        self.served = object()
        self.served_pricer = object()
        self.n = 0

    def forecaster(self):
        return self.served

    def pricer(self):
        return self.served_pricer

    def swap(self, **kw):
        self.n += 1
        return SimpleNamespace(generation=self.n)


def ev(kind, tick):
    return SimpleNamespace(kind=kind, tick=tick, sim_date=f"d{tick}", value=2.0, threshold=1.0)


def node(monkeypatch):
    inf, ctrl = FakeInference(), FakeController()
    r = LocalRetrainer("pos", inf)
    r._controller = ctrl
    clones = []
    monkeypatch.setattr(LocalRetrainer, "_clone_forecaster",
                        staticmethod(lambda served, ds: clones.append(served) or object()))
    monkeypatch.setattr(LocalRetrainer, "_clone_pricer",
                        staticmethod(lambda served: clones.append(served) or object()))
    return r, ctrl, inf, clones


def test_same_kind_dropped_other_kind_deferred(monkeypatch):
    r, ctrl, inf, clones = node(monkeypatch)
    first = r.on_drift(ev("rl", 1))
    again = r.on_drift(ev("rl", 2))
    fc1 = r.on_drift(ev("forecasting", 3))
    fc2 = r.on_drift(ev("forecasting", 4))
    assert again.skipped and "in flight" in again.error
    assert fc1.skipped and "superseded" in fc1.error and not fc1.deferred
    assert fc2.deferred and not fc2.skipped
    time.sleep(0.05)
    assert ctrl.order == []                      # nothing overlaps the pricer
    ctrl.gate.set()
    assert r.wait(10)
    assert ctrl.order == [("rl", "d1"), ("fc", "d4")]
    assert first.ok and fc2.ok and not fc2.deferred and inf.n == 2
    assert not r.busy


def test_forecaster_retrains_a_copy_of_the_served_model(monkeypatch):
    r, ctrl, inf, clones = node(monkeypatch)
    rec = r.on_drift(ev("forecasting", 1), blocking=True)
    assert rec.ok and clones == [inf.served]
    assert ctrl.forecaster is not inf.served


def test_pricer_retrains_a_copy_of_the_served_model(monkeypatch):
    # Training the served PPO in place left a graph-attached tensor on its
    # policy, and the NEXT retrain's SDFT teacher deepcopy failed on it.
    r, ctrl, inf, clones = node(monkeypatch)
    ctrl.gate.set()
    rec = r.on_drift(ev("rl", 1), blocking=True)
    assert rec.ok and clones == [inf.served_pricer]
    assert ctrl.pricer is not inf.served_pricer


def test_child_process_only_when_asked_and_only_for_stateless_strategies(monkeypatch):
    from edge_system.config import SYSTEM_CONFIG
    models = dict(SYSTEM_CONFIG["models"])
    monkeypatch.setitem(SYSTEM_CONFIG, "models", models)
    inf = FakeInference()
    models.pop("retrain_process", None)
    assert not LocalRetrainer("pos", inf, "sdft")._use_child()      # default: in-process
    models["retrain_process"] = "subprocess"
    assert LocalRetrainer("pos", inf, "sdft")._use_child()
    # Replay / EWC keep buffers in the controller across retrains.
    assert not LocalRetrainer("pos", inf, "replay")._use_child()
    assert not LocalRetrainer("pos", inf, "ewc")._use_child()


def test_child_failure_is_a_failed_record_not_a_dead_node(monkeypatch):
    r, ctrl, inf, clones = node(monkeypatch)
    monkeypatch.setattr(LocalRetrainer, "_use_child", lambda self: True)

    def boom(self, rec):
        raise RuntimeError("retrain child exited 1: CUDA out of memory")
    monkeypatch.setattr(LocalRetrainer, "_run_in_child", boom)
    rec = r.on_drift(ev("forecasting", 1), blocking=True)
    assert not rec.ok and "exited 1" in rec.error and inf.n == 0 and not r.busy
