"""The live clock treats a model update as overnight work: the next trading day
does not open while a channel is retraining, so the lag is one day on any
machine at any speed. Nodes are faked; no services run."""

import threading
import time

import pytest

from edge_system.config import SYSTEM_CONFIG
from edge_system.sim import service


@pytest.fixture
def run(monkeypatch):
    sim = dict(SYSTEM_CONFIG["sim"])
    monkeypatch.setitem(SYSTEM_CONFIG, "sim", sim)
    r = service.Runner()
    r.busy = []
    monkeypatch.setattr(r, "_retraining", lambda: list(r.busy))
    return r


def test_no_retrain_no_hold(run):
    assert run._hold() is False
    assert run.state(brief=True)["waiting_for"] == []


def test_holds_while_a_channel_retrains_then_lets_go(run):
    run.busy = ["web"]
    assert run._hold() is True
    assert run.state(brief=True)["waiting_for"] == ["web"]
    assert run._hold() is True
    run.busy = []
    assert run._hold() is False
    assert run.state(brief=True)["waiting_for"] == []
    kinds = [e["kind"] for e in run.log]
    assert kinds.count("retrain") == 1 and kinds.count("model") == 1


def test_switched_off_never_holds(run):
    SYSTEM_CONFIG["sim"]["hold_for_retrain"] = False
    run.busy = ["web"]
    assert run._hold() is False


def test_a_stuck_fit_releases_the_clock_after_the_cap(run):
    SYSTEM_CONFIG["sim"]["retrain_hold_max_s"] = 0.05
    run.busy = ["web"]
    assert run._hold() is True
    time.sleep(0.1)
    assert run._hold() is False          # gave up: trade on the old model
    assert run._hold() is False          # and do not hold again for this fit
    assert any(e["kind"] == "warn" for e in run.log)
    run.busy = []
    run._hold()
    run.busy = ["web"]                   # a later fit holds the clock again
    assert run._hold() is True


def test_next_day_pressed_during_a_hold_trades_once_the_model_lands(run, monkeypatch):
    traded = []
    monkeypatch.setattr(run, "trade", lambda: traded.append(time.time()))
    monkeypatch.setattr(run, "_watch_nodes", lambda d: None, raising=False)
    run.mode, run.busy = "paused", ["pos"]
    with run.lock:
        run._step = True
    run.wake.set()
    threading.Thread(target=run._loop, daemon=True).start()
    time.sleep(1.2)
    assert traded == [] and run.waiting_for == ["pos"]
    run.busy = []
    deadline = time.time() + 3
    while not traded and time.time() < deadline:
        time.sleep(0.05)
    assert len(traded) == 1
    time.sleep(0.7)
    assert len(traded) == 1              # one click, one day
