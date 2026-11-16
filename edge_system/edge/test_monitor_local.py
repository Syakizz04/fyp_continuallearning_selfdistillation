"""The node's drift monitor: the fixed RL floor is unchanged, and the rolling
detector warms up, calibrates, ignores noise and catches a slump."""

import random

import pytest

from edge_system.config import SYSTEM_CONFIG
from edge_system.edge.monitor_local import LocalDriftMonitor

CAL = {"forecasting": {"mase_mu": 1.0, "mase_sigma": 0.1}}


@pytest.fixture
def models(monkeypatch):
    m = dict(SYSTEM_CONFIG["models"])
    monkeypatch.setitem(SYSTEM_CONFIG, "models", m)
    return m


def feed(mon, days, start=0):
    fired = []
    for t, (rev, ref) in enumerate(days, start=start):
        fired += mon.update(tick=t, sim_date=f"d{t}", profit_index=rev / ref,
                            revenue=rev, reference=ref)
    return fired


def noisy(n, level=0.85, ref=100.0, sd=0.15, seed=0):
    rng = random.Random(seed)
    return [(max(ref * level * (1 + rng.gauss(0, sd)), 0.0), ref) for _ in range(n)]


def test_fixed_is_default_and_unchanged(models):
    models.pop("rl_detector", None)
    mon = LocalDriftMonitor("pos", CAL)
    assert mon.rl_mode == "fixed" and mon.rl_threshold == 1.0
    fired = [mon.update(tick=t, sim_date="d", profit_index=0.9) for t in range(2)]
    assert [len(f) for f in fired] == [0, 1]


def test_explicit_floor_pins_fixed(models):
    models["rl_detector"] = "rolling"
    mon = LocalDriftMonitor("pos", CAL, rl_floor=0.5)
    assert mon.rl_mode == "fixed" and mon.rl_threshold == 0.5


def test_rolling_warms_up_then_calibrates(models):
    models.update(rl_detector="rolling", rl_window_days=7, rl_warmup_days=28, rl_k_sigma=2.5)
    mon = LocalDriftMonitor("pos", CAL)
    assert feed(mon, noisy(27)) == []
    st = mon.status()
    assert st["rl_threshold"] is None and st["rl_warmup_left"] == 1
    feed(mon, noisy(1, seed=1), start=27)
    cal = mon.rl_calibration
    assert cal is not None and cal["mu"] == pytest.approx(0.85, abs=0.06)
    # Independent-day sigma of a 7-day mean: 0.85 * 0.15 / sqrt(7) ~ 0.048.
    assert cal["sigma"] == pytest.approx(0.048, rel=0.4)
    assert mon.rl_threshold == pytest.approx(cal["mu"] - 2.5 * cal["sigma"])
    # The stream carries the rolling value (what the threshold applies to).
    assert mon.stream[-1]["profit_index"] != mon.stream[-1]["profit_index_daily"]


def test_rolling_quiet_on_noise_and_fires_on_slump(models):
    models.update(rl_detector="rolling", rl_window_days=7, rl_warmup_days=28, rl_k_sigma=2.5)
    mon = LocalDriftMonitor("pos", CAL)
    feed(mon, noisy(28, seed=2))
    assert feed(mon, noisy(200, seed=3), start=28) == []
    slump = [(r * 0.6, f) for r, f in noisy(10, seed=4)]
    assert feed(mon, slump, start=228)


def test_recalibrate_restarts_warmup(models):
    models.update(rl_detector="rolling", rl_window_days=7, rl_warmup_days=14)
    mon = LocalDriftMonitor("pos", CAL)
    feed(mon, noisy(14))
    assert mon.rl_threshold is not None
    mon.recalibrate_rl()
    assert mon.rl_threshold is None and mon.status()["rl_warmup_left"] == 14
    # A slump during a fresh warm-up cannot fire: there is no floor yet.
    assert feed(mon, [(r * 0.3, f) for r, f in noisy(13)], start=14) == []


# ── Forecaster: the "realised" signal calibrates its own threshold ──────────

def test_fc_dataset_is_default_and_unchanged(models):
    models.pop("fc_signal", None)
    mon = LocalDriftMonitor("pos", CAL)
    assert mon.fc_mode == "dataset"
    assert mon.fc_threshold == pytest.approx(1.0 + 2.0 * 0.1)


def test_fc_realised_warms_up_fires_and_rewarms(models):
    models.update(fc_signal="realised", fc_warmup_days=28, fc_live_k=3.0,
                  fc_live_consecutive=3)
    mon = LocalDriftMonitor("pos", CAL)
    assert mon.fc_threshold is None and mon.status()["fc_warmup_left"] == 28
    rng = random.Random(0)
    normal = [1.4 + rng.gauss(0, 0.05) for _ in range(28)]
    fired = [e for t, v in enumerate(normal)
             for e in mon.update(tick=t, sim_date=f"d{t}", mase=v)]
    assert fired == [] and mon.fc_calibration is not None
    cal = mon.fc_calibration
    assert mon.fc_threshold == pytest.approx(cal["mu"] + 3.0 * cal["sigma"])
    assert mon.status()["fc_warmup_left"] == 0

    # Two breaches are not enough at 3 running; the third fires.
    high = [mon.update(tick=28 + i, sim_date="s", mase=3.0) for i in range(3)]
    assert [len(f) for f in high] == [0, 0, 1]
    assert high[2][0].kind == "forecasting"

    # A new forecaster starts a fresh warm-up: nothing fires while it runs.
    mon.recalibrate_fc()
    assert mon.fc_threshold is None and mon.status()["fc_warmup_left"] == 28
    assert all(mon.update(tick=40 + i, sim_date="w", mase=9.0) == [] for i in range(10))


def test_fc_realised_ignores_nan_during_warmup(models):
    models.update(fc_signal="realised", fc_warmup_days=3)
    mon = LocalDriftMonitor("pos", CAL)
    for v in (1.0, float("nan"), 1.1):
        mon.update(tick=0, sim_date="d", mase=v)
    assert mon.fc_calibration is None and mon.status()["fc_warmup_left"] == 1


def test_fc_unknown_signal_rejected(models):
    models["fc_signal"] = "bogus"
    with pytest.raises(ValueError):
        LocalDriftMonitor("pos", CAL)


def test_each_check_records_the_line_it_was_judged_against(models):
    """The floor is re-learned after every retrain, so the current line is not
    the line an old warning crossed. Each row keeps its own, and every warning
    sits beyond the line on its row - which is what the dashboard draws."""
    models.update(rl_detector="rolling", rl_window_days=7, rl_warmup_days=28, rl_k_sigma=2.5)
    mon = LocalDriftMonitor("pos", CAL)
    feed(mon, noisy(28))
    # Day 28 completes the warm-up and is already judged against the new floor.
    assert all(r["rl_threshold"] is None for r in mon.stream[:27])
    assert mon.stream[27]["rl_threshold"] == mon.rl_threshold
    first = mon.rl_threshold
    fired = feed(mon, noisy(20, level=0.4, seed=2), start=28)
    assert fired
    mon.recalibrate_rl()
    feed(mon, noisy(28, level=0.4, seed=3), start=48)
    fired += feed(mon, noisy(20, level=0.15, seed=4), start=76)
    assert mon.rl_threshold != first
    by_tick = {r["tick"]: r for r in mon.stream}
    for e in fired:
        row = by_tick[e.tick]
        assert row["rl_threshold"] == e.threshold
        assert row["profit_index"] < row["rl_threshold"]


# ── re-learning after a swap (models.relearn_min_days) ─────────────────────

def warmed_rl(models, relearn=7):
    models.update(rl_detector="rolling", rl_window_days=7, rl_warmup_days=28,
                  rl_k_sigma=2.5, relearn_min_days=relearn)
    mon = LocalDriftMonitor("pos", CAL)
    feed(mon, noisy(28))
    return mon


def test_first_calibration_still_waits_the_full_warm_up(models):
    """No previous spread exists on a fresh run, so no provisional line."""
    models.update(rl_detector="rolling", rl_window_days=7, rl_warmup_days=28,
                  rl_k_sigma=2.5, relearn_min_days=7)
    mon = LocalDriftMonitor("pos", CAL)
    feed(mon, noisy(27))
    assert mon.rl_threshold is None and mon.rl_provisional is None


def test_after_a_swap_a_provisional_floor_goes_up_after_one_window(models):
    mon = warmed_rl(models)
    sigma = mon.rl_calibration["sigma"]
    mon.recalibrate_rl()
    feed(mon, noisy(6, level=0.6, sd=0.02, seed=5), start=28)
    assert mon.rl_threshold is None                      # window still filling
    feed(mon, noisy(1, level=0.6, sd=0.02, seed=6), start=34)
    prov = mon.rl_provisional
    assert prov is not None and prov["days"] == 7 and prov["sigma"] == sigma
    # The new model's own level, not the old one's (~0.85).
    assert prov["mu"] == pytest.approx(0.6, abs=0.03)
    assert mon.rl_threshold == pytest.approx(prov["mu"] - 2.5 * sigma)
    assert mon.stream[-1]["rl_threshold"] == mon.rl_threshold
    feed(mon, noisy(21, level=0.6, sd=0.02, seed=7), start=35)
    assert mon.rl_calibration is not None and mon.rl_provisional is None
    assert mon.rl_calibration["warmup_days"] == 28


def test_relearn_zero_keeps_the_original_blind_warm_up(models):
    mon = warmed_rl(models, relearn=0)
    mon.recalibrate_rl()
    feed(mon, noisy(27, level=0.6, seed=8), start=28)
    assert mon.rl_threshold is None
    assert all(r["rl_threshold"] is None for r in mon.stream[28:])


def test_a_provisional_floor_catches_a_slump_in_the_new_models_first_month(models):
    mon = warmed_rl(models)
    mon.recalibrate_rl()
    feed(mon, noisy(10, level=0.85, seed=9), start=28)
    fired = feed(mon, noisy(8, level=0.3, sd=0.02, seed=10), start=38)
    assert fired and mon.rl_calibration is None          # before the old rule could look
    by_tick = {r["tick"]: r for r in mon.stream}
    for e in fired:
        assert by_tick[e.tick]["rl_threshold"] == e.threshold > by_tick[e.tick]["profit_index"]


def warmed_fc(models, relearn=7):
    models.update(fc_signal="realised", fc_warmup_days=28, fc_live_k=3.0,
                  fc_live_consecutive=3, relearn_min_days=relearn)
    mon = LocalDriftMonitor("pos", CAL)
    rng = random.Random(1)
    for t in range(28):
        mon.update(tick=t, sim_date=f"d{t}", mase=1.3 + rng.gauss(0, 0.05))
    return mon, rng


def test_forecast_line_after_a_swap_uses_the_new_level_and_the_old_spread(models):
    mon, rng = warmed_fc(models)
    sigma = mon.fc_calibration["sigma"]
    mon.recalibrate_fc()
    for t in range(28, 34):
        mon.update(tick=t, sim_date=f"d{t}", mase=1.0 + rng.gauss(0, 0.02))
    assert mon.fc_threshold is None
    mon.update(tick=34, sim_date="d34", mase=1.0)
    prov = mon.fc_provisional
    assert prov["days"] == 7 and prov["sigma"] == sigma
    assert mon.fc_threshold == pytest.approx(prov["mu"] + 3.0 * sigma)
    # Judged against the line recorded on its own row, then it refines it.
    fired = []
    for t in range(35, 40):
        fired += mon.update(tick=t, sim_date=f"d{t}", mase=3.0)
    by_tick = {r["tick"]: r for r in mon.stream}
    assert fired
    for e in fired:
        assert by_tick[e.tick]["fc_threshold"] == e.threshold < e.value


def test_refining_the_line_keeps_a_running_breach_streak(models):
    mon, rng = warmed_fc(models)
    mon.recalibrate_fc()
    for t in range(28, 35):
        mon.update(tick=t, sim_date=f"d{t}", mase=1.0)
    det = mon._fc
    mon.update(tick=35, sim_date="d35", mase=5.0)
    mon.update(tick=36, sim_date="d36", mase=5.0)
    assert mon._fc is det and det.run == 2               # same detector, streak intact
