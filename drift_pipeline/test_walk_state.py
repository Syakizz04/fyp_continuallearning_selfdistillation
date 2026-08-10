"""
Mid-walk resume must be indistinguishable from never having been interrupted.

That is the only property worth asserting here, and it is stronger than
"resume runs without crashing": a resume that quietly reset the detector
streaks or dropped the accumulated stream would still complete, and would still
produce a plausible-looking results file with the wrong number of triggers in
it. So the test runs the same short walk twice — once straight through, once
aborted and resumed — and compares them row by row.

`frozen` is the arm under test because it is the one whose walk is exactly
reproducible: it never retrains, so there is no cuDNN or RNG nondeterminism to
mask a real discrepancy. The retraining arms exercise more of `walk_state`
(model weights, the SDFT teacher, the replay buffer) but can only be checked for
equivalence approximately, which would make this test weaker rather than
stronger. Those paths are covered by the restore-side assertions below instead.

Skipped when the M5 data or the base checkpoint is absent — both are gitignored,
so a clean checkout cannot run this.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from .core_pipeline import CONFIG

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "processed_m5_v4"
CKPT = ROOT / "outputs" / "drift" / "checkpoints" / "base_v4"

pytestmark = pytest.mark.skipif(
    not (DATA / "rl_environment.csv").exists() or not (CKPT / "base_tft.ckpt").exists(),
    reason="needs data/processed_m5_v4 + checkpoints/base_v4 (both gitignored)")

N_CHECKS = 6
ABORT_AT = 4


class _Abort(Exception):
    pass


@pytest.fixture(scope="module")
def walk_env(tmp_path_factory):
    from .core_pipeline import prepare_drift_data
    from . import monitor as mon

    CONFIG["paths"]["demand_csv"] = str(DATA / "demand_forecasting.csv")
    CONFIG["paths"]["rl_csv"] = str(DATA / "rl_environment.csv")
    data = prepare_drift_data()
    data["checks"] = list(data["checks"])[:N_CHECKS]

    CONFIG["paths"]["results"] = str(tmp_path_factory.mktemp("walk_state"))
    # Calibration falls back to the shared results dir when the checkpoint has
    # none of its own; base_v4 carries one, so this stays self-contained.
    base = mon.load_base({
        "tft": str(CKPT / "base_tft.ckpt"),
        "dataset": str(CKPT / "base_tft_dataset.pkl"),
        "ppo": str(CKPT / "base_ppo.zip"),
        "meta": str(CKPT / "base_meta.json"),
        "calibration": str(CKPT / "calibration.json"),
    })
    return data, base


def _walk(data, base, ctrl, **kw):
    from . import monitor as mon
    return mon.walk_forward(data, base, arm=ctrl.strategy,
                            forecaster_provider=ctrl.forecaster_provider,
                            pricer_provider=ctrl.pricer_provider, **kw)


def test_resumed_walk_matches_uninterrupted(walk_env):
    from . import retrain as rt
    from . import walk_state as ws

    data, base = walk_env
    arm = "frozen"
    ws.clear(arm)

    straight = _walk(data, base, rt.RetrainController(arm, base, data))

    ctrl_b = rt.RetrainController(arm, base, data)

    def on_ckpt(next_idx, res, fc_det, rl_det):
        ws.save(ctrl_b, res, fc_det, rl_det, next_idx)
        if next_idx == ABORT_AT:
            raise _Abort()

    with pytest.raises(_Abort):
        _walk(data, base, ctrl_b, on_checkpoint=on_ckpt, checkpoint_every=2)

    assert ws.exists(arm)

    ctrl_c = rt.RetrainController(arm, base, data)
    saved = ws.load(arm)
    assert saved is not None
    resume_state = ws.restore(ctrl_c, saved)
    assert resume_state["next_idx"] == ABORT_AT

    resumed = _walk(data, base, ctrl_c, resume_state=resume_state)

    assert len(resumed.stream) == len(straight.stream) == N_CHECKS
    # Trigger counts are the assertion that actually catches a reset detector
    # streak: the stream can look right while the triggers derived from it do not.
    assert resumed.n_fc_triggers == straight.n_fc_triggers
    assert resumed.n_rl_triggers == straight.n_rl_triggers

    for i, (a, c) in enumerate(zip(straight.stream, resumed.stream)):
        assert a["date"] == c["date"], f"row {i} date"
        for key in ("mase", "smape", "cumulative_profit", "fc_trigger", "rl_trigger"):
            av, cv = a.get(key), c.get(key)
            if isinstance(av, float) and isinstance(cv, float) and av != av:
                assert cv != cv, f"row {i} [{key}]: NaN vs {cv}"
            else:
                assert av == cv, f"row {i} [{key}]: {av} vs {cv}"

    ws.clear(arm)
    assert not ws.exists(arm)
