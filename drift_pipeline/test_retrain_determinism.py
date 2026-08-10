"""
A retrain must depend on WHAT it is, not on what ran before it.

This is the regression guard for a confound that silently invalidated a
cross-run comparison. The forecaster and the pricer drew from one global RNG
stream, so a forecaster fine-tune's shuffle order and dropout masks depended on
how many PPO retrains had already happened. Two runs with a byte-identical
forecaster, dataset, calibration and retrain date produced post-retrain walk
MASE of 0.867 and 1.373 - because one had three pricer retrains before the
forecaster's first retrain and the other had one.

Worse, it is systematic between ARMS rather than merely noisy: E2's arms fired
11 / 6 / 7 pricer retrains, so every arm's forecaster drew from a different RNG
state by construction. No amount of averaging removes a confound that tracks the
thing being compared.

`RetrainController._seed_retrain` keys each fit on (seed, arm, date, model), so
the tests below assert the property that actually matters - burning arbitrary
randomness beforehand must not change the fitted weights - rather than merely
that two identical calls agree, which would pass even with the bug present.

Skipped when the M5 data or the base checkpoint is absent; both are gitignored.
"""

from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import pytest
import torch

from .core_pipeline import CONFIG, seed_for

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "processed_m5_v4"
CKPT = ROOT / "outputs" / "drift" / "checkpoints" / "base_v4"

pytestmark = pytest.mark.skipif(
    not (DATA / "rl_environment.csv").exists() or not (CKPT / "base_tft.ckpt").exists(),
    reason="needs data/processed_m5_v4 + checkpoints/base_v4 (both gitignored)")

ORIGIN = "2013-12-10"          # the date the sdft arm's first retrain actually fired


def test_seed_for_is_stable_across_processes():
    """sha256, not the builtin hash(), which is per-process salted for str."""
    assert seed_for(42, "sdft", "2013-12-10", "fc") == seed_for(42, "sdft", "2013-12-10", "fc")
    assert seed_for(42, "sdft", "2013-12-10", "fc") != seed_for(42, "sdft", "2013-12-10", "rl")
    assert seed_for(42, "sdft", "2013-12-10", "fc") != seed_for(43, "sdft", "2013-12-10", "fc")
    assert seed_for(42, "sdft", "2013-12-10", "fc") != seed_for(42, "replay", "2013-12-10", "fc")
    # Known-value pin: a change here silently reshuffles every seeded run.
    assert seed_for(42, "sdft", "2013-12-10", "fc") == 130706116


@pytest.fixture(scope="module")
def env():
    from .core_pipeline import prepare_drift_data
    from . import monitor as mon

    CONFIG["paths"]["demand_csv"] = str(DATA / "demand_forecasting.csv")
    CONFIG["paths"]["rl_csv"] = str(DATA / "rl_environment.csv")
    CONFIG["seed"] = 42
    CONFIG["retrain"]["retrain_epochs"] = 1        # keep the test minutes, not hours
    data = prepare_drift_data()
    base = mon.load_base({
        "tft": str(CKPT / "base_tft.ckpt"),
        "dataset": str(CKPT / "base_tft_dataset.pkl"),
        "ppo": str(CKPT / "base_ppo.zip"),
        "meta": str(CKPT / "base_meta.json"),
        "calibration": str(CKPT / "calibration.json"),
    })
    # Snapshotted BEFORE any fit: RetrainController takes base["forecaster"] by
    # reference and trains it in place, so without a pristine copy the second
    # call would start from the first call's output.
    pristine = {k: v.detach().cpu().clone()
                for k, v in base["forecaster"].state_dict().items()}
    return data, base, pristine


def _fc_weights_after_retrain(data, base, pristine, burn: int):
    """Fit once, having first consumed `burn` draws from the global RNG.

    `burn` stands in for the pricer retrains that precede a forecaster retrain in
    a real walk. If seeding is correct the fitted weights must not notice.
    """
    from . import retrain as rt

    ctrl = rt.RetrainController("sdft", base, data)
    ctrl.forecaster.teacher = None            # drop any teacher a prior fit left
    ctrl.forecaster.load_state_dict(pristine)

    for _ in range(burn):
        random.random(); np.random.rand(); torch.rand(1)

    ctrl.retrain_forecaster(ORIGIN, reason="drift")
    return {k: v.detach().cpu().clone()
            for k, v in ctrl.forecaster.state_dict().items()
            if not k.startswith("teacher.")}


def _max_abs_diff(a, b) -> float:
    return max((a[k] - b[k]).abs().max().item() for k in a)


def test_retrain_is_independent_of_prior_rng(env):
    """Burning RNG beforehand must perturb a fit no more than a re-run does.

    Not bit-equality: a GPU fit is not bit-reproducible even against itself
    (measured ~2e-6 across 533 tensors, with cudnn.deterministic already True -
    cuBLAS reductions and atomics). Demanding exactness would fail forever and
    say nothing. The meaningful property is RELATIVE: if prior RNG consumption
    still leaked in, `burn` would move the weights by far more than re-running
    the identical fit does, because it changes shuffle order and dropout masks
    rather than just float rounding.
    """
    data, base, pristine = env
    a = _fc_weights_after_retrain(data, base, pristine, burn=0)
    b = _fc_weights_after_retrain(data, base, pristine, burn=0)
    c = _fc_weights_after_retrain(data, base, pristine, burn=5000)

    assert set(a) == set(b) == set(c)
    rerun = _max_abs_diff(a, b)
    burned = _max_abs_diff(a, c)

    assert burned <= max(rerun * 10.0, 1e-4), (
        f"burning RNG moved the weights by {burned:.3e} while a plain re-run "
        f"moves them {rerun:.3e} - the retrain is still inheriting the global "
        f"stream, so arms with different pricer retrain counts are not comparable")
