"""
Tests for the arm set itself - which arms exist and how each one is wired.

The failure these guard against is silent. An arm that is configured but wired to
no trigger does not crash: it runs the full walk, never retrains, and produces
results identical to `frozen`. That looks like a finding ("the CL method made no
difference") rather than like a bug, and it costs ~90 minutes of GPU to produce.
`naive` was in exactly that state - the retrain dispatch handled it, but
`run_arm`'s trigger gate listed only ewc/replay/sdft, so it could never fire.
"""

from __future__ import annotations

import pytest

from drift_pipeline.core_pipeline import CONFIG
from drift_pipeline.retrain import DRIFT_TRIGGERED

#: Arms that legitimately do not take a drift trigger, and why.
NOT_DRIFT_TRIGGERED = {
    "frozen": "never retrains at all - it is the metric anchor",
    "periodic": "retrains on a fixed schedule via on_check_periodic",
}


def configured_arms():
    return [*CONFIG["baselines"], *CONFIG["strategies"]]


def test_every_configured_arm_is_wired_to_something():
    """No configured arm may be silently inert."""
    for arm in configured_arms():
        assert arm in DRIFT_TRIGGERED or arm in NOT_DRIFT_TRIGGERED, (
            f"arm {arm!r} is configured but takes neither a drift trigger nor a "
            f"schedule - it would run a full walk, never retrain, and return "
            f"results indistinguishable from 'frozen'")


def test_naive_control_is_present_and_drift_triggered():
    """The control that makes the headline comparison attributable.

    `naive` shares the trigger with the CL arms and differs only in the mechanism.
    Without it, an arm beating `frozen` cannot say whether its CL mechanism helped
    or whether retraining at the right moment did.
    """
    assert "naive" in configured_arms()
    assert "naive" in DRIFT_TRIGGERED


def test_frozen_anchor_is_present_and_never_triggered():
    assert "frozen" in configured_arms()
    assert "frozen" not in DRIFT_TRIGGERED


def test_naive_and_the_cl_arms_share_the_trigger():
    """The controls only isolate the mechanism if the trigger is held fixed."""
    for arm in ("replay", "sdft"):
        if arm in configured_arms():
            assert arm in DRIFT_TRIGGERED, (
                f"{arm} must fire on the same trigger as naive, or the two "
                f"differ in more than the CL mechanism and neither explains the other")


@pytest.mark.parametrize("arm", ["naive", "periodic"])
def test_plain_finetune_arms_fall_through_to_naive_dispatch(arm):
    """Both plain-fine-tune arms must reach the same training path.

    They are meant to differ ONLY in when they fire. If one picked up a CL
    method by accident the pair would stop being a matched comparison.
    """
    import inspect

    from drift_pipeline import retrain

    src = inspect.getsource(retrain.RetrainController.retrain_forecaster)
    # The dispatch names each CL arm explicitly and routes everything else to
    # plain fine-tuning; neither plain arm may appear in that branch list.
    for branch in ("ewc", "replay", "sdft"):
        assert f'strat == "{branch}"' in src, "dispatch shape changed"
    assert f'strat == "{arm}"' not in src, (
        f"{arm} has acquired its own branch in the forecasting retrain dispatch; "
        f"it is supposed to fall through to plain fine-tuning")
