"""
Tests for the demand-censoring channel.

The properties that matter are structural, not numerical: censoring must be
negative-only, bounded by the demand it hides, reproducible from a seed, and it
must leave the frame's shape untouched so the TFT's consecutive `time_idx`
survives. The scarcity-concentration test is the one that distinguishes this from
"subtract a random number" - the bias has to correlate with the state, or the
treatment is not the mechanism it claims to model.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from drift_pipeline.censoring import (E1_FILL_RATES, CensoringSpec,
                                      apply_censoring, assert_uncensored,
                                      censoring_report)


def make_frames(n_days: int = 120, n_items: int = 4, seed: int = 0):
    """A TFT-shaped frame and its matching RL frame, sharing the merge keys."""
    rng = np.random.default_rng(seed)
    rows_t, rows_r = [], []
    dates = pd.date_range("2013-01-01", periods=n_days, freq="D")
    for i in range(n_items):
        demand = rng.poisson(12.0, size=n_days).astype(float)
        # Tight stock on a contiguous stretch, so scarcity is not uniform.
        on_hand = np.full(n_days, 200.0)
        on_hand[30:60] = 3.0
        unmet = np.maximum(demand - on_hand, 0.0)
        for d, dem, oh, um in zip(dates, demand, on_hand, unmet):
            rows_t.append({"product_id": f"P{i}", "region_id": "CA_1",
                           "date": d, "demand": dem, "sell_price": 5.0})
            rows_r.append({"product_id": f"P{i}", "region_id": "CA_1",
                           "date": d, "realized_demand": dem,
                           "unmet_demand": um, "inventory_level": oh})
    return pd.DataFrame(rows_t), pd.DataFrame(rows_r)


# ── the control arm ───────────────────────────────────────────────────────────

def test_mode_none_is_the_identity():
    tft, rl = make_frames()
    out, rep = apply_censoring(tft, rl, CensoringSpec(mode="none"))
    pd.testing.assert_frame_equal(out, tft)
    assert rep["units_censored"] == 0.0
    assert rep["fill_rate"] == pytest.approx(1.0)


def test_unknown_mode_is_rejected():
    tft, rl = make_frames()
    with pytest.raises(ValueError, match="unknown censoring mode"):
        apply_censoring(tft, rl, CensoringSpec(mode="wishful"))


# ── intrinsic: the dataset's own stockouts ────────────────────────────────────

def test_intrinsic_equals_demand_minus_unmet():
    tft, rl = make_frames()
    out, _ = apply_censoring(tft, rl, CensoringSpec(mode="intrinsic"))
    expected = np.maximum(tft["demand"] - rl["unmet_demand"], 0.0)
    assert np.allclose(out["demand"], expected)


def test_intrinsic_is_the_floor_not_a_treatment():
    """A perfectly synchronised system still loses these units, so a policy arm
    must censor at least as much as intrinsic."""
    tft, rl = make_frames()
    _, floor = apply_censoring(tft, rl, CensoringSpec(mode="intrinsic"))
    _, policy = apply_censoring(
        tft, rl, CensoringSpec(mode="policy", fill_rate=0.70))
    assert policy["units_censored"] >= floor["units_censored"]


# ── policy: the E1 link ───────────────────────────────────────────────────────

@pytest.mark.parametrize("fill", [0.75, 0.717, 0.60, 0.45])
def test_policy_mode_hits_the_target_fill_rate(fill):
    tft, rl = make_frames()
    _, rep = apply_censoring(tft, rl, CensoringSpec(mode="policy", fill_rate=fill))
    assert rep["fill_rate"] == pytest.approx(fill, abs=0.01)
    assert rep["target_reachable"]


def test_target_above_the_intrinsic_floor_is_flagged_not_silently_missed():
    """Censoring only removes, so a fill rate above what the dataset's own
    stockouts already leave cannot be reached. Returning the floor is correct;
    returning it silently would make two sweep cells secretly identical."""
    tft, rl = make_frames()
    _, floor = apply_censoring(tft, rl, CensoringSpec(mode="intrinsic"))
    _, rep = apply_censoring(tft, rl, CensoringSpec(mode="policy", fill_rate=0.99))
    assert rep["fill_rate"] == pytest.approx(floor["fill_rate"])
    assert rep["target_reachable"] is False
    assert rep["fill_rate_requested"] == 0.99


def test_policy_never_invents_demand():
    """Censoring only removes. A fill rate above what intrinsic already leaves
    must not push observed sales back up."""
    tft, rl = make_frames()
    out, _ = apply_censoring(tft, rl, CensoringSpec(mode="policy", fill_rate=0.999))
    assert (out["demand"] <= tft["demand"] + 1e-9).all()


def test_censored_demand_is_never_negative():
    tft, rl = make_frames()
    out, _ = apply_censoring(tft, rl, CensoringSpec(mode="policy", fill_rate=0.30))
    assert (out["demand"] >= -1e-9).all()


def test_spec_from_e1_policy_names():
    tft, rl = make_frames()
    spec = CensoringSpec.for_policy("escrow_quota")
    assert spec.fill_rate == E1_FILL_RATES["escrow_quota"]
    _, rep = apply_censoring(tft, rl, spec)
    assert rep["fill_rate"] == pytest.approx(0.717, abs=0.01)
    with pytest.raises(ValueError, match="unknown policy"):
        CensoringSpec.for_policy("gossip")


def test_escrow_censors_more_than_strong_lock():
    """E1's ordering must survive into E2's treatment, or the two experiments
    are not measuring the same quantity."""
    tft, rl = make_frames()
    _, lock = apply_censoring(tft, rl, CensoringSpec.for_policy("strong_lock"))
    _, escrow = apply_censoring(tft, rl, CensoringSpec.for_policy("escrow_quota"))
    assert escrow["units_censored"] > lock["units_censored"]


# ── the mechanism, not just the magnitude ─────────────────────────────────────

def test_censoring_concentrates_on_scarce_days():
    """Refusals happen where stock is tight. If the bias were uniform it would
    carry no correlation with the features, which is both unrealistic and an
    easier problem than the real one."""
    tft, rl = make_frames()
    out, _ = apply_censoring(
        tft, rl, CensoringSpec(mode="policy", fill_rate=0.70, scarcity_power=1.0))
    lost = tft["demand"].to_numpy() - out["demand"].to_numpy()
    scarce = rl["inventory_level"].to_numpy() < 10.0
    assert lost[scarce].mean() > lost[~scarce].mean()


def test_scarcity_power_zero_spreads_censoring_out():
    tft, rl = make_frames()
    sharp, _ = apply_censoring(
        tft, rl, CensoringSpec(mode="policy", fill_rate=0.70, scarcity_power=2.0))
    flat, _ = apply_censoring(
        tft, rl, CensoringSpec(mode="policy", fill_rate=0.70, scarcity_power=0.0))
    scarce = rl["inventory_level"].to_numpy() < 10.0
    sharp_lost = tft["demand"].to_numpy() - sharp["demand"].to_numpy()
    flat_lost = tft["demand"].to_numpy() - flat["demand"].to_numpy()
    assert sharp_lost[scarce].sum() > flat_lost[scarce].sum()


def test_bias_is_systematically_negative():
    """The defining property: censoring is not noise. A model trained on it
    under-predicts, it does not merely predict less precisely."""
    tft, rl = make_frames()
    _, rep = apply_censoring(tft, rl, CensoringSpec(mode="policy", fill_rate=0.75))
    assert rep["mean_relative_bias"] > 0        # true minus observed, so > 0
    assert rep["units_observed"] < rep["units_true"]


# ── frame integrity: the TFT's requirements ───────────────────────────────────

def test_shape_index_and_other_columns_are_preserved():
    """`time_idx` must stay consecutive per series; a dropped or reordered row
    would be silently discarded later by filter_tft_eval_frame, not raised."""
    tft, rl = make_frames()
    out, _ = apply_censoring(tft, rl, CensoringSpec(mode="policy", fill_rate=0.6))
    assert list(out.columns) == list(tft.columns)
    assert out.index.equals(tft.index)
    assert len(out) == len(tft)
    pd.testing.assert_series_equal(out["sell_price"], tft["sell_price"])
    pd.testing.assert_series_equal(out["product_id"], tft["product_id"])


def test_duplicate_rl_keys_are_rejected_not_silently_joined():
    tft, rl = make_frames()
    with pytest.raises(ValueError, match="duplicate keys"):
        apply_censoring(tft, pd.concat([rl, rl.iloc[:5]]),
                        CensoringSpec(mode="intrinsic"))


def test_missing_rl_columns_are_reported():
    tft, rl = make_frames()
    with pytest.raises(KeyError, match="unmet_demand"):
        apply_censoring(tft, rl.drop(columns=["unmet_demand"]),
                        CensoringSpec(mode="intrinsic"))


def test_determinism():
    tft, rl = make_frames()
    spec = CensoringSpec(mode="policy", fill_rate=0.68, seed=7)
    a, _ = apply_censoring(tft, rl, spec)
    b, _ = apply_censoring(tft, rl, spec)
    assert np.allclose(a["demand"], b["demand"])


# ── the yardstick must stay true ──────────────────────────────────────────────

def test_assert_uncensored_passes_on_a_clean_frame():
    tft, rl = make_frames()
    assert_uncensored(tft, rl)


def test_assert_uncensored_catches_a_censored_evaluation_frame():
    """The failure this exists to prevent: scoring a model against the same bias
    it was trained on, which would look like success."""
    tft, rl = make_frames()
    bad, _ = apply_censoring(tft, rl, CensoringSpec(mode="policy", fill_rate=0.7))
    with pytest.raises(AssertionError, match="evaluation frame is censored"):
        assert_uncensored(bad, rl)


def test_censor_data_dict_adds_keys_without_touching_the_originals():
    """The originals must survive untouched: the monitor scores against them."""
    from drift_pipeline.censoring import censor_data_dict

    tft, rl = make_frames()
    data = {"tft_full": tft, "tft_base": tft.iloc[:200].copy(), "rl_full": rl}
    out = censor_data_dict(data, CensoringSpec(mode="policy", fill_rate=0.7))

    assert "tft_full_censored" in out and "tft_base_censored" in out
    pd.testing.assert_frame_equal(out["tft_full"], tft)          # untouched
    assert out["tft_full_censored"]["demand"].sum() < tft["demand"].sum()
    assert out["censoring"]["fill_rate"] == pytest.approx(0.7, abs=0.01)


def test_censor_data_dict_none_mode_adds_no_censored_frames():
    """An arm that does not opt in must get the uncensored control, not a
    silently corrupted frame."""
    from drift_pipeline.censoring import censor_data_dict

    tft, rl = make_frames()
    out = censor_data_dict(
        {"tft_full": tft, "tft_base": tft, "rl_full": rl},
        CensoringSpec(mode="none"))
    assert "tft_full_censored" not in out
    assert out["censoring"]["units_censored"] == 0.0


def test_controller_trains_on_censored_and_falls_back_when_absent():
    """The wiring: `_recent_tft` must prefer the censored frame, and must fall
    back to the true one so a run without censoring is the control arm."""
    from drift_pipeline.censoring import censor_data_dict
    from drift_pipeline.retrain import RetrainController

    tft, rl = make_frames(n_days=400)
    tft = tft.rename(columns={})              # keep the shape the slicer expects
    data = {"tft_full": tft, "tft_base": tft, "rl_full": rl}
    base = {"train_ds": None, "forecaster": None, "pricer": None}

    plain = RetrainController("sdft", base, data)
    censored = RetrainController("sdft", base,
                                 censor_data_dict(data, CensoringSpec(
                                     mode="policy", fill_rate=0.7)))

    origin = tft["date"].max()
    a = plain._recent_tft(origin)["demand"].sum()
    b = censored._recent_tft(origin)["demand"].sum()
    assert b < a, "censored controller should train on fewer observed units"
    assert plain._base_tft()["demand"].sum() > censored._base_tft()["demand"].sum()


def test_report_accounts_for_every_unit():
    tft, rl = make_frames()
    out, rep = apply_censoring(tft, rl, CensoringSpec(mode="policy", fill_rate=0.72))
    assert rep["units_observed"] + rep["units_censored"] == pytest.approx(
        rep["units_true"])
    assert rep["n_rows"] == len(tft)


# ── reading unmet_demand back off disk ────────────────────────────────────────

def test_attach_unmet_demand_reads_the_column_back(tmp_path):
    """`RL_REQUIRED_COLUMNS` drops unmet_demand, so E2 has to restore it."""
    from drift_pipeline.censoring import attach_unmet_demand

    _, rl = make_frames(n_days=30, n_items=2)
    csv = tmp_path / "rl_environment.csv"
    rl.to_csv(csv, index=False)

    narrowed = rl.drop(columns=["unmet_demand"])
    restored = attach_unmet_demand(narrowed, str(csv))
    assert restored["unmet_demand"].to_numpy() == pytest.approx(
        rl["unmet_demand"].to_numpy())
    assert len(restored) == len(narrowed)
    # The caller's frame must not be mutated in place.
    assert "unmet_demand" not in narrowed.columns


def test_attach_unmet_demand_is_a_noop_when_already_present(tmp_path):
    from drift_pipeline.censoring import attach_unmet_demand

    _, rl = make_frames(n_days=10, n_items=1)
    # Passing a path that does not exist proves the CSV was never read.
    assert attach_unmet_demand(rl, str(tmp_path / "absent.csv")) is rl


def test_attach_unmet_demand_fills_rows_the_csv_lacks(tmp_path):
    """`filter_to_base_trainable` keeps series the CSV may not cover; those
    censor by 0 rather than becoming NaN and poisoning the arithmetic."""
    from drift_pipeline.censoring import attach_unmet_demand

    _, rl = make_frames(n_days=20, n_items=2)
    partial = rl[rl["product_id"] == "P0"]
    csv = tmp_path / "rl_partial.csv"
    partial.to_csv(csv, index=False)

    restored = attach_unmet_demand(rl.drop(columns=["unmet_demand"]), str(csv))
    assert restored["unmet_demand"].notna().all()
    assert restored.loc[restored["product_id"] == "P1", "unmet_demand"].eq(0).all()


def test_attach_unmet_demand_rejects_duplicate_keys(tmp_path):
    from drift_pipeline.censoring import attach_unmet_demand

    _, rl = make_frames(n_days=10, n_items=1)
    csv = tmp_path / "dupes.csv"
    pd.concat([rl, rl]).to_csv(csv, index=False)
    with pytest.raises(ValueError, match="duplicate"):
        attach_unmet_demand(rl.drop(columns=["unmet_demand"]), str(csv))
