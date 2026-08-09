"""
Tests for the pricing environment's observation vector.

These exist because of what an audit of the real data found: four of the then
thirteen state slots - `is_pre_raya_window`, `is_pre_cny_window`,
`viral_shock_active`, `any_shock_active` - were constant zero on M5. They were
FYP1's Malaysian retail calendar and synthetic shock generator, carried onto
Walmart California data where they have no analogue.

Nothing failed. A constant input is harmless to a network, the env constructed
fine, PPO trained fine, and the results looked entirely plausible - 31% of the
observation was simply inert. That is the failure mode worth a test: not a
crash, but a state vector that quietly stops carrying what it claims to.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from drift_pipeline.core_pipeline import CONFIG, ENV_REGIME_FLAGS
from drift_pipeline.trainers import DynamicPricingEnv

#: Every slot of `_get_obs`, in order, with the frame column that feeds it.
#: Kept explicit so that adding a slot without adding it here fails loudly.
SLOTS = [
    ("demand_forecast", "demand_forecast_norm"),
    ("inventory", "inventory_norm"),
    ("competitor_price", "competitor_price_norm"),
    ("day_of_week", "day_of_week"),
    ("month", "month"),
    ("is_weekend", "is_weekend"),
    ("snap", "snap"),
    ("is_event", "is_event"),
    ("elasticity", "elasticity_coefficient"),
]


def make_rl_frame(n: int = 240, seed: int = 0) -> pd.DataFrame:
    """A frame in which every observation slot genuinely varies."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2013-01-01", periods=n, freq="D")
    demand = rng.poisson(6.0, n).astype(float)
    base_price = 5.0 + rng.normal(0, 0.4, n)
    return pd.DataFrame({
        "product_id": "P0",
        "region_id": "CA_1",
        "date": dates,
        "day_of_week": dates.dayofweek,
        "month": dates.month,
        "is_weekend": (dates.dayofweek >= 5).astype(float),
        # Both are real M5 flags; rates approximate the observed ones.
        "snap": rng.binomial(1, 0.33, n).astype(float),
        "is_event": rng.binomial(1, 0.08, n).astype(float),
        "elasticity_coefficient": rng.uniform(-2.0, -0.3, n),
        "demand_forecast_norm": rng.uniform(0.01, 0.9, n),
        "inventory_norm": rng.uniform(0.01, 0.9, n),
        "competitor_price_norm": rng.uniform(0.5, 1.0, n),
        "base_price": base_price,
        "realized_demand": demand,
        "inventory_level": rng.uniform(1, 200, n),
        "stockout_flag": rng.binomial(1, 0.05, n).astype(float),
    })


def test_state_dim_matches_what_get_obs_returns():
    env = DynamicPricingEnv(make_rl_frame())
    obs, _ = env.reset()
    assert obs.shape == (DynamicPricingEnv.STATE_DIM,)
    assert env.observation_space.shape == (DynamicPricingEnv.STATE_DIM,)
    assert len(SLOTS) == DynamicPricingEnv.STATE_DIM, (
        "SLOTS is out of step with STATE_DIM - a slot was added or removed "
        "without updating this test's map")


def test_no_observation_slot_is_constant():
    """The regression guard. A slot that never moves carries no information.

    Every column feeding the state varies in the fixture, so any constant slot
    means the env is not reading the column it claims to - which is exactly how
    four Malaysian-calendar slots sat inert on M5 without anything failing.
    """
    env = DynamicPricingEnv(make_rl_frame())
    obs = np.stack([env._get_obs(i) for i in range(len(env.df))])
    constant = [SLOTS[j][0] for j in range(obs.shape[1])
                if np.unique(obs[:, j]).size <= 1]
    assert not constant, (
        f"observation slot(s) {constant} are constant across a frame in which "
        f"every source column varies")


def test_observations_are_finite():
    env = DynamicPricingEnv(make_rl_frame())
    obs = np.stack([env._get_obs(i) for i in range(len(env.df))])
    assert np.isfinite(obs).all()


def test_regime_flags_are_real_m5_columns():
    """No Malaysian calendar, no synthetic shocks.

    M5 is Walmart California 2011-2016. `snap` (SNAP benefit distribution days)
    and `is_event` (M5 calendar events) are real columns in the source data;
    Raya, CNY and the viral-shock generator are not, and materialising them here
    produced constant-zero features named after events that never happened.
    """
    assert ENV_REGIME_FLAGS == ["snap", "is_event"]
    banned = {"is_mega_sale", "is_ramadan", "is_pre_raya_window",
              "is_pre_cny_window", "viral_shock_active", "any_shock_active"}
    assert not banned & set(ENV_REGIME_FLAGS)
    # The remap that fed the Malaysian slot names from M5 columns is gone too;
    # leaving it would let the old names come back through the data layer.
    assert "state_flag_map" not in CONFIG["rl"]


@pytest.mark.parametrize("missing", ["snap", "is_event"])
def test_missing_regime_column_does_not_crash(missing):
    """`.get(col, pd.Series(0))` falls back to a LENGTH-1 Series, so a missing
    column raises IndexError on the second step rather than defaulting to zero.
    build_rl_features materialises these; this pins the env's own tolerance."""
    frame = make_rl_frame().drop(columns=[missing])
    env = DynamicPricingEnv(frame)
    for i in range(3):
        assert np.isfinite(env._get_obs(i)).all()
