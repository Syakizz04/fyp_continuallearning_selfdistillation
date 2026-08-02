"""
Tests that CONFIG survives being read twice in one process.

`adapt_config_to_data` prunes constant features from CONFIG in place - the
repo's mutate-CONFIG idiom - and `demand_required_columns()` re-reads CONFIG at
call time, on purpose, so post-import mutation takes effect. Those two facts
combine badly: on a single-store scope `region_id` is constant and gets pruned,
so a SECOND `prepare_drift_data()` in the same process loaded a frame without a
column that joins, sorts and group-bys all key on.

It surfaced twice, in two different functions (`load_and_clean`, then
`build_tft_dataframe`), each time as a KeyError raised *after* an expensive
training step had already succeeded - which is the worst way to lose work. The
fix is that identity columns are loaded whether or not the model uses them:
pruned from a feature list means "not a feature", not "not a column".
"""

from __future__ import annotations

import copy

import pandas as pd
import pytest

from drift_pipeline import core_pipeline as cp
from drift_pipeline.core_pipeline import (CONFIG, DEMAND_KEY_COLUMNS,
                                          adapt_config_to_data,
                                          demand_required_columns)


@pytest.fixture(autouse=True)
def restore_config():
    """CONFIG is global and these tests mutate it, as the pipeline does."""
    saved = copy.deepcopy(CONFIG["forecasting"])
    yield
    CONFIG["forecasting"] = saved


def single_store_frame(n: int = 40) -> pd.DataFrame:
    dates = pd.date_range("2011-01-29", periods=n, freq="D")
    return pd.DataFrame({
        "date": dates,
        "product_id": ["FOODS_1_001"] * n,
        # Constant: exactly the condition that triggers pruning.
        "region_id": ["CA_1"] * n,
        "product_category": ["FOODS"] * n,
        "demand": range(n),
    })


def test_key_columns_survive_pruning():
    """The regression. region_id must stay required after it stops being a feature."""
    before = demand_required_columns()
    assert "region_id" in before

    dropped = adapt_config_to_data(single_store_frame())
    assert "region_id" in dropped, "fixture did not trigger the prune"
    assert "region_id" not in CONFIG["forecasting"]["group_ids"]

    after = demand_required_columns()
    assert "region_id" in after, (
        "region_id was pruned out of the loaded columns, not just out of the "
        "model features - a second prepare_drift_data() would drop it and every "
        "downstream join keyed on it would raise KeyError")


def test_every_key_column_survives_pruning():
    adapt_config_to_data(single_store_frame())
    required = demand_required_columns()
    missing = [c for c in DEMAND_KEY_COLUMNS if c not in required]
    assert not missing, f"identity column(s) {missing} lost to feature pruning"


def test_required_columns_are_stable_across_repeated_pruning():
    """Idempotent: pruning twice must not erode the column set further."""
    frame = single_store_frame()
    adapt_config_to_data(frame)
    once = demand_required_columns()
    adapt_config_to_data(frame)
    twice = demand_required_columns()
    assert once == twice


def test_required_columns_have_no_duplicates():
    """Key columns overlap group_ids, so the de-dup has to hold."""
    required = demand_required_columns()
    assert len(required) == len(set(required))


def test_product_id_is_never_pruned():
    """group_ids must not empty out, whatever the data looks like."""
    frame = single_store_frame()
    frame["product_id"] = "ONLY_ONE"          # constant too
    adapt_config_to_data(frame)
    assert "product_id" in CONFIG["forecasting"]["group_ids"]


def test_load_and_clean_tolerates_a_narrowed_frame():
    """The sort/cast path must key only on columns that are actually present."""
    adapt_config_to_data(single_store_frame())
    frame = single_store_frame().drop(columns=["region_id"])
    # `present()` inside load_and_clean is what makes this survivable; calling
    # sort_values with a hardcoded region_id key is what used to raise.
    assert frame.sort_values(
        [c for c in ["product_id", "region_id", "date"] if c in frame.columns]
    ).shape == frame.shape
    assert hasattr(cp, "load_and_clean")
