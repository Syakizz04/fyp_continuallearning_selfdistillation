"""The realised-demand overlay: traded days take what the channel saw, scaled
up by its share - ordered into the check's truth frame, sold into the censored
training frame - and every other row keeps the dataset."""

import pandas as pd
import pytest

from edge_system.edge.inference import LocalInference


def node_with(frame):
    inf = LocalInference("pos")
    inf._data = {"tft_full": frame}
    return inf


def frame():
    dates = pd.date_range("2013-01-01", periods=4)
    return pd.DataFrame({"product_id": ["A"] * 4 + ["B"] * 4,
                         "date": list(dates) * 2, "demand": [5.0] * 8})


def test_overlay_scales_by_share_and_splits_truth_from_sales():
    inf = node_with(frame())
    daily = {"A": [{"sim_date": "2013-01-03", "ordered": 4, "sold": 3},
                   {"sim_date": "2013-01-04", "ordered": 2, "sold": 2}]}
    assert inf.apply_realised(daily, share=0.5) == 2
    truth, obs = inf._realised_truth, inf._realised_obs
    a_t = truth.set_index("date")["demand"].tolist()
    a_o = obs.set_index("date")["demand"].tolist()
    assert a_t == [5.0, 5.0, 8.0, 4.0]          # untraded days keep the dataset
    assert a_o == [5.0, 5.0, 6.0, 4.0]          # refused orders never became sales
    assert set(truth["product_id"]) == {"A"}    # only the SKUs this node trades
    assert inf.data["tft_full_censored"] is obs  # what the retrain reads
    assert inf._realised_skus == {"A"}


def test_overlay_empty_is_a_no_op():
    inf = node_with(frame())
    assert inf.apply_realised({}, share=0.5) == 0
    assert inf.apply_realised({"Z": [{"sim_date": "2013-01-01", "ordered": 1, "sold": 1}]},
                              share=0.5) == 0
    assert inf._realised_truth is None and "tft_full_censored" not in inf.data
