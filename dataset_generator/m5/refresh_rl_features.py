"""
Rebuild the two synthetic pricing features that did not survive an audit.

    python -m dataset_generator.m5.refresh_rl_features            # v3 -> v4
    python -m dataset_generator.m5.refresh_rl_features --competitor-store WI_1
    python -m dataset_generator.m5.refresh_rl_features --report-only

This is a *post-process*, not a rebuild. `build_m5.py` re-estimates elasticity
and re-simulates inventory, and neither needs to change: `inventory_level`,
`unmet_demand`, `stockout_flag` and `elasticity_coefficient` are carried over
bit-identical, so E1's measured fill rates and the conservation audit still
describe this dataset. Only the two audited columns move.
`demand_forecasting.csv` is copied untouched, so the base TFT checkpoint stays
valid.

## What was wrong

**competitor_price** was the median `sell_price` of the *same store's*
department. Measured against the item's own price it correlated 0.117, and the
competitor/own ratio had a standard deviation of 1.297 - larger than its mean of
1.235, meaning the "competitor" was routinely triple or a third of the focal
price. Real competing retailers on an identical product sit within a much
narrower band. It was also endogenous: the focal item is inside the department
being aggregated, so its own price moved its own competitor's price.

The replacement is the **same item's actual sell_price at a different store**.
That is observed data, not a proxy - a real price, set independently, for the
identical product, with its own promotional calendar. A cross-state store is the
default so the comparison is not dominated by chain-wide pricing policy.

**demand_forecast** was a trailing 28-day mean of realized demand. Honest and
correctly lagged, but it is not what a deployed node would price against: the
node runs a TFT, so the pricer should consume the forecaster's output. The
replacement is the base TFT's 1-step-ahead prediction.

## The limitation this deliberately keeps

The forecast is generated once, by the **base** model, and is identical for every
CL arm. It is not regenerated as an arm's forecaster drifts. That keeps the RL
comparison clean - arms differ only in their own CL mechanism, not in the
environment they inhabit - at the cost that forecaster degradation does not
propagate into pricing. Wiring the *live* forecaster into the environment is the
more realistic design and the more confounded one; state it as future work.

Consequence: the first `encoder_length` days of each series have no TFT window
and keep the old rolling-mean value. Those rows are reported, not hidden.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

DEFAULT_SRC = PROJECT_ROOT / "data" / "processed_m5_v3"
DEFAULT_OUT = PROJECT_ROOT / "data" / "processed_m5_v4"
DEFAULT_RAW = PROJECT_ROOT / "data" / "m5_raw"
DEFAULT_CKPT = PROJECT_ROOT / "outputs" / "drift" / "checkpoints" / "base_cover"

#: Cross-state by default. Same chain, so prices stay plausible for the same
#: product; different state, so the promotional calendar and local competition
#: are genuinely distinct rather than a copy of the focal store's policy.
DEFAULT_COMPETITOR_STORE = "TX_1"


# ─── Competitor price ────────────────────────────────────────────────────────

def cross_store_competitor(rl: pd.DataFrame, raw_dir: Path,
                           store: str) -> tuple[pd.Series, Dict]:
    """Same item, different store, real observed weekly price."""
    cal = pd.read_csv(raw_dir / "calendar.csv", usecols=["date", "wm_yr_wk"])
    cal["date"] = pd.to_datetime(cal["date"])

    prices = pd.read_csv(raw_dir / "sell_prices.csv")
    prices = prices[prices["store_id"].eq(store)]
    if prices.empty:
        raise SystemExit(f"no rows for store {store!r} in sell_prices.csv")
    prices = (prices.rename(columns={"item_id": "product_id",
                                     "sell_price": "competitor_price"})
                    [["product_id", "wm_yr_wk", "competitor_price"]])

    keys = rl[["product_id", "date"]].copy()
    keys["date"] = pd.to_datetime(keys["date"])
    keys = keys.merge(cal, on="date", how="left")
    merged = keys.merge(prices, on=["product_id", "wm_yr_wk"], how="left")
    if len(merged) != len(rl):
        raise ValueError(f"competitor join changed row count "
                         f"({len(rl)} -> {len(merged)})")

    comp = merged["competitor_price"]
    raw_missing = float(comp.isna().mean())

    # An item not yet stocked at the comparison store has no price that week.
    # Carry the last known one forward within the item, then fall back to the
    # focal store's own price - a competitor assumed to match you is the
    # conservative assumption, and it keeps the column free of NaN.
    comp = (comp.groupby(merged["product_id"]).ffill()
                .groupby(merged["product_id"]).bfill())
    filled_from_own = comp.isna()
    comp = comp.fillna(rl["base_price"])

    return comp.to_numpy(), {
        "competitor_store": store,
        "weeks_unpriced_at_competitor": raw_missing,
        "rows_fallen_back_to_own_price": float(filled_from_own.mean()),
    }


# ─── TFT forecast ────────────────────────────────────────────────────────────

def tft_one_step_forecast(ckpt_dir: Path, src: Path) -> tuple[pd.DataFrame, Dict]:
    """1-step-ahead base-TFT prediction for every (product_id, date) it can cover."""
    from drift_pipeline.core_pipeline import CONFIG, prepare_drift_data

    CONFIG["paths"]["demand_csv"] = str(src / "demand_forecasting.csv")
    CONFIG["paths"]["rl_csv"] = str(src / "rl_environment.csv")

    from drift_pipeline import monitor as mon
    from drift_pipeline.trainers import (filter_tft_eval_frame,
                                         make_tft_dataset, min_tft_rows)

    data = prepare_drift_data()
    base = mon.load_base({
        "tft": str(ckpt_dir / "base_tft.ckpt"),
        "dataset": str(ckpt_dir / "base_tft_dataset.pkl"),
        "ppo": str(ckpt_dir / "base_ppo.zip"),
        "meta": str(ckpt_dir / "base_meta.json"),
        "calibration": str(ckpt_dir / "calibration.json"),
    })

    full = filter_tft_eval_frame(data["tft_full"], min_length=min_tft_rows())
    ds = make_tft_dataset(full, train=False, training_dataset=base["train_ds"])
    dl = ds.to_dataloader(train=False, shuffle=False,
                          batch_size=CONFIG["forecasting"]["batch_size"],
                          num_workers=0)
    pred = base["forecaster"].predict(dl, mode="prediction", return_index=True)

    # Column 0 of the 14-day horizon is the 1-step-ahead value, which is the
    # like-for-like replacement for the shift(1) rolling mean it supersedes.
    out = pred.index.copy()
    out["tft_forecast"] = np.asarray(pred.output[:, 0].cpu(), dtype=float)

    # `allow_missing_timesteps=True` emits extra samples at the tail of each
    # series whose 14-day decoder runs past the end; they all report the same
    # final decoder start. Column 0 is a genuine 1-step-ahead value for that
    # index in every one of them, so keeping the first is lossless. ~2% of rows.
    n_before = len(out)
    out = out.drop_duplicates(["product_id", "time_idx"], keep="first")
    n_truncated = n_before - len(out)

    # Map time_idx back to a date PER SERIES. time_idx is consecutive within a
    # group but each series has its own origin, so a global time_idx -> date
    # lookup silently mis-dates every item but the first and emits duplicate
    # (product_id, date) pairs.
    lookup = (full[["product_id", "time_idx", "date"]]
              .astype({"product_id": str})
              .drop_duplicates(["product_id", "time_idx"]))
    out["product_id"] = out["product_id"].astype(str)
    out = out.merge(lookup, on=["product_id", "time_idx"], how="inner")
    out = out[["product_id", "date", "tft_forecast"]]
    if out.duplicated(["product_id", "date"]).any():
        raise ValueError("forecast has duplicate (product_id, date) keys")
    # A negative unit forecast is not a quantity; the quantile head can emit one.
    out["tft_forecast"] = out["tft_forecast"].clip(lower=0.0)
    return out, {"tft_rows_predicted": int(len(out)),
                 "tail_windows_deduped": int(n_truncated)}


# ─── Driver ──────────────────────────────────────────────────────────────────

def refresh(src: Path, out: Path, raw_dir: Path, ckpt_dir: Path,
            store: str) -> Dict:
    rl = pd.read_csv(src / "rl_environment.csv")
    rl["date"] = pd.to_datetime(rl["date"])
    stats: Dict = {"n_rows": int(len(rl)), "source": str(src)}

    old_comp = rl["competitor_price"].to_numpy(dtype=float)
    old_fc = rl["demand_forecast"].to_numpy(dtype=float)

    print(f"  competitor price <- store {store} (same item, real weekly price)")
    comp, cstats = cross_store_competitor(rl, raw_dir, store)
    rl["competitor_price"] = comp
    stats.update(cstats)

    print("  demand_forecast  <- base TFT, 1-step-ahead")
    fc, fstats = tft_one_step_forecast(ckpt_dir, src)
    stats.update(fstats)
    merged = rl[["product_id", "date"]].merge(fc, on=["product_id", "date"],
                                              how="left")
    if len(merged) != len(rl):
        raise ValueError("forecast join changed row count")
    covered = merged["tft_forecast"].notna()
    # Rows before the first full encoder window keep the rolling mean rather
    # than a fabricated value.
    rl["demand_forecast"] = np.where(covered, merged["tft_forecast"], old_fc)
    stats["forecast_from_tft_share"] = float(covered.mean())

    out.mkdir(parents=True, exist_ok=True)
    for name in ("demand_forecasting.csv", "elasticity_report.csv"):
        if (src / name).exists():
            shutil.copy2(src / name, out / name)
    rl.to_csv(out / "rl_environment.csv", index=False)

    # ── before/after, so the change is auditable rather than asserted ───────
    base_price = rl["base_price"].to_numpy(dtype=float)
    realized = rl["realized_demand"].to_numpy(dtype=float)

    def corr(a, b):
        return float(np.corrcoef(a, b)[0, 1])

    def ratio_stats(c):
        r = c / np.where(base_price > 0, base_price, np.nan)
        return float(np.nanmean(r)), float(np.nanstd(r))

    old_m, old_s = ratio_stats(old_comp)
    new_m, new_s = ratio_stats(comp)
    stats["competitor"] = {
        "corr_with_own_price_before": corr(old_comp, base_price),
        "corr_with_own_price_after": corr(comp, base_price),
        "ratio_mean_before": old_m, "ratio_std_before": old_s,
        "ratio_mean_after": new_m, "ratio_std_after": new_s,
        "distinct_before": int(pd.Series(old_comp).nunique()),
        "distinct_after": int(pd.Series(comp).nunique()),
    }
    new_fc = rl["demand_forecast"].to_numpy(dtype=float)
    stats["forecast"] = {
        "corr_with_realized_before": corr(old_fc, realized),
        "corr_with_realized_after": corr(new_fc, realized),
        "mae_before": float(np.mean(np.abs(old_fc - realized))),
        "mae_after": float(np.mean(np.abs(new_fc - realized))),
    }
    return stats


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--src", type=Path, default=DEFAULT_SRC)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW)
    ap.add_argument("--base-ckpt", type=Path, default=DEFAULT_CKPT)
    ap.add_argument("--competitor-store", default=DEFAULT_COMPETITOR_STORE)
    ap.add_argument("--report-only", action="store_true",
                    help="print the existing dataset's stats and exit")
    args = ap.parse_args(argv)

    if args.report_only:
        rl = pd.read_csv(args.src / "rl_environment.csv")
        bp = rl["base_price"].to_numpy(dtype=float)
        c = rl["competitor_price"].to_numpy(dtype=float)
        print(f"corr(competitor, base_price) = {np.corrcoef(c, bp)[0,1]:.4f}")
        r = c / np.where(bp > 0, bp, np.nan)
        print(f"competitor/own ratio         = {np.nanmean(r):.3f} +/- {np.nanstd(r):.3f}")
        return 0

    print(f"\nRefreshing pricing features: {args.src.name} -> {args.out.name}\n")
    stats = refresh(args.src, args.out, args.raw_dir, args.base_ckpt,
                    args.competitor_store)

    c, f = stats["competitor"], stats["forecast"]
    print(f"\n  competitor_price (store {stats['competitor_store']})")
    print(f"    corr with own price   {c['corr_with_own_price_before']:+.3f} -> "
          f"{c['corr_with_own_price_after']:+.3f}")
    print(f"    competitor/own ratio  {c['ratio_mean_before']:.3f}+/-{c['ratio_std_before']:.3f}"
          f" -> {c['ratio_mean_after']:.3f}+/-{c['ratio_std_after']:.3f}")
    print(f"    distinct values       {c['distinct_before']} -> {c['distinct_after']}")
    print(f"    fell back to own price on {stats['rows_fallen_back_to_own_price']:.2%} of rows")
    print(f"\n  demand_forecast (base TFT, 1-step-ahead)")
    print(f"    corr with realized    {f['corr_with_realized_before']:.3f} -> "
          f"{f['corr_with_realized_after']:.3f}")
    print(f"    MAE vs realized       {f['mae_before']:.3f} -> {f['mae_after']:.3f}")
    print(f"    TFT-covered rows      {stats['forecast_from_tft_share']:.2%} "
          f"(rest keep the rolling mean: no encoder window yet)")

    import json
    (args.out / "refresh_report.json").write_text(json.dumps(stats, indent=2,
                                                            default=float))
    print(f"\n-> {args.out / 'rl_environment.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
