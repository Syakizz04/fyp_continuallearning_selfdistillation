"""
Which demand signal should the pricer read: the 28-day rolling mean or the TFT?

Commit 2943253 kept the rolling mean because the ORIGINAL base TFT measured
worse, over 181,412 rows:

    signal                    MAE    corr     std     bias
    28-day rolling mean     1.489   0.804   3.892   +0.003
    TFT h=0                 2.432   0.505   5.202   +0.067

That TFT no longer exists. The rebuilt one calibrates at MASE 0.891 against
1.481 (see CLAUDE.md, "The November 2026 rebuild"). This re-measures the same
table with the same code path, refresh_rl_features.tft_one_step_forecast, on
the same rows, so the decision can be re-made on the model actually served.

It reads only. It does not rewrite rl_environment.csv and does not change the
pricer's input. Switching the pricer to the TFT would mean rebuilding v4 with
`--forecast-source tft` and retraining the pricer. That would change E2's
pricing side, although not its arm comparison, because the forecast is
generated once and is identical across arms.

    python -m experiments.forecast_signal_check
Writes results/forecast_signal/summary.json.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def row(name: str, pred: np.ndarray, real: np.ndarray) -> dict:
    return {"signal": name, "MAE": float(np.mean(np.abs(pred - real))),
            "corr": float(np.corrcoef(pred, real)[0, 1]), "std": float(np.std(pred)),
            "bias": float(np.mean(pred - real))}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--data-dir", type=Path, default=ROOT / "data" / "processed_m5_v4")
    ap.add_argument("--ckpt-dir", type=Path,
                    default=ROOT / "outputs" / "drift" / "checkpoints" / "base_v4")
    ap.add_argument("--out", type=Path, default=ROOT / "results" / "forecast_signal")
    args = ap.parse_args()

    from dataset_generator.m5.refresh_rl_features import tft_one_step_forecast

    rl = pd.read_csv(args.data_dir / "rl_environment.csv", parse_dates=["date"])
    fc, fstats = tft_one_step_forecast(args.ckpt_dir, args.data_dir)
    fc["date"] = pd.to_datetime(fc["date"])
    m = rl.merge(fc, on=["product_id", "date"], how="inner")
    m = m[m["tft_forecast"].notna()]
    real = m["realized_demand"].to_numpy(float)
    table = [row("28-day rolling mean", m["demand_forecast"].to_numpy(float), real),
             row("TFT h=0", m["tft_forecast"].to_numpy(float), real)]
    out = {"rows": int(len(m)), "ckpt_dir": str(args.ckpt_dir), "data_dir": str(args.data_dir),
           "table": table, "tft_stats": fstats,
           "original_2943253": [
               {"signal": "28-day rolling mean", "MAE": 1.489, "corr": 0.804, "std": 3.892, "bias": 0.003},
               {"signal": "TFT h=0", "MAE": 2.432, "corr": 0.505, "std": 5.202, "bias": 0.067}]}
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "summary.json").write_text(json.dumps(out, indent=2, default=float) + "\n")

    print(f"\nover {len(m):,} rows (original: 181,412)")
    print(f"  {'signal':22s} {'MAE':>7s} {'corr':>7s} {'std':>7s} {'bias':>7s}")
    for r in table:
        print(f"  {r['signal']:22s} {r['MAE']:7.3f} {r['corr']:7.3f} {r['std']:7.3f} {r['bias']:+7.3f}")
    better = min(table, key=lambda r: r["MAE"])["signal"]
    print(f"\nlower MAE: {better}   -> {args.out / 'summary.json'}")


if __name__ == "__main__":
    main()
