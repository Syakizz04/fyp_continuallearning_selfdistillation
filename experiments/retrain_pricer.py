"""
Retrain ONLY the base PPO pricer, against a different inventory environment.

    python -m experiments.retrain_pricer --data data/processed_m5_v3 --tag realinv

Produces a complete, loadable checkpoint directory at
`outputs/drift/checkpoints/base_<tag>/` which an edge node can be pointed at with
`FYP_BASE_CKPT_DIR` (already in `edge_system.config._SERVICE_KEYS`), leaving the
original `base/` untouched as the comparison arm.

## Why the TFT is copied rather than retrained

`rebuild_inventory.py` copies `demand_forecasting.csv` through byte-for-byte, so
the forecasting side of the new dataset is *identical*. Retraining the TFT on
identical data would burn the expensive part of base training to arrive at a
model that differs only by RNG - and would then confound the pricing comparison
with a forecaster that is not the same one. Copying makes the forecaster a
literal constant across the two arms.

The forecasting half of `calibration.json` is copied for the same reason. The RL
half IS recomputed, because the pricer changed and its profit reference is what
the drift monitor thresholds against.

## Why this is worth doing at all

The pricing agent's `inventory_level` state feature was trained against an
instant-replenishment loop that ran out of stock on 0.05% of days. It never
signalled scarcity, so the agent learned to ignore it: sweeping that input across
its full range moved under 5% of its pricing decisions. Any experiment that
degrades the inventory signal and looks for a response was therefore measuring an
artefact of the data generator rather than the hypothesis. Retraining against a
lead-time policy with a ~5% stockout rate is what gives the feature something to
say. Whether the agent then actually listens is measured, not assumed - see
`--verify`.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path
from typing import Dict, Optional

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def retrain(data_dir: Path, tag: str, *, timesteps: Optional[int] = None) -> Path:
    from drift_pipeline.base_training import calibrate_pricer, train_base_pricer
    from drift_pipeline.core_pipeline import CONFIG, prepare_drift_data
    from drift_pipeline import core_pipeline as dcfg

    if timesteps is not None:
        CONFIG["rl"]["total_timesteps_per_task"] = timesteps

    src_ckpt = Path(CONFIG["paths"]["checkpoints"]) / "base"
    dst_ckpt = Path(CONFIG["paths"]["checkpoints"]) / f"base_{tag}"
    if not src_ckpt.exists():
        raise SystemExit(f"no existing base checkpoints at {src_ckpt}")
    dst_ckpt.mkdir(parents=True, exist_ok=True)

    data = prepare_drift_data(str(data_dir / "demand_forecasting.csv"),
                              str(data_dir / "rl_environment.csv"))

    started = time.perf_counter()
    model = train_base_pricer(data["rl_base"])
    train_s = time.perf_counter() - started

    cal_start, cal_end = dcfg.calibration_window()
    rl_calib = calibrate_pricer(model, data["rl_full"], cal_start, cal_end)

    # The forecaster is a literal constant across arms.
    for name in ("base_tft.ckpt", "base_tft_dataset.pkl", "base_meta.json"):
        shutil.copy2(src_ckpt / name, dst_ckpt / name)
    model.save(str(dst_ckpt / "base_ppo.zip"))

    old_calib = json.loads(
        (Path(CONFIG["paths"]["results"]) / "calibration.json").read_text())
    new_calib = {**old_calib, "rl": rl_calib}
    new_calib.setdefault("provenance", {}).update({
        "pricer_retrained_from": str(data_dir),
        "pricer_timesteps": CONFIG["rl"]["total_timesteps_per_task"],
        "forecaster": "copied unchanged from base/",
    })
    (dst_ckpt / "calibration.json").write_text(json.dumps(new_calib, indent=2))

    print(f"\n  trained in {train_s / 60:.1f} min -> {dst_ckpt}")
    print(f"  rl profit reference: mu={rl_calib['ref_profit_mu']:.4f} "
          f"sigma={rl_calib['ref_profit_sigma']:.4f} "
          f"(was mu={old_calib['rl']['ref_profit_mu']:.4f} "
          f"sigma={old_calib['rl']['ref_profit_sigma']:.4f})")
    return dst_ckpt


def verify(data_dir: Path, ckpt_dir: Path, n_states: int = 300) -> Dict:
    """
    Measure whether the retrained agent actually uses `inventory_level`.

    The whole point of the retrain, stated as a number: what share of pricing
    decisions change when the inventory input is degraded? Reported at several
    degradation levels, because a usable experimental treatment has to move the
    outcome *smoothly* - an agent that ignores the feature until it falls off a
    cliff is no more usable than one that ignores it entirely.
    """
    import pandas as pd
    from edge_system.config import SYSTEM_CONFIG
    from edge_system.edge.inference import LocalInference

    SYSTEM_CONFIG["paths"]["data_dir"] = str(data_dir)
    SYSTEM_CONFIG["paths"]["base_ckpt_dir"] = str(ckpt_dir)

    inf = LocalInference("verify")
    inf.load(data_dir=str(data_dir))

    rl = inf.data["rl_full"]
    rng = np.random.default_rng(0)
    skus = sorted(rl["product_id"].unique())
    states = []
    for _ in range(n_states):
        sku = skus[rng.integers(len(skus))]
        sub = rl[rl["product_id"] == sku]
        row = sub.iloc[rng.integers(len(sub))]
        states.append((sku, str(row["date"].date()), float(row["inventory_level"])))

    out = {}
    for frac in (0.75, 0.5, 0.25, 0.0):
        changed, deltas = 0, []
        for sku, date, inv in states:
            a = inf.price(sku, sim_date=date, inventory_level=inv)["tier"]
            b = inf.price(sku, sim_date=date, inventory_level=inv * frac)["tier"]
            changed += a != b
            deltas.append(abs(a - b))
        out[frac] = {"changed": changed / len(states),
                     "mean_abs_dtier": float(np.mean(deltas))}
        print(f"  belief at {frac:>5.0%} of truth -> "
              f"{out[frac]['changed']:>6.1%} of decisions change, "
              f"mean|dtier|={out[frac]['mean_abs_dtier']:.4f}")
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Retrain the base PPO pricer only.")
    ap.add_argument("--data", default="data/processed_m5_v3")
    ap.add_argument("--tag", default="realinv")
    ap.add_argument("--timesteps", type=int, default=None)
    ap.add_argument("--inventory-constrained", action="store_true",
                    help="cap sales at stock on hand and charge for lost sales, "
                         "so inventory_level has a causal path to reward at all")
    ap.add_argument("--lost-sale-penalty", type=float, default=0.5)
    ap.add_argument("--verify", action="store_true",
                    help="measure inventory sensitivity after training")
    ap.add_argument("--verify-only", action="store_true")
    args = ap.parse_args(argv)

    data_dir = (PROJECT_ROOT / args.data).resolve()

    if args.inventory_constrained:
        # Set before any env is constructed: DynamicPricingEnv reads these in
        # __init__, per the repo's mutate-CONFIG-in-place convention.
        from drift_pipeline.core_pipeline import CONFIG
        import hybrid_pipeline.core_pipeline as hcfg
        for cfg in (CONFIG, hcfg.CONFIG):
            cfg["rl"]["inventory_constrained"] = True
            cfg["rl"]["lost_sale_penalty"] = args.lost_sale_penalty
        print(f"  reward: inventory-constrained "
              f"(lost-sale penalty {args.lost_sale_penalty})")

    ckpt = Path()
    if not args.verify_only:
        ckpt = retrain(data_dir, args.tag, timesteps=args.timesteps)
    else:
        from drift_pipeline.core_pipeline import CONFIG
        ckpt = Path(CONFIG["paths"]["checkpoints"]) / f"base_{args.tag}"

    if args.verify or args.verify_only:
        print("\nInventory sensitivity of the retrained pricer:")
        verify(data_dir, ckpt)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
