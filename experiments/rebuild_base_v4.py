"""
Rebuild outputs/drift/checkpoints/base_v4/ from data/processed_m5_v4/.

    python -m experiments.rebuild_base_v4

Recovered on 2026-11-21 after the original checkpoints were lost. It follows
the recipe the commit history records for how base_v4 was made:

  1. drift_pipeline.base_training trains the base TFT (and a first PPO) and
     calibrates both -> checkpoints/base/.
     The TFT trains on demand_forecasting.csv, which is byte-identical from
     v1 to v4, so training it on v4's copy is the same as on the original.
  2. experiments.retrain_pricer copies that TFT unchanged and retrains only the
     PPO pricer on v4 (9-dim state, inventory-constrained reward, which is now
     the CONFIG default) -> checkpoints/base_v4/ (commit 4f7faf4).

Calibration goes to outputs/drift/results_rebuild/, NOT outputs/drift/results/.
That directory holds the FYP1 drift-walk results from the old laptop, and
save_base would overwrite their calibration.json.

A retrained model is a new draw, not a bit copy: even the original GPU fits
were not bit-reproducible. Measure how close it is with
`python -m experiments.compare_rebuild --walk-checks 6`.
"""

from __future__ import annotations

import shutil
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

DATA = PROJECT_ROOT / "data" / "processed_m5_v4"


def main() -> int:
    from drift_pipeline import core_pipeline as cp
    from drift_pipeline import base_training as bt

    if not (DATA / "rl_environment.csv").exists():
        raise SystemExit(f"missing {DATA}; build it first (dataset_generator/m5/README.md)")

    results = PROJECT_ROOT / "outputs" / "drift" / "results_rebuild"
    results.mkdir(parents=True, exist_ok=True)
    cp.CONFIG["paths"]["results"] = str(results)
    cp.CONFIG["paths"]["demand_csv"] = str(DATA / "demand_forecasting.csv")
    cp.CONFIG["paths"]["rl_csv"] = str(DATA / "rl_environment.csv")
    ckpt_root = Path(cp.CONFIG["paths"]["checkpoints"])
    print(f"[rebuild] device={cp.DEVICE} results={results} checkpoints={ckpt_root}", flush=True)

    t0 = time.perf_counter()
    data = cp.prepare_drift_data()
    res = bt.run_base_training_and_calibration(data)
    shutil.copy2(res["paths"]["calibration"], ckpt_root / "base" / "calibration.json")
    print(f"[rebuild] step 1 done in {(time.perf_counter() - t0) / 60:.1f} min "
          f"-> {ckpt_root / 'base'}", flush=True)

    t1 = time.perf_counter()
    from experiments.retrain_pricer import retrain
    dst = retrain(DATA, "v4")
    print(f"[rebuild] step 2 done in {(time.perf_counter() - t1) / 60:.1f} min -> {dst}", flush=True)
    print(f"[rebuild] total {(time.perf_counter() - t0) / 60:.1f} min", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
