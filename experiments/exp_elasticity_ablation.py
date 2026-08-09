"""
E3 - elasticity ablation. Robustness check on the pricing side: does the RL
pricer's behaviour depend on which elasticity estimate it was trained against?

    python -m experiments.exp_elasticity_ablation --dry-run
    python -m experiments.exp_elasticity_ablation --build-ckpt
    python -m experiments.exp_elasticity_ablation

Not part of the CL argument E2 makes - see CLAUDE.md's experiment table. This
exists to survive the question "does the pricing side's data quality bug
matter", not to support the forgetting claim.

## The two sides

* `old`         - checkpoints/base, trained against data/processed_m5's
                  original estimate_elasticity(): 57% of items were arbitrary
                  (25% a hard -1.5 fallback, 32% clip-pinned at [-3.0, -0.2]).
* `re_grounded` - checkpoints/base_<tag>, trained against
                  data/processed_m5_v2's three-level empirical-Bayes shrinkage
                  (0% fallback, 0% clip-pinned - see
                  dataset_generator/m5/elasticity.py). Built on demand via
                  `experiments.retrain_pricer`, which copies the TFT
                  forecaster unchanged from checkpoints/base/ -
                  demand_forecasting.csv does not change between dataset
                  versions, only rl_environment.csv's elasticity_coefficient
                  does - so only the PPO pricer differs between the two sides.
                  NOTE: retrain_pricer.retrain() always copies its TFT from
                  checkpoints/base/ specifically, regardless of --old-ckpt.

## Why `frozen`, not a CL sweep

This is not a forgetting question - there is no drift trigger under test here,
and running naive/replay/sdft would confound "what the pricer learned from its
elasticity input" with "how each CL mechanism happened to retrain it". Both
sides run the `frozen` strategy (never retrains), which isolates exactly the
one thing under test.

## Two comparisons, because aggregate agreement can hide item-level disagreement

1. Walk-forward `walk_profit_mean` / `walk_mase_mean` for each side, off the
   same drift_stream_frozen.csv schema E2 reads - do aggregate outcomes move?
   Read directly per side rather than through metrics.build_accuracy_table,
   which compares arms sharing ONE results directory (E2's 4 arms in one
   cell); here there is one arm ('frozen') per side, each in its own
   directory, so a direct side-by-side read is the right shape.
2. Paired price-tier comparison: the SAME (SKU, date, inventory_level) state
   priced by both checkpoints - what share of individual pricing decisions
   actually change when the elasticity value the pricer was trained on is
   corrected? Modelled directly on `experiments.retrain_pricer.verify()`,
   which asks the analogous question for inventory sensitivity. This is the
   number that actually answers "robust or not" - two similar aggregate
   profit numbers could still hide compensating errors at the SKU level.

## The one thing this file checks before trusting either comparison

`_assert_same_grid` confirms old and re-grounded datasets cover the same
(product_id, date) rows. They are meant to differ ONLY in
elasticity_coefficient; if the grid itself differs, the paired comparison
would be pricing different states under the two checkpoints, not the same
states under two elasticity beliefs, and the result would not mean what it
claims to.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from rich.console import Console  # noqa: E402

from drift_pipeline import monitor as mon  # noqa: E402
from drift_pipeline import retrain as rt  # noqa: E402
from drift_pipeline.core_pipeline import CONFIG, prepare_drift_data  # noqa: E402
# Reused rather than reimplemented - same per-cell results-dir redirection and
# GPU-freeing E2 already relies on.
from experiments.exp_staleness_cl import free_gpu, results_dir  # noqa: E402

console = Console()

#: There is no drift trigger under test - see module docstring.
ARM = "frozen"

DEFAULT_OLD_DATA = PROJECT_ROOT / "data" / "processed_m5"
DEFAULT_OLD_CKPT = PROJECT_ROOT / "outputs" / "drift" / "checkpoints" / "base"
DEFAULT_NEW_DATA = PROJECT_ROOT / "data" / "processed_m5_v2"

_CKPT_FILES = ("base_tft.ckpt", "base_tft_dataset.pkl", "base_ppo.zip",
              "base_meta.json", "calibration.json")


def new_ckpt_dir(tag: str) -> Path:
    return PROJECT_ROOT / "outputs" / "drift" / "checkpoints" / f"base_{tag}"


def base_paths_with_fallback(ckpt_dir: Path) -> Dict[str, str]:
    """Same shape as exp_staleness_cl.base_paths, but calibration falls back
    to the shared results dir when the checkpoint has none of its own - true
    for checkpoints/base/, which predates experiments.retrain_pricer and was
    never given a private calibration.json (checkpoints/base_v4/ has one; the
    original checkpoints/base/ does not). Mirrors the identical fallback in
    edge_system/edge/inference.py's LocalInference.load()."""
    calib = ckpt_dir / "calibration.json"
    if not calib.exists():
        calib = Path(CONFIG["paths"]["results"]) / "calibration.json"
    return {
        "tft": str(ckpt_dir / "base_tft.ckpt"),
        "dataset": str(ckpt_dir / "base_tft_dataset.pkl"),
        "ppo": str(ckpt_dir / "base_ppo.zip"),
        "meta": str(ckpt_dir / "base_meta.json"),
        "calibration": str(calib),
    }


def _assert_same_grid(old_data: Path, new_data: Path) -> None:
    old_rl = pd.read_csv(old_data / "rl_environment.csv", usecols=["product_id", "date"])
    new_rl = pd.read_csv(new_data / "rl_environment.csv", usecols=["product_id", "date"])
    old_keys = set(zip(old_rl["product_id"], old_rl["date"]))
    new_keys = set(zip(new_rl["product_id"], new_rl["date"]))
    if old_keys != new_keys:
        raise SystemExit(
            f"old and re-grounded datasets do not share the same (sku, date) "
            f"grid ({len(new_keys - old_keys)} new-only, "
            f"{len(old_keys - new_keys)} old-only) - the paired price "
            f"comparison would be pricing different states, not the same "
            f"states under two elasticity beliefs.")


def ensure_regrounded_checkpoint(tag: str, data_dir: Path, *, build: bool) -> Path:
    ckpt_dir = new_ckpt_dir(tag)
    if all((ckpt_dir / n).exists() for n in _CKPT_FILES):
        console.print(f"  [dim]re-grounded checkpoint already exists at {ckpt_dir}[/dim]")
        return ckpt_dir
    if not build:
        raise SystemExit(
            f"no re-grounded checkpoint at {ckpt_dir}. Build it first:\n"
            f"  python -m experiments.retrain_pricer --data {data_dir} --tag {tag}\n"
            f"or pass --build-ckpt to have this script do it (needs a GPU).")
    console.print(f"  [yellow]building re-grounded checkpoint[/yellow] -> {ckpt_dir}")
    from experiments.retrain_pricer import retrain as retrain_pricer
    return retrain_pricer(data_dir, tag)


def _finite_mean(s: pd.Series) -> float:
    s = pd.to_numeric(s, errors="coerce")
    s = s[np.isfinite(s)]
    return float(s.mean()) if len(s) else float("nan")


def run_side(name: str, data_dir: Path, ckpt_dir: Path, cell_dir: Path,
            max_checks: Optional[int] = None) -> Dict:
    """Run the `frozen` walk on one (data, checkpoint) side. Loads the base
    BEFORE redirecting CONFIG['paths']['results'] to cell_dir, same ordering
    exp_staleness_cl.run_cell uses - the default calibration fallback in
    base_paths_with_fallback reads from the ORIGINAL results dir, which would
    resolve to the wrong (per-cell) path if reordered."""
    started = time.perf_counter()
    CONFIG["paths"]["demand_csv"] = str(data_dir / "demand_forecasting.csv")
    CONFIG["paths"]["rl_csv"] = str(data_dir / "rl_environment.csv")
    data = prepare_drift_data()
    if max_checks is not None:
        data["checks"] = list(data["checks"])[:max_checks]

    free_gpu()
    base = mon.load_base(base_paths_with_fallback(ckpt_dir))

    with results_dir(cell_dir):
        out = rt.run_arm(ARM, data, base=base)
        ctrl = out["controller"]

    n_fc, n_rl = ctrl.stats.n_fc_retrains, ctrl.stats.n_rl_retrains
    del ctrl, out, base
    free_gpu()
    return {"side": name, "n_fc_retrains": n_fc, "n_rl_retrains": n_rl,
            "wall_seconds": time.perf_counter() - started}


def build_walk_comparison(cell_old: Path, cell_new: Path) -> pd.DataFrame:
    rows = []
    for name, cell_dir in (("old", cell_old), ("re_grounded", cell_new)):
        df = pd.read_csv(cell_dir / f"drift_stream_{ARM}.csv", parse_dates=["date"])
        rows.append({
            "side": name,
            "n_checks": len(df),
            "walk_mase_mean": _finite_mean(df["mase"]),
            "walk_smape_mean": _finite_mean(df["smape"]),
            "walk_profit_mean": _finite_mean(df["cumulative_profit"]),
        })
    return pd.DataFrame(rows).set_index("side")


def paired_price_comparison(old_data: Path, old_ckpt: Path, new_data: Path,
                            new_ckpt: Path, n_states: int = 300,
                            seed: int = 0) -> Dict:
    """Modelled on experiments.retrain_pricer.verify(): sample states from the
    new (re-grounded) dataset's grid, then price the SAME (sku, date,
    inventory_level) under both checkpoints. _assert_same_grid has already
    confirmed both datasets cover this grid, so 'new' vs 'old' here is purely
    about which checkpoint answers, not which data it was sampled from."""
    from edge_system.config import SYSTEM_CONFIG
    from edge_system.edge.inference import LocalInference

    SYSTEM_CONFIG["paths"]["data_dir"] = str(new_data)
    SYSTEM_CONFIG["paths"]["base_ckpt_dir"] = str(new_ckpt)
    inf_new = LocalInference("e3_re_grounded")
    inf_new.load(data_dir=str(new_data))

    SYSTEM_CONFIG["paths"]["data_dir"] = str(old_data)
    SYSTEM_CONFIG["paths"]["base_ckpt_dir"] = str(old_ckpt)
    inf_old = LocalInference("e3_old")
    inf_old.load(data_dir=str(old_data))

    rl = inf_new.data["rl_full"]
    rng = np.random.default_rng(seed)
    skus = sorted(rl["product_id"].unique())
    states = []
    for _ in range(n_states):
        sku = skus[rng.integers(len(skus))]
        sub = rl[rl["product_id"] == sku]
        row = sub.iloc[rng.integers(len(sub))]
        states.append((sku, str(row["date"].date()), float(row["inventory_level"])))

    changed, deltas = 0, []
    for sku, date, inv in states:
        a = inf_old.price(sku, sim_date=date, inventory_level=inv)["tier"]
        b = inf_new.price(sku, sim_date=date, inventory_level=inv)["tier"]
        changed += a != b
        deltas.append(abs(a - b))

    out = {"n_states": n_states,
          "share_decisions_changed": changed / n_states,
          "mean_abs_dtier": float(np.mean(deltas))}
    console.print(f"  old vs re-grounded -> {out['share_decisions_changed']:.1%} of "
                  f"pricing decisions change, mean|dtier|={out['mean_abs_dtier']:.4f}")
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="E3: elasticity ablation (old vs re-grounded).")
    ap.add_argument("--old-data", type=Path, default=DEFAULT_OLD_DATA)
    ap.add_argument("--old-ckpt", type=Path, default=DEFAULT_OLD_CKPT)
    ap.add_argument("--new-data", type=Path, default=DEFAULT_NEW_DATA)
    ap.add_argument("--tag", default="elasticity_v2",
                    help="checkpoint tag for the re-grounded pricer; produces "
                         "outputs/drift/checkpoints/base_<tag>/")
    ap.add_argument("--build-ckpt", action="store_true",
                    help="train the re-grounded checkpoint if missing (needs a "
                         "GPU) - see experiments.retrain_pricer")
    ap.add_argument("--n-states", type=int, default=300,
                    help="paired price-tier comparison sample size")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-checks", type=int, default=None,
                    help="truncate the walk to the first N weekly checks - for "
                         "smoke-testing the wiring only, not a comparable result")
    ap.add_argument("--out", default="e3")
    ap.add_argument("--dry-run", action="store_true",
                    help="check paths and grid alignment, then stop. No models "
                         "are loaded")
    args = ap.parse_args(argv)

    for name, path in (("old data", args.old_data), ("old ckpt", args.old_ckpt),
                       ("new data", args.new_data)):
        if not path.exists():
            console.print(f"[red]{name} not found:[/red] {path}")
            return 1

    console.print(f"[bold]E3[/bold]: old ({args.old_data.name}) vs "
                  f"re_grounded ({args.new_data.name}), arm={ARM}")

    _assert_same_grid(args.old_data, args.new_data)
    console.print("  [green]OK[/green] old and re-grounded share the same (sku, date) grid")

    new_ckpt = new_ckpt_dir(args.tag)
    have_new_ckpt = all((new_ckpt / n).exists() for n in _CKPT_FILES)
    console.print(f"  re-grounded checkpoint: "
                  + ("found" if have_new_ckpt else
                     f"MISSING at {new_ckpt} - build with: "
                     f"python -m experiments.retrain_pricer --data "
                     f"{args.new_data} --tag {args.tag}  (or pass --build-ckpt)"))

    if args.dry_run:
        console.print("\n[bold]Dry run[/bold] - plan resolved above, no models loaded.")
        return 0

    new_ckpt = ensure_regrounded_checkpoint(args.tag, args.new_data, build=args.build_ckpt)

    root = Path(CONFIG["paths"]["results"]) / args.out
    root.mkdir(parents=True, exist_ok=True)
    cell_old, cell_new = root / "old", root / "re_grounded"

    console.print("\n[bold magenta]== old ==[/bold magenta]")
    row_old = run_side("old", args.old_data, args.old_ckpt, cell_old,
                       max_checks=args.max_checks)
    console.print(f"  done in {row_old['wall_seconds'] / 60:.1f} min "
                  f"({row_old['n_fc_retrains']} FC + {row_old['n_rl_retrains']} RL retrains)")

    console.print("\n[bold magenta]== re_grounded ==[/bold magenta]")
    row_new = run_side("re_grounded", args.new_data, new_ckpt, cell_new,
                       max_checks=args.max_checks)
    console.print(f"  done in {row_new['wall_seconds'] / 60:.1f} min "
                  f"({row_new['n_fc_retrains']} FC + {row_new['n_rl_retrains']} RL retrains)")

    walk = build_walk_comparison(cell_old, cell_new)
    console.print("\n[bold]Walk-forward comparison[/bold] (frozen arm, both sides)")
    console.print(walk.to_string())

    console.print(f"\n[bold]Paired price-tier comparison[/bold] ({args.n_states} states)")
    paired = paired_price_comparison(args.old_data, args.old_ckpt, args.new_data,
                                     new_ckpt, n_states=args.n_states, seed=args.seed)

    out_json = root / f"{args.out}_summary.json"
    out_json.write_text(json.dumps({
        "walk": walk.reset_index().to_dict(orient="records"),
        "paired_price_comparison": paired,
        "old_data": str(args.old_data), "old_ckpt": str(args.old_ckpt),
        "new_data": str(args.new_data), "new_ckpt": str(new_ckpt),
        "seed": args.seed,
    }, indent=2, default=float))
    console.print(f"\n[green]->[/green] {out_json}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
