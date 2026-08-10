"""
E5 - the SDFT alpha sweep. Is replay-free SDFT a viable substitute, or is it
just too conservative at the one alpha E2 happened to run?

    python -m experiments.exp_sdft_alpha --dry-run
    python -m experiments.exp_sdft_alpha
    python -m experiments.exp_sdft_alpha --alphas 0.95 --censoring none --max-checks 6

## Why this exists

E2 measured SDFT at `sdft_alpha = 0.50` and found it barely adapts: at
escrow_quota its walk MASE is 1.419 against frozen's 1.474 and replay's 1.020,
so it recovers about 12% of the adaptation replay delivers, while at that same
level forgetting MORE than plain naive fine-tuning (0.097 vs 0.053). Read as a
verdict on self-distillation that is damning. Read as a measurement it is
incomplete, because `trainers.py` applies

    loss = alpha * task_loss + (1 - alpha) * distillation

so alpha=0.5 spends half of every gradient step re-fitting the pre-drift
teacher. That is not an incidental hyperparameter; it IS the adaptation-versus-
retention dial the experiment is about, and E2 reports a single arbitrary point
on it.

The claim under test here is narrower than E2's and closer to what the method
is actually for: not that SDFT beats replay, but that some alpha makes it a
usable **replay-free** substitute - competitive on the walk while storing no
past data (privacy) and paying a 1x teacher instead of replay's ~706x buffer
(see E4). If no alpha achieves that, the negative result is worth far more with
a curve behind it than with one point.

## The trap this file exists to avoid

`monitor.load_base` REBINDS `CONFIG["cl"]` from the checkpoint's `base_meta.json`
(which stores alpha=0.5), so editing `core_pipeline.CONFIG` does nothing - it is
overwritten on every base load. But `CLTFT.cl_cfg` holds a *reference* to
whatever dict is installed there and re-reads `cl_cfg["sdft_alpha"]` on every
training step. So the injection has to be an **in-place write, after load_base**.
Rebinding (`CONFIG["cl"] = {...}`) would leave the already-built forecaster
pointing at the old dict and silently sweep nothing at all - every cell would
run alpha=0.5 and the results would look like a flat, uninteresting curve.
`assert_alpha_applied` re-reads it off the model to make that failure loud.

## Why frozen is copied rather than re-run

`build_forgetting_table` differences each arm against an anchor's probes in the
same directory, and `frozen` never retrains - so it is invariant to alpha AND to
censoring. E2 already produced it for both levels. Copying is not a shortcut, it
avoids six identical walks.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from rich.console import Console  # noqa: E402

from drift_pipeline import base_training as bt  # noqa: E402
from drift_pipeline import metrics as mt  # noqa: E402
from drift_pipeline import monitor as mon  # noqa: E402
from drift_pipeline import retrain as rt  # noqa: E402
from drift_pipeline.censoring import (assert_uncensored,  # noqa: E402
                                      attach_unmet_demand, censor_data_dict)
from drift_pipeline.core_pipeline import CONFIG, prepare_drift_data  # noqa: E402
# Reused wholesale - same base paths, same per-cell results redirection, same
# GPU hygiene and resume semantics E2 runs on.
from experiments.exp_staleness_cl import (DEFAULT_BASE_CKPT,  # noqa: E402
                                          DEFAULT_DATA_DIR, base_paths,
                                          build_tables, copy_static_arm,
                                          free_gpu, results_dir, spec_for)

console = Console()

ARM = "sdft"
ANCHOR = "frozen"

#: 0.5 is deliberately absent: E2 already ran it at both levels, and those rows
#: are this sweep's baseline. Rising alpha = more task loss = more adaptation.
DEFAULT_ALPHAS = [0.7, 0.85, 0.95]

#: The control and the strongest treatment. `strong_lock` is a middle point the
#: alpha question does not need - it would cost 3 more cells to interpolate a
#: curve between two levels that already bracket the effect.
DEFAULT_CENSORING = ["none", "escrow_quota"]

#: Where E2's frozen anchor is read from.
E2_ROOT = PROJECT_ROOT / "outputs" / "drift" / "results" / "e2"


def alpha_tag(alpha: float) -> str:
    """Filesystem-safe, sorts correctly: 0.85 -> 'alpha_085'."""
    return f"alpha_{str(alpha).replace('.', '')}"


def assert_alpha_applied(base: Dict, alpha: float) -> None:
    """Read alpha back off the built model, not out of CONFIG.

    CONFIG agreeing with itself proves nothing - the whole failure mode is the
    forecaster holding a DIFFERENT dict than the one that was written to.
    """
    got = base["forecaster"].cl_cfg.get("sdft_alpha")
    if got != alpha:
        raise SystemExit(
            f"sdft_alpha did not reach the model: forecaster.cl_cfg has {got!r}, "
            f"expected {alpha!r}. CONFIG['cl'] was probably rebound rather than "
            f"mutated in place - see this module's docstring.")


def seed_anchor(cell_dir: Path, censoring: str) -> bool:
    """Copy E2's frozen artifacts in so the forgetting table has its anchor."""
    src = E2_ROOT / censoring
    if not (src / f"probe_scores_{ANCHOR}.json").exists():
        return False
    copy_static_arm(src, cell_dir, ANCHOR)
    return True


def run_cell(censoring: str, alpha: float, data: Dict, cell_dir: Path,
             ckpt_dir: Path, checkpoint_every: int = 0,
             save_model: bool = True) -> Dict:
    """One (censoring, alpha) cell. Fresh base per cell, so no state leaks."""
    started = time.perf_counter()

    free_gpu()
    paths = base_paths(ckpt_dir)
    base = mon.load_base(paths)

    # THE injection. In place, after load_base - see the module docstring.
    CONFIG["cl"]["sdft_alpha"] = alpha
    assert_alpha_applied(base, alpha)

    with results_dir(cell_dir):
        out = rt.run_arm(ARM, data, base=base, checkpoint_every=checkpoint_every)
        ctrl = out["controller"]
        scores = mt.score_model_on_probes(ctrl.forecaster, ctrl.pricer,
                                          data, ctrl.train_ds)
        mt.save_probe_scores(ARM, scores)
        n_fc = ctrl.stats.n_fc_retrains
        n_rl = ctrl.stats.n_rl_retrains

        # Written BEFORE the controller is dropped - this is the only moment the
        # walk's trained models exist. Lands beside the cell's metrics so one
        # results tarball carries both the numbers and the model they describe.
        if save_model:
            model_dir = bt.save_serving_checkpoint(
                ctrl.forecaster, ctrl.pricer,
                dst_dir=cell_dir / "model", src_ckpt_dir=ckpt_dir,
                calibration_path=paths["calibration"],
                provenance={"experiment": "e5", "arm": ARM,
                            "censoring": censoring, "sdft_alpha": alpha,
                            "n_fc_retrains": n_fc, "n_rl_retrains": n_rl})
            console.print(f"    [green]model[/green] -> {model_dir}")

    del ctrl, out, base
    free_gpu()
    return {"censoring": censoring, "sdft_alpha": alpha,
            "n_fc_retrains": n_fc, "n_rl_retrains": n_rl,
            "wall_seconds": time.perf_counter() - started}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="E5: SDFT alpha sweep.")
    ap.add_argument("--alphas", nargs="*", type=float, default=DEFAULT_ALPHAS)
    ap.add_argument("--censoring", nargs="*", default=DEFAULT_CENSORING)
    ap.add_argument("--scarcity-power", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="e5")
    ap.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    ap.add_argument("--base-ckpt", type=Path, default=DEFAULT_BASE_CKPT)
    ap.add_argument("--checkpoint-every", type=int,
                    default=int(os.environ.get("FYP_CHECKPOINT_EVERY", "20")))
    ap.add_argument("--max-checks", type=int, default=None,
                    help="truncate the walk - for smoke-testing the wiring only")
    ap.add_argument("--no-save-model", action="store_true",
                    help="skip writing each cell's trained model. By default a "
                         "loadable checkpoint dir (~7 MB) is written to "
                         "<cell>/model/ - the walk's models exist only at that "
                         "moment, and are otherwise dropped once probe-scored")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(argv)

    for name, path in (("data dir", args.data_dir), ("base ckpt", args.base_ckpt)):
        if not path.exists():
            console.print(f"[red]{name} not found:[/red] {path}")
            return 1
    CONFIG["paths"]["demand_csv"] = str(args.data_dir / "demand_forecasting.csv")
    CONFIG["paths"]["rl_csv"] = str(args.data_dir / "rl_environment.csv")
    CONFIG["seed"] = args.seed

    root = Path(CONFIG["paths"]["results"]) / args.out
    root.mkdir(parents=True, exist_ok=True)
    n_cells = len(args.censoring) * len(args.alphas)
    console.print(f"[bold]E5[/bold]: {len(args.censoring)} censoring x "
                  f"{len(args.alphas)} alpha = {n_cells} cells -> {root}")
    console.print(f"  alphas={args.alphas}  (0.5 is E2's baseline, not re-run)")

    # Anchor availability is checked BEFORE any GPU work: without it every cell
    # would run to completion and then fail to produce a forgetting number.
    missing = [c for c in args.censoring
               if not (E2_ROOT / c / f"probe_scores_{ANCHOR}.json").exists()]
    if missing:
        console.print(f"[red]no E2 '{ANCHOR}' anchor for: {missing}[/red]\n"
                      f"  expected {E2_ROOT}/<censoring>/probe_scores_{ANCHOR}.json\n"
                      f"  extract results.tar.gz first, or run E2's frozen arm.")
        return 1
    console.print(f"  [green]OK[/green] E2 {ANCHOR} anchor found for "
                  f"{', '.join(args.censoring)}")

    if args.dry_run:
        for c in args.censoring:
            for a in args.alphas:
                console.print(f"  would run {c} x alpha={a} -> "
                              f"{root / c / alpha_tag(a)}")
        console.print("\n[bold]Dry run[/bold] - no models loaded.")
        return 0

    data = prepare_drift_data()
    data["rl_full"] = attach_unmet_demand(data["rl_full"], CONFIG["paths"]["rl_csv"])
    if args.max_checks is not None:
        data["checks"] = list(data["checks"])[:args.max_checks]
        console.print(f"  [yellow]TRUNCATED to {args.max_checks} checks - "
                      f"smoke run, not comparable[/yellow]")

    rows: List[Dict] = []
    for censoring in args.censoring:
        console.print(f"\n[bold magenta]== censoring: {censoring} ==[/bold magenta]")
        spec = spec_for(censoring, seed=args.seed,
                        scarcity_power=args.scarcity_power)
        cdata = censor_data_dict(data, spec)
        assert_uncensored(cdata["tft_full"], cdata["rl_full"])
        console.print(f"  fill={cdata['censoring']['fill_rate']:.3f}")

        for alpha in args.alphas:
            cell_dir = root / censoring / alpha_tag(alpha)
            cell_dir.mkdir(parents=True, exist_ok=True)
            if (cell_dir / f"probe_scores_{ARM}.json").exists() and not args.force:
                console.print(f"  [dim]skip alpha={alpha} (already done)[/dim]")
            else:
                console.print(f"  [bold]{censoring} x alpha={alpha}[/bold] ...")
                row = run_cell(censoring, alpha, cdata, cell_dir, args.base_ckpt,
                               checkpoint_every=args.checkpoint_every,
                               save_model=not args.no_save_model)
                rows.append(row)
                console.print(f"    done in {row['wall_seconds'] / 60:.1f} min "
                              f"({row['n_fc_retrains']} FC retrains)")
            seed_anchor(cell_dir, censoring)
            build_tables(cell_dir, [ANCHOR, ARM], anchor=ANCHOR)

    # ── Roll-up ─────────────────────────────────────────────────────────────
    summary: List[Dict] = []
    for censoring in args.censoring:
        for alpha in args.alphas:
            eff_path = root / censoring / alpha_tag(alpha) / "metrics_efficiency.csv"
            if not eff_path.exists():
                continue
            eff = pd.read_csv(eff_path, index_col=0)
            if ARM not in eff.index:
                continue
            r = eff.loc[ARM]
            summary.append({
                "censoring": censoring, "sdft_alpha": alpha,
                **{k: r.get(k) for k in (
                    "forgetting_mase_base_era", "adaptation_mase_walk_era",
                    "walk_mase_mean", "walk_mase_median",
                    "profit_index_vs_frozen", "n_fc_retrains", "n_rl_retrains")},
            })

    if summary:
        df = pd.DataFrame(summary)
        out_csv = root / f"{args.out}_summary.csv"
        df.to_csv(out_csv, index=False)
        (root / f"{args.out}_config.json").write_text(json.dumps({
            "alphas": args.alphas, "censoring": args.censoring,
            "seed": args.seed, "data_dir": str(args.data_dir),
            "base_ckpt": str(args.base_ckpt),
            "anchor_from": str(E2_ROOT),
        }, indent=2, default=float))
        console.print(f"\n[green]->[/green] {out_csv}\n")
        console.print("[bold]walk_mase_mean[/bold] (replay~1.02, frozen=1.474 - "
                      "lower = adapted more)")
        console.print(df.pivot(index="censoring", columns="sdft_alpha",
                               values="walk_mase_mean").to_string())
        console.print("\n[bold]forgetting_mase_base_era[/bold] "
                      "(naive: 0.139 none / 0.053 escrow_quota)")
        console.print(df.pivot(index="censoring", columns="sdft_alpha",
                               values="forgetting_mase_base_era").to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
