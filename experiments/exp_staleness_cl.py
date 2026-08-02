"""
E2 - demand censoring as a drift source. The headline experiment.

    python -m experiments.exp_staleness_cl --dry-run          # plan only, no GPU
    python -m experiments.exp_staleness_cl                    # full 9-cell sweep
    python -m experiments.exp_staleness_cl --censoring none escrow_quota
    python -m experiments.exp_staleness_cl --arms frozen sdft --force

Sweeps `censoring x CL arm` and asks whether replay-free SDFT degrades more
gracefully than replay when the training signal itself is corrupted by the
inventory sync layer.

## The mechanism

E1 measured what each sync policy costs as a fill rate: strong_lock 82.7%,
escrow_quota 71.7%. The refused share is demand that existed and was never
recorded. Retrain a forecaster on what the node observed and it learns the
product is less wanted than it is - a systematically negative bias whose severity
is set by the sync policy. So E1's independent variable *is* E2's treatment; the
two experiments measure one quantity from two sides rather than being separate
stories.

## The claim under test

This is a continual-learning experiment; the system is the apparatus that
generates the treatment, not the object of study. What makes the setting unusual
in CL terms is that the label corruption is **endogenous** - the deployed
learner produces it, and it feeds back. Under-forecast, and the node orders less;
order less, and more demand goes unserved; more unserved demand, and the next
retrain sees an even more understated target. Noisy-label CL generally assumes
the noise arrives from outside.

The mechanism claim: replay stores past windows and keeps re-teaching them, so
under censoring the buffer preserves the *old* bias long after it was recorded.
SDFT distils from the current teacher and stores no past data, so it has nothing
stale to re-teach. If that is right, the sdft-vs-replay gap should WIDEN as the
fill rate falls.

Four arms, because two different controls are needed to make that claim
attributable:

* `frozen`  - the ANCHOR. Never retrains, so it never ingests censored data.
              Every delta is measured against it, and any arm that does worse
              has been actively harmed by its own training.
* `naive`   - the CONTROL. Same drift trigger as the CL arms, no CL mechanism.
              Without it, an arm that beats frozen cannot say whether the win
              came from its mechanism or merely from retraining at the right
              moment.
* `replay`  - the incumbent SDFT's replay-free claim is made against.
* `sdft`    - the proposed method.

Read the result on **forgetting**, not on mean walk error: accuracy over the walk
confounds adapting to the new regime with keeping what was known about the old
one, and the whole question here is the second.

## The one rule this file enforces

**Train on censored demand, evaluate against true demand.** `RetrainController`
reads `tft_*_censored` for training while the monitor scores against `tft_full`,
and `assert_uncensored` re-checks the evaluation frame at every level. Censoring
both sides would let a model look accurate by faithfully reproducing its own
bias.

## Why each cell gets its own results directory

Every artifact `run_arm` writes is keyed by arm alone - `drift_stream_sdft.csv`,
`memory_sdft.csv`. Running one arm at three censoring levels into one directory
would silently overwrite, leaving three cells' worth of GPU time represented by
whichever finished last. `CONFIG["paths"]["results"]` is repointed per cell
instead, which also makes the sweep resumable: a cell whose probe scores already
exist on disk is complete and gets skipped.
"""

from __future__ import annotations

import argparse
import gc
import json
import shutil
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from rich.console import Console  # noqa: E402

from drift_pipeline import metrics as mt  # noqa: E402
from drift_pipeline import monitor as mon  # noqa: E402
from drift_pipeline import retrain as rt  # noqa: E402
from drift_pipeline.censoring import (CensoringSpec, assert_uncensored,  # noqa: E402
                                      attach_unmet_demand, censor_data_dict)
from drift_pipeline.core_pipeline import CONFIG, prepare_drift_data  # noqa: E402

console = Console()

#: The treatment axis. `none` is the control; `intrinsic` is the floor a
#: *perfectly* synchronised system would still hit (the dataset's own stockouts),
#: so it separates "censoring hurts" from "this policy's censoring hurts"; the
#: two policy names take their fill rates from E1.
CENSORING = ["none", "intrinsic", "strong_lock", "escrow_quota"]
DEFAULT_CENSORING = ["none", "strong_lock", "escrow_quota"]

#: EWC is dropped. It under-adapted in the first run - it is a regulariser that
#: pulls toward the base weights, which is the wrong response to a real regime
#: change - and at ~1.3 h per cell it would cost a quarter of the sweep to
#: re-confirm that. See the docstring for why the other four all earn their place.
ARMS = ["frozen", "naive", "replay", "sdft"]

#: The column the sweep is read on. Forgetting, not accuracy: this is a continual
#: learning experiment, and mean walk error confounds "adapted well to the new
#: regime" with "kept what it knew about the old one". `forgetting_mase_base_era`
#: separates them - it is each arm's final model re-scored on the BASE-era probe
#: windows, differenced against frozen, so >0 means the arm lost pre-drift
#: knowledge it once had.
HEADLINE = "forgetting_mase_base_era"

#: Arms that never retrain are censoring-invariant: they only ever run the base
#: model, so their result is identical in every cell of a row. Running one and
#: copying is not a shortcut, it is avoiding three identical hours of GPU.
STATIC_ARMS = {"frozen"}

#: E2 runs on v4 + base_cover, NOT on drift_pipeline's defaults (processed_m5 +
#: `base`), and the difference is not cosmetic:
#:
#: * v3 regenerated ONLY the inventory columns, taking stockout days from 0.05%
#:   to 5.1%. `unmet_demand` is the censoring signal, so on the default dataset
#:   the intrinsic floor is ~0 and `mode="intrinsic"` would be an empty
#:   treatment indistinguishable from the control.
#: * v4 rebuilt the two pricing features that failed an audit: `competitor_price`
#:   is now a real cross-store price for the same item rather than the focal
#:   store's own department median, and `demand_forecast` is the base TFT's
#:   1-step-ahead prediction rather than a 28-day rolling mean. Inventory columns
#:   and elasticity carry over from v3 bit-identical, so E1 still describes it.
#: * `demand_forecasting.csv` is byte-identical across every version, so the TFT
#:   side is unaffected - only the RL frame changes.
#: * `base_cover` is the pricer retrained against realistic lead-time inventory
#:   and is what `edge_system` serves. Its PPO checkpoint must match the 9-dim
#:   observation, so it is retrained whenever the state vector changes.
#:
#: `base_cover` carries its own calibration.json, so the drift thresholds come
#: from the matching base rather than from the default one.
DEFAULT_DATA_DIR = PROJECT_ROOT / "data" / "processed_m5_v4"
DEFAULT_BASE_CKPT = PROJECT_ROOT / "outputs" / "drift" / "checkpoints" / "base_v4"


def base_paths(ckpt_dir: Path) -> Dict[str, str]:
    """Explicit paths for `load_base`, which otherwise hardcodes `checkpoints/base`."""
    return {
        "tft": str(ckpt_dir / "base_tft.ckpt"),
        "dataset": str(ckpt_dir / "base_tft_dataset.pkl"),
        "ppo": str(ckpt_dir / "base_ppo.zip"),
        "meta": str(ckpt_dir / "base_meta.json"),
        "calibration": str(ckpt_dir / "calibration.json"),
    }


def spec_for(name: str, *, seed: int, scarcity_power: float = 1.0) -> CensoringSpec:
    if name == "none":
        return CensoringSpec(mode="none", seed=seed)
    if name == "intrinsic":
        return CensoringSpec(mode="intrinsic", seed=seed,
                             scarcity_power=scarcity_power)
    return CensoringSpec.for_policy(name, seed=seed,
                                    scarcity_power=scarcity_power)


@contextmanager
def results_dir(path: Path):
    """Repoint CONFIG['paths']['results'] for the duration of a cell.

    CONFIG is mutated in place rather than subclassed (the project-wide
    convention) and every helper re-reads it at call time, so this is enough to
    redirect `save_walk`, the retrain log, the memory log and the probe scores
    together. Restored in a `finally` because `mon.load_base` reads
    `calibration.json` from this same key and must see the canonical directory.
    """
    previous = CONFIG["paths"]["results"]
    CONFIG["paths"]["results"] = str(path)
    Path(path).mkdir(parents=True, exist_ok=True)
    try:
        yield Path(path)
    finally:
        CONFIG["paths"]["results"] = previous


def free_gpu() -> None:
    """Between cells. Each arm holds a TFT, a PPO model and its CL state; on a
    4 GB card the next cell's base load fails if the previous one is still resident."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()


def cell_is_done(cell_dir: Path, arm: str) -> bool:
    """Probe scores are written last, so their presence means the cell finished.

    Keying resume off `drift_stream_*.csv` would treat a run killed during
    probe-scoring as complete and silently drop that arm from the forgetting table.
    """
    return (cell_dir / f"probe_scores_{arm}.json").exists()


def load_probe_scores(cell_dir: Path, arms: List[str]) -> Dict[str, List[Dict]]:
    out = {}
    for arm in arms:
        p = cell_dir / f"probe_scores_{arm}.json"
        if p.exists():
            out[arm] = json.loads(p.read_text())["probes"]
    return out


def run_cell(censoring: str, arm: str, data: Dict, cell_dir: Path,
             ckpt_dir: Path) -> Dict:
    """One (censoring, arm) cell. Loads a FRESH base so no state leaks between cells."""
    started = time.perf_counter()

    # Loaded BEFORE the results directory is repointed, and with explicit paths
    # so the calibration comes from the matching base checkpoint set.
    free_gpu()
    base = mon.load_base(base_paths(ckpt_dir))

    with results_dir(cell_dir):
        out = rt.run_arm(arm, data, base=base)
        ctrl = out["controller"]

        # Probe-score immediately and drop the controller, rather than holding
        # every arm's models live until the end as `run_phase5` does. Nine cells
        # of resident TFT+PPO does not fit on a 4 GB card.
        scores = mt.score_model_on_probes(ctrl.forecaster, ctrl.pricer,
                                          data, ctrl.train_ds)
        mt.save_probe_scores(arm, scores)

        n_fc = ctrl.stats.n_fc_retrains
        n_rl = ctrl.stats.n_rl_retrains
        peak = ctrl.memlog.summary()

    cl_peak_mb = float("nan")
    if not peak.empty:
        sel = peak.loc[peak["component"] == "cl_state_total", "peak_mb"]
        if len(sel):
            cl_peak_mb = float(sel.iloc[0])

    del ctrl, out, base
    free_gpu()

    return {
        "censoring": censoring, "arm": arm,
        "n_fc_retrains": n_fc, "n_rl_retrains": n_rl,
        "cl_state_peak_mb": cl_peak_mb,
        "wall_seconds": time.perf_counter() - started,
    }


def copy_static_arm(src: Path, dst: Path, arm: str) -> None:
    """Reuse a never-retraining arm's artifacts at another censoring level."""
    for pattern in (f"*_{arm}.csv", f"*_{arm}.json"):
        for path in src.glob(pattern):
            shutil.copy2(path, dst / path.name)


def build_tables(cell_dir: Path, arms: List[str], anchor: str = "frozen") -> Optional[pd.DataFrame]:
    """Rebuild the accuracy / forgetting / efficiency tables from what is on disk.

    Reads rather than reusing in-memory results so a resumed sweep produces the
    same tables as an uninterrupted one.
    """
    with results_dir(cell_dir):
        probe_by_arm = load_probe_scores(cell_dir, arms)
        if anchor not in probe_by_arm:
            console.print(f"  [yellow]no '{anchor}' probes in {cell_dir.name}; "
                          f"skipping tables[/yellow]")
            return None
        streams = mt.load_streams(list(probe_by_arm))
        accuracy = mt.build_accuracy_table(streams, reference=anchor)
        forgetting = mt.build_forgetting_table(probe_by_arm, anchor=anchor)
        efficiency = mt.build_efficiency_table(
            mt.load_retrain_logs(list(probe_by_arm)), accuracy, forgetting)
        accuracy.to_csv(cell_dir / "metrics_accuracy.csv")
        forgetting.to_csv(cell_dir / "metrics_forgetting.csv")
        efficiency.to_csv(cell_dir / "metrics_efficiency.csv")
    return efficiency


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="E2: censoring x CL arm sweep.")
    ap.add_argument("--censoring", nargs="*", default=DEFAULT_CENSORING,
                    choices=CENSORING)
    ap.add_argument("--arms", nargs="*", default=ARMS)
    ap.add_argument("--scarcity-power", type=float, default=1.0,
                    help="how sharply extra refusals land on days when stock is "
                         "tight. 0 = uniform, which would be both unrealistic and "
                         "EASIER for a model to shrug off")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="e2")
    ap.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR,
                    help="v3 by default: it is the only version whose stockout "
                         "rate (5.1%% of days) makes censoring measurable")
    ap.add_argument("--base-ckpt", type=Path, default=DEFAULT_BASE_CKPT,
                    help="base_cover by default: the pricer trained against "
                         "realistic inventory, which is what edge_system serves")
    ap.add_argument("--max-checks", type=int, default=None,
                    help="truncate the walk to the first N weekly checks. For "
                         "smoke-testing the wiring only - a short walk sees fewer "
                         "regimes and its numbers are not comparable to a full run")
    ap.add_argument("--force", action="store_true",
                    help="re-run cells that already have results on disk")
    ap.add_argument("--no-reuse-static", action="store_true",
                    help="actually re-run 'frozen' at every censoring level "
                         "instead of copying it. It cannot differ - it never "
                         "trains - so this is only for verifying that claim")
    ap.add_argument("--dry-run", action="store_true",
                    help="report the plan and the censoring each level achieves, "
                         "then stop. No models are loaded")
    args = ap.parse_args(argv)

    for name, path in (("data dir", args.data_dir), ("base ckpt", args.base_ckpt)):
        if not path.exists():
            console.print(f"[red]{name} not found:[/red] {path}")
            return 1
    CONFIG["paths"]["demand_csv"] = str(args.data_dir / "demand_forecasting.csv")
    CONFIG["paths"]["rl_csv"] = str(args.data_dir / "rl_environment.csv")

    root = Path(CONFIG["paths"]["results"]) / args.out
    root.mkdir(parents=True, exist_ok=True)

    console.print(f"[bold]E2[/bold]: {len(args.censoring)} censoring x "
                  f"{len(args.arms)} arms = {len(args.censoring) * len(args.arms)} cells "
                  f"-> {root}")
    console.print(f"  data={args.data_dir.name}  base={args.base_ckpt.name}")

    # Prepared BEFORE any base is loaded: `load_base` overwrites
    # CONFIG["forecasting"] with the base's persisted (already column-pruned)
    # copy, and `prepare_drift_data` would then fail looking for region_id.
    data = prepare_drift_data()
    data["rl_full"] = attach_unmet_demand(data["rl_full"], CONFIG["paths"]["rl_csv"])
    if args.max_checks is not None:
        data["checks"] = list(data["checks"])[:args.max_checks]
    console.print(f"  data: {len(data['tft_full']):,} TFT rows, "
                  f"{len(data['checks'])} weekly checks"
                  + (" [yellow](TRUNCATED - smoke run)[/yellow]"
                     if args.max_checks is not None else ""))

    rows: List[Dict] = []
    reports: Dict[str, Dict] = {}
    first_static: Dict[str, Path] = {}

    for censoring in args.censoring:
        console.print(f"\n[bold magenta]== censoring: {censoring} ==[/bold magenta]")
        spec = spec_for(censoring, seed=args.seed,
                        scarcity_power=args.scarcity_power)
        cell_dir = root / censoring
        cell_dir.mkdir(parents=True, exist_ok=True)

        cdata = censor_data_dict(data, spec)
        report = cdata["censoring"]
        reports[censoring] = report

        # The yardstick must stay true even when the training frames do not.
        assert_uncensored(cdata["tft_full"], cdata["rl_full"])

        if report.get("target_reachable") is False:
            console.print(
                f"  [yellow]{censoring}: asked for fill "
                f"{report['fill_rate_requested']:.3f}, floor is "
                f"{report['fill_rate']:.3f}[/yellow] - the dataset's own stockouts "
                f"already censor more than this, so the cell is NOT the treatment "
                f"it is named after")

        console.print(f"  fill={report['fill_rate']:.3f} "
                      f"censored={report['units_censored']:,.0f} units "
                      f"({report['censored_share']:.1%}) "
                      f"rows_hit={report['rows_affected_share']:.1%} "
                      f"rel_bias={report['mean_relative_bias']:.3f}")
        (cell_dir / "censoring_report.json").write_text(
            json.dumps({**report, "scarcity_power": spec.scarcity_power,
                        "seed": spec.seed}, indent=2, default=float))
        # Makes the cell directory self-describing for the dashboard, which
        # otherwise has to know which base checkpoint set the thresholds came from.
        cal = args.base_ckpt / "calibration.json"
        if cal.exists():
            shutil.copy2(cal, cell_dir / "calibration.json")

        if args.dry_run:
            continue

        for arm in args.arms:
            if cell_is_done(cell_dir, arm) and not args.force:
                console.print(f"  [dim]skip {arm} (already done)[/dim]")
                continue

            reuse = (arm in STATIC_ARMS and not args.no_reuse_static
                     and arm in first_static)
            if reuse:
                console.print(f"  [dim]copy {arm} from "
                              f"{first_static[arm].name} (never retrains)[/dim]")
                copy_static_arm(first_static[arm], cell_dir, arm)
                continue

            console.print(f"  [bold]{censoring} x {arm}[/bold] ...")
            row = run_cell(censoring, arm, cdata, cell_dir, args.base_ckpt)
            rows.append(row)
            console.print(f"    done in {row['wall_seconds'] / 60:.1f} min "
                          f"({row['n_fc_retrains']} FC + {row['n_rl_retrains']} RL retrains)")
            if arm in STATIC_ARMS:
                first_static[arm] = cell_dir

        eff = build_tables(cell_dir, args.arms)
        if eff is not None:
            cols = [c for c in (HEADLINE, "adaptation_mase_walk_era",
                                "walk_mase_mean", "n_fc_retrains")
                    if c in eff.columns]
            console.print(eff[cols].to_string())

    if args.dry_run:
        console.print("\n[bold]Dry run[/bold] - censoring levels resolved above, "
                      "no models loaded.")
        return 0

    # ── Roll-up: one long-form row per (censoring, arm) ─────────────────────
    summary: List[Dict] = []
    for censoring in args.censoring:
        cell_dir = root / censoring
        eff_path = cell_dir / "metrics_efficiency.csv"
        if not eff_path.exists():
            continue
        eff = pd.read_csv(eff_path, index_col=0)
        rep = reports.get(censoring, {})
        for arm, r in eff.iterrows():
            summary.append({
                "censoring": censoring,
                "fill_rate": rep.get("fill_rate"),
                "censored_share": rep.get("censored_share"),
                "arm": arm,
                # Forgetting first: it is what the experiment is about, and
                # column order is what a reader skims.
                **{k: r.get(k) for k in (
                    "forgetting_mase_base_era", "adaptation_mase_walk_era",
                    "walk_mase_mean", "walk_mase_median", "walk_smape_mean",
                    "profit_index_vs_frozen", "n_fc_retrains", "n_rl_retrains",
                    "fc_epochs_total", "rl_steps_total")},
            })

    if summary:
        df = pd.DataFrame(summary)
        out_csv = root / f"{args.out}_summary.csv"
        df.to_csv(out_csv, index=False)
        (root / f"{args.out}_config.json").write_text(json.dumps({
            "censoring": args.censoring, "arms": args.arms,
            "scarcity_power": args.scarcity_power, "seed": args.seed,
            "data_dir": str(args.data_dir), "base_ckpt": str(args.base_ckpt),
            "reports": reports,
        }, indent=2, default=float))
        console.print(f"\n[green]->[/green] {out_csv}\n")
        console.print(f"[bold]{HEADLINE}[/bold] "
                      f"(>0 = lost pre-drift knowledge; frozen is the anchor, "
                      f"so its row is 0 by construction)")
        console.print(df.pivot(index="censoring", columns="arm",
                               values=HEADLINE).to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
