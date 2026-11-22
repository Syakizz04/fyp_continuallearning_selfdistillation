"""
Read seed replicates as PAIRED deltas, and say whether a gap is real.

    python -m experiments.compare_replicates
    python -m experiments.compare_replicates --censoring none --reference naive

The comparison has to be paired within seed. Arms sharing a seed share the same
censoring draw and the same per-retrain seeding, so a seed's idiosyncrasies
cancel in the difference; comparing arm means across seeds would instead pool
the very variance the replicates exist to measure.

The verdict is deliberately conservative. A gap counts as real only if its SIGN
is the same in every seed AND its smallest magnitude exceeds the spread of the
deltas themselves. With three seeds that is a weak test - it is meant to stop
you claiming a difference that is not there, not to certify one that is.

This rule is pre-registered for the post-rebuild E2 re-run: see
experiments/PREREGISTRATION_E2.md. It asks for 5 seeds, the primary metric
forgetting_mase_base_era, and the primary contrast sdft - replay. The mean
delta, t-based 95% interval and sign-test p printed beside each verdict are
DESCRIPTIVE. The verdict is the sign + spread rule alone.
"""

from __future__ import annotations

import argparse
import glob
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from rich.console import Console  # noqa: E402

console = Console()

#: Lower is better for both, so a NEGATIVE delta means the arm beat the reference.
METRICS = ["forgetting_mase_base_era", "walk_mase_mean"]
#: PREREGISTRATION_E2.md: fewer seeds than this is not the pre-registered run.
MIN_SEEDS = 5


def describe(d: pd.Series) -> str:
    """Mean delta, t-based 95% CI over seeds, exact two-sided sign-test p."""
    n = len(d)
    mean = float(d.mean())
    if n > 1:
        half = float(stats.t.ppf(0.975, n - 1) * d.std(ddof=1) / np.sqrt(n))
        ci = f"[{mean - half:+.4f}, {mean + half:+.4f}]"
    else:
        ci = "[n/a]"
    pos, neg = int((d > 0).sum()), int((d < 0).sum())
    p = float(stats.binomtest(min(pos, neg), pos + neg, 0.5).pvalue) if pos + neg else 1.0
    return f"mean={mean:+.4f} ci95={ci} sign-p={p:.3f}"


def load(censoring: str) -> pd.DataFrame:
    rows = []
    for eff in sorted(glob.glob(str(PROJECT_ROOT / "outputs" / "drift" / "results"
                                    / "rep_s*" / censoring / "metrics_efficiency.csv"))):
        seed = Path(eff).parents[1].name.replace("rep_s", "")
        df = pd.read_csv(eff, index_col=0)
        for arm, r in df.iterrows():
            rows.append({"seed": seed, "arm": arm,
                         **{m: r.get(m) for m in METRICS if m in df.columns}})
    return pd.DataFrame(rows)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Paired seed-replicate comparison.")
    ap.add_argument("--censoring", default="escrow_quota")
    ap.add_argument("--reference", default="replay",
                    help="arm every other arm is differenced against")
    args = ap.parse_args(argv)

    df = load(args.censoring)
    if df.empty:
        console.print(f"[red]no replicates found[/red] for censoring={args.censoring}\n"
                      f"  expected outputs/drift/results/rep_s<seed>/{args.censoring}/"
                      f"metrics_efficiency.csv\n  run: bash experiments/run_replicates.sh")
        return 1

    seeds = sorted(df.seed.unique())
    console.print(f"[bold]Replicates[/bold] censoring={args.censoring}  "
                  f"seeds={seeds}  arms={sorted(df.arm.unique())}\n")
    if len(seeds) < MIN_SEEDS:
        console.print(f"[yellow]only {len(seeds)} seed(s); the pre-registered run "
                      f"(experiments/PREREGISTRATION_E2.md) uses {MIN_SEEDS}. "
                      f"Treat verdicts as provisional.[/yellow]\n")

    for metric in METRICS:
        if metric not in df.columns:
            continue
        piv = df.pivot(index="seed", columns="arm", values=metric)
        console.print(f"[bold]{metric}[/bold] (lower is better)")
        console.print(piv.round(4).to_string())

        # frozen never trains: identical across seeds is the sanity check.
        if "frozen" in piv.columns:
            spread = float(piv["frozen"].max() - piv["frozen"].min())
            ok = "[green]OK[/green]" if spread < 1e-6 else "[red]MOVED[/red]"
            console.print(f"  frozen spread across seeds: {spread:.3e}  {ok}")

        if args.reference in piv.columns:
            console.print(f"\n  paired deltas vs [bold]{args.reference}[/bold] "
                          f"(negative = beats it)")
            for arm in [c for c in piv.columns if c not in (args.reference, "frozen")]:
                d = (piv[arm] - piv[args.reference]).dropna()
                if d.empty:
                    continue
                same_sign = bool((d > 0).all() or (d < 0).all())
                spread = float(d.max() - d.min())
                decisive = same_sign and float(d.abs().min()) > spread
                verdict = ("[green]consistent[/green]" if decisive else
                           "[yellow]not separable[/yellow]")
                console.print(f"    {arm:8s} " +
                              "  ".join(f"{s}:{v:+.4f}" for s, v in d.items()) +
                              f"   spread={spread:.4f}  {verdict}")
                console.print(f"             [dim]{describe(d)}[/dim]")
        console.print()

    console.print("[dim]'not separable' means the arms cannot be ranked at this "
                  "sample size - a real finding, not a failed run.[/dim]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
