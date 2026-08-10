"""
Pick the RL drift floor from data instead of asserting one.

    python -m experiments.analyze_rl_floor
    python -m experiments.analyze_rl_floor --stream outputs/drift/results/drift_stream_frozen.csv

The RL detector fires when `profit_index` sits below a floor. Choosing that
floor is a judgement call, so it should be made against the pricer's actual
operating band rather than by picking a round number - the flat 1.0 floor this
replaces looked reasonable precisely because normalisation puts the reference
mean at exactly 1.0, hiding that it sits at the CENTRE of the distribution.

Reads `frozen`'s walk, because a never-retraining arm's profit trace is the
closest thing to a no-treatment baseline: whatever it does is what the pricer
does when nothing intervenes. Accepts either a finished `drift_stream_*.csv` or
a partial `walk_state_*/stream.json`, so a killed run is still usable evidence.

What to look for: a floor below the normal operating band (roughly the 5-15%
quantile) that still fires often enough for RL continual learning to be a real
comparison. Too high and every arm retrains on noise; too low and the pricer
never retrains and the RL half of the experiment says nothing.
"""

from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from drift_pipeline.core_pipeline import CONFIG  # noqa: E402
from drift_pipeline.monitor import (rederive_triggers,  # noqa: E402
                                    rl_profit_floor)

FULL_WALK_CHECKS = 157

SEARCH = [
    "outputs/drift/results/e2/*/drift_stream_frozen.csv",
    "outputs/drift/results/e2/*/walk_state_frozen/stream.json",
    "outputs/drift/results/drift_stream_frozen.csv",
]


def load_stream(explicit: str | None):
    candidates = [explicit] if explicit else []
    for pattern in SEARCH:
        candidates.extend(sorted(glob.glob(str(PROJECT_ROOT / pattern))))
    for path in candidates:
        if not path or not Path(path).exists():
            continue
        if path.endswith(".json"):
            rows = json.loads(Path(path).read_text())["stream"]
        else:
            rows = pd.read_csv(path).to_dict("records")
        if rows:
            return rows, path
    return None, None


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Choose the RL drift floor from data.")
    ap.add_argument("--stream", default=None,
                    help="drift_stream_frozen.csv or walk_state_frozen/stream.json")
    ap.add_argument("--calibration",
                    default="outputs/drift/checkpoints/base_v4/calibration.json")
    args = ap.parse_args(argv)

    rows, src = load_stream(args.stream)
    if rows is None:
        print("no frozen stream found - run the frozen arm first:\n"
              "  python -m experiments.exp_staleness_cl --censoring none --arms frozen")
        return 1

    cal_path = PROJECT_ROOT / args.calibration
    if not cal_path.exists():
        print(f"calibration not found: {cal_path}")
        return 1
    cal = json.loads(cal_path.read_text())
    mu = cal["rl"]["ref_profit_mu"]
    sigma = cal["rl"].get("ref_profit_sigma")

    n = len(rows)
    pi = pd.to_numeric(pd.Series([r.get("profit_index") for r in rows]),
                       errors="coerce").dropna()

    print(f"stream      : {Path(src).relative_to(PROJECT_ROOT)}  ({n} checks"
          + (f", PARTIAL - full walk is {FULL_WALK_CHECKS}"
             if n < FULL_WALK_CHECKS else "") + ")")
    print(f"calibration : mu={mu:.2f} sigma={sigma}"
          + (f"  CV={sigma / abs(mu):.4f}" if sigma else "  (no sigma - flat floor)"))
    print()
    print(f"profit_index: mean={pi.mean():.3f} std={pi.std():.3f} "
          f"min={pi.min():.3f} max={pi.max():.3f}")
    for q in (0.05, 0.10, 0.25, 0.50):
        print(f"   {q:>4.0%} quantile: {pi.quantile(q):.3f}")
    print(f"   share below 1.0 (the old flat floor): {(pi < 1.0).mean():.1%}")
    print()

    cons = CONFIG["drift"]["rl_consecutive"]
    print(f"{'k':>5} {'floor':>8} {'triggers':>9} {'rate':>7} {'projected/157':>14}")
    for k in CONFIG["drift"].get("rl_k_sensitivity", [0.0, 0.5, 1.0, 1.5, 2.0, 2.5]):
        floor = rl_profit_floor(cal, {**CONFIG["drift"], "rl_k_sigma": k})
        fired = rederive_triggers(rows, mu=floor, sigma=0.0, k=0.0,
                                  consecutive=cons, metric="profit_index",
                                  direction="below")
        rate = len(fired) / n if n else 0.0
        print(f"{k:>5.1f} {floor:>8.4f} {len(fired):>9} {rate:>6.1%} "
              f"{round(rate * FULL_WALK_CHECKS):>14}")

    print(f"\nconsecutive={cons}. Under pure below-floor noise at rate p, a run of "
          f"{cons}\nfires about every {1 / 0.5 + 1 / 0.5 ** 2:.0f} checks when p=0.5 "
          f"(~{FULL_WALK_CHECKS / 6:.0f} over the walk) - a count near that is "
          f"indistinguishable\nfrom variance rather than evidence of drift.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
