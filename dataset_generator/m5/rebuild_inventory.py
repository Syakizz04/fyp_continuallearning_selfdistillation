"""
Regenerate ONLY the inventory columns of an existing rl_environment.csv.

    python -m dataset_generator.m5.rebuild_inventory \
        --src data/processed_m5_v2 --dst data/processed_m5_v3

## Why this exists instead of just rebuilding from raw M5

Rebuilding with `build_m5 --stores CA_1 --top-n-items 100` does **not** reproduce
the same 100 items: the run above came back with 182,557 rows against v2's
184,412, and a visibly different elasticity distribution (median -0.79 vs -0.60).
One item differed, and with it every elasticity that borrows strength from the
department pool.

That would wreck the experiment. The point of the new dataset is to change
**one** thing - whether the inventory signal is realistic - so that a pricing
agent trained on it can be compared against one trained on v2. If the item set
and the elasticities move at the same time, the comparison measures all three at
once and attributes the result to whichever we happened to name.

So this derives from v2 in place: every column except the inventory ones is
copied through untouched, guaranteeing a controlled comparison by construction
rather than by hoping a sampler is deterministic.

`demand_forecasting.csv` is copied byte-for-byte, so the existing base TFT
checkpoint stays valid and only the RL side changes.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from edge_system.inventory.replenishment import simulate_inventory  # noqa: E402


def rebuild(src: Path, dst: Path, *, cover_days: float, reorder_frac: float,
            lead_time_days: int, verbose: bool = True) -> pd.DataFrame:
    dst.mkdir(parents=True, exist_ok=True)

    rl = pd.read_csv(src / "rl_environment.csv", parse_dates=["date"])
    before_stockout = float(rl["stockout_flag"].mean())

    out = []
    for (item, store), g in rl.groupby(["product_id", "region_id"], sort=False):
        g = g.sort_values("date").copy()
        trace = simulate_inventory(
            g["realized_demand"].to_numpy(dtype=float),
            cover_days=cover_days, reorder_frac=reorder_frac,
            lead_time_days=lead_time_days,
        )
        g["inventory_level"] = trace.on_hand
        g["stockout_flag"] = trace.stockout
        g["unmet_demand"] = trace.unmet
        out.append(g)

    new = pd.concat(out).sort_index()
    new.to_csv(dst / "rl_environment.csv", index=False)

    # Copied, not regenerated: the forecasting side must stay identical or the
    # existing base TFT checkpoint silently stops matching its training data.
    shutil.copy2(src / "demand_forecasting.csv", dst / "demand_forecasting.csv")
    for extra in ("elasticity_report.csv",):
        if (src / extra).exists():
            shutil.copy2(src / extra, dst / extra)

    if verbose:
        served = new["realized_demand"] - new["unmet_demand"]
        fill = served.sum() / max(new["realized_demand"].sum(), 1e-9)
        print(f"  rows            : {len(new):,} (unchanged: {len(new) == len(rl)})")
        print(f"  stockout rate   : {before_stockout:.4%} -> "
              f"{new['stockout_flag'].mean():.4%}")
        print(f"  fill rate       : {fill:.2%}")
        print(f"  censored units  : {new['unmet_demand'].sum():,.0f} "
              f"({new['unmet_demand'].sum() / max(new['realized_demand'].sum(), 1e-9):.2%} "
              f"of demand)")
        print(f"  inventory CV    : "
              f"{rl['inventory_level'].std() / rl['inventory_level'].mean():.3f} -> "
              f"{new['inventory_level'].std() / new['inventory_level'].mean():.3f}")
        # The control: everything else must be untouched.
        untouched = [c for c in rl.columns
                     if c not in ("inventory_level", "stockout_flag")]
        same = all(rl[c].equals(new[c]) for c in untouched)
        print(f"  other columns identical to source: {same}")
        if not same:
            raise SystemExit("REFUSING: a column other than inventory changed.")
    return new


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--src", default="data/processed_m5_v2")
    ap.add_argument("--dst", default="data/processed_m5_v3")
    ap.add_argument("--cover-days", type=float, default=10.0)
    ap.add_argument("--reorder-frac", type=float, default=0.5)
    ap.add_argument("--lead-time-days", type=int, default=3)
    args = ap.parse_args(argv)

    print(f"Rebuilding inventory: {args.src} -> {args.dst}")
    print(f"  policy: cover={args.cover_days}d reorder={args.reorder_frac} "
          f"lead={args.lead_time_days}d")
    rebuild(Path(args.src), Path(args.dst), cover_days=args.cover_days,
            reorder_frac=args.reorder_frac, lead_time_days=args.lead_time_days)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
