"""
pipeline.py
-----------
Master orchestrator for the synthetic dataset generation pipeline.

Usage:
    python pipeline.py                  # Full pipeline (Stage 1 + Stage 2)
    python pipeline.py --stage1-only    # Demand forecasting dataset only
    python pipeline.py --validate       # Run after generation to produce plots

Pipeline stages:
    Stage 1: Generate demand_forecasting.csv
    Stage 2: Generate rl_environment.csv (reads Stage 1 output)
    Validate: Generate diagnostic plots
"""

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

ROOT = Path(__file__).parent
OUTPUT_DIR = ROOT / "output"
OUTPUT_DIR.mkdir(exist_ok=True)

DEMAND_CSV = OUTPUT_DIR / "demand_forecasting.csv"
RL_CSV     = OUTPUT_DIR / "rl_environment.csv"


def load_config() -> dict:
    with open(ROOT / "config" / "config.yaml", "r") as f:
        return yaml.safe_load(f)


def print_header(title: str):
    print("\n" + "=" * 60)
    print(f"  {title}")
    print("=" * 60)


# ---------------------------------------------------------------------------
# STAGE 1: Demand Forecasting Dataset
# ---------------------------------------------------------------------------

def run_stage1(config: dict, rng: np.random.Generator) -> pd.DataFrame:
    print_header("STAGE 1 — Demand Forecasting Dataset")

    from generators.calendar import build_calendar_features
    from generators.demand   import generate_demand, _build_viral_shock_map
    from generators.features import build_all_features

    start_date = config["simulation"]["start_date"]
    end_date   = config["simulation"]["end_date"]
    date_range = pd.date_range(start=start_date, end=end_date, freq="D")

    print(f"\n  Date range : {start_date} → {end_date} ({len(date_range)} days)")
    print(f"  SKUs       : {len(config['skus'])}")
    print(f"  Regions    : {len(config['regions'])}")
    print(f"  Total rows : {len(date_range) * len(config['skus']) * len(config['regions']):,}")

    # Build calendar once (shared across all SKUs and regions)
    print("\n  Building Malaysian calendar features...")
    calendar_df = build_calendar_features(date_range)

    # Build viral shock map once
    viral_shock_map = _build_viral_shock_map(date_range)
    print(f"  Viral shock events loaded: {len(set(k[0] for k in viral_shock_map)):} shock-days")

    all_frames = []
    total = len(config["skus"]) * len(config["regions"])
    count = 0

    for sku_cfg in config["skus"]:
        for region_cfg in config["regions"]:
            count += 1
            print(f"  [{count}/{total}] Generating demand: {sku_cfg['id']} × {region_cfg['id']}...", end=" ")
            t0 = time.time()

            # Generate raw demand signal
            demand_df = generate_demand(
                date_range=date_range,
                calendar_df=calendar_df.copy(),
                sku_config=sku_cfg,
                region_config=region_cfg,
                seasonality_config=config["seasonality"],
                payday_config=config["payday"],
                rng=rng,
                viral_shock_map=viral_shock_map,
            )

            # Add all derived features
            demand_df = build_all_features(
                df=demand_df,
                sku_config=sku_cfg,
                task_configs=config["cl_tasks"],
                rng=rng,
            )

            all_frames.append(demand_df)
            print(f"done ({time.time()-t0:.1f}s, {len(demand_df):,} rows)")

    full_df = pd.concat(all_frames, ignore_index=True)
    full_df = full_df.sort_values(["product_id", "region_id", "date"]).reset_index(drop=True)

    # Save
    print(f"\n  Saving to {DEMAND_CSV}...")
    full_df.to_csv(DEMAND_CSV, index=False)
    print(f"  Saved: {len(full_df):,} rows × {len(full_df.columns)} columns")
    print(f"  File size: {DEMAND_CSV.stat().st_size / 1024:.1f} KB")

    return full_df


# ---------------------------------------------------------------------------
# STAGE 2: RL Environment Dataset
# ---------------------------------------------------------------------------

def run_stage2(config: dict, rng: np.random.Generator, demand_df: pd.DataFrame = None) -> pd.DataFrame:
    print_header("STAGE 2 — RL Environment Dataset")

    from generators.rl_environment import generate_rl_environment

    if demand_df is None:
        if not DEMAND_CSV.exists():
            raise FileNotFoundError(
                f"Stage 1 output not found at {DEMAND_CSV}. "
                "Run Stage 1 first: python pipeline.py --stage1-only"
            )
        print(f"\n  Loading Stage 1 output from {DEMAND_CSV}...")
        demand_df = pd.read_csv(DEMAND_CSV, parse_dates=["date"])
        print(f"  Loaded: {len(demand_df):,} rows")

    print(f"\n  Generating RL environment for {len(demand_df):,} state transitions...")
    t0 = time.time()

    rl_df = generate_rl_environment(
        demand_df=demand_df,
        sku_configs=config["skus"],
        region_configs=config["regions"],
        competitor_config=config["competitor"],
        inventory_config=config["inventory"],
        rng=rng,
    )

    print(f"  Generated in {time.time()-t0:.1f}s")
    print(f"\n  Saving to {RL_CSV}...")
    rl_df.to_csv(RL_CSV, index=False)
    print(f"  Saved: {len(rl_df):,} rows × {len(rl_df.columns)} columns")
    print(f"  File size: {RL_CSV.stat().st_size / 1024:.1f} KB")

    return rl_df


# ---------------------------------------------------------------------------
# VALIDATION
# ---------------------------------------------------------------------------

def run_validation():
    print_header("VALIDATION — Generating Diagnostic Plots")
    import sys
    sys.path.insert(0, str(ROOT))
    from validate.visualise import main as vis_main
    vis_main()


# ---------------------------------------------------------------------------
# DATASET COLUMN MANIFEST
# ---------------------------------------------------------------------------

def print_column_manifest(demand_df: pd.DataFrame, rl_df: pd.DataFrame):
    print_header("COLUMN MANIFEST")

    print("\n  DEMAND FORECASTING DATASET columns:")
    for i, col in enumerate(demand_df.columns, 1):
        dtype = str(demand_df[col].dtype)
        null_count = demand_df[col].isnull().sum()
        print(f"    {i:3d}. {col:<45} [{dtype}] nulls={null_count}")

    print(f"\n  RL ENVIRONMENT DATASET — additional columns over Stage 1:")
    stage1_cols = set(demand_df.columns)
    for i, col in enumerate(rl_df.columns, 1):
        if col not in stage1_cols:
            dtype = str(rl_df[col].dtype)
            null_count = rl_df[col].isnull().sum()
            print(f"    {i:3d}. {col:<45} [{dtype}] nulls={null_count}")


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Synthetic Malaysian E-Commerce Dataset Generator"
    )
    parser.add_argument("--stage1-only", action="store_true",
                        help="Generate only the demand forecasting dataset")
    parser.add_argument("--stage2-only", action="store_true",
                        help="Generate only the RL dataset (Stage 1 must exist)")
    parser.add_argument("--validate", action="store_true",
                        help="Run validation plots only (both CSVs must exist)")
    parser.add_argument("--manifest", action="store_true",
                        help="Print column manifest after generation")
    args = parser.parse_args()

    config = load_config()
    seed   = config["simulation"]["random_seed"]
    rng    = np.random.default_rng(seed)

    print_header("SYNTHETIC MALAYSIAN E-COMMERCE DATASET GENERATOR")
    print(f"\n  Config loaded")
    print(f"  Random seed : {seed}")
    print(f"  Period      : {config['simulation']['start_date']} → {config['simulation']['end_date']}")
    print(f"  SKUs        : {[s['id'] for s in config['skus']]}")
    print(f"  Regions     : {[r['id'] for r in config['regions']]}")

    pipeline_start = time.time()

    if args.validate:
        run_validation()

    elif args.stage1_only:
        demand_df = run_stage1(config, rng)
        if args.manifest:
            print_column_manifest(demand_df, demand_df)

    elif args.stage2_only:
        rl_df = run_stage2(config, rng)

    else:
        # Full pipeline
        demand_df = run_stage1(config, rng)
        rl_df     = run_stage2(config, rng, demand_df=demand_df)

        if args.manifest:
            print_column_manifest(demand_df, rl_df)

        # Auto-run validation after full pipeline
        run_validation()

    total_time = time.time() - pipeline_start
    print_header(f"PIPELINE COMPLETE — {total_time:.1f}s total")
    print(f"\n  Output files:")
    if DEMAND_CSV.exists():
        print(f"    {DEMAND_CSV}  ({DEMAND_CSV.stat().st_size/1024:.1f} KB)")
    if RL_CSV.exists():
        print(f"    {RL_CSV}  ({RL_CSV.stat().st_size/1024:.1f} KB)")
    plots = list((OUTPUT_DIR / "plots").glob("*.png"))
    if plots:
        print(f"    {len(plots)} plot(s) in output/plots/")
    print()


if __name__ == "__main__":
    main()
