"""
M5 -> FYP pipeline adapter.

Reads the raw Kaggle "M5 Forecasting - Accuracy" files and emits the two CSVs
the continual-learning pipeline consumes, in the exact column contract of
`core_pipeline.demand_required_columns()` / the RL environment:

    data/processed_m5/demand_forecasting.csv   (TFT forecasting)
    data/processed_m5/rl_environment.csv       (semi-synthetic PPO pricing)

Scope (locked design): cat_id == "FOODS", store_id in {CA_1..CA_4}.

Raw inputs expected under --raw-dir (default data/m5_raw/):
    sales_train_evaluation.csv   (or sales_train_validation.csv)
    calendar.csv
    sell_prices.csv

Download (Kaggle, requires competition acceptance):
    kaggle competitions download -c m5-forecasting-accuracy
    unzip into data/m5_raw/

The forecasting CSV is real M5. The RL CSV is *semi-synthetic*: elasticity is
calibrated from real M5 price variation (log-log regression); inventory,
competitor price, base price, demand_forecast reference, and stockouts are
modeled. This matches the agreed design.

Usage:
    python -m dataset_generator.m5.build_m5                 # build from raw
    python -m dataset_generator.m5.build_m5 --self-test     # synth mini-M5, validate
    python -m dataset_generator.m5.build_m5 --raw-dir D:/m5 --out-dir data/processed_m5
"""

from __future__ import annotations

import argparse
import tempfile
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd

from .elasticity import (
    SAFETY_CLIP, ElasticityReport, Prior, build_price_instrument,
    estimate_elasticities, fit_external_prior, load_prior,
)

try:
    from rich.console import Console
    console = Console()
    def log(msg: str) -> None: console.print(msg)
except Exception:  # rich optional
    def log(msg: str) -> None: print(msg)

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# ── Locked scope ────────────────────────────────────────────────────────────
KEEP_CAT    = "FOODS"
KEEP_STORES = ["CA_1", "CA_2", "CA_3", "CA_4"]

# Event types in M5 -> small integer code for a known-real feature.
EVENT_TYPE_CODE = {
    "NA": 0, "Sporting": 1, "Cultural": 2, "National": 3, "Religious": 4,
}

# Semi-synthetic RL knobs.
# Elasticity is no longer a clip + constant fallback — see elasticity.py for the
# hierarchical (empirical-Bayes) estimator that replaced it. The knobs that used
# to live here (ELASTICITY_CLIP, ELASTICITY_FALLBACK, MIN_ELAST_OBS,
# MIN_LOGPRICE_STD) now live there.
UNIT_COST_FRAC    = 0.55           # cost = 0.55 * base_price (mirrors RL env)
# Replenishment. See edge_system/inventory/replenishment.py for why the lead
# time is the parameter that matters: with instant delivery (the original
# behaviour) stock ran out on 0.05% of days, so `inventory_level` never signalled
# scarcity and the pricing agent correctly learned to ignore it. Cover 10 with a
# 3-day lead gives ~5% stockout days at a ~98% fill rate, which is where real
# grocery retail operates.
INV_COVER_DAYS    = 10             # order-up-to level, in days of mean demand
INV_REORDER_FRAC  = 0.5            # reorder when position < 50% of that level
INV_LEAD_TIME_DAYS = 3             # 0 reproduces the old instant-refill loop


# ─── Raw loading ─────────────────────────────────────────────────────────────

def _find_sales_file(raw_dir: Path) -> Path:
    for name in ("sales_train_evaluation.csv", "sales_train_validation.csv"):
        p = raw_dir / name
        if p.exists():
            return p
    raise FileNotFoundError(
        f"No sales_train_evaluation.csv / sales_train_validation.csv in {raw_dir}. "
        "Download the M5 competition data into that folder."
    )


def load_raw(raw_dir: Path) -> Dict[str, pd.DataFrame]:
    raw_dir = Path(raw_dir)
    sales_path = _find_sales_file(raw_dir)
    calendar_path = raw_dir / "calendar.csv"
    prices_path   = raw_dir / "sell_prices.csv"
    for p in (calendar_path, prices_path):
        if not p.exists():
            raise FileNotFoundError(f"Missing required M5 file: {p}")

    log(f"[bold]Loading raw M5[/bold] from {raw_dir}")
    sales    = pd.read_csv(sales_path)
    calendar = pd.read_csv(calendar_path, parse_dates=["date"])
    prices   = pd.read_csv(prices_path)
    log(f"  sales    : {sales.shape[0]:,} series x {sales.shape[1]} cols")
    log(f"  calendar : {calendar.shape[0]:,} days")
    log(f"  prices   : {prices.shape[0]:,} (store,item,week) rows")
    return {"sales": sales, "calendar": calendar, "prices": prices}


# ─── Long-format build (real M5) ─────────────────────────────────────────────

def _select_items(vol: pd.Series, n: int, mode: str, seed: int) -> pd.Index:
    """Pick `n` item_ids from a per-item total-volume series by `mode`:
      top         -- the n highest-volume items (cleanest signal; selection-biased
                     toward stable staples, which can suppress regime drift).
      random      -- a uniform random sample (representative; default for results).
      stratified  -- ~proportional across volume deciles (spans intermittent->staple).
    """
    vol = vol.sort_values(ascending=False)
    if n >= len(vol):
        return vol.index
    if mode == "top":
        return vol.head(n).index
    rng = np.random.default_rng(seed)
    if mode == "random":
        return pd.Index(rng.choice(vol.index.to_numpy(), size=n, replace=False))
    if mode == "stratified":
        ranks = vol.rank(method="first", ascending=False)
        deciles = ((ranks - 1) // (len(vol) / 10.0)).astype(int).clip(0, 9)
        picks: list = []
        per = n // 10
        for d in range(10):
            members = vol.index[deciles.values == d].to_numpy()
            k = min(per, len(members))
            if k:
                picks.extend(rng.choice(members, size=k, replace=False))
        if len(picks) < n:                       # top up to exactly n
            pool = np.setdiff1d(vol.index.to_numpy(), np.array(picks, dtype=object))
            if len(pool):
                picks.extend(rng.choice(pool, size=min(n - len(picks), len(pool)),
                                        replace=False))
        return pd.Index(picks)
    raise ValueError(f"unknown sample_mode {mode!r} (use top/random/stratified)")


def build_long(sales: pd.DataFrame, calendar: pd.DataFrame,
               prices: pd.DataFrame, dept: str | None = None,
               top_n_items: int | None = None, stores: list | None = None,
               sample_mode: str = "top", seed: int = 42,
               active_before: str | None = None) -> pd.DataFrame:
    """Filter to FOODS x selected CA stores, melt to daily long, join calendar+prices.

    Optional scope reducers (applied before melt, so they cut memory/disk too):
      dept          -- keep only one department, e.g. "FOODS_3" (real-result scope).
      stores        -- subset of CA stores (default all 4); e.g. ["CA_1"].
      active_before -- keep only items whose first sale is before this date, so every
                       sampled item is mature by the base-fit window (avoids items the
                       base model can never see -> 'unknown category' at walk-eval).
      top_n_items   -- keep only N item_ids, chosen per `sample_mode`.
      sample_mode   -- top | random | stratified (how the N items are chosen).
    """
    id_cols = ["id", "item_id", "dept_id", "cat_id", "store_id", "state_id"]
    d_cols  = [c for c in sales.columns if c.startswith("d_")]
    stores = stores or KEEP_STORES

    sub = sales[sales["cat_id"].eq(KEEP_CAT) & sales["store_id"].isin(stores)].copy()
    log(f"  filtered : {len(sub):,} series (FOODS x {len(stores)} store(s): {','.join(stores)})")
    if sub.empty:
        raise ValueError("Filter produced 0 series; check cat_id/store_id values in raw data.")

    if dept is not None:
        sub = sub[sub["dept_id"].eq(dept)].copy()
        log(f"  dept     : {len(sub):,} series (dept_id == {dept})")
        if sub.empty:
            raise ValueError(f"dept={dept!r} produced 0 series; check dept_id values.")

    if active_before is not None:
        cutoff = pd.Timestamp(active_before)
        d2date = dict(zip(calendar["d"], pd.to_datetime(calendar["date"])))
        nz = sub[d_cols].ne(0)
        first_d = nz.idxmax(axis=1).where(nz.any(axis=1))          # NaN if never sold
        first_date = first_d.map(d2date)
        sub = sub.assign(_first=first_date.values)
        item_first = sub.groupby("item_id")["_first"].min()        # earliest across stores
        keep = item_first[item_first < cutoff].index
        before = sub["item_id"].nunique()
        sub = sub[sub["item_id"].isin(keep)].drop(columns="_first").copy()
        log(f"  active   : {sub['item_id'].nunique()} items first-sold before "
            f"{cutoff.date()} (dropped {before - sub['item_id'].nunique()} late-introduced)")

    if top_n_items is not None:
        vol = sub.groupby("item_id")[d_cols].sum().sum(axis=1)
        keep_items = _select_items(vol, top_n_items, sample_mode, seed)
        sub = sub[sub["item_id"].isin(keep_items)].copy()
        log(f"  {sample_mode}-{top_n_items} : {len(sub):,} series "
            f"({sub['item_id'].nunique()} items x {sub['store_id'].nunique()} store(s), "
            f"seed={seed})")

    long = sub.melt(id_vars=id_cols, value_vars=d_cols,
                    var_name="d", value_name="demand")

    cal = calendar[["d", "date", "wm_yr_wk", "month",
                    "event_name_1", "event_type_1", "snap_CA"]].copy()
    long = long.merge(cal, on="d", how="left")

    # Weekly sell price keyed by (store_id, item_id, wm_yr_wk).
    long = long.merge(prices, on=["store_id", "item_id", "wm_yr_wk"], how="left")

    # Rows before an item's first sale have no price -> drop (standard M5 handling).
    before = len(long)
    long = long[long["sell_price"].notna()].copy()
    log(f"  joined   : {len(long):,} rows ({before - len(long):,} pre-availability rows dropped)")

    long = long.sort_values(["item_id", "store_id", "date"]).reset_index(drop=True)
    return long


def _calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    d = df["date"]
    df["day_of_week"]  = d.dt.dayofweek            # 0=Mon..6=Sun
    df["day_of_month"] = d.dt.day
    df["week_of_year"] = d.dt.isocalendar().week.astype(int)
    df["quarter"]      = d.dt.quarter
    df["is_weekend"]   = d.dt.dayofweek.isin([5, 6]).astype(int)
    df["snap"]         = df["snap_CA"].fillna(0).astype(int)
    df["is_event"]     = df["event_name_1"].notna().astype(int)
    df["event_type_code"] = (
        df["event_type_1"].fillna("NA").map(EVENT_TYPE_CODE).fillna(0).astype(int)
    )
    return df


# ─── Forecasting CSV ─────────────────────────────────────────────────────────

FORECAST_COLUMNS = [
    "date", "product_id", "region_id", "product_category", "demand",
    "day_of_week", "day_of_month", "week_of_year", "month", "quarter",
    "is_weekend", "sell_price", "snap", "is_event", "event_type_code",
]


def build_forecast_csv(long: pd.DataFrame) -> pd.DataFrame:
    df = long.copy()
    df["product_id"]       = df["item_id"]
    df["region_id"]        = df["store_id"]
    df["product_category"] = df["dept_id"]      # FOODS_1/2/3 (cat is constant)
    df["demand"]           = df["demand"].clip(lower=0).astype(float)
    out = df[FORECAST_COLUMNS].copy()
    return out


# ─── Semi-synthetic RL layer ─────────────────────────────────────────────────

def _elasticity_report(df: pd.DataFrame, external_prior: Optional[Prior]) -> ElasticityReport:
    """Run the hierarchical estimator over the whole frame at once.

    Elasticity is now estimated jointly rather than series-by-series: items
    borrow strength from their department and from an external real-retail
    prior, so a series with no price variation degrades to the prior instead of
    snapping to a hard-coded constant. See elasticity.py.
    """
    return estimate_elasticities(df, external_prior=external_prior)


def _simulate_inventory(demand: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    (s,S) replenishment WITH a delivery lead time.

    Delegates to `edge_system.inventory.replenishment`, which is the same policy
    the live simulation runs, so the environment the agent trains against and the
    one it is deployed into are the same model rather than two similar-looking
    loops that drift apart.

    Returns `(on_hand, stockout_flag, unmet_demand)`. The third is new and is the
    **censoring** signal: units demanded but not served. On those days observed
    sales understate true demand, so a forecaster retrained on observed sales
    learns the product is less wanted than it is.
    """
    from edge_system.inventory.replenishment import simulate_inventory

    trace = simulate_inventory(
        demand,
        cover_days=INV_COVER_DAYS,
        reorder_frac=INV_REORDER_FRAC,
        lead_time_days=INV_LEAD_TIME_DAYS,
    )
    return trace.on_hand, trace.stockout, trace.unmet


def _simulate_inventory_legacy(demand: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """The original instant-replenishment loop, kept as the E3-style control arm.

    Preserved rather than deleted because the difference between this and the
    lead-time policy is what explains the pricing agent's insensitivity to
    inventory, and that comparison belongs in the report."""
    mean_d = max(float(np.mean(demand)), 1e-6)
    target = INV_COVER_DAYS * mean_d
    reorder = INV_REORDER_FRAC * target
    inv = np.empty(len(demand), dtype=float)
    stock = target
    out_flag = np.zeros(len(demand), dtype=int)
    for i, d in enumerate(demand):
        inv[i] = stock
        if stock < d:
            out_flag[i] = 1
        stock = max(stock - d, 0.0)
        if stock < reorder:
            stock = target              # next-period replenishment to target
    return inv, out_flag


def build_rl_csv(long: pd.DataFrame,
                 external_prior: Optional[Prior] = None) -> tuple[pd.DataFrame, ElasticityReport]:
    df = long.copy()
    df["product_id"] = df["item_id"]
    df["region_id"]  = df["store_id"]

    # Competitor price proxy: dept-level median sell_price per (store, week).
    df["competitor_price"] = (
        df.groupby(["store_id", "dept_id", "wm_yr_wk"])["sell_price"]
          .transform("median")
    )

    # Estimated once for all series, so items can borrow strength from each other.
    report = _elasticity_report(df, external_prior)

    rows = []
    for (item, store), g in df.groupby(["item_id", "store_id"], sort=False):
        g = g.sort_values("date")
        elast = report.values[(item, store)]
        demand = g["demand"].clip(lower=0).to_numpy(dtype=float)

        # base_price: trailing 28-day median of actual price (rolling reference).
        base_price = (
            g["sell_price"].rolling(28, min_periods=1).median().to_numpy(dtype=float)
        )
        # demand_forecast reference: trailing 28-day mean of realized demand.
        dforecast = (
            pd.Series(demand).rolling(28, min_periods=1).mean().shift(1)
              .bfill().to_numpy(dtype=float)
        )
        inv, stockout, unmet = _simulate_inventory(demand)

        sub = pd.DataFrame({
            "date":         g["date"].to_numpy(),
            "product_id":   item,
            "region_id":    store,
            "day_of_week":  g["day_of_week"].to_numpy(),
            "month":        g["month"].to_numpy(),
            "is_weekend":   g["is_weekend"].to_numpy(),
            "snap":         g["snap"].to_numpy(),
            "is_event":     g["is_event"].to_numpy(),
            "event_type_code": g["event_type_code"].to_numpy(),
            "elasticity_coefficient": elast,
            "demand_forecast":  dforecast,
            "inventory_level":  inv,
            "competitor_price": g["competitor_price"].to_numpy(),
            "base_price":       base_price,
            "realized_demand":  demand,
            "stockout_flag":    stockout,
            # Demand that arrived and could not be served. Not consumed by the
            # pricing env; carried so E2 can measure demand censoring without
            # re-deriving it.
            "unmet_demand":     unmet,
        })
        rows.append(sub)

    rl = pd.concat(rows, ignore_index=True)
    rl = rl.sort_values(["product_id", "region_id", "date"]).reset_index(drop=True)
    return rl, report


# ─── Orchestration ───────────────────────────────────────────────────────────

def build(raw_dir: Path, out_dir: Path, dept: str | None = None,
          top_n_items: int | None = None, stores: list | None = None,
          sample_mode: str = "top", seed: int = 42,
          active_before: str | None = None,
          external_prior: Optional[Prior] = None) -> Dict[str, Path]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    raw = load_raw(raw_dir)
    long = build_long(raw["sales"], raw["calendar"], raw["prices"],
                      dept=dept, top_n_items=top_n_items, stores=stores,
                      sample_mode=sample_mode, seed=seed, active_before=active_before)
    long = _calendar_features(long)

    # Hausman instrument for the elasticity regression: the same item's price in
    # the stores we did NOT model. Built from the unfiltered price table, so it
    # uses data already on disk and does not widen the modelling scope.
    focal_stores = sorted(long["store_id"].unique())
    try:
        instrument = build_price_instrument(raw["prices"], focal_stores)
        long = long.merge(instrument, on=["item_id", "wm_yr_wk"], how="left")
    except ValueError as exc:
        # Every store is in scope, so there is no excluded store to instrument
        # with. Degrade to OLS rather than fail the build; the elasticity report
        # records the estimator actually used.
        log(f"  [yellow]no price instrument available ({exc}); using OLS[/yellow]")

    log("[bold]Building forecasting CSV[/bold]")
    fc = build_forecast_csv(long)
    fc_path = out_dir / "demand_forecasting.csv"
    fc.to_csv(fc_path, index=False)
    log(f"  -> {fc_path}  ({len(fc):,} rows x {fc.shape[1]} cols, "
        f"{fc['product_id'].nunique()} items x {fc['region_id'].nunique()} stores)")

    log("[bold]Building semi-synthetic RL CSV[/bold] (elasticity calibrated from real prices)")
    rl, elast_report = build_rl_csv(long, external_prior=external_prior)
    rl_path = out_dir / "rl_environment.csv"
    rl.to_csv(rl_path, index=False)
    log(f"  -> {rl_path}  ({len(rl):,} rows x {rl.shape[1]} cols)")
    log("  " + elast_report.summary().replace("\n", "\n  "))

    # Per-item elasticity provenance, for the before/after figure and so the
    # report can state exactly which items were data-driven vs prior-driven.
    elast_path = out_dir / "elasticity_report.csv"
    pd.DataFrame([
        {"item_id": k[0], "store_id": k[1],
         "elasticity": v,
         "raw_beta": elast_report.raw_betas[k],
         "se": elast_report.estimates[k].se,
         "method": elast_report.estimates[k].method,
         "first_stage_f": elast_report.estimates[k].first_stage_f,
         "n_obs": elast_report.estimates[k].n_obs,
         "identified": elast_report.estimates[k].identified}
        for k, v in elast_report.values.items()
    ]).to_csv(elast_path, index=False)
    log(f"  -> {elast_path}")

    date_min, date_max = fc["date"].min(), fc["date"].max()
    log(f"[green]Done.[/green] date span {date_min.date()} -> {date_max.date()}")
    return {"forecast": fc_path, "rl": rl_path}


# ─── Self-test fixture (synthetic mini-M5) ──────────────────────────────────

def _make_mini_m5(raw_dir: Path, n_days: int = 200, n_items: int = 6) -> None:
    """Write a tiny but schema-faithful M5 raw triple so the adapter can be
    validated end-to-end without the real download."""
    raw_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(42)
    dates = pd.date_range("2011-01-29", periods=n_days, freq="D")
    d_ids = [f"d_{i+1}" for i in range(n_days)]
    # ~weekly wm_yr_wk groups
    wm = 11101 + (np.arange(n_days) // 7)

    events = np.where(rng.random(n_days) < 0.05, "SuperBowl", np.nan)
    etypes = np.where(pd.notna(events), "Sporting", np.nan)
    calendar = pd.DataFrame({
        "date": dates, "wm_yr_wk": wm, "d": d_ids,
        "month": dates.month,
        "event_name_1": events, "event_type_1": etypes,
        "snap_CA": rng.integers(0, 2, n_days),
    })

    depts  = ["FOODS_1", "FOODS_2", "FOODS_3"]
    stores = KEEP_STORES
    price_rows, sales_rows = [], []
    for it in range(n_items):
        item_id = f"FOODS_{(it % 3) + 1}_{it+1:03d}"
        dept = depts[it % 3]
        for store in stores:
            # price varies week to week to create elasticity signal
            base = rng.uniform(2.0, 8.0)
            wk_price = {w: max(0.5, base * rng.uniform(0.8, 1.2)) for w in np.unique(wm)}
            elast_true = rng.uniform(-2.5, -0.5)
            row_demand = []
            for k in range(n_days):
                p = wk_price[wm[k]]
                mu = 20 * (p / base) ** elast_true * (1 + 0.3 * calendar["snap_CA"].iloc[k])
                row_demand.append(max(0, int(rng.poisson(max(mu, 0.1)))))
                price_rows.append({"store_id": store, "item_id": item_id,
                                   "wm_yr_wk": wm[k], "sell_price": p})
            sales_rows.append({
                "id": f"{item_id}_{store}_evaluation",
                "item_id": item_id, "dept_id": dept, "cat_id": "FOODS",
                "store_id": store, "state_id": "CA",
                **{d_ids[k]: row_demand[k] for k in range(n_days)},
            })
    sales = pd.DataFrame(sales_rows)
    prices = (pd.DataFrame(price_rows)
              .drop_duplicates(["store_id", "item_id", "wm_yr_wk"]))
    sales.to_csv(raw_dir / "sales_train_evaluation.csv", index=False)
    calendar.to_csv(raw_dir / "calendar.csv", index=False)
    prices.to_csv(raw_dir / "sell_prices.csv", index=False)


def self_test() -> None:
    log("[bold yellow]SELF-TEST[/bold yellow]: synthesizing mini-M5 and running adapter")
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        raw = tmp / "raw"
        out = tmp / "out"
        _make_mini_m5(raw)
        paths = build(raw, out)

        fc = pd.read_csv(paths["forecast"], parse_dates=["date"])
        rl = pd.read_csv(paths["rl"], parse_dates=["date"])

        # Contract checks
        assert list(fc.columns) == FORECAST_COLUMNS, "forecast columns drifted"
        assert fc["demand"].ge(0).all(), "negative demand"
        assert fc[["product_id", "region_id", "date"]].duplicated().sum() == 0, "dup keys"
        for col in ("elasticity_coefficient", "base_price", "realized_demand",
                    "competitor_price", "inventory_level", "stockout_flag"):
            assert col in rl.columns, f"RL missing {col}"
        assert rl["elasticity_coefficient"].between(*SAFETY_CLIP).all(), "elasticity out of range"
        assert rl["elasticity_coefficient"].lt(0).all(), "elasticity must be negative"
        assert rl["stockout_flag"].isin([0, 1]).all(), "bad stockout flag"
        assert len(fc) == len(rl), "fc/rl row mismatch"
        log(f"[green]SELF-TEST PASSED[/green]  fc={fc.shape}  rl={rl.shape}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Build FYP CSVs from raw M5 data.")
    ap.add_argument("--raw-dir", default=str(PROJECT_ROOT / "data" / "m5_raw"))
    ap.add_argument("--out-dir", default=str(PROJECT_ROOT / "data" / "processed_m5"))
    ap.add_argument("--self-test", action="store_true",
                    help="Synthesize a mini-M5 and validate the adapter end-to-end.")
    ap.add_argument("--dept", default=None,
                    help="Restrict to one department, e.g. FOODS_3 (real-result scope).")
    ap.add_argument("--stores", default=None,
                    help="Comma-separated CA store subset, e.g. CA_1 (default: all 4).")
    ap.add_argument("--top-n-items", type=int, default=None,
                    help="Keep only N items (selected by --sample-mode).")
    ap.add_argument("--sample-mode", default="top", choices=["top", "random", "stratified"],
                    help="How to choose the N items: top volume / random / volume-stratified.")
    ap.add_argument("--seed", type=int, default=42, help="Seed for random/stratified sampling.")
    ap.add_argument("--active-before", default=None,
                    help="Keep only items first-sold before this date (e.g. 2012-09-01) "
                         "so every item is mature by the base-fit window.")
    ap.add_argument("--external-prior", default=None,
                    help="Elasticity prior source. Either a prior.json written by "
                         "save_prior(), or a CSV of external real-retail prices to fit "
                         "one from (see --prior-cols). Supplies the prior ONLY -- never "
                         "joined to M5. Omitted => placeholder prior.")
    ap.add_argument("--prior-cols", default="PRODUCT_ID,unit_price,units,WEEK_NO",
                    help="item,price,qty[,time] column names inside --external-prior.")
    ap.add_argument("--prior-controls", default="display,mailer",
                    help="Exogenous controls for the external elasticity fit. On "
                         "dunnhumby these observe the promotion confound directly.")
    args = ap.parse_args()

    if args.self_test:
        self_test()
        return

    external_prior = None
    if args.external_prior:
        if str(args.external_prior).endswith(".json"):
            external_prior = load_prior(args.external_prior)
            log(f"  loaded external prior: {external_prior}")
        else:
            cols = [c.strip() for c in args.prior_cols.split(",")]
            if len(cols) < 3:
                ap.error("--prior-cols needs at least item,price,qty")
            controls = tuple(c.strip() for c in args.prior_controls.split(",") if c.strip())
            external_prior = fit_external_prior(
                args.external_prior, item_col=cols[0], price_col=cols[1],
                qty_col=cols[2], time_col=cols[3] if len(cols) > 3 else None,
                control_cols=controls,
            )

    stores = [s.strip() for s in args.stores.split(",")] if args.stores else None
    build(Path(args.raw_dir), Path(args.out_dir),
          dept=args.dept, top_n_items=args.top_n_items, stores=stores,
          sample_mode=args.sample_mode, seed=args.seed, active_before=args.active_before,
          external_prior=external_prior)


if __name__ == "__main__":
    main()
