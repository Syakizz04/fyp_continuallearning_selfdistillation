# Core configuration + M5 data prep for the drift-triggered CL experiment.
# Safe to import: defines CONFIG and helpers, seeds RNGs, makes output dirs;
# does not load data or train until prepare_drift_data() is called.

import os
import random
import logging
import warnings
import faulthandler
from pathlib import Path
from typing import Dict, List, Tuple

faulthandler.enable()

import numpy as np
import pandas as pd
import torch
from rich.console import Console
from rich.table import Table

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
logging.getLogger("lightning").setLevel(logging.WARNING)
logging.getLogger("lightning.pytorch").setLevel(logging.WARNING)
logging.getLogger("pytorch_lightning").setLevel(logging.WARNING)

console = Console()
PROJECT_ROOT = Path(__file__).resolve().parents[1]

SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BF16_OK = False
if torch.cuda.is_available():
    major, _ = torch.cuda.get_device_capability()
    BF16_OK = major >= 8


# ─── Cell: Central Configuration ─────────────────────────────────────────────
# Drift-triggered experiment on M5 (FOODS x CA_1..CA_4). Edit here only.

CONFIG = {
    # ── Paths (M5 processed CSVs + drift outputs) ──────────────────────────
    "paths": {
        "demand_csv"  : str(PROJECT_ROOT / "data" / "processed_m5" / "demand_forecasting.csv"),
        "rl_csv"      : str(PROJECT_ROOT / "data" / "processed_m5" / "rl_environment.csv"),
        "checkpoints" : str(PROJECT_ROOT / "outputs" / "drift" / "checkpoints"),
        "results"     : str(PROJECT_ROOT / "outputs" / "drift" / "results"),
        "logs"        : str(PROJECT_ROOT / "outputs" / "drift" / "logs"),
        "plots"       : str(PROJECT_ROOT / "outputs" / "drift" / "plots"),
    },

    # ── Drift-experiment timeline ──────────────────────────────────────────
    # Base model trained once on [base_start, base_end]; the last
    # `calibration_tail_weeks` of that span are held out (not trained on) to
    # calibrate the "normal" error/profit reference that sets drift thresholds.
    "timeline": {
        "base_start"            : "2011-01-29",
        "base_end"              : "2012-12-31",
        "calibration_tail_weeks": 8,
        "walk_start"            : "2013-01-01",
        "walk_end"              : "2015-12-31",
        "check_every_days"      : 7,    # weekly drift check
        "eval_window_weeks"     : 4,    # rolling window the drift metric averages over
    },

    # ── Retraining strategies + reference baselines ────────────────────────
    # EWC removed from the default set: it under-adapted in the first run, which
    # is what a pull-toward-base-weights regulariser does when the regime has
    # genuinely changed. The code path is unchanged, so re-adding "ewc" here
    # restores it.
    "strategies"   : ["replay", "sdft"],
    # `frozen` is the metric ANCHOR (the never-retrained reference every delta is
    # measured against). `naive` is the CONTROL (drift-triggered, no CL
    # mechanism). These are different jobs and neither substitutes for the other:
    # without `naive`, an arm that beats `frozen` cannot say whether the win came
    # from its CL mechanism or merely from retraining at the right moment.
    #
    # `periodic` is the mirror control — plain fine-tuning on a fixed schedule,
    # so it isolates the TRIGGER rather than the mechanism. It is the heaviest arm
    # (~24 retrains) and is left out by default for runtime; re-add it here to
    # restore it, the on_check_periodic path is unchanged.
    "baselines"    : ["frozen", "naive"],
    "retrain": {
        "scope"             : "recent_window+replay",
        "recent_window_weeks": 8,       # data window a triggered retrain trains on
        "warm_start"        : True,     # fine-tune current weights, not from scratch
        "periodic_every_weeks": 13,     # ~quarterly scheduled retrain (baseline arm)
        "retrain_epochs"    : 5,        # fine-tune epochs per forecasting retrain
        "retrain_timesteps" : 15_000,   # PPO steps per RL retrain
    },

    # ── Drift detection ────────────────────────────────────────────────────
    # Forecasting: retrain when windowed MASE exceeds mu + k*sigma of the
    #   calibration error, for `fc_consecutive` checks in a row.
    # RL: retrain when rolling profit_index drops below rl_profit_floor for
    #   `rl_consecutive` checks in a row.
    # NOTE: the raw windowed-error stream + calibration (mu, sigma) are logged
    # every check so triggers at k=1.5/2.0/2.5 are re-derivable WITHOUT rerunning.
    "drift": {
        "fc_k_sigma"      : 2.0,
        "fc_consecutive"  : 2,
        "rl_profit_floor" : 1.0,
        "rl_consecutive"  : 2,
        "k_sensitivity"   : [1.5, 2.0, 2.5],   # reported post-hoc from logged stream
    },

    # ── Forgetting probes ──────────────────────────────────────────────────
    # Fixed historical eval windows (one per calendar quarter). After every
    # retrain, the current model is re-scored on all *past* probes; degradation
    # = forgetting. Method-independent, so comparable across asynchronous arms.
    "probes": {
        "cadence"     : "quarterly",
        "window_days" : 14,             # = prediction_length (one forecast horizon)
    },

    # ── Forecasting (TFT) — M5-native features ─────────────────────────────
    "forecasting": {
        "encoder_length"        : 60,
        "prediction_length"     : 14,
        "hidden_size"           : 128,
        "attention_head_size"   : 4,
        "dropout"               : 0.1,
        "hidden_continuous_size": 32,
        "output_size"           : 7,
        "learning_rate"         : 3e-4,
        "batch_size"            : 256,
        "max_epochs"            : 12,    # trimmed: TFT val_loss plateaus by ~epoch 1-2 on M5
        "early_stop_patience"   : 3,     # stop 3 epochs after best val_loss
        "early_stop_min_delta"  : 1e-3,  # ignore sub-1e-3 "improvements" (plateau noise)
        "allow_unknown_categories": True,  # walk-eval may see series the base encoder lacks
        "gradient_clip"         : 0.1,
        "num_workers"           : 4,
        "group_ids"             : ["product_id", "region_id"],   # item x store
        "target"                : "demand",
        "known_reals": [
            "time_idx", "day_of_week", "day_of_month", "week_of_year",
            "month", "quarter", "is_weekend",
            "sell_price", "snap", "is_event", "event_type_code",
        ],
        "static_categoricals"   : ["product_id", "region_id", "product_category"],
        "static_reals"          : [],
        "time_varying_unknown_reals": ["demand"],
    },

    # ── RL (PPO) ───────────────────────────────────────────────────────────
    "rl": {
        "learning_rate"         : 3e-4,
        "n_steps"               : 2048,
        "batch_size"            : 64,
        "n_epochs"              : 10,
        "gamma"                 : 0.99,
        "gae_lambda"            : 0.95,
        "clip_range"            : 0.2,
        "ent_coef"              : 0.01,
        "vf_coef"               : 0.5,
        "max_grad_norm"         : 0.5,
        "total_timesteps_per_task": 50_000,
        # Sales cap at stock on hand and lost sales are charged for, so
        # `inventory_level` has a causal path to reward at all. Without this it
        # is an observation the agent cannot be rewarded or punished for using,
        # and it correctly learns to ignore it. Paired with the cover-based
        # inventory scaling in DynamicPricingEnv._precompute; together they took
        # the agent's response to a degraded stock signal from ~4% of decisions
        # to 35-70%. Enabled by default from FYP2 onward - it changes the reward,
        # so FYP1's RL numbers are NOT comparable to runs made with it on.
        "inventory_constrained": True,
        "lost_sale_penalty": 0.5,
        "eval_episodes"         : 6,     # walk-forward re-evals policy every check; 6 balances noise vs cost
        "price_tiers"           : [
            -0.10, -0.08, -0.06, -0.04, -0.02, 0.00,
             0.02,  0.04,  0.06,  0.08,  0.10,
        ],
        "n_actions"             : 11,
        "policy"                : "MlpPolicy",
        "net_arch"              : [256, 256],
    },

    # ── CL method hyperparameters ──────────────────────────────────────────
    "cl": {
        "ewc_lambda"            : 400.0,
        "ewc_fisher_samples"    : 200,
        "replay_buffer_size"    : 2000,
        "replay_mix_ratio"      : 0.30,
        "recall_buffer_capacity": 20_000,
        "recall_mix_n_steps"    : 512,
        "sdft_alpha"            : 0.50,
        "sdft_kl_coef"          : 0.10,
    },

    # ── Hardware ───────────────────────────────────────────────────────────
    "hardware": {
        "device"            : DEVICE,
        "precision"         : "32",
        "compile"           : os.environ.get("FYP_TORCH_COMPILE", "0") == "1",
        "compile_mode"      : "reduce-overhead",
        "num_workers"       : int(os.environ.get("FYP_NUM_WORKERS", "0")),
        "pin_memory"        : torch.cuda.is_available(),
        "persistent_workers": False,
    },

    # ── Metrics (grid-anchored; frozen base = reference, not naive) ─────────
    "metrics": {
        "forecasting_primary"  : "mase",
        "forecasting_secondary": ["smape", "rmse"],
        "rl_primary"           : ["profit_index", "cumulative_profit"],
        "rl_secondary"         : ["pricing_regret", "avg_episode_reward"],
        "efficiency"           : ["n_retrains", "cumulative_train_steps"],
        "retention"            : ["forgetting"],
    },
}


# ─── Column contracts (M5) ───────────────────────────────────────────────────

#: Columns that identify a row rather than feed the model. They are loaded
#: whether or not the model uses them, because `adapt_config_to_data` prunes
#: CONFIG's feature lists in place and this function re-reads CONFIG: on a
#: single-store scope it drops `region_id`, so a SECOND `prepare_drift_data()`
#: call in the same process would load a frame without it and every downstream
#: join, sort and group-by keyed on it would raise KeyError. That is not
#: hypothetical - it killed `retrain_pricer --verify` twice, in two different
#: functions, each time after the training had already succeeded.
#:
#: Being pruned from the model's feature lists means "not a feature". It does not
#: mean "not a column".
DEMAND_KEY_COLUMNS = ["date", "product_id", "region_id"]


def demand_required_columns() -> List[str]:
    """Columns the TFT forecasting pipeline needs (auto-derived from CONFIG)."""
    fc = CONFIG["forecasting"]
    required = (
        DEMAND_KEY_COLUMNS
        + fc["group_ids"]
        + fc["static_categoricals"]
        + [fc["target"]]
        + [c for c in fc["known_reals"] if c != "time_idx"]
    )
    return list(dict.fromkeys(required))


RL_REQUIRED_COLUMNS = [
    "date", "product_id", "region_id",
    "day_of_week", "month", "is_weekend",
    "snap", "is_event", "event_type_code",
    "elasticity_coefficient", "demand_forecast", "inventory_level",
    "competitor_price", "base_price", "realized_demand", "stockout_flag",
]

# Regime flags DynamicPricingEnv reads into its state vector.
#
# This list used to carry FYP1's Malaysian retail calendar - is_mega_sale,
# is_ramadan, is_pre_raya_window, is_pre_cny_window - plus two synthetic shock
# flags, and `build_rl_features` materialised all six on M5 so the env's
# length-1 `.get()` fallback would not raise. Measured on the real data, four of
# them were constant zero: 4 of 13 observation slots carried no information at
# all, and the other two were M5's `snap` and `is_event` wearing Malaysian
# names. M5 is Walmart California; there is no Raya, no CNY and no viral shock
# generator.
#
# The env now reads M5's own flags under their own names. Two real regime
# features, honestly labelled, instead of six slots of which four were dead.
ENV_REGIME_FLAGS = ["snap", "is_event"]


def select_required_columns(df: pd.DataFrame, required: List[str], name: str) -> pd.DataFrame:
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"{name} dataset is missing required columns: {missing}")
    dropped = [c for c in df.columns if c not in required]
    if dropped:
        preview = ", ".join(dropped[:8]) + ("..." if len(dropped) > 8 else "")
        console.print(f"  [dim]{name}: ignoring {len(dropped)} unused column(s): {preview}[/dim]")
    return df.loc[:, required].copy()


# ─── Data loading / cleaning / feature engineering ───────────────────────────

def load_and_clean(demand_path: str, rl_path: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    console.print("[bold]Loading M5 datasets...[/bold]")
    demand_df = pd.read_csv(demand_path, parse_dates=["date"])
    rl_df     = pd.read_csv(rl_path,     parse_dates=["date"])

    demand_df = select_required_columns(demand_df, demand_required_columns(), "Demand")
    rl_df     = select_required_columns(rl_df, RL_REQUIRED_COLUMNS, "RL")

    console.print(f"  Demand CSV  : {len(demand_df):,} rows x {demand_df.shape[1]} cols")
    console.print(f"  RL CSV      : {len(rl_df):,} rows x {rl_df.shape[1]} cols")

    # Sort/cast only by columns that survived selection. `adapt_config_to_data`
    # prunes constant features (region_id on a single-store scope) from
    # CONFIG["forecasting"], and `demand_required_columns()` re-reads CONFIG, so a
    # SECOND call to this function in the same process gets a narrower frame than
    # the first. Hardcoding the key list makes that second call a KeyError - which
    # is what it did to `retrain_pricer --verify`, after the training had already
    # succeeded.
    def present(frame, cols):
        return [c for c in cols if c in frame.columns]

    sort_keys = ["product_id", "region_id", "date"]
    demand_df = demand_df.sort_values(present(demand_df, sort_keys))
    for col in present(demand_df, ["product_id", "region_id", "product_category"]):
        if demand_df[col].dtype == object:
            demand_df[col] = demand_df[col].astype("category")
    num_cols = demand_df.select_dtypes(include=[np.number]).columns
    demand_df[num_cols] = demand_df[num_cols].fillna(0)

    rl_df = rl_df.sort_values(present(rl_df, sort_keys))
    rl_num = rl_df.select_dtypes(include=[np.number]).columns
    rl_df[rl_num] = rl_df[rl_num].fillna(0)

    console.print("[green]✓ Data loaded and cleaned[/green]")
    return demand_df, rl_df


def build_tft_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Prepare demand_df for TimeSeriesDataSet; time_idx consecutive per group."""
    df = df.copy()
    min_date = df["date"].min()
    df["time_idx"] = (df["date"] - min_date).dt.days.astype(int)
    for col in ["product_id", "region_id", "product_category"]:
        df[col] = df[col].astype(str)
    for col in CONFIG["forecasting"]["known_reals"]:
        if col in df.columns:
            df[col] = df[col].astype(float)
        else:
            df[col] = 0.0
    df["demand"] = df["demand"].clip(lower=0).astype(float)
    return df.reset_index(drop=True)


def adapt_config_to_data(tft_df: pd.DataFrame) -> List[str]:
    """Drop constant group_ids / static_categoricals so the TFT carries no dead
    single-level embeddings. e.g. `region_id` is constant when one store is selected,
    `product_category` when one department is selected. Mutates CONFIG in place
    (the repo's mutate-CONFIG idiom) and returns the dropped columns. `product_id`
    is always protected so group_ids never empties."""
    fc = CONFIG["forecasting"]

    def prune(key: str, protect: set) -> List[str]:
        kept, dropped = [], []
        for col in fc[key]:
            if col in protect or col not in tft_df.columns:
                kept.append(col)
            elif tft_df[col].nunique(dropna=False) <= 1:
                dropped.append(col)
            else:
                kept.append(col)
        fc[key] = kept
        return dropped

    dropped = sorted(set(prune("group_ids", {"product_id"}))
                     | set(prune("static_categoricals", {"product_id"})))
    if dropped:
        console.print(f"  [yellow]auto-pruned constant feature(s): "
                      f"{', '.join(dropped)}[/yellow] -> group_ids={fc['group_ids']}, "
                      f"static_categoricals={fc['static_categoricals']}")
    return dropped


def build_rl_features(df: pd.DataFrame) -> pd.DataFrame:
    """Normalise the columns DynamicPricingEnv reads as observations."""
    df = df.copy()
    max_comp = df["competitor_price"].abs().max()
    if max_comp > 0:
        df["competitor_price_norm"] = df["competitor_price"] / max_comp
    dmax = df["demand_forecast"].max()
    if dmax > 0:
        df["demand_forecast_norm"] = df["demand_forecast"] / dmax
    inv_max = df["inventory_level"].max()
    df["inventory_norm"] = df["inventory_level"] / max(inv_max, 1)

    # DynamicPricingEnv reads these via df.get(col, pd.Series(0)) — that fallback
    # is a length-1 Series, so a missing column crashes indexing rather than
    # defaulting. Materialise them as full-length float columns. Both are real M5
    # columns now, so the 0.0 branch should never fire; it stays as a guard.
    for flag in ENV_REGIME_FLAGS:
        df[flag] = df[flag].astype(float) if flag in df.columns else 0.0
    return df


# ─── Timeline helpers (base / calibration / walk-forward / probes) ───────────

def base_train_window() -> Tuple[pd.Timestamp, pd.Timestamp]:
    t = CONFIG["timeline"]
    return pd.Timestamp(t["base_start"]), pd.Timestamp(t["base_end"])


def calibration_window() -> Tuple[pd.Timestamp, pd.Timestamp]:
    """Held-out tail of the base period used to calibrate 'normal' error/profit."""
    t = CONFIG["timeline"]
    base_end = pd.Timestamp(t["base_end"])
    cal_start = base_end - pd.Timedelta(weeks=t["calibration_tail_weeks"]) + pd.Timedelta(days=1)
    return cal_start, base_end


def base_fit_window() -> Tuple[pd.Timestamp, pd.Timestamp]:
    """Base period MINUS the calibration tail — what the base model trains on."""
    base_start, _ = base_train_window()
    cal_start, _  = calibration_window()
    return base_start, cal_start - pd.Timedelta(days=1)


def walk_forward_checks() -> List[pd.Timestamp]:
    """Weekly check anchor dates across the walk-forward span."""
    t = CONFIG["timeline"]
    return list(pd.date_range(t["walk_start"], t["walk_end"],
                              freq=f"{t['check_every_days']}D"))


def build_probe_windows() -> List[Dict]:
    """One probe per calendar quarter from base start through walk end.
    Each probe is a fixed (start,end) slice of `window_days`, anchored at a
    quarter end. Used to measure forgetting after each retrain."""
    t = CONFIG["timeline"]
    span_start = pd.Timestamp(t["base_start"])
    span_end   = pd.Timestamp(t["walk_end"])
    win = CONFIG["probes"]["window_days"]
    # Quarter-end alias: "QE" on pandas>=2.2, "Q" on older versions.
    try:
        q_ends = pd.date_range(span_start, span_end, freq="QE")
    except ValueError:
        q_ends = pd.date_range(span_start, span_end, freq="Q")
    probes = []
    for qe in q_ends:
        end   = min(qe, span_end)
        start = end - pd.Timedelta(days=win - 1)
        if start < span_start:
            continue
        probes.append({
            "name" : f"probe_{end.year}Q{((end.month - 1) // 3) + 1}",
            "start": start, "end": end,
        })
    return probes


def slice_by_dates(df: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    mask = (df["date"] >= start) & (df["date"] <= end)
    return df[mask].copy()


def filter_to_base_trainable(tft_df: pd.DataFrame, rl_df: pd.DataFrame
                             ) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Restrict both frames to (product_id, region_id) series that are present with
    enough rows in the base-fit window. The base model can only encode/normalize
    series it saw at base time; a series introduced later would crash walk-eval with
    'unknown category'. This keeps the experiment to 'known series, watch for drift'
    and is scope-agnostic (any dept/store/sample). Items dropped are logged."""
    bf0, bf1 = base_fit_window()
    fc = CONFIG["forecasting"]
    min_rows = fc["encoder_length"] + fc["prediction_length"]
    keys = ["product_id", "region_id"]

    base = tft_df[(tft_df["date"] >= bf0) & (tft_df["date"] <= bf1)]
    counts = base.groupby(keys, observed=True).size()
    keep = counts[counts >= min_rows].index            # MultiIndex of (product_id, region_id)

    n0 = tft_df.groupby(keys, observed=True).ngroups
    tft_mi = pd.MultiIndex.from_arrays([tft_df["product_id"], tft_df["region_id"]])
    rl_mi  = pd.MultiIndex.from_arrays([rl_df["product_id"],  rl_df["region_id"]])
    tft_df = tft_df[tft_mi.isin(keep)].copy()
    rl_df  = rl_df[rl_mi.isin(keep)].copy()
    n1 = len(keep)
    if n1 < n0:
        console.print(f"  [yellow]base-trainable filter: kept {n1}/{n0} series "
                      f"(dropped {n0 - n1} not present with >={min_rows} rows in base-fit)[/yellow]")
    return tft_df, rl_df


def prepare_drift_data(demand_path: str | None = None, rl_path: str | None = None) -> Dict:
    """Load M5, engineer features, and produce the drift-experiment splits.

    Returns a dict with the full frames plus base-fit / calibration / walk-forward
    slices and the probe windows. The walk-forward monitor (later phase) steps
    through `walk` using `walk_forward_checks()`.
    """
    demand_path = demand_path or CONFIG["paths"]["demand_csv"]
    rl_path     = rl_path or CONFIG["paths"]["rl_csv"]

    demand_df, rl_df = load_and_clean(demand_path, rl_path)
    tft_df = build_tft_dataframe(demand_df)
    adapt_config_to_data(tft_df)        # drop constant region_id / product_category
    rl_df  = build_rl_features(rl_df)
    tft_df, rl_df = filter_to_base_trainable(tft_df, rl_df)   # drop late-introduced series

    bf_start, bf_end   = base_fit_window()
    cal_start, cal_end = calibration_window()
    walk_start = pd.Timestamp(CONFIG["timeline"]["walk_start"])
    walk_end   = pd.Timestamp(CONFIG["timeline"]["walk_end"])

    return {
        "tft_full"   : tft_df,
        "rl_full"    : rl_df,
        "tft_base"   : slice_by_dates(tft_df, bf_start, bf_end),
        "rl_base"    : slice_by_dates(rl_df,  bf_start, bf_end),
        "tft_calib"  : slice_by_dates(tft_df, cal_start, cal_end),
        "rl_calib"   : slice_by_dates(rl_df,  cal_start, cal_end),
        "tft_walk"   : slice_by_dates(tft_df, walk_start, walk_end),
        "rl_walk"    : slice_by_dates(rl_df,  walk_start, walk_end),
        "probes"     : build_probe_windows(),
        "checks"     : walk_forward_checks(),
    }


# ─── Output dirs / diagnostics ───────────────────────────────────────────────

def ensure_output_dirs() -> None:
    for key, path in CONFIG["paths"].items():
        if key not in {"demand_csv", "rl_csv"}:
            Path(path).mkdir(parents=True, exist_ok=True)


def print_timeline_summary(data: Dict) -> None:
    bf_start, bf_end   = base_fit_window()
    cal_start, cal_end = calibration_window()
    table = Table(show_header=True, header_style="bold cyan", title="Drift experiment timeline")
    table.add_column("Phase", style="bold")
    table.add_column("Span")
    table.add_column("TFT rows")
    table.add_column("RL rows")
    table.add_row("Base fit",     f"{bf_start.date()} -> {bf_end.date()}",
                  f"{len(data['tft_base']):,}",  f"{len(data['rl_base']):,}")
    table.add_row("Calibration",  f"{cal_start.date()} -> {cal_end.date()}",
                  f"{len(data['tft_calib']):,}", f"{len(data['rl_calib']):,}")
    table.add_row("Walk-forward",
                  f"{CONFIG['timeline']['walk_start']} -> {CONFIG['timeline']['walk_end']}",
                  f"{len(data['tft_walk']):,}",  f"{len(data['rl_walk']):,}")
    console.print(table)
    console.print(f"  Weekly checks : {len(data['checks'])}")
    console.print(f"  Probes        : {len(data['probes'])} "
                  f"({', '.join(p['name'] for p in data['probes'][:6])}"
                  f"{'...' if len(data['probes']) > 6 else ''})")


ensure_output_dirs()
