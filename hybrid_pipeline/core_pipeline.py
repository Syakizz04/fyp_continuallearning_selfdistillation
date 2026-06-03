# Core pipeline utilities for the FYP continual-learning experiment.
# This module is safe to import: it defines configuration and helpers, but does not
# load datasets or start training until you call prepare_data()/run helpers.

import os
import random
import logging
import warnings
from pathlib import Path
from typing import Dict, List, Tuple

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


# ─── Cell: Central Configuration ─────────────────────────────────────────────
# Single source of truth for all hyperparameters, paths, and settings.
# Edit only this cell to adjust experiment settings.

SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# ── Hardware detection ─────────────────────────────────────────────────────
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

BF16_OK = False
if torch.cuda.is_available():
    major, minor = torch.cuda.get_device_capability()
    BF16_OK = major >= 8

CONFIG = {
    # ── Paths ──────────────────────────────────────────────────────────────
    "paths": {
        "demand_csv"  : str(PROJECT_ROOT / "data" / "processed" / "demand_forecasting.csv"),
        "rl_csv"      : str(PROJECT_ROOT / "data" / "processed" / "rl_environment.csv"),
        "checkpoints" : str(PROJECT_ROOT / "outputs" / "hybrid" / "checkpoints"),
        "results"     : str(PROJECT_ROOT / "outputs" / "hybrid" / "results"),
        "logs"        : str(PROJECT_ROOT / "outputs" / "hybrid" / "logs"),
        "plots"       : str(PROJECT_ROOT / "outputs" / "hybrid" / "plots"),
    },

    # ── Tasks (6 CL tasks) ────────────────────────────────────────────────
    "tasks": [
        {"task_id": 1, "name": "Baseline_2023_H1",
         "start": "2023-01-01", "end": "2023-05-31",   "regime": "baseline"},
        {"task_id": 2, "name": "MegaSale_2023",
         "start": "2023-06-01", "end": "2023-12-31",   "regime": "mega_sale"},
        {"task_id": 3, "name": "Baseline_2024_H1",
         "start": "2024-01-01", "end": "2024-05-31",   "regime": "baseline"},
        {"task_id": 4, "name": "MegaSale_2024",
         "start": "2024-06-01", "end": "2024-12-31",   "regime": "mega_sale"},
        {"task_id": 5, "name": "Baseline_2025_H1",
         "start": "2025-01-01", "end": "2025-05-31",   "regime": "baseline"},
        {"task_id": 6, "name": "MegaSale_2025",
         "start": "2025-06-01", "end": "2025-12-31",   "regime": "mega_sale"},
    ],

    # ── Model Types & CL Methods ─────────────────────────────────────────
    "model_types": ["forecasting", "rl"],
    "cl_methods": {
        "forecasting": ["naive", "recall_ewc", "adaptive_drift", "drift_adaptive_replay_ewc"],
        "rl":          ["naive", "recall_ewc", "adaptive_drift", "drift_adaptive_replay_ewc"],
    },

    # ── Forecasting (TFT) Hyperparameters ────────────────────────────────
    "forecasting": {
        "encoder_length"       : 60,
        "prediction_length"    : 14,
        "hidden_size"          : 128,
        "attention_head_size"  : 4,
        "dropout"              : 0.1,
        "hidden_continuous_size": 32,
        "output_size"          : 7,    # 7 quantile outputs
        "learning_rate"        : 3e-4,
        "batch_size"           : 256,
        "max_epochs"           : 20,
        "early_stop_patience"  : 5,
        "gradient_clip"        : 0.1,
        "num_workers"          : 4,
        # Series: all 3 SKUs × 3 regions = 9 time series
        "group_ids"            : ["product_id", "region_id"],
        "target"               : "demand",
        "known_reals": [
            "time_idx", "day_of_week", "day_of_month", "week_of_year",
            "month", "quarter", "is_weekend", "is_federal_holiday",
            "is_ramadan", "ramadan_day", "is_pre_raya_window",
            "is_pre_cny_window", "is_mega_sale", "mega_sale_base_magnitude",
            "is_school_holiday", "is_government_payday_window",
            "is_private_payday_window", "social_media_index",
            "competitor_activity_index", "marketing_spend_myr",
        ],
        "static_categoricals"  : ["product_id", "region_id", "product_category"],
        "static_reals"         : [],
        "time_varying_unknown_reals": ["demand"],
    },

    # ── RL (PPO) Hyperparameters ──────────────────────────────────────────
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
        "eval_episodes"         : 10,
        # Action space: 11 discrete price tiers from -10% to +10%.
        "price_tiers"           : [
            -0.10, -0.08, -0.06, -0.04, -0.02,
             0.00,
             0.02,  0.04,  0.06,  0.08,  0.10,
        ],
        "n_actions"             : 11,
        # State feature dimension (computed below)
        "policy"                : "MlpPolicy",
        "net_arch"              : [256, 256],
    },

    # ── CL Method Hyperparameters ─────────────────────────────────────────
    "cl": {
        # EWC
        "ewc_lambda"            : 400.0,
        "ewc_fisher_samples"    : 200,   # batches to estimate Fisher
        # Replay (Forecasting)
        "replay_buffer_size"    : 2000,  # (x, y) pairs per task in buffer
        "replay_mix_ratio"      : 0.30,  # fraction of batch from replay
        # RECALL (RL replay)
        "recall_buffer_capacity": 20_000, # transitions per task
        "recall_mix_n_steps"    : 512,   # past transitions per PPO update
        # SDFT (Forecasting)
        "sdft_alpha"            : 0.50,  # convex blend: alpha*task + (1-alpha)*distill
        # SDFT (RL)
        "sdft_kl_coef"          : 0.10,  # KL penalty coefficient
        # Drift-adaptive replay-free CL
        "drift_ewc_min_scale"   : 0.25,
        "drift_ewc_max_scale"   : 1.25,
        "drift_distill_max_scale": 1.00,
        # Drift-adaptive replay + EWC
        "drift_replay_min_scale": 0.50,
        "drift_replay_max_scale": 1.25,
    },

    # ── RTX 5090 Optimizations ────────────────────────────────────────────
    "hardware": {
        "device"          : DEVICE,
        "precision"       : "32",
        # torch.compile can be unstable with Lightning + PyTorch Forecasting RNNs
        # across repeated continual-learning fits. Opt in with FYP_TORCH_COMPILE=1.
        "compile"         : os.environ.get("FYP_TORCH_COMPILE", "0") == "1",
        "compile_mode"    : "reduce-overhead",
        # Keep workers at 0 by default for notebooks/Vast.ai to avoid fork
        # deprecation spam and child-process deadlocks in already-threaded kernels.
        "num_workers"     : int(os.environ.get("FYP_NUM_WORKERS", "0")),
        "pin_memory"      : torch.cuda.is_available(),
        "persistent_workers": False,
    },

    # ── Metrics ───────────────────────────────────────────────────────────
    "metrics": {
        "forecasting_primary"  : "mase",
        "forecasting_secondary": ["smape", "rmse"],
        "rl_primary"           : ["cumulative_profit", "avg_episode_reward"],
        "rl_secondary"         : ["pricing_regret"],
        "cl_metrics"           : ["bwt", "fwt", "forgetting", "avg_forgetting"],
    },
}

def demand_required_columns() -> List[str]:
    """Columns required by the TFT forecasting pipeline."""
    fc = CONFIG["forecasting"]
    required = (
        ["date"]
        + fc["group_ids"]
        + fc["static_categoricals"]
        + [fc["target"]]
        + [c for c in fc["known_reals"] if c != "time_idx"]
    )
    return list(dict.fromkeys(required))


RL_REQUIRED_COLUMNS = [
    "date", "product_id",
    "region_id", "day_of_week", 
    "month", "is_weekend",
    "is_mega_sale", "is_ramadan",
    "is_pre_raya_window", "is_pre_cny_window",
    "viral_shock_active", "any_shock_active",
    "elasticity_coefficient", "demand_forecast",
    "inventory_level", "competitor_price",
    "base_price", "realized_demand",
    "stockout_flag",
]


def select_required_columns(df: pd.DataFrame, required: List[str], dataset_name: str) -> pd.DataFrame:
    """Keep only columns used by the training/evaluation pipeline."""
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"{dataset_name} dataset is missing required columns: {missing}")

    dropped = [col for col in df.columns if col not in required]
    if dropped:
        preview = ", ".join(dropped[:8])
        suffix = "..." if len(dropped) > 8 else ""
        console.print(
            f"  [dim]{dataset_name}: ignoring {len(dropped)} unused column(s): "
            f"{preview}{suffix}[/dim]"
        )

    return df.loc[:, required].copy()

# ── Create output directories ─────────────────────────────────────────────────
for key, path in CONFIG["paths"].items():
    if key != "demand_csv" and key != "rl_csv":
        Path(path).mkdir(parents=True, exist_ok=True)

# ── Checkpoint naming helper ──────────────────────────────────────────────────
def ckpt_path(model_type: str, cl_method: str, task_id: int, suffix="") -> str:
    base = Path(CONFIG["paths"]["checkpoints"]) / model_type / cl_method
    base.mkdir(parents=True, exist_ok=True)
    name = f"task{task_id}{('_' + suffix) if suffix else ''}.ckpt"
    return str(base / name)


# ─── Cell: Data Loading, Cleaning & Preprocessing ────────────────────────────

def load_and_clean(demand_path: str, rl_path: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    console.print("[bold]Loading datasets...[/bold]")
    demand_df = pd.read_csv(demand_path, parse_dates=["date"])
    rl_df     = pd.read_csv(rl_path,     parse_dates=["date"])

    demand_df = select_required_columns(demand_df, demand_required_columns(), "Demand")
    rl_df = select_required_columns(rl_df, RL_REQUIRED_COLUMNS, "RL")

    console.print(f"  Demand CSV  : {len(demand_df):,} rows × {demand_df.shape[1]} cols")
    console.print(f"  RL CSV      : {len(rl_df):,} rows × {rl_df.shape[1]} cols")

    # ── Clean demand_df ────────────────────────────────────────────────────
    demand_df = demand_df.sort_values(["product_id", "region_id", "date"])

    # Encode categoricals
    for col in ["product_id", "region_id", "product_category"]:
        if demand_df[col].dtype == object:
            demand_df[col] = demand_df[col].astype("category")

    # Fill remaining NaN with 0
    num_cols = demand_df.select_dtypes(include=[np.number]).columns
    demand_df[num_cols] = demand_df[num_cols].fillna(0)

    # ── Clean rl_df ────────────────────────────────────────────────────────
    rl_df = rl_df.sort_values(["product_id", "region_id", "date"])
    rl_num = rl_df.select_dtypes(include=[np.number]).columns
    rl_df[rl_num] = rl_df[rl_num].fillna(0)

    console.print(f"[green]✓ Data loaded and cleaned[/green]")
    return demand_df, rl_df


# ─── Cell: Feature Engineering for TFT ───────────────────────────────────────

def build_tft_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Prepare demand_df for PyTorch Forecasting TimeSeriesDataSet.
    Key requirement: time_idx must be a consecutive integer per group.
    We use a global time_idx across all dates (same for all groups).
    """
    df = df.copy()

    # Global time index: days since simulation start
    min_date = df["date"].min()
    df["time_idx"] = (df["date"] - min_date).dt.days.astype(int)

    # Encode categoricals as integers for pytorch_forecasting
    for col in ["product_id", "region_id", "product_category"]:
        df[col] = df[col].astype(str)

    # Ensure all known_reals exist and are float
    for col in CONFIG["forecasting"]["known_reals"]:
        if col in df.columns:
            df[col] = df[col].astype(float)
        else:
            df[col] = 0.0  # placeholder if column missing

    # Clip demand to [0, inf)
    df["demand"] = df["demand"].clip(lower=0).astype(float)

    return df.reset_index(drop=True)


def build_rl_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalise RL state features.
    These will be used as the state vector in the Gymnasium environment.
    """
    df = df.copy()

    # Normalise only the columns consumed by DynamicPricingEnv observations.
    max_competitor_price = df["competitor_price"].abs().max()
    if max_competitor_price > 0:
        df["competitor_price_norm"] = df["competitor_price"] / max_competitor_price

    demand_max = df["demand_forecast"].max()
    if demand_max > 0:
        df["demand_forecast_norm"] = df["demand_forecast"] / demand_max
    inv_max = df["inventory_level"].max()
    df["inventory_norm"] = df["inventory_level"] / max(inv_max, 1)

    return df


# ─── Cell: Split Data into 6 Continual Learning Tasks ────────────────────────

def split_tasks(
    tft_df: pd.DataFrame,
    rl_df:  pd.DataFrame,
    tasks:  List[Dict],
) -> Tuple[List[pd.DataFrame], List[pd.DataFrame]]:
    """
    Return two lists: tft_task_dfs[i] and rl_task_dfs[i]
    Each element is the subset of data for that task.
    """
    tft_tasks, rl_tasks = [], []

    for task in tasks:
        t_start = pd.Timestamp(task["start"])
        t_end   = pd.Timestamp(task["end"])

        tft_mask = (tft_df["date"] >= t_start) & (tft_df["date"] <= t_end)
        rl_mask  = (rl_df["date"]  >= t_start) & (rl_df["date"]  <= t_end)

        tft_sub = tft_df[tft_mask].copy()
        rl_sub  = rl_df[rl_mask].copy()

        tft_tasks.append(tft_sub)
        rl_tasks.append(rl_sub)

    return tft_tasks, rl_tasks


def is_vast_ai() -> bool:
    """Return True when the code appears to be running on a Vast.ai instance."""
    return bool(os.path.exists("/etc/vast_ai_instance") or os.environ.get("VAST_AI"))


def print_gpu_diagnostics() -> None:
    """Print the active CPU/GPU runtime so a notebook run can verify Vast.ai setup."""
    import platform
    import multiprocessing

    console.print("\n[bold]Runtime diagnostics[/bold]")
    console.print(f"  Python         : {platform.python_version()}")
    console.print(f"  PyTorch        : {torch.__version__}")
    console.print(f"  CUDA available : {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        free, total = torch.cuda.mem_get_info(0)
        console.print(f"  GPU            : {props.name}")
        console.print(f"  CUDA version   : {torch.version.cuda}")
        console.print(f"  Compute cap    : {props.major}.{props.minor}")
        console.print(f"  BF16 supported : {torch.cuda.is_bf16_supported()}")
        console.print(f"  VRAM free      : {free/1e9:.1f} / {total/1e9:.1f} GB")
    console.print(f"  CPU cores      : {multiprocessing.cpu_count()}")
    console.print(f"  Vast.ai        : {'YES' if is_vast_ai() else 'NO'}")
    console.print(f"  Active device  : {DEVICE.upper()}")
    console.print(f"  Precision      : {CONFIG['hardware']['precision']}")


def configure_vast_ai(
    data_dir: str | None = None,
    output_dir: str | None = None,
    require_gpu: bool = False,
) -> Dict:
    """
    Adjust paths for a rented Vast.ai machine without changing training code.

    data_dir should contain demand_forecasting.csv and rl_environment.csv.
    output_dir is where checkpoints/results/logs/plots will be written.
    """
    if require_gpu and not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. Choose a Vast.ai image with NVIDIA drivers and a CUDA PyTorch build.")

    if data_dir is not None:
        data_path = Path(data_dir).expanduser().resolve()
        CONFIG["paths"]["demand_csv"] = str(data_path / "demand_forecasting.csv")
        CONFIG["paths"]["rl_csv"] = str(data_path / "rl_environment.csv")

    if output_dir is not None:
        out_path = Path(output_dir).expanduser().resolve()
        CONFIG["paths"]["checkpoints"] = str(out_path / "checkpoints")
        CONFIG["paths"]["results"] = str(out_path / "results")
        CONFIG["paths"]["logs"] = str(out_path / "logs")
        CONFIG["paths"]["plots"] = str(out_path / "plots")

    CONFIG["hardware"]["device"] = DEVICE
    CONFIG["hardware"]["precision"] = "32"
    CONFIG["hardware"]["compile"] = os.environ.get("FYP_TORCH_COMPILE", "0") == "1"
    CONFIG["hardware"]["pin_memory"] = torch.cuda.is_available()

    ensure_output_dirs()
    print_gpu_diagnostics()
    return CONFIG


def ensure_output_dirs() -> None:
    """Create configured output directories."""
    for key, path in CONFIG["paths"].items():
        if key not in {"demand_csv", "rl_csv"}:
            Path(path).mkdir(parents=True, exist_ok=True)


def prepare_data(
    demand_path: str | None = None,
    rl_path: str | None = None,
) -> Tuple[List[pd.DataFrame], List[pd.DataFrame], pd.DataFrame, pd.DataFrame]:
    """Load, clean, engineer features, and split both datasets into CL tasks."""
    demand_path = demand_path or CONFIG["paths"]["demand_csv"]
    rl_path = rl_path or CONFIG["paths"]["rl_csv"]
    demand_df, rl_df = load_and_clean(demand_path, rl_path)
    tft_df = build_tft_dataframe(demand_df)
    rl_df = build_rl_features(rl_df)
    tft_tasks, rl_tasks = split_tasks(tft_df, rl_df, CONFIG["tasks"])
    return tft_tasks, rl_tasks, tft_df, rl_df


def print_task_summary(tft_tasks: List[pd.DataFrame], rl_tasks: List[pd.DataFrame]) -> None:
    """Print row counts for each continual-learning task."""
    table = Table(show_header=True, header_style="bold cyan")
    table.add_column("Task", style="bold")
    table.add_column("Name")
    table.add_column("Period")
    table.add_column("TFT rows")
    table.add_column("RL rows")
    for i, task in enumerate(CONFIG["tasks"]):
        table.add_row(
            str(task["task_id"]),
            task["name"],
            f"{task['start']} -> {task['end']}",
            f"{len(tft_tasks[i]):,}",
            f"{len(rl_tasks[i]):,}",
        )
    console.print(table)


ensure_output_dirs()
