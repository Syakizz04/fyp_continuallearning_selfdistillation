"""
Model and continual-learning machinery for the deployed FYP2 system.

Vendored from `hybrid_pipeline/trainers.py` (plus the three PPO callbacks that
lived in `hybrid_pipeline/experiment_runner.py`) so that `drift_pipeline` and
`edge_system` are self-contained. Before this, the entire FYP2 stack imported
its engine from an FYP1 *variant* package that FYP1 itself did not use - the
dependency ran backwards, and a config-syncing shim existed only to reconcile
the two packages' separate CONFIG dicts.

The code is a deliberate byte-level copy, not a rewrite: the base checkpoints in
`outputs/drift/checkpoints/` were trained by it, and `build_cltft` has to
reconstruct exactly that architecture for `base_tft.ckpt` to load. Changing
anything here risks silently failing to restore the base model.

Dropped as unused by FYP2: `compute_bwt_fwt` / `compute_forgetting` (drift has
its own `metrics.py`), and `ResultsLogger` / `CheckpointManager` (the drift
pipeline writes its own long-form CSVs).

The one substantive change is the config source. This module now reads
`drift_pipeline.core_pipeline.CONFIG` directly, so `base_training.sync_config()`
no longer has to copy drift's config into hybrid's before every call.
"""

import copy
import random
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import gymnasium as gym
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning as L
from darts import TimeSeries
from darts.metrics import mase as darts_mase
from gymnasium import spaces
from lightning.pytorch.callbacks import EarlyStopping, LearningRateMonitor
from lightning.pytorch.loggers import TensorBoardLogger
from pytorch_forecasting import TemporalFusionTransformer, TimeSeriesDataSet
from pytorch_forecasting.data import GroupNormalizer
from pytorch_forecasting.metrics import QuantileLoss
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv

from .core_pipeline import CONFIG, DEVICE, SEED, console

# ─── Cell: PyTorch Forecasting TimeSeriesDataSet Builder ─────────────────────
from pytorch_forecasting.data import GroupNormalizer

# Minimum rows needed for one encoder+prediction window. Keep this dynamic
# because smoke tests/notebooks may shrink CONFIG after importing the module.
def min_tft_rows() -> int:
    fc = CONFIG["forecasting"]
    return fc["encoder_length"] + fc["prediction_length"]


MIN_ROWS_NEEDED = min_tft_rows()

def make_tft_dataset(
    df: pd.DataFrame,
    train: bool = True,
    training_dataset: Optional[TimeSeriesDataSet] = None,
    allow_missing_timesteps: bool = True,
    predict_mode: bool = False,
) -> TimeSeriesDataSet:
    """
    Build a TimeSeriesDataSet for a given task's data slice.
    For validation/test splits, pass the training_dataset to align normalisation.
    """
    df = df.copy()

    # PyTorch Forecasting requires time_idx to be integer, not float/string/datetime.
    if "time_idx" not in df.columns:
        raise ValueError("Missing required column: time_idx")

    if df["time_idx"].isna().any():
        raise ValueError("time_idx contains missing values")

    df["time_idx"] = df["time_idx"].astype("int64")

    # Group IDs and static categoricals must be string/categorical-friendly.
    for col in CONFIG["forecasting"]["group_ids"]:
        if col in df.columns:
            df[col] = df[col].astype(str)

    for col in CONFIG["forecasting"]["static_categoricals"]:
        if col in df.columns:
            df[col] = df[col].astype(str)

    # Target should be numeric.
    df["demand"] = pd.to_numeric(df["demand"], errors="coerce")
    if df["demand"].isna().any():
        raise ValueError("demand contains non-numeric or missing values after conversion")

    enc_len  = CONFIG["forecasting"]["encoder_length"]
    pred_len = CONFIG["forecasting"]["prediction_length"]

    known_reals = [c for c in CONFIG["forecasting"]["known_reals"] if c in df.columns]

    # Known real covariates should be numeric.
    for col in known_reals:
        df[col] = pd.to_numeric(df[col], errors="coerce")
        if df[col].isna().any():
            df[col] = df[col].fillna(0)

    if train or training_dataset is None:
        ds = TimeSeriesDataSet(
            df,
            time_idx="time_idx",
            target="demand",
            group_ids=CONFIG["forecasting"]["group_ids"],
            min_encoder_length=enc_len // 2,
            max_encoder_length=enc_len,
            min_prediction_length=1,
            max_prediction_length=pred_len,
            static_categoricals=CONFIG["forecasting"]["static_categoricals"],
            static_reals=[],
            time_varying_known_reals=known_reals,
            time_varying_unknown_reals=["demand"],
            target_normalizer=GroupNormalizer(
                groups=CONFIG["forecasting"]["group_ids"],
                transformation="softplus",
            ),
            add_relative_time_idx=True,
            add_target_scales=True,
            add_encoder_length=True,
            allow_missing_timesteps=allow_missing_timesteps,
        )
    else:
        ds = TimeSeriesDataSet.from_dataset(
            training_dataset,
            df,
            predict=predict_mode,
            stop_randomization=True,
            allow_missing_timesteps=allow_missing_timesteps,
        )

    return ds


def filter_tft_eval_frame(df: pd.DataFrame, min_length: int = None) -> pd.DataFrame:
    """
    Keep only groups with enough rows for at least one encoder/decoder window.
    This prevents TimeSeriesDataSet from filtering everything and then failing
    downstream in sklearn scalers with a 0-sample array.
    """
    fc = CONFIG["forecasting"]
    group_ids = fc["group_ids"]
    min_length = min_length or min_tft_rows()

    if df is None or df.empty:
        return pd.DataFrame(columns=[] if df is None else df.columns)

    kept = []
    dropped = []
    for group_key, group_df in df.groupby(group_ids, observed=False):
        if len(group_df) >= min_length:
            kept.append(group_df)
        else:
            dropped.append((group_key, len(group_df)))

    if dropped:
        preview = ", ".join(f"{key}:{n}" for key, n in dropped[:5])
        suffix = "..." if len(dropped) > 5 else ""
        console.print(
            f"  [yellow]Skipped {len(dropped)} eval group(s) with fewer than "
            f"{min_length} rows: {preview}{suffix}[/yellow]"
        )

    if not kept:
        console.print(
            f"  [yellow]No valid eval groups remain after requiring {min_length} rows[/yellow]"
        )
        return df.iloc[0:0].copy()

    return pd.concat(kept, axis=0).sort_values(group_ids + ["time_idx"]).reset_index(drop=True)


def make_tft_loaders(
    train_df: pd.DataFrame,
    val_df:   Optional[pd.DataFrame] = None,
    training_dataset: Optional[TimeSeriesDataSet] = None,
) -> Tuple:
    """
    Build train DataLoader (and optionally val DataLoader).
    Returns (train_ds, train_loader, val_loader_or_None).
    """
    hw = CONFIG["hardware"]
    fc = CONFIG["forecasting"]

    train_df = train_df.sort_values(CONFIG["forecasting"]["group_ids"] + ["time_idx"]).copy()
    enc_len = fc["encoder_length"]
    min_rows_needed = min_tft_rows()

    # 80/20 split within task if no explicit val_df given. Keep encoder context
    # in val_df so each series has enough history for validation windows.
    if val_df is None and len(train_df) > min_rows_needed * 2:
        unique_times = np.sort(train_df["time_idx"].unique())
        cutoff = int(len(unique_times) * 0.8)
        split_idx = unique_times[cutoff]
        val_start = max(unique_times[0], split_idx - enc_len)
        val_df = train_df[train_df["time_idx"] >= val_start].copy()
        train_df = train_df[train_df["time_idx"] < split_idx].copy()
    elif val_df is not None:
        val_df = val_df.sort_values(CONFIG["forecasting"]["group_ids"] + ["time_idx"]).copy()

    train_ds = make_tft_dataset(train_df, train=True)
    val_base_ds = training_dataset or train_ds

    loader_kwargs = dict(
        batch_size   = fc["batch_size"],
        num_workers  = hw["num_workers"],
        pin_memory   = hw["pin_memory"],
        persistent_workers = hw["persistent_workers"] if hw["num_workers"] > 0 else False,
    )

    train_loader = train_ds.to_dataloader(train=True, shuffle=True,  **loader_kwargs)

    val_loader = None
    if val_df is not None and len(val_df) >= min_rows_needed:
        val_df = filter_tft_eval_frame(val_df, min_length=min_rows_needed)
        if val_df.empty:
            return train_ds, train_loader, None
        val_ds     = make_tft_dataset(val_df, train=False, training_dataset=val_base_ds)
        val_loader = val_ds.to_dataloader(train=False, shuffle=False, **loader_kwargs)

    return train_ds, train_loader, val_loader




# ─── Cell: CL-Aware TFT Model ────────────────────────────────────────────────
# Subclass of TemporalFusionTransformer that injects EWC / SDFT losses
# into training_step. Replay is handled at the DataLoader level.

class CLTFT(TemporalFusionTransformer):
    """
    CL-aware Temporal Fusion Transformer.

    Supported cl_method values:
        "naive"  — standard fine-tuning, no forgetting mitigation
        "ewc"    — adds diagonal EWC penalty to loss
        "replay" — no model change (replay mixed into DataLoader externally)
        "sdft"   — adds soft distillation loss from frozen teacher model
    """

    def __init__(self, *args, cl_method: str = "naive", cl_cfg: dict = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.cl_method  = cl_method
        self.cl_cfg     = cl_cfg or CONFIG["cl"]

        # EWC state
        self.ewc_fisher  : Dict[str, torch.Tensor] = {}
        self.ewc_optparams: Dict[str, torch.Tensor] = {}
        # Device-resident copies of Fisher/opt-params, built lazily in _ewc_loss
        # and reused across steps; invalidated whenever Fisher is recomputed.
        self._ewc_dev_cache = None

        # SDFT teacher (frozen copy of previous model)
        self.teacher     : Optional["CLTFT"] = None
        self.drift_score = 0.0
        self.adaptive_ewc_scale = 1.0
        self.adaptive_distill_scale = 1.0

    def _move_to_device(self, obj):
        if isinstance(obj, torch.Tensor):
            return obj.to(self.device)
        if isinstance(obj, dict):
            return {k: self._move_to_device(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [self._move_to_device(v) for v in obj]
        if isinstance(obj, tuple):
            return tuple(self._move_to_device(v) for v in obj)
        return obj

    def _prediction_from_output(self, output):
        if isinstance(output, dict):
            return output.get("prediction", output.get("output"))
        if hasattr(output, "prediction"):
            return output.prediction
        return output

    def _add_auxiliary_loss(self, step_output, aux_loss: torch.Tensor):
        if isinstance(step_output, dict):
            if "loss" not in step_output:
                raise TypeError("Expected training_step output dict to contain a 'loss' key")
            merged = dict(step_output)
            merged["loss"] = merged["loss"] + aux_loss
            return merged
        if isinstance(step_output, torch.Tensor):
            return step_output + aux_loss
        raise TypeError(f"Unsupported training_step output type: {type(step_output)!r}")

    def create_log(self, *args, **kwargs):
        """Disable PyTorch Forecasting's figure logging; scalar logs still work."""
        return {}

    def log_interpretation(self, *args, **kwargs):
        """Disable TFT interpretation figures, which can crash TensorBoard image conversion."""
        return None

    def on_epoch_end(self, *args, **kwargs):
        """Skip TFT epoch-end interpretation logging."""
        return None

    # ── Training step override ─────────────────────────────────────────────
    def training_step(self, batch, batch_idx):
        step_output = super().training_step(batch, batch_idx)

        if self.cl_method in {"ewc", "recall_ewc"} and self.ewc_fisher:
            ewc_penalty = self._ewc_loss()
            self.log("ewc_penalty", ewc_penalty.detach(), prog_bar=False, on_step=True)
            step_output = self._add_auxiliary_loss(step_output, ewc_penalty)

        elif self.cl_method == "drift_adaptive_replay_ewc" and self.ewc_fisher:
            ewc_penalty = self._ewc_loss() * self.adaptive_ewc_scale
            self.log("drift_replay_ewc_penalty", ewc_penalty.detach(), prog_bar=False, on_step=True)
            step_output = self._add_auxiliary_loss(step_output, ewc_penalty)

        elif self.cl_method == "adaptive_drift":
            if self.ewc_fisher:
                ewc_penalty = self._ewc_loss() * self.adaptive_ewc_scale
                self.log("adaptive_ewc_penalty", ewc_penalty.detach(), prog_bar=False, on_step=True)
                step_output = self._add_auxiliary_loss(step_output, ewc_penalty)

            if self.teacher is not None and self.adaptive_distill_scale > 0:
                # Additive, drift-scaled distillation (this method's own design).
                # Keep the (1-alpha) weighting that used to live inside _sdft_loss.
                distill_w    = (1.0 - self.cl_cfg["sdft_alpha"])
                sdft_penalty = distill_w * self._sdft_loss(batch) * self.adaptive_distill_scale
                self.log("adaptive_distill_penalty", sdft_penalty.detach(), prog_bar=False, on_step=True)
                step_output = self._add_auxiliary_loss(step_output, sdft_penalty)

        elif self.cl_method == "sdft" and self.teacher is not None:
            alpha   = self.cl_cfg["sdft_alpha"]
            distill = self._sdft_loss(batch)
            self.log("sdft_penalty", distill.detach(), prog_bar=False, on_step=True)
            # Convex blend: alpha * task_loss + (1 - alpha) * distillation.
            # The task loss must actually be scaled by alpha — super() returns it
            # unweighted, so this is where the SDFT trade-off is really applied.
            if isinstance(step_output, dict):
                step_output = dict(step_output)
                step_output["loss"] = alpha * step_output["loss"] + (1.0 - alpha) * distill
            else:
                step_output = alpha * step_output + (1.0 - alpha) * distill

        return step_output

    # ── EWC penalty ────────────────────────────────────────────────────────
    def _ewc_loss(self) -> torch.Tensor:
        lam = self.cl_cfg["ewc_lambda"]
        # Fisher/opt-params are stored on CPU (for cross-task accumulation and
        # checkpointing). Move them to the compute device ONCE and cache, instead
        # of re-transferring the whole model's tensors on every training step.
        if self._ewc_dev_cache is None:
            self._ewc_dev_cache = (
                {n: f.to(self.device) for n, f in self.ewc_fisher.items()},
                {n: p.to(self.device) for n, p in self.ewc_optparams.items()},
            )
        fisher_dev, opt_dev = self._ewc_dev_cache

        penalty = torch.tensor(0.0, device=self.device)
        for name, param in self.named_parameters():
            if name in fisher_dev:
                penalty += (fisher_dev[name] * (param - opt_dev[name]).pow(2)).sum()
        return (lam / 2.0) * penalty

    def compute_and_store_fisher(self, dataloader, n_batches: int = 200):
        """
        Compute diagonal Fisher Information Matrix over dataloader.
        Called AFTER training on a task, BEFORE moving to the next.
        """
        self.train()
        fisher_acc: Dict[str, torch.Tensor] = {
            n: torch.zeros_like(p)
            for n, p in self.named_parameters() if p.requires_grad
        }

        count = 0
        for batch in dataloader:
            if count >= n_batches:
                break
            try:
                self.zero_grad()
                x, y = batch
                x = self._move_to_device(x)
                y = self._move_to_device(y)

                out  = self(x)
                pred = self._prediction_from_output(out)

                # Use sum of output as proxy loss for Fisher estimation
                loss = pred.sum()
                loss.backward()

                for name, param in self.named_parameters():
                    if param.requires_grad and param.grad is not None:
                        fisher_acc[name] += param.grad.detach().pow(2)
                count += 1
            except Exception:
                continue

        if count > 0:
            for name in fisher_acc:
                fisher_acc[name] /= count
                # Accumulate Fisher across tasks
                if name in self.ewc_fisher:
                    self.ewc_fisher[name] = (
                        self.ewc_fisher[name].cpu() + fisher_acc[name].cpu()
                    )
                else:
                    self.ewc_fisher[name] = fisher_acc[name].cpu()

        # Store optimal parameters at this task
        self.ewc_optparams = {
            n: p.detach().cpu().clone()
            for n, p in self.named_parameters() if p.requires_grad
        }
        # Fisher/opt-params changed → drop the device cache so _ewc_loss rebuilds it.
        self._ewc_dev_cache = None
        self.train()
        console.print(f"  [cyan]Fisher computed over {count} batches[/cyan]")

    # ── SDFT: self-distillation loss ───────────────────────────────────────
    def _sdft_loss(self, batch) -> torch.Tensor:
        """Raw self-distillation term: MSE between student and frozen-teacher
        predictions. No temperature — for MSE regression a temperature is just a
        constant rescaling (MSE(a/T, b/T) = MSE(a, b)/T^2) and changes nothing.
        No alpha here either; callers apply their own weighting (convex blend for
        the base 'sdft' method, drift-scaled additive term for 'adaptive_drift')."""
        x, y = batch
        x_dev = self._move_to_device(x)

        with torch.no_grad():
            self.teacher.eval()
            self.teacher.to(self.device)
            t_out  = self.teacher(x_dev)
            t_pred = self._prediction_from_output(t_out)

        s_out  = self(x_dev)
        s_pred = self._prediction_from_output(s_out)

        return F.mse_loss(s_pred, t_pred.detach())

    def store_teacher(self):
        """Store a frozen deep copy of self as the teacher for next task."""
        self.teacher = copy.deepcopy(self)
        self.teacher.teacher = None
        self.teacher.eval()
        for p in self.teacher.parameters():
            p.requires_grad_(False)

    def set_adaptive_strengths(
        self,
        drift_score: float,
        ewc_scale: float,
        distill_scale: float,
    ) -> None:
        """Set replay-free adaptive regularization strengths for this task."""
        self.drift_score = float(drift_score)
        self.adaptive_ewc_scale = float(ewc_scale)
        self.adaptive_distill_scale = float(distill_scale)


def build_cltft(training_dataset: TimeSeriesDataSet, cl_method: str) -> CLTFT:
    """Instantiate CLTFT from a training dataset."""
    fc = CONFIG["forecasting"]
    model = CLTFT.from_dataset(
        training_dataset,
        learning_rate          = fc["learning_rate"],
        hidden_size            = fc["hidden_size"],
        attention_head_size    = fc["attention_head_size"],
        dropout                = fc["dropout"],
        hidden_continuous_size = fc["hidden_continuous_size"],
        output_size            = fc["output_size"],
        loss                   = QuantileLoss(),
        log_interval           = -1,
        log_val_interval       = -1,
        reduce_on_plateau_patience = 3,
        cl_method              = cl_method,
        cl_cfg                 = CONFIG["cl"],
    )
    return model




# ─── Cell: Gymnasium Dynamic Pricing Environment ──────────────────────────────

class DynamicPricingEnv(gym.Env):
    """
    Discrete-action dynamic pricing environment backed by rl_environment.csv.

    State (9 features):
        [demand_forecast_norm, inventory_norm, competitor_price_norm,
         day_of_week/6, month/12, is_weekend, snap, is_event, elasticity_norm]

    This was 13 until an audit measured each slot on the real data. Four -
    is_pre_raya_window, is_pre_cny_window, viral_shock_active, any_shock_active -
    were constant zero, being FYP1's Malaysian calendar and synthetic shock
    generator carried onto Walmart California data. Two more were M5's `snap`
    and `is_event` renamed to `is_mega_sale` and `is_ramadan` by a config remap.
    Constant inputs are harmless to a network, but describing a 13-dim state when
    9 dimensions carry signal is not, so the dead slots are gone and the real
    ones go by their real names.

    Action (Discrete 11):
        Price adjustment tiers from -10% to +10% in 2% increments.
        Applied to base_price to set the simulated price.

    Reward:
        Computed online from simulated profit margin and penalised by
        0.15 * revenue if stockout occurs, then scaled by 1/1000.
    """

    metadata = {"render_modes": []}

    STATE_DIM   = 9

    def __init__(self, task_df: pd.DataFrame, seed: int = SEED):
        super().__init__()
        self.df      = task_df.reset_index(drop=True)
        self.n_steps = len(self.df)
        self._seed   = seed
        self.price_tiers = list(CONFIG["rl"]["price_tiers"])

        # Off by default: turning it on changes the reward, so FYP1's RL numbers
        # would no longer be comparable. Read from CONFIG at construction time so
        # a scenario can enable it the same way everything else is configured.
        self.inventory_constrained = bool(
            CONFIG["rl"].get("inventory_constrained", False))
        self.lost_sale_penalty = float(
            CONFIG["rl"].get("lost_sale_penalty", 0.5))

        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(self.STATE_DIM,), dtype=np.float32,
        )
        self.action_space = spaces.Discrete(len(self.price_tiers))

        # Precompute needed columns
        self._precompute()
        self.reset(seed=seed)

    def _precompute(self):
        df = self.df

        def column(col, fallback):
            """Full-length float array for `col`, or `fallback` everywhere.

            NOT `df.get(col, pd.Series(x))`: that fallback is a length-1 Series,
            so a missing column does not default - it raises IndexError on the
            second step, once the env indexes past position 0. The failure lands
            in `_get_obs` at run time rather than at construction, which is the
            worst place for it.
            """
            if col not in df.columns:
                return np.full(len(df), fallback, dtype=np.float32)
            return df[col].to_numpy(dtype=np.float32)

        def safe_norm(col, fallback=0.0):
            v = column(col, fallback)
            m = np.abs(v).max()
            return v / m if m > 0 else v

        self._demand_forecast = safe_norm("demand_forecast_norm",  0.5)
        self._inventory       = safe_norm("inventory_norm",        0.5)
        self._comp_price      = safe_norm("competitor_price_norm",  0.5)
        self._day_of_week     = column("day_of_week", 0.0) / 6.0
        self._month           = column("month",       1.0) / 12.0
        self._is_weekend      = column("is_weekend",  0.0)
        # Real M5 regime signal, under its own name. `snap` marks SNAP benefit
        # distribution days, which are a genuine FOODS demand driver (32.9% of
        # days); `is_event` marks M5 calendar events (8.0%).
        self._snap            = column("snap",     0.0)
        self._is_event        = column("is_event", 0.0)
        self._elasticity      = safe_norm("elasticity_coefficient", -1.5)

        # Price info
        self._base_price      = column("base_price",      100.0)
        self._realized_demand = column("realized_demand",  50.0)
        self._unit_cost       = column("base_price",       50.0) * 0.55
        self._stockout        = column("stockout_flag",     0.0)
        # RAW inventory, not the normalised observation. Used only when
        # `inventory_constrained` is on - see step().
        self._inventory_raw   = column("inventory_level", np.inf)

        if self.inventory_constrained and "inventory_level" in df.columns:
            # Re-scale the inventory OBSERVATION to days of cover.
            #
            # `safe_norm` divides by the maximum across the whole frame, which is
            # the largest stock level of the busiest SKU. On this panel that is
            # 205 units, so a slow-moving SKU sitting on 1, 2 or 3 units maps to
            # 0.005, 0.010, 0.015 - indistinguishable to the network, and those
            # are precisely the states where stock should change the price. The
            # measured symptom was an agent that responded to inventory LEAST
            # when stock was scarcest (0% of decisions at under 2 days of cover,
            # against 6% when stock was ample).
            #
            # Days of cover is the scale the decision actually turns on: "two
            # days left" means the same thing for a fast and a slow SKU, whereas
            # "ten units" does not. Clipped at two weeks, beyond which more stock
            # makes no difference to a pricing decision.
            mean_d = (df.groupby("product_id")["realized_demand"].transform("mean")
                      if "product_id" in df.columns
                      else pd.Series(df["realized_demand"].mean(), index=df.index))
            self._mean_demand = np.maximum(
                mean_d.to_numpy(dtype=np.float32), 1e-6).astype(np.float32)
            cover = (df["inventory_level"].to_numpy(dtype=np.float32)
                     / self._mean_demand)
            self._inventory = np.clip(cover / self.COVER_CLIP_DAYS,
                                      0.0, 1.0).astype(np.float32)
        else:
            self._mean_demand = None

        # Oracle best reward per step, precomputed once and vectorized over price
        # tiers. The oracle depends only on the data (not the policy), so there is
        # no need to recompute it per step/episode during evaluation.
        tiers   = np.asarray(self.price_tiers, dtype=np.float32)            # (T,)
        base_p  = np.maximum(self._base_price, 1e-6)[:, None]               # (N,1)
        price   = self._base_price[:, None] * (1.0 + tiers)[None, :]        # (N,T)
        ratio   = price / base_p                                           # (N,T)
        demand  = np.maximum(self._realized_demand, 0.0)[:, None]          # (N,1)
        adj     = demand * np.power(ratio, self._elasticity[:, None])      # (N,T)
        adj     = np.maximum(0.0, adj)
        if self.inventory_constrained:
            # The oracle must face the same constraint as the policy, or regret
            # and profit_index are measured against an unreachable benchmark.
            stock = self._inventory_raw[:, None]
            sold  = np.minimum(adj, stock)
            lost  = adj - sold
            profit = ((price - self._unit_cost[:, None]) * sold
                      - self.lost_sale_penalty * price * lost)
        else:
            profit = (price - self._unit_cost[:, None]) * adj              # (N,T)
        self._opt_reward = (profit.max(axis=1) / 1000.0).astype(np.float32)  # (N,)

    #: Stock beyond this many days of cover makes no difference to a price.
    COVER_CLIP_DAYS = 14.0

    def inventory_obs(self, idx: int, inventory_level: float) -> float:
        """
        Map a RAW stock level onto this row's inventory observation slot.

        Exposed so a serving process can substitute a different stock figure -
        a node's stale belief, say - without having to reconstruct the
        normalisation. Reconstructing it is how the served observation and the
        trained one drift apart, and under the cover-based scaling it is not even
        possible to invert from a single row, because the divisor is per-SKU.
        """
        if self._mean_demand is not None:
            cover = inventory_level / float(self._mean_demand[idx])
            return float(np.clip(cover / self.COVER_CLIP_DAYS, 0.0, 1.0))
        # Legacy path: inventory was normalised twice (global max in
        # build_rl_features, then this frame's max), so recover the composite
        # divisor empirically from a row where the normalised value is non-zero.
        raw = self._inventory_raw
        norm = np.asarray(self._inventory, dtype=float)
        usable = np.flatnonzero((norm > 0) & np.isfinite(norm) & (raw > 0))
        if not len(usable):
            return 0.0
        i = int(usable[0])
        scale = float(raw[i] / norm[i])
        return float(inventory_level / scale) if scale else 0.0

    def _get_obs(self, idx: int) -> np.ndarray:
        return np.array([
            self._demand_forecast[idx],
            self._inventory[idx],
            self._comp_price[idx],
            float(self._day_of_week[idx]),
            float(self._month[idx]),
            self._is_weekend[idx],
            self._snap[idx],
            self._is_event[idx],
            self._elasticity[idx],
        ], dtype=np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        return self._get_obs(0), {}

    def step(self, action: int):
        idx   = self.current_step
        tier  = self.price_tiers[int(action)]
        price = float(self._base_price[idx]) * (1.0 + tier)
        cost  = float(self._unit_cost[idx])

        # Price-elasticity adjusted demand (simplified inline)
        base_demand = max(float(self._realized_demand[idx]), 0)
        elasticity  = float(self._elasticity[idx])
        base_p      = float(self._base_price[idx])
        price_ratio = price / max(base_p, 1e-6)
        adj_demand  = base_demand * (price_ratio ** elasticity)
        adj_demand  = max(0.0, round(adj_demand))

        if self.inventory_constrained:
            # Sales are capped by stock on hand, and demand generated but not
            # served is charged for.
            #
            # WITHOUT this, `inventory_level` is an observation with no causal
            # path to reward: the agent cannot be rewarded or punished for
            # pricing against it, so it correctly learns to ignore it. That was
            # measured - sweeping the inventory input across its whole range
            # moved under 5% of pricing decisions, and retraining on a realistic
            # inventory series did NOT change that, because the defect is here in
            # the reward rather than in the data.
            #
            # With it, the economics the project claims to model actually bind:
            # cutting price into low stock manufactures demand that cannot be
            # served, so the agent has a reason to raise price as stock falls -
            # and a reason to care whether its stock figure is correct, which is
            # the premise of the whole staleness experiment.
            stock       = float(self._inventory_raw[idx])
            sold        = min(adj_demand, stock)
            lost        = adj_demand - sold
            revenue     = price * sold
            profit_margin = (price - cost) * sold
            stockout_pen  = self.lost_sale_penalty * price * lost
            # Endogenous now: a consequence of the agent's own pricing against
            # the stock it had, not a flag read off the dataset row.
            stockout      = 1.0 if lost > 0 else 0.0
        else:
            # Original behaviour, preserved so FYP1's RL results stay comparable.
            # `stockout` here is exogenous - read from the dataset row, identical
            # whatever the agent does - so it shifts the reward without ever
            # depending on the action.
            revenue       = price * adj_demand
            profit_margin = (price - cost) * adj_demand
            stockout      = float(self._stockout[idx])
            stockout_pen  = 0.15 * revenue * stockout
            sold, lost    = adj_demand, 0.0

        reward = (profit_margin - stockout_pen) / 1000.0  # scale to ~[-1, 5]
        reward = float(np.clip(reward, -5.0, 5.0))

        self.current_step += 1
        done = self.current_step >= self.n_steps
        obs  = self._get_obs(min(self.current_step, self.n_steps - 1))

        info = {
            "price"        : price,
            "adj_demand"   : adj_demand,
            "revenue"      : revenue,
            "profit_margin": profit_margin,
            "stockout"     : stockout,
        }
        return obs, reward, done, False, info

    def optimal_reward(self, idx: int) -> float:
        """Oracle best reward at step idx — O(1) lookup into the precomputed
        per-step array (see _precompute)."""
        return float(self._opt_reward[idx])


def make_pricing_env(task_df: pd.DataFrame) -> DummyVecEnv:
    return DummyVecEnv([lambda df=task_df: DynamicPricingEnv(df)])




# ─── Cell: Metric Functions ───────────────────────────────────────────────────

# ── Forecasting Metrics ────────────────────────────────────────────────────────

def compute_mase(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_train: np.ndarray,
    seasonality: int = 7,
) -> float:
    """
    Mean Absolute Scaled Error using a historical in-sample baseline.

    Darts requires explicit time alignment where the insample series ends before
    the prediction series begins. The experiment evaluates flattened batches
    from multiple product/region series, so using Darts' default indexes makes
    the train and prediction series start at the same point and raises:
    "insample series must start before the pred_series". Computing MASE directly
    avoids that indexing failure while preserving the metric definition.
    """
    y_true = np.asarray(y_true, dtype=float).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=float).reshape(-1)
    y_train = np.asarray(y_train, dtype=float).reshape(-1)

    if len(y_true) == 0 or len(y_pred) == 0:
        return np.nan
    if len(y_true) != len(y_pred):
        n = min(len(y_true), len(y_pred))
        y_true = y_true[:n]
        y_pred = y_pred[:n]

    mae = np.mean(np.abs(y_true - y_pred))
    if len(y_train) <= seasonality:
        naive = np.mean(np.abs(np.diff(y_train))) if len(y_train) > 1 else np.nan
    else:
        naive = np.mean(np.abs(y_train[seasonality:] - y_train[:-seasonality]))

    if not np.isfinite(naive) or naive <= 1e-8:
        return np.nan
    return float(mae / naive)


def compute_smape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    denom = (np.abs(y_true) + np.abs(y_pred)) / 2.0
    return float(np.mean(np.abs(y_true - y_pred) / np.maximum(denom, 1e-8)) * 100)


def compute_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def evaluate_forecasting(
    model: CLTFT,
    val_loader,
    task_train_df: pd.DataFrame,
    task_id: int,
) -> Dict[str, float]:
    """
    Run inference on val_loader and compute MASE, sMAPE, RMSE.
    Returns dict of metric_name → value.
    """
    was_training = model.training
    model.eval()
    all_preds, all_true = [], []

    try:
        with torch.no_grad():
            for batch in val_loader:
                x, y = batch
                if hasattr(model, "_move_to_device"):
                    x = model._move_to_device(x)
                    y = model._move_to_device(y)
                else:
                    device = next(model.parameters()).device
                    x = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                         for k, v in x.items()}
                    if isinstance(y, torch.Tensor):
                        y = y.to(device)
                    elif isinstance(y, (list, tuple)):
                        y = type(y)(v.to(device) if isinstance(v, torch.Tensor) else v for v in y)

                out  = model(x)
                if isinstance(out, dict):
                    pred = out["prediction"]
                elif hasattr(out, "prediction"):
                    pred = out.prediction
                else:
                    pred = out
                # Use median quantile (index 3 of 7 quantiles)
                if pred.dim() == 3:
                    pred = pred[:, :, 3]
                true = y[0] if isinstance(y, (list, tuple)) else y
                all_preds.append(pred.detach().float().cpu().numpy())
                all_true.append(true.detach().float().cpu().numpy())
    except Exception as e:
        console.print(f"  [red]Forecast eval error: {e}[/red]")
        return {"mase": np.nan, "smape": np.nan, "rmse": np.nan}
    finally:
        if was_training:
            model.train()

    if not all_preds:
        return {"mase": np.nan, "smape": np.nan, "rmse": np.nan}

    y_pred  = np.concatenate([p.reshape(-1) for p in all_preds])
    y_true  = np.concatenate([t.reshape(-1) for t in all_true])
    y_train = task_train_df["demand"].values.astype(float)

    return {
        "mase" : compute_mase(y_true, y_pred, y_train),
        "smape": compute_smape(y_true, y_pred),
        "rmse" : compute_rmse(y_true, y_pred),
    }


# ── RL Metrics ─────────────────────────────────────────────────────────────────

def evaluate_rl(
    model: PPO,
    task_df: pd.DataFrame,
    n_episodes: int = None,   # accepted for API compatibility; see note below
) -> Dict[str, float]:
    """
    Run the PPO agent on task_df and compute pricing metrics.

    DynamicPricingEnv is fully deterministic (no randomness in reset/step) and we
    evaluate with a deterministic policy, so every episode over a given task is
    byte-for-byte identical. We therefore run exactly one pass and ignore
    n_episodes — running N would only multiply cumulative_profit by N (an
    artifact) while leaving the per-step reward/regret means unchanged.
    """
    env = DynamicPricingEnv(task_df)

    obs, _ = env.reset()
    ep_reward, ep_profit = 0.0, 0.0
    total_regret = 0.0
    total_steps  = 0
    done = False

    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, _, info = env.step(int(action))
        ep_reward += reward
        ep_profit += info.get("profit_margin", 0.0)

        # Regret: oracle - actual (both in raw MYR)
        oracle = env.optimal_reward(min(env.current_step - 1, env.n_steps - 1))
        total_regret += max(0.0, oracle - reward)
        total_steps  += 1

    return {
        "avg_episode_reward": float(ep_reward),
        "cumulative_profit" : float(ep_profit),
        "pricing_regret"    : float(total_regret / max(total_steps, 1)),
    }


# ─── Cell: Shared CL Engines (EWC, Replay, SDFT) ─────────────────────────────

# ── 1. Forecasting Replay Buffer ───────────────────────────────────────────────

class ForecastingReplayBuffer:
    """
    Stores (batch_x, batch_y) samples from past tasks.
    Keeps memory roughly balanced per task and samples evenly across tasks.
    """

    def __init__(self, capacity: int):
        self.capacity = capacity
        self.buffer_by_task: Dict[int, List[Tuple]] = defaultdict(list)
        self._seen_by_task: Dict[int, int] = defaultdict(int)

    def _task_capacity(self) -> int:
        n_tasks = max(len(self.buffer_by_task), 1)
        return max(1, self.capacity // n_tasks)

    def _rebalance(self):
        cap = self._task_capacity()
        for task_id, task_buf in list(self.buffer_by_task.items()):
            if len(task_buf) > cap:
                self.buffer_by_task[task_id] = random.sample(task_buf, cap)

    def add_from_loader(self, dataloader, task_id: int, n_samples: int = None):
        n_samples = n_samples or self.capacity
        added = 0
        task_buf = self.buffer_by_task[int(task_id)]
        self._rebalance()

        for batch in dataloader:
            if added >= n_samples:
                break
            x, y = batch
            # Store CPU tensors to save GPU memory
            x_cpu = {k: v.cpu() if isinstance(v, torch.Tensor) else v for k, v in x.items()}
            y_cpu = [v.cpu() if isinstance(v, torch.Tensor) else v for v in y]
            sample = (x_cpu, y_cpu)

            task_cap = self._task_capacity()
            seen = self._seen_by_task[int(task_id)]
            if len(task_buf) < task_cap:
                task_buf.append(sample)
            else:
                # Per-task reservoir sampling.
                idx = random.randint(0, seen)
                if idx < task_cap:
                    task_buf[idx] = sample
            self._seen_by_task[int(task_id)] += 1
            added += 1
        self._rebalance()

    def sample(self, n: int) -> List[Tuple]:
        n = min(n, len(self))
        tasks = [task_id for task_id, task_buf in self.buffer_by_task.items() if task_buf]
        if n <= 0 or not tasks:
            return []

        samples = []
        while len(samples) < n and tasks:
            random.shuffle(tasks)
            for task_id in tasks:
                task_buf = self.buffer_by_task[task_id]
                if task_buf:
                    samples.append(random.choice(task_buf))
                if len(samples) >= n:
                    break
        return samples

    def __len__(self):
        return sum(len(task_buf) for task_buf in self.buffer_by_task.values())


# ── 2. RECALL-style RL Replay Buffer ──────────────────────────────────────────

class RLReplayBuffer:
    """
    Stores (obs, action, reward, next_obs, done) tuples from past RL tasks.
    Used by RECALL-style method to augment PPO rollouts with task-balanced
    sampling.
    """

    def __init__(self, capacity: int):
        self.capacity = capacity
        self.transitions_by_task: Dict[int, List[Tuple]] = defaultdict(list)
        self._seen_by_task: Dict[int, int] = defaultdict(int)

    def _task_capacity(self) -> int:
        n_tasks = max(len(self.transitions_by_task), 1)
        return max(1, self.capacity // n_tasks)

    def _rebalance(self):
        cap = self._task_capacity()
        for task_id, task_buf in list(self.transitions_by_task.items()):
            if len(task_buf) > cap:
                self.transitions_by_task[task_id] = random.sample(task_buf, cap)

    def add_from_env_rollout(self, env: DynamicPricingEnv, policy: PPO, n: int, task_id: int):
        """Collect n transitions from env using policy."""
        obs, _ = env.reset()
        added  = 0
        task_id = int(task_id)
        task_buf = self.transitions_by_task[task_id]
        self._rebalance()

        while added < n:
            action, _ = policy.predict(obs, deterministic=False)
            next_obs, reward, done, _, _ = env.step(int(action))
            transition = (
                obs.copy(),
                int(action),
                float(reward),
                next_obs.copy(),
                bool(done),
            )

            task_cap = self._task_capacity()
            seen = self._seen_by_task[task_id]
            if len(task_buf) < task_cap:
                task_buf.append(transition)
            else:
                idx = random.randint(0, seen)
                if idx < task_cap:
                    task_buf[idx] = transition

            obs = next_obs
            if done:
                obs, _ = env.reset()
            self._seen_by_task[task_id] += 1
            added += 1
        self._rebalance()

    def sample(self, n: int) -> Dict[str, np.ndarray]:
        n = min(n, len(self))
        tasks = [task_id for task_id, task_buf in self.transitions_by_task.items() if task_buf]
        selected = []
        while len(selected) < n and tasks:
            random.shuffle(tasks)
            for task_id in tasks:
                task_buf = self.transitions_by_task[task_id]
                if task_buf:
                    selected.append(random.choice(task_buf))
                if len(selected) >= n:
                    break

        obs, actions, rewards, next_obs, dones = zip(*selected) if selected else ([], [], [], [], [])
        return {
            "obs"     : np.array(obs),
            "actions" : np.array(actions),
            "rewards" : np.array(rewards),
            "next_obs": np.array(next_obs),
            "dones"   : np.array(dones),
        }

    def __len__(self):
        return sum(len(task_buf) for task_buf in self.transitions_by_task.values())


# ── 3. SDFT Teacher Store (RL) ────────────────────────────────────────────────

class RLTeacherStore:
    """
    Stores a frozen copy of the previous PPO policy for SDFT KL penalty.
    """

    def __init__(self):
        self.teacher_policy = None

    def store(self, ppo_model: PPO):
        self.teacher_policy = copy.deepcopy(ppo_model.policy)
        self.teacher_policy.eval()
        for p in self.teacher_policy.parameters():
            p.requires_grad_(False)

    def kl_penalty(self, ppo_model: PPO, obs_tensor: torch.Tensor) -> torch.Tensor:
        """
        KL(old_policy || current_policy) for discrete action distributions.
        obs_tensor: shape (B, state_dim)
        """
        if self.teacher_policy is None:
            return torch.tensor(0.0)

        device = next(ppo_model.policy.parameters()).device
        obs_tensor = obs_tensor.to(device)

        with torch.no_grad():
            t_dist = self.teacher_policy.get_distribution(obs_tensor)
            t_logp = t_dist.distribution.logits  # (B, n_actions)

        s_dist = ppo_model.policy.get_distribution(obs_tensor)
        s_logp = s_dist.distribution.logits

        kl = F.kl_div(
            F.log_softmax(s_logp, dim=-1),
            F.softmax(t_logp.detach(), dim=-1),
            reduction="batchmean",
        )
        return kl


# ── 4. EWC Engine for PPO ─────────────────────────────────────────────────────

class PPOEWCEngine:
    """
    Diagonal EWC for SB3 PPO policy parameters.
    """

    def __init__(self, lambda_ewc: float):
        self.lambda_ewc  = lambda_ewc
        self.fisher      : Dict[str, torch.Tensor] = {}
        self.opt_params  : Dict[str, torch.Tensor] = {}

    def compute_fisher(self, ppo_model: PPO, env: DynamicPricingEnv, n_steps: int = 1000):
        policy = ppo_model.policy
        device = next(policy.parameters()).device
        fisher_acc = {n: torch.zeros_like(p) for n, p in policy.named_parameters() if p.requires_grad}

        obs, _    = env.reset()
        count     = 0
        while count < n_steps:
            obs_t = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
            policy.zero_grad()
            dist   = policy.get_distribution(obs_t)
            log_p  = dist.distribution.logits.sum()
            log_p.backward()
            for name, param in policy.named_parameters():
                if param.requires_grad and param.grad is not None:
                    fisher_acc[name] += param.grad.detach().pow(2)
            action, _ = ppo_model.predict(obs, deterministic=False)
            obs, _, done, _, _ = env.step(int(action))
            if done:
                obs, _ = env.reset()
            count += 1

        for name in fisher_acc:
            fisher_acc[name] /= n_steps
            if name in self.fisher:
                self.fisher[name] = self.fisher[name].cpu() + fisher_acc[name].cpu()
            else:
                self.fisher[name] = fisher_acc[name].cpu()

        self.opt_params = {
            n: p.detach().cpu().clone()
            for n, p in policy.named_parameters() if p.requires_grad
        }
        console.print(f"  [cyan]PPO Fisher computed over {count} steps[/cyan]")

    def penalty(self, policy: nn.Module) -> torch.Tensor:
        penalty = torch.tensor(0.0, device=next(policy.parameters()).device)
        for name, param in policy.named_parameters():
            if name in self.fisher:
                fisher   = self.fisher[name].to(param.device)
                opt_p    = self.opt_params[name].to(param.device)
                penalty += (fisher * (param - opt_p).pow(2)).sum()
        return (self.lambda_ewc / 2.0) * penalty



# ─── PPO continual-learning callbacks ─────────────────────────────────────────
# Vendored from hybrid_pipeline/experiment_runner.py, where they sat beside the
# FYP1 task-loop they were written for. They are model machinery, not
# orchestration, so they belong here.
# ── RECALL Callback ───────────────────────────────────────────────────────────

class RECALLCallback(BaseCallback):
    """
    RECALL-style replay-enhanced CL callback for PPO.
    After each rollout collection, injects past transitions into the rollout buffer
    by creating a supplementary batch that modifies the policy gradient signal.

    Implementation: We augment the training by running an additional
    supervised imitation step on replayed experiences after each PPO update.
    This preserves past policy behaviour without modifying the core PPO algorithm.
    """

    def __init__(self, replay_buf: RLReplayBuffer, mix_n: int, device: str, bc_coef: float = 0.1):
        super().__init__(verbose=0)
        self.replay_buf = replay_buf
        self.mix_n      = mix_n
        self.device     = device
        self.bc_coef    = bc_coef

    def _on_step(self) -> bool:
        return True

    def _on_rollout_end(self):
        """Called after rollout collection, before PPO update."""
        if len(self.replay_buf) < self.mix_n:
            return

        batch   = self.replay_buf.sample(self.mix_n)
        obs_t   = torch.tensor(batch["obs"],     dtype=torch.float32, device=self.device)
        acts_t  = torch.tensor(batch["actions"], dtype=torch.long,    device=self.device)

        policy  = self.model.policy
        # Compute log-prob under current policy
        dist    = policy.get_distribution(obs_t)
        log_p   = dist.log_prob(acts_t)

        # Imitation loss: maximise log-prob of past actions (behaviour cloning)
        bc_loss = -log_p.mean() * self.bc_coef   # weighted to not dominate PPO loss
        policy.optimizer.zero_grad()
        bc_loss.backward()
        policy.optimizer.step()


# ── SDFT Callback ─────────────────────────────────────────────────────────────

class SDFTCallback(BaseCallback):
    """
    SDFT for RL: KL penalty between frozen old-task policy and current policy.
    Applied as an auxiliary gradient step after each PPO rollout.
    """

    def __init__(self, teacher_store: RLTeacherStore, kl_coef: float, device: str):
        super().__init__(verbose=0)
        self.teacher_store = teacher_store
        self.kl_coef       = kl_coef
        self.device        = device

    def _on_step(self) -> bool:
        return True

    def _on_rollout_end(self):
        if self.teacher_store.teacher_policy is None:
            return

        policy = self.model.policy
        # Sample recent observations from rollout buffer
        try:
            rollout_data = next(self.model.rollout_buffer.get(64))
            obs_t = rollout_data.observations.to(self.device)
        except (StopIteration, AttributeError):
            return

        kl = self.teacher_store.kl_penalty(self.model, obs_t)
        kl_loss = self.kl_coef * kl

        policy.optimizer.zero_grad()
        kl_loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), 0.5)
        policy.optimizer.step()


# ── EWC Callback ──────────────────────────────────────────────────────────────

class EWCCallbackRL(BaseCallback):
    """Applies EWC penalty after each PPO rollout update."""

    def __init__(self, ewc_engine: PPOEWCEngine, device: str, scale: float = 1.0):
        super().__init__(verbose=0)
        self.ewc_engine = ewc_engine
        self.device     = device
        self.scale      = scale

    def _on_step(self) -> bool:
        return True

    def _on_rollout_end(self):
        if not self.ewc_engine.fisher:
            return
        policy  = self.model.policy
        penalty = self.ewc_engine.penalty(policy) * self.scale
        policy.optimizer.zero_grad()
        penalty.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), 0.5)
        policy.optimizer.step()
