# Training components for the FYP continual-learning experiment.

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

from .core_pipeline import CONFIG, DEVICE, SEED, ckpt_path, console


# ─── Cell: PyTorch Forecasting TimeSeriesDataSet Builder ─────────────────────
from pytorch_forecasting.data import GroupNormalizer

# Minimum rows needed for one encoder+prediction window
MIN_ROWS_NEEDED = CONFIG["forecasting"]["encoder_length"] + CONFIG["forecasting"]["prediction_length"]

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

    # 80/20 split within task if no explicit val_df given. Keep encoder context
    # in val_df so each series has enough history for validation windows.
    if val_df is None and len(train_df) > MIN_ROWS_NEEDED * 2:
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
    if val_df is not None and len(val_df) >= MIN_ROWS_NEEDED:
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

        # SDFT teacher (frozen copy of previous model)
        self.teacher     : Optional["CLTFT"] = None

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

    # ── Training step override ─────────────────────────────────────────────
    def training_step(self, batch, batch_idx):
        step_output = super().training_step(batch, batch_idx)

        if self.cl_method == "ewc" and self.ewc_fisher:
            ewc_penalty = self._ewc_loss()
            self.log("ewc_penalty", ewc_penalty.detach(), prog_bar=False, on_step=True)
            step_output = self._add_auxiliary_loss(step_output, ewc_penalty)

        elif self.cl_method == "sdft" and self.teacher is not None:
            sdft_penalty = self._sdft_loss(batch)
            self.log("sdft_penalty", sdft_penalty.detach(), prog_bar=False, on_step=True)
            step_output = self._add_auxiliary_loss(step_output, sdft_penalty)

        return step_output

    # ── EWC penalty ────────────────────────────────────────────────────────
    def _ewc_loss(self) -> torch.Tensor:
        penalty = torch.tensor(0.0, device=self.device)
        lam     = self.cl_cfg["ewc_lambda"]
        for name, param in self.named_parameters():
            if name in self.ewc_fisher:
                fisher   = self.ewc_fisher[name].to(self.device)
                opt_p    = self.ewc_optparams[name].to(self.device)
                penalty += (fisher * (param - opt_p).pow(2)).sum()
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
        self.train()
        console.print(f"  [cyan]Fisher computed over {count} batches[/cyan]")

    # ── SDFT: soft regression distillation loss ────────────────────────────
    def _sdft_loss(self, batch) -> torch.Tensor:
        alpha  = self.cl_cfg["sdft_alpha"]
        T      = self.cl_cfg["sdft_temperature"]

        x, y = batch
        x_dev = self._move_to_device(x)

        with torch.no_grad():
            self.teacher.eval()
            self.teacher.to(self.device)
            t_out  = self.teacher(x_dev)
            t_pred = self._prediction_from_output(t_out)

        s_out  = self(x_dev)
        s_pred = self._prediction_from_output(s_out)

        # Soft MSE distillation (scaled by temperature)
        sdft_loss = F.mse_loss(s_pred / T, t_pred.detach() / T)
        return (1.0 - alpha) * sdft_loss  # alpha already in main loss from super()

    def store_teacher(self):
        """Store a frozen deep copy of self as the teacher for next task."""
        self.teacher = copy.deepcopy(self)
        self.teacher.eval()
        for p in self.teacher.parameters():
            p.requires_grad_(False)


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
        log_interval           = 20,
        reduce_on_plateau_patience = 3,
        cl_method              = cl_method,
        cl_cfg                 = CONFIG["cl"],
    )
    return model




# ─── Cell: Gymnasium Dynamic Pricing Environment ──────────────────────────────

class DynamicPricingEnv(gym.Env):
    """
    Discrete-action dynamic pricing environment backed by rl_environment.csv.

    State (13 features):
        [demand_forecast_norm, inventory_norm, competitor_price_norm,
         day_of_week/6, month/12, is_weekend, is_mega_sale,
         is_ramadan, is_pre_raya_window, is_pre_cny_window,
         viral_shock_active, any_shock_active, elasticity_norm]

    Action (Discrete 5):
        0: -10%  1: -5%  2: 0%  3: +5%  4: +10%
        Applied to base_price to set current_price.

    Reward:
        profit_margin_myr (clipped to [-1000, 5000]) / 1000
        Penalised by 0.15 * revenue if stockout occurs.
    """

    metadata = {"render_modes": []}

    PRICE_TIERS = [-0.10, -0.05, 0.0, +0.05, +0.10]
    STATE_DIM   = 13

    def __init__(self, task_df: pd.DataFrame, seed: int = SEED):
        super().__init__()
        self.df      = task_df.reset_index(drop=True)
        self.n_steps = len(self.df)
        self._seed   = seed

        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(self.STATE_DIM,), dtype=np.float32,
        )
        self.action_space = spaces.Discrete(len(self.PRICE_TIERS))

        # Precompute needed columns
        self._precompute()
        self.reset(seed=seed)

    def _precompute(self):
        df = self.df
        def safe_norm(col, fallback=0.0):
            if col not in df.columns:
                return np.full(len(df), fallback, dtype=np.float32)
            v = df[col].values.astype(np.float32)
            m = np.abs(v).max()
            return v / m if m > 0 else v

        self._demand_forecast = safe_norm("demand_forecast_norm",  0.5)
        self._inventory       = safe_norm("inventory_norm",        0.5)
        self._comp_price      = safe_norm("competitor_price_norm",  0.5)
        self._day_of_week     = df.get("day_of_week",  pd.Series(0)).values / 6.0
        self._month           = df.get("month",         pd.Series(1)).values / 12.0
        self._is_weekend      = df.get("is_weekend",    pd.Series(0)).values.astype(np.float32)
        self._is_mega         = df.get("is_mega_sale",  pd.Series(0)).values.astype(np.float32)
        self._is_ramadan      = df.get("is_ramadan",    pd.Series(0)).values.astype(np.float32)
        self._is_raya         = df.get("is_pre_raya_window",pd.Series(0)).values.astype(np.float32)
        self._is_cny          = df.get("is_pre_cny_window", pd.Series(0)).values.astype(np.float32)
        self._viral           = df.get("viral_shock_active", pd.Series(0)).values.astype(np.float32)
        self._any_shock       = df.get("any_shock_active",  pd.Series(0)).values.astype(np.float32)
        self._elasticity      = safe_norm("elasticity_coefficient", -1.5)

        # Price info
        self._base_price      = df.get("base_price",      pd.Series(100)).values.astype(np.float32)
        self._realized_demand = df.get("realized_demand", pd.Series(50)).values.astype(np.float32)
        self._unit_cost       = df.get("base_price",      pd.Series(50)).values.astype(np.float32) * 0.55
        self._stockout        = df.get("stockout_flag",   pd.Series(0)).values.astype(np.float32)

    def _get_obs(self, idx: int) -> np.ndarray:
        return np.array([
            self._demand_forecast[idx],
            self._inventory[idx],
            self._comp_price[idx],
            float(self._day_of_week[idx]),
            float(self._month[idx]),
            self._is_weekend[idx],
            self._is_mega[idx],
            self._is_ramadan[idx],
            self._is_raya[idx],
            self._is_cny[idx],
            self._viral[idx],
            self._any_shock[idx],
            self._elasticity[idx],
        ], dtype=np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        return self._get_obs(0), {}

    def step(self, action: int):
        idx   = self.current_step
        tier  = self.PRICE_TIERS[int(action)]
        price = float(self._base_price[idx]) * (1.0 + tier)
        cost  = float(self._unit_cost[idx])

        # Price-elasticity adjusted demand (simplified inline)
        base_demand = max(float(self._realized_demand[idx]), 0)
        elasticity  = float(self._elasticity[idx])
        base_p      = float(self._base_price[idx])
        price_ratio = price / max(base_p, 1e-6)
        adj_demand  = base_demand * (price_ratio ** elasticity)
        adj_demand  = max(0.0, round(adj_demand))

        # Revenue & profit
        revenue       = price * adj_demand
        profit_margin = (price - cost) * adj_demand
        stockout      = float(self._stockout[idx])
        stockout_pen  = 0.15 * revenue * stockout

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
        """Compute oracle best reward at step idx (exhaustive over tiers)."""
        best = -np.inf
        for tier in self.PRICE_TIERS:
            price = float(self._base_price[idx]) * (1.0 + tier)
            cost  = float(self._unit_cost[idx])
            demand = max(float(self._realized_demand[idx]), 0)
            e     = float(self._elasticity[idx])
            base_p = float(self._base_price[idx])
            adj   = demand * ((price / max(base_p, 1e-6)) ** e)
            profit = (price - cost) * max(0.0, adj)
            if profit > best:
                best = profit
        return float(best) / 1000.0


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
    Mean Absolute Scaled Error using Darts.
    y_train is the in-sample series used to compute the naive seasonal baseline.
    """
    try:
        ts_true  = TimeSeries.from_values(y_true.reshape(-1, 1))
        ts_pred  = TimeSeries.from_values(y_pred.reshape(-1, 1))
        ts_train = TimeSeries.from_values(y_train.reshape(-1, 1))
        return float(darts_mase(ts_true, ts_pred, ts_train, m=seasonality))
    except Exception:
        # Fallback: manual MASE
        n       = len(y_true)
        mae     = np.mean(np.abs(y_true - y_pred))
        naive   = np.mean(np.abs(y_train[seasonality:] - y_train[:-seasonality]))
        return float(mae / max(naive, 1e-8))


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
    n_episodes: int = None,
) -> Dict[str, float]:
    """
    Run the PPO agent on task_df and compute pricing metrics.
    """
    n_episodes = n_episodes or CONFIG["rl"]["eval_episodes"]
    env = DynamicPricingEnv(task_df)

    episode_rewards, episode_profits = [], []
    total_regret = 0.0
    total_steps  = 0

    for ep in range(n_episodes):
        obs, _ = env.reset()
        ep_reward, ep_profit = 0.0, 0.0
        done = False

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, _, info = env.step(int(action))
            ep_reward  += reward
            ep_profit  += info.get("profit_margin", 0.0)

            # Regret: oracle - actual (both in raw MYR)
            oracle = env.optimal_reward(min(env.current_step - 1, env.n_steps - 1))
            regret = max(0.0, oracle - reward)
            total_regret += regret
            total_steps  += 1

        episode_rewards.append(ep_reward)
        episode_profits.append(ep_profit)

    avg_reward  = float(np.mean(episode_rewards))
    cum_profit  = float(np.sum(episode_profits))
    avg_regret  = float(total_regret / max(total_steps, 1))

    return {
        "avg_episode_reward": avg_reward,
        "cumulative_profit" : cum_profit,
        "pricing_regret"    : avg_regret,
    }


# ── CL Metrics (BWT / FWT) ────────────────────────────────────────────────────

def compute_bwt_fwt(
    perf_matrix: np.ndarray,
    primary_is_higher_better: bool = True,
) -> Tuple[float, float]:
    """
    Compute Backward Transfer (BWT) and Forward Transfer (FWT).

    perf_matrix[i, j] = performance on task j evaluated after training on task i.
    Shape: (n_tasks, n_tasks). Upper-triangle is -inf (task not seen yet).

    BWT = (1/(T-1)) * sum_{i=1}^{T-1} (R_{T,i} - R_{i,i})
    FWT = (1/(T-1)) * sum_{i=2}^{T}   (R_{i-1,i} - b_i)
          where b_i = perf_matrix[0, i] (naive baseline = Task-1 model on task i)
    """
    T = perf_matrix.shape[0]
    if T < 2:
        return 0.0, 0.0

    # BWT: how much does final model forget earlier tasks?
    bwt_vals = []
    for i in range(T - 1):
        r_final  = perf_matrix[T - 1, i]
        r_at_i   = perf_matrix[i, i]
        if not (np.isnan(r_final) or np.isnan(r_at_i)):
            diff = r_final - r_at_i
            bwt_vals.append(diff if primary_is_higher_better else -diff)
    bwt = float(np.mean(bwt_vals)) if bwt_vals else 0.0

    # FWT: does past learning help on new tasks?
    fwt_vals = []
    for i in range(1, T):
        r_transfer = perf_matrix[i - 1, i]
        r_baseline = perf_matrix[0, i]          # naive CL result used as baseline
        if not (np.isnan(r_transfer) or np.isnan(r_baseline)):
            diff = r_transfer - r_baseline
            fwt_vals.append(diff if primary_is_higher_better else -diff)
    fwt = float(np.mean(fwt_vals)) if fwt_vals else 0.0

    return bwt, fwt




# ─── Cell: Shared CL Engines (EWC, Replay, SDFT) ─────────────────────────────

# ── 1. Forecasting Replay Buffer ───────────────────────────────────────────────

class ForecastingReplayBuffer:
    """
    Stores (batch_x, batch_y) samples from past tasks.
    Uses reservoir sampling to maintain a fixed-size buffer.
    """

    def __init__(self, capacity: int):
        self.capacity = capacity
        self.buffer: List[Tuple] = []
        self._count = 0

    def add_from_loader(self, dataloader, n_samples: int = None):
        n_samples = n_samples or self.capacity
        added = 0
        for batch in dataloader:
            if added >= n_samples:
                break
            x, y = batch
            # Store CPU tensors to save GPU memory
            x_cpu = {k: v.cpu() if isinstance(v, torch.Tensor) else v for k, v in x.items()}
            y_cpu = [v.cpu() if isinstance(v, torch.Tensor) else v for v in y]
            sample = (x_cpu, y_cpu)

            if len(self.buffer) < self.capacity:
                self.buffer.append(sample)
            else:
                # Reservoir sampling
                idx = random.randint(0, self._count)
                if idx < self.capacity:
                    self.buffer[idx] = sample
            self._count += 1
            added += 1

    def sample(self, n: int) -> List[Tuple]:
        n = min(n, len(self.buffer))
        return random.sample(self.buffer, n)

    def __len__(self):
        return len(self.buffer)


# ── 2. RECALL-style RL Replay Buffer ──────────────────────────────────────────

class RLReplayBuffer:
    """
    Stores (obs, action, reward, next_obs, done) tuples from past RL tasks.
    Used by RECALL-style method to augment PPO rollouts.
    """

    def __init__(self, capacity: int):
        self.capacity = capacity
        self.obs       : List[np.ndarray] = []
        self.actions   : List[int]        = []
        self.rewards   : List[float]      = []
        self.next_obs  : List[np.ndarray] = []
        self.dones     : List[bool]       = []

    def add_from_env_rollout(self, env: DynamicPricingEnv, policy: PPO, n: int):
        """Collect n transitions from env using policy."""
        obs, _ = env.reset()
        added  = 0
        while added < n:
            action, _ = policy.predict(obs, deterministic=False)
            next_obs, reward, done, _, _ = env.step(int(action))
            self.obs.append(obs.copy())
            self.actions.append(int(action))
            self.rewards.append(float(reward))
            self.next_obs.append(next_obs.copy())
            self.dones.append(bool(done))
            obs = next_obs
            if done:
                obs, _ = env.reset()
            added += 1
            # Trim to capacity (FIFO)
            if len(self.obs) > self.capacity:
                self.obs.pop(0); self.actions.pop(0)
                self.rewards.pop(0); self.next_obs.pop(0); self.dones.pop(0)

    def sample(self, n: int) -> Dict[str, np.ndarray]:
        n    = min(n, len(self.obs))
        idxs = random.sample(range(len(self.obs)), n)
        return {
            "obs"     : np.array([self.obs[i]      for i in idxs]),
            "actions" : np.array([self.actions[i]  for i in idxs]),
            "rewards" : np.array([self.rewards[i]  for i in idxs]),
            "next_obs": np.array([self.next_obs[i] for i in idxs]),
            "dones"   : np.array([self.dones[i]    for i in idxs]),
        }

    def __len__(self):
        return len(self.obs)


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




# ─── Cell: Results Logger & Checkpoint Manager ───────────────────────────────

class ResultsLogger:
    """
    Logs all metric evaluations into a structured DataFrame.
    Columns: model_type, cl_method, train_task_id, eval_task_id,
             metric_name, metric_value, timestamp
    """

    def __init__(self):
        self._records: List[Dict] = []
        # perf_matrix[model_type][cl_method] = 2D np array (n_tasks x n_tasks)
        self._perf_matrices: Dict[str, Dict[str, np.ndarray]] = defaultdict(dict)
        self._n_tasks = len(CONFIG["tasks"])

    def log(
        self,
        model_type   : str,
        cl_method    : str,
        train_task_id: int,
        eval_task_id : int,
        metrics      : Dict[str, float],
    ):
        ts = datetime.now().isoformat(timespec="seconds")
        for metric_name, value in metrics.items():
            self._records.append({
                "model_type"    : model_type,
                "cl_method"     : cl_method,
                "train_task_id" : train_task_id,
                "eval_task_id"  : eval_task_id,
                "metric_name"   : metric_name,
                "metric_value"  : value,
                "timestamp"     : ts,
            })

    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame(self._records)

    def save(self, path: str = None):
        path = path or str(Path(CONFIG["paths"]["results"]) / "all_metrics.csv")
        self.to_dataframe().to_csv(path, index=False)
        console.print(f"  Results saved → {path}")

    def get_perf_matrix(
        self,
        model_type : str,
        cl_method  : str,
        metric_name: str,
    ) -> np.ndarray:
        """
        Build (n_tasks × n_tasks) performance matrix for BWT/FWT computation.
        mat[i, j] = metric value on task j after training on task i+1.
        """
        df  = self.to_dataframe()
        sub = df[
            (df["model_type"] == model_type) &
            (df["cl_method"]  == cl_method)  &
            (df["metric_name"]== metric_name)
        ]
        n   = self._n_tasks
        mat = np.full((n, n), np.nan)
        for _, row in sub.iterrows():
            i = int(row["train_task_id"]) - 1
            j = int(row["eval_task_id"])  - 1
            if 0 <= i < n and 0 <= j < n:
                mat[i, j] = row["metric_value"]
        return mat


class CheckpointManager:
    """
    Saves and loads model checkpoints.
    Keeps the best checkpoint per method based on primary metric.
    """

    def __init__(self):
        self._best: Dict[str, float] = {}
        self._best_path: Dict[str, str] = {}

    def save_forecasting(
        self,
        model    : CLTFT,
        trainer  : L.Trainer,
        cl_method: str,
        task_id  : int,
        metric   : float,
    ):
        path = ckpt_path("forecasting", cl_method, task_id)
        trainer.save_checkpoint(path)

        key = f"forecasting_{cl_method}"
        # Lower MASE is better
        if key not in self._best or (not np.isnan(metric) and metric < self._best.get(key, np.inf)):
            best_path = ckpt_path("forecasting", cl_method, task_id, "best")
            trainer.save_checkpoint(best_path)
            self._best[key]      = metric
            self._best_path[key] = best_path
            console.print(f"  [green]★ New best {cl_method} MASE={metric:.4f}[/green]")

    def save_rl(self, model: PPO, cl_method: str, task_id: int, metric: float):
        path = ckpt_path("rl", cl_method, task_id).replace(".ckpt", "")
        model.save(path)

        key = f"rl_{cl_method}"
        # Higher cumulative_profit is better
        if key not in self._best or (not np.isnan(metric) and metric > self._best.get(key, -np.inf)):
            best_path = ckpt_path("rl", cl_method, task_id, "best").replace(".ckpt", "")
            model.save(best_path)
            self._best[key]      = metric
            self._best_path[key] = best_path


# ── Instantiate global logger and checkpoint manager ──────────────────────────
LOGGER  = ResultsLogger()
CKPT_MGR = CheckpointManager()


