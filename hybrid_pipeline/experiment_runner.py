# Experiment orchestration and plotting.

import random
import time
import traceback
import logging
import warnings
from pathlib import Path
from typing import Dict, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import lightning as L
from lightning.pytorch.callbacks import EarlyStopping, LearningRateMonitor
from lightning.pytorch.loggers import TensorBoardLogger
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv

from .core_pipeline import CONFIG, DEVICE, SEED, ckpt_path, console, prepare_data, print_task_summary
from .trainers import (
    CKPT_MGR, LOGGER, CLTFT, DynamicPricingEnv, ForecastingReplayBuffer,
    PPOEWCEngine, RLReplayBuffer, RLTeacherStore, TimeSeriesDataSet,
    build_cltft, compute_bwt_fwt, compute_forgetting, evaluate_forecasting, evaluate_rl,
    filter_tft_eval_frame, make_pricing_env, make_tft_dataset, make_tft_loaders,
    min_tft_rows,
)

warnings.filterwarnings("ignore", category=DeprecationWarning)
logging.getLogger("lightning").setLevel(logging.WARNING)
logging.getLogger("lightning.pytorch").setLevel(logging.WARNING)
logging.getLogger("pytorch_lightning").setLevel(logging.WARNING)


FORECAST_DRIFT_COLUMNS = [
    "demand", "day_of_week", "month", "quarter", "is_weekend",
    "is_federal_holiday", "is_ramadan", "is_pre_raya_window",
    "is_pre_cny_window", "is_mega_sale", "mega_sale_base_magnitude",
    "is_school_holiday", "social_media_index",
    "competitor_activity_index", "marketing_spend_myr",
]

RL_DRIFT_COLUMNS = [
    "demand_forecast", "inventory_level", "competitor_price", "base_price",
    "realized_demand", "stockout_flag", "elasticity_coefficient",
    "day_of_week", "month", "is_weekend", "is_mega_sale", "is_ramadan",
    "is_pre_raya_window", "is_pre_cny_window", "viral_shock_active",
    "any_shock_active",
]


def compute_distribution_drift(
    previous_df: Optional[pd.DataFrame],
    current_df: pd.DataFrame,
    columns: list[str],
) -> float:
    """Return a bounded [0, 1] drift score from normalized mean shifts."""
    if previous_df is None or previous_df.empty or current_df.empty:
        return 0.0

    shifts = []
    for col in columns:
        if col not in previous_df.columns or col not in current_df.columns:
            continue
        prev = pd.to_numeric(previous_df[col], errors="coerce").dropna()
        curr = pd.to_numeric(current_df[col], errors="coerce").dropna()
        if prev.empty or curr.empty:
            continue
        pooled_std = np.sqrt((prev.var(ddof=0) + curr.var(ddof=0)) / 2.0)
        if not np.isfinite(pooled_std) or pooled_std <= 1e-8:
            shifts.append(0.0 if np.isclose(prev.mean(), curr.mean()) else 1.0)
        else:
            shifts.append(abs(curr.mean() - prev.mean()) / pooled_std)

    if not shifts:
        return 0.0

    raw_shift = float(np.mean(shifts))
    return float(np.clip(raw_shift / (1.0 + raw_shift), 0.0, 1.0))


def adaptive_strengths(drift_score: float) -> tuple[float, float]:
    """Map drift to EWC/distillation strength. More drift means more plasticity."""
    cl_cfg = CONFIG["cl"]
    retention = 1.0 - float(np.clip(drift_score, 0.0, 1.0))
    ewc_scale = cl_cfg["drift_ewc_min_scale"] + (
        cl_cfg["drift_ewc_max_scale"] - cl_cfg["drift_ewc_min_scale"]
    ) * retention
    distill_scale = cl_cfg["drift_distill_max_scale"] * retention
    return float(ewc_scale), float(distill_scale)


def adaptive_replay_scale(drift_score: float) -> float:
    """Map drift to replay strength. More drift means less replay pressure."""
    cl_cfg = CONFIG["cl"]
    retention = 1.0 - float(np.clip(drift_score, 0.0, 1.0))
    replay_scale = cl_cfg["drift_replay_min_scale"] + (
        cl_cfg["drift_replay_max_scale"] - cl_cfg["drift_replay_min_scale"]
    ) * retention
    return float(replay_scale)


# ─── Cell: Forecasting CL Trainers (Naive / EWC / Replay / SDFT) ─────────────

# Shared persistent state across tasks (per cl_method)
_fc_model_state : Dict[str, Optional[CLTFT]] = {m: None for m in CONFIG["cl_methods"]["forecasting"]}
_fc_replay_buf  : Dict[str, ForecastingReplayBuffer] = {
    "recall_ewc": ForecastingReplayBuffer(CONFIG["cl"]["replay_buffer_size"]),
    "drift_adaptive_replay_ewc": ForecastingReplayBuffer(CONFIG["cl"]["replay_buffer_size"]),
}
_fc_previous_task_df: Dict[str, Optional[pd.DataFrame]] = {
    "adaptive_drift": None,
    "drift_adaptive_replay_ewc": None,
}
# Keep Task-1 training data for MASE baseline
_task1_train_df: Optional[pd.DataFrame] = None


def _make_lightning_trainer(cl_method: str, task_id: int, has_val_loader: bool = True) -> L.Trainer:
    hw  = CONFIG["hardware"]
    fc  = CONFIG["forecasting"]
    log_dir = str(Path(CONFIG["paths"]["logs"]) / "forecasting" / cl_method / f"task{task_id}")

    callbacks = [LearningRateMonitor(logging_interval="epoch")]
    if has_val_loader:
        callbacks.insert(
            0,
            EarlyStopping(
                monitor="val_loss",
                min_delta=1e-4,
                patience=fc["early_stop_patience"],
                verbose=False,
                mode="min",
            ),
        )

    return L.Trainer(
        max_epochs          = fc["max_epochs"],
        accelerator         = "gpu" if DEVICE == "cuda" else "cpu",
        devices             = 1,
        precision           = hw["precision"],
        gradient_clip_val   = fc["gradient_clip"],
        enable_model_summary= False,
        enable_progress_bar = True,
        logger              = TensorBoardLogger(log_dir, name=""),
        callbacks           = callbacks,
        log_every_n_steps   = 5,
    )


def _prepare_forecast_model_for_fit(model: CLTFT) -> CLTFT:
    model.train()
    original = getattr(model, "_orig_mod", None)
    if original is not None:
        original.train()
    return model


def train_forecast_naive(task_id: int, train_loader, val_loader, train_ds, train_df) -> CLTFT:
    """Standard fine-tuning — no forgetting mitigation."""
    model = _fc_model_state["naive"]
    if model is None:
        model = build_cltft(train_ds, cl_method="naive")
        if CONFIG["hardware"].get("compile", False):
            model = torch.compile(model, mode=CONFIG["hardware"]["compile_mode"])
    trainer = _make_lightning_trainer("naive", task_id, has_val_loader=val_loader is not None)
    model = _prepare_forecast_model_for_fit(model)
    trainer.fit(model, train_loader, val_loader)
    _fc_model_state["naive"] = model
    return model, trainer


def train_forecast_recall_ewc(task_id: int, train_loader, val_loader, train_ds, train_df) -> CLTFT:
    """
    RECALL + EWC: replay old forecasting batches while protecting important
    parameters with EWC.
    """
    model = _fc_model_state["recall_ewc"]
    if model is None:
        model = build_cltft(train_ds, cl_method="recall_ewc")
        if CONFIG["hardware"].get("compile", False):
            model = torch.compile(model, mode=CONFIG["hardware"]["compile_mode"])

    buf = _fc_replay_buf["recall_ewc"]

    # If we have past data, create a mixed loader
    if len(buf) > 0 and task_id > 1:
        mix_ratio = CONFIG["cl"]["replay_mix_ratio"]

        class MixedLoader:
            """Interleaves current-task batches with replay batches."""
            def __init__(self, current_loader, replay_buf, mix_ratio):
                self.current   = current_loader
                self.buf       = replay_buf
                self.mix_ratio = mix_ratio

            def __iter__(self):
                for batch in self.current:
                    yield batch
                    # Inject a replay batch every few current batches
                    if random.random() < self.mix_ratio and len(self.buf) > 0:
                        replay_samples = self.buf.sample(1)
                        if replay_samples:
                            yield replay_samples[0]

            def __len__(self):
                return len(self.current)

            @property
            def dataset(self):
                return self.current.dataset

        mixed_loader = MixedLoader(train_loader, buf, mix_ratio)
    else:
        mixed_loader = train_loader

    trainer = _make_lightning_trainer("recall_ewc", task_id, has_val_loader=val_loader is not None)
    model = _prepare_forecast_model_for_fit(model)
    trainer.fit(model, mixed_loader, val_loader)

    # Store EWC state and current task batches for the next task.
    n_batches = CONFIG["cl"]["ewc_fisher_samples"]
    model.compute_and_store_fisher(train_loader, n_batches=n_batches)
    buf.add_from_loader(
        train_loader,
        task_id=task_id,
        n_samples=CONFIG["cl"]["replay_buffer_size"] // max(task_id, 1),
    )

    _fc_model_state["recall_ewc"] = model
    return model, trainer


def train_forecast_drift_adaptive_replay_ewc(task_id: int, train_loader, val_loader, train_ds, train_df) -> CLTFT:
    """
    Drift-adaptive replay + EWC: same replay/EWC mechanics as RECALL + EWC,
    but drift scales EWC strength and replay injection frequency.
    """
    method = "drift_adaptive_replay_ewc"
    model = _fc_model_state[method]
    if model is None:
        model = build_cltft(train_ds, cl_method=method)
        model.set_adaptive_strengths(0.0, *adaptive_strengths(0.0))
        if CONFIG["hardware"].get("compile", False):
            model = torch.compile(model, mode=CONFIG["hardware"]["compile_mode"])
        drift = 0.0
    else:
        drift = compute_distribution_drift(
            _fc_previous_task_df[method],
            train_df,
            FORECAST_DRIFT_COLUMNS,
        )
        ewc_scale, _ = adaptive_strengths(drift)
        model.set_adaptive_strengths(drift, ewc_scale, 0.0)

    ewc_scale, _ = adaptive_strengths(drift)
    replay_scale = adaptive_replay_scale(drift)
    mix_ratio = CONFIG["cl"]["replay_mix_ratio"] * replay_scale
    buf = _fc_replay_buf[method]

    if task_id > 1:
        console.print(
            f"  [cyan]Forecast drift={drift:.3f} "
            f"ewc_scale={ewc_scale:.3f} replay_mix={mix_ratio:.3f}[/cyan]"
        )

    if len(buf) > 0 and task_id > 1:
        class MixedLoader:
            """Interleaves current-task batches with task-balanced replay batches."""
            def __init__(self, current_loader, replay_buf, mix_ratio):
                self.current   = current_loader
                self.buf       = replay_buf
                self.mix_ratio = mix_ratio

            def __iter__(self):
                for batch in self.current:
                    yield batch
                    if random.random() < self.mix_ratio and len(self.buf) > 0:
                        replay_samples = self.buf.sample(1)
                        if replay_samples:
                            yield replay_samples[0]

            def __len__(self):
                return len(self.current)

            @property
            def dataset(self):
                return self.current.dataset

        mixed_loader = MixedLoader(train_loader, buf, mix_ratio)
    else:
        mixed_loader = train_loader

    trainer = _make_lightning_trainer(method, task_id, has_val_loader=val_loader is not None)
    model = _prepare_forecast_model_for_fit(model)
    trainer.fit(model, mixed_loader, val_loader)

    n_batches = CONFIG["cl"]["ewc_fisher_samples"]
    model.compute_and_store_fisher(train_loader, n_batches=n_batches)
    buf.add_from_loader(
        train_loader,
        task_id=task_id,
        n_samples=CONFIG["cl"]["replay_buffer_size"] // max(task_id, 1),
    )

    _fc_previous_task_df[method] = train_df.copy()
    _fc_model_state[method] = model
    return model, trainer


def train_forecast_adaptive_drift(task_id: int, train_loader, val_loader, train_ds, train_df) -> CLTFT:
    """
    Drift-adaptive replay-free CL: combines EWC and teacher distillation, then
    lowers their strength when the current task distribution drifts more.
    """
    model = _fc_model_state["adaptive_drift"]
    if model is None:
        model = build_cltft(train_ds, cl_method="adaptive_drift")
        model.set_adaptive_strengths(0.0, *adaptive_strengths(0.0))
        if CONFIG["hardware"].get("compile", False):
            model = torch.compile(model, mode=CONFIG["hardware"]["compile_mode"])
    else:
        model.store_teacher()
        drift = compute_distribution_drift(
            _fc_previous_task_df["adaptive_drift"],
            train_df,
            FORECAST_DRIFT_COLUMNS,
        )
        ewc_scale, distill_scale = adaptive_strengths(drift)
        model.set_adaptive_strengths(drift, ewc_scale, distill_scale)
        console.print(
            f"  [cyan]Forecast drift={drift:.3f} "
            f"ewc_scale={ewc_scale:.3f} distill_scale={distill_scale:.3f}[/cyan]"
        )

    trainer = _make_lightning_trainer("adaptive_drift", task_id, has_val_loader=val_loader is not None)
    model = _prepare_forecast_model_for_fit(model)
    trainer.fit(model, train_loader, val_loader)

    n_batches = CONFIG["cl"]["ewc_fisher_samples"]
    model.compute_and_store_fisher(train_loader, n_batches=n_batches)

    _fc_previous_task_df["adaptive_drift"] = train_df.copy()
    _fc_model_state["adaptive_drift"] = model
    return model, trainer


FORECAST_TRAINERS = {
    "naive"                    : train_forecast_naive,
    "recall_ewc"               : train_forecast_recall_ewc,
    "adaptive_drift"           : train_forecast_adaptive_drift,
    "drift_adaptive_replay_ewc": train_forecast_drift_adaptive_replay_ewc,
}




# ─── Cell: RL CL Trainers (Naive / EWC / RECALL / SDFT) ─────────────────────

# Shared persistent state across tasks (per cl_method)
_rl_model_state : Dict[str, Optional[PPO]] = {m: None for m in CONFIG["cl_methods"]["rl"]}
_rl_ewc_engine  : Dict[str, PPOEWCEngine]  = {
    "recall_ewc": PPOEWCEngine(CONFIG["cl"]["ewc_lambda"]),
    "adaptive_drift": PPOEWCEngine(CONFIG["cl"]["ewc_lambda"]),
    "drift_adaptive_replay_ewc": PPOEWCEngine(CONFIG["cl"]["ewc_lambda"]),
}
_rl_recall_buf  : Dict[str, RLReplayBuffer] = {
    "recall_ewc": RLReplayBuffer(CONFIG["cl"]["recall_buffer_capacity"]),
    "drift_adaptive_replay_ewc": RLReplayBuffer(CONFIG["cl"]["recall_buffer_capacity"]),
}
_rl_teacher     : Dict[str, RLTeacherStore] = {"adaptive_drift": RLTeacherStore()}
_rl_previous_task_df: Dict[str, Optional[pd.DataFrame]] = {
    "adaptive_drift": None,
    "drift_adaptive_replay_ewc": None,
}


def _make_ppo(env: DummyVecEnv) -> PPO:
    rl_cfg = CONFIG["rl"]
    hw_cfg = CONFIG["hardware"]
    policy_kwargs = dict(net_arch=rl_cfg["net_arch"])
    return PPO(
        policy          = rl_cfg["policy"],
        env             = env,
        learning_rate   = rl_cfg["learning_rate"],
        n_steps         = rl_cfg["n_steps"],
        batch_size      = rl_cfg["batch_size"],
        n_epochs        = rl_cfg["n_epochs"],
        gamma           = rl_cfg["gamma"],
        gae_lambda      = rl_cfg["gae_lambda"],
        clip_range      = rl_cfg["clip_range"],
        ent_coef        = rl_cfg["ent_coef"],
        vf_coef         = rl_cfg["vf_coef"],
        max_grad_norm   = rl_cfg["max_grad_norm"],
        policy_kwargs   = policy_kwargs,
        device          = DEVICE,
        verbose         = 0,
        seed            = SEED,
        tensorboard_log = str(Path(CONFIG["paths"]["logs"]) / "rl"),
    )


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


# ── RL Trainer Functions ───────────────────────────────────────────────────────

def _get_or_init_rl(cl_method: str, env: DummyVecEnv) -> PPO:
    """Get existing model or create fresh one; set env if existing."""
    model = _rl_model_state[cl_method]
    if model is None:
        model = _make_ppo(env)
    else:
        model.set_env(env)
    return model


def train_rl_naive(task_id: int, task_df: pd.DataFrame) -> PPO:
    """Standard PPO fine-tuning with no forgetting mitigation."""
    env   = make_pricing_env(task_df)
    model = _get_or_init_rl("naive", env)
    model.learn(
        total_timesteps = CONFIG["rl"]["total_timesteps_per_task"],
        reset_num_timesteps = False,
        tb_log_name     = f"naive_task{task_id}",
    )
    _rl_model_state["naive"] = model
    return model


def train_rl_recall_ewc(task_id: int, task_df: pd.DataFrame) -> PPO:
    """RECALL + EWC PPO: replay old transitions and protect important weights."""
    env    = make_pricing_env(task_df)
    buf    = _rl_recall_buf["recall_ewc"]
    engine = _rl_ewc_engine["recall_ewc"]
    model  = _get_or_init_rl("recall_ewc", env)

    mix_n = CONFIG["cl"]["recall_mix_n_steps"]
    callbacks = [
        RECALLCallback(buf, mix_n, DEVICE),
        EWCCallbackRL(engine, DEVICE),
    ]

    model.learn(
        total_timesteps     = CONFIG["rl"]["total_timesteps_per_task"],
        reset_num_timesteps = False,
        callback            = callbacks,
        tb_log_name         = f"recall_ewc_task{task_id}",
    )

    # Store EWC state and collect current task transitions into replay buffer.
    raw_env = DynamicPricingEnv(task_df)
    engine.compute_fisher(model, raw_env, n_steps=CONFIG["cl"]["ewc_fisher_samples"] * 5)
    buf.add_from_env_rollout(
        raw_env, model,
        n = CONFIG["cl"]["recall_buffer_capacity"] // max(task_id, 1),
        task_id = task_id,
    )

    _rl_model_state["recall_ewc"] = model
    return model


def train_rl_drift_adaptive_replay_ewc(task_id: int, task_df: pd.DataFrame) -> PPO:
    """
    Drift-adaptive RECALL + EWC PPO: same replay/EWC mechanics as RECALL + EWC,
    but drift scales EWC and replay imitation strength.
    """
    method = "drift_adaptive_replay_ewc"
    env    = make_pricing_env(task_df)
    buf    = _rl_recall_buf[method]
    engine = _rl_ewc_engine[method]
    model  = _get_or_init_rl(method, env)

    drift = compute_distribution_drift(
        _rl_previous_task_df[method],
        task_df,
        RL_DRIFT_COLUMNS,
    )
    ewc_scale, _ = adaptive_strengths(drift)
    replay_scale = adaptive_replay_scale(drift)
    mix_n = max(1, int(CONFIG["cl"]["recall_mix_n_steps"] * replay_scale))
    bc_coef = 0.1 * replay_scale

    if task_id > 1:
        console.print(
            f"  [cyan]RL drift={drift:.3f} "
            f"ewc_scale={ewc_scale:.3f} replay_mix_n={mix_n} replay_coef={bc_coef:.3f}[/cyan]"
        )

    callbacks = [
        RECALLCallback(buf, mix_n, DEVICE, bc_coef=bc_coef),
        EWCCallbackRL(engine, DEVICE, scale=ewc_scale),
    ]

    model.learn(
        total_timesteps     = CONFIG["rl"]["total_timesteps_per_task"],
        reset_num_timesteps = False,
        callback            = callbacks,
        tb_log_name         = f"{method}_task{task_id}",
    )

    raw_env = DynamicPricingEnv(task_df)
    engine.compute_fisher(model, raw_env, n_steps=CONFIG["cl"]["ewc_fisher_samples"] * 5)
    buf.add_from_env_rollout(
        raw_env, model,
        n = CONFIG["cl"]["recall_buffer_capacity"] // max(task_id, 1),
        task_id = task_id,
    )

    _rl_previous_task_df[method] = task_df.copy()
    _rl_model_state[method] = model
    return model


def train_rl_adaptive_drift(task_id: int, task_df: pd.DataFrame) -> PPO:
    """
    Drift-adaptive replay-free PPO: combines EWC and SDFT KL regularization,
    reducing both when the new task distribution drifts more.
    """
    env     = make_pricing_env(task_df)
    teacher = _rl_teacher["adaptive_drift"]
    engine  = _rl_ewc_engine["adaptive_drift"]
    model   = _get_or_init_rl("adaptive_drift", env)

    drift = compute_distribution_drift(
        _rl_previous_task_df["adaptive_drift"],
        task_df,
        RL_DRIFT_COLUMNS,
    )
    ewc_scale, distill_scale = adaptive_strengths(drift)

    callbacks = [EWCCallbackRL(engine, DEVICE, scale=ewc_scale)]
    if task_id > 1:
        teacher.store(model)
        if distill_scale > 0:
            callbacks.append(
                SDFTCallback(
                    teacher,
                    CONFIG["cl"]["sdft_kl_coef"] * distill_scale,
                    DEVICE,
                )
            )
        console.print(
            f"  [cyan]RL drift={drift:.3f} "
            f"ewc_scale={ewc_scale:.3f} distill_scale={distill_scale:.3f}[/cyan]"
        )

    model.learn(
        total_timesteps     = CONFIG["rl"]["total_timesteps_per_task"],
        reset_num_timesteps = False,
        callback            = callbacks,
        tb_log_name         = f"adaptive_drift_task{task_id}",
    )

    raw_env = DynamicPricingEnv(task_df)
    engine.compute_fisher(model, raw_env, n_steps=CONFIG["cl"]["ewc_fisher_samples"] * 5)

    _rl_previous_task_df["adaptive_drift"] = task_df.copy()
    _rl_model_state["adaptive_drift"] = model
    return model


RL_TRAINERS = {
    "naive"                    : train_rl_naive,
    "recall_ewc"               : train_rl_recall_ewc,
    "adaptive_drift"           : train_rl_adaptive_drift,
    "drift_adaptive_replay_ewc": train_rl_drift_adaptive_replay_ewc,
}




# ─── Cell: Main Training Loop ─────────────────────────────────────────────────
# Implements the required nested structure:
#
#   for model_type in ["forecasting", "rl"]:
#       for cl_method in cl_methods[model_type]:
#           for task in tasks:
#               train(...)
#               evaluate_on_all_seen_tasks(...)
#               save_metrics_and_checkpoint(...)


# ── Helper: build and evaluate forecasting on a given task ───────────────────

# Cache of (val_loader, filtered_eval_df) keyed by (id(eval_task_df), id(train_ds)).
# Cross-task evaluation re-scores the same task frames repeatedly with the same
# training dataset, so the TimeSeriesDataSet only needs to be built once. The
# cache is cleared per cl_method in run_experiment to avoid id() reuse collisions.
_eval_loader_cache: Dict[tuple, tuple] = {}


def run_forecast_on_task(
    model      : CLTFT,
    eval_task_df: pd.DataFrame,
    train_ds   : TimeSeriesDataSet,
) -> Dict[str, float]:
    """Build (or reuse a cached) val loader for eval_task_df and run metrics."""
    min_rows_needed = min_tft_rows()
    if len(eval_task_df) < min_rows_needed:
        return {"mase": np.nan, "smape": np.nan, "rmse": np.nan}
    try:
        cache_key = (id(eval_task_df), id(train_ds))
        cached    = _eval_loader_cache.get(cache_key)
        if cached is None:
            hw = CONFIG["hardware"]
            fc = CONFIG["forecasting"]
            filtered = filter_tft_eval_frame(
                eval_task_df,
                min_length=min_rows_needed,
            )
            if filtered.empty:
                return {"mase": np.nan, "smape": np.nan, "rmse": np.nan}

            loader_kwargs = dict(
                batch_size   = fc["batch_size"],
                num_workers  = hw["num_workers"],
                pin_memory   = hw["pin_memory"],
                persistent_workers = hw["persistent_workers"] if hw["num_workers"] > 0 else False,
            )
            eval_ds = make_tft_dataset(
                filtered,
                train=False,
                training_dataset=train_ds,
            )
            val_loader = eval_ds.to_dataloader(train=False, shuffle=False, **loader_kwargs)
            cached = (val_loader, filtered)
            _eval_loader_cache[cache_key] = cached

        val_loader, filtered = cached
        # MASE baseline = the evaluated task's own demand history (standard
        # per-series scaling). Using a fixed Task-1 baseline for every task
        # inflated MASE on the larger-scale mega-sale tasks.
        return evaluate_forecasting(model, val_loader, filtered, task_id=0)
    except Exception as e:
        console.print(f"  [red]eval_forecast error: {e}[/red]")
        return {"mase": np.nan, "smape": np.nan, "rmse": np.nan}


# ── Main loop ─────────────────────────────────────────────────────────────────

def run_experiment(tft_tasks, rl_tasks):
    tasks       = CONFIG["tasks"]
    n_tasks     = len(tasks)
    model_types = CONFIG["model_types"]
    cl_methods  = CONFIG["cl_methods"]

    exp_start = time.time()
    console.print("\n" + "=" * 62)
    console.print("[bold magenta]  CONTINUAL LEARNING EXPERIMENT START[/bold magenta]")
    console.print("=" * 62)

    for model_type in model_types:
        console.print(f"\n[bold blue]═══ MODEL TYPE: {model_type.upper()} ═══[/bold blue]")

        # Reset per-method model state for each model_type
        if model_type == "forecasting":
            for k in _fc_model_state: _fc_model_state[k] = None
            _fc_replay_buf["recall_ewc"] = ForecastingReplayBuffer(CONFIG["cl"]["replay_buffer_size"])
            _fc_replay_buf["drift_adaptive_replay_ewc"] = ForecastingReplayBuffer(CONFIG["cl"]["replay_buffer_size"])
            _fc_previous_task_df["adaptive_drift"] = None
            _fc_previous_task_df["drift_adaptive_replay_ewc"] = None
        else:
            for k in _rl_model_state:  _rl_model_state[k] = None
            _rl_ewc_engine["recall_ewc"] = PPOEWCEngine(CONFIG["cl"]["ewc_lambda"])
            _rl_ewc_engine["adaptive_drift"] = PPOEWCEngine(CONFIG["cl"]["ewc_lambda"])
            _rl_ewc_engine["drift_adaptive_replay_ewc"] = PPOEWCEngine(CONFIG["cl"]["ewc_lambda"])
            _rl_recall_buf["recall_ewc"] = RLReplayBuffer(CONFIG["cl"]["recall_buffer_capacity"])
            _rl_recall_buf["drift_adaptive_replay_ewc"] = RLReplayBuffer(CONFIG["cl"]["recall_buffer_capacity"])
            _rl_teacher["adaptive_drift"] = RLTeacherStore()
            _rl_previous_task_df["adaptive_drift"] = None
            _rl_previous_task_df["drift_adaptive_replay_ewc"] = None

        for cl_method in cl_methods[model_type]:
            console.print(f"\n[bold cyan]  ── CL Method: {cl_method} ──[/bold cyan]")

            # Reset model for this cl_method
            if model_type == "forecasting":
                _fc_model_state[cl_method] = None
            else:
                _rl_model_state[cl_method] = None

            # ── Training & Evaluation Loop ─────────────────────────────────
            trained_tft_ds : Optional[TimeSeriesDataSet] = None   # for eval alignment
            _eval_loader_cache.clear()  # fresh eval-loader cache per cl_method

            for task_idx, task in enumerate(tasks):
                task_id   = task["task_id"]
                task_name = task["name"]

                console.print(f"\n  [yellow]Task {task_id}/{n_tasks}: {task_name}[/yellow]")

                try:
                    # ───────────────────────────────────────────────────────
                    # TRAIN on current task
                    # ───────────────────────────────────────────────────────
                    if model_type == "forecasting":
                        t_df = tft_tasks[task_idx]

                        train_ds, train_loader, val_loader = make_tft_loaders(t_df)

                        if train_loader is None:
                            console.print(f"  [red]Insufficient data, skipping[/red]")
                            continue

                        model, trainer = FORECAST_TRAINERS[cl_method](
                            task_id, train_loader, val_loader, train_ds, t_df
                        )

                        if trained_tft_ds is None:
                            trained_tft_ds = train_ds

                        # Save checkpoint
                        dummy_mase = np.nan  # filled after eval below

                    else:  # rl
                        t_df  = rl_tasks[task_idx]
                        model = RL_TRAINERS[cl_method](task_id, t_df)

                    console.print(f"  [green]✓ Training complete[/green]")

                    # ───────────────────────────────────────────────────────
                    # EVALUATE on ALL SEEN TASKS (tasks 0..task_idx)
                    # ───────────────────────────────────────────────────────
                    console.print(f"  Evaluating on {task_idx + 1} seen task(s)...")
                    mase_scores = []

                    for eval_idx in range(task_idx + 1):
                        eval_task    = tasks[eval_idx]
                        eval_task_id = eval_task["task_id"]

                        if model_type == "forecasting":
                            eval_df  = tft_tasks[eval_idx]
                            metrics  = run_forecast_on_task(model, eval_df, trained_tft_ds)
                            mase_scores.append(metrics.get("mase", np.nan))

                        else:
                            eval_df  = rl_tasks[eval_idx]
                            metrics  = evaluate_rl(model, eval_df)

                        LOGGER.log(
                            model_type    = model_type,
                            cl_method     = cl_method,
                            train_task_id = task_id,
                            eval_task_id  = eval_task_id,
                            metrics       = metrics,
                            eval_phase    = "seen",
                        )
                        metric_str = "  ".join(f"{k}={v:.4f}" for k,v in metrics.items() if not np.isnan(v))
                        console.print(f"    Eval task {eval_task_id}: {metric_str}")

                    if task_idx + 1 < n_tasks:
                        console.print(f"  Evaluating on {n_tasks - task_idx - 1} future task(s) for FWT...")

                        for eval_idx in range(task_idx + 1, n_tasks):
                            eval_task    = tasks[eval_idx]
                            eval_task_id = eval_task["task_id"]

                            if model_type == "forecasting":
                                eval_df = tft_tasks[eval_idx]
                                metrics = run_forecast_on_task(model, eval_df, trained_tft_ds)
                            else:
                                eval_df = rl_tasks[eval_idx]
                                metrics = evaluate_rl(model, eval_df)

                            LOGGER.log(
                                model_type    = model_type,
                                cl_method     = cl_method,
                                train_task_id = task_id,
                                eval_task_id  = eval_task_id,
                                metrics       = metrics,
                                eval_phase    = "future",
                            )
                            metric_str = "  ".join(f"{k}={v:.4f}" for k,v in metrics.items() if not np.isnan(v))
                            console.print(f"    Future eval task {eval_task_id}: {metric_str}")

                    # ───────────────────────────────────────────────────────
                    # SAVE CHECKPOINT
                    # ───────────────────────────────────────────────────────
                    if model_type == "forecasting":
                        primary = float(np.nanmean(mase_scores))
                        CKPT_MGR.save_forecasting(model, trainer, cl_method, task_id, primary)
                    else:
                        recent_metrics = evaluate_rl(model, rl_tasks[task_idx])
                        primary = recent_metrics.get("cumulative_profit", -np.inf)
                        CKPT_MGR.save_rl(model, cl_method, task_id, primary)

                except Exception as e:
                    console.print(f"  [red]ERROR: task {task_id} {cl_method}: {e}[/red]")
                    traceback.print_exc()
                    continue

    # ── Derive profit_index, then save all results ──────────────────────────────
    append_profit_index_metric("rl")
    LOGGER.save()
    total_time = time.time() - exp_start
    console.print(f"\n[bold green]═══ EXPERIMENT COMPLETE ({total_time/60:.1f} min) ═══[/bold green]")





# ─── Cell: BWT / FWT Computation ─────────────────────────────────────────────

def append_profit_index_metric(model_type: str = "rl",
                               base_metric: str = "cumulative_profit") -> None:
    """
    Derive and log a normalized ``profit_index`` for the RL methods.

        profit_index[i, j] = method_profit[i, j] / naive_profit[j, j]

    The denominator is the *naive* CL method's profit on task j right after it
    trained task j (the diagonal of naive's performance matrix) — a fixed
    per-task reference. This keeps the numbers readable (~1.0) while preserving
    forgetting structure: BWT on profit_index is just the raw BWT divided by a
    positive per-task constant, and every method's forgetting (including naive's
    own off-diagonal drop) stays visible. Logged as a metric so all downstream
    tables / plots / BWT-FWT treat it uniformly. Idempotent across re-runs.
    """
    df = LOGGER.to_dataframe()
    if df.empty:
        return
    sub = df[(df["model_type"] == model_type) & (df["metric_name"] == base_metric)]
    if sub.empty:
        return

    # Drop any previously injected profit_index rows so re-runs don't duplicate.
    LOGGER._records = [
        r for r in LOGGER._records
        if not (r["model_type"] == model_type and r["metric_name"] == "profit_index")
    ]

    # naive[j, j] — naive's fresh profit on each task (train_task == eval_task).
    naive = sub[sub["cl_method"] == "naive"]
    denom = {
        int(r["eval_task_id"]): float(r["metric_value"])
        for _, r in naive.iterrows()
        if int(r["train_task_id"]) == int(r["eval_task_id"])
    }
    if not denom:
        console.print("  [yellow]profit_index: no naive baseline found; skipping[/yellow]")
        return

    for _, r in sub.iterrows():
        j = int(r["eval_task_id"])
        d = denom.get(j)
        if d is None or not np.isfinite(d) or abs(d) <= 1e-9:
            val = np.nan
        else:
            val = float(r["metric_value"]) / d
        LOGGER.log(
            model_type    = r["model_type"],
            cl_method     = r["cl_method"],
            train_task_id = int(r["train_task_id"]),
            eval_task_id  = j,
            metrics       = {"profit_index": val},
            eval_phase    = r.get("eval_phase", "seen"),
        )


PRIMARY_METRICS = {
    "forecasting": ("mase", False),             # (metric, higher_is_better)
    "rl":          ("profit_index", True),      # readable index vs naive fresh per-task
}


def compute_all_cl_metrics() -> pd.DataFrame:
    """
    Compute CL metrics for every (model_type, cl_method) combination.
    Returns a summary DataFrame.
    """
    records = []
    df      = LOGGER.to_dataframe()

    for model_type in CONFIG["model_types"]:
        metric_name, higher_better = PRIMARY_METRICS[model_type]
        fwt_baseline_mat = LOGGER.get_perf_matrix(model_type, "naive", metric_name)

        for cl_method in CONFIG["cl_methods"][model_type]:
            mat = LOGGER.get_perf_matrix(model_type, cl_method, metric_name)
            bwt, fwt = compute_bwt_fwt(
                mat,
                primary_is_higher_better=higher_better,
                fwt_baseline_matrix=fwt_baseline_mat,
            )
            forgetting, avg_forgetting = compute_forgetting(
                mat,
                primary_is_higher_better=higher_better,
            )

            # Average performance on seen tasks at final task
            n_tasks     = len(CONFIG["tasks"])
            final_row   = mat[n_tasks - 1, :]
            avg_final   = float(np.nanmean(final_row[~np.isnan(final_row)]))
            diag_vals   = [mat[i, i] for i in range(n_tasks) if not np.isnan(mat[i, i])]
            avg_online  = float(np.nanmean(diag_vals)) if diag_vals else np.nan

            records.append({
                "model_type"          : model_type,
                "cl_method"           : cl_method,
                "primary_metric"      : metric_name,
                "avg_final_perf"      : round(avg_final,  4),
                "avg_online_perf"     : round(avg_online, 4),
                "bwt"                 : round(bwt, 4),
                "fwt"                 : round(fwt, 4),
                "forgetting"          : round(forgetting, 4),
                "avg_forgetting"      : round(avg_forgetting, 4),
            })

    return pd.DataFrame(records)


def build_fwt_details(save: bool = True) -> pd.DataFrame:
    """Return per-target-task FWT contributions from future-task evaluations."""
    records = []
    tasks = CONFIG["tasks"]
    n_tasks = len(tasks)

    for model_type in CONFIG["model_types"]:
        metric_name, higher_better = PRIMARY_METRICS[model_type]
        baseline_mat = LOGGER.get_perf_matrix(model_type, "naive", metric_name)

        for cl_method in CONFIG["cl_methods"][model_type]:
            mat = LOGGER.get_perf_matrix(model_type, cl_method, metric_name)

            for target_idx in range(1, n_tasks):
                baseline_perf = baseline_mat[target_idx - 1, target_idx]
                transfer_perf = mat[target_idx - 1, target_idx]

                if np.isnan(baseline_perf) or np.isnan(transfer_perf):
                    fwt_value = np.nan
                else:
                    diff = transfer_perf - baseline_perf
                    fwt_value = diff if higher_better else -diff

                records.append({
                    "model_type"          : model_type,
                    "cl_method"           : cl_method,
                    "primary_metric"      : metric_name,
                    "target_task_id"      : tasks[target_idx]["task_id"],
                    "target_task_name"    : tasks[target_idx]["name"],
                    "baseline_cl_method"  : "naive",
                    "baseline_train_task" : tasks[target_idx - 1]["task_id"],
                    "transfer_train_task" : tasks[target_idx - 1]["task_id"],
                    "baseline_perf"       : baseline_perf,
                    "transfer_perf"       : transfer_perf,
                    "fwt"                 : fwt_value,
                })

    details = pd.DataFrame(records)
    if save and not details.empty:
        path = Path(CONFIG["paths"]["results"]) / "fwt_details.csv"
        details.to_csv(path, index=False)
        console.print(f"  FWT details saved -> {path}")
    return details


def build_forgetting_details(save: bool = True) -> pd.DataFrame:
    """Return per-task forgetting based on final vs best earlier performance."""
    records = []
    tasks = CONFIG["tasks"]
    n_tasks = len(tasks)

    for model_type in CONFIG["model_types"]:
        metric_name, higher_better = PRIMARY_METRICS[model_type]

        for cl_method in CONFIG["cl_methods"][model_type]:
            mat = LOGGER.get_perf_matrix(model_type, cl_method, metric_name)

            for eval_idx in range(n_tasks - 1):
                previous_scores = mat[eval_idx:n_tasks - 1, eval_idx]
                valid_mask = ~np.isnan(previous_scores)
                final_perf = mat[n_tasks - 1, eval_idx]

                if not valid_mask.any() or np.isnan(final_perf):
                    best_previous_perf = np.nan
                    best_train_task = np.nan
                    forgetting = np.nan
                else:
                    valid_scores = previous_scores[valid_mask]
                    valid_rows = np.arange(eval_idx, n_tasks - 1)[valid_mask]

                    if higher_better:
                        best_pos = int(np.argmax(valid_scores))
                        forgetting = float(valid_scores[best_pos] - final_perf)
                    else:
                        best_pos = int(np.argmin(valid_scores))
                        forgetting = float(final_perf - valid_scores[best_pos])

                    best_previous_perf = float(valid_scores[best_pos])
                    best_train_task = tasks[int(valid_rows[best_pos])]["task_id"]
                    forgetting = max(0.0, forgetting)

                records.append({
                    "model_type"          : model_type,
                    "cl_method"           : cl_method,
                    "primary_metric"      : metric_name,
                    "eval_task_id"        : tasks[eval_idx]["task_id"],
                    "eval_task_name"      : tasks[eval_idx]["name"],
                    "best_train_task"     : best_train_task,
                    "final_train_task"    : tasks[n_tasks - 1]["task_id"],
                    "best_previous_perf"  : best_previous_perf,
                    "final_perf"          : final_perf,
                    "forgetting"          : forgetting,
                })

    details = pd.DataFrame(records)
    if save and not details.empty:
        path = Path(CONFIG["paths"]["results"]) / "forgetting_details.csv"
        details.to_csv(path, index=False)
        console.print(f"  Forgetting details saved -> {path}")
    return details


def build_cl_summary() -> pd.DataFrame:
    cl_summary = compute_all_cl_metrics()
    build_fwt_details(save=True)
    build_forgetting_details(save=True)
    console.print("\n[bold]CL Summary (BWT / FWT / Forgetting):[/bold]")
    console.print(cl_summary.to_string(index=False))
    return cl_summary



# ─── Cell: Results Comparison Tables ─────────────────────────────────────────

def make_comparison_table(model_type: str, metric: str) -> pd.DataFrame:
    """Pivot table: CL methods × Tasks, showing metric value."""
    df   = LOGGER.to_dataframe()
    sub  = df[(df["model_type"]==model_type) & (df["metric_name"]==metric)]
    sub  = sub[sub["train_task_id"] == sub["eval_task_id"]]
    if sub.empty:
        return pd.DataFrame()
    pivot = sub.pivot_table(
        index   = "cl_method",
        columns = "eval_task_id",
        values  = "metric_value",
        aggfunc = "mean",
    ).round(4)
    pivot.columns = [f"Task {c}" for c in pivot.columns]
    return pivot


def print_and_save_comparison_tables(cl_summary: Optional[pd.DataFrame] = None) -> Dict[str, pd.DataFrame]:
    cl_summary = cl_summary if cl_summary is not None else build_cl_summary()
    tables = {
        "forecast_mase": make_comparison_table("forecasting", "mase"),
        "forecast_smape": make_comparison_table("forecasting", "smape"),
        "rl_profit_index": make_comparison_table("rl", "profit_index"),
        "rl_profit": make_comparison_table("rl", "cumulative_profit"),
        "rl_regret": make_comparison_table("rl", "pricing_regret"),
    }
    labels = {
        "forecast_mase": "FORECASTING - MASE (lower is better)",
        "forecast_smape": "FORECASTING - sMAPE",
        "rl_profit_index": "RL - PROFIT INDEX (vs naive fresh per-task, higher is better)",
        "rl_profit": "RL - CUMULATIVE PROFIT (raw MYR, reference)",
        "rl_regret": "RL - PRICING REGRET (lower is better)",
    }
    for name, table in tables.items():
        print("\n" + "=" * 62)
        print(f"  {labels[name]}")
        print("=" * 62)
        if not table.empty:
            print(table.to_string())
            table.to_csv(f"{CONFIG['paths']['results']}/{name}.csv")
    print("\n" + "=" * 62)
    print("  CL METRICS - BWT / FWT / FORGETTING SUMMARY")
    print("=" * 62)
    print(cl_summary.to_string(index=False))
    return tables

# ─── Cell: Visualization ─────────────────────────────────────────────────────

plt.style.use("seaborn-v0_8-whitegrid")
COLORS = {
    "naive"                    : "#E74C3C",
    "recall_ewc"               : "#3498DB",
    "adaptive_drift"           : "#9B59B6",
    "drift_adaptive_replay_ewc": "#16A085",
}
METHOD_LABELS = {
    "naive"                    : "Naive",
    "recall_ewc"               : "RECALL + EWC",
    "adaptive_drift"           : "Drift-adaptive CL",
    "drift_adaptive_replay_ewc": "Drift-adaptive Replay + EWC",
}
MARKERS = {
    "naive"                    : "o",
    "recall_ewc"               : "s",
    "adaptive_drift"           : "^",
    "drift_adaptive_replay_ewc": "D",
}
LINESTYLES = {
    "naive"                    : "-",
    "recall_ewc"               : "--",
    "adaptive_drift"           : "-.",
    "drift_adaptive_replay_ewc": ":",
}
TASK_NAMES = [t["name"].replace("_", "\n") for t in CONFIG["tasks"]]

def get_results_df() -> pd.DataFrame:
    """Get latest in-memory metrics, falling back to saved CSV after reload."""
    df = LOGGER.to_dataframe()
    if df.empty:
        csv_path = Path(CONFIG["paths"]["results"]) / "all_metrics.csv"
        if csv_path.exists():
            df = pd.read_csv(csv_path)
    return df


def plot_metric_over_tasks(model_type: str, metric: str, ylabel: str,
                           title: str, lower_better: bool = True,
                           filename: str = ""):
    results_df = get_results_df()
    if results_df.empty:
        console.print(f"  [yellow]Skipped {title}: no metric records available[/yellow]")
        return

    methods = CONFIG["cl_methods"][model_type]
    n_tasks = len(CONFIG["tasks"])
    x_offsets = np.linspace(-0.06, 0.06, len(methods)) if len(methods) > 1 else np.array([0.0])
    fig, ax = plt.subplots(figsize=(11, 5))

    for method, x_offset in zip(methods, x_offsets):
        sub = results_df[
            (results_df["model_type"] == model_type) &
            (results_df["cl_method"]  == method) &
            (results_df["metric_name"]== metric) &
            (results_df["train_task_id"] == results_df["eval_task_id"])   # diagonal only
        ].sort_values("eval_task_id")

        if sub.empty: continue
        x_values = sub["eval_task_id"].astype(float) + x_offset
        ax.plot(
            x_values,
            sub["metric_value"],
            marker=MARKERS.get(method, "o"),
            linestyle=LINESTYLES.get(method, "-"),
            linewidth=2,
            markersize=7,
            markeredgecolor="white",
            markeredgewidth=0.8,
            alpha=0.92,
            color=COLORS.get(method, "#555"),
            label=METHOD_LABELS.get(method, method),
            zorder=3,
        )

    ax.set_xlim(0.5, n_tasks + 0.5)
    ax.set_xticks(range(1, n_tasks + 1))
    ax.set_xticklabels([f"T{i}" for i in range(1, n_tasks + 1)], fontsize=10)
    ax.set_xlabel("Task (trained on)", fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.legend(fontsize=10)

    # Shade task regime backgrounds
    for i, task in enumerate(CONFIG["tasks"], 1):
        color = "#FFF3CD" if task["regime"] == "baseline" else "#D6EAF8"
        ax.axvspan(i - 0.45, i + 0.45, alpha=0.3, color=color, zorder=0)

    plt.tight_layout()
    fpath = f"{CONFIG['paths']['plots']}/{filename or metric}_{model_type}.png"
    plt.savefig(fpath, dpi=150, bbox_inches="tight")
    plt.show()
    console.print(f"  Saved: {fpath}")


def plot_performance_matrix(model_type: str, metric: str, title: str,
                            lower_better: bool = True):
    """Heatmap of perf_matrix[i,j] = eval on task j after training on task i."""
    methods = CONFIG["cl_methods"][model_type]
    n = len(methods)
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 4), sharey=True)
    if n == 1: axes = [axes]

    for ax, method in zip(axes, methods):
        mat = LOGGER.get_perf_matrix(model_type, method, metric)
        cmap = "RdYlGn_r" if lower_better else "RdYlGn"
        im = ax.imshow(mat, cmap=cmap, aspect="auto")
        ax.set_title(METHOD_LABELS.get(method, method), fontsize=11, fontweight="bold")
        ax.set_xlabel("Eval Task")
        ax.set_ylabel("Train Task (up to)")
        n_tasks = len(CONFIG["tasks"])
        ax.set_xticks(range(n_tasks)); ax.set_xticklabels([f"T{i+1}" for i in range(n_tasks)], fontsize=8)
        ax.set_yticks(range(n_tasks)); ax.set_yticklabels([f"T{i+1}" for i in range(n_tasks)], fontsize=8)
        plt.colorbar(im, ax=ax, label=metric, fraction=0.046)
        # Mark diagonal
        for i in range(n_tasks):
            ax.plot(i, i, 'w*', markersize=10)

    fig.suptitle(f"{title} — Performance Matrix\n★ = online performance (diagonal)",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    fpath = f"{CONFIG['paths']['plots']}/perfmat_{metric}_{model_type}.png"
    plt.savefig(fpath, dpi=150, bbox_inches="tight")
    plt.show()
    console.print(f"  Saved: {fpath}")


def plot_bwt_fwt_bar(cl_summary: pd.DataFrame):
    """Bar chart comparing transfer and forgetting metrics."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for ax, model_type in zip(axes, CONFIG["model_types"]):
        sub     = cl_summary[cl_summary["model_type"] == model_type]
        methods = sub["cl_method"].tolist()
        x       = np.arange(len(methods))
        w       = 0.25

        bwt_vals = sub["bwt"].values
        fwt_vals = sub["fwt"].values
        forgetting_vals = sub["avg_forgetting"].values if "avg_forgetting" in sub else np.zeros(len(sub))
        colors   = [COLORS.get(m, "#888") for m in methods]

        bars1 = ax.bar(x - w, bwt_vals, w, label="BWT",
                       color=[c + "99" for c in colors], edgecolor="black", linewidth=0.8)
        bars2 = ax.bar(x, fwt_vals, w, label="FWT",
                       color=colors,  edgecolor="black", linewidth=0.8)
        bars3 = ax.bar(x + w, forgetting_vals, w, label="Avg Forgetting",
                       color=[c + "55" for c in colors], edgecolor="black", linewidth=0.8)

        ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
        ax.set_xticks(x)
        ax.set_xticklabels([METHOD_LABELS.get(m, m) for m in methods], fontsize=10, rotation=15)
        ax.set_title(f"{model_type.capitalize()} - Transfer & Forgetting", fontsize=12, fontweight="bold")
        ax.set_ylabel("Metric value\n(BWT/FWT higher is better; forgetting lower is better)", fontsize=9)
        ax.legend(fontsize=10)

        # Annotate
        for bar in list(bars1) + list(bars2) + list(bars3):
            h = bar.get_height()
            ax.annotate(f"{h:.3f}", xy=(bar.get_x() + bar.get_width()/2, h),
                        xytext=(0, 3 if h >= 0 else -10),
                        textcoords="offset points", ha="center", fontsize=8)

    plt.suptitle("Continual Learning Metrics (BWT / FWT / Avg Forgetting)", fontsize=13, fontweight="bold")
    plt.tight_layout()
    fpath = f"{CONFIG['paths']['plots']}/cl_transfer_forgetting_comparison.png"
    plt.savefig(fpath, dpi=150, bbox_inches="tight")
    plt.show()
    console.print(f"  Saved: {fpath}")


def plot_final_summary_table(cl_summary: pd.DataFrame):
    """Render cl_summary as a styled matplotlib table."""
    fig, ax = plt.subplots(figsize=(15, 3.8))
    ax.axis("off")

    display = cl_summary.copy()
    display["cl_method"]  = display["cl_method"].map(lambda m: METHOD_LABELS.get(m, m))
    display["model_type"] = display["model_type"].str.capitalize()
    display = display.rename(columns={
        "model_type"    : "Model",
        "cl_method"     : "Method",
        "primary_metric": "Primary Metric",
        "avg_final_perf": "Avg Final",
        "avg_online_perf": "Avg Online",
        "bwt"           : "BWT ↑",
        "fwt"           : "FWT ↑",
        "forgetting"    : "Forgetting ↓",
        "avg_forgetting": "Avg Forgetting ↓",
    })

    table = ax.table(
        cellText  = display.values,
        colLabels = display.columns,
        cellLoc   = "center",
        loc       = "center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.8)

    # Header style
    for j in range(len(display.columns)):
        table[0, j].set_facecolor("#2C3E50")
        table[0, j].set_text_props(color="white", fontweight="bold")

    # Highlight the proposed adaptive method.
    for i, row in enumerate(display.itertuples(), 1):
        if "Drift-adaptive" in str(row.Method):
            for j in range(len(display.columns)):
                table[i, j].set_facecolor("#EBF5FB")

    plt.title("Final Experiment Results Summary", fontsize=13, fontweight="bold", pad=20)
    plt.tight_layout()
    fpath = f"{CONFIG['paths']['plots']}/final_summary_table.png"
    plt.savefig(fpath, dpi=150, bbox_inches="tight")
    plt.show()
    console.print(f"  Saved: {fpath}")


def generate_all_plots(cl_summary: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    cl_summary = cl_summary if cl_summary is not None else build_cl_summary()
    Path(CONFIG["paths"]["plots"]).mkdir(exist_ok=True)
    console.print("\n[bold]Generating plots...[/bold]")
    plot_metric_over_tasks("forecasting", "mase", "MASE", "Forecasting MASE per Task (down is better)", True, "mase_online")
    plot_metric_over_tasks("forecasting", "smape", "sMAPE (%)", "Forecasting sMAPE per Task", True, "smape_online")
    plot_performance_matrix("forecasting", "mase", "Forecasting MASE", lower_better=True)
    plot_metric_over_tasks("rl", "profit_index", "Profit Index (vs naive fresh)", "RL Profit Index per Task (up is better)", False, "profit_index_online")
    plot_metric_over_tasks("rl", "pricing_regret", "Pricing Regret", "RL Pricing Regret per Task (down is better)", True, "regret_online")
    plot_performance_matrix("rl", "profit_index", "RL Profit Index", lower_better=False)
    plot_bwt_fwt_bar(cl_summary)
    plot_final_summary_table(cl_summary)
    console.print("[bold green]All plots saved to plots/[/bold green]")
    return cl_summary


if __name__ == "__main__":
    tft_tasks, rl_tasks, _, _ = prepare_data()
    print_task_summary(tft_tasks, rl_tasks)
    run_experiment(tft_tasks, rl_tasks)
    summary = build_cl_summary()
    print_and_save_comparison_tables(summary)
    generate_all_plots(summary)

