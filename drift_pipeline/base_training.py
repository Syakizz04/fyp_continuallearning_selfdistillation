# Phase 2: base-model training + drift calibration for the M5 drift experiment.
#
# Trains ONE robust TFT forecaster and ONE PPO pricer on the base-fit window,
# then characterises "normal" behaviour on the held-out calibration tail so the
# drift thresholds (forecasting mu+k*sigma; RL profit floor) are grounded in data.
#
# Reuses the model/CL machinery in hybrid_pipeline.trainers. Because those
# trainers read hybrid_pipeline.core_pipeline.CONFIG, we first sync drift's shared
# config keys into it (the repo's mutate-CONFIG-in-place idiom) — no third copy
# of the 1500-line trainer module.

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import torch
import lightning as L
from lightning.pytorch.callbacks import EarlyStopping, LearningRateMonitor
from lightning.pytorch.loggers import TensorBoardLogger
from stable_baselines3 import PPO

from . import core_pipeline as dcfg
from .core_pipeline import CONFIG, DEVICE, SEED, console, slice_by_dates

import hybrid_pipeline.core_pipeline as hcfg
from hybrid_pipeline.trainers import (
    build_cltft, make_tft_dataset, make_tft_loaders, make_pricing_env,
    filter_tft_eval_frame, min_tft_rows, evaluate_forecasting, evaluate_rl,
)

_SHARED_KEYS = ["forecasting", "rl", "cl", "hardware", "paths"]


def sync_config() -> None:
    """Push drift's shared CONFIG keys into hybrid's CONFIG so hybrid trainers
    operate on M5 settings (M5 known_reals, item x store groups, drift paths).
    Idempotent; call before any training/eval."""
    for key in _SHARED_KEYS:
        hcfg.CONFIG[key] = copy.deepcopy(CONFIG[key])
    # hybrid's TimeSeriesDataSet code references CONFIG["forecasting"] only;
    # keep its other keys intact. Ensure output dirs exist under outputs/drift.
    dcfg.ensure_output_dirs()


# ─── Lightning trainer ───────────────────────────────────────────────────────

def _make_trainer(tag: str, has_val: bool) -> L.Trainer:
    hw, fc = CONFIG["hardware"], CONFIG["forecasting"]
    log_dir = str(Path(CONFIG["paths"]["logs"]) / "forecasting" / tag)
    callbacks = [LearningRateMonitor(logging_interval="epoch")]
    if has_val:
        callbacks.insert(0, EarlyStopping(
            monitor="val_loss", min_delta=fc.get("early_stop_min_delta", 1e-4),
            patience=fc["early_stop_patience"], mode="min", verbose=True,
        ))
    return L.Trainer(
        max_epochs           = fc["max_epochs"],
        accelerator          = "gpu" if DEVICE == "cuda" else "cpu",
        devices              = 1,
        precision            = hw["precision"],
        gradient_clip_val    = fc["gradient_clip"],
        enable_model_summary = False,
        enable_progress_bar  = True,
        enable_checkpointing = False,   # we save the base explicitly; avoids checkpoints/ pileup
        logger               = TensorBoardLogger(log_dir, name=""),
        callbacks            = callbacks,
        log_every_n_steps    = 5,
    )


# ─── Base model training ─────────────────────────────────────────────────────

def train_base_forecaster(tft_base: pd.DataFrame) -> Dict:
    """Train the base TFT on the base-fit window. Returns model + dataset template
    + the training frame (the latter feeds the MASE seasonal-naive denominator)."""
    sync_config()
    console.print("[bold]Training base forecaster (TFT)[/bold] "
                  f"on {len(tft_base):,} rows...")
    train_ds, train_loader, val_loader = make_tft_loaders(tft_base)
    model = build_cltft(train_ds, cl_method="naive")   # base = plain TFT, no CL penalty
    trainer = _make_trainer("base", has_val=val_loader is not None)
    model.train()
    trainer.fit(model, train_loader, val_loader)
    console.print("[green]✓ Base forecaster trained[/green]")
    return {"model": model, "train_ds": train_ds, "train_df": tft_base.copy(),
            "trainer": trainer}


def train_base_pricer(rl_base: pd.DataFrame) -> PPO:
    """Train the base PPO pricer on the base-fit window."""
    sync_config()
    budget = CONFIG["rl"]["total_timesteps_per_task"]
    console.print(f"[bold]Training base pricer (PPO)[/bold] for {budget:,} steps...")
    env = make_pricing_env(rl_base)
    rl = CONFIG["rl"]
    model = PPO(
        policy=rl["policy"], env=env, learning_rate=rl["learning_rate"],
        n_steps=rl["n_steps"], batch_size=rl["batch_size"], n_epochs=rl["n_epochs"],
        gamma=rl["gamma"], gae_lambda=rl["gae_lambda"], clip_range=rl["clip_range"],
        ent_coef=rl["ent_coef"], vf_coef=rl["vf_coef"], max_grad_norm=rl["max_grad_norm"],
        policy_kwargs=dict(net_arch=rl["net_arch"]), device=DEVICE, verbose=0, seed=SEED,
        tensorboard_log=str(Path(CONFIG["paths"]["logs"]) / "rl"),
    )
    model.learn(total_timesteps=budget, tb_log_name="base", reset_num_timesteps=True)
    console.print("[green]✓ Base pricer trained[/green]")
    return model


# ─── Windowed evaluation helpers ─────────────────────────────────────────────

def weekly_anchors(start: pd.Timestamp, end: pd.Timestamp,
                   horizon_days: int, step_days: int) -> List[pd.Timestamp]:
    """Forecast-origin dates D such that [D, D+horizon) fits within [start, end]."""
    last = end - pd.Timedelta(days=horizon_days)
    if last < start:
        return []
    return list(pd.date_range(start, last, freq=f"{step_days}D"))


def _eval_loader(frame: pd.DataFrame, train_ds):
    frame = filter_tft_eval_frame(frame, min_length=min_tft_rows())
    if frame.empty:
        return None
    ds = make_tft_dataset(frame, train=False, training_dataset=train_ds)
    return ds.to_dataloader(train=False, shuffle=False,
                            batch_size=CONFIG["forecasting"]["batch_size"], num_workers=0)


def window_forecast_error(model, tft_full: pd.DataFrame, train_ds,
                          base_train_df: pd.DataFrame, origin: pd.Timestamp) -> Dict:
    """MASE/sMAPE/RMSE for a single forecast at `origin`: encoder pulled from the
    `encoder_length` days before origin, targets = the `prediction_length` days
    from origin. Returns NaNs if the window can't form (handled upstream)."""
    fc = CONFIG["forecasting"]
    enc, pred = fc["encoder_length"], fc["prediction_length"]
    win_start = origin - pd.Timedelta(days=enc)
    win_end   = origin + pd.Timedelta(days=pred - 1)
    frame = slice_by_dates(tft_full, win_start, win_end)
    loader = _eval_loader(frame, train_ds)
    if loader is None:
        return {"mase": np.nan, "smape": np.nan, "rmse": np.nan}
    return evaluate_forecasting(model, loader, base_train_df, task_id=0)


def window_rl_profit(model: PPO, rl_full: pd.DataFrame, origin: pd.Timestamp) -> Dict:
    """PPO profit/reward/regret over the [origin, origin+prediction_length) window."""
    pred = CONFIG["forecasting"]["prediction_length"]
    frame = slice_by_dates(rl_full, origin, origin + pd.Timedelta(days=pred - 1))
    if frame.empty:
        return {"cumulative_profit": np.nan, "avg_episode_reward": np.nan,
                "pricing_regret": np.nan}
    return evaluate_rl(model, frame)


# ─── Calibration ─────────────────────────────────────────────────────────────

def calibrate_forecaster(model, tft_full, train_ds, base_train_df,
                         cal_start, cal_end) -> Dict:
    """Characterise 'normal' forecast error on the held-out calibration tail.
    Stores every window's error so thresholds at any k are re-derivable later."""
    sync_config()
    fc = CONFIG["forecasting"]
    step = int(CONFIG["timeline"].get("calib_step_days", 3))
    anchors = weekly_anchors(cal_start, cal_end, fc["prediction_length"], step)
    console.print(f"[bold]Calibrating forecaster[/bold] over {len(anchors)} windows "
                  f"({cal_start.date()} -> {cal_end.date()}, step {step}d)...")
    windows = []
    for d in anchors:
        m = window_forecast_error(model, tft_full, train_ds, base_train_df, d)
        windows.append({"date": str(d.date()), **m})
    mases = np.array([w["mase"] for w in windows if np.isfinite(w["mase"])], dtype=float)
    mu  = float(np.mean(mases)) if mases.size else float("nan")
    sig = float(np.std(mases, ddof=1)) if mases.size > 1 else float("nan")
    thresholds = {f"k={k}": (mu + k * sig) for k in CONFIG["drift"]["k_sensitivity"]}
    console.print(f"  normal MASE: mu={mu:.4f}  sigma={sig:.4f}  "
                  f"(n={mases.size})  thresholds={ {k: round(v,3) for k,v in thresholds.items()} }")
    return {"mase_mu": mu, "mase_sigma": sig, "n_windows": int(mases.size),
            "step_days": step, "thresholds": thresholds, "windows": windows}


def calibrate_pricer(model, rl_full, cal_start, cal_end) -> Dict:
    """Characterise the base policy's 'normal' profit level on the calibration tail.
    This profit level is the RL trigger reference: profit_index = current/this, and
    a sustained profit_index < floor signals the policy is losing money vs normal."""
    sync_config()
    fc = CONFIG["forecasting"]
    step = int(CONFIG["timeline"].get("calib_step_days", 3))
    anchors = weekly_anchors(cal_start, cal_end, fc["prediction_length"], step)
    console.print(f"[bold]Calibrating pricer[/bold] over {len(anchors)} windows...")
    windows = []
    for d in anchors:
        r = window_rl_profit(model, rl_full, d)
        windows.append({"date": str(d.date()), **r})
    profits = np.array([w["cumulative_profit"] for w in windows
                        if np.isfinite(w["cumulative_profit"])], dtype=float)
    mu  = float(np.mean(profits)) if profits.size else float("nan")
    sig = float(np.std(profits, ddof=1)) if profits.size > 1 else float("nan")
    console.print(f"  normal profit/window: mu={mu:.2f}  sigma={sig:.2f}  "
                  f"(n={profits.size})  floor(profit_index)={CONFIG['drift']['rl_profit_floor']}")
    return {"ref_profit_mu": mu, "ref_profit_sigma": sig, "n_windows": int(profits.size),
            "step_days": step, "profit_floor": CONFIG["drift"]["rl_profit_floor"],
            "windows": windows}


# ─── Persistence ─────────────────────────────────────────────────────────────

def save_base(fc_bundle: Dict, ppo_model: PPO,
              fc_calib: Dict, rl_calib: Dict) -> Dict[str, str]:
    """Persist base checkpoints (TFT + dataset template + PPO) and calibration.json."""
    ckpt_dir = Path(CONFIG["paths"]["checkpoints"]) / "base"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    res_dir = Path(CONFIG["paths"]["results"])
    res_dir.mkdir(parents=True, exist_ok=True)

    tft_ckpt = ckpt_dir / "base_tft.ckpt"
    ds_path  = ckpt_dir / "base_tft_dataset.pkl"
    ppo_path = ckpt_dir / "base_ppo.zip"
    meta_path = ckpt_dir / "base_meta.json"
    calib_path = res_dir / "calibration.json"

    fc_bundle["trainer"].save_checkpoint(str(tft_ckpt))
    fc_bundle["train_ds"].save(str(ds_path))
    ppo_model.save(str(ppo_path))

    # Persist the exact forecasting/cl config the base was trained with, so the
    # monitor rebuilds a matching architecture before load_state_dict (otherwise
    # a smoke-trained model won't load into a full-size rebuild, and vice versa).
    meta_path.write_text(json.dumps(
        {"forecasting": CONFIG["forecasting"], "cl": CONFIG["cl"]}, indent=2))

    calibration = {
        "config": {
            "base_fit"     : [str(d.date()) for d in dcfg.base_fit_window()],
            "calibration"  : [str(d.date()) for d in dcfg.calibration_window()],
            "fc_k_sigma"   : CONFIG["drift"]["fc_k_sigma"],
            "fc_consecutive": CONFIG["drift"]["fc_consecutive"],
            "rl_profit_floor": CONFIG["drift"]["rl_profit_floor"],
            "rl_consecutive": CONFIG["drift"]["rl_consecutive"],
            "k_sensitivity": CONFIG["drift"]["k_sensitivity"],
        },
        "forecasting": fc_calib,
        "rl": rl_calib,
    }
    calib_path.write_text(json.dumps(calibration, indent=2))
    console.print(f"[green]✓ Saved base checkpoints + calibration[/green] -> {ckpt_dir}")
    return {"tft": str(tft_ckpt), "dataset": str(ds_path), "ppo": str(ppo_path),
            "meta": str(meta_path), "calibration": str(calib_path)}


def run_base_training_and_calibration(data: Dict) -> Dict:
    """End-to-end Phase 2: train base TFT+PPO, calibrate, save. `data` comes from
    core_pipeline.prepare_drift_data()."""
    sync_config()
    cal_start, cal_end = dcfg.calibration_window()

    fc_bundle = train_base_forecaster(data["tft_base"])
    ppo_model = train_base_pricer(data["rl_base"])

    fc_calib = calibrate_forecaster(
        fc_bundle["model"], data["tft_full"], fc_bundle["train_ds"],
        fc_bundle["train_df"], cal_start, cal_end,
    )
    rl_calib = calibrate_pricer(ppo_model, data["rl_full"], cal_start, cal_end)

    paths = save_base(fc_bundle, ppo_model, fc_calib, rl_calib)
    return {"fc_bundle": fc_bundle, "ppo_model": ppo_model,
            "fc_calib": fc_calib, "rl_calib": rl_calib, "paths": paths}
