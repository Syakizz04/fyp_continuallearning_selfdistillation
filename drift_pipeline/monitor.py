# Phase 3: walk-forward drift monitor for the M5 drift experiment.
#
# Loads the base TFT+PPO and calibration, walks forward weekly through the
# walk-forward span, and at each check computes the forecasting error (windowed
# MASE) and the RL profit_index against the calibrated reference. The RAW stream
# is logged every check so triggers at any k are re-derivable WITHOUT rerunning.
#
# Drift detection is debounced (N consecutive breaches). The retrain dispatch
# (Phase 4) plugs into the `on_trigger` callback; Phase 3 runs the FROZEN base
# (no retrain) — which both validates detection and is one of our baseline arms.

from __future__ import annotations

import copy
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional

import numpy as np
import pandas as pd
import torch
from stable_baselines3 import PPO

from . import core_pipeline as dcfg
from .core_pipeline import CONFIG, DEVICE, console
from . import base_training as bt

from .trainers import build_cltft, TimeSeriesDataSet


# ─── Base model + calibration loading ────────────────────────────────────────

def load_base(paths: Optional[Dict[str, str]] = None) -> Dict:
    """Restore base TFT (architecture-matched + state_dict), base PPO, and the
    calibration reference. Applies the persisted base config so the rebuilt TFT
    matches the checkpoint regardless of smoke/full settings."""
    ckpt_dir = Path(CONFIG["paths"]["checkpoints"]) / "base"
    res_dir  = Path(CONFIG["paths"]["results"])
    paths = paths or {
        "tft":  str(ckpt_dir / "base_tft.ckpt"),
        "dataset": str(ckpt_dir / "base_tft_dataset.pkl"),
        "ppo":  str(ckpt_dir / "base_ppo.zip"),
        "meta": str(ckpt_dir / "base_meta.json"),
        "calibration": str(res_dir / "calibration.json"),
    }

    # Re-apply the exact base-train forecasting/cl config.
    meta = json.loads(Path(paths["meta"]).read_text())
    CONFIG["forecasting"] = meta["forecasting"]
    CONFIG["cl"]          = meta["cl"]
    bt.sync_config()

    train_ds = TimeSeriesDataSet.load(paths["dataset"])
    model = build_cltft(train_ds, cl_method="naive")
    state = torch.load(paths["tft"], map_location=DEVICE)
    model.load_state_dict(state["state_dict"] if "state_dict" in state else state)
    model.to(DEVICE).eval()

    pricer = PPO.load(paths["ppo"], device=DEVICE)
    calibration = json.loads(Path(paths["calibration"]).read_text())

    console.print("[green]✓ Loaded base TFT + PPO + calibration[/green]")
    return {"forecaster": model, "train_ds": train_ds, "pricer": pricer,
            "calibration": calibration}


# ─── Debounced threshold detector ────────────────────────────────────────────

class DebouncedDetector:
    """Fires when `value` breaches `threshold` for `consecutive` checks in a row.
    direction='above' -> breach when value > threshold (forecasting error);
    direction='below' -> breach when value < threshold (RL profit_index)."""

    def __init__(self, threshold: float, consecutive: int, direction: str = "above"):
        self.threshold = threshold
        self.consecutive = consecutive
        self.direction = direction
        self.run = 0

    def update(self, value: float) -> bool:
        if value is None or not np.isfinite(value):
            self.run = 0                    # NaN window: no signal, reset streak
            return False
        breached = (value > self.threshold) if self.direction == "above" \
                   else (value < self.threshold)
        self.run = self.run + 1 if breached else 0
        if self.run >= self.consecutive:
            self.run = 0                    # reset after firing (next streak is fresh)
            return True
        return False


def rederive_triggers(stream: List[Dict], *, mu: float, sigma: float, k: float,
                      consecutive: int, metric: str = "mase",
                      direction: str = "above") -> List[str]:
    """Replay the logged stream at a different k and return the trigger dates.
    For RL, pass metric='profit_index', direction='below', and the floor as
    `mu` with sigma=0, k=0 (threshold = floor)."""
    threshold = mu + k * sigma
    det = DebouncedDetector(threshold, consecutive, direction)
    fired = []
    for row in stream:
        if det.update(row.get(metric)):
            fired.append(row["date"])
    return fired


# ─── Walk-forward monitor ────────────────────────────────────────────────────

@dataclass
class WalkResult:
    arm: str
    stream: List[Dict] = field(default_factory=list)      # per-check raw metrics
    triggers: List[Dict] = field(default_factory=list)    # fired events
    n_fc_triggers: int = 0
    n_rl_triggers: int = 0


def walk_forward(
    data: Dict,
    base: Dict,
    arm: str = "frozen",
    forecaster_provider: Optional[Callable] = None,
    pricer_provider: Optional[Callable] = None,
    on_trigger: Optional[Callable] = None,
    on_check: Optional[Callable] = None,
    resume_state: Optional[Dict] = None,
    on_checkpoint: Optional[Callable] = None,
    checkpoint_every: int = 0,
) -> WalkResult:
    """Step weekly through the walk-forward span, logging the raw drift stream and
    firing debounced triggers.

    forecaster_provider()/pricer_provider() return the model to use at each check
    (defaults: the frozen base). on_trigger(kind, origin, ctx) is called when a
    trigger fires — Phase 4 wires retraining here; Phase 3 leaves it None.
    on_check(origin, idx) runs before each check's metrics — used by the periodic
    baseline arm to retrain on a fixed schedule (independent of drift).

    resume_state (from drift_pipeline.walk_state) restarts a partially-walked arm:
    it restores the accumulated stream, the trigger counts and — easy to forget —
    the detectors' consecutive-breach streaks, which would otherwise reset and
    under-trigger. on_checkpoint(next_idx, res, fc_det, rl_det) is invoked every
    `checkpoint_every` checks so the caller can snapshot; 0 disables it.
    """
    bt.sync_config()
    fc_cfg, drift = CONFIG["forecasting"], CONFIG["drift"]
    calib = base["calibration"]

    mu, sigma = calib["forecasting"]["mase_mu"], calib["forecasting"]["mase_sigma"]
    fc_threshold = mu + drift["fc_k_sigma"] * sigma
    ref_profit   = calib["rl"]["ref_profit_mu"]
    profit_floor = drift["rl_profit_floor"]

    fc_det = DebouncedDetector(fc_threshold, drift["fc_consecutive"], "above")
    rl_det = DebouncedDetector(profit_floor, drift["rl_consecutive"], "below")

    get_fc = forecaster_provider or (lambda: base["forecaster"])
    get_rl = pricer_provider or (lambda: base["pricer"])

    res = WalkResult(arm=arm)
    checks = data["checks"]
    start_idx = 0
    if resume_state:
        res.stream = list(resume_state.get("stream", []))
        res.triggers = list(resume_state.get("triggers", []))
        res.n_fc_triggers = int(resume_state.get("n_fc_triggers", 0))
        res.n_rl_triggers = int(resume_state.get("n_rl_triggers", 0))
        fc_det.run = int(resume_state.get("fc_det_run", 0))
        rl_det.run = int(resume_state.get("rl_det_run", 0))
        start_idx = int(resume_state.get("next_idx", 0))

    console.print(f"[bold]Walk-forward[/bold] arm='{arm}' over {len(checks)} weekly checks "
                  f"(FC thr={fc_threshold:.3f} @k={drift['fc_k_sigma']}, "
                  f"RL floor={profit_floor})...")
    if start_idx:
        console.print(f"  [cyan]resuming at check {start_idx}/{len(checks)}[/cyan] "
                      f"({res.n_fc_triggers} FC + {res.n_rl_triggers} RL triggers so far)")

    for idx, origin in enumerate(checks):
        if idx < start_idx:
            continue
        if on_check is not None:
            on_check(origin, idx)
        fc_model = get_fc()
        rl_model = get_rl()

        fc_err = bt.window_forecast_error(
            fc_model, data["tft_full"], base["train_ds"], data["tft_base"], origin)
        rl_prof = bt.window_rl_profit(rl_model, data["rl_full"], origin)

        cum = rl_prof.get("cumulative_profit")
        profit_index = (float(cum) / ref_profit) if (cum is not None
                        and np.isfinite(cum) and abs(ref_profit) > 1e-9) else np.nan

        row = {
            "date": str(pd.Timestamp(origin).date()),
            "arm": arm,
            "mase":  fc_err.get("mase"),
            "smape": fc_err.get("smape"),
            "rmse":  fc_err.get("rmse"),
            "cumulative_profit": cum,
            "profit_index": profit_index,
            "avg_episode_reward": rl_prof.get("avg_episode_reward"),
            "pricing_regret": rl_prof.get("pricing_regret"),
            "fc_threshold": fc_threshold,
            "fc_mu": mu, "fc_sigma": sigma,
            "rl_profit_floor": profit_floor, "rl_ref_profit": ref_profit,
        }

        if fc_det.update(row["mase"]):
            res.n_fc_triggers += 1
            ev = {"date": row["date"], "kind": "forecasting", "mase": row["mase"],
                  "threshold": fc_threshold}
            res.triggers.append(ev)
            row["fc_trigger"] = True
            if on_trigger:
                on_trigger("forecasting", origin, {"data": data, "base": base, "row": row})
        else:
            row["fc_trigger"] = False

        if rl_det.update(row["profit_index"]):
            res.n_rl_triggers += 1
            ev = {"date": row["date"], "kind": "rl", "profit_index": row["profit_index"],
                  "floor": profit_floor}
            res.triggers.append(ev)
            row["rl_trigger"] = True
            if on_trigger:
                on_trigger("rl", origin, {"data": data, "base": base, "row": row})
        else:
            row["rl_trigger"] = False

        res.stream.append(row)

        # After the row is appended, so the checkpoint's `next_idx` means "the
        # first check NOT yet in the stream" and a resume neither repeats nor
        # skips one.
        if (on_checkpoint is not None and checkpoint_every > 0
                and (idx + 1) % checkpoint_every == 0 and (idx + 1) < len(checks)):
            on_checkpoint(idx + 1, res, fc_det, rl_det)

    console.print(f"  [cyan]{arm}[/cyan]: {res.n_fc_triggers} forecasting + "
                  f"{res.n_rl_triggers} RL triggers over {len(checks)} checks")
    return res


# ─── Persistence + k-sensitivity report ──────────────────────────────────────

def save_walk(res: WalkResult, calibration: Dict) -> Dict[str, str]:
    """Persist the raw stream (CSV) + triggers (JSON), and the k-sensitivity table
    re-derived from the logged stream at k in CONFIG['drift']['k_sensitivity']."""
    res_dir = Path(CONFIG["paths"]["results"])
    res_dir.mkdir(parents=True, exist_ok=True)
    stream_path  = res_dir / f"drift_stream_{res.arm}.csv"
    trig_path    = res_dir / f"triggers_{res.arm}.json"
    ksens_path   = res_dir / f"k_sensitivity_{res.arm}.json"

    pd.DataFrame(res.stream).to_csv(stream_path, index=False)

    mu = calibration["forecasting"]["mase_mu"]
    sigma = calibration["forecasting"]["mase_sigma"]
    cons = CONFIG["drift"]["fc_consecutive"]
    ksens = {}
    for k in CONFIG["drift"]["k_sensitivity"]:
        dates = rederive_triggers(res.stream, mu=mu, sigma=sigma, k=k,
                                  consecutive=cons, metric="mase", direction="above")
        ksens[f"k={k}"] = {"threshold": mu + k * sigma,
                           "n_triggers": len(dates), "dates": dates}

    trig_path.write_text(json.dumps(
        {"arm": res.arm, "n_fc_triggers": res.n_fc_triggers,
         "n_rl_triggers": res.n_rl_triggers, "events": res.triggers}, indent=2))
    ksens_path.write_text(json.dumps(ksens, indent=2))

    console.print(f"[green]✓ Saved drift stream + triggers + k-sensitivity[/green] -> {res_dir}")
    console.print("  k-sensitivity (forecasting retrains): "
                  + "  ".join(f"k={k.split('=')[1]}:{v['n_triggers']}"
                              for k, v in ksens.items()))
    return {"stream": str(stream_path), "triggers": str(trig_path),
            "k_sensitivity": str(ksens_path)}


def run_frozen_walk(data: Dict, base: Optional[Dict] = None) -> Dict:
    """Phase 3 entry point: load base (if not given), walk forward with the frozen
    model, persist the stream + triggers + k-sensitivity."""
    base = base or load_base()
    res = walk_forward(data, base, arm="frozen")
    paths = save_walk(res, base["calibration"])
    return {"result": res, "paths": paths, "base": base}
