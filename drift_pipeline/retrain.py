# Phase 4: drift-triggered retrain dispatch for the M5 drift experiment.
#
# Each ARM is an independent walk-forward with its own controller and trigger
# history:
#   frozen   — never retrain (lower bound; from Phase 3)
#   periodic — retrain every N weeks on a fixed schedule (plain fine-tune)
#   naive    — drift-triggered retrain, NO continual-learning mechanism
#   ewc      — drift-triggered retrain, EWC-protected
#   replay   — drift-triggered retrain, replay-augmented
#   sdft     — drift-triggered retrain, self-distillation
#
# `naive` and `periodic` are both plain fine-tuning and differ only in WHEN they
# fire, which is exactly why both are needed. Against `frozen` alone, a CL arm
# that wins has two candidate explanations — the CL mechanism worked, or
# retraining at the right moment worked — and no way to separate them. `naive`
# shares the trigger with the CL arms and differs only in the mechanism, so it
# isolates the mechanism; `periodic` shares the mechanism and differs only in the
# trigger, so it isolates the trigger. Without `naive` the headline comparison
# cannot attribute its own result.
#
# On a forecasting trigger we retrain the TFT on the recent window (reusing BASE
# normalization so the input space stays fixed); on an RL trigger we continue PPO
# on the recent window. EWC Fisher / replay buffers / teachers are seeded from the
# base regime so the protections actually preserve pre-drift knowledge.

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import torch
import lightning as L
from stable_baselines3 import PPO

from .core_pipeline import CONFIG, DEVICE, console, slice_by_dates
from . import base_training as bt
from . import monitor as mon
from . import walk_state as ws
from .memory_accounting import MemoryLog, reset_peak

from .trainers import (
    build_cltft, make_tft_dataset, make_pricing_env, DynamicPricingEnv,
    ForecastingReplayBuffer, RLReplayBuffer, RLTeacherStore, PPOEWCEngine,
    RECALLCallback, EWCCallbackRL, SDFTCallback,
)


class _MixedLoader:
    """Interleaves current-window batches with replay batches (forecasting)."""
    def __init__(self, current_loader, replay_buf, mix_ratio):
        self.current, self.buf, self.mix_ratio = current_loader, replay_buf, mix_ratio
    def __iter__(self):
        import random as _r
        for batch in self.current:
            yield batch
            if _r.random() < self.mix_ratio and len(self.buf) > 0:
                s = self.buf.sample(1)
                if s:
                    yield s[0]
    def __len__(self):
        return len(self.current)
    @property
    def dataset(self):
        return self.current.dataset


@dataclass
class ArmStats:
    strategy: str
    n_fc_retrains: int = 0
    n_rl_retrains: int = 0
    fc_epochs_total: int = 0
    rl_steps_total: int = 0
    retrain_log: List[Dict] = field(default_factory=list)


class RetrainController:
    """Holds the current forecaster/pricer for one arm and retrains them on demand."""

    def __init__(self, strategy: str, base: Dict, data: Dict,
                 memlog: Optional[MemoryLog] = None):
        self.strategy = strategy
        self.data = data
        self.train_ds = base["train_ds"]
        self.forecaster = base["forecaster"]
        self.pricer = base["pricer"]
        self.stats = ArmStats(strategy)

        # CL state (seeded lazily from the base regime on first retrain).
        # NOTE for E4: every arm allocates all four, so process RSS cannot
        # attribute memory to a mechanism — see memory_accounting.
        self.fc_replay = ForecastingReplayBuffer(CONFIG["cl"]["replay_buffer_size"])
        self.rl_buf    = RLReplayBuffer(CONFIG["cl"]["recall_buffer_capacity"])
        self.rl_ewc    = PPOEWCEngine(CONFIG["cl"]["ewc_lambda"])
        self.rl_teacher = RLTeacherStore()
        self._fc_ewc_seeded = False
        self._fc_replay_seeded = False
        self._rl_ewc_seeded = False
        self._rl_buf_seeded = False
        self._fc_idx = 0
        self._rl_idx = 0
        self._last_periodic: Optional[pd.Timestamp] = None

        # E4 instrumentation. Always on: a snapshot is a few hundred microseconds
        # of pointer walking, and making it opt-in would mean the measurement is
        # missing from exactly the long runs worth measuring. The 'init' snapshot
        # is the zero point every later event is read against.
        self.memlog = memlog if memlog is not None else MemoryLog(strategy)
        self.memlog.snapshot(self, event="init")

    # ── providers (what the monitor evaluates at each check) ────────────────
    def forecaster_provider(self):
        return self.forecaster

    def pricer_provider(self):
        return self.pricer

    # ── window helpers ──────────────────────────────────────────────────────
    def _recent_tft(self, origin) -> pd.DataFrame:
        fc = CONFIG["forecasting"]
        weeks = CONFIG["retrain"]["recent_window_weeks"]
        # Pad by encoder_length so series have full training sequences.
        start = pd.Timestamp(origin) - pd.Timedelta(weeks=weeks) \
                - pd.Timedelta(days=fc["encoder_length"])
        # TRAINING sees what the node observed, which under a sync policy is
        # sales rather than demand. Evaluation keeps using the uncensored
        # `tft_full` via the monitor - see drift_pipeline/censoring.py. Absent
        # the censored key this is the uncensored control, by design.
        source = self.data.get("tft_full_censored", self.data["tft_full"])
        return slice_by_dates(source, start, pd.Timestamp(origin))

    def _base_tft(self) -> pd.DataFrame:
        """Base-regime frame used to seed EWC Fisher and the replay buffer.

        Also censored: the protections are supposed to preserve what the node
        actually learned, not a truth it never saw.
        """
        return self.data.get("tft_base_censored", self.data["tft_base"])

    def _recent_rl(self, origin) -> pd.DataFrame:
        weeks = CONFIG["retrain"]["recent_window_weeks"]
        start = pd.Timestamp(origin) - pd.Timedelta(weeks=weeks)
        return slice_by_dates(self.data["rl_full"], start, pd.Timestamp(origin))

    def _set_fc_teacher(self):
        """Freeze the current forecaster as the SDFT teacher via a state_dict copy.
        Avoids copy.deepcopy(model) — the TFT can hold non-leaf tensors cached by a
        forward pass during the walk-forward eval, which breaks deepcopy."""
        teacher = build_cltft(self.train_ds, cl_method="naive")
        # The forecaster may already carry a previous '.teacher' submodule (prior
        # SDFT retrain), so its state_dict has 'teacher.*' keys the fresh teacher
        # has no slots for — keep only the base TFT weights.
        sd = {k: v for k, v in self.forecaster.state_dict().items()
              if not k.startswith("teacher.")}
        teacher.load_state_dict(sd)
        teacher.to(DEVICE).eval()
        for p in teacher.parameters():
            p.requires_grad_(False)
        teacher.teacher = None
        self.forecaster.teacher = teacher

    def _fc_loader(self, df: pd.DataFrame):
        """Training loader that REUSES base normalization (from_dataset)."""
        ds = make_tft_dataset(df, train=False, training_dataset=self.train_ds)
        return ds.to_dataloader(train=True, shuffle=True,
                                batch_size=CONFIG["forecasting"]["batch_size"],
                                num_workers=0)

    def _fc_trainer(self) -> L.Trainer:
        hw = CONFIG["hardware"]
        return L.Trainer(
            max_epochs=CONFIG["retrain"]["retrain_epochs"],
            accelerator="gpu" if DEVICE == "cuda" else "cpu", devices=1,
            precision=hw["precision"], gradient_clip_val=CONFIG["forecasting"]["gradient_clip"],
            enable_model_summary=False, enable_progress_bar=False, logger=False,
            enable_checkpointing=False,   # retrained model kept in memory; no per-retrain ckpt
            log_every_n_steps=5,
        )

    # ── forecasting retrain ─────────────────────────────────────────────────
    def retrain_forecaster(self, origin, reason: str):
        df = self._recent_tft(origin)
        loader = self._fc_loader(df)
        model = self.forecaster
        strat = self.strategy
        # Peak VRAM is reported per-retrain, not cumulatively — see reset_peak.
        reset_peak()

        if strat == "ewc":
            model.cl_method = "ewc"
            if not self._fc_ewc_seeded:    # anchor on base regime
                base_loader = self._fc_loader(self._base_tft())
                model.compute_and_store_fisher(base_loader, CONFIG["cl"]["ewc_fisher_samples"])
                self._fc_ewc_seeded = True
            train_loader = loader
        elif strat == "replay":
            model.cl_method = "replay"
            if not self._fc_replay_seeded:
                base_loader = self._fc_loader(self._base_tft())
                self.fc_replay.add_from_loader(base_loader, task_id=0)
                self._fc_replay_seeded = True
            train_loader = (_MixedLoader(loader, self.fc_replay, CONFIG["cl"]["replay_mix_ratio"])
                            if len(self.fc_replay) > 0 else loader)
        elif strat == "sdft":
            model.cl_method = "sdft"
            self._set_fc_teacher()         # freeze pre-drift model as teacher
            train_loader = loader
        else:                               # naive / periodic — plain fine-tune
            model.cl_method = "naive"
            train_loader = loader

        model.train()
        self._fc_trainer().fit(model, train_loader)

        # accumulate protections for the NEXT retrain
        if strat == "ewc":
            model.compute_and_store_fisher(loader, CONFIG["cl"]["ewc_fisher_samples"])
        elif strat == "replay":
            self.fc_replay.add_from_loader(loader, task_id=self._fc_idx + 1)

        self.forecaster = model
        self._fc_idx += 1
        self.stats.n_fc_retrains += 1
        self.stats.fc_epochs_total += CONFIG["retrain"]["retrain_epochs"]
        self.stats.retrain_log.append(
            {"date": str(pd.Timestamp(origin).date()), "model": "forecasting",
             "strategy": strat, "reason": reason, "window_rows": int(len(df))})
        # After the protections have been accumulated for the NEXT retrain, so
        # the row reflects what this arm is now carrying forward.
        self.memlog.snapshot(self, event=f"retrain_fc_{reason}", date=origin,
                             model_type="forecasting")

    # ── RL retrain ──────────────────────────────────────────────────────────
    def retrain_pricer(self, origin, reason: str):
        df = self._recent_rl(origin)
        if df.empty:
            return
        env = make_pricing_env(df)
        model = self.pricer
        model.set_env(env)
        strat = self.strategy
        reset_peak()
        steps = CONFIG["retrain"]["retrain_timesteps"]
        fisher_steps = CONFIG["cl"]["ewc_fisher_samples"] * 5
        cbs = []

        if strat == "ewc":
            if not self._rl_ewc_seeded:
                self.rl_ewc.compute_fisher(model, DynamicPricingEnv(self.data["rl_base"]), fisher_steps)
                self._rl_ewc_seeded = True
            cbs.append(EWCCallbackRL(self.rl_ewc, DEVICE))
        elif strat == "replay":
            if not self._rl_buf_seeded:
                self.rl_buf.add_from_env_rollout(
                    DynamicPricingEnv(self.data["rl_base"]), model,
                    n=CONFIG["cl"]["recall_buffer_capacity"] // 2, task_id=0)
                self._rl_buf_seeded = True
            cbs.append(RECALLCallback(self.rl_buf, CONFIG["cl"]["recall_mix_n_steps"], DEVICE))
        elif strat == "sdft":
            self.rl_teacher.store(model)
            cbs.append(SDFTCallback(self.rl_teacher, CONFIG["cl"]["sdft_kl_coef"], DEVICE))
        # periodic: no callbacks (plain PPO continuation)

        model.learn(total_timesteps=steps, reset_num_timesteps=False,
                    callback=cbs or None)

        if strat == "ewc":
            self.rl_ewc.compute_fisher(model, DynamicPricingEnv(df), fisher_steps)
        elif strat == "replay":
            self.rl_buf.add_from_env_rollout(
                DynamicPricingEnv(df), model,
                n=CONFIG["cl"]["recall_buffer_capacity"] // 2, task_id=self._rl_idx + 1)

        self.pricer = model
        self._rl_idx += 1
        self.stats.n_rl_retrains += 1
        self.stats.rl_steps_total += steps
        self.stats.retrain_log.append(
            {"date": str(pd.Timestamp(origin).date()), "model": "rl",
             "strategy": strat, "reason": reason, "window_rows": int(len(df))})
        self.memlog.snapshot(self, event=f"retrain_rl_{reason}", date=origin,
                             model_type="rl")

    # ── hooks wired into the monitor ────────────────────────────────────────
    def on_trigger(self, kind, origin, ctx):
        if kind == "forecasting":
            console.print(f"    [yellow]drift retrain[/yellow] FC @ {pd.Timestamp(origin).date()}")
            self.retrain_forecaster(origin, reason="drift")
        elif kind == "rl":
            console.print(f"    [yellow]drift retrain[/yellow] RL @ {pd.Timestamp(origin).date()}")
            self.retrain_pricer(origin, reason="drift")

    def on_check_periodic(self, origin, idx):
        # Cadence-independent: retrain once `periodic_every_weeks` of wall-clock
        # time has elapsed since the last scheduled retrain (anchored at first check).
        origin = pd.Timestamp(origin)
        period_days = CONFIG["retrain"]["periodic_every_weeks"] * 7
        if self._last_periodic is None:
            self._last_periodic = origin
            return
        if (origin - self._last_periodic).days >= period_days:
            console.print(f"    [blue]periodic retrain[/blue] @ {origin.date()}")
            self.retrain_forecaster(origin, reason="periodic")
            self.retrain_pricer(origin, reason="periodic")
            self._last_periodic = origin


# ─── Arm orchestration ───────────────────────────────────────────────────────

#: Arms that retrain when the drift detector fires. `naive` belongs here despite
#: having no CL mechanism: it is the control that holds the trigger fixed so the
#: mechanism is the only thing varying. The retrain dispatch already routes
#: anything unrecognised to plain fine-tuning (`cl_method = "naive"`), so adding
#: it needed no new training path — only the trigger it was never wired to.
DRIFT_TRIGGERED = ("naive", "ewc", "replay", "sdft")


def run_arm(strategy: str, data: Dict, base: Optional[Dict] = None,
            checkpoint_every: int = 0, resume: bool = True) -> Dict:
    """Run one arm end-to-end. Loads a FRESH base (isolation) unless given one.

    checkpoint_every>0 snapshots the walk every N checks so an interrupted cell
    restarts from there instead of from zero — see drift_pipeline.walk_state.
    `resume` picks up such a snapshot if one is present; it defaults True because
    a stale checkpoint is only written by a run of this same cell that did not
    finish, which is exactly the case worth resuming.
    """
    base = base or mon.load_base()
    ctrl = RetrainController(strategy, base, data)

    on_trigger = ctrl.on_trigger if strategy in DRIFT_TRIGGERED else None
    on_check   = ctrl.on_check_periodic if strategy == "periodic" else None

    resume_state = None
    if resume:
        saved = ws.load(strategy)
        if saved is not None:
            resume_state = ws.restore(ctrl, saved)
            console.print(f"  [cyan]restored mid-walk state[/cyan] for '{strategy}' "
                          f"at check {resume_state['next_idx']}")

    on_checkpoint = None
    if checkpoint_every > 0:
        def on_checkpoint(next_idx, res_, fc_det, rl_det):        # noqa: F811
            ws.save(ctrl, res_, fc_det, rl_det, next_idx)

    res = mon.walk_forward(
        data, base, arm=strategy,
        forecaster_provider=ctrl.forecaster_provider,
        pricer_provider=ctrl.pricer_provider,
        on_trigger=on_trigger, on_check=on_check,
        resume_state=resume_state, on_checkpoint=on_checkpoint,
        checkpoint_every=checkpoint_every,
    )
    paths = mon.save_walk(res, base["calibration"])
    # The walk is over: a multi-GB buffer snapshot for a finished cell is waste,
    # and leaving it would make a re-run resume from the end of a completed walk.
    ws.clear(strategy)

    # persist the retrain log (when/why each arm retrained)
    log_path = Path(CONFIG["paths"]["results"]) / f"retrain_log_{strategy}.json"
    log_path.write_text(json.dumps({
        "strategy": strategy,
        "n_fc_retrains": ctrl.stats.n_fc_retrains,
        "n_rl_retrains": ctrl.stats.n_rl_retrains,
        "fc_epochs_total": ctrl.stats.fc_epochs_total,
        "rl_steps_total": ctrl.stats.rl_steps_total,
        "events": ctrl.stats.retrain_log,
    }, indent=2))
    paths["retrain_log"] = str(log_path)

    # E4: one row per (event, component). Written per arm rather than once at
    # the end so a run that dies on arm 3 still leaves arms 1-2 measured.
    mem_path = ctrl.memlog.save(
        Path(CONFIG["paths"]["results"]) / f"memory_{strategy}.csv")
    paths["memory_log"] = str(mem_path)

    peak = ctrl.memlog.summary()
    if not peak.empty:
        cl_peak = peak.loc[peak["component"] == "cl_state_total", "peak_mb"]
        if len(cl_peak):
            console.print(f"    [cyan]CL state peak[/cyan]: "
                          f"{float(cl_peak.iloc[0]):.1f} MB")

    console.print(f"[green]✓ arm '{strategy}'[/green]: "
                  f"{ctrl.stats.n_fc_retrains} FC + {ctrl.stats.n_rl_retrains} RL retrains  "
                  f"(fc_epochs={ctrl.stats.fc_epochs_total}, rl_steps={ctrl.stats.rl_steps_total})")
    return {"result": res, "controller": ctrl, "paths": paths}


def run_all_arms(data: Dict, arms: Optional[List[str]] = None) -> Dict[str, Dict]:
    """Run every arm (each loads its own fresh base for isolation).
    Default order: frozen, periodic, then the CL strategies (ewc/replay/sdft)."""
    if arms is None:
        arms = [*CONFIG["baselines"], *CONFIG["strategies"]]   # config-driven arm set
    seen, ordered = set(), []
    for a in arms:                          # de-dup, preserve order
        if a not in seen:
            seen.add(a); ordered.append(a)
    out = {}
    for arm in ordered:
        console.print(f"\n[bold magenta]== ARM: {arm} ==[/bold magenta]")
        out[arm] = run_arm(arm, data)
    return out
