"""
In-process model loading and inference.

This module is what makes the system "edge" in the sense the supervisor
specified: the TFT and PPO are loaded into **this process's memory once at
startup** and every prediction is a local function call. There is no inference
service to call out to, and no network hop between a request and a model.

It deliberately reuses `drift_pipeline.monitor.load_base()` rather than
reimplementing checkpoint loading. That function already handles the fiddly
part - restoring the exact base-train `forecasting`/`cl` config from
`base_meta.json` before rebuilding the architecture, so the rebuilt TFT matches
the checkpoint regardless of whatever CONFIG happens to hold at import time.
Duplicating that logic here is how the served model and the experiment model
silently drift apart.

Note the import cost: pulling this in drags in torch, lightning,
pytorch-forecasting and stable-baselines3. That is fine for an edge node, which
needs them to retrain anyway, but it is why the dashboard reads CSVs instead of
importing this.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd


@dataclass
class ModelVersion:
    """Identity of the models currently being served.

    `generation` increments on every local retrain, so the control plane and the
    dashboard can tell which node is serving what without comparing weights.
    """
    generation: int = 0
    source: str = "base"            # "base" | "sdft" | "replay" | "ewc"
    loaded_at: float = field(default_factory=time.time)
    retrained_at: Optional[float] = None
    trigger_reason: Optional[str] = None

    def as_dict(self) -> Dict:
        return {
            "generation": self.generation,
            "source": self.source,
            "loaded_at": self.loaded_at,
            "retrained_at": self.retrained_at,
            "trigger_reason": self.trigger_reason,
        }


class LocalInference:
    """
    Holds this node's forecaster and pricer in memory and serves from them.

    Thread-safety: `_swap_lock` guards model replacement. Retraining runs in a
    background thread and produces a *new* model object, which is swapped in
    under the lock. Readers take the lock only to grab a reference, so requests
    keep being served from the old model for the whole duration of a retrain -
    which is the behaviour a deployed system needs, and also what makes the
    "retrain locally without going offline" claim true rather than aspirational.
    """

    def __init__(self, node: str) -> None:
        self.node = node
        self._swap_lock = threading.RLock()
        self._forecaster = None
        self._pricer = None
        self._train_ds = None
        self._calibration: Optional[Dict] = None
        self._data: Optional[Dict] = None
        self.version = ModelVersion()
        self.load_seconds: Optional[float] = None
        self._env = None
        self._row_index: Dict[tuple, int] = {}
        self._inv_scale: float = 1.0

    # ── Startup ─────────────────────────────────────────────────────────────

    def load(self, *, data_dir: Optional[str] = None) -> None:
        """Load base checkpoints and the walk-forward data into this process."""
        started = time.perf_counter()

        # Imported lazily: this is the heavy import, and keeping it inside the
        # call means `import edge_system.edge` stays cheap for anything that
        # only needs the dataclasses above.
        from drift_pipeline import monitor as mon
        from drift_pipeline.core_pipeline import CONFIG as DRIFT_CONFIG
        from drift_pipeline.core_pipeline import prepare_drift_data

        from ..config import SYSTEM_CONFIG

        data_dir = data_dir or SYSTEM_CONFIG["paths"]["data_dir"]
        demand_path = str(Path(data_dir) / "demand_forecasting.csv")
        rl_path = str(Path(data_dir) / "rl_environment.csv")

        # prepare_drift_data mutates the drift CONFIG (adapt_config_to_data drops
        # constant columns). It must run BEFORE load_base, which rebuilds the TFT
        # architecture against that config.
        self._data = prepare_drift_data(demand_path, rl_path)

        # `load_base()` defaults to `<drift checkpoints>/base`, which silently
        # ignores `paths.base_ckpt_dir` - so pointing a node at an alternative
        # checkpoint (the pricer retrained on realistic inventory, say) would
        # have loaded the original and reported success. Build the paths here.
        ckpt_dir = Path(SYSTEM_CONFIG["paths"]["base_ckpt_dir"])
        if not ckpt_dir.exists():
            raise FileNotFoundError(f"no base checkpoints at {ckpt_dir}")
        # Calibration lives beside the checkpoint when one was produced by
        # `experiments.retrain_pricer`, and in the shared results dir otherwise.
        calib = ckpt_dir / "calibration.json"
        if not calib.exists():
            calib = Path(DRIFT_CONFIG["paths"]["results"]) / "calibration.json"

        base = mon.load_base({
            "tft": str(ckpt_dir / "base_tft.ckpt"),
            "dataset": str(ckpt_dir / "base_tft_dataset.pkl"),
            "ppo": str(ckpt_dir / "base_ppo.zip"),
            "meta": str(ckpt_dir / "base_meta.json"),
            "calibration": str(calib),
        })
        self.ckpt_dir = str(ckpt_dir)
        self._forecaster = base["forecaster"]
        self._pricer = base["pricer"]
        self._train_ds = base["train_ds"]
        self._calibration = base["calibration"]

        self.load_seconds = time.perf_counter() - started
        self.version = ModelVersion(generation=0, source="base")

    @property
    def ready(self) -> bool:
        return self._forecaster is not None and self._pricer is not None

    # ── Accessors used by the monitor and the retrainer ─────────────────────

    def forecaster(self):
        with self._swap_lock:
            return self._forecaster

    def pricer(self):
        with self._swap_lock:
            return self._pricer

    @property
    def data(self) -> Dict:
        if self._data is None:
            raise RuntimeError("LocalInference.load() has not been called")
        return self._data

    @property
    def calibration(self) -> Dict:
        if self._calibration is None:
            raise RuntimeError("LocalInference.load() has not been called")
        return self._calibration

    @property
    def base_bundle(self) -> Dict:
        """The dict shape `RetrainController` expects for its `base` argument."""
        return {
            "forecaster": self.forecaster(),
            "pricer": self.pricer(),
            "train_ds": self._train_ds,
            "calibration": self.calibration,
        }

    def swap(self, *, forecaster=None, pricer=None, source: str,
             reason: Optional[str] = None) -> ModelVersion:
        """Atomically replace the served models after a local retrain."""
        with self._swap_lock:
            if forecaster is not None:
                self._forecaster = forecaster
            if pricer is not None:
                self._pricer = pricer
            self.version = ModelVersion(
                generation=self.version.generation + 1,
                source=source,
                loaded_at=self.version.loaded_at,
                retrained_at=time.time(),
                trigger_reason=reason,
            )
            # The env is a property of the DATA, not the weights, so it survives
            # a retrain untouched. Rebuilding it here would re-run `_precompute`
            # over the whole panel on every hot-swap for no reason.
            return self.version

    # ── Inference ───────────────────────────────────────────────────────────

    def price(self, sku: str, *, sim_date: str, inventory_level: float,
              demand_forecast: Optional[float] = None) -> Dict:
        """
        Ask the local PPO agent for a price tier for one SKU.

        `inventory_level` is supplied by the caller rather than read from the
        dataset, because in the deployed system it is the node's **stale view**
        from the escrow layer, not ground truth. Threading it through as an
        argument is what makes E2 possible: substituting a stale value for the
        true one is the entire manipulation.
        """
        env, idx = self._env_for(sku, sim_date)
        obs = self._observation(env, idx, inventory_level, demand_forecast)

        action, _ = self.pricer().predict(obs, deterministic=True)
        action = int(np.asarray(action).ravel()[0])

        # DynamicPricingEnv applies tiers as base_price * (1 + tier), NOT as a
        # bare multiplier - see its _precompute().
        tier = float(env.price_tiers[action])
        base_price = float(env._base_price[idx])

        return {
            "sku": sku,
            "action": action,
            "tier": tier,
            "price": base_price * (1.0 + tier),
            "base_price": base_price,
            "inventory_level_used": inventory_level,
            "model_generation": self.version.generation,
        }

    def forecast(self, sku: str, *, sim_date: str) -> Dict:
        """Local TFT forecast for one SKU as of `sim_date`."""
        import torch  # noqa: PLC0415
        from hybrid_pipeline.trainers import (  # noqa: PLC0415
            make_tft_dataset, filter_tft_eval_frame,
        )
        from drift_pipeline.core_pipeline import CONFIG as DRIFT_CONFIG

        fc_cfg = DRIFT_CONFIG["forecasting"]
        end = pd.Timestamp(sim_date)
        start = end - pd.Timedelta(days=fc_cfg["encoder_length"] + fc_cfg["prediction_length"])

        df = self.data["tft_full"]
        window = df[(df["product_id"] == sku) & (df["date"] > start) & (df["date"] <= end)]
        window = filter_tft_eval_frame(window)
        if window.empty:
            # Deliberate: short series return NaN rather than raising, matching
            # the offline pipeline's behaviour so metrics stay comparable.
            return {"sku": sku, "forecast": None,
                    "reason": "insufficient history for encoder+prediction length"}

        ds = make_tft_dataset(window, train=False, training_dataset=self._train_ds,
                              predict_mode=True)
        loader = ds.to_dataloader(train=False, batch_size=64, num_workers=0)
        model = self.forecaster()
        with torch.no_grad():
            preds = model.predict(loader, mode="prediction")
        arr = np.asarray(preds.cpu() if hasattr(preds, "cpu") else preds, dtype=float)
        return {
            "sku": sku,
            "forecast": float(np.nanmean(arr)),
            "horizon": fc_cfg["prediction_length"],
            "model_generation": self.version.generation,
        }

    # ── Internals ───────────────────────────────────────────────────────────

    def _row(self, sku: str, sim_date: str) -> pd.Series:
        rl = self.data["rl_full"]
        m = rl[(rl["product_id"] == sku) & (rl["date"] == pd.Timestamp(sim_date))]
        if m.empty:
            raise KeyError(f"no RL row for {sku} on {sim_date}")
        return m.iloc[0]

    # State-vector positions, from DynamicPricingEnv._get_obs(). Named here so
    # the override below is legible instead of a bare magic index.
    _OBS_DEMAND_FORECAST = 0
    _OBS_INVENTORY = 1

    def _shared_env(self):
        """
        ONE env over the whole multi-SKU panel, built once.

        The obvious implementation - an env per SKU, built from that SKU's slice
        - is wrong, and silently so. `DynamicPricingEnv._precompute` normalises
        its columns by the max **within the frame it is given**, including
        `elasticity_coefficient`, which is constant within a SKU. Slice per SKU
        and that column normalises to exactly -1.0 for every item: the
        `elasticity_norm` state feature loses all its information, and every SKU
        looks equally price-sensitive to the agent.

        Training builds the env from a multi-SKU frame (`make_pricing_env` on the
        base window), so the agent learned on `elasticity_norm` spread across
        roughly [-1, -0.02]. Serving it a constant -1.0 is train/serve skew in a
        state feature - the agent would be answering a question it was never
        asked. Sharing one panel-wide env keeps the normalisation basis the same
        kind of object as the one it trained on.

        Residual caveat, worth a line in the limitations: the basis here is the
        full walk-forward panel while training used the base window, so the
        divisors are close but not identical.
        """
        from hybrid_pipeline.trainers import DynamicPricingEnv  # noqa: PLC0415

        if self._env is None:
            rl = (self.data["rl_full"]
                  .sort_values(["product_id", "date"]).reset_index(drop=True))
            self._env = DynamicPricingEnv(rl)
            self._row_index = {
                (pid, date): i for i, (pid, date)
                in enumerate(zip(rl["product_id"], rl["date"]))
            }
            self._inv_scale = 1.0  # superseded by env.inventory_obs()
        return self._env

    def _env_for(self, sku: str, sim_date: str):
        """Return (env, row index) for this SKU at this date."""
        env = self._shared_env()
        idx = self._row_index.get((sku, pd.Timestamp(sim_date)))
        if idx is None:
            raise KeyError(f"no RL row for {sku} on {sim_date}")
        return env, idx

    @staticmethod
    def _inventory_scale(env) -> float:
        """
        Recover the divisor that maps raw `inventory_level` onto the env's
        observation slot.

        Inventory is normalised **twice**: `build_rl_features` divides by the
        global max to make `inventory_norm`, then the env's `safe_norm` divides
        again by that column's max within this SKU's slice. Rather than
        reconstruct both constants, invert the composition empirically from a
        row where the normalised value is non-zero - that stays correct even if
        either normalisation changes.
        """
        raw = env.df["inventory_level"].to_numpy(dtype=float)
        norm = np.asarray(env._inventory, dtype=float)
        usable = np.flatnonzero((norm > 0) & np.isfinite(norm) & (raw > 0))
        if not len(usable):
            return 1.0
        i = int(usable[0])
        return float(raw[i] / norm[i])

    def _observation(self, env, idx: int, inventory_level: float,
                     demand_forecast: Optional[float]) -> np.ndarray:
        """Build the state vector, overriding inventory with the node's view."""
        obs = np.asarray(env._get_obs(idx), dtype=np.float32).copy()

        # Ask the env to do the mapping rather than inverting it here: under the
        # cover-based scaling the divisor is per-SKU, so it cannot be recovered
        # from a single row, and a serving-side reimplementation is exactly how
        # the served observation stops matching the trained one.
        obs[self._OBS_INVENTORY] = np.float32(
            env.inventory_obs(idx, float(inventory_level)))

        if demand_forecast is not None:
            dmax = float(np.max(env.df["demand_forecast"].to_numpy(dtype=float)))
            if dmax > 0:
                obs[self._OBS_DEMAND_FORECAST] = np.float32(demand_forecast / dmax)
        return obs
