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
        self.sdft_alpha: Optional[float] = None
        # Forecaster-specific provenance. `version.generation` counts BOTH
        # models' retrains, so it cannot answer "when did the forecast last
        # change?" on its own.
        self.fc_generation: int = 0
        self.fc_updated_at: Optional[float] = None
        self.fc_updated_sim_date: Optional[str] = None
        self._env = None
        self._row_index: Dict[tuple, int] = {}
        self._inv_scale: float = 1.0
        # What this node has actually seen trade, overlaid on the dataset for the
        # SKUs it sells (see `apply_realised`). None until the first check.
        self._realised_truth: Optional[pd.DataFrame] = None
        self._realised_obs: Optional[pd.DataFrame] = None
        self.realised_days: int = 0
        self._realised_sku_set: set = set()
        # Pricing drift signal (see `pricing_outcome`): the stock view each
        # quote was priced against, by (sku, sim_date), so the after-the-day
        # question is asked of the same state; and the critic's line onto
        # the training reward, fitted once per pricer object.
        self._quoted_inventory: Dict[tuple, float] = {}
        self._critic_fit_for: Optional[int] = None
        self._critic_line: Optional[tuple] = None
        self._critic_lock = threading.Lock()

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
        # In place, AFTER load_base: load_base rebinds DRIFT_CONFIG["cl"] to the
        # dict the forecaster holds (base_meta.json's alpha=0.5), so this is the
        # one write the model actually reads. Read back off the model rather
        # than trusted - a rebind instead of a mutation would leave the node
        # silently retraining at 0.5.
        alpha = float(SYSTEM_CONFIG["models"]["sdft_alpha"])
        DRIFT_CONFIG["cl"]["sdft_alpha"] = alpha
        got = base["forecaster"].cl_cfg.get("sdft_alpha")
        if got != alpha:
            raise RuntimeError(f"sdft_alpha did not reach the forecaster: it holds "
                               f"{got!r}, expected {alpha!r}")
        self.sdft_alpha = alpha

        self.ckpt_dir = str(ckpt_dir)
        self._forecaster = base["forecaster"]
        self._pricer = base["pricer"]
        self._train_ds = base["train_ds"]
        self._calibration = base["calibration"]

        self.load_seconds = time.perf_counter() - started
        self.version = ModelVersion(generation=0, source="base")
        self.fc_generation = 0
        self.fc_updated_at = self.version.loaded_at
        self.fc_updated_sim_date = None          # None = the base checkpoint

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
             reason: Optional[str] = None,
             sim_date: Optional[str] = None) -> ModelVersion:
        """Atomically replace the served models after a local retrain."""
        with self._swap_lock:
            if forecaster is not None:
                self._forecaster = forecaster
                self.fc_generation += 1
                self.fc_updated_at = time.time()
                self.fc_updated_sim_date = sim_date
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
        self._remember_quote(sku, sim_date, inventory_level)

        return {
            "sku": sku,
            "action": action,
            "tier": tier,
            "price": base_price * (1.0 + tier),
            "base_price": base_price,
            "inventory_level_used": inventory_level,
            "model_generation": self.version.generation,
        }

    #: Quantiles surfaced to the operator as the uncertainty band.
    BAND = (0.1, 0.5, 0.9)
    #: Days of realised demand returned alongside the forecast.
    HISTORY_DAYS = 28

    # ── Pricing drift signal: the pricer's own expectation vs the outcome ────
    #
    # The pricing counterpart of the forecast check. The forecaster is judged
    # on its own predictions against what happened (MASE); the pricer carries
    # its own prediction too - PPO's critic, V(s), the return it expects from a
    # state - so it is judged the same way: the day's training reward, summed,
    # against what the critic expected, summed.
    #
    # The critic is asked AFTER the day, with the demand that actually turned
    # up (units requested, refused included - a shop sees those) in its demand
    # slot instead of the forecast. Asked with the forecast it explained 18-46%
    # of the reward and, like the revenue index it replaces, mostly tracked
    # demand swings; with hindsight demand, day-to-day demand noise sits on both
    # sides and cancels, leaving how well the price did. No true demand curve
    # is used: only the node's own critic, its quotes, and the orders it saw.
    #
    # What it actually measures (experiments/critic_compare.py, 2026-11-22).
    # Price is not among V(s)'s inputs (DynamicPricingEnv._get_obs), so V
    # cannot tell which tier was charged. Price reaches it only through the
    # hindsight units, which also scale the actual reward, so the units cancel.
    # What is left is a per-unit-margin gauge: it falls when prices sit below
    # what this policy usually charges. Measured on the rebuilt base_v4, over
    # 17 non-overlapping trials:
    #   * It catches a pricer stuck at -10% (pos 14/17 vs revenue 1/17).
    #   * It does not catch one stuck at base or at +10%. The floor is
    #     one-sided, and +10% raises margin.
    #   * It does not catch a 2x price-sensitive world (an elasticity shift
    #     looks like ordinary demand).
    #   * A one-number markup index, reward over M x base x sold, matches it,
    #     and beats it on the marketplace.
    # So treat it as a markup alarm, not a judge of pricing quality. The earlier
    # "13/15 and 12/15" claims came from a lost scratch script on the old
    # pricer, and they do not reproduce.
    #
    # The action never changes the next state (step reads the next dataset
    # row). But PPO trains with gamma 0.99 over SKU-ordered rows, so V(s) is
    # mostly a discounted return over the same SKU's coming days, not today's
    # reward. That is why the fitted slope b is tiny (~0.014). The line
    # r = a + b V is fitted on the training window. The monitor's warm-up
    # absorbs whatever offset the live world adds: the sim applies raw
    # elasticity, while the env applies a normalised one.

    #: Quotes kept for the after-the-day question, in days.
    QUOTE_DAYS = 3

    def _remember_quote(self, sku: str, sim_date: str, inventory_level: float) -> None:
        with self._critic_lock:
            self._quoted_inventory[(sku, sim_date)] = float(inventory_level)
            if len(self._quoted_inventory) > 4096:            # bounded; oldest dates first
                keep = sorted({d for _, d in self._quoted_inventory})[-self.QUOTE_DAYS:]
                self._quoted_inventory = {k: v for k, v in self._quoted_inventory.items()
                                          if k[1] in keep}

    @staticmethod
    def _reward(env, idx, price, sold, lost):
        """DynamicPricingEnv.step's reward, for given units (arrays)."""
        cost = env._unit_cost[idx]
        if env.inventory_constrained:
            r = (price - cost) * sold - env.lost_sale_penalty * price * lost
        else:
            r = (price - cost) * sold - 0.15 * price * sold * env._stockout[idx]
        return np.clip(r / 1000.0, -5.0, 5.0)

    def _critic_values(self, pricer, obs: np.ndarray) -> np.ndarray:
        import torch  # noqa: PLC0415
        with torch.no_grad():
            t = torch.as_tensor(obs, dtype=torch.float32, device=pricer.policy.device)
            return pricer.policy.predict_values(t).cpu().numpy().ravel()

    def critic_line(self) -> tuple:
        """(a, b) with expected reward = a + b V(s), fitted on the training
        window for the pricer now served. Refitted after a pricer swap (the new
        critic has its own scale); a few seconds over ~60k rows."""
        pricer = self.pricer()
        with self._critic_lock:
            if self._critic_fit_for == id(pricer) and self._critic_line is not None:
                return self._critic_line
        import torch  # noqa: PLC0415
        env = self._shared_env()
        base_dates = pd.to_datetime(self.data["rl_base"]["date"]).unique()
        idx = np.flatnonzero(pd.to_datetime(env.df["date"]).isin(base_dates).to_numpy())
        obs = np.stack([env._get_obs(int(i)) for i in idx]).astype(np.float32)
        with torch.no_grad():
            t = torch.as_tensor(obs, device=pricer.policy.device)
            acts = pricer.policy._predict(t, deterministic=True).cpu().numpy().ravel()
        tiers = np.asarray(env.price_tiers, dtype=float)[acts]
        base = env._base_price[idx].astype(float)
        price = base * (1.0 + tiers)
        demand = np.maximum(env._realized_demand[idx].astype(float), 0.0)
        asked = np.maximum(0.0, np.round(demand * (price / np.maximum(base, 1e-6))
                                         ** env._elasticity[idx].astype(float)))
        stock = env._inventory_raw[idx].astype(float) if env.inventory_constrained else asked
        sold = np.minimum(asked, stock)
        r = self._reward(env, idx, price, sold, asked - sold)
        obs[:, self._OBS_DEMAND_FORECAST] = asked / self._demand_divisor(env)
        v = self._critic_values(pricer, obs)
        b, a = np.polyfit(v, r, 1)
        line = (float(a), float(b))
        with self._critic_lock:
            self._critic_fit_for, self._critic_line = id(pricer), line
        return line

    @staticmethod
    def _demand_divisor(env) -> float:
        dmax = float(np.max(env.df["demand_forecast"].to_numpy(dtype=float)))
        return dmax if dmax > 0 else 1.0

    def pricing_outcome(self, *, sim_date: str, outcomes, share: float) -> Optional[tuple]:
        """(actual, expected) training reward over one channel's day.

        `outcomes` holds, per SKU, the price charged and the units requested /
        sold / refused at this channel. The channel sees `share` of each SKU's
        demand, so units are scaled up by 1/share onto the SKU-level demand the
        pricer was trained on (the ratio is unaffected; the critic's demand slot
        is not). None when nothing traded."""
        if share <= 0:
            return None
        env = self._shared_env()
        day = pd.Timestamp(sim_date)
        rows, price, asked, sold, lost, inv = [], [], [], [], [], []
        for o in outcomes:
            i = self._row_index.get((o["sku"], day))
            if i is None or o.get("price") is None:
                continue
            rows.append(i)
            price.append(float(o["price"]))
            asked.append(float(o["requested"]) / share)
            sold.append(float(o["sold"]) / share)
            lost.append(float(o["refused"]) / share)
            with self._critic_lock:
                q = self._quoted_inventory.get((o["sku"], sim_date))
            inv.append(float(env._inventory_raw[i]) if q is None else q)
        if not rows:
            return None
        a, b = self.critic_line()
        rows_a = np.asarray(rows)
        r = self._reward(env, rows_a, np.asarray(price), np.asarray(sold), np.asarray(lost))
        obs = np.stack([self._observation(env, i, q, d) for i, q, d in zip(rows, inv, asked)])
        v = self._critic_values(self.pricer(), obs)
        return float(np.sum(r)), float(np.sum(a + b * v))

    def forecast(self, sku: str, *, sim_date: str) -> Dict:
        """
        The next `prediction_length` days of demand for one SKU, as seen on
        `sim_date`, with a p10-p90 band.

        Forward-looking: the encoder ends ON `sim_date` and the decoder covers
        the following days. The decoder still needs those days' known
        covariates (calendar, sell_price, snap, events) - which a retailer has in
        advance - but never their demand: `demand` is the only unknown real, and
        TFT reads unknowns from the encoder alone.

        (The previous version anchored the window to END on `sim_date`, i.e. it
        "forecast" the fortnight that had just happened. Nothing called it, so
        nothing depended on that.)
        """
        import torch  # noqa: PLC0415
        from drift_pipeline.trainers import (  # noqa: PLC0415
            make_tft_dataset, filter_tft_eval_frame,
        )
        from drift_pipeline.core_pipeline import CONFIG as DRIFT_CONFIG

        fc_cfg = DRIFT_CONFIG["forecasting"]
        enc, horizon = fc_cfg["encoder_length"], fc_cfg["prediction_length"]
        origin = pd.Timestamp(sim_date)
        start = origin - pd.Timedelta(days=enc)            # exclusive
        end = origin + pd.Timedelta(days=horizon)          # inclusive

        # A SKU this node trades is forecast from what it has observed selling,
        # so a demand shock shows up in the forecast; others from the dataset.
        obs = self._realised_obs
        df = obs if obs is not None and sku in self._realised_skus else self.data["tft_full"]
        window = df[(df["product_id"] == sku) & (df["date"] > start) & (df["date"] <= end)]
        window = filter_tft_eval_frame(window)
        if window.empty:
            # Deliberate: short series return no forecast rather than raising,
            # matching the offline pipeline's NaN-not-crash behaviour.
            return {"sku": sku, "forecast": None, "daily": [],
                    "reason": "insufficient history for encoder+prediction length",
                    **self._forecaster_meta()}

        ds = make_tft_dataset(window, train=False, training_dataset=self._train_ds,
                              predict_mode=True)
        loader = ds.to_dataloader(train=False, batch_size=64, num_workers=0)
        model = self.forecaster()
        with torch.no_grad():
            q = model.predict(loader, mode="quantiles")
        q = np.asarray(q.cpu() if hasattr(q, "cpu") else q, dtype=float)[0]  # (horizon, n_q)

        # Recent realised demand, for context on the chart. Only days up to and
        # including `sim_date` - what the node has actually observed.
        seen = window[window["date"] <= origin].tail(self.HISTORY_DAYS)
        history = [{"date": pd.Timestamp(d).strftime("%Y-%m-%d"), "actual": float(v)}
                   for d, v in zip(seen["date"], seen["demand"])]

        levels = [float(x) for x in model.loss.quantiles]
        idx = {b: levels.index(b) for b in self.BAND}
        dates = pd.date_range(origin + pd.Timedelta(days=1), periods=q.shape[0], freq="D")
        # Demand is a count: a negative quantile is a regression artefact, not
        # information, and would draw a band below zero.
        daily = [{"date": d.strftime("%Y-%m-%d"),
                  "p10": max(0.0, q[i, idx[0.1]]),
                  "p50": max(0.0, q[i, idx[0.5]]),
                  "p90": max(0.0, q[i, idx[0.9]])}
                 for i, d in enumerate(dates)]
        return {
            "sku": sku,
            "as_of": origin.strftime("%Y-%m-%d"),
            "horizon": int(q.shape[0]),
            # Kept for older callers: mean daily p50 over the horizon.
            "forecast": float(np.mean([r["p50"] for r in daily])),
            "total_p50": float(sum(r["p50"] for r in daily)),
            "daily": daily,
            "history": history,
            **self._forecaster_meta(),
        }

    def _forecaster_meta(self) -> Dict:
        """When the forecaster being served was last changed - the operator's
        'last updated', which is NOT the same as the node's model generation
        (a pricer retrain bumps that too)."""
        return {
            "model_generation": self.version.generation,
            "forecaster_generation": self.fc_generation,
            "forecaster_updated_at": self.fc_updated_at,
            "forecaster_updated_sim_date": self.fc_updated_sim_date,
        }

    def realised_mase(self, *, sim_date: str) -> Optional[float]:
        """
        This node's forecast error on demand it has ALREADY observed.

        The offline walk scores `window_forecast_error(origin)` over
        `[origin - enc, origin + pred)` - it looks ahead of its origin, which is
        fine offline. A deployed node on `sim_date` only knows demand up to that
        day, so the same window is anchored to END on `sim_date`: origin =
        sim_date - (pred - 1). Same function, same pooled-rolling definition the
        drift threshold was calibrated on, only realised data.
        """
        from drift_pipeline import base_training as bt  # noqa: PLC0415
        from drift_pipeline.core_pipeline import CONFIG as DRIFT_CONFIG

        pred = DRIFT_CONFIG["forecasting"]["prediction_length"]
        origin = pd.Timestamp(sim_date) - pd.Timedelta(days=pred - 1)
        out = bt.window_forecast_error(self.forecaster(), self.data["tft_full"],
                                       self._train_ds, self.data["tft_base"], origin)
        m = out.get("mase")
        return float(m) if m is not None and np.isfinite(m) else None

    # ── Realised demand (the live forecaster check) ─────────────────────────

    @property
    def _realised_skus(self) -> set:
        f = self._realised_obs
        return set() if f is None else self._realised_sku_set

    def apply_realised(self, daily: Dict[str, list], *, share: float) -> int:
        """Overlay what this node has seen trade onto the dataset, for the SKUs
        it sells. Returns how many realised days the overlay covers.

        `daily` is the inventory service's per-SKU series for THIS node's
        channel. A node forecasts SKU-level demand but only sees its own
        channel, so its counts are scaled up by the channel's share - an edge
        node estimating the whole from its part, which is also why each node's
        drift signal (and so its retrain timing) differs.

        Two frames, following E2's rule (drift_pipeline/censoring.py):
          truth - everything customers ORDERED, granted or refused. The check's
                  yardstick: a demand shock moves it.
          obs   - what was SOLD. Refused orders never become sales, so this is
                  censored demand. Training and the shown forecast use it,
                  because a deployed node cannot record what it refused.
        Days not yet traded keep the dataset's history - what the node knew at
        deployment.
        """
        if not daily:
            return 0
        skus = sorted(daily)
        base = self.data["tft_full"]
        frame = base[base["product_id"].isin(skus)].copy()
        if frame.empty:
            return 0
        rows = [(sku, pd.Timestamp(r["sim_date"]), r.get("ordered", 0), r.get("sold", 0))
                for sku, series in daily.items() for r in series]
        real = pd.DataFrame(rows, columns=["product_id", "date", "ordered", "sold"])
        real["date"] = pd.to_datetime(real["date"])
        scale = 1.0 / max(share, 1e-6)
        key = frame[["product_id", "date"]].copy()
        key["date"] = pd.to_datetime(key["date"])
        merged = key.merge(real, on=["product_id", "date"], how="left")
        if len(merged) != len(frame):
            raise ValueError("realised overlay changed the row count")
        seen = merged["ordered"].notna().to_numpy()
        truth, obs = frame.copy(), frame.copy()
        truth.loc[seen, "demand"] = merged.loc[seen, "ordered"].to_numpy(float) * scale
        obs.loc[seen, "demand"] = merged.loc[seen, "sold"].to_numpy(float) * scale
        with self._swap_lock:
            self._realised_truth, self._realised_obs = truth, obs
            self._realised_sku_set = set(skus)
            # The retrain controller reads this key (drift_pipeline/retrain.py
            # `_recent_tft`): it fine-tunes on the observed, censored sales of
            # the SKUs this node trades.
            self._data["tft_full_censored"] = obs
        self.realised_days = int(real["date"].nunique())
        return self.realised_days

    def realised_forecast_error(self, *, sim_date: str) -> Optional[float]:
        """MASE of the forecast this node would have issued `prediction_length`
        days ago, against realised demand since - for the SKUs it trades.

        One full-encoder forecast per SKU (predict_mode), anchored so its last
        decoder day is `sim_date`. Unlike `realised_mase`, which pools every
        window across a ~74-day span of all 100 dataset series (slow, and a
        10-day shock on 10 of them barely moves it), this reacts within days
        and costs ~0.06 s. It is NOT the metric the offline threshold was
        calibrated on, so the monitor calibrates its own (models.fc_signal).
        """
        from drift_pipeline.core_pipeline import CONFIG as DRIFT_CONFIG  # noqa: PLC0415
        from drift_pipeline.trainers import (  # noqa: PLC0415
            evaluate_forecasting, filter_tft_eval_frame, make_tft_dataset,
        )
        truth = self._realised_truth
        if truth is None:
            return None
        fc = DRIFT_CONFIG["forecasting"]
        end = pd.Timestamp(sim_date)
        lo = end - pd.Timedelta(days=fc["prediction_length"] + fc["encoder_length"] - 1)
        window = filter_tft_eval_frame(truth[(truth["date"] >= lo) & (truth["date"] <= end)])
        if window.empty:
            return None
        ds = make_tft_dataset(window, train=False, training_dataset=self._train_ds,
                              predict_mode=True)
        loader = ds.to_dataloader(train=False, shuffle=False, batch_size=64, num_workers=0)
        base = self.data["tft_base"]
        scale_df = base[base["product_id"].isin(self._realised_sku_set)]
        m = evaluate_forecasting(self.forecaster(), loader, scale_df, task_id=0).get("mase")
        return float(m) if m is not None and np.isfinite(m) else None

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
        from drift_pipeline.trainers import DynamicPricingEnv  # noqa: PLC0415

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
