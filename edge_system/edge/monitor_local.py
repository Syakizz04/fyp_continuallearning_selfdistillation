"""
Streaming drift detection for one edge node.

The offline pipeline evaluates drift by walking a whole timeline in one call
(`drift_pipeline.monitor.walk_forward`). A deployed node cannot do that - it sees
one tick at a time and has to decide, now, whether to retrain.

This wraps the **same** `DebouncedDetector` and the **same** calibrated
thresholds so a node fires on exactly the criterion the experiment used. Sharing
the detector matters: a node that drifted from the offline definition would make
the deployed results incomparable with the FYP1 results they are meant to extend.

The opt-in exceptions are the "rolling" RL detector (`models.rl_detector`) and
the "realised" forecaster signal (`models.fc_signal`), which the viva demo uses
and no experiment does.
"""

from __future__ import annotations

import math
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class DriftEvent:
    tick: int
    sim_date: str
    kind: str                # "forecasting" | "rl"
    value: float
    threshold: float
    node: str
    ts: float = field(default_factory=time.time)

    def as_dict(self) -> Dict:
        return {
            "tick": self.tick, "sim_date": self.sim_date, "kind": self.kind,
            "value": self.value, "threshold": self.threshold,
            "node": self.node, "ts": self.ts,
        }


class LocalDriftMonitor:
    """
    Per-node debounced drift detection over a live metric stream.

    Two independent detectors, matching the offline arms:

      forecasting - fires when MASE runs ABOVE mu + k*sigma
      rl          - fires when profit_index runs BELOW a floor

    Both are debounced: a single bad window is noise, `consecutive` bad windows
    in a row is drift. That debouncing is why the offline runs retrained once or
    twice rather than constantly.

    The RL floor has two forms (SYSTEM_CONFIG["models"]["rl_detector"]):
    "fixed" is the offline config's flat floor on each day's index; "rolling"
    sums revenue and reference over a window before dividing, and calibrates
    its own floor on the node's first weeks of trading.
    """

    def __init__(self, node: str, calibration: Dict,
                 *, fc_k: Optional[float] = None,
                 rl_floor: Optional[float] = None,
                 consecutive: Optional[int] = None) -> None:
        from drift_pipeline.core_pipeline import CONFIG as DRIFT_CONFIG
        from drift_pipeline.monitor import DebouncedDetector
        from edge_system.config import SYSTEM_CONFIG

        drift = DRIFT_CONFIG["drift"]
        models = SYSTEM_CONFIG["models"]
        self.node = node

        mu = calibration["forecasting"]["mase_mu"]
        sigma = calibration["forecasting"]["mase_sigma"]
        k = fc_k if fc_k is not None else drift["fc_k_sigma"]
        # "dataset": the offline metric and its calibrated threshold.
        # "realised": the node's own realised-demand error (inference.py
        # `realised_forecast_error`), a different metric, so the threshold is
        # calibrated here on the node's first checks instead.
        self.fc_mode = "dataset" if fc_k is not None else models.get("fc_signal", "dataset")
        if self.fc_mode not in ("dataset", "realised"):
            raise ValueError(f"unknown fc_signal {self.fc_mode!r}")
        self.fc_threshold: Optional[float] = (mu + k * sigma if self.fc_mode == "dataset"
                                              else None)

        # An explicit floor is a caller pinning the old behaviour.
        self.rl_mode = ("fixed" if rl_floor is not None
                        else models.get("rl_detector", "fixed"))
        if self.rl_mode not in ("fixed", "rolling"):
            raise ValueError(f"unknown rl_detector {self.rl_mode!r}")
        self.rl_threshold: Optional[float] = (
            rl_floor if rl_floor is not None
            else drift["rl_profit_floor"] if self.rl_mode == "fixed"
            else None)                       # rolling: set once warmed up

        # Forecasting and RL debounce independently in the offline config, so
        # they do here too - collapsing them to one value would change the
        # trigger criterion and break comparability with the FYP1 runs.
        fc_n = consecutive if consecutive is not None else drift["fc_consecutive"]
        rl_n = consecutive if consecutive is not None else drift["rl_consecutive"]

        self._Detector = DebouncedDetector
        if self.fc_mode == "realised":
            fc_n = consecutive if consecutive is not None else int(models.get("fc_live_consecutive", 3))
        self._fc_n = fc_n
        self._fc = (DebouncedDetector(self.fc_threshold, fc_n, "above")
                    if self.fc_threshold is not None else None)
        self._fc_warmup = max(int(models.get("fc_warmup_days", 28)), 2)
        self._fc_k = float(models.get("fc_live_k", 3.0))
        self._fc_calib: List[float] = []
        self.fc_calibration: Optional[Dict] = None
        self.fc_provisional: Optional[Dict] = None
        self._rl_n = rl_n
        self._rl = (DebouncedDetector(self.rl_threshold, rl_n, "below")
                    if self.rl_threshold is not None else None)

        # Rolling mode: the last W days of (revenue, reference), and the days
        # the floor is calibrated on. Calibration needs at least a full window.
        self._rl_window = max(int(models.get("rl_window_days", 7)), 1)
        self._rl_warmup = max(int(models.get("rl_warmup_days", 28)),
                              self._rl_window, 2)
        self._rl_k = float(models.get("rl_k_sigma", 2.5))
        # Re-learning after a swap (config `relearn_min_days`): the spread
        # each detector last calibrated, kept so a provisional line can go up
        # before a new warm-up completes. None until a first calibration.
        self._relearn_min = max(int(models.get("relearn_min_days", 0)), 0)
        self._rl_sigma_prev: Optional[float] = None
        self._fc_sigma_prev: Optional[float] = None
        self._days: deque = deque(maxlen=self._rl_window)
        self._calib: List[tuple] = []
        self.rl_calibration: Optional[Dict] = None
        self.rl_provisional: Optional[Dict] = None

        self.stream: List[Dict] = []
        self.events: List[DriftEvent] = []
        self.calibration = calibration

    def update(self, *, tick: int, sim_date: str,
               mase: Optional[float] = None,
               profit_index: Optional[float] = None,
               revenue: Optional[float] = None,
               reference: Optional[float] = None) -> List[DriftEvent]:
        """Feed one check's metrics. Returns any triggers that fired."""
        daily = profit_index
        if self.rl_mode == "rolling" and profit_index is not None:
            profit_index = self._roll(profit_index, revenue, reference)
        row = {
            "tick": tick, "date": sim_date,
            "mase": mase, "profit_index": profit_index,
            "profit_index_daily": daily,
        }
        self.stream.append(row)
        # The floor a value is judged against moves: calibrated after a
        # warm-up, cleared and re-learned after each retrain. Record the one in
        # force for this check (read before the detectors run, which is what
        # they compare against), so a chart can draw the line each warning
        # actually crossed instead of today's line over the whole history.
        row["fc_threshold"] = self.fc_threshold if self._fc is not None else None
        row["rl_threshold"] = self.rl_threshold if self._rl is not None else None

        fired: List[DriftEvent] = []
        # Judge first, against the line recorded on this row; then let the
        # check teach a warm-up. Without a provisional line `_fc` is None for
        # the whole warm-up, so this is the original order of events.
        if mase is not None and self._fc is not None and self._fc.update(mase):
            fired.append(DriftEvent(tick, sim_date, "forecasting", mase,
                                    self.fc_threshold, self.node))
        if (mase is not None and self.fc_mode == "realised" and self.fc_calibration is None
                and math.isfinite(mase)):
            self._fc_calib.append(mase)
            if len(self._fc_calib) >= self._fc_warmup:
                self._calibrate_fc()
            else:
                self._provisional_fc()
        if (profit_index is not None and self._rl is not None
                and self._rl.update(profit_index)):
            fired.append(DriftEvent(tick, sim_date, "rl", profit_index,
                                    self.rl_threshold, self.node))

        self.events.extend(fired)
        return fired

    # ── Rolling RL floor ────────────────────────────────────────────────────

    def _roll(self, index: float, revenue: Optional[float],
              reference: Optional[float]) -> Optional[float]:
        """Add one day; return the window's ratio of sums, calibrating the floor
        once the warm-up is complete. A caller that sends only the ratio counts
        as reference 1, which degrades to a mean of daily ratios."""
        if revenue is None or reference is None:
            revenue, reference = index, 1.0
        if not (math.isfinite(revenue) and math.isfinite(reference)) or reference <= 0:
            return None
        self._days.append((revenue, reference))
        if self.rl_calibration is None:
            self._calib.append((revenue, reference))
            if len(self._calib) >= self._rl_warmup:
                self._calibrate()
            else:
                self._provisional_rl()
        ref = sum(f for _, f in self._days)
        return sum(r for r, _ in self._days) / ref if ref > 0 else None

    def _calibrate(self) -> None:
        """mu = the warm-up's ratio of sums. sigma = the spread a W-day window
        would have if days were independent, from each day's residual against
        mu. The obvious alternative - the std of the warm-up's own rolling
        values - rests on a handful of overlapping windows, and in replay it
        let pos false-fire 40-100 times a year."""
        ref = [f for _, f in self._calib]
        mu = sum(r for r, _ in self._calib) / sum(ref)
        resid = [r - mu * f for r, f in self._calib]
        m = sum(resid) / len(resid)
        sd_day = math.sqrt(sum((e - m) ** 2 for e in resid) / (len(resid) - 1))
        sigma = sd_day / (math.sqrt(self._rl_window) * (sum(ref) / len(ref)))
        floor = mu - self._rl_k * sigma
        self.rl_calibration = {"mu": mu, "sigma": sigma, "k": self._rl_k,
                               "window_days": self._rl_window,
                               "warmup_days": len(self._calib)}
        self.rl_provisional = None
        self._set_rl_line(floor)

    def _set_rl_line(self, floor: float) -> None:
        """Move the floor, keeping a running breach streak across a refinement."""
        self.rl_threshold = floor
        if self._rl is None:
            self._rl = self._Detector(floor, self._rl_n, "below")
        else:
            self._rl.threshold = floor

    def _provisional_rl(self) -> None:
        """A floor from the new model's own days so far, on the spread the last
        calibration measured. Only after a swap (a previous spread exists), and
        only once a whole window is the new model's, so the rolling value no
        longer mixes in days the old model priced."""
        n = len(self._calib)
        if (self._rl_sigma_prev is None or self._relearn_min <= 0
                or n < max(self._relearn_min, self._rl_window)):
            return
        ref = sum(f for _, f in self._calib)
        if ref <= 0:
            return
        mu = sum(r for r, _ in self._calib) / ref
        self.rl_provisional = {"mu": mu, "sigma": self._rl_sigma_prev, "k": self._rl_k,
                               "days": n}
        self._set_rl_line(mu - self._rl_k * self._rl_sigma_prev)

    def _calibrate_fc(self) -> None:
        vals = self._fc_calib
        mu = sum(vals) / len(vals)
        sd = math.sqrt(sum((v - mu) ** 2 for v in vals) / (len(vals) - 1))
        self.fc_calibration = {"mu": mu, "sigma": sd, "k": self._fc_k,
                               "warmup_days": len(vals), "consecutive": self._fc_n}
        self.fc_provisional = None
        self._set_fc_line(mu + self._fc_k * sd)

    def _set_fc_line(self, threshold: float) -> None:
        self.fc_threshold = threshold
        if self._fc is None:
            self._fc = self._Detector(threshold, self._fc_n, "above")
        else:
            self._fc.threshold = threshold

    def _provisional_fc(self) -> None:
        """As `_provisional_rl`: the new forecaster's mean error so far plus the
        spread the last calibration measured. A replay found no in-sample
        flattery in a swap's first checks (lower in 4 of 7, higher in 3), so
        they count."""
        n = len(self._fc_calib)
        if self._fc_sigma_prev is None or self._relearn_min <= 0 or n < self._relearn_min:
            return
        mu = sum(self._fc_calib) / n
        self.fc_provisional = {"mu": mu, "sigma": self._fc_sigma_prev, "k": self._fc_k,
                               "days": n}
        self._set_fc_line(mu + self._fc_k * self._fc_sigma_prev)

    def recalibrate_fc(self) -> None:
        """Fresh warm-up after the forecaster is replaced - the new model's
        normal error level is its own, as with the pricer. Its spread is
        kept for the provisional line (see `relearn_min_days`)."""
        if self.fc_mode != "realised":
            return
        if self.fc_calibration is not None:
            self._fc_sigma_prev = self.fc_calibration["sigma"]
        self._fc_calib = []
        self.fc_calibration = None
        self.fc_provisional = None
        self.fc_threshold = None
        self._fc = None

    def recalibrate_rl(self) -> None:
        """Start a fresh warm-up. Called when the pricer is replaced: a new
        policy prices differently, so its profit settles at a new level, and
        judging it against the old policy's floor would retrain it for being
        different rather than for being worse."""
        if self.rl_mode != "rolling":
            return
        if self.rl_calibration is not None:
            self._rl_sigma_prev = self.rl_calibration["sigma"]
        self._calib = []
        self._days.clear()
        self.rl_calibration = None
        self.rl_provisional = None
        self.rl_threshold = None
        self._rl = None

    # ── Views ───────────────────────────────────────────────────────────────

    def status(self) -> Dict:
        warming = self.rl_mode == "rolling" and self.rl_calibration is None
        fc_warming = self.fc_mode == "realised" and self.fc_calibration is None
        return {
            "node": self.node,
            "checks": len(self.stream),
            "fc_threshold": self.fc_threshold,
            "rl_threshold": self.rl_threshold,
            "fc_streak": self._fc.run if self._fc is not None else 0,
            "fc_signal": self.fc_mode,
            "fc_calibration": self.fc_calibration,
            # A line up before the warm-up completed (after a swap): level so
            # far + the previous spread. None once fully calibrated.
            "fc_provisional": self.fc_provisional,
            "fc_warmup_left": (max(self._fc_warmup - len(self._fc_calib), 0)
                               if fc_warming else 0),
            "rl_streak": self._rl.run if self._rl is not None else 0,
            "rl_detector": self.rl_mode,
            "rl_calibration": self.rl_calibration,
            "rl_provisional": self.rl_provisional,
            "rl_warmup_left": (max(self._rl_warmup - len(self._calib), 0)
                               if warming else 0),
            # The forecaster and pricing checks append separate rows, so the
            # newest row usually carries only one of the two.
            "last_mase": next((r["mase"] for r in reversed(self.stream)
                               if r.get("mase") is not None), None),
            "last_profit_index": next((r["profit_index"] for r in reversed(self.stream)
                                       if r.get("profit_index") is not None), None),
            "n_events": len(self.events),
        }

    def recent_events(self, limit: int = 20) -> List[Dict]:
        return [e.as_dict() for e in self.events[-limit:]]
