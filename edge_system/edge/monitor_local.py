"""
Streaming drift detection for one edge node.

The offline pipeline evaluates drift by walking a whole timeline in one call
(`drift_pipeline.monitor.walk_forward`). A deployed node cannot do that - it sees
one tick at a time and has to decide, now, whether to retrain.

This wraps the **same** `DebouncedDetector` and the **same** calibrated
thresholds so a node fires on exactly the criterion the experiment used. Sharing
the detector matters: a node that drifted from the offline definition would make
the deployed results incomparable with the FYP1 results they are meant to extend.
"""

from __future__ import annotations

import time
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
    """

    def __init__(self, node: str, calibration: Dict,
                 *, fc_k: Optional[float] = None,
                 rl_floor: Optional[float] = None,
                 consecutive: Optional[int] = None) -> None:
        from drift_pipeline.core_pipeline import CONFIG as DRIFT_CONFIG
        from drift_pipeline.monitor import DebouncedDetector

        drift = DRIFT_CONFIG["drift"]
        self.node = node

        mu = calibration["forecasting"]["mase_mu"]
        sigma = calibration["forecasting"]["mase_sigma"]
        k = fc_k if fc_k is not None else drift["fc_k_sigma"]
        self.fc_threshold = mu + k * sigma
        self.rl_threshold = (rl_floor if rl_floor is not None
                             else drift["rl_profit_floor"])
        # Forecasting and RL debounce independently in the offline config, so
        # they do here too - collapsing them to one value would change the
        # trigger criterion and break comparability with the FYP1 runs.
        fc_n = consecutive if consecutive is not None else drift["fc_consecutive"]
        rl_n = consecutive if consecutive is not None else drift["rl_consecutive"]

        self._fc = DebouncedDetector(self.fc_threshold, fc_n, "above")
        self._rl = DebouncedDetector(self.rl_threshold, rl_n, "below")

        self.stream: List[Dict] = []
        self.events: List[DriftEvent] = []
        self.calibration = calibration

    def update(self, *, tick: int, sim_date: str,
               mase: Optional[float] = None,
               profit_index: Optional[float] = None) -> List[DriftEvent]:
        """Feed one check's metrics. Returns any triggers that fired."""
        self.stream.append({
            "tick": tick, "date": sim_date,
            "mase": mase, "profit_index": profit_index,
        })

        fired: List[DriftEvent] = []
        if mase is not None and self._fc.update(mase):
            fired.append(DriftEvent(tick, sim_date, "forecasting", mase,
                                    self.fc_threshold, self.node))
        if profit_index is not None and self._rl.update(profit_index):
            fired.append(DriftEvent(tick, sim_date, "rl", profit_index,
                                    self.rl_threshold, self.node))

        self.events.extend(fired)
        return fired

    # ── Views ───────────────────────────────────────────────────────────────

    def status(self) -> Dict:
        last = self.stream[-1] if self.stream else {}
        return {
            "node": self.node,
            "checks": len(self.stream),
            "fc_threshold": self.fc_threshold,
            "rl_threshold": self.rl_threshold,
            "fc_streak": self._fc.run,
            "rl_streak": self._rl.run,
            "last_mase": last.get("mase"),
            "last_profit_index": last.get("profit_index"),
            "n_events": len(self.events),
        }

    def recent_events(self, limit: int = 20) -> List[Dict]:
        return [e.as_dict() for e in self.events[-limit:]]
