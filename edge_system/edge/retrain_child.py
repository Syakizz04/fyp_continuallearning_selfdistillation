"""
One retrain, in a process of its own (`models.retrain_process: "subprocess"`).

Why: a retrain inside the long-lived node process leaves ~1.2 GB committed
(~0.8 GB resident) that never goes back to the OS - Lightning's fit, the
allocators' high-water marks - so three nodes that have each retrained once
hold ~2.3 GB more than at startup, which is what tipped the demo laptop into
low memory. A child process returns all of it when it exits.

What runs here is the node's own code, not a copy: `LocalInference.load()` -
the same data, base checkpoints and sdft_alpha a node starts with - then the
served weights the parent wrote, the realised-sales frame the parent built,
and `RetrainController.retrain_forecaster` / `retrain_pricer` exactly as the
in-process path calls them. The parent loads the result and hot-swaps it.

Only strategies with no CL state beyond the model itself can move here
(`CHILD_STRATEGIES`): SDFT's teacher is the current model, which the parent
sends. Replay and EWC keep buffers / Fisher terms inside the controller across
retrains, so they stay in-process.

Job folder (written by the parent, `retrain_local.LocalRetrainer`):
    job.json          kind, origin, reason, node
    forecaster.pt     served forecaster state_dict      (kind "forecasting")
    pricer.zip        served PPO, SB3 save format       (kind "rl")
    realised.pkl      the node's realised-sales frame   (optional)
and on success the child adds `result.pt` / `result.zip` and `result.json`.

    python -m edge_system.edge.retrain_child <job folder>
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

#: Strategies whose whole CL state is the model the parent sends.
CHILD_STRATEGIES = {"sdft", "naive", "periodic"}


def strip_teacher(state: dict) -> dict:
    """A forecaster's weights without an attached SDFT teacher: the teacher is
    rebuilt from the student at the next retrain, so shipping it is waste."""
    return {k: v for k, v in state.items() if not k.startswith("teacher.")}


def run(job_dir: Path) -> dict:
    import pandas as pd
    import torch

    from drift_pipeline.retrain import RetrainController

    from .inference import LocalInference

    job = json.loads((job_dir / "job.json").read_text())
    started = time.time()

    inf = LocalInference(job["node"])
    inf.load()
    if (job_dir / "realised.pkl").exists():
        inf._data["tft_full_censored"] = pd.read_pickle(job_dir / "realised.pkl")

    ctrl = RetrainController(job["strategy"], inf.base_bundle, inf.data)
    if job["kind"] == "forecasting":
        state = torch.load(job_dir / "forecaster.pt", map_location="cpu", weights_only=True)
        ctrl.forecaster.load_state_dict(strip_teacher(state))
        ctrl.retrain_forecaster(job["origin"], job["reason"])
        torch.save(strip_teacher(ctrl.forecaster.state_dict()), job_dir / "result.pt")
    else:
        from stable_baselines3 import PPO
        ctrl.pricer = PPO.load(job_dir / "pricer.zip", device=ctrl.pricer.device)
        ctrl.retrain_pricer(job["origin"], job["reason"])
        ctrl.pricer.save(job_dir / "result.zip")

    out = {"ok": True, "seconds": time.time() - started}
    (job_dir / "result.json").write_text(json.dumps(out))
    return out


if __name__ == "__main__":
    run(Path(sys.argv[1]))
