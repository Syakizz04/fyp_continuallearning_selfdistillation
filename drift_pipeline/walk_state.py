"""
Mid-walk checkpointing, so an interrupted cell resumes where it stopped.

The sweep already resumes at CELL granularity — a cell whose `probe_scores_*`
exist is skipped. That is the right unit when cells are cheap. They are not: a
retraining arm walks 157 weekly checks, retraining the forecaster ~16 times and
the pricer ~35 times along the way, and losing one at check 150 costs the same
as losing it at check 2. This module adds the missing granularity.

## What has to be saved, and why all of it

A walk's position is not just "which check". Resuming needs everything the next
check reads:

* **the two models** — the whole point; they are what retraining mutated;
* **the CL state** — the replay buffer, EWC Fisher/anchors, the RL teacher, and
  the `_seeded` flags that say whether each has been initialised from the base
  regime. Dropping these would silently convert a resumed `replay` arm into
  something closer to `naive` for the rest of its walk, which is precisely the
  comparison E2 exists to make;
* **the detectors' streaks** — `DebouncedDetector` fires only after N
  consecutive breaches, so a detector reset mid-streak under-triggers;
* **the accumulated stream and trigger list** — the results themselves;
* **`ArmStats` and the `MemoryLog` rows** — retrain counts and E4's per-event
  memory record, both of which are reported outputs rather than scratch state;
* **the RNG streams** — the replay buffers use reservoir sampling and PPO draws
  actions stochastically while training.

## What this does NOT promise

Resume is *statistically* equivalent, not bit-identical. cuDNN kernel selection
and dataloader worker seeding are not fully captured by the RNG states above, so
a resumed cell can differ from an uninterrupted one in the last decimal places.
That is a far smaller perturbation than the alternative (starting over, or
resuming with an empty replay buffer), but it is worth stating rather than
implying exactness this cannot deliver.

## Cost

The forecaster is ~4 MB and the pricer smaller, but a saturated replay buffer
reaches ~3.1 GB, and it is the bulk of every save. Checkpointing is therefore
periodic rather than per-check — see `--checkpoint-every`. Writes go to a
sibling `.tmp` directory and are swapped in by rename, so a crash *during* a
save leaves the previous checkpoint intact rather than a half-written one.

Note for the rent-stop-resume workflow: this state lives on the box's disk. It
survives a vast.ai **Stop** (the volume persists) but not a **Destroy**. To move
between instances, copy `walk_state_<arm>/` off with the results.
"""

from __future__ import annotations

import json
import random
import shutil
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd
import torch

from .core_pipeline import CONFIG, DEVICE
from .trainers import build_cltft

#: Bumped when the on-disk layout changes in a way that makes older state
#: unreadable. A mismatch is ignored rather than crashed on — a stale checkpoint
#: should cost a restart, not the whole run.
STATE_VERSION = 1


def state_dir(arm: str) -> Path:
    return Path(CONFIG["paths"]["results"]) / f"walk_state_{arm}"


def _files(d: Path) -> Dict[str, Path]:
    return {"meta": d / "meta.json", "stream": d / "stream.json",
            "forecaster": d / "forecaster.pt", "pricer": d / "pricer.zip",
            "cl": d / "cl_state.pt"}


def _jsonable(o):
    if isinstance(o, np.generic):
        return o.item()
    return str(o)


def exists(arm: str) -> bool:
    return _files(state_dir(arm))["meta"].exists()


def clear(arm: str) -> None:
    """Drop the checkpoint. Called once a cell finishes — the walk is over, and
    a multi-GB buffer snapshot for a completed cell is pure waste."""
    for d in (state_dir(arm), Path(str(state_dir(arm)) + ".tmp")):
        if d.exists():
            shutil.rmtree(d, ignore_errors=True)


def save(ctrl, res, fc_det, rl_det, next_idx: int) -> Path:
    """Snapshot a cell mid-walk. `next_idx` is the check to resume AT."""
    final = state_dir(ctrl.strategy)
    tmp = Path(str(final) + ".tmp")
    if tmp.exists():
        shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True, exist_ok=True)
    f = _files(tmp)

    fc = ctrl.forecaster
    has_teacher = getattr(fc, "teacher", None) is not None
    torch.save({"state_dict": fc.state_dict()}, f["forecaster"])
    ctrl.pricer.save(str(f["pricer"]))

    # The buffers hold CPU tensors / numpy arrays / plain tuples, and the RL
    # teacher an nn.Module, so torch.save pickles them directly. Reloading
    # therefore requires the same codebase — fine for resuming a run, which is
    # the only thing this is for.
    torch.save({
        "fc_replay": ctrl.fc_replay, "rl_buf": ctrl.rl_buf,
        "rl_ewc": ctrl.rl_ewc, "rl_teacher": ctrl.rl_teacher,
        "rng": {
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            "numpy": np.random.get_state(),
            "python": random.getstate(),
        },
    }, f["cl"])

    f["stream"].write_text(json.dumps(
        {"stream": res.stream, "triggers": res.triggers}, default=_jsonable))

    f["meta"].write_text(json.dumps({
        "version": STATE_VERSION,
        "strategy": ctrl.strategy,
        "next_idx": int(next_idx),
        "n_fc_triggers": int(res.n_fc_triggers),
        "n_rl_triggers": int(res.n_rl_triggers),
        "fc_det_run": int(fc_det.run),
        "rl_det_run": int(rl_det.run),
        "has_fc_teacher": bool(has_teacher),
        "fc_cl_method": getattr(fc, "cl_method", "naive"),
        "stats": {
            "n_fc_retrains": ctrl.stats.n_fc_retrains,
            "n_rl_retrains": ctrl.stats.n_rl_retrains,
            "fc_epochs_total": ctrl.stats.fc_epochs_total,
            "rl_steps_total": ctrl.stats.rl_steps_total,
            "retrain_log": ctrl.stats.retrain_log,
        },
        "flags": {
            "_fc_ewc_seeded": ctrl._fc_ewc_seeded,
            "_fc_replay_seeded": ctrl._fc_replay_seeded,
            "_rl_ewc_seeded": ctrl._rl_ewc_seeded,
            "_rl_buf_seeded": ctrl._rl_buf_seeded,
            "_fc_idx": ctrl._fc_idx, "_rl_idx": ctrl._rl_idx,
            "_last_periodic": (str(ctrl._last_periodic)
                               if ctrl._last_periodic is not None else None),
        },
        "memlog": {"rows": ctrl.memlog.rows, "event_idx": ctrl.memlog._event_idx},
    }, indent=2, default=_jsonable))

    # Swap by rename so an interrupted save cannot destroy the last good state.
    old = Path(str(final) + ".old")
    if old.exists():
        shutil.rmtree(old, ignore_errors=True)
    if final.exists():
        final.rename(old)
    tmp.rename(final)
    shutil.rmtree(old, ignore_errors=True)
    return final


def load(arm: str) -> Optional[Dict]:
    """Read a checkpoint's metadata, or None if there isn't a usable one."""
    d = state_dir(arm)
    f = _files(d)
    if not f["meta"].exists():
        return None
    try:
        meta = json.loads(f["meta"].read_text())
        payload = json.loads(f["stream"].read_text())
    except (json.JSONDecodeError, OSError):
        return None
    if meta.get("version") != STATE_VERSION or meta.get("strategy") != arm:
        return None
    return {"meta": meta, "stream": payload["stream"],
            "triggers": payload["triggers"], "dir": d}


def restore(ctrl, state: Dict) -> Dict:
    """Rehydrate `ctrl` in place. Returns the resume payload `walk_forward` needs.

    The forecaster is restored by state_dict into the freshly-built base model
    rather than by unpickling a module, so the architecture still comes from
    `build_cltft` exactly as `load_base` produced it. An SDFT teacher has to be
    ATTACHED before loading, because it contributes `teacher.*` keys that a
    teacher-less model has no slots for.
    """
    from stable_baselines3 import PPO           # noqa: PLC0415  (heavy import)

    meta, f = state["meta"], _files(state["dir"])

    if meta.get("has_fc_teacher"):
        teacher = build_cltft(ctrl.train_ds, cl_method="naive")
        teacher.teacher = None
        ctrl.forecaster.teacher = teacher
    blob = torch.load(f["forecaster"], map_location=DEVICE)
    ctrl.forecaster.load_state_dict(blob["state_dict"])
    ctrl.forecaster.to(DEVICE)
    ctrl.forecaster.cl_method = meta.get("fc_cl_method", "naive")
    if meta.get("has_fc_teacher"):
        ctrl.forecaster.teacher.to(DEVICE).eval()
        for p in ctrl.forecaster.teacher.parameters():
            p.requires_grad_(False)

    ctrl.pricer = PPO.load(str(f["pricer"]), device=DEVICE)

    # weights_only=False is explicit: these are container objects, not tensors,
    # and torch >= 2.6 defaults the flag to True.
    cl = torch.load(f["cl"], map_location=DEVICE, weights_only=False)
    ctrl.fc_replay = cl["fc_replay"]
    ctrl.rl_buf = cl["rl_buf"]
    ctrl.rl_ewc = cl["rl_ewc"]
    ctrl.rl_teacher = cl["rl_teacher"]

    rng = cl.get("rng") or {}
    if rng.get("torch") is not None:
        torch.set_rng_state(rng["torch"].cpu().to(torch.uint8))
    if rng.get("cuda") is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([s.cpu().to(torch.uint8) for s in rng["cuda"]])
    if rng.get("numpy") is not None:
        np.random.set_state(tuple(rng["numpy"]))
    if rng.get("python") is not None:
        random.setstate(tuple((x if not isinstance(x, list) else tuple(x))
                              for x in rng["python"]))

    st = meta["stats"]
    ctrl.stats.n_fc_retrains = st["n_fc_retrains"]
    ctrl.stats.n_rl_retrains = st["n_rl_retrains"]
    ctrl.stats.fc_epochs_total = st["fc_epochs_total"]
    ctrl.stats.rl_steps_total = st["rl_steps_total"]
    ctrl.stats.retrain_log = list(st["retrain_log"])

    fl = meta["flags"]
    ctrl._fc_ewc_seeded = fl["_fc_ewc_seeded"]
    ctrl._fc_replay_seeded = fl["_fc_replay_seeded"]
    ctrl._rl_ewc_seeded = fl["_rl_ewc_seeded"]
    ctrl._rl_buf_seeded = fl["_rl_buf_seeded"]
    ctrl._fc_idx = fl["_fc_idx"]
    ctrl._rl_idx = fl["_rl_idx"]
    ctrl._last_periodic = (pd.Timestamp(fl["_last_periodic"])
                           if fl["_last_periodic"] else None)

    ctrl.memlog.rows = list(meta["memlog"]["rows"])
    ctrl.memlog._event_idx = int(meta["memlog"]["event_idx"])

    return {"next_idx": int(meta["next_idx"]),
            "stream": state["stream"], "triggers": state["triggers"],
            "n_fc_triggers": meta["n_fc_triggers"],
            "n_rl_triggers": meta["n_rl_triggers"],
            "fc_det_run": meta["fc_det_run"], "rl_det_run": meta["rl_det_run"]}
