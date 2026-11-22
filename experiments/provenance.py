"""
Stamp an experiment output directory with what produced it.

    from experiments.provenance import write_run_meta
    write_run_meta(cell_dir, data_dir=..., ckpt_dir=..., extra={...})

writes `run_meta.json` with:
  * git commit, branch, and whether the tree had uncommitted changes;
  * Python and key package versions;
  * the resolved drift CONFIG at that moment, after every in-place mutation
    (CONFIG is mutated, never subclassed, so only a runtime dump shows what
    actually ran);
  * sha256 of every data and checkpoint file used, and whether they match the
    committed artifacts.json;
  * the command line, and anything the caller adds (seed, arm, censoring).

Why: the August E2 tables could not have been tied to a model even if they
had survived. Nothing recorded which data or checkpoint produced them, and
"which model was this?" is the question the November rebuild had to answer
by forensics. No hostname or user name is recorded.
"""
from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
import time
from importlib import metadata
from pathlib import Path
from typing import Dict, Optional

ROOT = Path(__file__).resolve().parents[1]
PACKAGES = ("torch", "lightning", "pytorch-forecasting", "stable-baselines3", "numpy",
            "pandas", "scikit-learn", "gymnasium", "darts")


def _git(*args: str) -> str:
    try:
        return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True,
                              check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _sha256(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _hash_dir(d: Optional[Path]) -> Dict[str, str]:
    if d is None or not Path(d).exists():
        return {}
    out = {}
    for p in sorted(Path(d).rglob("*")):
        if p.is_file():
            try:
                rel = p.resolve().relative_to(ROOT).as_posix()
            except ValueError:
                rel = p.as_posix()
            out[rel] = _sha256(p)
    return out


def _manifest_check(hashes: Dict[str, str]) -> Optional[bool]:
    """True if every hashed file is in artifacts.json with the same sha256.
    None if there is no manifest, or none of these files are in it."""
    man = ROOT / "artifacts.json"
    if not man.exists():
        return None
    recorded = {f["path"]: f["sha256"]
                for g in json.loads(man.read_text())["groups"].values() for f in g["files"]}
    known = [p for p in hashes if p in recorded]
    if not known:
        return None
    return all(recorded[p] == hashes[p] for p in known) and len(known) == len(hashes)


def _versions() -> Dict[str, str]:
    out = {"python": platform.python_version()}
    for name in PACKAGES:
        try:
            out[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            out[name] = "not installed"
    try:
        import torch  # noqa: PLC0415
        out["cuda"] = str(torch.version.cuda)
        out["cuda_available"] = str(torch.cuda.is_available())
    except Exception:  # noqa: BLE001 - provenance must never break a run
        pass
    return out


def write_run_meta(out_dir: Path, *, data_dir: Optional[Path] = None,
                   ckpt_dir: Optional[Path] = None, extra: Optional[Dict] = None,
                   config: Optional[Dict] = None) -> Path:
    """Write out_dir/run_meta.json and return its path. Never raises on a
    missing optional piece: a provenance failure must not cost a GPU run."""
    from drift_pipeline.core_pipeline import CONFIG  # noqa: PLC0415

    data_h, ckpt_h = _hash_dir(data_dir), _hash_dir(ckpt_dir)
    meta = {
        "written": time.strftime("%Y-%m-%d %H:%M:%S"),
        "git": {"commit": _git("rev-parse", "HEAD"),
                "branch": _git("rev-parse", "--abbrev-ref", "HEAD"),
                "dirty": bool(_git("status", "--porcelain", "--untracked-files=no"))},
        "argv": sys.argv,
        "versions": _versions(),
        "data": {"dir": str(data_dir) if data_dir else None, "sha256": data_h,
                 "matches_artifacts_json": _manifest_check(data_h)},
        "checkpoint": {"dir": str(ckpt_dir) if ckpt_dir else None, "sha256": ckpt_h,
                       "matches_artifacts_json": _manifest_check(ckpt_h)},
        "config": config if config is not None else CONFIG,
        **(extra or {}),
    }
    out = Path(out_dir) / "run_meta.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(meta, indent=2, default=str) + "\n")
    return out
