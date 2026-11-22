"""
A manifest of the gitignored artifacts the FYP2 results depend on.

Why this exists: when the old laptop died, the data and checkpoints went with
it. Nothing recorded which data a checkpoint had been trained on, so proving
that the rebuilt forecaster differed from the original took a forensic
exercise (outputs/drift/REBUILD_SUMMARY.txt) instead of a single hash
comparison. artifacts.json is committed. The files it describes are not.

    python -m experiments.artifacts            # (re)write artifacts.json from disk
    python -m experiments.artifacts --verify   # check every file on disk against it

Run --verify before an experiment. On a fresh machine, if it fails, the data or
checkpoints are not the ones the committed results came from. Re-fetch them, or
rebuild them with dataset_generator/m5/README.md and experiments/rebuild_base_v4.py.

Each group records the groups it was built from ("built_from"), so the chain
raw -> v2 -> v3 -> v4 -> base_v4 can be checked by hash. Compare the
checkpoint's recorded data with what is on disk.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "artifacts.json"

# name -> (directory, built_from, how to rebuild)
GROUPS: Dict[str, tuple] = {
    # Everything at the canonical paths is the OLD LAPTOP's, recovered on
    # 2026-11-22 with experiments/laptop_transfer.py. These are the files the
    # FYP1 results and the deployed system use. The November rebuild is kept
    # under data/_rebuilt/ and checkpoints/rebuilt/ and is not used.
    "m5_raw": ("data/m5_raw", [],
               "m5-forecasting-accuracy.zip from github.com/KunalArora/kaggle-m5-forecasting "
               "(zip md5 4123b9f46f7c24ad237fe3b89dfc821b); see dataset_generator/m5/README.md"),
    "external_pricing": ("data/external_pricing", [],
                         "old laptop (dunnhumby download + the fitted prior.json)"),
    "processed_fyp1": ("data/processed", [], "old laptop (FYP1 synthetic data)"),
    "processed_m5_v1": ("data/processed_m5", ["m5_raw"], "old laptop; not rebuildable"),
    "processed_m5_v2": ("data/processed_m5_v2", ["m5_raw", "external_pricing"], "old laptop"),
    "processed_m5_v3": ("data/processed_m5_v3", ["processed_m5_v2"], "old laptop"),
    "processed_m5_v4": ("data/processed_m5_v4", ["processed_m5_v3"], "old laptop"),
    "ckpt_base": ("outputs/drift/checkpoints/base", ["processed_m5_v4"], "old laptop; not rebuildable"),
    "ckpt_base_v4": ("outputs/drift/checkpoints/base_v4", ["processed_m5_v4"],
                     "old laptop; not rebuildable. The deployed system loads this one"),
    "ckpt_base_cover": ("outputs/drift/checkpoints/base_cover", ["processed_m5_v4"], "old laptop"),
    "ckpt_base_invconstr": ("outputs/drift/checkpoints/base_invconstr", ["processed_m5_v4"], "old laptop"),
    "ckpt_base_realinv": ("outputs/drift/checkpoints/base_realinv", ["processed_m5_v4"], "old laptop"),
    # The November 2026 rebuild, kept for reference (results/rebuild_2026-11/).
    "rebuilt_data": ("data/_rebuilt", ["m5_raw"],
                     "dataset_generator/m5/README.md, rebuilt 2026-11 while the laptop was dead"),
    "rebuilt_ckpt": ("outputs/drift/checkpoints/rebuilt", ["rebuilt_data"], "experiments.rebuild_base_v4"),
}


def sha256(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def scan(directory: Path) -> List[Dict]:
    files = sorted(p for p in directory.rglob("*") if p.is_file())
    return [{"path": p.relative_to(ROOT).as_posix(), "bytes": p.stat().st_size,
             "sha256": sha256(p)} for p in files]


def git_head() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True,
                              text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def write() -> int:
    out = {"written": time.strftime("%Y-%m-%d %H:%M:%S"), "git_head_when_written": git_head(),
           "groups": {}}
    for name, (rel, built_from, how) in GROUPS.items():
        d = ROOT / rel
        if not d.exists():
            print(f"  MISSING  {name:16s} {rel} (left out of the manifest)")
            continue
        files = scan(d)
        out["groups"][name] = {"dir": rel, "built_from": built_from, "rebuild": how,
                               "files": files}
        mb = sum(f["bytes"] for f in files) / 1e6
        print(f"  {name:16s} {len(files):3d} files  {mb:8.1f} MB")
    MANIFEST.write_text(json.dumps(out, indent=2) + "\n")
    print(f"wrote {MANIFEST.relative_to(ROOT)}")
    return 0


def verify() -> int:
    if not MANIFEST.exists():
        print("no artifacts.json; run `python -m experiments.artifacts` first")
        return 2
    man = json.loads(MANIFEST.read_text())
    bad = 0
    for name, g in man["groups"].items():
        problems = []
        recorded = {f["path"]: f for f in g["files"]}
        for path, f in recorded.items():
            p = ROOT / path
            if not p.exists():
                problems.append(f"missing {path}")
            elif p.stat().st_size != f["bytes"] or sha256(p) != f["sha256"]:
                problems.append(f"changed {path}")
        d = ROOT / g["dir"]
        if d.exists():
            extra = [p.relative_to(ROOT).as_posix() for p in d.rglob("*")
                     if p.is_file() and p.relative_to(ROOT).as_posix() not in recorded]
            problems += [f"not in manifest {e}" for e in extra]
        status = "OK" if not problems else "FAIL"
        bad += bool(problems)
        print(f"  {status:4s} {name:16s} {g['dir']}")
        for pr in problems[:8]:
            print(f"         {pr}")
        if len(problems) > 8:
            print(f"         ... and {len(problems) - 8} more")
    print("all artifacts match the manifest" if not bad else
          f"{bad} group(s) differ from the manifest the committed results were produced with")
    return 1 if bad else 0


def main() -> None:
    ap = argparse.ArgumentParser(description="Write or verify artifacts.json")
    ap.add_argument("--verify", action="store_true")
    args = ap.parse_args()
    sys.exit(verify() if args.verify else write())


if __name__ == "__main__":
    main()
