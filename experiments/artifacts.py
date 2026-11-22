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
    "m5_raw": ("data/m5_raw", [],
               "m5-forecasting-accuracy.zip from github.com/KunalArora/kaggle-m5-forecasting "
               "(zip md5 4123b9f46f7c24ad237fe3b89dfc821b); see dataset_generator/m5/README.md"),
    "external_prior": ("data/external_pricing", [],
                       "prior.json, re-entered by hand: mu -0.524, tau 0.313"),
    "processed_m5_v2": ("data/processed_m5_v2", ["m5_raw", "external_prior"],
                        "dataset_generator.m5.build_m5 (flags in dataset_generator/m5/README.md)"),
    "processed_m5_v3": ("data/processed_m5_v3", ["processed_m5_v2"],
                        "dataset_generator.m5.rebuild_inventory"),
    "processed_m5_v4": ("data/processed_m5_v4", ["processed_m5_v3"],
                        "dataset_generator.m5.refresh_rl_features"),
    "ckpt_base": ("outputs/drift/checkpoints/base", ["processed_m5_v4"],
                  "experiments.rebuild_base_v4 (step 1)"),
    "ckpt_base_v4": ("outputs/drift/checkpoints/base_v4", ["processed_m5_v4"],
                     "experiments.rebuild_base_v4 (step 2: retrain_pricer)"),
    # The ORIGINAL models, recovered from the old laptop on 2026-11-22 with
    # experiments/laptop_transfer.py. Not rebuildable: the training run cannot
    # be repeated. They produced results/august_2026_original/.
    "ckpt_original": ("outputs/drift/checkpoints/original", ["processed_m5_v4"],
                      "not rebuildable; recovered from the old laptop (recovered/old_laptop/_manifest.json has the hashes)"),
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
