"""
Survey a recovered drive (the old laptop's) for FYP2 artifacts. Read-only.

    python -m experiments.recover_scan "D:/Users/MDT/Downloads/projects/fyp_continuallearning_selfdistillation"
    python -m experiments.recover_scan D:/ --out results/recovery/scan.json

It answers the questions that decide what to do with the old files:

1. Data: is each recovered data file byte-identical to the rebuilt one in
   artifacts.json? It reports MATCH or DIFFERS per group. data/processed_m5
   (v1) has no rebuilt copy, so it is flagged as the one irreplaceable dataset.
   The original TFT was most likely trained on v1 (outputs/drift/REBUILD_SUMMARY.txt).
2. Checkpoints: is each base_tft.ckpt / base_ppo.zip the rebuilt one, or a
   different model? A different base_tft.ckpt is probably the ORIGINAL
   forecaster, the one behind every August E2 number and the FYP1 drift
   results. Its calibration.json mu is printed: about 1.481 means original,
   about 0.891 means rebuilt.
3. Results: the E2/E5/replicate tables that were lost because outputs/ is
   gitignored. Every CSV/JSON under an outputs/ folder is listed with its size
   and mtime.

Nothing is copied or modified. What to restore, and whether the original
models replace the rebuilt baseline, is a decision for after reading this.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

from experiments.artifacts import MANIFEST, ROOT, sha256

CKPT_NAMES = {"base_tft.ckpt", "base_ppo.zip", "base_tft_dataset.pkl", "calibration.json", "base_meta.json"}
RESULT_SUFFIXES = {".csv", ".json", ".txt", ".parquet"}
SKIP_DIRS = {".git", ".venv", "venv", "node_modules", "__pycache__", "$Recycle.Bin", "Windows",
             "Program Files", "Program Files (x86)", "ProgramData", "AppData"}


def walk(root: Path):
    stack = [root]
    while stack:
        d = stack.pop()
        try:
            entries = list(d.iterdir())
        except (PermissionError, OSError):
            continue
        for p in entries:
            if p.is_dir():
                if p.name not in SKIP_DIRS:
                    stack.append(p)
            elif p.is_file():
                yield p


def load_manifest():
    man = json.loads(MANIFEST.read_text())
    by_hash, by_name = {}, defaultdict(list)
    for g, info in man["groups"].items():
        for f in info["files"]:
            by_hash[f["sha256"]] = (g, f["path"])
            by_name[Path(f["path"]).name].append((g, f))
    return by_hash, by_name


def calib_mu(p: Path):
    try:
        c = json.loads(p.read_text())
    except Exception:
        return None
    for key in ("forecasting", "fc", "tft"):
        v = c.get(key)
        if isinstance(v, dict):
            for k in ("mase_mu", "mu"):
                if isinstance(v.get(k), (int, float)):
                    return round(v[k], 3)
    return c.get("mu") if isinstance(c.get("mu"), (int, float)) else None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("root", type=Path, help="folder on the recovered drive (a project copy, or a whole drive)")
    ap.add_argument("--out", type=Path, default=None, help="also write the report as JSON")
    a = ap.parse_args()
    if not a.root.exists():
        print(f"{a.root} does not exist"); return 2
    if a.root.resolve() == ROOT:
        print("that is this repo; point it at the recovered copy"); return 2

    by_hash, by_name = load_manifest()
    known_names = set(by_name) | CKPT_NAMES
    t0 = time.time()
    data, ckpts, results, v1 = [], [], [], []
    for p in walk(a.root):
        parts = {q.lower() for q in p.parts}
        rel = p.as_posix()
        if "processed_m5" in parts:                       # v1: no rebuilt copy exists
            v1.append({"path": rel, "bytes": p.stat().st_size})
        if p.name in known_names:
            h = sha256(p)
            hit = by_hash.get(h)
            row = {"path": rel, "bytes": p.stat().st_size, "sha256": h,
                   "matches": f"{hit[0]}:{hit[1]}" if hit else None,
                   "same_name_in": sorted({g for g, _ in by_name.get(p.name, [])})}
            if p.name in CKPT_NAMES:
                if p.name == "calibration.json":
                    row["calibration_mu"] = calib_mu(p)
                ckpts.append(row)
            else:
                data.append(row)
        elif p.suffix.lower() in RESULT_SUFFIXES and "outputs" in parts:
            st = p.stat()
            results.append({"path": rel, "bytes": st.st_size,
                            "mtime": time.strftime("%Y-%m-%d %H:%M", time.localtime(st.st_mtime))})

    def show(title, rows, fmt):
        print(f"\n{title} ({len(rows)})")
        for r in rows:
            print("  " + fmt(r))

    show("DATA files with a rebuilt counterpart", data,
         lambda r: f"{'MATCH  ' if r['matches'] else 'DIFFERS'} {r['path']}")
    show("CHECKPOINT files", ckpts,
         lambda r: (f"{'REBUILT ' if r['matches'] else 'ORIGINAL?'} {r['path']}"
                    + (f"   calibration mu {r['calibration_mu']}" if r.get("calibration_mu") is not None else "")))
    print(f"\nv1 data (data/processed_m5, not rebuildable): {len(v1)} files, "
          f"{sum(r['bytes'] for r in v1) / 1e6:.1f} MB")
    for r in v1[:10]:
        print(f"  {r['path']}")
    results.sort(key=lambda r: r["mtime"], reverse=True)
    show("RESULT files under outputs/ (newest first, first 60)", results[:60],
         lambda r: f"{r['mtime']}  {r['bytes']:>10,d}  {r['path']}")

    orig = [r for r in ckpts if r["path"].endswith("base_tft.ckpt") and not r["matches"]]
    print("\nSUMMARY")
    print(f"  data files differing from the rebuild: {sum(not r['matches'] for r in data)} of {len(data)}")
    print(f"  base_tft.ckpt not matching the rebuild (candidate originals): {len(orig)}")
    print(f"  v1 dataset found: {'yes' if v1 else 'no'}")
    print(f"  result files under outputs/: {len(results)}")
    print(f"  scanned in {time.time() - t0:.0f}s; nothing was copied or changed")
    if a.out:
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(json.dumps({"root": a.root.as_posix(), "data": data, "checkpoints": ckpts,
                                     "v1": v1, "results": results}, indent=2) + "\n")
        print(f"  report -> {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
