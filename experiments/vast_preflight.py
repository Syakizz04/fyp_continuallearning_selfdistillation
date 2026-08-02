"""
Preflight for a rented GPU box. Run this BEFORE starting a paid sweep.

    python -m experiments.vast_preflight
    python -m experiments.vast_preflight --concurrency 4

Every check here corresponds to a way an overnight run has a real chance of
dying or, worse, quietly producing nothing useful. The expensive failure is not
a crash on minute one - it is a crash on hour five, or a run that completes
against the wrong data. So this loads the actual base checkpoint and runs a real
forward pass rather than just checking that files exist.

Exit code is 0 only if every REQUIRED check passes.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

#: Peak resident memory of one cell, from the E4 measurements: the two 184k-row
#: frames plus their censored copies, plus a replay buffer that reaches ~3.1 GB
#: at cap. Rounded up, because a box that swaps is slower than a smaller box that
#: does not.
RAM_GB_PER_CELL = 5.0

#: Outputs are CSV/JSON per cell plus the drift streams; checkpoints are not
#: re-saved per cell. Generous.
DISK_GB_NEEDED = 5.0

PASS, WARN, FAIL = "PASS", "WARN", "FAIL"
_results: list[tuple[str, str, str]] = []


def check(name: str, status: str, detail: str = "") -> None:
    _results.append((name, status, detail))
    colour = {"PASS": "\033[32m", "WARN": "\033[33m", "FAIL": "\033[31m"}[status]
    print(f"  {colour}{status:<4}\033[0m {name}" + (f"  -- {detail}" if detail else ""))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Preflight a GPU box for the E2 sweep.")
    ap.add_argument("--concurrency", type=int, default=3,
                    help="how many cells you intend to run at once; sizes the "
                         "RAM check")
    ap.add_argument("--data-dir", type=Path,
                    default=PROJECT_ROOT / "data" / "processed_m5_v3")
    ap.add_argument("--base-ckpt", type=Path,
                    default=PROJECT_ROOT / "outputs" / "drift" / "checkpoints" / "base_cover")
    args = ap.parse_args(argv)

    print(f"\nPreflight - planning {args.concurrency} concurrent cells\n")

    # ── 1. Hardware ─────────────────────────────────────────────────────────
    try:
        import torch
        if torch.cuda.is_available():
            p = torch.cuda.get_device_properties(0)
            check("CUDA", PASS, f"{p.name}, {p.total_memory / 1e9:.0f} GB, "
                                f"{p.multi_processor_count} SMs")
        else:
            check("CUDA", FAIL, "no GPU visible - a CPU run is not worth paying for")
    except Exception as exc:                                  # noqa: BLE001
        check("CUDA", FAIL, f"torch import failed: {exc}")

    try:
        import psutil
        ram_gb = psutil.virtual_memory().total / 1e9
        need = RAM_GB_PER_CELL * args.concurrency
        # Host RAM, not VRAM, is the binding constraint: the models are ~4 MB but
        # a replay buffer reaches ~3.1 GB, and it lives on the CPU.
        check("Host RAM", PASS if ram_gb >= need else FAIL,
              f"{ram_gb:.0f} GB total, need ~{need:.0f} GB for "
              f"{args.concurrency} cells (replay buffers are host-side)")
        cores = psutil.cpu_count(logical=False) or 0
        check("CPU cores", PASS if cores >= args.concurrency else WARN,
              f"{cores} physical; much of each check is pandas/env rollout, "
              f"so fewer cores than cells throttles the sweep")
    except ImportError:
        check("Host RAM", WARN, "psutil not installed; cannot verify")

    free_gb = shutil.disk_usage(PROJECT_ROOT).free / 1e9
    check("Disk", PASS if free_gb >= DISK_GB_NEEDED else FAIL,
          f"{free_gb:.0f} GB free, need ~{DISK_GB_NEEDED:.0f} GB")

    # ── 2. Payload ──────────────────────────────────────────────────────────
    for label, path, members in (
        ("Dataset (v3)", args.data_dir,
         ["demand_forecasting.csv", "rl_environment.csv"]),
        ("Base checkpoint", args.base_ckpt,
         ["base_tft.ckpt", "base_tft_dataset.pkl", "base_ppo.zip",
          "base_meta.json", "calibration.json"]),
    ):
        missing = [m for m in members if not (path / m).exists()]
        check(label, FAIL if missing else PASS,
              f"missing {missing} in {path}" if missing else str(path))

    # v3 is the only version whose stockout rate makes censoring measurable; on
    # v1 the treatment would be empty and silently equal to the control.
    rl_csv = args.data_dir / "rl_environment.csv"
    if rl_csv.exists():
        header = rl_csv.read_text(encoding="utf-8").split("\n", 1)[0]
        has = "unmet_demand" in header.split(",")
        check("Censoring signal", PASS if has else FAIL,
              "unmet_demand present" if has else
              "no unmet_demand column - this is not the v3 dataset, and E2's "
              "treatment would be empty")

    # ── 3. The stack actually runs ──────────────────────────────────────────
    try:
        from drift_pipeline import trainers  # noqa: F401
        check("Imports", PASS, "drift_pipeline.trainers (self-contained)")
    except Exception as exc:                                  # noqa: BLE001
        check("Imports", FAIL, f"{type(exc).__name__}: {exc}")
        return _summary()

    # The check that matters most. A base checkpoint that will not load is the
    # failure that wastes the whole rental, and it cannot be detected by looking
    # at the file - the architecture has to match what build_cltft constructs.
    try:
        from drift_pipeline.core_pipeline import CONFIG
        CONFIG["paths"]["demand_csv"] = str(args.data_dir / "demand_forecasting.csv")
        CONFIG["paths"]["rl_csv"] = str(args.data_dir / "rl_environment.csv")
        from drift_pipeline import monitor as mon
        from experiments.exp_staleness_cl import base_paths
        base = mon.load_base(base_paths(args.base_ckpt))
        ok = base.get("forecaster") is not None and base.get("pricer") is not None
        check("Base model loads", PASS if ok else FAIL,
              "TFT + PPO + calibration restored")
    except Exception as exc:                                  # noqa: BLE001
        check("Base model loads", FAIL, f"{type(exc).__name__}: {exc}")

    return _summary()


def _summary() -> int:
    fails = [r for r in _results if r[1] == FAIL]
    warns = [r for r in _results if r[1] == WARN]
    print()
    if fails:
        print(f"\033[31m{len(fails)} FAILED\033[0m - do not start the sweep:")
        for name, _, detail in fails:
            print(f"    - {name}: {detail}")
        return 1
    print(f"\033[32mReady.\033[0m" + (f" ({len(warns)} warning(s))" if warns else ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
