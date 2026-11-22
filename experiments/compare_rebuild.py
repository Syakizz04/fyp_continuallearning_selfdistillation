"""
Compare a rebuilt M5 dataset + base checkpoints against the lost originals.

    python -m experiments.compare_rebuild                  # every stage that exists
    python -m experiments.compare_rebuild --walk-checks 6  # + fingerprint walk (CPU: minutes)
    python -m experiments.compare_rebuild --walk-checks 157  # whole frozen walk (CPU: ~35 min)

## Why this exists

The original `data/processed_m5_v4/` and `outputs/drift/checkpoints/base_v4/`
were lost with the old laptop (Nov 2026). Rebuilding from raw M5 is not
guaranteed to reproduce them: `rebuild_inventory.py` records that re-running
`build_m5 --stores CA_1 --top-n-items 100` produced 182,557 rows against v2's
184,412, with one item different, and the department-pooled elasticities of
every other item shifted with it. So the rebuild has to be measured against the
original, not assumed to match it.

Each stage is checked in the order the pipeline produces it. The first DIFF
says where the rebuild diverged. Everything after that stage will inherit the
divergence, so those later DIFFs are consequences, not separate findings.

## Where the reference numbers come from

None of the original artefacts survive. The references are what was recorded
about them at the time:

  * `experiments/reference/old_laptop_drift_stream_frozen.csv` and
    `old_laptop_calibration.json` - copied from the `dashboard-deploy` branch
    (FYP1 drift walk). The forecasting half belongs to the same base TFT used by
    v1-v4, because `demand_forecasting.csv` was byte-identical across them. The
    RL half belongs to the ORIGINAL v1 `base` pricer (ref profit 3346.98), NOT
    base_v4, so only the forecasting numbers are used from it.
  * Every other number is quoted from a commit message or docstring, which is
    named next to it below.

The frozen arm's walk MASE depends only on the forecaster (it never retrains,
and pricing does not feed into the forecasting metric), so a rebuilt TFT can be
compared per check against the saved stream. Its profit columns cannot: those
came from the v1 pricer.

The old 100-item list was not recorded anywhere that survived, so item
identity cannot be checked. Only the count and the distributional summaries can.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

REF_DIR = PROJECT_ROOT / "experiments" / "reference"
REF_STREAM = REF_DIR / "old_laptop_drift_stream_frozen.csv"
REF_CALIB = REF_DIR / "old_laptop_calibration.json"


# ─── Reference values (old laptop) ──────────────────────────────────────────
# (value, tolerance, source). Tolerances are "close enough that a reader would
# call it the same result", not statistical tests.

REF = {
    # rebuild_inventory.py docstring
    "v2_rows":               (184_412, 0,      "rebuild_inventory.py docstring"),
    "v2_items":              (100,     0,      "build scope: CA_1, 100 items stratified"),
    "v2_elasticity_median":  (-0.60,   0.05,   "rebuild_inventory.py docstring"),
    # dataset_generator/m5/README.md
    "v2_identified":         (54,      0,      "m5/README.md"),
    "v2_wrong_sign":         (21,      0,      "m5/README.md"),
    "v2_no_variation":       (25,      0,      "m5/README.md"),
    "prior_mu":              (-0.524,  0.01,   "m5/README.md"),
    "prior_tau":             (0.313,   0.01,   "m5/README.md"),
    # commit 4f7faf4 / CLAUDE.md
    "v3_stockout_rate":      (0.051,   0.005,  "commit 4f7faf4 (5.1% stockout days)"),
    "v4_comp_corr":          (0.995,   0.005,  "commit 4f7faf4"),
    "v4_comp_ratio_mean":    (0.971,   0.01,   "commit 4f7faf4"),
    "v4_comp_ratio_std":     (0.072,   0.01,   "commit 4f7faf4"),
    # commit 4f7faf4: base_v4 pricer's calibration profit reference
    "pricer_ref_profit_mu":  (3971.00, 0.05,   "commit 4f7faf4 (relative tolerance)"),
    # commit 4f7faf4 / 577105e / CLAUDE.md
    "walk6_mase":            (1.469544, 1e-6,  "CLAUDE.md; first 6 checks of the frozen stream"),
}


# ─── Reporting ───────────────────────────────────────────────────────────────

@dataclass
class Row:
    stage: str
    metric: str
    old: object
    new: object
    status: str          # SAME / DIFF / MISSING / INFO
    note: str = ""


ROWS: List[Row] = []


def _fmt(v) -> str:
    if v is None:
        return "-"
    if isinstance(v, float):
        if math.isnan(v):
            return "nan"
        return f"{v:,.6f}".rstrip("0").rstrip(".") if abs(v) < 100 else f"{v:,.2f}"
    if isinstance(v, int):
        return f"{v:,}"
    return str(v)


def compare(stage: str, metric: str, key: str, new, relative: bool = False) -> None:
    old, tol, src = REF[key]
    if new is None:
        ROWS.append(Row(stage, metric, old, None, "MISSING", src))
        return
    diff = abs(new - old) / abs(old) if relative else abs(new - old)
    ROWS.append(Row(stage, metric, old, new, "SAME" if diff <= tol else "DIFF", src))


def missing(stage: str, what: str) -> None:
    ROWS.append(Row(stage, what, None, None, "MISSING", "not built yet"))


def info(stage: str, metric: str, old, new, note: str = "") -> None:
    ROWS.append(Row(stage, metric, old, new, "INFO", note))


def print_report() -> int:
    w = [max(len(r.stage) for r in ROWS), max(len(r.metric) for r in ROWS)]
    head = f"{'stage':<{w[0]}}  {'metric':<{w[1]}}  {'old laptop':>14}  {'rebuild':>14}  status   source"
    print("\n" + head + "\n" + "-" * len(head))
    for r in ROWS:
        print(f"{r.stage:<{w[0]}}  {r.metric:<{w[1]}}  {_fmt(r.old):>14}  "
              f"{_fmt(r.new):>14}  {r.status:<7}  {r.note}")
    first = next((r for r in ROWS if r.status == "DIFF"), None)
    print()
    if first:
        print(f"First divergence: [{first.stage}] {first.metric}. Later stages inherit it -")
        print("treat downstream results as a fresh run, not a reproduction.")
    elif any(r.status == "MISSING" for r in ROWS):
        print("No divergence in the stages that exist; MISSING stages are not built yet.")
    else:
        print("Every recorded number reproduces.")
    return 1 if first else 0


# ─── Stages ──────────────────────────────────────────────────────────────────

def _md5(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def check_v2(data_root: Path) -> None:
    import pandas as pd
    d = data_root / "processed_m5_v2"
    rl_path, rep_path = d / "rl_environment.csv", d / "elasticity_report.csv"
    if not rl_path.exists():
        missing("v2", f"{rl_path.relative_to(PROJECT_ROOT)}")
        return
    rl = pd.read_csv(rl_path, usecols=["product_id"])
    compare("v2", "rows", "v2_rows", int(len(rl)))
    compare("v2", "items", "v2_items", int(rl["product_id"].nunique()))
    if not rep_path.exists():
        missing("v2", "elasticity_report.csv")
        return
    rep = pd.read_csv(rep_path)
    compare("v2", "elasticity median (per item)", "v2_elasticity_median",
            float(rep["elasticity"].median()))
    ident = rep["identified"].astype(str).str.lower().eq("true")
    wrong = (~ident) & (rep["raw_beta"] > 0)
    compare("v2", "items identified from own data", "v2_identified", int(ident.sum()))
    compare("v2", "items discarded as wrong-sign", "v2_wrong_sign", int(wrong.sum()))
    compare("v2", "items with no price variation", "v2_no_variation",
            int((~ident & ~wrong).sum()))


def check_prior(data_root: Path) -> None:
    p = data_root / "external_pricing" / "prior.json"
    if not p.exists():
        info("prior", "prior.json", "fitted", None,
             "absent: build_m5 falls back to a placeholder prior and v2 WILL differ")
        return
    d = json.loads(p.read_text())
    compare("prior", "mu", "prior_mu", float(d["mu"]))
    compare("prior", "tau", "prior_tau", math.sqrt(float(d["tau2"])))


def check_v3(data_root: Path) -> None:
    import pandas as pd
    p = data_root / "processed_m5_v3" / "rl_environment.csv"
    if not p.exists():
        missing("v3", f"{p.relative_to(PROJECT_ROOT)}")
        return
    rl = pd.read_csv(p, usecols=["stockout_flag"])
    compare("v3", "stockout rate", "v3_stockout_rate", float(rl["stockout_flag"].mean()))


def check_v4(data_root: Path) -> None:
    import numpy as np
    import pandas as pd
    d = data_root / "processed_m5_v4"
    p = d / "rl_environment.csv"
    if not p.exists():
        missing("v4", f"{p.relative_to(PROJECT_ROOT)}")
        return
    rl = pd.read_csv(p, usecols=["competitor_price", "base_price"])
    comp = rl["competitor_price"].to_numpy(float)
    base = rl["base_price"].to_numpy(float)
    ok = np.isfinite(comp) & np.isfinite(base)
    compare("v4", "corr(competitor, own price)", "v4_comp_corr",
            float(np.corrcoef(comp[ok], base[ok])[0, 1]))
    r = comp / np.where(base > 0, base, np.nan)
    compare("v4", "competitor/own ratio mean", "v4_comp_ratio_mean", float(np.nanmean(r)))
    compare("v4", "competitor/own ratio std", "v4_comp_ratio_std", float(np.nanstd(r)))
    # Internal consistency, not a comparison with the old laptop: the base TFT is
    # only valid on v4 if the forecasting CSV came through v2 -> v4 untouched.
    v2 = data_root / "processed_m5_v2" / "demand_forecasting.csv"
    if v2.exists() and (d / "demand_forecasting.csv").exists():
        same = _md5(v2) == _md5(d / "demand_forecasting.csv")
        ROWS.append(Row("v4", "demand_forecasting.csv == v2's", "identical",
                        "identical" if same else "CHANGED",
                        "SAME" if same else "DIFF", "rebuild_inventory/refresh copy it through"))


def check_checkpoints(ckpt: Path) -> None:
    if not (ckpt / "base_tft.ckpt").exists():
        missing("ckpt", f"{ckpt.relative_to(PROJECT_ROOT)}/base_tft.ckpt")
        return
    cal_path = ckpt / "calibration.json"
    if not cal_path.exists():
        missing("ckpt", "calibration.json")
        return
    new = json.loads(cal_path.read_text())
    old = json.loads(REF_CALIB.read_text())
    nf, of = new["forecasting"], old["forecasting"]
    # A retrained TFT is a different draw; "same" = within one calibration sigma.
    sig = of["mase_sigma"]
    for k, label in (("mase_mu", "TFT calibration MASE mu"),
                     ("mase_sigma", "TFT calibration MASE sigma")):
        diff = abs(nf[k] - of[k])
        ROWS.append(Row("ckpt", label, of[k], nf[k], "SAME" if diff <= sig else "DIFF",
                        "old_laptop_calibration.json (tol = 1 old sigma)"))
    thr_old, thr_new = of["thresholds"]["k=2.0"], nf["thresholds"]["k=2.0"]
    info("ckpt", "drift threshold (k=2)", thr_old, thr_new,
         "sets when retrains fire; a shift moves every arm's schedule")
    compare("ckpt", "pricer ref profit mu", "pricer_ref_profit_mu",
            float(new["rl"]["ref_profit_mu"]), relative=True)


def check_walk(data_root: Path, ckpt: Path, n_checks: int) -> None:
    import pandas as pd

    v4 = data_root / "processed_m5_v4"
    if not (v4 / "rl_environment.csv").exists() or not (ckpt / "base_tft.ckpt").exists():
        missing("walk", "needs v4 data + checkpoints")
        return

    from drift_pipeline.core_pipeline import CONFIG, prepare_drift_data
    from drift_pipeline import monitor as mon
    from drift_pipeline import retrain as rt

    CONFIG["paths"]["demand_csv"] = str(v4 / "demand_forecasting.csv")
    CONFIG["paths"]["rl_csv"] = str(v4 / "rl_environment.csv")
    # Never write into outputs/drift/results: that directory holds the only
    # other copy of the old laptop's result files.
    CONFIG["paths"]["results"] = tempfile.mkdtemp(prefix="compare_rebuild_")

    data = prepare_drift_data()
    data["checks"] = list(data["checks"])[:n_checks]
    base = mon.load_base({
        "tft": str(ckpt / "base_tft.ckpt"),
        "dataset": str(ckpt / "base_tft_dataset.pkl"),
        "ppo": str(ckpt / "base_ppo.zip"),
        "meta": str(ckpt / "base_meta.json"),
        "calibration": str(ckpt / "calibration.json"),
    })
    ctrl = rt.RetrainController("frozen", base, data)
    res = mon.walk_forward(data, base, arm="frozen",
                           forecaster_provider=ctrl.forecaster_provider,
                           pricer_provider=ctrl.pricer_provider)

    new = pd.DataFrame(res.stream)[["date", "mase"]]
    new["date"] = pd.to_datetime(new["date"])
    old = pd.read_csv(REF_STREAM, usecols=["date", "mase"], parse_dates=["date"])
    m = old.merge(new, on="date", suffixes=("_old", "_new"))
    if m.empty:
        info("walk", "check dates", "2013-01-01..", "no overlap",
             "walk calendar changed; cannot compare per check")
        return

    if n_checks >= 6:
        compare("walk", "frozen MASE, first 6 checks", "walk6_mase",
                float(m["mase_new"].head(6).mean()))
    info("walk", f"frozen MASE mean ({len(m)} checks)",
         float(m["mase_old"].mean()), float(m["mase_new"].mean()))
    info("walk", "max |per-check MASE diff|", 0.0,
         float((m["mase_new"] - m["mase_old"]).abs().max()))
    out = PROJECT_ROOT / "outputs" / "drift" / "compare_rebuild_walk.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    m.to_csv(out, index=False)
    info("walk", "per-check table", "", str(out.relative_to(PROJECT_ROOT)))


# ─── Main ────────────────────────────────────────────────────────────────────

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--data-root", default=str(PROJECT_ROOT / "data"))
    ap.add_argument("--ckpt", default=str(PROJECT_ROOT / "outputs" / "drift" / "checkpoints" / "base_v4"))
    ap.add_argument("--walk-checks", type=int, default=0,
                    help="run the frozen walk for N checks (0 = skip; 6 = the recorded "
                         "fingerprint; 157 = the whole walk)")
    args = ap.parse_args()

    data_root, ckpt = Path(args.data_root), Path(args.ckpt)
    for path in (REF_STREAM, REF_CALIB):
        if not path.exists():
            raise SystemExit(f"reference file missing: {path}")

    check_prior(data_root)
    check_v2(data_root)
    check_v3(data_root)
    check_v4(data_root)
    check_checkpoints(ckpt)
    if args.walk_checks:
        check_walk(data_root, ckpt, args.walk_checks)
    return print_report()


if __name__ == "__main__":
    raise SystemExit(main())
