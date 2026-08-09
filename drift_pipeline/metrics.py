# Phase 5: grid-anchored metrics + probe-based forgetting for the drift experiment.
#
# Turns the per-arm walk streams (Phase 3/4) and the arms' FINAL models into the
# three headline tables:
#   1. accuracy   — grid-anchored forecast error + RL profit per arm over the walk,
#                   with profit expressed as an index vs the frozen baseline.
#   2. forgetting — re-score each arm's FINAL model on fixed quarterly PROBE windows.
#                   Degradation on BASE-era probes (pre-drift knowledge) = forgetting;
#                   improvement on WALK-era probes = adaptation benefit. The frozen
#                   arm's final model == the base, so it is the method-independent
#                   anchor every other arm is differenced against.
#   3. efficiency — retrain counts / train-steps vs accuracy vs forgetting. This is
#                   the accuracy-vs-#retrains tradeoff that IS the result.
#
# All three are derived from on-disk artifacts (drift_stream_*.csv, retrain_log_*.json)
# plus a fresh probe re-score of each arm's live final model, so Phase 5 runs in the
# same process as Phase 4 (the final models live in the returned controllers).

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from .core_pipeline import CONFIG, console
from . import base_training as bt


# ─── Probe re-scoring (forgetting basis) ─────────────────────────────────────

def _probe_era(probe: Dict) -> str:
    """A probe is 'base' if it ends before the walk starts (pre-drift knowledge),
    else 'walk' (an era the arm could adapt to)."""
    walk_start = pd.Timestamp(CONFIG["timeline"]["walk_start"])
    return "base" if pd.Timestamp(probe["end"]) < walk_start else "walk"


def score_model_on_probes(forecaster, pricer, data: Dict, train_ds) -> List[Dict]:
    """Score one arm's FINAL forecaster + pricer on every fixed probe window.

    Each probe is a `window_days`-long slice anchored at a quarter end; we forecast
    from its start over one horizon. MASE uses the SAME base-era seasonal-naive
    denominator (`data['tft_base']`) for every arm/probe, so the numbers are
    grid-anchored and directly comparable across asynchronously-retrained arms.
    """
    rows: List[Dict] = []
    for p in data["probes"]:
        origin = pd.Timestamp(p["start"])
        fc = bt.window_forecast_error(
            forecaster, data["tft_full"], train_ds, data["tft_base"], origin)
        rl = bt.window_rl_profit(pricer, data["rl_full"], origin)
        rows.append({
            "name": p["name"], "era": _probe_era(p),
            "start": str(pd.Timestamp(p["start"]).date()),
            "end":   str(pd.Timestamp(p["end"]).date()),
            "mase":  fc.get("mase"), "smape": fc.get("smape"), "rmse": fc.get("rmse"),
            "cumulative_profit": rl.get("cumulative_profit"),
            "avg_episode_reward": rl.get("avg_episode_reward"),
            "pricing_regret": rl.get("pricing_regret"),
        })
    return rows


def save_probe_scores(arm: str, scores: List[Dict]) -> str:
    res_dir = Path(CONFIG["paths"]["results"]); res_dir.mkdir(parents=True, exist_ok=True)
    path = res_dir / f"probe_scores_{arm}.json"
    path.write_text(json.dumps({"arm": arm, "probes": scores}, indent=2))
    return str(path)


# ─── Loading on-disk artifacts ───────────────────────────────────────────────

def load_streams(arms: List[str]) -> Dict[str, pd.DataFrame]:
    res_dir = Path(CONFIG["paths"]["results"])
    out = {}
    for arm in arms:
        p = res_dir / f"drift_stream_{arm}.csv"
        if p.exists():
            out[arm] = pd.read_csv(p, parse_dates=["date"])
    return out


def load_retrain_logs(arms: List[str]) -> Dict[str, Dict]:
    res_dir = Path(CONFIG["paths"]["results"])
    out = {}
    for arm in arms:
        p = res_dir / f"retrain_log_{arm}.json"
        out[arm] = json.loads(p.read_text()) if p.exists() else {
            "n_fc_retrains": 0, "n_rl_retrains": 0,
            "fc_epochs_total": 0, "rl_steps_total": 0}
    return out


def _finite_mean(s: pd.Series) -> float:
    s = pd.to_numeric(s, errors="coerce")
    s = s[np.isfinite(s)]
    return float(s.mean()) if len(s) else float("nan")


# ─── Table 1: grid-anchored accuracy ─────────────────────────────────────────

def build_accuracy_table(streams: Dict[str, pd.DataFrame],
                         reference: str = "frozen") -> pd.DataFrame:
    """Per-arm mean/median forecast error and RL profit over the walk. Profit is
    also reported as an index vs the `reference` arm (frozen baseline) on matching
    check dates — the design's comparison reference, distinct from the calibration
    trigger reference already in the stream's `profit_index` column."""
    ref = streams.get(reference)
    ref_profit = (ref.set_index("date")["cumulative_profit"] if ref is not None else None)

    recs = []
    for arm, df in streams.items():
        rec = {
            "arm": arm,
            "n_checks": int(len(df)),
            "walk_mase_mean":   _finite_mean(df["mase"]),
            "walk_mase_median": float(pd.to_numeric(df["mase"], errors="coerce").median()),
            "walk_smape_mean":  _finite_mean(df["smape"]),
            "walk_profit_mean": _finite_mean(df["cumulative_profit"]),
            "n_fc_drift_checks": int(df.get("fc_trigger", pd.Series(dtype=bool)).sum()),
            "n_rl_drift_checks": int(df.get("rl_trigger", pd.Series(dtype=bool)).sum()),
        }
        if ref_profit is not None:
            a = df.set_index("date")["cumulative_profit"]
            joined = pd.concat([a.rename("arm"), ref_profit.rename("ref")], axis=1).dropna()
            joined = joined[joined["ref"].abs() > 1e-9]
            rec["profit_index_vs_frozen"] = (
                float((joined["arm"] / joined["ref"]).mean()) if len(joined) else float("nan"))
        recs.append(rec)
    return pd.DataFrame(recs).set_index("arm")


# ─── Table 2: probe-based forgetting ─────────────────────────────────────────

def build_forgetting_table(probe_by_arm: Dict[str, List[Dict]],
                           anchor: str = "frozen") -> pd.DataFrame:
    """Difference each arm's final-model probe error against the anchor (frozen =
    base) on the SAME probes. Positive base-era MASE delta = the arm forgot
    pre-drift knowledge; positive walk-era improvement = it adapted to new eras."""
    anchor_probes = {p["name"]: p for p in probe_by_arm[anchor]}

    recs = []
    for arm, probes in probe_by_arm.items():
        base_fmase, walk_improve = [], []
        base_pdrop = []
        raw_base, raw_walk = [], []
        for p in probes:
            ap = anchor_probes.get(p["name"])
            if ap is None:
                continue
            am, rm = p.get("mase"), ap.get("mase")
            if am is not None and rm is not None and np.isfinite(am) and np.isfinite(rm):
                if p["era"] == "base":
                    base_fmase.append(am - rm)        # >0 -> worse than base => forgetting
                    raw_base.append(am)
                else:
                    walk_improve.append(rm - am)      # >0 -> better than base => adaptation
                    raw_walk.append(am)
            # RL profit retention on base-era probes (relative drop vs anchor)
            apr, rpr = p.get("cumulative_profit"), ap.get("cumulative_profit")
            if (p["era"] == "base" and apr is not None and rpr is not None
                    and np.isfinite(apr) and np.isfinite(rpr) and abs(rpr) > 1e-9):
                base_pdrop.append((rpr - apr) / rpr)  # >0 -> lost profit on old era

        recs.append({
            "arm": arm,
            "forgetting_mase_base_era":  float(np.mean(base_fmase)) if base_fmase else float("nan"),
            "adaptation_mase_walk_era":  float(np.mean(walk_improve)) if walk_improve else float("nan"),
            "forgetting_profit_base_era": float(np.mean(base_pdrop)) if base_pdrop else float("nan"),
            "probe_mase_base_mean": float(np.mean(raw_base)) if raw_base else float("nan"),
            "probe_mase_walk_mean": float(np.mean(raw_walk)) if raw_walk else float("nan"),
            "n_base_probes": len(base_fmase), "n_walk_probes": len(walk_improve),
        })
    return pd.DataFrame(recs).set_index("arm")


# ─── Table 3: efficiency (the accuracy-vs-#retrains tradeoff) ─────────────────

def build_efficiency_table(retrain_logs: Dict[str, Dict],
                           accuracy: pd.DataFrame,
                           forgetting: pd.DataFrame) -> pd.DataFrame:
    recs = []
    for arm, log in retrain_logs.items():
        recs.append({
            "arm": arm,
            "n_fc_retrains": log.get("n_fc_retrains", 0),
            "n_rl_retrains": log.get("n_rl_retrains", 0),
            "n_retrains_total": log.get("n_fc_retrains", 0) + log.get("n_rl_retrains", 0),
            "fc_epochs_total": log.get("fc_epochs_total", 0),
            "rl_steps_total": log.get("rl_steps_total", 0),
        })
    eff = pd.DataFrame(recs).set_index("arm")
    acc_cols = [c for c in ("walk_mase_mean", "profit_index_vs_frozen") if c in accuracy]
    fg_cols  = [c for c in ("forgetting_mase_base_era", "forgetting_profit_base_era",
                            "adaptation_mase_walk_era")
                if c in forgetting]
    eff = eff.join(accuracy[acc_cols], how="left").join(forgetting[fg_cols], how="left")
    return eff


# ─── Orchestration ───────────────────────────────────────────────────────────

def run_phase5(data: Dict, arms_out: Dict[str, Dict],
               anchor: str = "frozen") -> Dict:
    """Score every arm's final model on the probe grid, then build + persist the
    accuracy / forgetting / efficiency tables. `arms_out` is the dict returned by
    `retrain.run_all_arms` (each value carries the arm's live controller)."""
    bt.sync_config()
    res_dir = Path(CONFIG["paths"]["results"]); res_dir.mkdir(parents=True, exist_ok=True)
    arms = list(arms_out.keys())

    # 1. Probe re-score each arm's FINAL model.
    console.print(f"[bold]Phase 5[/bold]: probe-scoring {len(arms)} arms over "
                  f"{len(data['probes'])} probe windows...")
    probe_by_arm: Dict[str, List[Dict]] = {}
    for arm in arms:
        ctrl = arms_out[arm]["controller"]
        scores = score_model_on_probes(ctrl.forecaster, ctrl.pricer, data, ctrl.train_ds)
        save_probe_scores(arm, scores)
        probe_by_arm[arm] = scores
        n_fin = sum(1 for s in scores if s["mase"] is not None and np.isfinite(s["mase"]))
        console.print(f"  [cyan]{arm}[/cyan]: {n_fin}/{len(scores)} probes scored")

    if anchor not in probe_by_arm:
        anchor = arms[0]
        console.print(f"  [yellow]frozen anchor missing; using '{anchor}' as forgetting anchor[/yellow]")

    # 2. Build tables.
    streams = load_streams(arms)
    retrain_logs = load_retrain_logs(arms)
    accuracy   = build_accuracy_table(streams, reference=anchor)
    forgetting = build_forgetting_table(probe_by_arm, anchor=anchor)
    efficiency = build_efficiency_table(retrain_logs, accuracy, forgetting)

    # 3. Persist.
    accuracy.to_csv(res_dir / "metrics_accuracy.csv")
    forgetting.to_csv(res_dir / "metrics_forgetting.csv")
    efficiency.to_csv(res_dir / "metrics_efficiency.csv")
    (res_dir / "metrics_summary.json").write_text(json.dumps({
        "accuracy":   accuracy.reset_index().to_dict(orient="records"),
        "forgetting": forgetting.reset_index().to_dict(orient="records"),
        "efficiency": efficiency.reset_index().to_dict(orient="records"),
    }, indent=2, default=float))

    _print_tables(accuracy, forgetting, efficiency)
    console.print(f"[green]✓ Phase 5 metrics saved[/green] -> {res_dir}")
    return {"accuracy": accuracy, "forgetting": forgetting,
            "efficiency": efficiency, "probe_scores": probe_by_arm}


def _print_tables(accuracy: pd.DataFrame, forgetting: pd.DataFrame,
                  efficiency: pd.DataFrame) -> None:
    from rich.table import Table
    t = Table(title="Efficiency — retrains vs accuracy vs forgetting",
              header_style="bold cyan")
    cols = ["arm", "n_fc_retrains", "n_rl_retrains", "walk_mase_mean",
            "profit_index_vs_frozen", "forgetting_mase_base_era",
            "adaptation_mase_walk_era"]
    for c in cols:
        t.add_column(c, justify="right" if c != "arm" else "left")
    for arm, r in efficiency.iterrows():
        def f(x):
            return "-" if (x is None or (isinstance(x, float) and not np.isfinite(x))) else (
                f"{x:.3f}" if isinstance(x, float) else str(x))
        t.add_row(str(arm), f(r["n_fc_retrains"]), f(r["n_rl_retrains"]),
                  f(r["walk_mase_mean"]), f(r.get("profit_index_vs_frozen")),
                  f(r["forgetting_mase_base_era"]), f(r["adaptation_mase_walk_era"]))
    console.print(t)


def run_drift_experiment(data: Dict, arms: Optional[List[str]] = None) -> Dict:
    """End-to-end Phase 3-5 driver: run every arm's walk-forward (+ retrains), then
    compute the grid-anchored metrics / forgetting / efficiency tables."""
    from . import retrain as rt
    arms_out = rt.run_all_arms(data, arms=arms)
    metrics = run_phase5(data, arms_out)
    return {"arms": arms_out, "metrics": metrics}
