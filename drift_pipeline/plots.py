# Phase 6: figures for the drift-triggered CL experiment.
#
# Reads only on-disk artifacts written by Phases 3-5 (drift_stream_*.csv,
# retrain_log_*.json, probe_scores_*.json, metrics_*.csv) so it can be run
# standalone after an experiment. Renders to CONFIG['paths']['plots'].
#
#   1. error_timeline       — walk MASE per arm vs the frozen baseline, retrain
#                             markers, and the drift threshold band.
#   2. profit_timeline      — profit index vs frozen over the walk.
#   3. retrain_counts       — FC/RL retrains per arm (the cost axis).
#   4. accuracy_vs_retrains — the headline tradeoff scatter (+ forgetting panel).
#   5. forgetting           — base-era forgetting vs walk-era adaptation per arm.

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .core_pipeline import CONFIG, console

_ARM_ORDER = ["frozen", "periodic", "ewc", "replay", "sdft"]
_COLORS = {"frozen": "#777777", "periodic": "#1f77b4", "ewc": "#2ca02c",
           "replay": "#ff7f0e", "sdft": "#d62728"}


def _res_dir() -> Path:
    return Path(CONFIG["paths"]["results"])


def _plot_dir() -> Path:
    p = Path(CONFIG["paths"]["plots"]); p.mkdir(parents=True, exist_ok=True)
    return p


def _arms_present() -> List[str]:
    found = [a for a in _ARM_ORDER
             if (_res_dir() / f"drift_stream_{a}.csv").exists()]
    return found


def _load_stream(arm: str) -> Optional[pd.DataFrame]:
    p = _res_dir() / f"drift_stream_{arm}.csv"
    return pd.read_csv(p, parse_dates=["date"]) if p.exists() else None


def _load_json(name: str) -> Optional[Dict]:
    p = _res_dir() / name
    return json.loads(p.read_text()) if p.exists() else None


def _retrain_dates(arm: str, model: str) -> List[pd.Timestamp]:
    log = _load_json(f"retrain_log_{arm}.json")
    if not log:
        return []
    return [pd.Timestamp(e["date"]) for e in log.get("events", [])
            if e.get("model") == model]


# ─── 1. Error timeline ───────────────────────────────────────────────────────

def plot_error_timeline(arms: List[str]) -> Path:
    fig, ax = plt.subplots(figsize=(12, 5))
    thr = None
    for arm in arms:
        df = _load_stream(arm)
        if df is None:
            continue
        ax.plot(df["date"], pd.to_numeric(df["mase"], errors="coerce"),
                label=arm, color=_COLORS.get(arm), lw=1.6,
                alpha=0.95 if arm != "frozen" else 0.7,
                ls="--" if arm == "frozen" else "-")
        if thr is None and "fc_threshold" in df:
            thr = float(pd.to_numeric(df["fc_threshold"], errors="coerce").dropna().iloc[0])
        # retrain markers
        for d in _retrain_dates(arm, "forecasting"):
            ax.axvline(d, color=_COLORS.get(arm), ls=":", lw=0.7, alpha=0.5)
    if thr is not None:
        ax.axhline(thr, color="black", ls="-.", lw=1.0,
                   label=f"drift threshold (k={CONFIG['drift']['fc_k_sigma']})")
    ax.set_title("Forecast error over the walk-forward (grid-anchored MASE)")
    ax.set_xlabel("date"); ax.set_ylabel("windowed MASE")
    ax.legend(ncol=3, fontsize=8); ax.grid(alpha=0.3)
    out = _plot_dir() / "error_timeline.png"
    fig.tight_layout(); fig.savefig(out, dpi=130); plt.close(fig)
    return out


# ─── 2. Profit timeline ──────────────────────────────────────────────────────

def plot_profit_timeline(arms: List[str]) -> Path:
    frozen = _load_stream("frozen")
    ref = (frozen.set_index("date")["cumulative_profit"] if frozen is not None else None)
    fig, ax = plt.subplots(figsize=(12, 5))
    for arm in arms:
        df = _load_stream(arm)
        if df is None:
            continue
        if ref is not None:
            a = df.set_index("date")["cumulative_profit"]
            j = pd.concat([a.rename("a"), ref.rename("r")], axis=1).dropna()
            j = j[j["r"].abs() > 1e-9]
            ax.plot(j.index, j["a"] / j["r"], label=arm, color=_COLORS.get(arm),
                    lw=1.6, ls="--" if arm == "frozen" else "-")
        for d in _retrain_dates(arm, "rl"):
            ax.axvline(d, color=_COLORS.get(arm), ls=":", lw=0.7, alpha=0.5)
    ax.axhline(1.0, color="black", ls="-.", lw=1.0, label="parity vs frozen")
    ax.set_title("RL profit index vs frozen baseline over the walk-forward")
    ax.set_xlabel("date"); ax.set_ylabel("profit / frozen profit")
    ax.legend(ncol=3, fontsize=8); ax.grid(alpha=0.3)
    out = _plot_dir() / "profit_timeline.png"
    fig.tight_layout(); fig.savefig(out, dpi=130); plt.close(fig)
    return out


# ─── 3. Retrain counts ───────────────────────────────────────────────────────

def plot_retrain_counts(arms: List[str]) -> Path:
    fc, rl = [], []
    for arm in arms:
        log = _load_json(f"retrain_log_{arm}.json") or {}
        fc.append(log.get("n_fc_retrains", 0)); rl.append(log.get("n_rl_retrains", 0))
    x = np.arange(len(arms)); w = 0.38
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(x - w / 2, fc, w, label="forecasting", color="#4c72b0")
    ax.bar(x + w / 2, rl, w, label="RL", color="#dd8452")
    for i, (a, b) in enumerate(zip(fc, rl)):
        ax.text(x[i] - w / 2, a, str(a), ha="center", va="bottom", fontsize=8)
        ax.text(x[i] + w / 2, b, str(b), ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x); ax.set_xticklabels(arms)
    ax.set_title("Retrains per arm (the cost axis)")
    ax.set_ylabel("# retrains"); ax.legend(); ax.grid(axis="y", alpha=0.3)
    out = _plot_dir() / "retrain_counts.png"
    fig.tight_layout(); fig.savefig(out, dpi=130); plt.close(fig)
    return out


# ─── 4. The headline tradeoff scatter ────────────────────────────────────────

def plot_accuracy_vs_retrains() -> Optional[Path]:
    p = _res_dir() / "metrics_efficiency.csv"
    if not p.exists():
        return None
    eff = pd.read_csv(p)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for _, r in eff.iterrows():
        arm = r["arm"]; c = _COLORS.get(arm, "#333")
        n = r["n_retrains_total"]
        axes[0].scatter(n, r["walk_mase_mean"], s=120, color=c, zorder=3)
        axes[0].annotate(arm, (n, r["walk_mase_mean"]),
                         textcoords="offset points", xytext=(6, 4), fontsize=9)
        if "forgetting_mase_base_era" in r:
            axes[1].scatter(n, r["forgetting_mase_base_era"], s=120, color=c, zorder=3)
            axes[1].annotate(arm, (n, r["forgetting_mase_base_era"]),
                             textcoords="offset points", xytext=(6, 4), fontsize=9)
    axes[0].set_title("Accuracy vs cost"); axes[0].set_xlabel("# retrains (FC+RL)")
    axes[0].set_ylabel("mean walk MASE (lower = better)"); axes[0].grid(alpha=0.3)
    axes[1].axhline(0, color="black", lw=0.8)
    axes[1].set_title("Forgetting vs cost"); axes[1].set_xlabel("# retrains (FC+RL)")
    axes[1].set_ylabel("base-era MASE increase vs frozen (lower = better)")
    axes[1].grid(alpha=0.3)
    fig.suptitle("Drift-triggered CL: the retrain-count tradeoff", fontweight="bold")
    out = _plot_dir() / "accuracy_vs_retrains.png"
    fig.tight_layout(); fig.savefig(out, dpi=130); plt.close(fig)
    return out


# ─── 5. Forgetting / adaptation ──────────────────────────────────────────────

def plot_forgetting() -> Optional[Path]:
    p = _res_dir() / "metrics_forgetting.csv"
    if not p.exists():
        return None
    fg = pd.read_csv(p)
    x = np.arange(len(fg)); w = 0.38
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(x - w / 2, fg["forgetting_mase_base_era"], w,
           label="base-era forgetting (MASE +)", color="#c44e52")
    ax.bar(x + w / 2, fg["adaptation_mase_walk_era"], w,
           label="walk-era adaptation (MASE -)", color="#55a868")
    ax.axhline(0, color="black", lw=0.8)
    ax.set_xticks(x); ax.set_xticklabels(fg["arm"])
    ax.set_title("Forgetting vs adaptation (final model on fixed probe windows)")
    ax.set_ylabel("MASE delta vs frozen base"); ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    out = _plot_dir() / "forgetting.png"
    fig.tight_layout(); fig.savefig(out, dpi=130); plt.close(fig)
    return out


# ─── Orchestration ───────────────────────────────────────────────────────────

def generate_all_plots() -> Dict[str, str]:
    arms = _arms_present()
    if not arms:
        console.print("[yellow]No drift streams found; run the arms first.[/yellow]")
        return {}
    console.print(f"[bold]Phase 6[/bold]: rendering plots for arms {arms}...")
    out = {
        "error_timeline":  str(plot_error_timeline(arms)),
        "profit_timeline": str(plot_profit_timeline(arms)),
        "retrain_counts":  str(plot_retrain_counts(arms)),
    }
    sc = plot_accuracy_vs_retrains()
    if sc: out["accuracy_vs_retrains"] = str(sc)
    fg = plot_forgetting()
    if fg: out["forgetting"] = str(fg)
    console.print(f"[green]✓ Saved {len(out)} figures[/green] -> {_plot_dir()}")
    return out
