"""Data access for the dashboard.

Reads the on-disk artifacts of the two pipelines directly (CSV / JSON) — no import
of the training code, so there is no torch / GPU dependency. The ``periodic`` arm is
dropped everywhere: it is no longer part of the ablation.

  initial pipeline -> outputs/results/        (task-based CL; train x eval matrix)
  drift   pipeline -> outputs/drift/results/  (drift-triggered; weekly walk stream)
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
INITIAL_DIR = ROOT / "outputs" / "results"
DRIFT_DIR = ROOT / "outputs" / "drift" / "results"

DROP_ARMS = {"periodic"}

# Task sequence (initial pipeline) — alternating regimes, by design.
TASKS = [
    (1, "Baseline 2023 H1", "baseline"),
    (2, "Mega-sale 2023",   "mega_sale"),
    (3, "Baseline 2024 H1", "baseline"),
    (4, "Mega-sale 2024",   "mega_sale"),
    (5, "Baseline 2025 H1", "baseline"),
    (6, "Mega-sale 2025",   "mega_sale"),
]
TASK_SHORT = {1: "T1", 2: "T2", 3: "T3", 4: "T4", 5: "T5", 6: "T6"}
MEGA_SALE_TASKS = [2, 4, 6]


# ── Initial pipeline ─────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def load_initial() -> pd.DataFrame:
    """Long-form metrics: model_type, cl_method, train/eval task, metric, value."""
    fp = INITIAL_DIR / "all_metrics.csv"
    if not fp.exists():
        return pd.DataFrame()
    df = pd.read_csv(fp)
    df = df[~df["cl_method"].isin(DROP_ARMS)].copy()
    for c in ("train_task_id", "eval_task_id"):
        df[c] = df[c].astype(int)
    df["metric_value"] = pd.to_numeric(df["metric_value"], errors="coerce")
    return df


def initial_methods(df: pd.DataFrame, model_type: str, metric: str) -> list[str]:
    sub = df[(df["model_type"] == model_type) & (df["metric_name"] == metric)]
    return sorted(sub["cl_method"].unique().tolist())


def matrix(df: pd.DataFrame, method: str, model_type: str, metric: str) -> pd.DataFrame:
    """train_task_id (rows) x eval_task_id (cols) value matrix for one method."""
    sub = df[(df["cl_method"] == method) & (df["model_type"] == model_type)
             & (df["metric_name"] == metric)]
    if sub.empty:
        return pd.DataFrame()
    m = sub.pivot_table(index="train_task_id", columns="eval_task_id",
                        values="metric_value", aggfunc="mean")
    return m.reindex(index=range(1, 7), columns=range(1, 7))


def online_series(df: pd.DataFrame, model_type: str, metric: str) -> pd.DataFrame:
    """Diagonal (train==eval) per method across tasks: index=task, cols=method."""
    sub = df[(df["model_type"] == model_type) & (df["metric_name"] == metric)
             & (df["train_task_id"] == df["eval_task_id"])]
    if sub.empty:
        return pd.DataFrame()
    return sub.pivot_table(index="eval_task_id", columns="cl_method",
                           values="metric_value", aggfunc="mean")


def retention_series(df: pd.DataFrame, eval_task: int, model_type: str,
                     metric: str) -> pd.DataFrame:
    """Fixed eval task scored as training walks forward: index=train task, cols=method."""
    sub = df[(df["model_type"] == model_type) & (df["metric_name"] == metric)
             & (df["eval_task_id"] == eval_task)
             & (df["train_task_id"] >= eval_task)]
    if sub.empty:
        return pd.DataFrame()
    return sub.pivot_table(index="train_task_id", columns="cl_method",
                           values="metric_value", aggfunc="mean")


@st.cache_data(show_spinner=False)
def initial_summary(model_type: str = "forecasting",
                    metric: str = "mase") -> pd.DataFrame:
    """Per-method leaderboard: mean online error and a simple forgetting score
    (final-model error on a task minus that task's online error, averaged)."""
    df = load_initial()
    rows = []
    for method in initial_methods(df, model_type, metric):
        m = matrix(df, method, model_type, metric)
        if m.empty:
            continue
        diag = pd.Series({t: m.loc[t, t] for t in range(1, 7) if t in m.index and t in m.columns})
        final = 6
        forget = []
        if final in m.index:
            for t in range(1, final):
                if t in m.columns and t in diag.index:
                    a, b = m.loc[final, t], diag.get(t)
                    if pd.notna(a) and pd.notna(b):
                        forget.append(a - b)
        rows.append({
            "method": method,
            "online_mean": diag.mean(skipna=True),
            "final_task": m.loc[final].mean(skipna=True) if final in m.index else float("nan"),
            "forgetting": (sum(forget) / len(forget)) if forget else float("nan"),
        })
    return pd.DataFrame(rows).set_index("method") if rows else pd.DataFrame()


# ── Drift pipeline ───────────────────────────────────────────────────────────
DRIFT_ARMS = ["frozen", "ewc", "replay", "sdft"]


@st.cache_data(show_spinner=False)
def load_drift_streams() -> pd.DataFrame:
    """Concatenated weekly walk streams, one row per (date, arm)."""
    frames = []
    for arm in DRIFT_ARMS:
        fp = DRIFT_DIR / f"drift_stream_{arm}.csv"
        if not fp.exists():
            continue
        d = pd.read_csv(fp, parse_dates=["date"])
        frames.append(d)
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True)
    return out[~out["arm"].isin(DROP_ARMS)]


@st.cache_data(show_spinner=False)
def load_drift_events() -> pd.DataFrame:
    """Retrain events across arms: date, arm, model (forecasting/rl), window_rows."""
    rows = []
    for arm in DRIFT_ARMS:
        fp = DRIFT_DIR / f"retrain_log_{arm}.json"
        if not fp.exists():
            continue
        obj = json.loads(fp.read_text())
        for ev in obj.get("events", []):
            rows.append({
                "date": pd.Timestamp(ev["date"]),
                "arm": arm,
                "model": ev.get("model"),
                "window_rows": ev.get("window_rows"),
            })
    return pd.DataFrame(rows)


@st.cache_data(show_spinner=False)
def load_drift_efficiency() -> pd.DataFrame:
    fp = DRIFT_DIR / "metrics_efficiency.csv"
    if not fp.exists():
        return pd.DataFrame()
    df = pd.read_csv(fp)
    return df[~df["arm"].isin(DROP_ARMS)].set_index("arm")


@st.cache_data(show_spinner=False)
def load_drift_forgetting() -> pd.DataFrame:
    fp = DRIFT_DIR / "metrics_forgetting.csv"
    if not fp.exists():
        return pd.DataFrame()
    df = pd.read_csv(fp)
    return df[~df["arm"].isin(DROP_ARMS)].set_index("arm")


@st.cache_data(show_spinner=False)
def drift_window() -> tuple[pd.Timestamp, pd.Timestamp]:
    s = load_drift_streams()
    if s.empty:
        return (pd.NaT, pd.NaT)
    return (s["date"].min(), s["date"].max())
