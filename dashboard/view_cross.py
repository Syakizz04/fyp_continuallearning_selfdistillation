"""Cross-pipeline page: the Replay vs SDFT ablation under both CL regimes.

The two pipelines use different anchors (Naive for initial, Frozen for drift) and
different axes, so this is a curated side-by-side, not a merged table. The point is
whether the ranking of Replay against SDFT holds across task-based and
drift-triggered continual learning.
"""
from __future__ import annotations

import plotly.graph_objects as go
import streamlit as st

import loader as L
import theme as T

PAIR = ["replay", "sdft"]


def _initial_online() -> go.Figure:
    piv = L.online_series(L.load_initial(), "forecasting", "mase")
    fig = go.Figure()
    if piv.empty:
        return T.style(fig, height=420)
    labels = []
    for method in [m for m in PAIR if m in piv.columns]:
        y = piv[method]
        T.line(fig, list(piv.index), list(y), method,
               hovertemplate=f"{T.label_of(method)} · task %{{x}}<br>MASE %{{y:.3f}}<extra></extra>")
        last = y.dropna()
        if not last.empty:
            labels.append((last.index[-1], last.iloc[-1], T.label_of(method),
                           T.color_of(method)))
    T.right_labels(fig, labels)
    fig.update_xaxes(tickmode="array", tickvals=list(piv.index),
                     ticktext=[L.TASK_SHORT[t] for t in piv.index], title="Task")
    fig.update_yaxes(title="MASE")
    return T.style(fig, height=420, title="Task-based CL — online MASE")


def _drift_walk() -> go.Figure:
    s = L.load_drift_streams()
    fig = go.Figure()
    if s.empty:
        return T.style(fig, height=420)
    labels = []
    for arm in [a for a in PAIR if a in s["arm"].unique()]:
        d = s[s["arm"] == arm].sort_values("date")
        T.line(fig, d["date"], d["mase"], arm,
               hovertemplate=f"{T.label_of(arm)} · %{{x|%Y-%m-%d}}<br>MASE %{{y:.3f}}<extra></extra>")
        labels.append((d["date"].iloc[-1], d["mase"].iloc[-1], T.label_of(arm),
                       T.color_of(arm)))
    T.right_labels(fig, labels)
    fig.update_xaxes(title="Walk-forward date", dtick="M6", tickformat="%Y-%m")
    fig.update_yaxes(title="MASE")
    return T.style(fig, height=420, title="Drift-triggered CL — walk MASE")


def _initial_rl() -> go.Figure:
    piv = L.online_series(L.load_initial(), "rl", "cumulative_profit")
    fig = go.Figure()
    if piv.empty or "naive" not in piv.columns:
        return T.style(fig, height=420)
    idx = piv.div(piv["naive"], axis=0)        # index relative to Naive per task
    fig.add_hline(y=1.0, line=dict(color=T.ALERT, width=1.3, dash="dash"), opacity=0.7)
    labels = []
    for method in [m for m in PAIR if m in idx.columns]:
        y = idx[method]
        T.line(fig, list(idx.index), list(y), method,
               hovertemplate=f"{T.label_of(method)} · task %{{x}}<br>index %{{y:.3f}}<extra></extra>")
        last = y.dropna()
        if not last.empty:
            labels.append((last.index[-1], last.iloc[-1], T.label_of(method),
                           T.color_of(method)))
    T.right_labels(fig, labels)
    fig.update_xaxes(tickmode="array", tickvals=list(idx.index),
                     ticktext=[L.TASK_SHORT[t] for t in idx.index], title="Task")
    fig.update_yaxes(title="Profit index vs Naive")
    return T.style(fig, height=420, title="Task-based CL — profit index")


def _drift_profit() -> go.Figure:
    s = L.load_drift_streams()
    fig = go.Figure()
    if s.empty:
        return T.style(fig, height=420)
    fig.add_hline(y=1.0, line=dict(color=T.ALERT, width=1.3, dash="dash"), opacity=0.7)
    labels = []
    for arm in [a for a in PAIR if a in s["arm"].unique()]:
        d = s[s["arm"] == arm].sort_values("date").copy()
        d["sm"] = d["profit_index"].rolling(4, min_periods=1).mean()
        T.line(fig, d["date"], d["sm"], arm,
               hovertemplate=f"{T.label_of(arm)} · %{{x|%Y-%m-%d}}<br>index %{{y:.3f}}<extra></extra>")
        labels.append((d["date"].iloc[-1], d["sm"].iloc[-1], T.label_of(arm),
                       T.color_of(arm)))
    T.right_labels(fig, labels)
    fig.update_xaxes(title="Walk-forward date", dtick="M6", tickformat="%Y-%m")
    fig.update_yaxes(title="Profit index (4-week mean)")
    return T.style(fig, height=420, title="Drift-triggered CL — profit index")


def render() -> None:
    st.header("Replay vs SDFT — the ablation across both regimes")
    st.caption(
        "Same two methods, two continual-learning paradigms. Colours are fixed: "
        f"{T.label_of('replay')} slate-blue, {T.label_of('sdft')} teal."
    )

    eff = L.load_drift_efficiency()
    summ = L.initial_summary("forecasting", "mase")

    # KPI tiles: headline numbers for each method, one regime each
    cols = st.columns(4)
    if not summ.empty:
        for i, m in enumerate(PAIR):
            if m in summ.index:
                cols[i].metric(f"{T.label_of(m)} — task-based online MASE",
                               f"{summ.loc[m, 'online_mean']:.3f}", delta_color="off",
                               help="Mean online forecast error across the six tasks "
                                    "in the synthetic pipeline. Lower is better.")
    if not eff.empty:
        for j, m in enumerate(PAIR):
            if m in eff.index:
                cols[2 + j].metric(f"{T.label_of(m)} — drift walk MASE",
                                   f"{eff.loc[m, 'walk_mase_mean']:.3f}",
                                   f"{int(eff.loc[m, 'n_retrains_total'])} retrains",
                                   delta_color="off",
                                   help="Mean forecast error over the weekly walk in "
                                        "the drift pipeline, with total retrains shown "
                                        "below. Lower error, fewer retrains is better.")

    st.divider()

    st.subheader("Forecasting")
    a, b = st.columns(2)
    with a:
        st.plotly_chart(_initial_online(), width="stretch",
                        config={"displayModeBar": False})
        st.caption("Synthetic pipeline. Anchor: Naive.")
    with b:
        st.plotly_chart(_drift_walk(), width="stretch",
                        config={"displayModeBar": False})
        st.caption("Drift pipeline. Anchor: Frozen.")

    st.divider()

    st.subheader("Dynamic pricing")
    c, d = st.columns(2)
    with c:
        st.plotly_chart(_initial_rl(), width="stretch",
                        config={"displayModeBar": False})
        st.caption("Synthetic pipeline. Profit index vs Naive per task; red dashed = "
                   "parity with Naive.")
    with d:
        st.plotly_chart(_drift_profit(), width="stretch",
                        config={"displayModeBar": False})
        st.caption("Drift pipeline. Profit index vs frozen; red dashed = break-even.")

    st.divider()

    st.subheader("Side-by-side summary")
    rows = []
    for m in PAIR:
        row = {"Method": T.label_of(m)}
        if not summ.empty and m in summ.index:
            row["Task-based online MASE"] = summ.loc[m, "online_mean"]
            row["Task-based forgetting ΔMASE"] = summ.loc[m, "forgetting"]
        if not eff.empty and m in eff.index:
            row["Drift walk MASE"] = eff.loc[m, "walk_mase_mean"]
            row["Drift profit index"] = eff.loc[m, "profit_index_vs_frozen"]
            row["Drift retrains"] = int(eff.loc[m, "n_retrains_total"])
            row["Drift base-era forgetting"] = eff.loc[m, "forgetting_mase_base_era"]
        rows.append(row)
    import pandas as pd
    tbl = pd.DataFrame(rows).set_index("Method")
    st.dataframe(tbl.style.format({
        "Task-based online MASE": "{:.3f}",
        "Task-based forgetting ΔMASE": "{:+.3f}",
        "Drift walk MASE": "{:.3f}",
        "Drift profit index": "{:.3f}",
        "Drift base-era forgetting": "{:+.3f}",
    }), width="stretch")
    st.caption("Lower error and more-negative forgetting are better; fewer retrains "
               "is cheaper. The ranking is consistent if one method wins on both "
               "regimes.")
