"""Initial pipeline page: task-based continual learning over six alternating tasks.

Hero is the forgetting matrix (train task x eval task). Supporting panels show the
online error across the task sequence and the retention of an early task.
"""
from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
import streamlit as st

import loader as L
import theme as T


def _heatmap(method: str, metric: str) -> go.Figure:
    m = L.matrix(L.load_initial(), method, "forecasting", metric)
    fig = go.Figure()
    if m.empty:
        return T.style(fig, height=520)
    z = m.values.astype(float)
    text = np.where(np.isnan(z), "", np.vectorize(lambda v: f"{v:.2f}")(z))
    fig.add_trace(go.Heatmap(
        z=z,
        x=[L.TASK_SHORT[c] for c in m.columns],
        y=[L.TASK_SHORT[r] for r in m.index],
        colorscale=T.TEAL_SCALE,
        colorbar=dict(title=dict(text=metric.upper(), font=dict(family=T.FONT, size=13)),
                      thickness=14, outlinecolor=T.INK, outlinewidth=1, len=0.9),
        zmin=np.nanmin(z), zmax=np.nanmax(z),
        text=text, texttemplate="%{text}",
        textfont=dict(family=T.MONO, size=14, color=T.INK),
        hovertemplate="train %{y} · eval %{x}<br>" + metric.upper() + " %{z:.3f}<extra></extra>",
        xgap=2, ygap=2,
    ))
    fig.update_layout(
        xaxis=dict(title="Evaluated on task", side="bottom"),
        yaxis=dict(title="After training task", autorange="reversed"),
        margin=dict(l=110, r=40, t=64, b=56),
    )
    return T.style(fig, height=520,
                   title=f"Forgetting matrix — {T.label_of(method)} ({metric.upper()})")


def _online(metric: str) -> go.Figure:
    piv = L.online_series(L.load_initial(), "forecasting", metric)
    fig = go.Figure()
    if piv.empty:
        return T.style(fig, height=380)
    for t in L.MEGA_SALE_TASKS:                       # shade mega-sale regimes
        fig.add_vrect(x0=t - 0.5, x1=t + 0.5, fillcolor="#EFEDE6",
                      line_width=0, layer="below")
    labels = []
    for method in sorted(piv.columns, key=T.order_key):
        y = piv[method]
        T.line(fig, list(piv.index), list(y), method,
               hovertemplate=f"{T.label_of(method)} · task %{{x}}<br>"
                             + metric.upper() + " %{y:.3f}<extra></extra>")
        last = y.dropna()
        if not last.empty:
            labels.append((last.index[-1], last.iloc[-1], T.label_of(method),
                           T.color_of(method)))
    T.right_labels(fig, labels)
    fig.update_xaxes(tickmode="array", tickvals=list(piv.index),
                     ticktext=[L.TASK_SHORT[t] for t in piv.index], title="Task")
    fig.update_yaxes(title=metric.upper())
    return T.style(fig, height=380, title=f"Online error across the task sequence")


def _retention(eval_task: int, metric: str) -> go.Figure:
    piv = L.retention_series(L.load_initial(), eval_task, "forecasting", metric)
    fig = go.Figure()
    if piv.empty:
        return T.style(fig, height=380)
    labels = []
    for method in sorted(piv.columns, key=T.order_key):
        y = piv[method]
        T.line(fig, list(piv.index), list(y), method,
               hovertemplate=f"{T.label_of(method)} · after task %{{x}}<br>"
                             + metric.upper() + " %{y:.3f}<extra></extra>")
        last = y.dropna()
        if not last.empty:
            labels.append((last.index[-1], last.iloc[-1], T.label_of(method),
                           T.color_of(method)))
    T.right_labels(fig, labels)
    fig.update_xaxes(tickmode="array", tickvals=list(piv.index),
                     ticktext=[L.TASK_SHORT[t] for t in piv.index],
                     title="Model after training task")
    fig.update_yaxes(title=metric.upper())
    return T.style(fig, height=380,
                   title=f"Retention of task {eval_task} as training continues")


def _rl_profit_index() -> go.Figure:
    """Online pricing profit per task, indexed to Naive fine-tuning (= 1.0)."""
    piv = L.online_series(L.load_initial(), "rl", "cumulative_profit")
    fig = go.Figure()
    if piv.empty or "naive" not in piv.columns:
        return T.style(fig, height=360)
    idx = piv.div(piv["naive"], axis=0)        # index relative to Naive per task
    fig.add_hline(y=1.0, line=dict(color=T.ALERT, width=1.3, dash="dash"), opacity=0.7)
    labels = []
    for method in sorted([c for c in idx.columns if c != "naive"], key=T.order_key):
        y = idx[method]
        T.line(fig, list(idx.index), list(y), method,
               hovertemplate=f"{T.label_of(method)} · task %{{x}}<br>"
                             "index %{y:.3f}<extra></extra>")
        last = y.dropna()
        if not last.empty:
            labels.append((last.index[-1], last.iloc[-1], T.label_of(method),
                           T.color_of(method)))
    T.right_labels(fig, labels)
    fig.update_xaxes(tickmode="array", tickvals=list(idx.index),
                     ticktext=[L.TASK_SHORT[t] for t in idx.index], title="Task")
    fig.update_yaxes(title="Profit index vs Naive")
    return T.style(fig, height=360, title="Profit index vs Naive")


def _rl_cumulative() -> go.Figure:
    """Online cumulative pricing profit per task (raw MYR)."""
    piv = L.online_series(L.load_initial(), "rl", "cumulative_profit")
    fig = go.Figure()
    if piv.empty:
        return T.style(fig, height=360)
    labels = []
    for method in sorted(piv.columns, key=T.order_key):
        y = piv[method]
        T.line(fig, list(piv.index), list(y), method,
               hovertemplate=f"{T.label_of(method)} · task %{{x}}<br>"
                             "profit %{y:.0f}<extra></extra>")
        last = y.dropna()
        if not last.empty:
            labels.append((last.index[-1], last.iloc[-1], T.label_of(method),
                           T.color_of(method)))
    T.right_labels(fig, labels)
    fig.update_xaxes(tickmode="array", tickvals=list(piv.index),
                     ticktext=[L.TASK_SHORT[t] for t in piv.index], title="Task")
    fig.update_yaxes(title="Cumulative profit (MYR)")
    return T.style(fig, height=360, title="Cumulative profit (MYR)")


def render() -> None:
    df = L.load_initial()
    st.header("Synthetic pipeline — task-based continual learning")
    st.caption(
        "Six tasks trained in sequence, regimes alternating baseline and mega-sale. "
        "Catastrophic forgetting is read off the matrix: the diagonal is online "
        "performance, the lower triangle is how earlier tasks degrade as training "
        "continues. Anchor method is Naive fine-tuning."
    )

    if df.empty:
        st.info("No synthetic-pipeline results found under outputs/results/.")
        return

    summ = L.initial_summary("forecasting", "mase")
    c1, c2, c3, c4 = st.columns(4)
    if not summ.empty:
        best_on = summ["online_mean"].idxmin()
        best_ret = summ["forgetting"].idxmin()
        c1.metric("Best online MASE",
                  f"{summ.loc[best_on, 'online_mean']:.3f}", T.label_of(best_on),
                  delta_color="off",
                  help="Mean of the matrix diagonal: error on each task right after "
                       "training on it, averaged over the six tasks. Lower is better.")
        c2.metric("Least forgetting (ΔMASE)",
                  f"{summ.loc[best_ret, 'forgetting']:+.3f}", T.label_of(best_ret),
                  delta_color="off",
                  help="Final model's error on earlier tasks minus their online error, "
                       "averaged. Positive = forgot; negative = backward transfer.")
        c3.metric("Methods compared", f"{summ.shape[0]:d}",
                  help="CL methods with forecasting results in this run "
                       "(Naive, EWC, Replay, SDFT).")
    c4.metric("Tasks", "6", "3 baseline · 3 mega-sale", delta_color="off",
              help="The six-task sequence; regimes alternate baseline and mega-sale "
                   "by design, so forgetting is observable.")

    st.divider()

    # Hero — forgetting matrix
    methods = L.initial_methods(df, "forecasting", "mase")
    default = "sdft" if "sdft" in methods else methods[0]
    left, right = st.columns([5, 1])
    with right:
        st.write("")
        method = st.radio("Method", methods, index=methods.index(default),
                          format_func=T.label_of, key="init_method")
        metric = st.radio("Error metric", ["mase", "smape"], index=0,
                          format_func=str.upper, key="init_metric")
    with left:
        st.plotly_chart(_heatmap(method, metric), width="stretch",
                        config={"displayModeBar": False})
    st.caption(
        "Read down a column to see one task's error grow as later tasks overwrite it. "
        "Lower (lighter) is better."
    )

    st.divider()

    # Supporting panels — unequal split
    a, b = st.columns([3, 2])
    with a:
        st.plotly_chart(_online(metric), width="stretch",
                        config={"displayModeBar": False})
        st.caption("Diagonal of the matrix. Shaded bands mark mega-sale regimes.")
    with b:
        eval_task = st.selectbox("Track retention of", [1, 2, 3, 4, 5],
                                 format_func=lambda t: f"Task {t} — {dict((i,n) for i,n,_ in L.TASKS)[t]}",
                                 key="init_retention")
        st.plotly_chart(_retention(eval_task, metric), width="stretch",
                        config={"displayModeBar": False})

    st.divider()

    st.subheader("Dynamic pricing (PPO)")
    rl_methods = L.initial_methods(df, "rl", "cumulative_profit")
    if not rl_methods:
        st.caption("No RL metrics recorded for this run.")
    else:
        view = st.radio("Pricing metric", ["Profit index", "Cumulative profit"],
                        index=0, horizontal=True, key="init_rl_metric")
        if view == "Profit index":
            st.plotly_chart(_rl_profit_index(), width="stretch",
                            config={"displayModeBar": False})
            st.caption("Online pricing profit per task relative to Naive fine-tuning; "
                       "red dashed = parity with Naive (below it = worse than Naive).")
        else:
            st.plotly_chart(_rl_cumulative(), width="stretch",
                            config={"displayModeBar": False})
            st.caption("Online cumulative pricing profit per task (raw MYR).")
