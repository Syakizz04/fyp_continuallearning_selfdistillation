"""Landing page: frame the two experiments and the question they share."""
from __future__ import annotations

import streamlit as st

import loader as L
import theme as T


def render() -> None:
    st.title("Continual learning under regime shift")
    st.caption(
        "Catastrophic forgetting in demand forecasting and dynamic pricing, compared "
        "across two continual-learning designs. Proposed method: SDFT (self-"
        "distillation). Reference methods: Replay and EWC."
    )

    df = L.load_initial()
    eff = L.load_drift_efficiency()
    lo, hi = L.drift_window()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Pipelines", "2", "task-based · drift-triggered", delta_color="off",
              help="Two continual-learning designs compared: the task-based synthetic "
                   "pipeline and the drift-triggered M5 pipeline.")
    if not df.empty:
        c2.metric("Synthetic-pipeline records", f"{len(df):,}",
                  help="Number of individual metric measurements logged across all "
                       "methods, tasks and metrics in the synthetic pipeline.")
    if not eff.empty:
        adapt = eff[eff.index != "frozen"]
        if not adapt.empty:
            best = adapt["walk_mase_mean"].idxmin()
            c3.metric("Best drift walk MASE", f"{eff.loc[best, 'walk_mase_mean']:.3f}",
                      T.label_of(best), delta_color="off",
                      help="Lowest mean forecast error (MASE) over the weekly walk, "
                           "among the adaptive methods. Lower is better.")
    if lo is not None and hi is not None and not L.load_drift_streams().empty:
        c4.metric("Drift walk window", f"{lo:%Y}–{hi:%Y}", "weekly", delta_color="off",
                  help="Date span of the drift pipeline's forward walk, evaluated "
                       "once per week.")

    st.divider()

    a, b = st.columns(2)
    with a:
        st.subheader("Synthetic pipeline")
        st.markdown(
            "Trained on **synthetic data**. Six tasks trained in sequence, regimes alternating baseline and "
            "mega-sale. Forgetting is measured from the full train-task by eval-task "
            "matrix: the diagonal is online accuracy, the lower triangle is retention "
            "of earlier tasks. Anchor method is Naive fine-tuning."
        )
        st.markdown(
            "Methods: **Naive** · **EWC** · **Replay** · **SDFT**."
        )
    with b:
        st.subheader("Drift pipeline")
        st.markdown(
            "Trained on a **subset of the real M5 (Walmart) sales data** — FOODS items "
            "in store CA_1. One base model is walked forward through 2013–2015. A drift "
            "detector watches forecast error and pricing profit; each strategy retrains "
            "only when drift fires. Cost is the number of retrains. Anchor method is "
            "Frozen (never retrains)."
        )
        st.markdown(
            "Arms: **Frozen** · **EWC** · **Replay** · **SDFT**."
        )

    st.divider()
    sc = T.STRATEGY_COLOR
    st.markdown(
        "**Reading the colours.** One fixed colour per strategy across every chart and "
        "both pipelines — "
        f"<span style='color:{sc['sdft']};font-weight:600'>SDFT</span> teal, "
        f"<span style='color:{sc['replay']};font-weight:600'>Replay</span> slate-blue, "
        f"<span style='color:{sc['ewc']};font-weight:600'>EWC</span> bronze, "
        f"<span style='color:{sc['naive']};font-weight:600'>baselines</span> grey. ",
        unsafe_allow_html=True,
    )
