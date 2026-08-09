"""
Model Health: what each node is currently serving, and when it decided to relearn.

The system retrains on **detected** drift rather than on a schedule, so the two
things worth seeing are the detector's margin (how close the error is running to
its trigger threshold) and what firing actually cost. The second is where the
project's premise gets tested: replay-free CL is motivated by edge memory limits,
so the footprint panel is evidence, not decoration.

Thresholds are calibrated, not chosen: mu + k*sigma of the base model's error on
a held-out tail. That is why the threshold line is drawn from the data rather
than hardcoded.
"""
from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import live as LV
import theme as T

MB = 1024 * 1024


# ── Served models ────────────────────────────────────────────────────────────

def _node_cards() -> bool:
    """Live node status. Returns whether any node answered."""
    nodes = LV.live_nodes() if LV.http_available() else pd.DataFrame()
    if nodes.empty or not nodes.get("up", pd.Series(dtype=bool)).any():
        return False
    cols = st.columns(len(nodes))
    for col, (_, r) in zip(cols, nodes.iterrows()):
        with col:
            st.markdown(f"**{r['node']}**")
            if not r.get("up"):
                st.caption("offline")
                continue
            st.metric("Model generation", r.get("generation") or "—")
            st.caption(
                f"strategy `{r.get('strategy') or '—'}` · "
                f"loaded in {r.get('load_seconds') or float('nan'):.1f}s · "
                f"{int(r.get('n_retrains') or 0)} retrains")
            if r.get("retrain_in_flight"):
                st.warning("retraining now — old model still serving",
                           icon=":material/autorenew:")
    return True


def _registry_panel() -> None:
    models = LV.registry_models()
    nodes = LV.registry_nodes()
    if models.empty and nodes.empty:
        st.caption("Control-plane registry is empty — no node has registered a "
                   "model version yet.")
        return
    if not nodes.empty:
        st.markdown("**Registered nodes**")
        st.dataframe(nodes, width="stretch", hide_index=True)
    if not models.empty:
        st.markdown("**Model versions**")
        st.dataframe(models.head(50), width="stretch", hide_index=True)


# ── Drift vs threshold ───────────────────────────────────────────────────────

def _drift_fig(arms: list[str]) -> go.Figure:
    """Windowed forecast error per arm, with its calibrated threshold and the
    checks where a retrain actually fired."""
    fig = go.Figure()
    drawn = False
    labels = []
    for arm in arms:
        s = LV.drift_stream(arm)
        if s.empty or "mase" not in s.columns:
            continue
        d = s.sort_values("date")
        col = T.color_of(arm)
        fig.add_trace(go.Scatter(
            x=d["date"], y=d["mase"], mode="lines",
            line=dict(color=col, width=T.width_of(arm)),
            hovertemplate=f"{T.label_of(arm)} · %{{x|%Y-%m-%d}}<br>"
                          "MASE %{y:.3f}<extra></extra>"))
        labels.append((d["date"].iloc[-1], d["mase"].iloc[-1], T.label_of(arm), col))
        drawn = True

        rt = LV.retrain_log(arm)
        fc = rt[rt["model"] == "forecasting"] if not rt.empty else rt
        if not fc.empty:
            pts = d[d["date"].isin(fc["date"])]
            if not pts.empty:
                fig.add_trace(go.Scatter(
                    x=pts["date"], y=pts["mase"], mode="markers",
                    marker=dict(symbol=T.marker_of(arm), size=12, color=col,
                                line=dict(width=1.5, color="#FFFFFF")),
                    hovertemplate=f"{T.label_of(arm)} retrained · "
                                  "%{x|%Y-%m-%d}<extra></extra>"))
    if drawn:
        # The threshold is logged with every check, so it is read back rather
        # than recomputed - a drawn line that disagreed with the one the
        # detector used would be worse than no line.
        s0 = LV.drift_stream(arms[0])
        if "fc_threshold" in s0.columns and s0["fc_threshold"].notna().any():
            thr = float(s0["fc_threshold"].dropna().iloc[0])
            fig.add_hline(y=thr, line=dict(color=T.ALERT, width=1.3, dash="dash"),
                          opacity=0.8)
            fig.add_annotation(x=s0["date"].max(), y=thr, text="trigger threshold",
                               showarrow=False, xanchor="right", yshift=12,
                               font=dict(family=T.FONT, size=13, color=T.ALERT))
        T.right_labels(fig, labels)
    fig.update_xaxes(title="Walk-forward date", dtick="M6", tickformat="%Y-%m")
    fig.update_yaxes(title="Windowed MASE")
    return T.style(fig, height=480, title="Forecast error against the trigger threshold")


def _retrain_fig(arms: list[str]) -> go.Figure:
    """When each arm decided to relearn. Cadence is the story, not the count."""
    fig = go.Figure()
    rows = [LV.retrain_log(a) for a in arms]
    rows = [r for r in rows if not r.empty]
    if not rows:
        return T.style(fig, height=300)
    ev = pd.concat(rows, ignore_index=True)
    order = sorted(ev["arm"].unique(), key=T.order_key)
    for i, arm in enumerate(order):
        d = ev[ev["arm"] == arm]
        for model, symbol in (("forecasting", "circle"), ("rl", "square")):
            sub = d[d["model"] == model]
            if sub.empty:
                continue
            fig.add_trace(go.Scatter(
                x=sub["date"], y=[i] * len(sub), mode="markers",
                marker=dict(symbol=symbol, size=12, color=T.color_of(arm),
                            line=dict(width=1.2, color="#FFFFFF")),
                hovertemplate=f"{T.label_of(arm)} · {model}<br>"
                              "%{x|%Y-%m-%d}<extra></extra>"))
    fig.update_yaxes(tickmode="array", tickvals=list(range(len(order))),
                     ticktext=[T.label_of(a) for a in order], title=None)
    fig.update_xaxes(title="Walk-forward date", dtick="M6", tickformat="%Y-%m")
    return T.style(fig, height=300,
                   title="Retrain events  ·  circle = forecaster, square = pricer")


# ── Memory footprint (E4) ────────────────────────────────────────────────────

def _memory_fig(arms: list[str]) -> tuple[go.Figure, pd.DataFrame]:
    """What each CL mechanism costs in RAM as the stream lengthens."""
    fig = go.Figure()
    summary = []
    labels = []
    for arm in arms:
        m = LV.memory_log(arm)
        if m.empty or "component" not in m.columns:
            continue
        tot = m[m["component"] == "cl_state_total"].sort_values("event_idx")
        if tot.empty:
            continue
        col = T.color_of(arm)
        fig.add_trace(go.Scatter(
            x=tot["event_idx"], y=tot["mb"], mode="lines",
            line=dict(color=col, width=T.width_of(arm)),
            hovertemplate=f"{T.label_of(arm)} · event %{{x}}<br>"
                          "%{y:.1f} MB<extra></extra>"))
        labels.append((tot["event_idx"].iloc[-1], tot["mb"].iloc[-1],
                       T.label_of(arm), col))
        summary.append({"arm": T.label_of(arm),
                        "peak_mb": float(tot["mb"].max()),
                        "final_mb": float(tot["mb"].iloc[-1])})
    if labels:
        T.right_labels(fig, labels)
    fig.update_xaxes(title="Retrain event")
    fig.update_yaxes(title="CL state held (MB)")
    return (T.style(fig, height=380, title="What each method keeps in memory"),
            pd.DataFrame(summary))


# ── Page ─────────────────────────────────────────────────────────────────────

def render() -> None:
    st.title("Model health")
    st.caption(
        "Each node runs its own forecaster and pricer in-process, watches its own "
        "error, and retrains locally when drift fires. Nothing but inventory state "
        "and model metadata leaves the node — no raw history is ever shipped."
    )

    st.subheader("Currently served")
    if not _node_cards():
        st.info(
            "**No nodes responding.** The panels below read the recorded walk-"
            "forward runs instead, which is what the drift detector and the "
            "retrain logic actually did over 2013–2015.",
            icon=":material/history:")
    st.divider()

    arms = LV.drift_arms()
    if not arms:
        st.warning("No drift streams under `outputs/drift/results/`. Run "
                   "`python -m experiments.exp_staleness_cl` to produce them.")
        return

    with st.sidebar:
        st.markdown("### Arms")
        picked = st.multiselect(
            "Strategies", arms, default=arms,
            format_func=T.label_of,
            help="Each arm is an independent walk with its own trigger history.")
    picked = sorted(picked or arms, key=T.order_key)

    st.plotly_chart(_drift_fig(picked), width="stretch",
                    config={"displayModeBar": False})
    st.caption(
        "The threshold is **calibrated, not chosen**: mu + k·sigma of the base "
        "model's error on a held-out tail of its own training period. A retrain "
        "fires only after the error stays above it for consecutive checks, so a "
        "single noisy week cannot trigger one."
    )

    st.divider()
    st.plotly_chart(_retrain_fig(picked), width="stretch",
                    config={"displayModeBar": False})

    st.divider()
    st.subheader("Memory footprint")
    fig, summary = _memory_fig(picked)
    if summary.empty:
        st.caption(
            "No `memory_*.csv` yet — E4's footprint is recorded by the same runs "
            "that produce the walk, so it appears once a sweep has been run."
        )
    else:
        left, right = st.columns([3, 2])
        with left:
            st.plotly_chart(fig, width="stretch",
                            config={"displayModeBar": False})
        with right:
            st.dataframe(summary.set_index("arm").round(1),
                         width="stretch")
            st.caption(
                "The project's case for replay-free CL rests on edge memory "
                "limits, and this is the measurement behind it: a replay buffer "
                "grows with the stream, while a distillation teacher is one "
                "model copy and stays flat no matter how long the node runs."
            )

    st.divider()
    st.subheader("Control plane registry")
    _registry_panel()
