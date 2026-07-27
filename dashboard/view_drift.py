"""Drift pipeline page: drift-triggered continual learning on a weekly walk.

One base model is walked forward through 2013-2015; each strategy retrains only when
drift fires. Hero is the forecast-error timeline with retrain markers. Supporting
panels: profit index over the walk and the accuracy-vs-retrains trade-off.
"""
from __future__ import annotations

import plotly.graph_objects as go
import streamlit as st

import loader as L
import theme as T


def _error_timeline() -> go.Figure:
    s = L.load_drift_streams()
    ev = L.load_drift_events()
    fig = go.Figure()
    if s.empty:
        return T.style(fig, height=540)

    labels = []
    fc = ev[ev["model"] == "forecasting"] if not ev.empty else ev
    for arm in sorted(s["arm"].unique(), key=T.order_key):
        d = s[s["arm"] == arm].sort_values("date").copy()
        col = T.color_of(arm)
        # Raw weekly series, one bold line per method.
        fig.add_trace(go.Scatter(
            x=d["date"], y=d["mase"], mode="lines",
            line=dict(color=col, width=T.width_of(arm)),
            hovertemplate=f"{T.label_of(arm)} · %{{x|%Y-%m-%d}}<br>"
                          "MASE %{y:.3f}<extra></extra>"))
        labels.append((d["date"].iloc[-1], d["mase"].iloc[-1], T.label_of(arm), col))

        # Forecast retrains, on the line; method colour, one shape per method.
        if not fc.empty:
            pts = d[d["date"].isin(fc[fc["arm"] == arm]["date"])]
            if not pts.empty:
                fig.add_trace(go.Scatter(
                    x=pts["date"], y=pts["mase"], mode="markers",
                    marker=dict(symbol=T.marker_of(arm), size=13, color=col,
                                line=dict(width=1.5, color="#FFFFFF")),
                    hovertemplate=f"{T.label_of(arm)} retrained · "
                                  "%{x|%Y-%m-%d}<extra></extra>"))
    T.right_labels(fig, labels)
    fig.update_xaxes(title="Walk-forward date", dtick="M6", tickformat="%Y-%m")
    fig.update_yaxes(title="MASE")
    return T.style(fig, height=540, title="Forecast error, 2013–2015")


def _profit_timeline() -> go.Figure:
    s = L.load_drift_streams()
    ev = L.load_drift_events()
    fig = go.Figure()
    if s.empty:
        return T.style(fig, height=380)
    # Frozen reference at 1.0
    fig.add_hline(y=1.0, line=dict(color=T.ALERT, width=1.3, dash="dash"), opacity=0.7)
    labels = []
    rl = ev[ev["model"] == "rl"] if not ev.empty else ev
    for arm in sorted(s["arm"].unique(), key=T.order_key):
        if arm == "frozen":
            continue                                  # frozen is the 1.0 reference
        d = s[s["arm"] == arm].sort_values("date").copy()
        d["sm"] = d["profit_index"].rolling(4, min_periods=1).mean()
        col = T.color_of(arm)
        # 4-week rolling mean only — the raw series is too noisy here.
        fig.add_trace(go.Scatter(
            x=d["date"], y=d["sm"], mode="lines",
            line=dict(color=col, width=T.width_of(arm)),
            hovertemplate=f"{T.label_of(arm)} · %{{x|%Y-%m-%d}}<br>"
                          "index (4-wk) %{y:.3f}<extra></extra>"))
        labels.append((d["date"].iloc[-1], d["sm"].iloc[-1], T.label_of(arm), col))

        # Policy retrains, on the trend line; oxblood, one shape per method.
        if not rl.empty:
            pts = d[d["date"].isin(rl[rl["arm"] == arm]["date"])]
            if not pts.empty:
                fig.add_trace(go.Scatter(
                    x=pts["date"], y=pts["sm"], mode="markers",
                    marker=dict(symbol=T.marker_of(arm), size=12, color=col,
                                line=dict(width=1.5, color="#FFFFFF")),
                    hovertemplate=f"{T.label_of(arm)} retrained · "
                                  "%{x|%Y-%m-%d}<extra></extra>"))
    T.right_labels(fig, labels)
    fig.update_xaxes(title="Walk-forward date", dtick="M6", tickformat="%Y-%m")
    fig.update_yaxes(title="Profit index (4-week mean)")
    return T.style(fig, height=540, title="Profit index, 2013–2015")


def _cost_scatter(x_col: str, y_col: str, x_title: str, y_title: str,
                  title: str, ref_line: float | None = None) -> go.Figure:
    """One model's accuracy against its own retraining cost, one point per arm."""
    eff = L.load_drift_efficiency()
    fig = go.Figure()
    if eff.empty:
        return T.style(fig, height=400)
    if ref_line is not None:
        fig.add_hline(y=ref_line, line=dict(color=T.ALERT, width=1.3, dash="dash"),
                      opacity=0.7)
    for arm in sorted(eff.index, key=T.order_key):
        r = eff.loc[arm]
        fig.add_trace(go.Scatter(
            x=[r[x_col]], y=[r[y_col]], mode="markers+text",
            marker=dict(size=18, color=T.color_of(arm),
                        line=dict(width=1.2, color=T.INK)),
            text=[T.label_of(arm)], textposition="middle right",
            textfont=dict(family=T.FONT, size=14, color=T.INK),
            hovertemplate=f"{T.label_of(arm)}<br>{x_title} %{{x}}<br>"
                          f"{y_title} %{{y:.3f}}<extra></extra>",
        ))
    fig.update_xaxes(title=x_title)
    fig.update_yaxes(title=y_title)
    fig.update_layout(margin=dict(r=90))
    return T.style(fig, height=400, title=title)


def _forgetting_bars() -> go.Figure:
    fg = L.load_drift_forgetting()
    fig = go.Figure()
    if fg.empty:
        return T.style(fig, height=360)
    arms = [a for a in sorted(fg.index, key=T.order_key)]
    colors = [T.color_of(a) for a in arms]
    fig.add_trace(go.Bar(
        x=[T.label_of(a) for a in arms],
        y=[fg.loc[a, "forgetting_mase_base_era"] for a in arms],
        marker=dict(color=colors, line=dict(width=1, color=T.INK)),
        hovertemplate="%{x}<br>base-era ΔMASE %{y:+.3f}<extra></extra>",
    ))
    fig.add_hline(y=0.0, line=dict(color=T.INK, width=1))
    fig.update_yaxes(title="Base-era ΔMASE  (− = backward transfer)")
    return T.style(fig, height=360, title="Forgetting on fixed base-era probes")


def render() -> None:
    s = L.load_drift_streams()
    eff = L.load_drift_efficiency()
    st.header("Drift pipeline — drift-triggered continual learning")
    lo, hi = L.drift_window()
    span = "" if s.empty else f"{lo:%Y-%m-%d} to {hi:%Y-%m-%d}, weekly"
    st.caption(
        "A single base model walks forward in time; each strategy retrains only when "
        f"a drift detector fires. Walk window: {span}. Anchor method is Frozen "
        "(never retrains)."
    )
    if s.empty:
        st.info("No drift-pipeline results found under outputs/drift/results/.")
        return

    # KPI tiles from the efficiency table
    if not eff.empty:
        adapt = eff[eff.index != "frozen"]
        best = adapt["walk_mase_mean"].idxmin()
        fewest = adapt["n_retrains_total"].idxmin()
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Best walk MASE", f"{eff.loc[best, 'walk_mase_mean']:.3f}",
                  T.label_of(best), delta_color="off",
                  help="Lowest mean forecast error over the walk, among the adaptive "
                       "arms. Lower is better.")
        c2.metric("Frozen walk MASE", f"{eff.loc['frozen', 'walk_mase_mean']:.3f}",
                  "anchor", delta_color="off",
                  help="The never-retrained baseline. All profit and forgetting "
                       "figures are measured relative to this anchor.")
        c3.metric("Fewest retrains", f"{int(eff.loc[fewest, 'n_retrains_total'])}",
                  T.label_of(fewest), delta_color="off",
                  help="Total drift-triggered retrains (forecast + policy) for the "
                       "most economical adaptive arm. Fewer is cheaper.")
        c4.metric("Weekly checks", f"{len(s[s['arm'] == best]):d}",
                  help="Number of forward-walk evaluation points, one per week.")

    st.divider()

    # Hero — error timeline
    st.plotly_chart(_error_timeline(), width="stretch",
                    config={"displayModeBar": False})
    st.caption("Weekly forecast error per method. Markers flag each method's forecast "
               "retrains — one shape per method (SDFT circle, Replay square, EWC "
               "diamond). Frozen never retrains; the adaptive arms drop sharply after "
               "the 2013-12 retrain.")

    st.divider()

    # Profit index — full width, hero size
    st.plotly_chart(_profit_timeline(), width="stretch",
                    config={"displayModeBar": False})
    st.caption("Pricing profit relative to the frozen policy; red dashed line is "
               "break-even (below it = worse than frozen). Line is the 4-week rolling "
               "mean (raw series omitted); markers flag policy retrains, one shape per "
               "method.")

    st.divider()

    # Accuracy vs retraining cost — forecast and policy split, side by side
    e1, e2 = st.columns(2)
    with e1:
        st.plotly_chart(
            _cost_scatter("n_fc_retrains", "walk_mase_mean", "Forecast retrains",
                          "Mean walk MASE", "Forecast accuracy vs retraining cost"),
            width="stretch", config={"displayModeBar": False})
        st.caption("Lower-left is better: low error, few forecast retrains.")
    with e2:
        st.plotly_chart(
            _cost_scatter("n_rl_retrains", "profit_index_vs_frozen", "Policy retrains",
                          "Profit index vs frozen", "Pricing return vs retraining cost",
                          ref_line=1.0),
            width="stretch", config={"displayModeBar": False})
        st.caption("Upper-left is better: high profit, few policy retrains.")

    st.divider()

    c, d = st.columns([2, 3])
    with c:
        st.plotly_chart(_forgetting_bars(), width="stretch",
                        config={"displayModeBar": False})
    with d:
        st.subheader("Efficiency table")
        cols = {
            "n_fc_retrains": "FC retrains", "n_rl_retrains": "RL retrains",
            "walk_mase_mean": "Walk MASE", "profit_index_vs_frozen": "Profit idx",
            "forgetting_mase_base_era": "Forget (base)",
            "adaptation_mase_walk_era": "Adapt (walk)",
        }
        show = eff[list(cols)].rename(columns=cols).copy()
        show.index = [T.label_of(a) for a in show.index]
        st.dataframe(show.style.format({
            "Walk MASE": "{:.3f}", "Profit idx": "{:.3f}",
            "Forget (base)": "{:+.3f}", "Adapt (walk)": "{:+.3f}",
        }), width="stretch")
        st.caption("Forget(base) below zero is backward transfer; Adapt(walk) is "
                   "improvement on the moving regime.")
