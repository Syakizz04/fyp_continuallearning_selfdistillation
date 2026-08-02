"""
Live Operations: the shared stock pool, and what each channel believes about it.

The hero is **staleness**, not throughput. Correctness is unconditional in this
system - escrow cannot oversell - so the interesting quantity is not whether it
works but what it costs, and the cost is that a node prices and sells against a
view of stock that is systematically behind the truth. That gap is the input E2
studies, so it is the thing this page is built to show.

Staleness is read off the event log rather than recomputed: it was captured at
the moment each decision was made, and it cannot be reconstructed afterwards
because the true figure has moved on by then.
"""
from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import live as LV
import theme as T


# ── Mode + run selection ─────────────────────────────────────────────────────

def _mode_control() -> tuple[str, str | None]:
    """Pick live vs recorded, and which recorded run. Returns (mode, policy)."""
    modes = LV.available_modes()
    probe = LV.probe()

    with st.sidebar:
        st.markdown("### Source")
        if modes == [LV.NONE]:
            st.warning("No live services and no recorded run.")
            return LV.NONE, None
        labels = {LV.LIVE: "Live system", LV.RECORDED: "Recorded run"}
        mode = st.radio("Data source", modes,
                        format_func=lambda m: labels.get(m, m),
                        index=len(modes) - 1 if LV.RECORDED in modes else 0,
                        help="Recorded replays a completed run from its event "
                             "log. Live polls the running services.")
        policy = None
        if mode == LV.RECORDED:
            pol = LV.recorded_policies()
            if pol.empty:
                st.warning("Event log has no runs.")
                return LV.NONE, None
            policy = st.selectbox("Sync policy", pol["policy"].tolist())
            row = pol[pol["policy"] == policy].iloc[0]
            st.caption(f"{int(row['events']):,} events · {int(row['skus'])} SKUs · "
                       f"{int(row['nodes'])} channels\n\n"
                       f"{row['first_date']} → {row['last_date']}")
        else:
            st.caption(f"inventory {'up' if probe['inventory'] else 'down'} · "
                       f"control {'up' if probe['control'] else 'down'}")
            for n, ok in (probe.get("nodes") or {}).items():
                st.caption(f"{'●' if ok else '○'} {n}")
    return mode, policy


# ── Tiles ────────────────────────────────────────────────────────────────────

def _tiles(mode: str, policy: str | None) -> None:
    if mode == LV.LIVE:
        m = LV.live_metrics()
        oversell = m.get("oversell_units")
        rejection = m.get("rejection_rate")
        p50, p99 = m.get("latency_p50_ms"), m.get("latency_p99_ms")
        hops = m.get("roundtrips_per_reserve")
        stale = None
    else:
        runs = LV.recorded_summaries()
        r = runs[runs["policy"] == policy] if not runs.empty else pd.DataFrame()
        # A policy may have several runs (delay sweep); aggregate rather than
        # silently showing whichever happened to be first.
        oversell = int(r["oversell_units"].sum()) if not r.empty else None
        rejection = float(r["rejection_rate"].mean()) if not r.empty else None
        p50 = float(r["latency_p50_ms"].mean()) if not r.empty else None
        p99 = float(r["latency_p99_ms"].mean()) if not r.empty else None
        hops = float(r["roundtrips_per_reserve"].mean()) if not r.empty else None
        s = LV.recorded_staleness(policy)
        stale = float(s["staleness"].mean()) if not s.empty else None

    def fmt(v, spec=".2f", suffix=""):
        return "—" if v is None or pd.isna(v) else f"{v:{spec}}{suffix}"

    c = st.columns(5)
    # Oversell first and always: it is the safety property the whole design
    # exists to guarantee, so a non-zero value has to be impossible to miss.
    c[0].metric("Oversell units", "0" if oversell == 0 else fmt(oversell, ",.0f"),
                delta=None if not oversell else "invariant broken",
                delta_color="inverse")
    c[1].metric("Rejection rate", fmt(rejection, ".1%"))
    c[2].metric("Mean staleness", fmt(stale, ".1f", " units"))
    c[3].metric("Reserve latency p50", fmt(p50, ".2f", " ms"))
    c[4].metric("Central hops / reserve", fmt(hops, ".2f"))
    if oversell:
        st.error(f"**{oversell:,.0f} units oversold.** For `strong_lock` and "
                 f"`escrow_quota` that would be a bug; for `eventual` it is the "
                 f"expected result and the reason it is the control arm.")


# ── Hero: staleness ──────────────────────────────────────────────────────────

def _staleness_fig(mode: str, policy: str | None) -> go.Figure:
    fig = go.Figure()
    if mode != LV.RECORDED:
        return T.style(fig, height=460)
    s = LV.recorded_staleness(policy)
    if s.empty:
        return T.style(fig, height=460)

    labels = []
    for node in sorted(s["node"].dropna().unique()):
        d = s[s["node"] == node].sort_values("tick")
        col = LV.channel_color(node)
        fig.add_trace(go.Scatter(
            x=d["tick"], y=d["staleness"], mode="lines",
            line=dict(color=col, width=2.8),
            hovertemplate=f"{node} · tick %{{x}}<br>"
                          "stale by %{y:.1f} units<extra></extra>"))
        labels.append((d["tick"].iloc[-1], d["staleness"].iloc[-1], node, col))
    T.right_labels(fig, labels)
    fig.update_xaxes(title="Simulated day (tick)")
    fig.update_yaxes(title="True available − node's view (units)")
    return T.style(fig, height=460,
                   title="How far behind each channel's stock view runs")


def _stock_fig(mode: str, policy: str | None) -> go.Figure:
    """Truth vs belief per SKU. The gap is the quantity, not the levels."""
    fig = go.Figure()
    if mode == LV.RECORDED:
        d = LV.recorded_stock(policy, limit_skus=14)
    else:
        d = LV.live_stock(LV.live_sku_names(14))
    if d.empty:
        return T.style(fig, height=460)

    truth = d.groupby("sku")["true_available"].max().sort_values()
    order = truth.index.tolist()
    fig.add_trace(go.Bar(
        x=truth.reindex(order).values, y=order, orientation="h",
        marker=dict(color="#ECEAE3", line=dict(color=T.GRID, width=1)),
        name="True available", hovertemplate="true %{x:.0f}<extra></extra>"))
    for node in sorted(d["node"].dropna().unique()):
        sub = d[d["node"] == node].set_index("sku").reindex(order)
        fig.add_trace(go.Scatter(
            x=sub["node_view"], y=order, mode="markers", name=node,
            marker=dict(color=LV.channel_color(node), size=10,
                        line=dict(color="#FFFFFF", width=1.2)),
            hovertemplate=f"{node} believes %{{x:.0f}}<extra></extra>"))
    fig.update_layout(showlegend=True, barmode="overlay",
                      legend=dict(orientation="h", y=1.06, x=0))
    fig.update_xaxes(title="Units")
    fig.update_yaxes(title=None)
    return T.style(fig, height=460,
                   title="Stock on hand vs what each channel thinks it has")


def _flow_fig(mode: str, policy: str | None) -> go.Figure:
    fig = go.Figure()
    if mode != LV.RECORDED:
        return T.style(fig, height=340)
    f = LV.recorded_flow(policy)
    if f.empty:
        return T.style(fig, height=340)
    fig.add_trace(go.Bar(x=f["tick"], y=f["granted"], name="Granted",
                         marker_color=T.ACCENT,
                         hovertemplate="tick %{x}<br>granted %{y}<extra></extra>"))
    fig.add_trace(go.Bar(x=f["tick"], y=f["refused"], name="Refused",
                         marker_color=T.ALERT,
                         hovertemplate="tick %{x}<br>refused %{y}<extra></extra>"))
    fig.update_layout(barmode="stack", showlegend=True,
                      legend=dict(orientation="h", y=1.08, x=0))
    fig.update_xaxes(title="Simulated day (tick)")
    fig.update_yaxes(title="Reservations")
    return T.style(fig, height=340, title="Reservation outcomes per day")


# ── Page ─────────────────────────────────────────────────────────────────────

def render() -> None:
    st.title("Live operations")
    st.caption(
        "Three sales channels drawing on one stock pool per SKU. Escrow keeps the "
        "pool arithmetically safe; the cost lands as a stale, conservative view at "
        "each node — which is what corrupts the training signal downstream."
    )

    mode, policy = _mode_control()
    if mode == LV.NONE:
        st.info(
            "**Nothing to show yet.** Start the system with "
            "`python -m edge_system.run_system --scenario smoke --ticks 30`, "
            "or run `python -m experiments.exp_sync` to leave a recorded event "
            "log this page can replay."
        )
        if not LV.http_available():
            st.caption(
                "`httpx` is not installed, so live mode is unavailable here. That "
                "is expected on the slim viewer requirements — install "
                "`requirements-system.txt` to poll a running system."
            )
        return

    if mode == LV.RECORDED:
        st.info(
            f"**Recorded run** — replaying the `{policy}` event log. These are real "
            f"decisions from a completed run, not a simulation of a simulation. "
            f"Dates are the simulated business calendar (M5, 2013–2015), not "
            f"wall-clock time.", icon=":material/history:")

    _tiles(mode, policy)
    st.divider()

    st.plotly_chart(_staleness_fig(mode, policy), width="stretch",
                    config={"displayModeBar": False})
    st.caption(
        "A node spends its own escrow quota without coordinating, so it cannot see "
        "units held in the other channels' quotas. It therefore believes it has "
        "**less** than the pool actually holds — the gap above — and refuses orders "
        "the pool could have served. Each drop is a quota refill resynchronising it."
    )

    st.divider()
    left, right = st.columns([1, 1])
    with left:
        st.plotly_chart(_stock_fig(mode, policy), width="stretch",
                        config={"displayModeBar": False})
    with right:
        st.plotly_chart(_flow_fig(mode, policy), width="stretch",
                        config={"displayModeBar": False})
        st.caption(
            "Refused reservations are not lost sales in the ordinary sense — they "
            "are demand that existed and was never recorded. That is exactly the "
            "censoring E2 applies to the forecaster's training target."
        )

    st.divider()
    st.subheader("Recent events")
    ev = (LV.recorded_events(policy=policy, limit=250) if mode == LV.RECORDED
          else LV.live_events(limit=250))
    if ev.empty:
        st.caption("No events.")
        return
    cols = [c for c in ["sim_date", "tick", "kind", "node", "sku", "qty",
                        "granted", "true_available", "node_view",
                        "staleness_units", "latency_ms"] if c in ev.columns]
    kinds = sorted(ev["kind"].dropna().unique()) if "kind" in ev.columns else []
    pick = st.multiselect("Event kind", kinds, default=kinds)
    view = ev[ev["kind"].isin(pick)] if pick and "kind" in ev.columns else ev
    st.dataframe(view[cols], width="stretch", hide_index=True, height=340)
