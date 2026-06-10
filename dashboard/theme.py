"""Shared visual language for the dashboard.

One strategy -> colour mapping, one Plotly template, and a small set of helpers,
so every figure on every page reads as the same instrument. Import and use these;
do not set ad-hoc colours or fonts in the page modules.
"""
from __future__ import annotations

import plotly.graph_objects as go
import plotly.io as pio

# ── Palette ──────────────────────────────────────────────────────────────────
INK      = "#1A1A1A"   # near-black text / axes
PAPER    = "#FCFCFA"   # near-white page
PANEL    = "#FFFFFF"   # plot face
GRID     = "#ECEAE3"   # one faint horizontal rule
ACCENT   = "#0E6E6E"   # deep teal — the single accent (also SDFT)
ALERT    = "#A8322D"   # oxblood — reserved for trigger / retrain markers ONLY

# One fixed colour per strategy, shared across both pipelines and every chart.
# SDFT is the most saturated line (the proposed method); baselines are muted grey.
STRATEGY_COLOR = {
    "sdft":   ACCENT,     # deep teal, most saturated
    "replay": "#3D5A80",  # slate-blue
    "recall": "#3D5A80",  # RECALL == replay family -> same colour
    "ewc":    "#9A7B4F",  # muted bronze
    "naive":  "#9AA0A6",  # baseline grey (initial pipeline anchor)
    "frozen": "#6B7177",  # baseline grey (drift pipeline anchor)
}

# Heavier lines for the methods under test; baselines stay quiet.
STRATEGY_WIDTH = {
    "sdft": 3.6, "replay": 3.0, "recall": 3.0,
    "ewc": 2.6, "naive": 2.2, "frozen": 2.2,
}

STRATEGY_LABEL = {
    "sdft": "SDFT", "replay": "Replay", "recall": "Replay",
    "ewc": "EWC", "naive": "Naive", "frozen": "Frozen",
}

# Draw order: baselines first (underneath), SDFT last (on top).
STRATEGY_ORDER = ["naive", "frozen", "ewc", "replay", "recall", "sdft"]

# Retrain markers share the single alert colour; shape distinguishes the method.
STRATEGY_MARKER = {
    "sdft": "circle", "replay": "square", "recall": "square",
    "ewc": "diamond", "naive": "triangle-up", "frozen": "x",
}

# Single-hue sequential ramp for matrices (white -> teal); higher = worse error.
TEAL_SCALE = [[0.0, "#FFFFFF"], [0.5, "#7FB3B3"], [1.0, ACCENT]]

FONT   = "IBM Plex Sans, sans-serif"
MONO   = "IBM Plex Mono, monospace"


def color_of(name: str) -> str:
    return STRATEGY_COLOR.get(name, "#9AA0A6")


def width_of(name: str) -> float:
    return STRATEGY_WIDTH.get(name, 2.4)


def label_of(name: str) -> str:
    return STRATEGY_LABEL.get(name, name.upper())


def marker_of(name: str) -> str:
    return STRATEGY_MARKER.get(name, "circle")


def order_key(name: str) -> int:
    return STRATEGY_ORDER.index(name) if name in STRATEGY_ORDER else 99


# ── Plotly template ──────────────────────────────────────────────────────────
def _axis(grid: bool) -> dict:
    """Hairline mirrored box; optional single faint gridline."""
    return dict(
        showline=True, linecolor=INK, linewidth=1.1, mirror=True,
        ticks="outside", tickcolor=INK, ticklen=5, tickwidth=1.1,
        tickfont=dict(family=FONT, size=14, color=INK),
        title=dict(font=dict(family=FONT, size=15, color=INK)),
        showgrid=grid, gridcolor=GRID, gridwidth=1,
        zeroline=False,
    )


def _register_template() -> None:
    t = go.layout.Template()
    t.layout = go.Layout(
        font=dict(family=FONT, size=15, color=INK),
        title=dict(
            font=dict(family=FONT, size=19, color=INK),
            x=0.0, xanchor="left", xref="paper", y=0.97, yanchor="top",
        ),
        paper_bgcolor=PANEL,
        plot_bgcolor=PANEL,
        colorway=[STRATEGY_COLOR[k] for k in ("sdft", "replay", "ewc", "frozen", "naive")],
        xaxis=_axis(grid=False),                 # x gridlines off
        yaxis=_axis(grid=True),                  # one faint horizontal rule
        margin=dict(l=72, r=124, t=64, b=56),    # right room for direct labels
        showlegend=False,                        # we direct-label instead
        hoverlabel=dict(
            font=dict(family=FONT, size=13, color=INK),
            bgcolor="#FFFFFF", bordercolor=INK,
        ),
        hovermode="x unified",
        dragmode=False,
    )
    pio.templates["fyp"] = t


_register_template()


def style(fig: go.Figure, height: int = 460, title: str | None = None) -> go.Figure:
    """Apply the shared template and common chrome to a figure."""
    fig.update_layout(template="fyp", height=height)
    if title is not None:
        fig.update_layout(title_text=title)
    return fig


def right_labels(fig: go.Figure, items: list[tuple[float, float, str, str]]) -> go.Figure:
    """Direct-label lines at their right end. items = (x, y, text, colour)."""
    for x, y, text, color in items:
        fig.add_annotation(
            x=x, y=y, text=text, showarrow=False,
            xanchor="left", xshift=10, yanchor="middle",
            font=dict(family=FONT, size=14, color=color),
        )
    fig.update_layout(showlegend=False)
    return fig


def line(fig: go.Figure, x, y, strategy: str, dash: str | None = None,
         hovertemplate: str | None = None) -> None:
    """Add one strategy line styled from the shared mapping (no marker glow)."""
    fig.add_trace(go.Scatter(
        x=x, y=y, mode="lines", name=label_of(strategy),
        line=dict(color=color_of(strategy), width=width_of(strategy),
                  dash=dash or "solid"),
        connectgaps=False,
        hovertemplate=hovertemplate,
    ))


# ── Page CSS (fonts + tile typography) ───────────────────────────────────────
CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500&family=IBM+Plex+Sans:wght@400;500;600&display=swap');

html, body, [class*="css"], .stMarkdown, .stMetric { font-family: 'IBM Plex Sans', sans-serif; }

/* Headline numbers in mono */
[data-testid="stMetricValue"] { font-family: 'IBM Plex Mono', monospace; font-weight: 500; color: #1A1A1A; }
[data-testid="stMetricLabel"] { font-size: 0.80rem; letter-spacing: .02em; color: #5A5A55; }
[data-testid="stMetricDelta"] { font-family: 'IBM Plex Mono', monospace; }

/* Tighten the page; ruled, sectioned feel */
.block-container { padding-top: 2.2rem; padding-bottom: 3rem; max-width: 1320px; }
hr { border-color: #D8D6CE; }
h1, h2, h3 { font-weight: 600; letter-spacing: -.01em; }
h1 { font-size: 1.85rem; }
code, pre, kbd { font-family: 'IBM Plex Mono', monospace; }
[data-testid="stDataFrame"] { font-family: 'IBM Plex Mono', monospace; }

/* Hide the header anchor link ("clip") icons for cleaner titles */
[data-testid="stHeaderActionElements"] { display: none; }
</style>
"""
