"""Continual-learning results dashboard (initial + drift pipelines).

Run from the project root:

    streamlit run dashboard/app.py

Viewer only: reads on-disk artifacts under outputs/ and outputs/drift/. No training
code or GPU is imported.
"""
from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent))  # allow flat module imports

import theme as T            # noqa: E402
import view_home             # noqa: E402
import view_initial          # noqa: E402
import view_drift            # noqa: E402
import view_cross            # noqa: E402

st.set_page_config(
    page_title="Continual learning under regime shift",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded",
)
st.markdown(T.CSS, unsafe_allow_html=True)

pages = [
    st.Page(view_home.render, title="Overview", url_path="overview", default=True),
    st.Page(view_initial.render, title="Synthetic pipeline", url_path="synthetic"),
    st.Page(view_drift.render, title="Drift pipeline", url_path="drift"),
    st.Page(view_cross.render, title="Replay vs SDFT", url_path="ablation"),
]
nav = st.navigation(pages, position="sidebar")
nav.run()
