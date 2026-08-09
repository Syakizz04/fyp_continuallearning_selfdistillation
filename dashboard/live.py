"""
Data access for the two deployed-system pages.

The FYP1 result pages read finished CSVs. These two describe a system that may or
may not be running right now, so there are three states and all of them have to
look deliberate:

  live      - the services are up; poll their HTTP endpoints
  recorded  - no services, but a previous run left an event log and summaries
  none      - neither; say so plainly instead of rendering empty axes

**Recorded is the default, not the fallback.** A viva demo that depends on Redis
plus five uvicorn processes surviving the moment is a demo that fails in the
moment; and `dashboard-deploy` on Streamlit Cloud has no services at all, ever.
The recorded path reads the same quantities from `events.db`, which is written by
the same code, so the two modes agree by construction rather than by maintenance.

Two hard constraints, both load-bearing:

* **No torch, no training imports.** The dashboard must stay installable from the
  slim viewer requirements.
* **httpx is optional.** It lives in requirements-system.txt, NOT
  requirements.txt, because the deploy branch deliberately excludes the system
  runtime. Importing it at module scope would break that branch on import, so it
  is imported lazily and its absence simply means live mode is unavailable.

SQLite is opened read-only (`mode=ro`) so that inspecting a run cannot disturb a
run that is still writing.
"""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
STATE_DIR = ROOT / "outputs" / "system" / "state"
RESULTS_DIR = ROOT / "outputs" / "system" / "results"
EVENTS_DB = STATE_DIR / "events.db"
REGISTRY_DB = STATE_DIR / "registry.db"
DRIFT_RESULTS = ROOT / "outputs" / "drift" / "results"

LIVE, RECORDED, NONE = "live", "recorded", "none"

#: Mirrors edge_system.config defaults. Duplicated rather than imported: importing
#: edge_system pulls the runtime, and this file must stay viewer-only.
INVENTORY_URL = "http://127.0.0.1:8001"
CONTROL_URL = "http://127.0.0.1:8000"
NODE_URLS = {"pos": "http://127.0.0.1:8010",
             "web": "http://127.0.0.1:8011",
             "marketplace": "http://127.0.0.1:8012"}

#: Channel colours. Distinct from theme.STRATEGY_COLOR on purpose - a channel is
#: not a CL method, and reusing those hues would imply a correspondence that does
#: not exist.
CHANNEL_COLOR = {"pos": "#0E6E6E", "web": "#3D5A80", "marketplace": "#9A7B4F"}


def channel_color(name: str) -> str:
    return CHANNEL_COLOR.get(name, "#9AA0A6")


# ── HTTP (optional) ──────────────────────────────────────────────────────────

def _httpx():
    """Lazy, optional. Absent on the slim deploy requirements - see module docs."""
    try:
        import httpx
        return httpx
    except ImportError:
        return None


def http_available() -> bool:
    return _httpx() is not None


def _get(url: str, path: str, timeout: float = 1.5):
    """GET returning parsed JSON, or None on any failure.

    Deliberately swallows everything: a dashboard must not raise because a
    service it is describing went away mid-render. Callers distinguish "no data"
    from "zero" by checking for None.
    """
    httpx = _httpx()
    if httpx is None:
        return None
    try:
        r = httpx.get(f"{url}{path}", timeout=timeout)
        r.raise_for_status()
        return r.json()
    except Exception:                                          # noqa: BLE001
        return None


@st.cache_data(ttl=3, show_spinner=False)
def probe() -> Dict:
    """Which pieces of the system are reachable. Cached briefly so one page
    render does not issue the same probe a dozen times."""
    if not http_available():
        return {"inventory": False, "control": False, "nodes": {}}
    inv = _get(INVENTORY_URL, "/health")
    ctl = _get(CONTROL_URL, "/health")
    nodes = {n: bool(_get(u, "/health", timeout=1.0)) for n, u in NODE_URLS.items()}
    return {
        "inventory": bool(inv and inv.get("ok")),
        "control": bool(ctl and ctl.get("ok")),
        "nodes": nodes,
        "inventory_health": inv or {},
    }


def has_recorded() -> bool:
    return EVENTS_DB.exists()


def available_modes() -> List[str]:
    modes = []
    if probe()["inventory"]:
        modes.append(LIVE)
    if has_recorded():
        modes.append(RECORDED)
    return modes or [NONE]


# ── Recorded: the event log ──────────────────────────────────────────────────

def _read_sql(db: Path, sql: str, params: tuple = ()) -> pd.DataFrame:
    if not db.exists():
        return pd.DataFrame()
    try:
        # mode=ro: never disturb a run that is still writing to this file.
        with sqlite3.connect(f"file:{db}?mode=ro", uri=True) as conn:
            return pd.read_sql(sql, conn, params=params)
    except Exception:                                          # noqa: BLE001
        return pd.DataFrame()


@st.cache_data(ttl=30, show_spinner=False)
def recorded_policies() -> pd.DataFrame:
    """One row per policy present in the log - the run selector's options."""
    return _read_sql(EVENTS_DB, """
        SELECT policy,
               COUNT(*)                AS events,
               COUNT(DISTINCT sku)     AS skus,
               COUNT(DISTINCT node)    AS nodes,
               MIN(sim_date)           AS first_date,
               MAX(sim_date)           AS last_date,
               MAX(tick)               AS ticks
        FROM events WHERE policy IS NOT NULL
        GROUP BY policy ORDER BY events DESC""")


@st.cache_data(ttl=30, show_spinner=False)
def recorded_events(policy: Optional[str] = None, kind: Optional[str] = None,
                    limit: int = 500) -> pd.DataFrame:
    where, params = [], []
    if policy:
        where.append("policy = ?"); params.append(policy)
    if kind:
        where.append("kind = ?"); params.append(kind)
    clause = (" WHERE " + " AND ".join(where)) if where else ""
    return _read_sql(EVENTS_DB,
                     f"SELECT * FROM events{clause} ORDER BY id DESC LIMIT ?",
                     tuple(params) + (limit,))


@st.cache_data(ttl=30, show_spinner=False)
def recorded_flow(policy: str) -> pd.DataFrame:
    """Per-tick reservation outcome, the shape the flow chart needs."""
    return _read_sql(EVENTS_DB, """
        SELECT tick, MIN(sim_date) AS sim_date,
               SUM(granted)               AS granted,
               SUM(1 - granted)           AS refused,
               SUM(qty * granted)         AS units_granted,
               AVG(latency_ms)            AS latency_ms
        FROM events WHERE kind = 'reserve' AND policy = ?
        GROUP BY tick ORDER BY tick""", (policy,))


@st.cache_data(ttl=30, show_spinner=False)
def recorded_staleness(policy: str) -> pd.DataFrame:
    """Per-tick, per-node staleness at decision time.

    Captured when the decision was made; it cannot be reconstructed afterwards,
    which is why the event log records it rather than deriving it.
    """
    return _read_sql(EVENTS_DB, """
        SELECT tick, node, MIN(sim_date) AS sim_date,
               AVG(staleness_units) AS staleness,
               AVG(true_available)  AS true_available,
               AVG(node_view)       AS node_view,
               COUNT(*)             AS n
        FROM events
        WHERE kind = 'reserve' AND policy = ? AND staleness_units IS NOT NULL
        GROUP BY tick, node ORDER BY tick""", (policy,))


@st.cache_data(ttl=30, show_spinner=False)
def recorded_stock(policy: str, limit_skus: int = 20) -> pd.DataFrame:
    """Latest observed true stock vs each node's belief, per SKU.

    Taken from the most recent reserve event per (sku, node), which is the last
    moment the two figures were both known.
    """
    df = _read_sql(EVENTS_DB, """
        SELECT e.sku, e.node, e.true_available, e.node_view, e.staleness_units,
               e.sim_date, e.tick
        FROM events e
        JOIN (SELECT sku, node, MAX(id) AS mid
              FROM events WHERE kind = 'reserve' AND policy = ?
              GROUP BY sku, node) m
          ON e.id = m.mid
        ORDER BY e.sku""", (policy,))
    if df.empty:
        return df
    keep = (df.groupby("sku")["staleness_units"].max()
              .sort_values(ascending=False).head(limit_skus).index)
    return df[df["sku"].isin(keep)]


@st.cache_data(ttl=30, show_spinner=False)
def recorded_summaries() -> pd.DataFrame:
    """Every sim_summary_*.json as one row - the runs that have been completed."""
    rows = []
    for fp in sorted(RESULTS_DIR.glob("sim_summary_*.json")):
        try:
            d = json.loads(fp.read_text())
        except Exception:                                      # noqa: BLE001
            continue
        m = d.get("inventory_metrics", {})
        rows.append({
            "run_id": d.get("run_id", fp.stem),
            "policy": d.get("policy"), "delay_ms": d.get("delay_ms"),
            "ticks": d.get("ticks"), "skus": d.get("skus"), "span": d.get("span"),
            "orders": d.get("orders"), "fill_rate": d.get("fill_rate"),
            "revenue": d.get("revenue"),
            "mean_staleness_units": d.get("mean_staleness_units"),
            "oversell_units": m.get("oversell_units"),
            "oversell_events": m.get("oversell_events"),
            "rejection_rate": m.get("rejection_rate"),
            "latency_p50_ms": m.get("latency_p50_ms"),
            "latency_p99_ms": m.get("latency_p99_ms"),
            "roundtrips_per_reserve": m.get("roundtrips_per_reserve"),
            "quota_refills": m.get("quota_refills"),
            "backend": m.get("backend"),
        })
    return pd.DataFrame(rows)


# ── Live: HTTP ───────────────────────────────────────────────────────────────

@st.cache_data(ttl=2, show_spinner=False)
def live_metrics() -> Dict:
    return _get(INVENTORY_URL, "/metrics") or {}


@st.cache_data(ttl=2, show_spinner=False)
def live_events(limit: int = 200) -> pd.DataFrame:
    data = _get(INVENTORY_URL, "/events", timeout=3.0)
    if not data:
        return pd.DataFrame()
    df = pd.DataFrame(data)
    return df.head(limit)


@st.cache_data(ttl=2, show_spinner=False)
def live_stock(skus: tuple) -> pd.DataFrame:
    """Per-SKU ground truth alongside each node's view and staleness."""
    rows = []
    for sku in skus:
        d = _get(INVENTORY_URL, f"/stock/{sku}")
        if not d:
            continue
        for node, view in (d.get("node_views") or {}).items():
            rows.append({
                "sku": sku, "node": node,
                "true_available": d.get("true_available"),
                "node_view": view,
                "staleness_units": (d.get("staleness") or {}).get(node),
                "estimate_error": (d.get("estimate_errors") or {}).get(node),
            })
    return pd.DataFrame(rows)


@st.cache_data(ttl=5, show_spinner=False)
def live_sku_names(limit: int = 20) -> tuple:
    """SKU ids seen in the recent event stream - /stock needs an id to ask about."""
    df = live_events(limit=400)
    if df.empty or "sku" not in df.columns:
        return ()
    return tuple(df["sku"].dropna().unique()[:limit])


@st.cache_data(ttl=3, show_spinner=False)
def live_nodes() -> pd.DataFrame:
    """Per-node health + served model version."""
    rows = []
    for node, url in NODE_URLS.items():
        h = _get(url, "/health", timeout=1.0)
        if not h:
            rows.append({"node": node, "up": False})
            continue
        m = _get(url, "/model", timeout=1.5) or {}
        version = m.get("version") or {}
        rows.append({
            "node": node, "up": bool(h.get("ok")),
            "strategy": h.get("strategy"),
            "uptime_s": h.get("uptime_s"),
            "generation": h.get("model_generation"),
            "load_seconds": h.get("load_seconds"),
            "retrain_in_flight": h.get("retrain_in_flight"),
            "version": version.get("version") or version.get("id"),
            "n_retrains": len(m.get("history") or []),
        })
    return pd.DataFrame(rows)


@st.cache_data(ttl=3, show_spinner=False)
def live_node_drift(node: str, limit: int = 60) -> Dict:
    url = NODE_URLS.get(node)
    return (_get(url, f"/drift?limit={limit}", timeout=2.0) or {}) if url else {}


# ── Recorded model health ────────────────────────────────────────────────────

@st.cache_data(ttl=30, show_spinner=False)
def registry_nodes() -> pd.DataFrame:
    # Table is `node_state`, not `nodes`. Getting this wrong is invisible:
    # `_read_sql` swallows the error and returns an empty frame, so the panel
    # renders as "registry is empty" rather than as a mistake.
    return _read_sql(REGISTRY_DB, "SELECT * FROM node_state")


@st.cache_data(ttl=30, show_spinner=False)
def registry_models() -> pd.DataFrame:
    return _read_sql(REGISTRY_DB,
                     "SELECT * FROM model_versions ORDER BY id DESC LIMIT 200")


@st.cache_data(ttl=30, show_spinner=False)
def drift_arms() -> List[str]:
    return sorted({p.stem.replace("drift_stream_", "")
                   for p in DRIFT_RESULTS.glob("drift_stream_*.csv")})


@st.cache_data(ttl=30, show_spinner=False)
def drift_stream(arm: str) -> pd.DataFrame:
    fp = DRIFT_RESULTS / f"drift_stream_{arm}.csv"
    if not fp.exists():
        return pd.DataFrame()
    return pd.read_csv(fp, parse_dates=["date"])


@st.cache_data(ttl=30, show_spinner=False)
def retrain_log(arm: str) -> pd.DataFrame:
    fp = DRIFT_RESULTS / f"retrain_log_{arm}.json"
    if not fp.exists():
        return pd.DataFrame()
    try:
        obj = json.loads(fp.read_text())
    except Exception:                                          # noqa: BLE001
        return pd.DataFrame()
    df = pd.DataFrame(obj.get("events", []))
    if not df.empty:
        df["date"] = pd.to_datetime(df["date"])
        df["arm"] = arm
    return df


@st.cache_data(ttl=30, show_spinner=False)
def memory_log(arm: str) -> pd.DataFrame:
    """E4's per-event footprint record, written by the same runs as the walk."""
    for fp in (DRIFT_RESULTS / f"memory_{arm}.csv",
               *DRIFT_RESULTS.glob(f"e2/*/memory_{arm}.csv")):
        if fp.exists():
            try:
                return pd.read_csv(fp)
            except Exception:                                  # noqa: BLE001
                continue
    return pd.DataFrame()
