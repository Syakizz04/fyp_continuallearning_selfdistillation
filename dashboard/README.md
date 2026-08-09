# Dashboard

A viewer-only Streamlit dashboard covering both halves of the project: the
**offline results** (FYP1's task-based CL and the drift-triggered walk) and the
**deployed system** (FYP2's shared inventory pool and per-node model health). It
reads on-disk artifacts and, when they are up, the running services — no training
code, torch, or GPU is imported.

## Run

From the project root (so `outputs/` resolves):

```powershell
.venv\Scripts\Activate.ps1
streamlit run dashboard/app.py
```

## Pages

- **Overview** — frames the two experiments and the shared question.
- **Synthetic pipeline** (the `initial_pipeline` package) — hero is the forgetting matrix (train task x eval task);
  supporting panels are online error across the six tasks and retention of an early
  task; a tab holds the PPO pricing metrics.
- **Drift pipeline** — hero is the forecast-error timeline 2013-2015 with retrain
  markers; supporting panels are the profit index over the walk and the
  accuracy-vs-retrains trade-off, plus the efficiency table.
- **Replay vs SDFT** — the ablation side-by-side under both regimes.

### Deployed system

- **Live operations** — the shared stock pool. Hero is **staleness**: how far each
  channel's view of stock runs behind the truth. Correctness is unconditional
  here (escrow cannot oversell), so the question is not whether it works but what
  it costs — and the cost is the stale, conservative node view that goes on to
  corrupt the training signal. Supporting panels: truth vs belief per SKU,
  reservation outcomes per day, and the raw event stream.
- **Model health** — what each node is serving, forecast error against the
  **calibrated** trigger threshold, when each arm retrained, and E4's memory
  footprint per method.

## The three source modes

The deployed pages describe a system that may or may not be running, so they
resolve one of three states and all three are designed for:

| Mode | When | Reads |
|------|------|-------|
| `live` | services are up | HTTP: `/metrics`, `/stock/{sku}`, `/events`, node `/health` + `/model` + `/drift` |
| `recorded` | a previous run left a log | `outputs/system/state/events.db`, `sim_summary_*.json`, `registry.db` |
| `none` | neither | an explanation of how to produce either |

**Recorded is the default, not a fallback.** A viva demo that needs Redis plus
five uvicorn processes to survive the moment is one that fails in the moment, and
`dashboard-deploy` on Streamlit Cloud has no services at all. Both modes read the
same quantities from the same writer, so they agree by construction.

Staleness is read off the event log rather than recomputed: it was captured when
each decision was made and cannot be reconstructed afterwards, because by then
the true figure has moved.

## Data sources

| Page | Reads |
|------|-------|
| Initial | `outputs/results/all_metrics.csv` |
| Drift | `outputs/drift/results/` — `drift_stream_*.csv`, `metrics_efficiency.csv`, `metrics_forgetting.csv`, `retrain_log_*.json` |
| Live operations | `outputs/system/state/events.db`, `outputs/system/results/sim_summary_*.json`, or the live services |
| Model health | `drift_stream_*.csv`, `retrain_log_*.json`, `memory_*.csv`, `registry.db`, or the live nodes |

The `periodic` arm is excluded from the FYP1 pages (no longer part of that
ablation); the deployed pages show whatever arms are actually on disk.

## Dependencies

`httpx` is imported **lazily** in `live.py` and its absence only disables live
mode. This is deliberate: it ships in `requirements-system.txt`, not
`requirements.txt`, because `dashboard-deploy` installs a slim viewer set.
Importing it at module scope would break that branch on import — `test_views.py`
pins this.

## Tests

```powershell
pytest dashboard/ -q
```

A page fails at *render* time, over the websocket, long after its HTTP route has
returned 200 — so hitting the URL proves nothing. `test_views.py` executes every
page with `AppTest` and asserts it raised nothing, including with no services, no
event log and no `httpx`.

## Styling

All visual choices live in three places, applied to every figure:

- `.streamlit/config.toml` — native light theme, deep-teal accent, IBM Plex fonts.
- `dashboard/theme.py` — the strategy->colour mapping, the shared Plotly template,
  and helpers (`line`, `right_labels`, `style`). One fixed colour per strategy
  across both pipelines; SDFT is the most saturated line, baselines are muted grey,
  oxblood is reserved for retrain markers.

To restyle the whole dashboard, edit `theme.py` — not the page modules.
