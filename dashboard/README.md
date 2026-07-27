# Results dashboard

A viewer-only Streamlit dashboard for the **initial** (task-based CL) and **drift**
(drift-triggered CL) pipelines. It reads the on-disk artifacts those pipelines
write — no training code, torch, or GPU is imported.

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

## Data sources

| Page | Reads |
|------|-------|
| Initial | `outputs/results/all_metrics.csv` |
| Drift | `outputs/drift/results/` — `drift_stream_*.csv`, `metrics_efficiency.csv`, `metrics_forgetting.csv`, `retrain_log_*.json` |

The `periodic` arm is excluded everywhere (no longer part of the ablation).

## Styling

All visual choices live in three places, applied to every figure:

- `.streamlit/config.toml` — native light theme, deep-teal accent, IBM Plex fonts.
- `dashboard/theme.py` — the strategy->colour mapping, the shared Plotly template,
  and helpers (`line`, `right_labels`, `style`). One fixed colour per strategy
  across both pipelines; SDFT is the most saturated line, baselines are muted grey,
  oxblood is reserved for retrain markers.

To restyle the whole dashboard, edit `theme.py` — not the page modules.
