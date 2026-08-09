# Edge node

One process per sales channel (`pos`, `web`, `marketplace`). Everything a
request touches is local: the TFT and PPO are loaded into **this process's
memory at startup**, drift is detected here, and retraining runs here. The only
outbound calls are to the inventory service to reserve stock and to the control
plane to report a new model version.

This is "edge" in the sense the supervisor specified — inference is local, no
inference service is called out to. There is no containerisation and no edge
hardware; see the plan for why.

## Files

| file | role |
|------|------|
| `inference.py` | Loads base checkpoints in-process; `price()` / `forecast()`. |
| `monitor_local.py` | Streaming wrapper around the offline `DebouncedDetector`. |
| `retrain_local.py` | Background SDFT/replay/EWC retrain + atomic hot-swap. |
| `service.py` | FastAPI. |

## Why it reuses the offline code

`inference.py` calls `drift_pipeline.monitor.load_base()` and
`retrain_local.py` wraps `drift_pipeline.retrain.RetrainController` rather than
reimplementing either. If the deployed retrain diverged from the measured one,
the FYP1 results would not transfer to the system and E2 would be comparing two
different things. Same reason `monitor_local.py` reuses `DebouncedDetector` and
the calibrated thresholds: a node that fired on a different criterion would not
be comparable with the offline arms.

## Serving during a retrain

Retraining runs on a background thread and builds a *new* model object; the old
one keeps answering requests until the new one is swapped in under
`_swap_lock`. A node never goes offline to retrain — otherwise "retrains locally"
would be aspirational rather than true.

Only one retrain runs at a time per node. A trigger arriving mid-retrain is
**recorded and dropped, not queued**: by the time the in-flight retrain lands,
the condition that fired the second trigger has usually been addressed, and
queueing produces exactly the "retrain churn" EWC showed in the first real run.

A failed retrain does not take the node down — it keeps serving the previous
generation, and the failure is recorded (`ok=0`, `error`) so it stays visible.

## The staleness mechanism (what E2 measures)

`/price` with no `inventory_level` asks the inventory service what **this node**
may sell — its escrow quota, not the global total. Measured end-to-end on a SKU
with 500 units stocked:

```
true_available   495          node_views  {pos: 10, web: 0, marketplace: 0}
                              staleness   {pos: 485, web: 495, marketplace: 495}

node view   0  ->  action  2   tier -6%    (believes it is out of stock)
node view  10  ->  action 10   tier +10%
```

The same underlying reality produces a **16-point pricing swing** purely from
sync staleness. This is the mechanism E2 quantifies, and it is live rather than
hypothetical.

**Calibration note for E2:** a raw quota is a very aggressive proxy for
`inventory_level` — the PPO was trained on true stock, so a view of 10 against a
true 495 is not merely stale but on a different scale, and would demolish the
agent rather than degrade it informatively. Staleness must therefore be
*graduated* for the experiment, which is what `sync.quota_refill_multiple`
controls: large multiple → node view close to truth (`none`), small → view near
zero (`severe`). Pick those levels deliberately when running E2; do not just
take the default.

## Running

```powershell
docker compose up -d redis
$env:PYTHONIOENCODING="utf-8"   # rich prints non-cp1252 glyphs on load
$env:FYP_NODE="pos"; $env:FYP_STRATEGY="sdft"
uvicorn edge_system.edge.service:app --port 8010
```

Model load takes ~10-12 s (TFT + PPO + the M5 frames). `/health` reports
`load_seconds` and the current `model_generation`.

## Endpoints

```
GET  /price     ?sku&sim_date[&inventory_level]   omit inventory_level to use the escrow view
GET  /forecast  ?sku&sim_date
POST /check     {tick, sim_date, mase?, profit_index?, blocking?}
GET  /model     current version + retrain history
GET  /drift     detector status, events, recent stream
GET  /health
```
