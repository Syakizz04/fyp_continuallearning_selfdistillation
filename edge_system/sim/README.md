# Simulation driver

Replays the M5 walk-forward window as a live feed against the running system.

```powershell
docker compose up -d redis
$env:PYTHONIOENCODING = "utf-8"

python -m edge_system.run_system --scenario smoke --ticks 30          # no models, ~2 s
python -m edge_system.run_system --scenario smoke_models --ticks 10   # real TFT+PPO, ~60 s
python -m edge_system.run_system --scenario sync_cell --policy strong_lock --delay-ms 200
python -m edge_system.run_system --scenario partition --policy escrow_quota
```

Results land in `outputs/system/results/` as `sim_ticks_<run_id>.csv` (one row per
tick per channel) and `sim_summary_<run_id>.json`. Service logs go to
`outputs/system/state/logs/<run_id>/`.

| module | role |
|---|---|
| `clock.py` | 1 tick = 1 M5 day; drops days the data cannot serve |
| `order_gen.py` | real demand + quoted price + estimated elasticity → discrete orders |
| `network.py` | the injectable latency/partition dial — E1's independent variable |
| `driver.py` | the tick loop, over HTTP |
| `../run_system.py` | process supervisor + named scenarios |

## What is simulated and what is not

Real: demand level (M5 `realized_demand`), prices (`base_price` from
`sell_prices.csv`), elasticity (hierarchically estimated — see
`dataset_generator/m5/README.md`), and the entire inventory algorithm, which
runs as a service over real HTTP against real Redis.

Simulated: the split of demand across three channels, the conversion of a daily
demand total into discrete order arrivals, and the network.

The demand model is
`units(p) = realized_demand * (p / base_price) ** elasticity` — deliberately the
same functional form `DynamicPricingEnv` computes its reward from, so a pricing
result measures the agent rather than a mismatch between the agent's training
environment and the world it was deployed into.

## Why the network is fake

Latency is the independent variable of E1 and the generator of staleness for E2.
An experiment has to *set* it. A real network — or a broker in front of one —
produces latency as an emergent property of batching and scheduling, so two runs
"at the same delay" are not comparable and a sweep is not reproducible. Making
it a dial is what turns the confounder into a treatment. The limitation to state
in the report: this models delay and partition, not loss, reordering or
congestion.

## Determinism

Order streams are seeded from `(seed, tick, sku, channel)` content, not from a
running RNG. Two sweep cells therefore see byte-identical customers regardless of
policy, hop count, or the order the driver happened to walk the SKUs in. If the
stream depended on control flow, every difference between policies would be
confounded with a difference in who walked through the door.

## Results so far

20 ticks, 15 SKUs, 50 ms one-way delay, Redis backend:

| policy | fill rate | oversell units | hops/order | p50 latency | p99 latency |
|---|---|---|---|---|---|
| `strong_lock` | 98.7% | 0 | 1.00 | 52.1 ms | 53.2 ms |
| `escrow_quota` | 93.3% | 0 | 0.42 | **0.0 ms** | 52.7 ms |
| `eventual` | 100.0% | **223** | 0.03 | 0.0 ms | 52.1 ms |

Escrow attains strong lock's zero oversell at eventual's *median* latency,
because most orders never leave the node. The cost is not latency and not
correctness — it is **fill rate**: 93.3% vs 98.7%. A node refuses orders it
could actually have served, because it cannot see stock held in other nodes'
quotas. That conservatism is the same quantity that reaches the pricing agent as
staleness, which is what E2 then studies.

### Partition onset

60 ticks, marketplace cut off for ticks 20–44, units sold **on the partitioned
channel**:

| policy | before | during | after | refused during |
|---|---|---|---|---|
| `strong_lock` | 68 | **0** | 48 | 97 |
| `escrow_quota` | 58 | **20** | 43 | 77 |
| `eventual` | 71 | **97** | 45 | 0 |

Strong lock is correct and completely unavailable; eventual is available and
wrong; escrow keeps trading on rights acquired before the link dropped, then
degrades to refusal. This is the CAP trade-off on a retail stock pool.

**The onset matters.** Partitioning a node from tick 0 makes escrow look
identical to strong lock, because acquiring the first quota needs a hop that
never succeeds. The advantage exists only for rights held *before* the failure,
so the scenario drops the link mid-run via `network.schedule`.

## Silent bugs found here

Recorded because every one produced complete, plausible, wrong output rather
than an error — the kind that survives into a report.

**1. Config does not cross a process boundary.** The offline pipelines mutate
`CONFIG` in place and it works because everything is one process. The services
are separate uvicorn children that import their own fresh `SYSTEM_CONFIG`, so
`--policy strong_lock` set a variable in the supervisor and nothing else. Three
"different" policy cells were all executed by `escrow_quota`, each returning a
full, well-formed result set. Caught only because `strong_lock` reported 44
`quota_refills` — a counter it never increments.

**2. A stale service answers `/health` cheerfully.** Services left from an
earlier session held the ports, the new children failed to bind and died, and
readiness was satisfied by the *old* processes — a different policy and last
week's stock. `ServiceProcess.preflight()` now refuses to start on a busy port.

**3. `paths.data_dir` did not reach the edge nodes.** The same boundary bug, in
the experiment it would have damaged most: **E3 repoints `data_dir`** between
`processed_m5` (old elasticity) and `processed_m5_v2` (re-grounded). Both arms
would have loaded the same data, and the ablation could only ever have reported
"elasticity makes no difference."

**4. `sync.reservation_ttl_s` was configured and dead.** Nothing passed it to
`reserve()`; the escrow core's own default was also `300.0`, so the two agreed
and the setting looked alive. Setting it to 60 in a scenario did nothing.

**5. `sync.quota_low_watermark` was read by no code at all.** The config
documented "refill when quota < 25% of last grant"; the policy actually refilled
only once an order could not be covered. A described mechanism that did not
exist.

### How the class is now closed

`config._SERVICE_KEYS` is a single manifest of every setting a child process
reads. It drives **both** directions — the env vars the supervisor exports and
the overrides a child applies at import — so the two cannot drift. On top of it:

- `/health` reports the **effective** config read back off the live policy
  object, not off `SYSTEM_CONFIG`.
- `run_system.verify_inventory_config()` compares every requested setting
  against that readback and refuses to run on any mismatch.
- `test_manifest_covers_every_config_section_a_child_process_reads` scans the
  service modules for `SYSTEM_CONFIG[...]` reads and fails if a section is
  missing from the manifest — mechanical, so it catches the *next* one rather
  than relying on someone remembering.

### One knob is wired but nearly inert, and that is a finding

`quota_low_watermark` now works as documented, and measurably almost nothing
happens: 300 orders of 2 units give 75 refills at watermark 0.0 and 76 at 0.9.
The coordination rate is set by `block size / order size`, not by trigger
timing — `_refill` takes a fixed block sized on the current order, so an earlier
trigger just shifts the same refill within its cycle.

This also rules out the obvious explanation for escrow's fill-rate cost. The
93.3% is **not** late refilling; it is structural, since a node cannot see stock
held in other nodes' quotas, and refilling sooner would if anything worsen it.
Making the watermark a real lever means refilling *to a target level* rather
than by a fixed block — a change to the method under test, so it is a Phase 5
decision rather than a quiet fix.

## A pre-existing skew this phase fixed

`LocalInference` built one `DynamicPricingEnv` per SKU. `_precompute` normalises
its columns by the max within the frame it is handed, and
`elasticity_coefficient` is constant within a SKU — so per-SKU slicing drove
`elasticity_norm` to exactly −1.0 for every item, erasing the feature. Training
builds the env from a multi-SKU frame, where the same feature spans roughly
[−1, −0.02]. The agent was being asked a question it had never been trained on,
and it would have made the E3 elasticity ablation meaningless.

Now one panel-wide env is shared. Residual caveat for the limitations section:
the normalisation basis is the full walk-forward panel while training used the
base window, so the divisors are close but not identical.

## Cost

The driver makes roughly `n_skus x n_channels` HTTP calls per tick, and in
`pricing: node` mode each one is a PPO forward pass. `sim.n_skus` is the main
runtime dial. Measured: 30 ticks x 10 SKUs static ≈ 1.5 s; 10 ticks x 5 SKUs with
models ≈ 56 s, plus ~10 s per node for the cold model load.
