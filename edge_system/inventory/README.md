# Inventory sync service

Shared stock across three sales channels contending for one pool per SKU.
This is the measured contribution of FYP2, not plumbing: it replaces the toy
`_simulate_inventory()` in `dataset_generator/m5/build_m5.py` — an (s,S) loop
with instant next-day replenishment that was the only fully fabricated part of
the dataset — with a real reservation algorithm driven by real demand.

## Layout

| file | role |
|------|------|
| `escrow.py` | The algorithm. Pure, no I/O, property-tested on its own. |
| `policies.py` | `strong_lock` / `eventual` / `escrow_quota` — E1's independent variable. |
| `store.py` | Durable central pool: Redis (Lua CAS) or SQLite (IMMEDIATE txn). |
| `events.py` | Append-only event log, off the hot path. |
| `service.py` | FastAPI wrapper. |

## The algorithm

Escrow (O'Neil, ACM TODS 1986) admits a decrement against `inf` — the lowest
value the field could reach once every in-flight transaction resolves — rather
than against the current value, so no interleaving can drive stock negative
without holding a lock.

The demarcation protocol (Barbará-Millá & García-Molina, VLDB Journal 1994)
distributes that: each node gets a private **quota** it spends with no
coordination, and the centre is consulted only on refill. Safety is arithmetic:

```
sum(node quotas) + central_free + reserved + committed == total
```

`SkuLedger.invariant_holds()` checks exactly this, and `BoundedCounter._check`
asserts it after every mutation — an invariant that is never verified is only a
comment. The same structure appears in modern form as the Bounded Counter CRDT
(Balegas et al., EuroSys 2015), whose motivating example is distributed retail
stock.

## The trade-off being measured

Correctness is unconditional. The cost lands somewhere else: a node knows its
own quota, **not** the global total, so its view of stock is systematically
stale and conservative. That is not just an engineering cost — it is the input
E2 studies, because it reaches the models through two channels:

1. **Pricing** — the PPO agent's `inventory_level` state feature is the node's
   stale view, so it prices against stock that may not reflect reality.
2. **Forecasting** — unanticipated stockouts censor observed demand, so the TFT
   retrains on corrupted targets.

`staleness()` and the `staleness_units` column in the event log capture this at
decision time; it cannot be reconstructed afterwards.

## Why `eventual` is allowed to be wrong

`EventualPolicy` sets `allow_oversell=True`, which switches the invariant check
off and lets stock go negative. That is deliberate — it is E1's control
condition. A run in which `eventual` never oversells means the contention
scenario is too weak to be measuring anything, which is why
`test_eventual_oversells_under_contention` asserts that it *does*.

## Why Redis

`take()` is a compare-and-decrement, which is a race if done as separate GET and
DECRBY. Redis runs it as an atomic Lua script.

The reason this matters is experimental, not performance: if `strong_lock` were
implemented with the same hand-written locking as `escrow_quota`, E1 would be
comparing one author's code against itself. Delegating strong consistency to a
standard, independent primitive removes that objection.

Stated honestly: with a single uvicorn worker the GIL already serialises the
service's handlers, so SQLite would be sufficient for correctness. Redis earns
its place under multiple workers and as the independent primitive above.

## Running

```powershell
docker compose up -d redis                  # start the pool backend
uvicorn edge_system.inventory.service:app --port 8001

pytest edge_system/inventory/ -v            # Redis tests skip if none is up
```

Set `FYP_POOL_BACKEND=redis|sqlite|auto`. **The E1 scripts must set this
explicitly** — which primitive served the control arm is part of the result,
not something to discover at runtime.

## Stated limitations

**Refill is a fixed block, not a top-up to a target.** `EscrowQuotaPolicy._refill`
takes `max(qty x refill_multiple, min_refill)` units — a block sized on the order
that triggered it, rather than replenishing the node to a target derived from its
demand rate. The consequence is measurable: the coordination rate is governed by
`block size / order size`, so `quota_low_watermark` (refill *before* running dry)
barely changes anything — **75 refills at watermark 0.0 versus 76 at 0.9**, over
300 orders of 2 units. The watermark is implemented and tested but ships
**disabled by default**, rather than at a tuned-looking value that would imply a
mechanism which does not bite.

This also rules out the obvious explanation for escrow's fill-rate cost. The
93.3% (against strong lock's 98.7%) is **not** late refilling — it is structural.
A node cannot see units held in other nodes' quotas, so it refuses orders the
pool could have served, and refilling sooner would if anything worsen that by
locking away more stock. Refill-to-target would make the watermark a real lever,
but it is a change to the method under test and was deliberately not made in
order to fix a measurement problem.

## Endpoints

```
POST /reserve       {node, sku, qty, tick?, sim_date?} -> {granted, reservation_id, node_view, latency_ms}
POST /commit        {reservation_id}
POST /release       {reservation_id}          Saga compensation
POST /sweep                                   release everything past TTL
POST /replenish     {sku, qty}
GET  /stock/{sku}                             ground truth + per-node views + staleness
GET  /metrics                                 E1 counters
GET  /events        ?limit&kind
POST /admin/reset   ?policy                   wipe between experiment cells
```
