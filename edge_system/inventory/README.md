# Inventory sync service

Shared stock across three sales channels contending for one pool per SKU.
It replaces the toy `_simulate_inventory()` in `dataset_generator/m5/build_m5.py`
— an (s,S) loop with instant next-day replenishment that was the only fully
fabricated part of the dataset — with a real reservation algorithm driven by
real demand.

## Setting: omnichannel retail, not pure e-commerce

The three channels are a physical till (`pos`), an online storefront (`web`) and a
marketplace listing (`marketplace`). They share one stock pool, which is what
makes the setting *omnichannel* rather than multichannel: the channels are not
independently stocked, so a unit sold at the till is a unit the website can no
longer promise. That shared pool is the entire source of the contention this
service manages.

This framing matches the data. M5 is Walmart — a physical grocery retailer — so a
dominant in-store channel is the correct shape, and Walmart is close to the
canonical omnichannel example (store + walmart.com + marketplace on one
inventory).

Two things are stated as assumptions rather than claimed as facts:

- **M5 records daily unit sales, not the channel a unit sold through.** The
  channel split, and the discrete order arrivals built from it, are a modelling
  overlay (`edge_system/sim/order_gen.py`), calibrated so that aggregating
  generated orders recovers the observed daily total. The daily total is real;
  the decomposition is not.
- **The channel weights (0.55 / 0.35 / 0.10) are an experimental choice.**
  Walmart's real online share over the M5 window was roughly 3%. At that split
  the pool is effectively single-channel, nothing contends, and every sync policy
  measures the same thing. The weights are set to put the pool under contention
  because contention is the regime under study — the same reasoning that governs
  `sim.initial_cover_days`.

## What this service is for

**This is apparatus, not the contribution.** The project's contribution is on the
continual-learning side — SDFT, and whether replay-free CL survives a corrupted
training signal. This service exists to *produce* that corruption in a controlled,
measurable way.

That is a real job, and it is the reason the code is held to the standard it is.
The treatment E2 applies is a fill rate, and a fill rate is only a credible
treatment if the mechanism that produced it is correct: if the escrow arithmetic
were wrong, the censoring E2 studies would be an artifact of a bug rather than a
property of the sync policy. Hence the invariant assertions and the property
tests. **Their role is to make the apparatus trustworthy, not to claim a result.**

**The algorithms are prior work.** Escrow is O'Neil (1986); the per-node quota is
the demarcation protocol (1994); the modern restatement is the Bounded Counter
CRDT (2015). They are implemented here in order to be *measured*, not proposed.

What is new is the measurement itself: the database literature evaluates these
protocols on throughput and correctness, never on **what they do to the training
signal of a model downstream**. A node that refuses orders it cannot see stock
for is destroying its own training data, and nobody has costed that. That cost is
the bridge from this service to the CL experiment — and it is a bridge, not a
destination. E1's fill rates (82.7% / 71.7%) are not the finding; they are the
treatment levels E2 consumes.

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
stale and conservative. That is not just an engineering cost — it is **the entire
reason this service is in the project**, because it reaches the models through two
channels:

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
