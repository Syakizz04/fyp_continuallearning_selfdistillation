"""
E1 - sync correctness and cost.

    python -m experiments.exp_sync                      # full sweep
    python -m experiments.exp_sync --ticks 20 --n-skus 10 --delays 0 50   # pilot

Sweeps `policy x one-way delay` and measures what each policy buys and what it
costs. No models are loaded, so this needs no GPU and runs in minutes - which is
why it runs before any GPU time is committed to E2.

## The claim under test

`escrow_quota` should attain `strong_lock`'s zero oversell at close to
`eventual`'s latency, because most orders never leave the node. If that holds,
the cost has to show up somewhere else, and the interesting result is *where*.

## Why the cells are comparable

Three things are held fixed across every cell, and each one would otherwise
confound the comparison:

* **The order stream.** Seeded from (seed, tick, sku, channel) content rather
  than a running RNG, so every cell sees byte-identical customers regardless of
  policy, hop count, or the order the driver walked the SKUs in.
* **The pool backend.** Pinned to Redis. `strong_lock`'s claim to be a credible
  control arm rests on its atomicity coming from a Lua compare-and-decrement
  rather than from our own escrow code, so a silent fallback to SQLite would
  invalidate the comparison it exists to make.
* **The services.** Started once and reset between cells, so no cell differs by
  process start-up state.

Delay is injected rather than real. It is the independent variable and has to be
a dial we set - see edge_system/sim/network.py.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List

import httpx
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from edge_system.config import SYSTEM_CONFIG, ensure_dirs, inventory_url  # noqa: E402
from edge_system.run_system import (ServiceProcess, build_services,  # noqa: E402
                                    check_redis)

POLICIES = ["strong_lock", "escrow_quota", "eventual"]
DELAYS_MS = [0.0, 50.0, 200.0, 1000.0]


def run_cell(policy: str, delay_ms: float, *, gen=None) -> Dict:
    """One (policy, delay) cell against the already-running services."""
    from edge_system.sim.driver import SimulationDriver

    httpx.post(f"{inventory_url()}/admin/reset", params={"policy": policy},
               timeout=30.0).raise_for_status()
    httpx.post(f"{inventory_url()}/admin/network", timeout=30.0,
               json={"delay_ms": delay_ms, "jitter_ms": 0.0,
                     "partition": []}).raise_for_status()

    health = httpx.get(f"{inventory_url()}/health", timeout=10.0).json()
    if health["policy"] != policy:
        raise SystemExit(f"asked for {policy!r}, service is running "
                         f"{health['policy']!r} - refusing to record this cell")
    if health["backend"] != "RedisPoolStore":
        raise SystemExit(f"expected Redis, got {health['backend']!r}")

    SYSTEM_CONFIG["sync"]["policy"] = policy
    SYSTEM_CONFIG["network"]["delay_ms"] = delay_ms

    run_id = f"e1_{policy}_{int(delay_ms)}ms"
    driver = SimulationDriver(run_id=run_id, pricing="static", verbose=False)
    if gen is not None:
        # Reuse the parsed frame across cells: it is 184k rows and identical
        # every time, so re-reading it 12 times is pure overhead.
        driver.gen = gen
        driver.skus = gen.skus(limit=SYSTEM_CONFIG["sim"].get("n_skus"))
    started = time.perf_counter()
    try:
        driver.run()
        summary = driver.summary
    finally:
        driver.close()

    m = summary["inventory_metrics"]
    return {
        "policy": policy,
        "delay_ms": delay_ms,
        "wall_seconds": time.perf_counter() - started,
        "orders": summary["orders"],
        "units_requested": summary["units_requested"],
        "units_granted": summary["units_granted"],
        "units_committed": summary["units_committed"],
        "fill_rate": summary["fill_rate"],
        "revenue": summary["revenue"],
        # The safety property.
        "oversell_units": m["oversell_units"],
        "oversell_events": m["oversell_events"],
        "oversell_rate": m["oversell_rate"],
        # The cost of that safety.
        "rejection_rate": m["rejection_rate"],
        # The coordination traded away.
        "central_roundtrips": m["central_roundtrips"],
        "roundtrips_per_reserve": m["roundtrips_per_reserve"],
        "quota_refills": m["quota_refills"],
        "latency_p50_ms": m["latency_p50_ms"],
        "latency_p99_ms": m["latency_p99_ms"],
        # The signal E2 inherits.
        "mean_staleness_units": summary["mean_staleness_units"],
        "mean_abs_estimate_error": summary["mean_abs_estimate_error"],
        "backend": health["backend"],
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="E1: sync policy x delay sweep.")
    ap.add_argument("--ticks", type=int, default=60)
    ap.add_argument("--n-skus", type=int, default=25)
    ap.add_argument("--cover-days", type=float, default=7.0,
                    help="opening stock. Low enough that the pool is contended - "
                         "with generous cover nothing runs out and every policy "
                         "looks identical")
    ap.add_argument("--policies", nargs="*", default=POLICIES)
    ap.add_argument("--delays", nargs="*", type=float, default=DELAYS_MS)
    ap.add_argument("--out", default="e1_sync")
    args = ap.parse_args(argv)

    SYSTEM_CONFIG["sim"].update({
        "max_ticks": args.ticks, "n_skus": args.n_skus, "pricing": "static",
        "initial_cover_days": args.cover_days,
        "replenish_cover_days": args.cover_days,
    })
    # Pinned, not discovered: which primitive served strong_lock is part of the
    # result.
    SYSTEM_CONFIG["redis"]["backend"] = "redis"
    check_redis()
    ensure_dirs()

    cells = [(p, d) for p in args.policies for d in args.delays]
    print(f"E1: {len(cells)} cells | {args.ticks} ticks x {args.n_skus} SKUs "
          f"| cover {args.cover_days}d")

    services: List[ServiceProcess] = build_services(with_models=False,
                                                    strategy="sdft")
    log_dir = Path(SYSTEM_CONFIG["paths"]["state_dir"]) / "logs" / args.out
    log_dir.mkdir(parents=True, exist_ok=True)

    started: List[ServiceProcess] = []
    rows: List[Dict] = []
    try:
        for svc in services:
            svc.preflight()
        for svc in services:
            svc.start(log_dir)
            started.append(svc)
            if not svc.wait_ready(120.0):
                print(f"{svc.name} failed to start:\n{svc.tail()}", file=sys.stderr)
                return 1

        from edge_system.sim.order_gen import OrderGenerator
        gen = OrderGenerator.from_csv(
            Path(SYSTEM_CONFIG["paths"]["data_dir"]) / "rl_environment.csv",
            seed=SYSTEM_CONFIG["sim"]["seed"],
            mean_basket=SYSTEM_CONFIG["sim"]["mean_basket"],
            channel_weights={c["name"]: c["weight"]
                             for c in SYSTEM_CONFIG["channels"]},
        )

        for i, (policy, delay) in enumerate(cells, 1):
            print(f"  [{i:>2}/{len(cells)}] {policy:<13} {delay:>6.0f} ms ... ",
                  end="", flush=True)
            row = run_cell(policy, delay, gen=gen)
            rows.append(row)
            print(f"oversell={row['oversell_units']:>4} "
                  f"fill={row['fill_rate']:>6.1%} "
                  f"p50={row['latency_p50_ms']:>7.1f}ms "
                  f"hops/order={row['roundtrips_per_reserve']:.2f} "
                  f"({row['wall_seconds']:.0f}s)")
    finally:
        for svc in reversed(started):
            svc.stop()

    out_dir = Path(SYSTEM_CONFIG["paths"]["results"])
    df = pd.DataFrame(rows)
    csv_path = out_dir / f"{args.out}.csv"
    df.to_csv(csv_path, index=False)
    (out_dir / f"{args.out}_config.json").write_text(json.dumps({
        "ticks": args.ticks, "n_skus": args.n_skus,
        "cover_days": args.cover_days, "policies": args.policies,
        "delays_ms": args.delays, "seed": SYSTEM_CONFIG["sim"]["seed"],
        "backend": rows[0]["backend"] if rows else None,
    }, indent=2))

    print(f"\n-> {csv_path}\n")
    print(df.pivot(index="delay_ms", columns="policy",
                   values="oversell_units").to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
