"""
The tick loop: drives the whole running system over HTTP for one scenario.

One tick is one M5 day, and each tick does exactly what a day in a deployed
retailer does:

    1. sweep     expire abandoned reservations (Saga compensation on a timer)
    2. replenish deliveries arrive for anything below its reorder point
    3. sell      quote a price per (SKU, channel), generate orders, reserve,
                 then commit or release
    4. check     feed the day's realised metrics to each node's drift monitor
    5. record    one row per (tick, channel) to the results CSV

Everything goes over HTTP to services running in their own processes. That is
slower than calling the policy objects directly, and it is the point: the
reservation latency E1 reports includes the serialisation, the loopback socket
and the simulated network delay, so it is a latency a deployed system would
actually pay rather than a function-call cost dressed up as one.

The driver holds no inventory state of its own. Whatever it reports about stock
it learned by asking the inventory service, exactly as a node would.
"""

from __future__ import annotations

import csv
import json
import random
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import httpx
import pandas as pd

from ..config import SYSTEM_CONFIG, channel_names, control_url, ensure_dirs, inventory_url
from .clock import SimClock, Tick
from .order_gen import OrderGenerator


@dataclass
class TickRecord:
    """One row of the results CSV: what one channel did on one simulated day."""

    tick: int
    sim_date: str
    channel: str
    policy: str
    delay_ms: float
    orders: int = 0
    units_requested: int = 0
    units_granted: int = 0
    units_refused: int = 0
    units_committed: int = 0
    units_released: int = 0
    revenue: float = 0.0
    mean_price: float = 0.0
    mean_tier: float = 0.0
    quote_failures: int = 0
    # Sync health, sampled from /stock across the SKUs this channel touched.
    mean_node_view: float = 0.0            # selling rights            (E1)
    mean_true_available: float = 0.0
    mean_staleness: float = 0.0            # truth - selling rights    (E1)
    mean_stock_estimate: float = 0.0       # what the node believes    (E2)
    mean_estimate_error: float = 0.0       # truth - belief, signed    (E2)
    reserve_latency_ms: float = 0.0
    model_generation: int = 0

    def as_row(self) -> Dict:
        return dict(self.__dict__)


class SimulationDriver:
    """Runs one scenario end to end and writes its results."""

    def __init__(self, *, run_id: str, pricing: Optional[str] = None,
                 verbose: bool = True) -> None:
        cfg = SYSTEM_CONFIG          # re-read at call time, per repo convention
        sim = cfg["sim"]

        self.run_id = run_id
        self.verbose = verbose
        self.pricing = pricing or sim.get("pricing", "node")
        self.channels = channel_names()
        self.rng = random.Random(sim["seed"])

        self.gen = OrderGenerator.from_csv(
            Path(cfg["paths"]["data_dir"]) / "rl_environment.csv",
            seed=sim["seed"], mean_basket=sim["mean_basket"],
            channel_weights={c["name"]: c["weight"] for c in cfg["channels"]},
        )
        self.skus = self.gen.skus(limit=sim.get("n_skus"))
        self.clock = SimClock.from_config(sim, available_dates=self.gen.dates)

        self.inventory = httpx.Client(base_url=inventory_url(), timeout=30.0)
        self.control = httpx.Client(base_url=control_url(), timeout=10.0)
        self.nodes = {
            c["name"]: httpx.Client(base_url=f"http://{cfg['host']}:{c['port']}",
                                    timeout=120.0)
            for c in cfg["channels"]
        }

        self.records: List[TickRecord] = []
        # Populated by write_results while the HTTP clients are still open, so
        # the caller can report it after tearing the stack down.
        self.summary: Dict = {}
        self.policy_name = "unknown"
        self.delay_ms = float(cfg["network"]["delay_ms"])
        self._targets: Dict[str, int] = {}
        self._reorder: Dict[str, int] = {}

    # ── Lifecycle ───────────────────────────────────────────────────────────

    def close(self) -> None:
        self.inventory.close()
        self.control.close()
        for c in self.nodes.values():
            c.close()

    def _log(self, msg: str) -> None:
        if self.verbose:
            print(f"[sim] {msg}", flush=True)

    def stock_up(self) -> None:
        """Open the store: initial delivery for every SKU in scope."""
        sim = SYSTEM_CONFIG["sim"]
        cover = sim["initial_cover_days"]
        replen = sim["replenish_cover_days"]
        frac = sim["reorder_point_frac"]

        for sku in self.skus:
            qty = self.gen.initial_stock(sku, cover)
            self._targets[sku] = self.gen.initial_stock(sku, replen)
            self._reorder[sku] = max(1, int(self._targets[sku] * frac))
            self.inventory.post("/replenish", json={"sku": sku, "qty": qty}
                                ).raise_for_status()

        health = self.inventory.get("/health").json()
        self.policy_name = health.get("policy", "unknown")
        self._log(f"stocked {len(self.skus)} SKUs | policy={self.policy_name} "
                  f"| backend={health.get('backend')}")

    # ── The loop ────────────────────────────────────────────────────────────

    def run(self) -> Path:
        self.stock_up()
        self._log(f"running {len(self.clock)} ticks ({self.clock.span}) "
                  f"| pricing={self.pricing}")

        for tick in self.clock:
            self._apply_network_schedule(tick)
            self.inventory.post("/sweep")
            self._replenish(tick)
            per_channel = {ch: TickRecord(tick=tick.index, sim_date=tick.date_str,
                                          channel=ch, policy=self.policy_name,
                                          delay_ms=self.delay_ms)
                           for ch in self.channels}
            for sku in self.skus:
                self._trade_sku(sku, tick, per_channel)
            self._finalise(per_channel)
            self._check_drift(tick, per_channel)
            self.records.extend(per_channel.values())

            if tick.index % 10 == 0 or tick.is_last:
                sold = sum(r.units_committed for r in per_channel.values())
                self._log(f"tick {tick.index:>4} {tick.date_str} sold={sold}")

        path = self.write_results()
        self._log(f"done in {self.clock.elapsed_s:.1f}s -> {path}")
        return path

    def _apply_network_schedule(self, tick: Tick) -> None:
        """
        Move the network dial part-way through a run.

        Partition **onset** is the case that separates the policies. Cut a node
        off from tick 0 and escrow_quota looks exactly like strong_lock, because
        acquiring the first quota needs a hop that never succeeds. Cut it off at
        tick 15 and the node keeps trading on rights it already holds until they
        run out - which is the availability argument for escrow, and it is only
        visible if the link drops after the escrow was granted.
        """
        for step in SYSTEM_CONFIG["network"].get("schedule", ()):
            if int(step.get("at_tick", -1)) != tick.index:
                continue
            body = {k: v for k, v in step.items() if k != "at_tick"}
            try:
                self.inventory.post("/admin/network", json=body).raise_for_status()
                self._log(f"tick {tick.index}: network -> {body}")
                if "delay_ms" in body:
                    self.delay_ms = float(body["delay_ms"])
            except httpx.HTTPError as exc:
                self._log(f"tick {tick.index}: network change FAILED ({exc})")

    def _replenish(self, tick: Tick) -> None:
        """Fixed (s,S) reordering. Out of scope for the RL agent, by design."""
        for sku in self.skus:
            try:
                stock = self.inventory.get(f"/stock/{sku}").json()
            except httpx.HTTPError:
                continue
            if stock["true_available"] < self._reorder[sku]:
                qty = self._targets[sku] - stock["true_available"]
                if qty > 0:
                    self.inventory.post("/replenish", json={
                        "sku": sku, "qty": int(qty),
                        "tick": tick.index, "sim_date": tick.date_str})

    def _trade_sku(self, sku: str, tick: Tick, out: Dict[str, TickRecord]) -> None:
        row = self.gen.row(sku, tick.date)
        if row is None:
            return

        try:
            stock = self.inventory.get(f"/stock/{sku}").json()
        except httpx.HTTPError:
            stock = None

        for ch in self.channels:
            rec = out[ch]
            quote = self._quote(ch, sku, tick, row)
            if quote is None:
                rec.quote_failures += 1
                continue
            price, tier, generation = quote
            rec.mean_price += price
            rec.mean_tier += tier
            rec.model_generation = max(rec.model_generation, generation)

            if stock is not None:
                rec.mean_node_view += stock["node_views"].get(ch, 0)
                rec.mean_true_available += stock["true_available"]
                rec.mean_staleness += stock["staleness"].get(ch, 0)
                rec.mean_stock_estimate += stock["stock_estimates"].get(ch, 0)
                rec.mean_estimate_error += stock["estimate_errors"].get(ch, 0)

            for order in self.gen.orders(sku, tick.date, ch, price, tick=tick.index):
                self._place(order, rec)

    def _quote(self, channel: str, sku: str, tick: Tick, row: Dict):
        """
        Ask the channel's own agent what to charge.

        In `node` mode the edge node fetches its **own escrow view** of stock and
        prices against that - the stale signal is never supplied by the driver,
        because supplying it would make the staleness an artefact of the test
        harness rather than a property of the system under test.
        """
        if self.pricing == "static":
            return row["base_price"], 0.0, 0

        try:
            r = self.nodes[channel].get(
                "/price", params={"sku": sku, "sim_date": tick.date_str})
            r.raise_for_status()
            body = r.json()
        except httpx.HTTPError:
            return None
        return (float(body["price"]), float(body["tier"]),
                int(body.get("model_generation", 0)))

    def _place(self, order, rec: TickRecord) -> None:
        rec.orders += 1
        rec.units_requested += order.qty
        try:
            r = self.inventory.post("/reserve", json={
                "node": order.channel, "sku": order.sku, "qty": order.qty,
                "tick": order.tick, "sim_date": order.sim_date})
            r.raise_for_status()
            res = r.json()
        except httpx.HTTPError:
            rec.units_refused += order.qty
            return

        rec.reserve_latency_ms += float(res.get("latency_ms", 0.0))
        if not res["granted"]:
            rec.units_refused += order.qty
            return

        rec.units_granted += order.qty
        if self.rng.random() < SYSTEM_CONFIG["sim"]["abandon_rate"]:
            self.inventory.post("/release", json={"reservation_id": res["reservation_id"]})
            rec.units_released += order.qty
        else:
            self.inventory.post("/commit", json={"reservation_id": res["reservation_id"]})
            rec.units_committed += order.qty
            rec.revenue += order.value

    def _finalise(self, per_channel: Dict[str, TickRecord]) -> None:
        """Turn the running sums into per-tick means."""
        for rec in per_channel.values():
            quoted = max(len(self.skus) - rec.quote_failures, 1)
            rec.mean_price /= quoted
            rec.mean_tier /= quoted
            rec.mean_node_view /= quoted
            rec.mean_true_available /= quoted
            rec.mean_staleness /= quoted
            rec.mean_stock_estimate /= quoted
            rec.mean_estimate_error /= quoted
            rec.reserve_latency_ms /= max(rec.orders, 1)

    def _check_drift(self, tick: Tick, per_channel: Dict[str, TickRecord]) -> None:
        """
        Hand the day's realised outcome to each node's local drift monitor.

        `profit_index` is the node's revenue against what it would have taken at
        the unchanged base price - the same ratio-to-reference shape the offline
        pipeline uses, so the monitor's calibrated threshold still means what it
        was calibrated to mean.
        """
        if self.pricing != "node":
            return
        for ch, rec in per_channel.items():
            reference = sum(
                self.gen.expected_units(sku, tick.date, None)
                * (self.gen.base_price(sku, tick.date) or 0.0)
                * SYSTEM_CONFIG["channels"][self.channels.index(ch)]["weight"]
                for sku in self.skus
            )
            profit_index = rec.revenue / reference if reference > 0 else None
            try:
                self.nodes[ch].post("/check", json={
                    "tick": tick.index, "sim_date": tick.date_str,
                    "profit_index": profit_index, "blocking": False})
            except httpx.HTTPError:
                pass

    # ── Output ──────────────────────────────────────────────────────────────

    def write_results(self) -> Path:
        ensure_dirs()
        out_dir = Path(SYSTEM_CONFIG["paths"]["results"])
        csv_path = out_dir / f"sim_ticks_{self.run_id}.csv"

        rows = [r.as_row() for r in self.records]
        with csv_path.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

        self.summary = self.summarise()
        (out_dir / f"sim_summary_{self.run_id}.json").write_text(
            json.dumps(self.summary, indent=2, default=str), encoding="utf-8")

        try:
            self.inventory.post("/admin/flush")
        except httpx.HTTPError:
            pass
        return csv_path

    def summarise(self) -> Dict:
        df = pd.DataFrame([r.as_row() for r in self.records])
        try:
            metrics = self.inventory.get("/metrics").json()
        except httpx.HTTPError:
            metrics = {}

        totals = {c: int(df[c].sum()) for c in
                  ("orders", "units_requested", "units_granted",
                   "units_refused", "units_committed", "units_released")}
        return {
            "run_id": self.run_id,
            "policy": self.policy_name,
            "pricing": self.pricing,
            "delay_ms": self.delay_ms,
            "network": SYSTEM_CONFIG["network"],
            "ticks": len(self.clock),
            "skus": len(self.skus),
            "span": self.clock.span,
            "wall_seconds": self.clock.elapsed_s,
            **totals,
            "revenue": float(df["revenue"].sum()),
            "fill_rate": totals["units_granted"] / max(totals["units_requested"], 1),
            "mean_staleness_units": float(df["mean_staleness"].mean()),
            "mean_estimate_error_units": float(df["mean_estimate_error"].mean()),
            "mean_abs_estimate_error": float(df["mean_estimate_error"].abs().mean()),
            "inventory_metrics": metrics,
        }
