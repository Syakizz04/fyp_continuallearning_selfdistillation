"""
Does the "critic" pricing drift signal catch a broken pricer better than the
revenue index it replaced?

Rebuilt on 2026-11-22. The original was a scratch script that was never
committed and went down with the old laptop. Its numbers survive only in
commit 847605f and in LocalInference.pricing_outcome's docstring: critic
13/15 and 12/15 against revenue 4/15 and 1/15, at 2-3.5 false retrains a
year. Those were measured on the old base_v4 pricer.

What it does
------------
It replays the demo's trading in-process, with no services, using the same
code the nodes run:
  * LocalInference.price: the pricer's quote;
  * LocalInference.pricing_outcome: the critic reading;
  * LocalDriftMonitor in rolling mode: the detector, with the demo's
    window 7, warm-up 28 and k 2.5.
Both signals feed separate monitors on the same days, orders and prices, so
the comparison is paired.

Worlds
  normal    : the pricer prices, demand follows the dataset elasticity.
  stuck     : from the onset date, every price is base x 0.90 (the lowest
              tier), whatever the pricer says. A broken pricer.
  sensitive : from the onset date, demand responds to price with 2x the
              elasticity the pricer was trained on. A changed world.

Trials
  For each of N onset dates across 2013-2014, start a fresh monitor at
  WARMUP_LEAD days before onset (enough to calibrate), inject the fault at
  onset, and say it was CAUGHT if the detector fires within HORIZON days.
  The detection delay is days from onset to the first fire.

False alarms
  The normal world over the whole window, continuously. Each fire is treated
  as the node would treat a retrain: the line is re-learned (recalibrate_rl).
  The pricer itself is NOT swapped, since that would change what is measured.
  Reported as fires per year per channel.

Simplifications (all apply equally to both signals)
  * Stock is one true pool per SKU with the driver's (s,S) replenishment.
    The node prices against true stock, not its escrow view, so there is no
    sync staleness. That keeps refusals to real stock-outs.
  * The SKUs are the demo's: top sellers, at most 21-day gaps, 5 per
    department, 10 SKUs.

Usage
    python -m experiments.critic_compare                # 15 onsets
    python -m experiments.critic_compare --onsets 5     # quicker
Writes outputs/system/critic_compare/{trials.csv,false_alarms.csv,summary.json}.
"""
from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from edge_system.config import SYSTEM_CONFIG, channel_names
from edge_system.sim.order_gen import OrderGenerator

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = PROJECT_ROOT / "outputs" / "system" / "critic_compare"

WINDOW = ("2013-01-01", "2014-12-31")
WARMUP_LEAD = 35        # days of normal trading before onset (28 warm-up + one window)
HORIZON = 28            # days after onset in which a fire counts as a catch
STUCK_TIER = -0.10
SENSITIVITY = 2.0
SIGNALS = ("revenue", "critic")


class World:
    """One replayed shop: shared stock, three channels, one pricer."""

    def __init__(self, inf, gen: OrderGenerator, skus: List[str], *, seed: int = 42):
        self.inf, self.gen, self.skus = inf, gen, skus
        sim = SYSTEM_CONFIG["sim"]
        self.channels = channel_names()
        self.weights = {c["name"]: float(c["weight"]) for c in SYSTEM_CONFIG["channels"]}
        self.abandon = float(sim["abandon_rate"])
        self.rng = random.Random(seed)
        cover0 = 10.0                                   # the demo's initial_cover_days
        self.target = {s: gen.initial_stock(s, sim["replenish_cover_days"]) for s in skus}
        self.reorder = {s: max(1, int(self.target[s] * sim["reorder_point_frac"]))
                        for s in skus}
        self.stock = {s: gen.initial_stock(s, cover0) for s in skus}

    def day(self, tick: int, date: pd.Timestamp, fault: Optional[str]) -> Dict[str, Dict]:
        """Trade one day. Returns, per channel, the revenue reading's parts and
        the per-SKU outcomes the critic reads."""
        d = date.strftime("%Y-%m-%d")
        for s in self.skus:                              # (s,S), start of day
            if self.stock[s] < self.reorder[s]:
                self.stock[s] = self.target[s]
        rev = {ch: 0.0 for ch in self.channels}
        outcomes: Dict[str, List[Dict]] = {ch: [] for ch in self.channels}
        for s in self.skus:
            row = self.gen.row(s, date)
            if row is None:
                continue
            q = self.inf.price(s, sim_date=d, inventory_level=float(self.stock[s]))
            price = q["price"]
            if fault == "stuck":
                price = q["base_price"] * (1.0 + STUCK_TIER)
            for ch in self.channels:
                o = {"sku": s, "price": float(price), "requested": 0, "sold": 0, "refused": 0}
                for order in self._orders(s, date, ch, price, tick, row, fault):
                    o["requested"] += order.qty
                    if self.stock[s] < order.qty:
                        o["refused"] += order.qty
                        continue
                    self.stock[s] -= order.qty
                    if self.rng.random() < self.abandon:
                        self.stock[s] += order.qty         # released
                        continue
                    o["sold"] += order.qty
                    rev[ch] += order.qty * float(price)
                outcomes[ch].append(o)
        out = {}
        for ch in self.channels:
            reference = sum(self.gen.expected_units(s, date, None)
                            * (self.gen.base_price(s, date) or 0.0) * self.weights[ch]
                            for s in self.skus)
            out[ch] = {"revenue": rev[ch], "reference": reference, "pricing": outcomes[ch]}
        return out

    def _orders(self, sku, date, ch, price, tick, row, fault):
        if fault != "sensitive":
            return self.gen.orders(sku, date, ch, price, tick=tick)
        # 2x elasticity: scale the channel weight so expected units match
        # demand * ratio**(2e). Same content-seeded arrival process.
        base = row["base_price"]
        if base <= 0:
            return self.gen.orders(sku, date, ch, price, tick=tick)
        ratio = max(price, 1e-6) / base
        extra = ratio ** ((SENSITIVITY - 1.0) * row["elasticity"])
        return self.gen.orders(sku, date, ch, price, tick=tick,
                               weight=self.weights[ch] * extra)


def make_monitors(inf):
    from edge_system.edge.monitor_local import LocalDriftMonitor
    return {(ch, sig): LocalDriftMonitor(ch, inf.calibration)
            for ch in channel_names() for sig in SIGNALS}


def feed(inf, monitors, world_out, tick: int, d: str) -> Dict[tuple, bool]:
    """The node's /check, both signals. Returns which (channel, signal) fired."""
    fired = {}
    for ch, got in world_out.items():
        ref = got["reference"]
        rev_idx = got["revenue"] / ref if ref > 0 else None
        m = monitors[(ch, "revenue")]
        fired[(ch, "revenue")] = bool(m.update(tick=tick, sim_date=d, profit_index=rev_idx,
                                               revenue=got["revenue"], reference=ref))
        res = inf.pricing_outcome(sim_date=d, outcomes=got["pricing"],
                                  share=world_weights()[ch])
        if res is None:
            actual = expected = idx = None
        else:
            actual, expected = res
            idx = actual / expected if expected > 0 else None
        m = monitors[(ch, "critic")]
        fired[(ch, "critic")] = bool(m.update(tick=tick, sim_date=d, profit_index=idx,
                                              revenue=actual, reference=expected))
    for key, f in fired.items():
        if f:
            monitors[key].recalibrate_rl()     # what a retrain does to the line
    return fired


def world_weights() -> Dict[str, float]:
    return {c["name"]: float(c["weight"]) for c in SYSTEM_CONFIG["channels"]}


def dates_between(gen: OrderGenerator, lo, hi) -> List[pd.Timestamp]:
    ds = sorted(set(gen.dates))
    return [d for d in ds if pd.Timestamp(lo) <= d <= pd.Timestamp(hi)]


def run_trial(inf, gen, skus, onset: pd.Timestamp, fault: str, all_dates) -> List[Dict]:
    start = onset - pd.Timedelta(days=WARMUP_LEAD)
    end = onset + pd.Timedelta(days=HORIZON - 1)
    days = [d for d in all_dates if start <= d <= end]
    world, mons = World(inf, gen, skus), make_monitors(inf)
    first: Dict[tuple, Optional[int]] = {k: None for k in mons}
    pre: Dict[tuple, int] = {k: 0 for k in mons}
    for d in days:
        tick = (d - pd.Timestamp(WINDOW[0])).days
        active = fault if d >= onset else None
        fired = feed(inf, mons, world.day(tick, d, active), tick, d.strftime("%Y-%m-%d"))
        for k, f in fired.items():
            if not f:
                continue
            if d < onset:
                pre[k] += 1
            elif first[k] is None:
                first[k] = (d - onset).days
    rows = []
    for (ch, sig), delay in first.items():
        rows.append({"fault": fault, "onset": onset.strftime("%Y-%m-%d"), "channel": ch,
                     "signal": sig, "caught": delay is not None, "delay_days": delay,
                     "fires_before_onset": pre[(ch, sig)]})
    return rows


def run_false_alarms(inf, gen, skus, all_dates) -> List[Dict]:
    days = [d for d in all_dates if pd.Timestamp(WINDOW[0]) <= d <= pd.Timestamp(WINDOW[1])]
    world, mons = World(inf, gen, skus), make_monitors(inf)
    rows = []
    for d in days:
        tick = (d - pd.Timestamp(WINDOW[0])).days
        ds = d.strftime("%Y-%m-%d")
        fired = feed(inf, mons, world.day(tick, d, None), tick, ds)
        for (ch, sig), f in fired.items():
            if f:
                rows.append({"date": ds, "channel": ch, "signal": sig})
    years = len(days) / 365.25
    return rows, years


def main() -> None:
    global SENSITIVITY, OUT_DIR
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--onsets", type=int, default=15)
    ap.add_argument("--skip-false-alarms", action="store_true")
    ap.add_argument("--sensitivity", type=float, default=2.0,
                    help="elasticity multiplier in the 'sensitive' world (default 2)")
    ap.add_argument("--out", default=None, help="output dir (default outputs/system/critic_compare)")
    args = ap.parse_args()
    SENSITIVITY = args.sensitivity
    if args.out:
        OUT_DIR = Path(args.out)

    # The demo's detector settings. Mutate in place: helpers read at call time.
    SYSTEM_CONFIG["models"].update({"rl_detector": "rolling", "rl_signal": "critic"})
    sim = SYSTEM_CONFIG["sim"]
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    from edge_system.edge.inference import LocalInference
    t0 = time.perf_counter()
    inf = LocalInference("replay")
    inf.load()
    a, b = inf.critic_line()
    print(f"[critic_compare] models from {inf.ckpt_dir} loaded in {time.perf_counter()-t0:.0f}s;"
          f" critic line r = {a:.4f} + {b:.4f} V")

    gen = OrderGenerator.from_csv(Path(SYSTEM_CONFIG["paths"]["data_dir"]) / "rl_environment.csv",
                                  seed=sim["seed"], mean_basket=sim["mean_basket"],
                                  channel_weights=world_weights())
    all_dates = sorted(set(gen.dates))
    demo_end = pd.Timestamp("2013-01-01") + pd.Timedelta(days=399)   # demo: 400 ticks
    skus = gen.skus(limit=10, order="top", max_gap_days=21, per_dept=5,
                    window=(pd.Timestamp("2013-01-01"), demo_end))
    print(f"[critic_compare] SKUs: {skus}")

    first_onset = pd.Timestamp(WINDOW[0]) + pd.Timedelta(days=WARMUP_LEAD)
    last_onset = pd.Timestamp(WINDOW[1]) - pd.Timedelta(days=HORIZON)
    onsets = pd.date_range(first_onset, last_onset, periods=args.onsets).normalize()

    trials: List[Dict] = []
    for fault in ("stuck", "sensitive"):
        for i, onset in enumerate(onsets, 1):
            t = time.perf_counter()
            trials += run_trial(inf, gen, skus, onset, fault, all_dates)
            print(f"[critic_compare] {fault} {i}/{len(onsets)} onset {onset.date()} "
                  f"({time.perf_counter()-t:.0f}s)", flush=True)
    tr = pd.DataFrame(trials)
    tr.to_csv(OUT_DIR / "trials.csv", index=False)

    summary: Dict = {"checkpoints": inf.ckpt_dir, "critic_line": [a, b], "skus": skus,
                     "onsets": [o.strftime("%Y-%m-%d") for o in onsets],
                     "horizon_days": HORIZON, "sensitivity": SENSITIVITY, "detector": {
                         k: SYSTEM_CONFIG["models"][k] for k in
                         ("rl_window_days", "rl_warmup_days", "rl_k_sigma", "relearn_min_days")},
                     "caught": {}, "median_delay_days": {}}
    for (fault, ch, sig), g in tr.groupby(["fault", "channel", "signal"]):
        key = f"{fault}/{ch}/{sig}"
        summary["caught"][key] = f"{int(g['caught'].sum())}/{len(g)}"
        dl = g["delay_days"].dropna()
        summary["median_delay_days"][key] = float(dl.median()) if len(dl) else None

    if not args.skip_false_alarms:
        t = time.perf_counter()
        fa, years = run_false_alarms(inf, gen, skus, all_dates)
        pd.DataFrame(fa, columns=["date", "channel", "signal"]).to_csv(
            OUT_DIR / "false_alarms.csv", index=False)
        per_year = {}
        for ch in channel_names():
            for sig in SIGNALS:
                n = sum(1 for r in fa if r["channel"] == ch and r["signal"] == sig)
                per_year[f"{ch}/{sig}"] = round(n / years, 2)
        summary["false_alarms_per_year"] = per_year
        print(f"[critic_compare] false-alarm year(s) done ({time.perf_counter()-t:.0f}s)")

    (OUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2))

    print("\n== caught within %d days (median delay) ==" % HORIZON)
    for fault in ("stuck", "sensitive"):
        for ch in channel_names():
            cells = [f"{sig} {summary['caught'][f'{fault}/{ch}/{sig}']}"
                     f" ({summary['median_delay_days'][f'{fault}/{ch}/{sig}']})"
                     for sig in SIGNALS]
            print(f"  {fault:9s} {ch:12s} " + "   ".join(cells))
    if "false_alarms_per_year" in summary:
        print("== false alarms per year, normal trading ==")
        for ch in channel_names():
            print(f"  {ch:12s} " + "   ".join(
                f"{sig} {summary['false_alarms_per_year'][f'{ch}/{sig}']}" for sig in SIGNALS))
    print(f"\nwritten to {OUT_DIR}")


if __name__ == "__main__":
    main()
