"""
How well do the node's pricing drift signals catch a pricing problem?

History
-------
The original replay behind commit 847605f was a scratch script that was never
committed and was lost with the old laptop. Its claims were critic 13/15 and
12/15 against revenue 4/15 and 1/15. Version 1 of this rebuild (2bd39ee)
reproduced the broken-pricer result but not the changed-world one.

A review on 2026-11-22 found three things:
  * The critic is redundant with a one-number markup index (below). Price is
    not in the pricer's observation, and the critic is asked with hindsight
    units, so the ratio reduces to a per-unit-margin gauge.
  * "Stuck at -10%" was the single most favourable fault for the critic.
  * The trials overlapped, and there were no intervals.

This version (v2) answers all three.

Signals, all fed the same days, orders and prices, so every comparison is paired
  revenue : channel revenue over expected revenue at base price. The original
            signal, built as SimulationDriver._check_drift does.
  critic  : the day's training reward over a + b V(s), with V(s) asked with
            hindsight demand. LocalInference.pricing_outcome, the current
            default for models.rl_signal.
  markup  : the day's training reward over M x sum(base_price x sold) / 1000.
            M is one number fitted on the training window from the pricer's
            own prices. No network. This is the baseline the critic has to
            beat to justify itself.
Each feeds its own LocalDriftMonitor in rolling mode, with the demo's window
7, warm-up 28 and k 2.5.

Faults (from onset)
  none        : no fault. A "catch" is a false alarm. Every other row is read
                against this one.
  stuck-10    : every price is base x 0.90, whatever the pricer says.
  stuck0      : every price is the base price (a fallback-to-list-price bug).
  stuck+10    : every price is base x 1.10.
  sensitive2  : demand responds to price with 2x the dataset elasticity.
  sensitive6  : the same at 6x.

Trials
  Non-overlapping blocks of WARMUP_LEAD + HORIZON days across 2013-2015. Each
  has a fresh world and fresh monitors, so the blocks share no state and no
  order stream. Caught = the detector fires within HORIZON days of onset.
  Shown with a Wilson 95% interval. Signals are compared with an exact
  McNemar test on the paired trials.

False alarms
  The normal world over the whole window, continuously. Each fire re-learns
  the line (recalibrate_rl), as a retrain would; the pricer itself is not
  swapped. Reported per year with an exact Poisson 95% interval.

Simplifications (all apply equally to every signal)
  * Stock is one true pool per SKU with the driver's (s,S) replenishment. The
    node prices against true stock, so there is no escrow staleness.
  * Channels are served in a fixed order within a day (pos, web,
    marketplace), as SimulationDriver does, so refusals fall on the later
    channels first.
  * The SKUs are the demo's: top sellers, gaps of at most 21 days, 5 per
    department, 10 SKUs, chosen on the demo's 2013 window.

Usage
    python -m experiments.critic_compare
Writes outputs/system/critic_compare/{trials.csv,false_alarms.csv,summary.json}.
"""
from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats

from edge_system.config import SYSTEM_CONFIG, channel_names
from edge_system.sim.order_gen import OrderGenerator

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = PROJECT_ROOT / "outputs" / "system" / "critic_compare"

WINDOW = ("2013-01-01", "2015-12-31")
WARMUP_LEAD = 35        # days of normal trading before onset (28 warm-up + one window)
HORIZON = 28            # days after onset in which a fire counts as a catch
SIGNALS = ("revenue", "critic", "markup")
FAULTS: Dict[str, Optional[Tuple[str, float]]] = {
    "none": None,
    "stuck-10": ("stuck", -0.10),
    "stuck0": ("stuck", 0.0),
    "stuck+10": ("stuck", 0.10),
    "sensitive2": ("sensitive", 2.0),
    "sensitive6": ("sensitive", 6.0),
}


# ─── The replayed shop ───────────────────────────────────────────────────────

class World:
    """One replayed shop: shared stock, three channels, one pricer."""

    def __init__(self, inf, gen: OrderGenerator, skus: List[str], *, seed: int = 42):
        self.inf, self.gen, self.skus = inf, gen, skus
        sim = SYSTEM_CONFIG["sim"]
        self.channels = channel_names()
        self.weights = world_weights()
        self.abandon = float(sim["abandon_rate"])
        self.rng = random.Random(seed)
        cover0 = 10.0                                   # the demo's initial_cover_days
        self.target = {s: gen.initial_stock(s, sim["replenish_cover_days"]) for s in skus}
        self.reorder = {s: max(1, int(self.target[s] * sim["reorder_point_frac"]))
                        for s in skus}
        self.stock = {s: gen.initial_stock(s, cover0) for s in skus}

    def day(self, tick: int, date: pd.Timestamp, fault: Optional[Tuple[str, float]]
            ) -> Dict[str, Dict]:
        """Trade one day. Returns, per channel, the revenue reading's parts and
        the per-SKU outcomes the critic and markup signals read."""
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
            if fault is not None and fault[0] == "stuck":
                price = q["base_price"] * (1.0 + fault[1])
            for ch in self.channels:
                o = {"sku": s, "price": float(price), "base_price": float(q["base_price"]),
                     "requested": 0, "sold": 0, "refused": 0}
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
        if fault is None or fault[0] != "sensitive":
            return self.gen.orders(sku, date, ch, price, tick=tick)
        # k x elasticity: scale the channel weight so expected units match
        # demand * ratio**(k e). Same content-seeded arrival process.
        base = row["base_price"]
        if base <= 0:
            return self.gen.orders(sku, date, ch, price, tick=tick)
        ratio = max(price, 1e-6) / base
        extra = ratio ** ((fault[1] - 1.0) * row["elasticity"])
        return self.gen.orders(sku, date, ch, price, tick=tick,
                               weight=self.weights[ch] * extra)


def world_weights() -> Dict[str, float]:
    return {c["name"]: float(c["weight"]) for c in SYSTEM_CONFIG["channels"]}


# ─── The markup baseline ─────────────────────────────────────────────────────

def markup_coefficient(inf) -> float:
    """M with training reward ~= M x base_price x sold / 1000, fitted over the
    training window at the pricer's own prices. It is built the same way
    LocalInference.critic_line builds its fit, so both baselines see the same
    rows."""
    import torch  # noqa: PLC0415
    env = inf._shared_env()
    pricer = inf.pricer()
    base_dates = pd.to_datetime(inf.data["rl_base"]["date"]).unique()
    idx = np.flatnonzero(pd.to_datetime(env.df["date"]).isin(base_dates).to_numpy())
    obs = np.stack([env._get_obs(int(i)) for i in idx]).astype(np.float32)
    with torch.no_grad():
        t = torch.as_tensor(obs, device=pricer.policy.device)
        acts = pricer.policy._predict(t, deterministic=True).cpu().numpy().ravel()
    base = env._base_price[idx].astype(float)
    price = base * (1.0 + np.asarray(env.price_tiers, dtype=float)[acts])
    demand = np.maximum(env._realized_demand[idx].astype(float), 0.0)
    asked = np.maximum(0.0, np.round(demand * (price / np.maximum(base, 1e-6))
                                     ** env._elasticity[idx].astype(float)))
    stock = env._inventory_raw[idx].astype(float) if env.inventory_constrained else asked
    sold = np.minimum(asked, stock)
    r = inf._reward(env, idx, price, sold, asked - sold)
    denom = float(np.sum(base * sold) / 1000.0)
    return float(np.sum(r) / denom) if denom > 0 else 0.0


def markup_outcome(inf, m_coef: float, outcomes, share: float) -> Optional[Tuple[float, float]]:
    """(actual, expected) for the markup signal. Actual is the same training
    reward pricing_outcome sums, so only the expectation differs."""
    if share <= 0:
        return None
    env = inf._shared_env()
    rows, price, sold, lost, base = [], [], [], [], []
    for o in outcomes:
        i = inf._row_index.get((o["sku"], pd.Timestamp(o["date"])))
        if i is None:
            continue
        rows.append(i)
        price.append(o["price"])
        sold.append(o["sold"] / share)
        lost.append(o["refused"] / share)
        base.append(o["base_price"])
    if not rows:
        return None
    r = inf._reward(env, np.asarray(rows), np.asarray(price), np.asarray(sold), np.asarray(lost))
    expected = m_coef * float(np.sum(np.asarray(base) * np.asarray(sold)) / 1000.0)
    return float(np.sum(r)), expected


# ─── Feeding the monitors ────────────────────────────────────────────────────

def make_monitors(inf):
    from edge_system.edge.monitor_local import LocalDriftMonitor
    return {(ch, sig): LocalDriftMonitor(ch, inf.calibration)
            for ch in channel_names() for sig in SIGNALS}


def _update(m, tick, d, actual, expected) -> bool:
    idx = actual / expected if (actual is not None and expected and expected > 0) else None
    return bool(m.update(tick=tick, sim_date=d, profit_index=idx,
                         revenue=actual if idx is not None else None,
                         reference=expected if idx is not None else None))


def feed(inf, m_coef, monitors, world_out, tick: int, d: str) -> Dict[tuple, bool]:
    """The node's /check, every signal. Returns which (channel, signal) fired."""
    fired = {}
    shares = world_weights()
    for ch, got in world_out.items():
        for o in got["pricing"]:
            o["date"] = d
        fired[(ch, "revenue")] = _update(monitors[(ch, "revenue")], tick, d,
                                         got["revenue"], got["reference"])
        res = inf.pricing_outcome(sim_date=d, outcomes=got["pricing"], share=shares[ch])
        fired[(ch, "critic")] = _update(monitors[(ch, "critic")], tick, d,
                                        *(res if res else (None, None)))
        res = markup_outcome(inf, m_coef, got["pricing"], shares[ch])
        fired[(ch, "markup")] = _update(monitors[(ch, "markup")], tick, d,
                                        *(res if res else (None, None)))
    for key, f in fired.items():
        if f:
            monitors[key].recalibrate_rl()     # what a retrain does to the line
    return fired


def run_trial(inf, m_coef, gen, skus, onset, fault_name, all_dates) -> List[Dict]:
    fault = FAULTS[fault_name]
    start = onset - pd.Timedelta(days=WARMUP_LEAD)
    end = onset + pd.Timedelta(days=HORIZON - 1)
    days = [d for d in all_dates if start <= d <= end]
    world, mons = World(inf, gen, skus), make_monitors(inf)
    first: Dict[tuple, Optional[int]] = {k: None for k in mons}
    pre: Dict[tuple, int] = {k: 0 for k in mons}
    for d in days:
        tick = (d - pd.Timestamp(WINDOW[0])).days
        active = fault if d >= onset else None
        fired = feed(inf, m_coef, mons, world.day(tick, d, active), tick, d.strftime("%Y-%m-%d"))
        for k, f in fired.items():
            if not f:
                continue
            if d < onset:
                pre[k] += 1
            elif first[k] is None:
                first[k] = (d - onset).days
    return [{"fault": fault_name, "onset": onset.strftime("%Y-%m-%d"), "channel": ch,
             "signal": sig, "caught": delay is not None, "delay_days": delay,
             "fires_before_onset": pre[(ch, sig)]}
            for (ch, sig), delay in first.items()]


def run_false_alarms(inf, m_coef, gen, skus, all_dates):
    days = [d for d in all_dates if pd.Timestamp(WINDOW[0]) <= d <= pd.Timestamp(WINDOW[1])]
    world, mons = World(inf, gen, skus), make_monitors(inf)
    rows = []
    for d in days:
        tick = (d - pd.Timestamp(WINDOW[0])).days
        ds = d.strftime("%Y-%m-%d")
        for (ch, sig), f in feed(inf, m_coef, mons, world.day(tick, d, None), tick, ds).items():
            if f:
                rows.append({"date": ds, "channel": ch, "signal": sig})
    return rows, len(days) / 365.25


# ─── Statistics ──────────────────────────────────────────────────────────────

def wilson(k: int, n: int, z: float = 1.96) -> Tuple[float, float]:
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    den = 1 + z * z / n
    mid = (p + z * z / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return (max(0.0, mid - half), min(1.0, mid + half))


def mcnemar_exact(a: pd.Series, b: pd.Series) -> Dict:
    """Exact McNemar on paired booleans: b10 = a caught and b missed."""
    b10 = int((a & ~b).sum())
    b01 = int((~a & b).sum())
    n = b10 + b01
    p = 1.0 if n == 0 else float(min(1.0, 2 * stats.binom.cdf(min(b10, b01), n, 0.5)))
    return {"a_only": b10, "b_only": b01, "p": round(p, 4)}


def poisson_ci(k: int, t: float) -> Tuple[float, float]:
    lo = 0.0 if k == 0 else stats.chi2.ppf(0.025, 2 * k) / 2
    hi = stats.chi2.ppf(0.975, 2 * (k + 1)) / 2
    return (lo / t, hi / t)


# ─── Main ────────────────────────────────────────────────────────────────────

def main() -> None:
    global OUT_DIR
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--faults", default=",".join(FAULTS),
                    help=f"comma-separated subset of {list(FAULTS)}")
    ap.add_argument("--max-trials", type=int, default=None,
                    help="cap the number of onsets (quick runs)")
    ap.add_argument("--skip-false-alarms", action="store_true")
    ap.add_argument("--out", default=None, help="output dir (default outputs/system/critic_compare)")
    args = ap.parse_args()
    if args.out:
        OUT_DIR = Path(args.out)
    faults = [f.strip() for f in args.faults.split(",") if f.strip()]
    unknown = [f for f in faults if f not in FAULTS]
    if unknown:
        raise SystemExit(f"unknown fault(s) {unknown}; have {list(FAULTS)}")

    # The demo's detector settings. Mutate in place: helpers read at call time.
    SYSTEM_CONFIG["models"].update({"rl_detector": "rolling", "rl_signal": "critic"})
    sim = SYSTEM_CONFIG["sim"]
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    from edge_system.edge.inference import LocalInference
    t0 = time.perf_counter()
    inf = LocalInference("replay")
    inf.load()
    a, b = inf.critic_line()
    m_coef = markup_coefficient(inf)
    print(f"[critic_compare] models from {inf.ckpt_dir} loaded in {time.perf_counter()-t0:.0f}s;"
          f" critic line r = {a:.4f} + {b:.4f} V; markup M = {m_coef:.4f}")

    gen = OrderGenerator.from_csv(Path(SYSTEM_CONFIG["paths"]["data_dir"]) / "rl_environment.csv",
                                  seed=sim["seed"], mean_basket=sim["mean_basket"],
                                  channel_weights=world_weights())
    all_dates = sorted(set(gen.dates))
    demo_end = pd.Timestamp("2013-01-01") + pd.Timedelta(days=399)   # demo: 400 ticks
    skus = gen.skus(limit=10, order="top", max_gap_days=21, per_dept=5,
                    window=(pd.Timestamp("2013-01-01"), demo_end))
    print(f"[critic_compare] SKUs: {skus}")

    # Non-overlapping blocks: [onset - WARMUP_LEAD, onset + HORIZON).
    block = WARMUP_LEAD + HORIZON
    lo, hi = pd.Timestamp(WINDOW[0]), pd.Timestamp(WINDOW[1])
    onsets = []
    start = lo
    while start + pd.Timedelta(days=block - 1) <= hi:
        onsets.append(start + pd.Timedelta(days=WARMUP_LEAD))
        start += pd.Timedelta(days=block)
    if args.max_trials:
        onsets = onsets[:args.max_trials]

    trials: List[Dict] = []
    for fault in faults:
        t = time.perf_counter()
        for onset in onsets:
            trials += run_trial(inf, m_coef, gen, skus, onset, fault, all_dates)
        print(f"[critic_compare] {fault}: {len(onsets)} trials ({time.perf_counter()-t:.0f}s)",
              flush=True)
    tr = pd.DataFrame(trials)
    tr.to_csv(OUT_DIR / "trials.csv", index=False)

    summary: Dict = {
        "version": 2, "checkpoints": inf.ckpt_dir, "critic_line": [a, b],
        "markup_coefficient": m_coef, "skus": skus, "window": WINDOW,
        "onsets": [o.strftime("%Y-%m-%d") for o in onsets],
        "warmup_lead_days": WARMUP_LEAD, "horizon_days": HORIZON,
        "detector": {k: SYSTEM_CONFIG["models"][k] for k in
                     ("rl_window_days", "rl_warmup_days", "rl_k_sigma", "relearn_min_days")},
        "caught": {}, "mcnemar_critic_vs": {},
    }
    for (fault, ch, sig), g in tr.groupby(["fault", "channel", "signal"]):
        k, n = int(g["caught"].sum()), len(g)
        dl = g["delay_days"].dropna()
        lo_ci, hi_ci = wilson(k, n)
        summary["caught"][f"{fault}/{ch}/{sig}"] = {
            "k": k, "n": n, "ci95": [round(lo_ci, 3), round(hi_ci, 3)],
            "median_delay_days": float(dl.median()) if len(dl) else None,
            "fires_before_onset": int(g["fires_before_onset"].sum())}
    for (fault, ch), g in tr.groupby(["fault", "channel"]):
        w = g.pivot(index="onset", columns="signal", values="caught").astype(bool)
        summary["mcnemar_critic_vs"][f"{fault}/{ch}"] = {
            other: mcnemar_exact(w["critic"], w[other]) for other in SIGNALS if other != "critic"}

    if not args.skip_false_alarms:
        t = time.perf_counter()
        fa, years = run_false_alarms(inf, m_coef, gen, skus, all_dates)
        pd.DataFrame(fa, columns=["date", "channel", "signal"]).to_csv(
            OUT_DIR / "false_alarms.csv", index=False)
        per_year = {}
        for ch in channel_names():
            for sig in SIGNALS:
                n = sum(1 for r in fa if r["channel"] == ch and r["signal"] == sig)
                lo_ci, hi_ci = poisson_ci(n, years)
                per_year[f"{ch}/{sig}"] = {"events": n, "per_year": round(n / years, 2),
                                           "ci95": [round(lo_ci, 2), round(hi_ci, 2)]}
        summary["false_alarms"] = {"years": round(years, 2), "by_channel_signal": per_year}
        print(f"[critic_compare] false-alarm run done ({time.perf_counter()-t:.0f}s)")

    (OUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2))

    def cell(fault, ch, sig):
        c = summary["caught"][f"{fault}/{ch}/{sig}"]
        return f"{c['k']:>2}/{c['n']} [{c['ci95'][0]:.2f},{c['ci95'][1]:.2f}]"

    print(f"\n== caught within {HORIZON} days, Wilson 95% CI "
          f"({len(onsets)} non-overlapping trials) ==")
    print(f"  {'fault':11s} {'channel':12s} " + "  ".join(f"{s:>18s}" for s in SIGNALS)
          + "   McNemar p (critic vs revenue / markup)")
    for fault in faults:
        for ch in channel_names():
            mc = summary["mcnemar_critic_vs"][f"{fault}/{ch}"]
            print(f"  {fault:11s} {ch:12s} " + "  ".join(f"{cell(fault, ch, s):>18s}" for s in SIGNALS)
                  + f"   {mc['revenue']['p']:.3f} / {mc['markup']['p']:.3f}")
    if "false_alarms" in summary:
        print(f"== false alarms per year, normal trading ({summary['false_alarms']['years']} y) ==")
        for ch in channel_names():
            parts = []
            for sig in SIGNALS:
                v = summary["false_alarms"]["by_channel_signal"][f"{ch}/{sig}"]
                parts.append(f"{sig} {v['per_year']} [{v['ci95'][0]},{v['ci95'][1]}]")
            print(f"  {ch:12s} " + "   ".join(parts))
    print(f"\nwritten to {OUT_DIR}")


if __name__ == "__main__":
    main()
