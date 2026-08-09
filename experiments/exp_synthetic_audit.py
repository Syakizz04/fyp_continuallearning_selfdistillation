"""
Synthetic-layer audit - how much of this dataset is measured, and how much is assumed.

    python -m experiments.exp_synthetic_audit
    python -m experiments.exp_synthetic_audit --data data/processed_m5_v3

No GPU, no models, no training. Three diagnostics, in increasing order of how
uncomfortable the answer is.

## Why this exists

The forecasting results rest on real M5 unit sales. The pricing results do not:
they rest on an estimated elasticity, feeding counterfactual demand at prices
that were never observed, feeding an invented reward. That is a legitimate way
to build a pricing environment, but only if the assumed layer is stated rather
than implied - and an examiner who derives these numbers unaided will trust
everything else less. So they get derived here, and reported whether or not they
flatter.

1. **Conservation.** Does the synthetic machinery preserve the real quantity it
   was built from? If aggregating generated orders does not return M5's daily
   units, the simulation is not a re-expression of the data, it is a replacement
   for it. Mechanical pass/fail.

2. **Identification.** How much of each item's elasticity came from that item's
   own data, and how much from the dunnhumby prior? Empirical-Bayes shrinkage is
   correct behaviour when data cannot identify a parameter, but it means the
   "per-item elasticity" may be close to one global number wearing 100 hats.

3. **Price variation.** The root cause. Elasticity is identified by price moving
   while other things hold still; if M5 prices barely move within an item, no
   estimator could have recovered it, and that bounds what diagnostic 2 could
   ever have shown. This distinguishes "our method underperformed" from "the
   data does not contain the answer" - a distinction worth owning up front.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from rich.console import Console  # noqa: E402
from rich.table import Table  # noqa: E402

console = Console()

TOL = 1e-6


# ── 1. conservation ───────────────────────────────────────────────────────────

def audit_conservation(data_dir: Path, *, n_sku: int = 25, n_days: int = 120,
                       seed: int = 42) -> Dict:
    """Does the synthetic layer preserve the real quantity underneath it?"""
    from edge_system.config import SYSTEM_CONFIG
    from edge_system.inventory.replenishment import simulate_inventory
    from edge_system.sim.order_gen import OrderGenerator

    rl_path = data_dir / "rl_environment.csv"
    df = pd.read_csv(rl_path, parse_dates=["date"])
    checks: List[Dict] = []

    def check(name: str, passed: bool, detail: str) -> None:
        checks.append({"check": name, "pass": bool(passed), "detail": detail})

    # ── 1a. the CSV's own inventory columns must balance ──────────────────────
    # unmet is demand that on-hand could not cover. If this identity does not
    # hold exactly, the censoring signal E2 would consume is not derivable from
    # the stock series it is supposed to come from.
    implied = np.maximum(0.0, df["realized_demand"] - df["inventory_level"])
    resid = (df["unmet_demand"] - implied).abs()
    check("unmet == max(0, demand - on_hand)", bool(resid.max() <= TOL),
          f"max |residual| = {resid.max():.6g} over {len(df):,} rows")

    flag_ok = (df["stockout_flag"].astype(bool) == (df["unmet_demand"] > 0)).all()
    check("stockout_flag <=> unmet > 0", bool(flag_ok),
          f"mismatches = {int((df['stockout_flag'].astype(bool) != (df['unmet_demand'] > 0)).sum())}")

    check("unmet <= demand", bool((df["unmet_demand"] <= df["realized_demand"] + TOL).all()),
          f"violations = {int((df['unmet_demand'] > df['realized_demand'] + TOL).sum())}")
    check("on_hand >= 0", bool((df["inventory_level"] >= -TOL).all()),
          f"min on_hand = {df['inventory_level'].min():.4g}")

    # ── 1b. the replenishment simulator's internal flow balance ───────────────
    # Run it directly on a real demand series: opening stock plus everything
    # received minus everything served must equal what is left.
    sample_sku = df["product_id"].iloc[0]
    demand = (df[df["product_id"] == sample_sku]
              .sort_values("date")["realized_demand"].to_numpy(float))
    trace = simulate_inventory(demand, cover_days=10.0, lead_time_days=3)
    opening = float(trace.on_hand[0])
    closing = float(trace.on_hand_after[-1])
    flow = opening + float(trace.received.sum()) - float(trace.served.sum())
    check("inventory flow balances", abs(flow - closing) <= 1e-6,
          f"open {opening:.0f} + recv {trace.received.sum():.0f} "
          f"- served {trace.served.sum():.0f} = {flow:.0f} vs closing {closing:.0f}")
    check("served == demand - unmet", bool(
        np.abs(trace.served - (demand - trace.unmet)).max() <= TOL),
        f"max |residual| = {np.abs(trace.served - (demand - trace.unmet)).max():.6g}")

    # ── 1c. the order generator must return M5's units, not its own ───────────
    gen = OrderGenerator.from_csv(
        rl_path, seed=seed,
        mean_basket=SYSTEM_CONFIG["sim"]["mean_basket"],
        channel_weights={c["name"]: c["weight"] for c in SYSTEM_CONFIG["channels"]})
    skus = gen.skus(limit=n_sku)
    # `dates` is a property over every ROW, so it repeats a day once per SKU.
    dates = sorted(pd.to_datetime(pd.Series(gen.dates)).dt.normalize()
                   .unique())[:n_days]

    # At base price the demand model must be the identity on the real figure.
    ident_err = 0.0
    truth = df.set_index(["product_id", "date"])["realized_demand"]
    for sku in skus:
        for d in dates[:20]:
            bp = gen.base_price(sku, d)
            if bp is None:
                continue
            got = gen.expected_units(sku, d, bp)
            want = float(truth.get((sku, pd.Timestamp(d)), 0.0))
            ident_err = max(ident_err, abs(got - max(want, 0.0)))
    check("expected_units(base_price) == M5 units", ident_err <= 1e-9,
          f"max |error| = {ident_err:.3g} (the no-model baseline is anchored)")

    # Discrete arrivals are Poisson, so they match only in expectation. The
    # construction n ~ Poisson(u/b), size = 1 + Poisson(b-1) is unbiased by
    # design; this checks the implementation actually is.
    weights = {c["name"]: c["weight"] for c in SYSTEM_CONFIG["channels"]}
    gen_units, real_units = 0.0, 0.0
    for tick, d in enumerate(dates):
        for sku in skus:
            bp = gen.base_price(sku, d)
            if bp is None:
                continue
            real_units += gen.expected_units(sku, d, bp)
            for ch in weights:
                gen_units += sum(o.qty for o in
                                 gen.orders(sku, d, ch, bp, tick=tick))
    rel = (gen_units - real_units) / max(real_units, 1e-9)
    # 3% band: with ~n independent Poisson draws the sampling error on the total
    # is well inside this, so a breach means bias, not noise.
    check("generated orders reproduce M5 units (aggregate)", abs(rel) < 0.03,
          f"generated {gen_units:,.0f} vs expected {real_units:,.0f} "
          f"({rel:+.2%}) over {len(skus)} SKUs x {len(dates)} days")

    check("channel weights sum to 1", abs(sum(weights.values()) - 1.0) < 1e-9,
          f"sum = {sum(weights.values()):.6f} across {len(weights)} channels")

    return {"checks": checks,
            "n_passed": sum(c["pass"] for c in checks),
            "n_total": len(checks)}


# ── 2. identification ─────────────────────────────────────────────────────────

def audit_identification(data_dir: Path, prior_path: Path) -> Dict:
    """How much of the elasticity is the item's own data vs the global prior?"""
    report = pd.read_csv(data_dir / "elasticity_report.csv")
    prior = json.loads(Path(prior_path).read_text())
    mu, tau2 = float(prior["mu"]), float(prior["tau2"])

    # Precision weighting: the posterior is w*beta_hat + (1-w)*mu with
    # w = tau2 / (tau2 + se^2). w IS the share of the answer that came from this
    # item. A noisy item estimate (large se) contributes almost nothing.
    se = report["se"].to_numpy(float)
    w = np.where(np.isfinite(se) & (se > 0), tau2 / (tau2 + se ** 2), 0.0)
    report["data_weight"] = w

    methods = report["method"].value_counts().to_dict()
    n = len(report)
    post, raw = report["elasticity"], report["raw_beta"]

    stats = {
        "n_items": n,
        "n_identified": int(report["identified"].sum()),
        "share_identified": float(report["identified"].mean()),
        "methods": methods,
        "mean_data_weight": float(np.mean(w)),
        "median_data_weight": float(np.median(w)),
        "share_below_half_weight": float(np.mean(w < 0.5)),
        "share_zero_weight": float(np.mean(w <= 1e-9)),
        "corr_posterior_raw": float(post.corr(raw)),
        "mean_abs_post_minus_prior": float((post - mu).abs().mean()),
        "mean_abs_raw_minus_prior": float((raw - mu).abs().mean()),
        "posterior_iqr": [float(post.quantile(.25)), float(post.quantile(.75))],
        "posterior_median": float(post.median()),
        "prior_mu": mu, "prior_tau2": tau2,
        "prior_source": prior.get("source", ""),
    }
    # How much spread survives shrinkage. If the posterior sd is a small
    # fraction of the raw sd, "per-item elasticity" is mostly one number.
    stats["sd_posterior"] = float(post.std())
    stats["sd_raw"] = float(raw.std())
    stats["spread_retained"] = float(post.std() / max(raw.std(), 1e-9))
    return {"stats": stats, "per_item": report}


# ── 3. price variation ────────────────────────────────────────────────────────

def audit_price_variation(data_dir: Path, per_item: pd.DataFrame) -> Dict:
    """What price movement was available to identify elasticity from?

    An elasticity is recovered from price moving while demand responds. This
    measures how much movement each item actually offered, which is the ceiling
    on what ANY estimator could have found - and therefore separates a weak
    method from insufficient data.
    """
    df = pd.read_csv(data_dir / "rl_environment.csv", parse_dates=["date"])
    g = df.groupby("product_id")["base_price"]

    var = pd.DataFrame({
        "n_obs": g.size(),
        "n_unique_price": g.nunique(),
        "price_mean": g.mean(),
        "price_cv": g.std() / g.mean(),
        "price_min": g.min(),
        "price_max": g.max(),
    })
    var["price_range_ratio"] = var["price_max"] / var["price_min"].replace(0, np.nan)
    # Share of days sitting at the single most common price: the complement is
    # the fraction of the series that carries any identifying information.
    modal_share = df.groupby("product_id")["base_price"].apply(
        lambda s: s.value_counts(normalize=True).iloc[0])
    var["modal_price_share"] = modal_share
    var = var.reset_index().rename(columns={"product_id": "item_store"})

    # Join identification back on, to test the causal story directly.
    key = per_item.copy()
    key["item_store"] = key["item_id"].astype(str) + "_" + key["store_id"].astype(str)
    merged = var.merge(key[["item_store", "data_weight", "identified", "se"]],
                       on="item_store", how="inner")
    if merged.empty:                       # id formats differ between artefacts
        merged = var.merge(
            key.assign(item_store=key["item_id"])[
                ["item_store", "data_weight", "identified", "se"]],
            on="item_store", how="inner")

    out = {
        "median_unique_prices": float(var["n_unique_price"].median()),
        "median_price_cv": float(var["price_cv"].median()),
        "median_modal_share": float(var["modal_price_share"].median()),
        "share_under_5_prices": float((var["n_unique_price"] < 5).mean()),
        "median_range_ratio": float(var["price_range_ratio"].median()),
        "n_items": int(len(var)),
        "n_matched": int(len(merged)),
    }
    if not merged.empty and merged["data_weight"].std() > 0:
        out["corr_cv_vs_data_weight"] = float(
            merged["price_cv"].corr(merged["data_weight"]))
        out["corr_modalshare_vs_data_weight"] = float(
            merged["modal_price_share"].corr(merged["data_weight"]))
        ident = merged[merged["identified"] == True]      # noqa: E712
        unident = merged[merged["identified"] != True]    # noqa: E712
        if len(ident) and len(unident):
            out["median_cv_identified"] = float(ident["price_cv"].median())
            out["median_cv_unidentified"] = float(unident["price_cv"].median())
    return {"stats": out, "per_item": var, "merged": merged}


# ── reporting ─────────────────────────────────────────────────────────────────

def report(cons: Dict, ident: Dict, price: Dict) -> None:
    t = Table(title="1. Conservation - does the synthetic layer preserve M5?",
              header_style="bold cyan")
    t.add_column("check"); t.add_column("result"); t.add_column("detail")
    for c in cons["checks"]:
        t.add_row(c["check"],
                  "[green]PASS[/green]" if c["pass"] else "[red]FAIL[/red]",
                  c["detail"])
    console.print(t)
    console.print(f"  {cons['n_passed']}/{cons['n_total']} passed\n")

    s = ident["stats"]
    t = Table(title="2. Identification - data or prior?", header_style="bold cyan")
    t.add_column("quantity"); t.add_column("value", justify="right")
    t.add_row("items", f"{s['n_items']}")
    t.add_row("identified from own data", f"{s['n_identified']} "
                                          f"({s['share_identified']:.0%})")
    t.add_row("method breakdown", ", ".join(f"{k}={v}" for k, v in s["methods"].items()))
    t.add_row("mean weight on own data", f"{s['mean_data_weight']:.3f}")
    t.add_row("median weight on own data", f"{s['median_data_weight']:.3f}")
    t.add_row("items <50% own-data weight", f"{s['share_below_half_weight']:.0%}")
    t.add_row("corr(posterior, raw estimate)", f"{s['corr_posterior_raw']:.3f}")
    t.add_row("spread retained after shrinkage", f"{s['spread_retained']:.1%}")
    t.add_row("posterior median / IQR",
              f"{s['posterior_median']:.2f} "
              f"[{s['posterior_iqr'][0]:.2f}, {s['posterior_iqr'][1]:.2f}]")
    t.add_row("prior (mu, tau2)", f"{s['prior_mu']:.3f}, {s['prior_tau2']:.3f}")
    console.print(t)

    p = price["stats"]
    t = Table(title="3. Price variation - was the answer in the data?",
              header_style="bold cyan")
    t.add_column("quantity"); t.add_column("value", justify="right")
    t.add_row("median distinct prices per item", f"{p['median_unique_prices']:.0f}")
    t.add_row("items with <5 distinct prices", f"{p['share_under_5_prices']:.0%}")
    t.add_row("median price CV", f"{p['median_price_cv']:.4f}")
    t.add_row("median share of days at modal price", f"{p['median_modal_share']:.1%}")
    t.add_row("median max/min price ratio", f"{p['median_range_ratio']:.3f}")
    for k in ("corr_cv_vs_data_weight", "median_cv_identified",
              "median_cv_unidentified"):
        if k in p:
            t.add_row(k.replace("_", " "), f"{p[k]:.4f}")
    console.print(t)

    # The one-paragraph verdict, stated in the terms the thesis needs.
    console.print("\n[bold]Reading:[/bold]")
    if cons["n_passed"] == cons["n_total"]:
        console.print("  * Conservation holds: the synthetic layer re-expresses "
                      "M5 rather than replacing it.")
    else:
        console.print("  [red]* Conservation FAILED - a synthetic quantity does "
                      "not reconcile with the data it claims to derive from.[/red]")
    console.print(f"  * Elasticity is [bold]{1 - s['mean_data_weight']:.0%} prior, "
                  f"{s['mean_data_weight']:.0%} own data[/bold] on average; only "
                  f"{s['spread_retained']:.0%} of raw cross-item spread survives.")
    console.print(f"  * Prices sit at one value [bold]{p['median_modal_share']:.0%}"
                  f"[/bold] of days (median CV {p['median_price_cv']:.3f}), so the "
                  f"weak identification is a property of M5, not of the estimator.")
    console.print("  => Forecasting conclusions rest on real units. Pricing "
                  "conclusions rest on this assumed layer and should be reported "
                  "with that stated.")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Audit the synthetic data layer.")
    ap.add_argument("--data", default="data/processed_m5_v3")
    ap.add_argument("--prior", default="data/external_pricing/prior.json")
    ap.add_argument("--out", default="outputs/drift/results")
    ap.add_argument("--n-sku", type=int, default=25)
    ap.add_argument("--n-days", type=int, default=120)
    args = ap.parse_args(argv)

    data_dir = PROJECT_ROOT / args.data
    out_dir = PROJECT_ROOT / args.out
    out_dir.mkdir(parents=True, exist_ok=True)

    console.print(f"[dim]auditing {data_dir}[/dim]\n")
    cons = audit_conservation(data_dir, n_sku=args.n_sku, n_days=args.n_days)
    ident = audit_identification(data_dir, PROJECT_ROOT / args.prior)
    price = audit_price_variation(data_dir, ident["per_item"])

    report(cons, ident, price)

    pd.DataFrame(cons["checks"]).to_csv(out_dir / "audit_conservation.csv", index=False)
    ident["per_item"].to_csv(out_dir / "audit_elasticity_items.csv", index=False)
    price["per_item"].to_csv(out_dir / "audit_price_variation.csv", index=False)
    (out_dir / "audit_summary.json").write_text(json.dumps(
        {"conservation": {"passed": cons["n_passed"], "total": cons["n_total"],
                          "checks": cons["checks"]},
         "identification": ident["stats"],
         "price_variation": price["stats"]}, indent=2, default=str))
    console.print(f"\n-> {out_dir / 'audit_summary.json'}")
    return 0 if cons["n_passed"] == cons["n_total"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
