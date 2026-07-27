"""
Hierarchical (empirical-Bayes) price-elasticity estimation for the M5 RL layer.

Replaces the flat `estimate_elasticity()` in build_m5.py, which clipped every
per-item OLS estimate to [-3.0, -0.2] and substituted a hard-coded -1.5 whenever
a series had too few rows or too little price variation. On the 100-item CA_1
scope that left ~57% of elasticities not identified from data (25% at the
fallback constant, 32% pinned at a clip bound).

The fix is to stop treating "no signal" as "use a constant" and treat it as
"use the prior", with the strength of the shrinkage set by how much the item's
own data actually says. Three levels:

    level 2  item   b_i ~ N(mu_dept,  se_i^2)     OLS + its standard error
    level 1  dept   mu_dept ~ N(mu_0, tau_0^2)    pooled across items in a dept
    level 0  prior  mu_0, tau_0^2                 external real-retail dataset

The posterior mean is the precision-weighted combination at each level, so an
item with strong price variation keeps its own estimate, and an item with none
degrades smoothly to its department mean and then to the global prior — never
snapping to an arbitrary constant.

**The external dataset supplies the prior only — never a row-level join.** It
shares no item or store key with M5, so joining would splice unrelated demand
and price series together. Supplying (mu_0, tau_0^2) is standard empirical
Bayes and is the defensible use of that data.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Sequence

import numpy as np
import pandas as pd

try:
    from rich.console import Console
    _console = Console()
    def log(msg: str) -> None: _console.print(msg)
except Exception:  # rich optional
    def log(msg: str) -> None: print(msg)


# ─── Knobs ───────────────────────────────────────────────────────────────────

MIN_ELAST_OBS    = 40      # rows below which the item OLS is not attempted at all
MIN_LOGPRICE_STD = 0.02    # log-price sd below which elasticity is unidentified
MIN_FIRST_STAGE_F = 10.0   # Staiger-Stock weak-instrument threshold
SAFETY_CLIP      = (-5.0, -0.05)
# Wide *safety* bound applied only after shrinkage, to keep the RL demand model
# from exploding. Deliberately much wider than the old (-3.0, -0.2): the prior,
# not the clip, is what now does the regularising. The share of items touching
# this bound is reported — it should be near zero, and if it is not, that is a
# finding about the data rather than something to silently absorb.

MAX_SE = 10.0              # se above this is treated as "no information"

# Placeholder prior, used only when no external dataset is available. mu_0
# matches the old hard-coded fallback so behaviour is conservative, but the wide
# tau_0 means any real item-level signal now dominates it instead of being
# discarded. REPLACE THIS by fitting a real external dataset — see
# `fit_external_prior()` — before quoting elasticity numbers in the report.
LITERATURE_PRIOR = (-1.5, 1.0**2)


# ─── Level 2: per-item OLS with a standard error ─────────────────────────────

@dataclass(frozen=True)
class ItemEstimate:
    """A single item's own evidence about its elasticity."""
    beta: float          # log-log coefficient (nan when not estimable)
    se: float            # its standard error (inf when not estimable)
    n_obs: int
    logprice_std: float
    method: str = "ols"          # "ols" | "iv" | "none"
    first_stage_f: float = np.nan

    @property
    def identified(self) -> bool:
        return np.isfinite(self.beta) and np.isfinite(self.se) and self.se < MAX_SE

    @property
    def precision(self) -> float:
        return 1.0 / self.se**2 if self.identified else 0.0


def _ols(X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, float]:
    """Least squares + residual variance. Raises on a degenerate system."""
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    dof = max(len(y) - X.shape[1], 1)
    return beta, float(resid @ resid) / dof


def fit_item_elasticity(
    group: pd.DataFrame,
    instrument_col: Optional[str] = None,
    control_cols: Sequence[str] = (),
) -> ItemEstimate:
    """
    Deseasonalized log-log demand curve for one (item, store) series:

        log(demand + 1) ~ b * log(price) + month fixed effects

    Returns the coefficient *and its standard error*; the SE is what lets the
    caller decide how much to trust this item during shrinkage.

    **Endogeneity.** Estimating this by OLS is biased: retail promotions bundle
    price cuts with display and feature advertising, and retailers raise prices
    into high-demand periods, so log(price) correlates with the demand shock. On
    the M5 CA_1 scope, naive OLS returns a *positive* elasticity for 31% of
    items — the wrong sign, i.e. "raise the price and demand rises".

    When `instrument_col` is given, the price is instead instrumented by two-
    stage least squares using a **Hausman instrument**: the same item's price in
    *other* stores in the same week. Prices co-move across stores through shared
    supply-side cost shocks (the variation we want), while another store's local
    demand shock is independent of this store's (the exclusion restriction).
    Estimates whose first-stage F falls below the Staiger-Stock threshold of 10
    are reported as weakly identified, so shrinkage discounts them rather than
    the caller silently trusting a weak-instrument estimate.

    `control_cols` adds exogenous regressors to both stages. On M5 there is
    nothing useful to put here - it records no promotion activity, which is the
    whole problem. On dunnhumby there is: `display` and `mailer` observe the
    confounder directly, so controlling for them yields an elasticity that is
    not contaminated by promotional lift. That is what qualifies dunnhumby to
    supply M5's prior.
    """
    g = group[(group["demand"] > 0) & (group["sell_price"] > 0)]
    if instrument_col:
        g = g[g[instrument_col].notna() & (g[instrument_col] > 0)]
    n = len(g)
    if n < MIN_ELAST_OBS:
        return ItemEstimate(np.nan, np.inf, n, 0.0, "none")

    logp = np.log(g["sell_price"].to_numpy(dtype=float))
    lp_std = float(logp.std())
    if lp_std < MIN_LOGPRICE_STD:
        # No usable price variation: the item genuinely carries no information
        # about its own elasticity. Report that rather than inventing a number.
        return ItemEstimate(np.nan, np.inf, n, lp_std, "none")

    logd = np.log(g["demand"].to_numpy(dtype=float) + 1.0)
    months = pd.get_dummies(g["month"].astype(int), drop_first=True).to_numpy(dtype=float)

    extra = [months]
    for col in control_cols:
        if col in g.columns:
            v = pd.to_numeric(g[col], errors="coerce").fillna(0.0).to_numpy(dtype=float)
            if v.std() > 0:          # a constant control adds nothing but collinearity
                extra.append(v.reshape(-1, 1))

    controls = np.column_stack([np.ones(n), *extra])   # exogenous regressors
    X = np.column_stack([controls[:, :1], logp, controls[:, 1:]])  # const, logp, rest

    if n - X.shape[1] <= 1:
        return ItemEstimate(np.nan, np.inf, n, lp_std, "none")

    try:
        if instrument_col:
            z = np.log(g[instrument_col].to_numpy(dtype=float))
            # Stage 1: price on the instrument + exogenous controls.
            Z = np.column_stack([controls[:, :1], z, controls[:, 1:]])
            gamma, s2_first = _ols(Z, logp)
            logp_hat = Z @ gamma

            # Partial F for the excluded instrument (Staiger-Stock).
            _, s2_restricted = _ols(controls, logp)
            dof1 = max(n - Z.shape[1], 1)
            rss_r = s2_restricted * max(n - controls.shape[1], 1)
            rss_u = s2_first * dof1
            f_stat = float((rss_r - rss_u) / (rss_u / dof1)) if rss_u > 0 else np.nan

            # Stage 2: demand on fitted price. The structural residual uses the
            # ACTUAL price, not the fitted one — this is what makes the 2SLS
            # standard error correct rather than the naive stage-2 OLS one.
            X_hat = np.column_stack([controls[:, :1], logp_hat, controls[:, 1:]])
            beta, _ = _ols(X_hat, logd)
            resid = logd - X @ beta
            dof2 = max(n - X.shape[1], 1)
            sigma2 = float(resid @ resid) / dof2
            var_b = sigma2 * float(np.linalg.pinv(X_hat.T @ X_hat)[1, 1])
            method = "iv"

            if not np.isfinite(f_stat) or f_stat < MIN_FIRST_STAGE_F:
                # Weak instrument: 2SLS is unreliable here. Keep the estimate but
                # widen its SE so the hierarchical prior does the work instead.
                var_b = max(var_b, MAX_SE**2)
                method = "iv-weak"
        else:
            beta, sigma2 = _ols(X, logd)
            var_b = sigma2 * float(np.linalg.pinv(X.T @ X)[1, 1])
            f_stat, method = np.nan, "ols"
    except (np.linalg.LinAlgError, ValueError):
        return ItemEstimate(np.nan, np.inf, n, lp_std, "none")

    b = float(beta[1])
    if not np.isfinite(b) or not np.isfinite(var_b) or var_b <= 0:
        return ItemEstimate(np.nan, np.inf, n, lp_std, "none")

    if b > 0:
        # Economic sign restriction. A positive price elasticity violates the law
        # of demand, so it is evidence that the estimate is confounded rather
        # than a measurement of the demand curve. On M5 the confounder is
        # unobserved promotion activity: price cuts run alongside display and
        # feature advertising that M5 does not record, so the two effects cannot
        # be separated. Tested and ruled out as fixable — adding snap/event
        # controls (31% -> 32%) and aggregating to the weekly price frequency
        # (31% -> 41%) both fail to remove it, and the Hausman instrument only
        # moves it to 28% because Walmart prices chain-wide, so other stores
        # carry the same national promotion shock (Bresnahan's critique of
        # Hausman instruments).
        #
        # Keep `beta` for the diagnostic, but set the SE to infinity so the item
        # counts as unidentified and shrinks fully to its prior. This is a sign
        # restriction, not a clip: the value comes from the hierarchy, not from
        # a constant.
        return ItemEstimate(b, np.inf, n, lp_std, "wrong-sign", f_stat)

    return ItemEstimate(b, float(np.sqrt(var_b)), n, lp_std, method, f_stat)


def build_price_instrument(
    prices: pd.DataFrame,
    focal_stores: Sequence[str],
    *,
    item_col: str = "item_id",
    store_col: str = "store_id",
    week_col: str = "wm_yr_wk",
    price_col: str = "sell_price",
) -> pd.DataFrame:
    """
    Hausman instrument: for each (item, week), the mean price of that item across
    stores *other than* the focal ones.

    Returns a frame keyed `[item_col, week_col]` with an `instrument_price`
    column, ready to merge onto the long frame. Pass the *unfiltered*
    `sell_prices.csv` — the whole point is to use the stores excluded from the
    modelling scope, which is why this costs no extra data.
    """
    focal = set(focal_stores)
    other = prices[~prices[store_col].isin(focal)]
    if other.empty:
        raise ValueError(
            f"no non-focal stores left to build the instrument from "
            f"(focal={sorted(focal)}); the Hausman instrument needs at least one "
            f"store outside the modelling scope"
        )
    inst = (
        other.groupby([item_col, week_col])[price_col]
             .mean()
             .rename("instrument_price")
             .reset_index()
    )
    log(f"  price instrument: {len(inst):,} (item, week) cells from "
        f"{other[store_col].nunique()} non-focal stores")
    return inst


# ─── Levels 1 and 0: pooling ─────────────────────────────────────────────────

@dataclass(frozen=True)
class Prior:
    """A normal prior over elasticity: mean and between-unit variance."""
    mu: float
    tau2: float

    @property
    def precision(self) -> float:
        return 1.0 / self.tau2 if self.tau2 > 0 else 0.0

    def __repr__(self) -> str:  # keeps the build log readable
        return f"Prior(mu={self.mu:.3f}, tau={np.sqrt(self.tau2):.3f})"


def _pool(estimates: Sequence[ItemEstimate], fallback: Prior) -> Prior:
    """
    Random-effects pool of identified item estimates (DerSimonian-Laird).

    Returns the pooled mean and the *between-item* variance tau^2 — the latter
    is what tells a downstream item how much departments genuinely differ, and
    therefore how hard to shrink toward the pool.
    """
    ident = [e for e in estimates if e.identified]
    if len(ident) < 2:
        return fallback

    b = np.array([e.beta for e in ident], dtype=float)
    w = np.array([e.precision for e in ident], dtype=float)
    if w.sum() <= 0:
        return fallback

    mu_fixed = float((w * b).sum() / w.sum())

    # DerSimonian-Laird moment estimator for between-unit variance.
    q = float((w * (b - mu_fixed) ** 2).sum())
    k = len(ident)
    denom = w.sum() - (w**2).sum() / w.sum()
    tau2 = max(0.0, (q - (k - 1)) / denom) if denom > 0 else 0.0

    if tau2 <= 0:
        # Items agree within their own noise; keep a small floor so shrinkage
        # stays finite rather than collapsing every item onto the pooled mean.
        tau2 = max(float(np.var(b)), 1e-3)

    # Re-pool with the random-effects weights now that tau^2 is known.
    w_re = 1.0 / (1.0 / w + tau2)
    mu = float((w_re * b).sum() / w_re.sum())
    return Prior(mu=mu, tau2=float(tau2))


def _posterior_mean(estimate: ItemEstimate, prior: Prior) -> float:
    """Precision-weighted combination of an item's own evidence and its prior."""
    p_data, p_prior = estimate.precision, prior.precision
    if p_data + p_prior <= 0:
        return prior.mu
    if p_data <= 0:
        return prior.mu           # no signal at all -> the prior *is* the answer
    return float((estimate.beta * p_data + prior.mu * p_prior) / (p_data + p_prior))


# ─── Level 0: the external prior ─────────────────────────────────────────────

def fit_external_prior(
    path: str | Path,
    *,
    item_col: str,
    price_col: str,
    qty_col: str,
    time_col: Optional[str] = None,
    min_obs: int = 20,
    control_cols: Sequence[str] = (),
) -> Prior:
    """
    Fit (mu_0, tau_0^2) from an external real-retail dataset with transaction- or
    scanner-level prices, by running the same log-log regression per product and
    pooling the results.

    This is the "bootstrap from an external pricing dataset" step. Only the two
    prior scalars cross over into the M5 pipeline — no rows, no join.

    See `load_dunnhumby()` for an adapter that puts dunnhumby "The Complete
    Journey" into the expected (item, price, qty) shape.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"external pricing dataset not found: {path}\n"
            "Download it into data/external_pricing/ first, or call "
            "build_prior_chain(external=None) to run with the placeholder prior."
        )

    df = pd.read_csv(path)
    missing = {item_col, price_col, qty_col} - set(df.columns)
    if missing:
        raise KeyError(f"{path.name} is missing expected columns: {sorted(missing)}")

    df = df[(df[price_col] > 0) & (df[qty_col] > 0)].copy()
    df = df.rename(columns={price_col: "sell_price", qty_col: "demand"})

    # The item regression wants a `month` column for the seasonal dummies. Use
    # the external dataset's own time column when it has one; otherwise fall
    # back to a single constant month (no seasonal control available).
    if time_col and time_col in df.columns:
        t = pd.to_datetime(df[time_col], errors="coerce")
        if t.notna().any():
            df["month"] = t.dt.month.fillna(1).astype(int)
        else:  # numeric day-index columns (dunnhumby DAY) -> pseudo-months
            df["month"] = (pd.to_numeric(df[time_col], errors="coerce")
                             .fillna(0).astype(int) // 30 % 12) + 1
    else:
        df["month"] = 1

    estimates = [
        fit_item_elasticity(g, control_cols=control_cols)
        for _, g in df.groupby(item_col, sort=False)
        if len(g) >= min_obs
    ]
    prior = _pool(estimates, fallback=Prior(*LITERATURE_PRIOR))
    n_ident = sum(e.identified for e in estimates)
    raw = np.array([e.beta for e in estimates if np.isfinite(e.beta)])
    log(f"  external prior from [bold]{path.name}[/bold]: {prior}")
    log(f"    {n_ident}/{len(estimates)} products identified (post sign-restriction); "
        f"{(raw > 0).mean():.0%} of estimable products were wrong-sign before it"
        f"{'; controls=' + ','.join(control_cols) if control_cols else ''}")
    return prior


def save_prior(prior: Prior, path: str | Path, **provenance) -> Path:
    """Persist a fitted prior so rebuilds do not repeat a multi-minute fit.

    Provenance is stored alongside the numbers: a prior with no record of which
    dataset and specification produced it is not reproducible, and this one is
    load-bearing for every elasticity in the RL environment.
    """
    import json
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(
        {"mu": prior.mu, "tau2": prior.tau2, **provenance}, indent=2))
    log(f"  prior -> {path}")
    return path


def load_prior(path: str | Path) -> Prior:
    import json
    d = json.loads(Path(path).read_text())
    return Prior(mu=float(d["mu"]), tau2=float(d["tau2"]))


def load_dunnhumby(raw_dir: str | Path, out_csv: str | Path,
                   *, with_causal: bool = True,
                   min_weeks: int = 20) -> Path:
    """
    Adapter for dunnhumby "The Complete Journey" -> the shape
    `fit_external_prior()` expects.

    Aggregates `transaction_data.csv` to **product x week**, pooled across all
    stores. Unit price is the pre-discount shelf price, recovered by subtracting
    the retailer and coupon discounts from the amount actually paid (both are
    stored as negative amounts).

    **Why not product x store x week**, which would match `causal_data.csv`'s key
    exactly: dunnhumby is a 2,500-household panel, not a scanner census. At that
    grain 74% of cells contain a single unit sold, the median product-store has
    ONE week of data, and within-product-store price variation is literally zero
    - so there is nothing to regress. Estimating there produces textbook
    attenuation: elasticities collapse toward zero (pooled mean -0.15) and the
    sign becomes a coin flip (49% positive). Pooling stores restores real weekly
    volume per product and makes the promotion cycle the price variation.

    With `with_causal=True` the `display` and `mailer` flags are joined on. Those
    are the point of using this dataset: they observe the in-store display and
    feature-advertising activity that confounds every price-elasticity regression
    and that M5 does not record at all.
    """
    raw_dir = Path(raw_dir)
    src = raw_dir / "transaction_data.csv"
    if not src.exists():
        raise FileNotFoundError(f"expected {src} (dunnhumby Complete Journey)")

    usecols = ["PRODUCT_ID", "STORE_ID", "WEEK_NO", "QUANTITY", "SALES_VALUE",
               "RETAIL_DISC", "COUPON_MATCH_DISC"]
    tx = pd.read_csv(src, usecols=lambda c: c.upper() in usecols)
    tx.columns = [c.upper() for c in tx.columns]
    tx = tx[(tx["QUANTITY"] > 0) & (tx["SALES_VALUE"] > 0)].copy()

    shelf = tx["SALES_VALUE"].astype(float)
    for disc in ("RETAIL_DISC", "COUPON_MATCH_DISC"):
        if disc in tx.columns:
            shelf = shelf - tx[disc].astype(float)
    tx["unit_price"] = shelf / tx["QUANTITY"].astype(float)
    # Drop implausible unit prices (returns, multi-buys, data errors).
    tx = tx[(tx["unit_price"] > 0.05) & (tx["unit_price"] < 100)]

    panel = (
        tx.groupby(["PRODUCT_ID", "WEEK_NO"])
          .agg(unit_price=("unit_price", "median"),
               units=("QUANTITY", "sum"),
               n_baskets=("QUANTITY", "size"))
          .reset_index()
    )

    if with_causal:
        causal_path = raw_dir / "causal_data.csv"
        if causal_path.exists():
            causal = pd.read_csv(
                causal_path, usecols=["PRODUCT_ID", "STORE_ID", "WEEK_NO",
                                      "display", "mailer"])
            # Both are categorical codes where "0"/"A" mean no activity; collapse
            # to binary, then average over stores to get the SHARE of stores
            # running the promotion that week - the right control once the
            # quantity side has been pooled across those same stores.
            causal["display"] = (causal["display"].astype(str) != "0").astype(int)
            causal["mailer"] = (~causal["mailer"].astype(str).isin(["0", "A"])).astype(int)
            weekly_causal = (causal.groupby(["PRODUCT_ID", "WEEK_NO"])
                                   [["display", "mailer"]].mean().reset_index())
            panel = panel.merge(weekly_causal, on=["PRODUCT_ID", "WEEK_NO"], how="left")
            panel[["display", "mailer"]] = panel[["display", "mailer"]].fillna(0.0)
            log(f"  joined causal_data: mean display share={panel.display.mean():.1%} "
                f"mailer share={panel.mailer.mean():.1%}")
        else:
            log("  [yellow]causal_data.csv not found — the prior will be "
                "confounded by unobserved promotion, same as M5[/yellow]")

    # A week's price is only meaningful if several baskets back it; a single
    # household's single purchase is noise, and noise in log(price) attenuates
    # the coefficient toward zero.
    panel = panel[panel["n_baskets"] >= 5]

    # Keep products with enough weeks for a per-product regression.
    counts = panel.groupby("PRODUCT_ID")["WEEK_NO"].transform("size")
    panel = panel[counts >= min_weeks]

    # `fit_item_elasticity` wants a `month` column for its seasonal dummies;
    # dunnhumby has no calendar, so derive pseudo-months from the week index.
    panel["month"] = ((panel["WEEK_NO"].astype(int) - 1) // 4 % 12) + 1

    out_csv = Path(out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    panel.to_csv(out_csv, index=False)
    log(f"  dunnhumby -> {out_csv} ({len(panel):,} product-week rows, "
        f"{panel.PRODUCT_ID.nunique():,} products, "
        f"median {panel.n_baskets.median():.0f} baskets/week)")
    return out_csv


# ─── The estimator, assembled ────────────────────────────────────────────────

@dataclass
class ElasticityReport:
    """What the shrinkage actually did — goes straight into the writeup."""
    values: Dict[tuple, float]        # (item_id, store_id) -> elasticity
    n_items: int
    n_identified: int
    n_clipped: int
    global_prior: Prior
    dept_priors: Dict[str, Prior]
    raw_betas: Dict[tuple, float]     # unshrunk estimate, for the before/after plot
    estimates: Dict[tuple, ItemEstimate]

    @property
    def identified_share(self) -> float:
        return self.n_identified / max(self.n_items, 1)

    @property
    def wrong_sign_share(self) -> float:
        """Share of estimable items whose raw coefficient came out positive.

        The headline endogeneity diagnostic: under OLS this ran at ~31% on the
        M5 CA_1 scope, which is what motivated the instrument.
        """
        raw = np.array([b for b in self.raw_betas.values() if np.isfinite(b)])
        return float((raw > 0).mean()) if len(raw) else 0.0

    def summary(self) -> str:
        v = np.array(list(self.values.values()), dtype=float)
        methods = pd.Series([e.method for e in self.estimates.values()]).value_counts()
        f_stats = np.array([e.first_stage_f for e in self.estimates.values()
                            if np.isfinite(e.first_stage_f)])
        lines = [
            f"elasticity: mean {v.mean():.3f}  median {np.median(v):.3f}  "
            f"min {v.min():.3f}  max {v.max():.3f}",
            f"  identified from own data : {self.n_identified}/{self.n_items} "
            f"({100 * self.identified_share:.0f}%)",
            f"  shrunk to a prior        : {self.n_items - self.n_identified} "
            f"(no hard-coded fallback)",
            f"  wrong-sign raw estimates : {100 * self.wrong_sign_share:.0f}% "
            f"(endogeneity diagnostic)",
            f"  touching the safety clip : {self.n_clipped} "
            f"({100 * self.n_clipped / max(self.n_items, 1):.0f}%)",
            f"  estimator                : "
            f"{', '.join(f'{k}={v_}' for k, v_ in methods.items())}",
        ]
        if len(f_stats):
            lines.append(f"  first-stage F            : median {np.median(f_stats):.1f}, "
                         f"{100 * (f_stats >= MIN_FIRST_STAGE_F).mean():.0f}% strong")
        lines.append(f"  global prior             : {self.global_prior}")
        return "\n".join(lines)


def estimate_elasticities(
    df: pd.DataFrame,
    *,
    external_prior: Optional[Prior] = None,
    item_key: Sequence[str] = ("item_id", "store_id"),
    dept_col: str = "dept_id",
    instrument_col: Optional[str] = "instrument_price",
) -> ElasticityReport:
    """
    Run the full three-level estimator over the long M5 frame.

    `df` needs: the `item_key` columns, `dept_col`, `demand`, `sell_price`,
    `month` — i.e. the frame `build_long()` already produces.

    Passing `external_prior=None` falls back to LITERATURE_PRIOR, which is a
    placeholder; fit a real one with `fit_external_prior()` for the report.
    """
    item_key = list(item_key)
    level0 = external_prior or Prior(*LITERATURE_PRIOR)
    if external_prior is None:
        log("  [yellow]no external prior supplied — using LITERATURE_PRIOR "
            "placeholder; fit a real one before quoting these numbers[/yellow]")

    # Level 2 — every item's own evidence, instrumented where possible.
    if instrument_col and instrument_col not in df.columns:
        log(f"  [yellow]no '{instrument_col}' column — falling back to OLS, which is "
            f"biased by promotion endogeneity; see build_price_instrument()[/yellow]")
        instrument_col = None

    estimates: Dict[tuple, ItemEstimate] = {}
    dept_of: Dict[tuple, str] = {}
    for key, g in df.groupby(item_key, sort=False):
        key = key if isinstance(key, tuple) else (key,)
        estimates[key] = fit_item_elasticity(g, instrument_col=instrument_col)
        dept_of[key] = str(g[dept_col].iloc[0]) if dept_col in g.columns else "_all"

    # Level 1 — pool within each department, itself shrunk toward the external
    # prior so a thin department does not get an overconfident mean.
    dept_priors: Dict[str, Prior] = {}
    for dept in sorted(set(dept_of.values())):
        members = [estimates[k] for k, d in dept_of.items() if d == dept]
        pooled = _pool(members, fallback=level0)
        mu = _posterior_mean(
            ItemEstimate(pooled.mu, np.sqrt(pooled.tau2), 0, 0.0), level0
        ) if pooled.tau2 > 0 else level0.mu
        dept_priors[dept] = Prior(mu=mu, tau2=pooled.tau2 or level0.tau2)

    # Posterior per item, then the wide safety clip.
    values: Dict[tuple, float] = {}
    raw: Dict[tuple, float] = {}
    n_clipped = 0
    for key, est in estimates.items():
        post = _posterior_mean(est, dept_priors[dept_of[key]])
        clipped = float(np.clip(post, *SAFETY_CLIP))
        if clipped != post:
            n_clipped += 1
        values[key] = clipped
        raw[key] = est.beta

    return ElasticityReport(
        values=values,
        n_items=len(estimates),
        n_identified=sum(e.identified for e in estimates.values()),
        n_clipped=n_clipped,
        global_prior=level0,
        dept_priors=dept_priors,
        raw_betas=raw,
        estimates=estimates,
    )
