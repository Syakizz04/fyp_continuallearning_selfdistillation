"""
Demand censoring - the second channel by which inventory sync error corrupts a model.

## The mechanism

A node cannot observe demand. It observes *sales*. When it refuses an order -
because its escrow quota is exhausted, or because the shelf is empty - the
customer's demand still existed, but nothing records it. Retrain the forecaster
on what it observed and it learns the product is less wanted than it is.

That error then feeds itself: a forecaster that under-predicts orders less stock,
which causes more refusals, which censors more demand. Censoring is not noise
added to a target; it is a **systematically negative, self-reinforcing bias whose
severity is set by the sync policy**.

## Why this is E2's stronger channel

E1 measured the cost of each sync policy as a fill rate: strong_lock 82.7%,
escrow_quota 71.7%, eventual 100% (while overselling). The refused share is
exactly the demand that never gets recorded. So the same dial that produced E1's
headline result produces this treatment - the two experiments measure one
quantity from two sides, rather than being separate stories bolted together.

It is also the channel that rests on **real data**. The pricing channel depends
on an elasticity that the audit found to be 71% prior and 29% data, driving
counterfactual demand at prices M5 never observed. Censoring needs none of that:
`unmet_demand` is derived from real M5 units and a stated replenishment policy,
and the target being corrupted is real observed sales.

## The one rule

**Train on censored demand, evaluate against true demand.** The censoring is the
treatment; the truth is the yardstick. `assert_uncensored` exists to make a
violation loud, because an evaluation frame that was quietly censored too would
show a model scoring well on its own bias.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

__all__ = ["CensoringSpec", "apply_censoring", "censoring_report",
           "assert_uncensored", "attach_unmet_demand", "MERGE_KEYS"]

MERGE_KEYS = ["product_id", "region_id", "date"]

#: Fill rates measured in E1, so a spec can name a policy instead of a number.
E1_FILL_RATES: Dict[str, float] = {
    "strong_lock": 0.827,
    "escrow_quota": 0.717,
    "eventual": 1.000,
}


@dataclass(frozen=True)
class CensoringSpec:
    """How much demand the node fails to observe, and where.

    mode
        ``none``      - control arm; the target is untouched.
        ``intrinsic`` - only the stockouts already in the dataset
                        (``unmet_demand``). This is what a *perfectly*
                        synchronised system would still lose, so it is the floor,
                        not a treatment.
        ``policy``    - intrinsic, plus the additional refusals a sync policy
                        causes, scaled to hit ``fill_rate``.
    fill_rate
        Target share of demanded units actually served, for ``policy`` mode.
        Defaults to escrow_quota's measured 71.7%.
    scarcity_power
        How sharply extra refusals concentrate on days when stock is tight
        relative to demand. 0 spreads them uniformly at random; higher values
        push them onto scarce days. Uniform censoring would be both unrealistic
        and *easier* for a model to shrug off, since it adds no correlation
        between the bias and the features - so the default is 1.0, not 0.
    """

    mode: str = "none"
    fill_rate: float = 0.717
    scarcity_power: float = 1.0
    seed: int = 42

    @classmethod
    def for_policy(cls, policy: str, **kw) -> "CensoringSpec":
        """Build a spec from an E1 policy name, so the link is explicit."""
        if policy not in E1_FILL_RATES:
            raise ValueError(f"unknown policy {policy!r}; "
                             f"expected one of {sorted(E1_FILL_RATES)}")
        return cls(mode="policy", fill_rate=E1_FILL_RATES[policy], **kw)

    @property
    def label(self) -> str:
        if self.mode == "none":
            return "none"
        if self.mode == "intrinsic":
            return "intrinsic"
        return f"policy@{self.fill_rate:.3f}"


def _scarcity_weights(demand: np.ndarray, on_hand: np.ndarray,
                      power: float) -> np.ndarray:
    """Relative propensity for an order to be refused on each day.

    A node refuses when what it can serve falls short of what is asked, so the
    propensity rises with demand and falls with stock. `+1` keeps a zero-stock
    day finite rather than infinite.
    """
    if power <= 0:
        return np.ones_like(demand, dtype=float)
    ratio = demand / (on_hand + 1.0)
    w = np.power(np.clip(ratio, 0.0, None), power)
    total = w.sum()
    return w / total if total > 0 else np.ones_like(w) / max(len(w), 1)


def attach_unmet_demand(rl_df: pd.DataFrame, rl_csv: str) -> pd.DataFrame:
    """Read `unmet_demand` back onto an already-prepared RL frame.

    `RL_REQUIRED_COLUMNS` in core_pipeline deliberately narrows the RL frame to
    what the pricing environment consumes, and `unmet_demand` is not part of
    that - the agent never sees it. Censoring does need it, but widening the
    pipeline's column contract to serve one experiment would change what every
    RL arm loads, so it is read back here instead.

    A left join, because `filter_to_base_trainable` drops late-introduced series
    from `rl_df`; rows the CSV cannot match censor by 0.
    """
    if "unmet_demand" in rl_df.columns:
        return rl_df

    raw = pd.read_csv(rl_csv, usecols=MERGE_KEYS + ["unmet_demand"])
    raw["date"] = pd.to_datetime(raw["date"])
    if raw.duplicated(MERGE_KEYS).any():
        raise ValueError(f"{rl_csv} has duplicate {MERGE_KEYS} - a join on it "
                         f"would multiply rows")

    out = rl_df.copy()
    keys = out[MERGE_KEYS].copy()
    keys["date"] = pd.to_datetime(keys["date"])
    merged = keys.merge(raw, on=MERGE_KEYS, how="left")
    if len(merged) != len(out):
        raise ValueError(f"unmet_demand join changed row count "
                         f"({len(out)} -> {len(merged)})")
    out["unmet_demand"] = merged["unmet_demand"].fillna(0.0).to_numpy()
    return out


def apply_censoring(tft_df: pd.DataFrame, rl_df: pd.DataFrame,
                    spec: CensoringSpec, *,
                    target: str = "demand") -> Tuple[pd.DataFrame, Dict]:
    """Return a copy of `tft_df` whose target is what the node would have seen.

    Only the target column changes. Row order, index and every other column are
    preserved, because the TFT requires `time_idx` to stay a consecutive integer
    per series - reordering or dropping rows here would silently shrink the
    dataset via `filter_tft_eval_frame` rather than fail.
    """
    if spec.mode not in {"none", "intrinsic", "policy"}:
        raise ValueError(f"unknown censoring mode {spec.mode!r}")

    out = tft_df.copy()
    if spec.mode == "none":
        return out, censoring_report(tft_df, out, spec, target=target)

    # Align the censoring signal onto the forecasting frame. A left join keeps
    # every TFT row even if the RL frame is missing one; those rows censor by 0.
    cols = MERGE_KEYS + ["unmet_demand", "inventory_level"]
    missing = [c for c in cols if c not in rl_df.columns]
    if missing:
        raise KeyError(f"rl_df is missing {missing}")

    left = out[MERGE_KEYS].copy()
    for frame in (left, ):
        frame["date"] = pd.to_datetime(frame["date"])
    right = rl_df[cols].copy()
    right["date"] = pd.to_datetime(right["date"])
    merged = left.merge(right, on=MERGE_KEYS, how="left")
    if len(merged) != len(out):
        raise ValueError(f"censoring join changed row count "
                         f"({len(out)} -> {len(merged)}); rl_df has duplicate keys")

    demand = out[target].to_numpy(dtype=float)
    unmet = merged["unmet_demand"].fillna(0.0).to_numpy(dtype=float)
    on_hand = merged["inventory_level"].fillna(0.0).to_numpy(dtype=float)

    # Floor: what the dataset's own stockouts already hide.
    observed = np.maximum(demand - unmet, 0.0)

    if spec.mode == "policy":
        total = float(demand.sum())
        want_served = spec.fill_rate * total
        already = float(observed.sum())
        extra = already - want_served
        if extra > 0:
            # Distribute the additional refusals over days, weighted by scarcity
            # and capped by what is left to take on each day. Redistribute the
            # overflow so the requested fill rate is actually reached rather
            # than merely approached.
            w = _scarcity_weights(demand, on_hand, spec.scarcity_power)
            remaining = extra
            headroom = observed.copy()
            for _ in range(8):
                if remaining <= 1e-9 or headroom.sum() <= 1e-9:
                    break
                active = headroom > 0
                wa = np.where(active, w, 0.0)
                if wa.sum() <= 0:
                    wa = active.astype(float)
                    if wa.sum() <= 0:
                        break
                take = np.minimum(headroom, remaining * wa / wa.sum())
                headroom -= take
                remaining -= float(take.sum())
            observed = headroom

    out[target] = observed
    return out, censoring_report(tft_df, out, spec, target=target)


def censoring_report(true_df: pd.DataFrame, censored_df: pd.DataFrame,
                     spec: Optional[CensoringSpec] = None, *,
                     target: str = "demand") -> Dict:
    """Quantify the bias introduced, in the terms a result table needs."""
    t = true_df[target].to_numpy(dtype=float)
    c = censored_df[target].to_numpy(dtype=float)
    lost = t - c
    total = float(t.sum())
    nonzero = t > 0
    achieved = float(c.sum() / total) if total > 0 else 1.0

    # A requested fill rate ABOVE the intrinsic floor is unreachable: censoring
    # only removes, so the dataset's own stockouts cannot be undone. Floor at
    # intrinsic and say so - silently returning a different treatment than the
    # one named is how a sweep ends up with two cells that are secretly equal.
    # Unreachable means we ended up serving LESS than asked, because the floor
    # sits below the target. Censoring more is always possible; censoring less
    # is not.
    requested = spec.fill_rate if (spec and spec.mode == "policy") else None
    reachable = True if requested is None else bool(achieved >= requested - 1e-3)

    return {
        "spec": spec.label if spec else "",
        "mode": spec.mode if spec else "",
        "n_rows": int(len(t)),
        "units_true": total,
        "units_observed": float(c.sum()),
        "units_censored": float(lost.sum()),
        "censored_share": float(lost.sum() / total) if total > 0 else 0.0,
        "fill_rate": achieved,
        "fill_rate_requested": requested,
        "target_reachable": reachable,
        "rows_affected": int((lost > 1e-9).sum()),
        "rows_affected_share": float((lost > 1e-9).mean()),
        # The bias a forecaster would internalise: mean relative understatement
        # on the rows that carry signal.
        "mean_relative_bias": float(
            np.mean(lost[nonzero] / t[nonzero])) if nonzero.any() else 0.0,
        "max_row_censored": float(lost.max()) if len(lost) else 0.0,
    }


def censor_data_dict(data: Dict, spec: CensoringSpec, *,
                     target: str = "demand") -> Dict:
    """Add censored TRAINING frames to a `prepare_drift_data` dict.

    Returns a shallow copy carrying two extra keys, ``tft_full_censored`` and
    ``tft_base_censored``. The originals are left in place and untouched, which
    is the point: `RetrainController` trains from the censored frames while the
    monitor keeps scoring against the true ones. Nothing is overwritten, so an
    arm that forgets to opt in gets the uncensored control rather than a silently
    corrupted yardstick.
    """
    out = dict(data)
    if spec.mode == "none":
        out["censoring"] = censoring_report(
            data["tft_full"], data["tft_full"], spec, target=target)
        return out

    rl_full = data["rl_full"]
    full, report = apply_censoring(data["tft_full"], rl_full, spec, target=target)
    base, _ = apply_censoring(data["tft_base"], rl_full, spec, target=target)
    out["tft_full_censored"] = full
    out["tft_base_censored"] = base
    out["censoring"] = report
    return out


def assert_uncensored(df: pd.DataFrame, rl_df: pd.DataFrame, *,
                      target: str = "demand", tol: float = 1e-6) -> None:
    """Fail loudly if an EVALUATION frame has been censored.

    Training on censored demand while scoring against censored demand would let
    a model look accurate by reproducing the bias it was trained on. The whole
    experiment depends on the yardstick staying true, so this is an assertion
    rather than a warning.
    """
    keys = df[MERGE_KEYS].copy()
    keys["date"] = pd.to_datetime(keys["date"])
    right = rl_df[MERGE_KEYS + ["realized_demand"]].copy()
    right["date"] = pd.to_datetime(right["date"])
    merged = keys.merge(right, on=MERGE_KEYS, how="left")
    truth = merged["realized_demand"].to_numpy(dtype=float)
    got = df[target].to_numpy(dtype=float)
    ok = np.isnan(truth) | (np.abs(got - truth) <= tol)
    if not ok.all():
        n = int((~ok).sum())
        raise AssertionError(
            f"evaluation frame is censored: {n} of {len(got)} rows differ from "
            f"realized_demand (max {np.nanmax(np.abs(got - truth)):.4g}). "
            f"Evaluate against TRUE demand - censoring is the treatment, not "
            f"the yardstick.")
