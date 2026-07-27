"""
Turns real M5 demand into discrete order arrivals on three sales channels.

The chain is:

    real M5 realized_demand  ->  price response  ->  channel split  ->  orders

Only the last two steps are simulated, and both are simulated in the smallest
way that produces something the inventory service can be tested against: the
demand *level* is never invented.

## The price response

    units(p) = realized_demand * (p / base_price) ** elasticity

Constant-elasticity, and deliberately the **same functional form the PPO agent
was trained against** - `DynamicPricingEnv._precompute` computes its reward as
`demand * (price/base_price) ** elasticity`. If the simulated world responded to
price differently from the agent's training environment, every pricing result
would be measuring that mismatch rather than the agent.

The elasticity used here is the **raw** `elasticity_coefficient` from
`rl_environment.csv` - the hierarchically-estimated value, IV-identified for 54
of the 100 items and shrunk to a department/external prior for the rest. This
is the world's true elasticity. Note that the env *normalises* elasticity before
using it as an exponent, so the agent's internal model of price response is a
scaled version of this one; see the note in `edge_system/README.md`.

## Arrivals

Expected units are split across channels by their configured share and turned
into discrete orders, because the inventory service is a *reservation* system:
what stresses it is the number of concurrent claims, not the aggregate volume.
Basket size is drawn as `1 + Poisson(mean_basket - 1)` so every order is for at
least one unit, and order count is Poisson with the mean chosen to reproduce the
target unit volume in expectation.

Every draw is seeded from `(seed, tick, sku, channel)` rather than from a
running RNG, so a cell of an E1 sweep sees **exactly** the same order stream as
every other cell. Demand is a control variable, not a source of variance: if
`strong_lock` and `escrow_quota` saw different orders, the comparison between
them would be worthless.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Order:
    """One customer's attempt to buy, before the inventory service sees it."""

    sku: str
    channel: str
    qty: int
    unit_price: float
    tick: int
    sim_date: str

    @property
    def value(self) -> float:
        return self.qty * self.unit_price


class OrderGenerator:
    """
    Demand model for the simulated storefront.

    Torch-free by construction - it reads `rl_environment.csv` with pandas and
    nothing else - so E1 can run the full sync sweep without loading a model.
    """

    def __init__(self, rl_frame: pd.DataFrame, *, seed: int = 42,
                 mean_basket: float = 1.8,
                 channel_weights: Optional[Dict[str, float]] = None) -> None:
        if mean_basket < 1.0:
            raise ValueError(f"mean_basket must be >= 1, got {mean_basket}")

        df = rl_frame.copy()
        df["date"] = pd.to_datetime(df["date"]).dt.normalize()
        self.df = df.sort_values(["product_id", "date"]).reset_index(drop=True)
        self.seed = int(seed)
        self.mean_basket = float(mean_basket)
        self.channel_weights = dict(channel_weights or {})

        # (sku, date) -> row. Built once: the tick loop hits this ~n_skus x
        # n_channels times per tick and a boolean mask over 184k rows each time
        # would dominate the runtime of a no-model E1 run.
        self._rows: Dict[tuple, Dict] = {
            (r.product_id, r.date): {
                "demand": float(r.realized_demand),
                "base_price": float(r.base_price),
                "elasticity": float(r.elasticity_coefficient),
                "competitor_price": float(r.competitor_price),
            }
            for r in self.df.itertuples(index=False)
        }

        agg = self.df.groupby("product_id")["realized_demand"]
        self._mean_daily: Dict[str, float] = agg.mean().to_dict()
        self._elasticity: Dict[str, float] = (
            self.df.groupby("product_id")["elasticity_coefficient"].first().to_dict()
        )

    # ── Data access ─────────────────────────────────────────────────────────

    @classmethod
    def from_csv(cls, path: str | Path, **kwargs) -> "OrderGenerator":
        return cls(pd.read_csv(path), **kwargs)

    def skus(self, limit: Optional[int] = None) -> List[str]:
        """SKUs in a stable order, so `--n-skus 10` picks the same ten every run."""
        out = sorted(self._mean_daily)
        return out if limit is None else out[:limit]

    @property
    def dates(self) -> pd.Series:
        return self.df["date"]

    def mean_daily_demand(self, sku: str) -> float:
        return float(self._mean_daily.get(sku, 0.0))

    def elasticity(self, sku: str) -> float:
        return float(self._elasticity.get(sku, -1.0))

    def row(self, sku: str, date) -> Optional[Dict]:
        return self._rows.get((sku, pd.Timestamp(date).normalize()))

    def base_price(self, sku: str, date) -> Optional[float]:
        r = self.row(sku, date)
        return None if r is None else r["base_price"]

    # ── The demand model ────────────────────────────────────────────────────

    def expected_units(self, sku: str, date, quoted_price: Optional[float] = None
                       ) -> float:
        """
        Units demanded across all channels at `quoted_price`.

        At `quoted_price == base_price` this returns the real M5 figure exactly,
        which keeps the no-model baseline anchored to observed data.
        """
        r = self.row(sku, date)
        if r is None:
            return 0.0
        demand = max(r["demand"], 0.0)
        if quoted_price is None or r["base_price"] <= 0:
            return demand
        ratio = max(quoted_price, 1e-6) / r["base_price"]
        return float(demand * ratio ** r["elasticity"])

    def orders(self, sku: str, date, channel: str, quoted_price: float, *,
               tick: int, weight: Optional[float] = None) -> List[Order]:
        """Discrete arrivals for one (SKU, channel) on one day."""
        share = self.channel_weights.get(channel, 1.0) if weight is None else weight
        units = self.expected_units(sku, date, quoted_price) * share
        if units <= 0:
            return []

        rng = self._rng(tick, sku, channel)
        extra = self.mean_basket - 1.0
        n_orders = int(rng.poisson(max(units / self.mean_basket, 1e-9)))
        if n_orders <= 0:
            return []

        sizes = 1 + (rng.poisson(extra, size=n_orders) if extra > 0
                     else np.zeros(n_orders, dtype=int))
        sim_date = pd.Timestamp(date).strftime("%Y-%m-%d")
        return [
            Order(sku=sku, channel=channel, qty=int(q), unit_price=float(quoted_price),
                  tick=tick, sim_date=sim_date)
            for q in sizes
        ]

    # ── Stock sizing (used by the driver's fixed (s,S) replenishment) ───────

    def initial_stock(self, sku: str, cover_days: float) -> int:
        """
        Opening stock as a multiple of mean daily demand.

        Ordering is explicitly out of scope - the PPO agent controls price only -
        so this stays a fixed policy. It is a *parameter of the experiment*
        though: a low cover forces contention on the pool, which is what E1 needs
        to see, and a high cover would make every policy look identical because
        nothing ever runs out.
        """
        return max(1, int(round(self.mean_daily_demand(sku) * cover_days)))

    # ── Internals ───────────────────────────────────────────────────────────

    def _rng(self, tick: int, sku: str, channel: str) -> np.random.Generator:
        """
        A generator seeded by content, not by call order.

        This is what makes the sweep cells comparable: the stream for
        (tick 41, FOODS_3_090, web) is identical no matter which policy is under
        test, how many hops it took, or in what order the driver happened to walk
        the SKUs. A shared running RNG would couple the demand stream to the
        policy's control flow and quietly confound every comparison.
        """
        key = f"{self.seed}|{tick}|{sku}|{channel}".encode()
        digest = hashlib.blake2b(key, digest_size=8).digest()
        return np.random.default_rng(int.from_bytes(digest, "big"))
