"""
Replenishment policy: how stock gets back on the shelf.

Pure and dependency-free (numpy only) so both the dataset builder and the live
simulation can share one policy instead of each carrying its own toy loop.

## Why this module exists

The original `dataset_generator/m5/_simulate_inventory` was an (s,S) loop with
**instant replenishment**: when stock fell below the reorder point it jumped back
to target the same night. The consequence was measured, and it is severe -
stock ran out on **0.05% of days**, five in ten thousand, and never left the band
between 10 and 21 days of cover.

That matters far beyond tidiness, because `inventory_level` is a **state feature
of the pricing agent**. The PPO was trained on a number that never once signalled
scarcity, so it correctly learned to ignore it: sweeping the inventory input
across its entire range changes under 5% of its pricing decisions. An experiment
that degrades the inventory signal and looks for a response is then measuring
almost nothing - not because the hypothesis is wrong, but because the training
data taught the agent the feature was useless.

## What makes stock actually run out

**Lead time.** An order placed today arrives in `lead_time_days`. Between those
two moments, demand keeps arriving and there is no way to react. That gap is what
produces genuine stockouts, and no amount of tuning the reorder point removes it
- it only trades stockouts against holding more stock, which is the real
trade-off a retailer faces and the one the agent should see.

The policy is textbook (s,S) on **inventory position** (on hand + on order),
rather than on hand alone. Ordering on on-hand alone re-orders every day of the
lead time because the earlier orders are invisible, which produces a stock
explosion on arrival.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np


@dataclass
class InventoryTrace:
    """Per-day outcome of running the policy against a demand series."""

    #: Stock on hand at the START of each day - what a pricing agent would see.
    on_hand: np.ndarray
    #: 1 on days demand exceeded what was available.
    stockout: np.ndarray
    #: Units demanded but not served. This is the CENSORING signal: on these
    #: days observed sales understate true demand, and a forecaster retrained on
    #: observed sales learns that the product is less wanted than it is.
    unmet: np.ndarray
    #: Units received that day.
    received: np.ndarray

    @property
    def stockout_rate(self) -> float:
        return float(self.stockout.mean())

    @property
    def fill_rate(self) -> float:
        """Share of demanded units actually served."""
        served = self.served.sum()
        total = served + self.unmet.sum()
        return float(served / total) if total > 0 else 1.0

    @property
    def served(self) -> np.ndarray:
        return np.maximum(self.on_hand_after, 0)

    @property
    def on_hand_after(self) -> np.ndarray:
        return self.on_hand - self.unmet


def simulate_inventory(
    demand: np.ndarray,
    *,
    cover_days: float = 10.0,
    reorder_frac: float = 0.5,
    lead_time_days: int = 3,
    initial_frac: float = 1.0,
) -> InventoryTrace:
    """
    Run (s,S) replenishment with a delivery lead time over a demand series.

    Parameters
    ----------
    cover_days
        Order-up-to level, in days of mean demand.
    reorder_frac
        Reorder when inventory position falls to this fraction of the level.
    lead_time_days
        Days between placing an order and receiving it. **The parameter that
        makes stockouts possible at all** - at 0 this degenerates to the
        instant-replenishment loop it replaces.
    initial_frac
        Opening stock as a fraction of the order-up-to level.
    """
    demand = np.asarray(demand, dtype=float)
    n = len(demand)
    mean_d = max(float(np.mean(demand)), 1e-6)

    level = max(cover_days * mean_d, 1.0)
    reorder_point = reorder_frac * level

    on_hand = np.zeros(n)
    stockout = np.zeros(n, dtype=int)
    unmet = np.zeros(n)
    received = np.zeros(n)

    stock = level * initial_frac
    pipeline: Dict[int, float] = {}

    for i in range(n):
        arriving = pipeline.pop(i, 0.0)
        stock += arriving
        received[i] = arriving

        # Recorded BEFORE the day's sales: this is the figure a pricing decision
        # for that day would be made against.
        on_hand[i] = stock

        want = demand[i]
        served = min(stock, want)
        if want > stock:
            stockout[i] = 1
            unmet[i] = want - stock
        stock -= served

        # Order on inventory POSITION, so orders already in transit count.
        position = stock + sum(pipeline.values())
        if position <= reorder_point:
            pipeline[i + max(1, lead_time_days)] = level - position

    return InventoryTrace(on_hand=on_hand, stockout=stockout,
                          unmet=unmet, received=received)


def calibrate_cover_days(
    demand: np.ndarray,
    *,
    target_stockout_rate: float = 0.04,
    lead_time_days: int = 3,
    reorder_frac: float = 0.5,
    lo: float = 1.0,
    hi: float = 40.0,
    tol: float = 0.002,
    max_iter: int = 40,
) -> float:
    """
    Find the cover that hits a target stockout rate, by bisection.

    Service level is the thing with a defensible target - real grocery retail
    runs at roughly 92-98% - whereas "days of cover" is an implementation detail
    whose meaning changes with demand variability. Setting the interpretable
    quantity and solving for the other keeps the dataset honest across SKUs with
    very different demand shapes.

    Monotone in `cover_days`: more stock, fewer stockouts.
    """
    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        rate = simulate_inventory(demand, cover_days=mid,
                                  reorder_frac=reorder_frac,
                                  lead_time_days=lead_time_days).stockout_rate
        if abs(rate - target_stockout_rate) <= tol:
            return mid
        if rate > target_stockout_rate:
            lo = mid          # too many stockouts -> hold more
        else:
            hi = mid
    return 0.5 * (lo + hi)
