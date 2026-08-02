"""
Tests for (s,S) replenishment.

This module had no tests, which is how `InventoryTrace.served` shipped defined as
`max(on_hand - unmet, 0)`. On any day without a stockout that returns the shelf
size instead of the sales, so `fill_rate` read near 100% and the flow balance was
off by the entire sales volume. Nothing consumed those properties, so nothing
broke - it was wrong, complete and plausible, which is the hard kind to catch.

The conservation tests below are the ones that would have caught it: they check
the arithmetic that must hold for the trace to describe a physical warehouse,
rather than checking that any particular number came out.
"""

from __future__ import annotations

import numpy as np
import pytest

from edge_system.inventory.replenishment import (calibrate_cover_days,
                                                 simulate_inventory)


def _steady(n: int = 200, rate: float = 10.0, seed: int = 0) -> np.ndarray:
    return np.random.default_rng(seed).poisson(rate, size=n).astype(float)


# ── conservation: the arithmetic of a physical warehouse ──────────────────────

def test_flow_balances():
    """Opening stock + everything received - everything sold = what is left."""
    demand = _steady()
    t = simulate_inventory(demand, cover_days=10.0, lead_time_days=3)
    opening = float(t.on_hand[0]) - float(t.received[0])   # before day-0 arrival
    closing = float(t.on_hand_after[-1])
    assert opening + t.received.sum() - t.served.sum() == pytest.approx(closing)


def test_served_plus_unmet_is_demand():
    """Every demanded unit is either sold or recorded as missed - never both,
    never neither. This is the identity the censoring signal depends on."""
    demand = _steady()
    t = simulate_inventory(demand, cover_days=4.0, lead_time_days=5)
    assert np.allclose(t.served + t.unmet, demand)
    assert np.allclose(t.demand, demand)


def test_served_is_min_of_stock_and_demand():
    demand = _steady()
    t = simulate_inventory(demand, cover_days=3.0, lead_time_days=4)
    assert np.allclose(t.served, np.minimum(t.on_hand, demand))


def test_served_is_not_merely_on_hand_when_stock_is_plentiful():
    """The regression guard. With generous cover there are no stockouts, so a
    formula in terms of `unmet` collapses to `on_hand` and looks fine."""
    demand = _steady(rate=5.0)
    t = simulate_inventory(demand, cover_days=60.0, lead_time_days=1)
    assert t.unmet.sum() == 0
    assert np.allclose(t.served, demand)
    assert not np.allclose(t.served, t.on_hand)      # would pass under the bug


def test_stock_never_goes_negative():
    demand = _steady(rate=30.0)
    t = simulate_inventory(demand, cover_days=2.0, lead_time_days=7)
    assert (t.on_hand >= 0).all()
    assert (t.on_hand_after >= -1e-9).all()


def test_stockout_flag_matches_unmet():
    demand = _steady(rate=25.0)
    t = simulate_inventory(demand, cover_days=2.0, lead_time_days=6)
    assert np.array_equal(t.stockout.astype(bool), t.unmet > 0)


def test_fill_rate_is_units_served_over_units_demanded():
    demand = _steady(rate=20.0)
    t = simulate_inventory(demand, cover_days=2.5, lead_time_days=5)
    assert t.fill_rate == pytest.approx(t.served.sum() / demand.sum())
    assert 0.0 < t.fill_rate < 1.0          # this run must actually miss some


# ── behaviour: the parameters must mean what the docstring says ───────────────

def test_lead_time_is_what_makes_stockouts_possible():
    """At zero lead time the model degenerates to instant refill - the toy loop
    this module replaced. The dataset's censoring signal exists only because
    deliveries take time."""
    demand = _steady(rate=15.0)
    instant = simulate_inventory(demand, cover_days=5.0, lead_time_days=0)
    delayed = simulate_inventory(demand, cover_days=5.0, lead_time_days=7)
    assert delayed.stockout_rate > instant.stockout_rate


def test_more_cover_means_fewer_stockouts():
    """Monotonicity is what `calibrate_cover_days` bisects on."""
    demand = _steady(rate=15.0)
    rates = [simulate_inventory(demand, cover_days=c, lead_time_days=3).stockout_rate
             for c in (2.0, 5.0, 10.0, 20.0)]
    assert rates == sorted(rates, reverse=True)


def test_orders_on_inventory_position_not_on_hand():
    """Ordering on on-hand alone re-orders every day of the lead time, stacking
    up duplicate deliveries. Position accounting means at most one order is in
    flight per reorder point crossing."""
    demand = _steady(rate=10.0)
    t = simulate_inventory(demand, cover_days=10.0, lead_time_days=5)
    n_deliveries = int((t.received > 0).sum())
    assert n_deliveries < len(demand) / 5


def test_calibrate_hits_the_target_stockout_rate():
    demand = _steady(rate=12.0)
    cover = calibrate_cover_days(demand, target_stockout_rate=0.05,
                                 lead_time_days=3)
    rate = simulate_inventory(demand, cover_days=cover,
                              lead_time_days=3).stockout_rate
    assert rate == pytest.approx(0.05, abs=0.02)


def test_zero_demand_series_is_handled():
    t = simulate_inventory(np.zeros(50), cover_days=10.0)
    assert t.unmet.sum() == 0 and t.served.sum() == 0
    assert t.fill_rate == 1.0
