"""The driver reports, per channel and SKU, the price charged and what came of
it - the inputs a node judges its pricer on (models.rl_signal = "critic").
Fakes only: no services, no models."""

from __future__ import annotations

import random
from types import SimpleNamespace

from edge_system.sim.driver import SimulationDriver, TickRecord


class FakeResponse:
    def __init__(self, body=None):
        self._body = body or {}

    def raise_for_status(self):
        return None

    def json(self):
        return self._body


class FakeInventory:
    """Grants the first `stock` units, then refuses."""

    def __init__(self, stock: int):
        self.stock = stock

    def post(self, path, json=None):
        if path == "/reserve":
            ok = json["qty"] <= self.stock
            if ok:
                self.stock -= json["qty"]
            return FakeResponse({"granted": ok, "reservation_id": "r", "latency_ms": 0.0})
        return FakeResponse()


class FakeNode:
    def __init__(self):
        self.checks = []

    def post(self, path, json=None):
        self.checks.append(json)
        return FakeResponse()


def driver(stock: int) -> SimulationDriver:
    d = object.__new__(SimulationDriver)
    d.inventory = FakeInventory(stock)
    d.rng = random.Random(0)
    d.channels = ["pos", "web"]
    d._outcomes = {"pos": {}, "web": {}}
    return d


def order(channel, sku, qty, value):
    return SimpleNamespace(channel=channel, sku=sku, qty=qty, value=value,
                           tick=0, sim_date="2013-01-01")


def test_outcomes_count_requested_sold_and_refused_per_sku(monkeypatch):
    from edge_system.config import SYSTEM_CONFIG
    monkeypatch.setitem(SYSTEM_CONFIG["sim"], "abandon_rate", 0.0)
    d = driver(stock=5)
    d._outcomes["pos"]["A"] = {"sku": "A", "price": 2.0, "requested": 0, "sold": 0, "refused": 0}
    rec = TickRecord(tick=0, sim_date="2013-01-01", channel="pos", policy="p", delay_ms=0.0)
    for q in (3, 2, 4):                     # 5 in stock: 3 + 2 sold, then 4 refused
        d._place(order("pos", "A", q, q * 2.0), rec)
    assert d._outcomes["pos"]["A"] == {"sku": "A", "price": 2.0,
                                       "requested": 9, "sold": 5, "refused": 4}
    assert (rec.units_requested, rec.units_committed, rec.units_refused) == (9, 5, 4)


def test_an_order_for_an_unquoted_sku_still_trades():
    """A SKU whose quote failed has no outcome row; its orders must not crash."""
    d = driver(stock=10)
    rec = TickRecord(tick=0, sim_date="2013-01-01", channel="web", policy="p", delay_ms=0.0)
    d._place(order("web", "Z", 2, 4.0), rec)
    assert rec.units_requested == 2 and d._outcomes["web"] == {}


def test_check_carries_each_channels_own_outcomes():
    d = driver(stock=10)
    d.pricing = "node"
    d.skus = ["A"]
    d.gen = SimpleNamespace(expected_units=lambda *a: 4.0, base_price=lambda *a: 2.0)
    d.nodes = {"pos": FakeNode(), "web": FakeNode()}
    d._outcomes["pos"]["A"] = {"sku": "A", "price": 2.2, "requested": 3, "sold": 3, "refused": 0}
    per = {ch: TickRecord(tick=0, sim_date="2013-01-01", channel=ch, policy="p", delay_ms=0.0)
           for ch in d.channels}
    d._check_drift(SimpleNamespace(index=0, date="2013-01-01", date_str="2013-01-01"), per)
    assert d.nodes["pos"].checks[0]["pricing"] == [d._outcomes["pos"]["A"]]
    assert d.nodes["web"].checks[0]["pricing"] == []
    # The revenue index still travels alongside, for rl_signal = "revenue".
    assert "revenue" in d.nodes["pos"].checks[0] and "reference" in d.nodes["pos"].checks[0]
