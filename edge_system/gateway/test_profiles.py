"""Product tags: each is earned by a measurement, and missing data never breaks
the products page."""

import pytest

from ..config import SYSTEM_CONFIG
from . import profiles as pr


@pytest.mark.parametrize("price,cv,snap,el,expect", [
    (0.88, 1.07, 1.45, -0.79, ["Value", "Volatile", "SNAP-driven", "Price-insensitive"]),
    (1.58, 0.38, 1.04, -2.36, ["Everyday", "Steady", "Price-sensitive"]),
    (5.98, 0.90, 1.09, -1.20, ["Premium"]),          # middle of every band: tier only
    (3.00, 0.50, 1.20, -1.50, ["Premium", "SNAP-driven", "Price-sensitive"]),  # edges
])
def test_tags_follow_the_thresholds(price, cv, snap, el, expect):
    assert pr.tags_for(price, cv, snap, el) == expect


def test_every_product_gets_exactly_one_price_tier():
    for p in pr.profiles().values():
        assert sum(t in ("Value", "Everyday", "Premium") for t in p["tags"]) == 1


def test_the_demo_products_are_told_apart():
    """The tags exist to separate the ten products the demo trades; if every one
    got the same tags they would be decoration."""
    data = pr.profiles()
    if not data:
        pytest.skip("rl_environment.csv not on disk")
    top = sorted(data, key=lambda s: -data[s]["mean_daily"])[:10]
    assert len({tuple(data[s]["tags"]) for s in top}) >= 5


def test_missing_data_gives_no_tags_not_an_error(monkeypatch, tmp_path):
    monkeypatch.setitem(SYSTEM_CONFIG["paths"], "data_dir", str(tmp_path))
    assert pr.profiles() == {}
