"""
Unit tests for the simulation driver's three primitives.

Deliberately torch-free and service-free: these test the demand model, the
clock and the network dial in isolation. The end-to-end check is
`python -m edge_system.run_system --scenario smoke --ticks 30`.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from edge_system.sim.clock import SimClock
from edge_system.sim.network import NetworkConditions, Partitioned, SimNetwork
from edge_system.sim.order_gen import OrderGenerator


# ─── Fixtures ────────────────────────────────────────────────────────────────

@pytest.fixture
def rl_frame() -> pd.DataFrame:
    dates = pd.date_range("2013-01-01", periods=40, freq="D")
    rows = []
    for sku, elast, demand, price in (("A", -0.5, 20.0, 10.0),
                                      ("B", -2.5, 8.0, 4.0)):
        for d in dates:
            rows.append({"date": d, "product_id": sku, "realized_demand": demand,
                         "base_price": price, "elasticity_coefficient": elast,
                         "competitor_price": price * 1.05})
    return pd.DataFrame(rows)


@pytest.fixture
def gen(rl_frame) -> OrderGenerator:
    return OrderGenerator(rl_frame, seed=7, mean_basket=1.8,
                          channel_weights={"pos": 0.55, "web": 0.35,
                                           "marketplace": 0.10})


# ─── Clock ───────────────────────────────────────────────────────────────────

def test_clock_spans_the_requested_window():
    clock = SimClock("2013-01-01", "2013-01-10")
    assert len(clock) == 10
    ticks = list(clock)
    assert ticks[0].is_first and ticks[-1].is_last
    assert ticks[0].date_str == "2013-01-01"
    assert [t.index for t in ticks] == list(range(10))


def test_clock_drops_days_the_data_cannot_serve(rl_frame):
    # Data stops on 2013-02-09; asking for March must not manufacture ticks.
    clock = SimClock("2013-01-01", "2013-03-31",
                     available_dates=rl_frame["date"])
    assert len(clock) == 40
    assert clock.dates[-1] == pd.Timestamp("2013-02-09")


def test_clock_max_ticks_truncates():
    assert len(SimClock("2013-01-01", "2013-12-31", max_ticks=5)) == 5


def test_clock_rejects_an_empty_window():
    with pytest.raises(ValueError, match="no simulable days"):
        SimClock("2013-01-01", "2013-01-10",
                 available_dates=pd.Series(pd.to_datetime(["2020-01-01"])))


# ─── Network ─────────────────────────────────────────────────────────────────

def test_network_sleeps_for_the_configured_delay():
    slept = []
    net = SimNetwork(NetworkConditions(delay_ms=200.0), sleep=slept.append)
    assert net.hop("pos") is True
    assert slept == [pytest.approx(0.2)]
    assert net.stats()["hops"] == 1


def test_network_jitter_stays_within_bounds():
    slept = []
    net = SimNetwork(NetworkConditions(delay_ms=100.0, jitter_ms=20.0),
                     sleep=slept.append)
    for _ in range(200):
        net.hop("pos")
    assert all(0.08 - 1e-9 <= s <= 0.12 + 1e-9 for s in slept)
    assert np.mean(slept) == pytest.approx(0.1, abs=0.005)


def test_partitioned_node_is_blocked_and_pays_no_delay():
    slept = []
    net = SimNetwork(NetworkConditions(delay_ms=500.0, partition={"marketplace"}),
                     sleep=slept.append)
    assert net.hop("marketplace") is False
    assert slept == []                      # a dropped hop costs no wall time
    assert net.hop("pos") is True
    assert net.stats()["hops_blocked"] == 1


def test_delay_raises_for_callers_that_want_an_exception():
    net = SimNetwork(NetworkConditions(partition={"web"}), sleep=lambda s: None)
    with pytest.raises(Partitioned):
        net.delay("web")


def test_configure_moves_between_sweep_cells():
    net = SimNetwork(sleep=lambda s: None)
    net.configure(delay_ms=50.0, partition=["pos"])
    assert net.conditions.delay_ms == 50.0
    assert net.is_partitioned("pos")
    net.configure(partition=[])
    assert not net.is_partitioned("pos")
    assert net.conditions.delay_ms == 50.0   # untouched fields survive


# ─── Demand model ────────────────────────────────────────────────────────────

def test_base_price_reproduces_real_demand_exactly(gen):
    """The anchor: at the observed price the model returns the observed units."""
    assert gen.expected_units("A", "2013-01-05", 10.0) == pytest.approx(20.0)
    assert gen.expected_units("A", "2013-01-05", None) == pytest.approx(20.0)


def test_price_response_follows_the_estimated_elasticity(gen):
    # A doubling at elasticity -0.5 -> 2 ** -0.5
    assert gen.expected_units("A", "2013-01-05", 20.0) == pytest.approx(20 * 2 ** -0.5)
    # B is far more elastic, so the same relative rise costs it much more.
    assert gen.expected_units("B", "2013-01-05", 8.0) == pytest.approx(8 * 2 ** -2.5)


def test_cutting_price_raises_demand(gen):
    assert gen.expected_units("A", "2013-01-05", 5.0) > 20.0


def test_unknown_sku_or_date_generates_nothing(gen):
    assert gen.expected_units("ZZZ", "2013-01-05", 10.0) == 0.0
    assert gen.expected_units("A", "1999-01-01", 10.0) == 0.0
    assert gen.orders("A", "1999-01-01", "pos", 10.0, tick=0) == []


def test_orders_are_positive_integers_carrying_the_quoted_price(gen):
    orders = gen.orders("A", "2013-01-05", "pos", 10.0, tick=3)
    assert orders
    assert all(o.qty >= 1 for o in orders)
    assert all(o.unit_price == 10.0 for o in orders)
    assert all(o.channel == "pos" and o.sku == "A" for o in orders)
    assert orders[0].value == orders[0].qty * 10.0


def test_order_stream_is_identical_across_policies():
    """
    The control that makes E1 a controlled experiment.

    Two generators built independently - as two sweep cells would be - must emit
    byte-identical order streams for the same tick, or any difference between
    policies could just be different customers walking in.
    """
    frame = pd.DataFrame([{"date": pd.Timestamp("2013-01-01"), "product_id": "A",
                           "realized_demand": 50.0, "base_price": 10.0,
                           "elasticity_coefficient": -1.0,
                           "competitor_price": 11.0}])
    a = OrderGenerator(frame, seed=42, channel_weights={"pos": 1.0})
    b = OrderGenerator(frame, seed=42, channel_weights={"pos": 1.0})
    assert (a.orders("A", "2013-01-01", "pos", 10.0, tick=9)
            == b.orders("A", "2013-01-01", "pos", 10.0, tick=9))


def test_streams_differ_across_ticks_and_channels(gen):
    """Content-seeding must not collapse into one repeated day."""
    per_tick = [len(gen.orders("A", "2013-01-05", "pos", 10.0, tick=t))
                for t in range(20)]
    assert len(set(per_tick)) > 1

    pos = gen.orders("A", "2013-01-05", "pos", 10.0, tick=1)
    web = gen.orders("A", "2013-01-05", "web", 10.0, tick=1)
    assert [o.qty for o in pos] != [o.qty for o in web]


def test_channel_split_matches_the_configured_weights(gen):
    """Arrivals are stochastic; over many days the shares must hold."""
    units = {ch: 0 for ch in ("pos", "web", "marketplace")}
    for tick, date in enumerate(pd.date_range("2013-01-01", periods=40)):
        for ch in units:
            units[ch] += sum(o.qty for o in
                             gen.orders("A", date, ch, 10.0, tick=tick))
    total = sum(units.values())
    assert units["pos"] / total == pytest.approx(0.55, abs=0.06)
    assert units["web"] / total == pytest.approx(0.35, abs=0.06)
    assert units["marketplace"] / total == pytest.approx(0.10, abs=0.05)


def test_generated_volume_tracks_expected_units(gen):
    """The Poisson arrivals must reproduce the demand level, not just its shape."""
    total = sum(sum(o.qty for o in gen.orders("A", date, "pos", 10.0, tick=t))
                for t, date in enumerate(pd.date_range("2013-01-01", periods=40)))
    expected = 20.0 * 0.55 * 40
    assert total == pytest.approx(expected, rel=0.15)


def test_initial_stock_scales_with_mean_daily_demand(gen):
    assert gen.mean_daily_demand("A") == pytest.approx(20.0)
    assert gen.initial_stock("A", cover_days=7.0) == 140
    assert gen.initial_stock("A", cover_days=0.0) == 1     # never zero-stocked


def test_sku_order_is_stable(gen):
    assert gen.skus() == ["A", "B"]
    assert gen.skus(limit=1) == ["A"]


# ─── Config propagation ──────────────────────────────────────────────────────
#
# Regression tests for a bug that produced complete, well-formed, wrong results:
# a scenario set sync.policy in the supervisor, the inventory service ran in its
# own process with its own fresh SYSTEM_CONFIG, and three "different" policy
# cells were all silently executed by escrow_quota. Nothing raised.

def test_service_env_carries_the_settings_a_child_process_needs():
    from edge_system.config import SYSTEM_CONFIG, service_env

    original = dict(SYSTEM_CONFIG["sync"])
    try:
        SYSTEM_CONFIG["sync"]["policy"] = "strong_lock"
        SYSTEM_CONFIG["sync"]["quota_refill_multiple"] = 7.5
        env = service_env()
        assert env["FYP_SYNC_POLICY"] == "strong_lock"
        assert env["FYP_QUOTA_REFILL"] == "7.5"
    finally:
        SYSTEM_CONFIG["sync"].update(original)


def test_manifest_covers_every_config_section_a_child_process_reads():
    """
    The mechanical guard: scan the modules that run in child processes for
    `SYSTEM_CONFIG["section"]` reads, and fail if any section is absent from the
    manifest. A list someone has to remember to update is exactly what failed
    the first time - this fails on its own when a new read is introduced.
    """
    import re
    from pathlib import Path

    from edge_system.config import SERVICE_SECTIONS

    root = Path(__file__).resolve().parents[1]
    pattern = re.compile(r'SYSTEM_CONFIG\[[\'"](\w+)[\'"]\]')

    found = {}
    for sub in ("inventory", "edge", "control"):
        for path in (root / sub).glob("*.py"):
            if path.name.startswith("test_"):
                continue
            for section in pattern.findall(path.read_text(encoding="utf-8")):
                found.setdefault(section, set()).add(f"{sub}/{path.name}")

    uncovered = {s: sorted(f) for s, f in found.items() if s not in SERVICE_SECTIONS}
    assert not uncovered, (
        f"these config sections are read inside child service processes but are "
        f"not in config._SERVICE_KEYS, so mutating them has no effect on a "
        f"running system: {uncovered}"
    )


def test_env_overrides_round_trip_through_the_manifest():
    """Whatever `service_env` exports, `apply_env_overrides` must restore."""
    from edge_system.config import (SYSTEM_CONFIG, apply_env_overrides,
                                    service_env)

    original = {s: dict(SYSTEM_CONFIG[s]) for s in ("sync", "network", "redis", "paths")}
    try:
        SYSTEM_CONFIG["sync"]["policy"] = "eventual"
        SYSTEM_CONFIG["sync"]["reservation_ttl_s"] = 45.0
        SYSTEM_CONFIG["sync"]["quota_low_watermark"] = 0.5
        SYSTEM_CONFIG["network"]["delay_ms"] = 200.0
        SYSTEM_CONFIG["network"]["partition"] = ["web"]
        SYSTEM_CONFIG["paths"]["data_dir"] = "data/processed_m5"
        exported = service_env()

        # Simulate a child process starting from defaults.
        for section, values in original.items():
            SYSTEM_CONFIG[section].update(values)
        applied = apply_env_overrides(exported)

        assert SYSTEM_CONFIG["sync"]["policy"] == "eventual"
        assert SYSTEM_CONFIG["sync"]["reservation_ttl_s"] == 45.0
        assert SYSTEM_CONFIG["sync"]["quota_low_watermark"] == 0.5
        assert SYSTEM_CONFIG["network"]["delay_ms"] == 200.0
        assert SYSTEM_CONFIG["network"]["partition"] == ["web"]
        # E3's ablation lives or dies on this one.
        assert SYSTEM_CONFIG["paths"]["data_dir"] == "data/processed_m5"
        assert "paths.data_dir" in applied
    finally:
        for section, values in original.items():
            SYSTEM_CONFIG[section].update(values)


def test_env_overrides_preserve_types():
    """A float knob arriving as the string '45.0' would break every comparison."""
    from edge_system.config import SYSTEM_CONFIG, apply_env_overrides

    original = dict(SYSTEM_CONFIG["sync"]), dict(SYSTEM_CONFIG["network"])
    try:
        apply_env_overrides({"FYP_RESERVATION_TTL": "45.5",
                             "FYP_DELAY_MS": "200",
                             "FYP_PARTITION": ""})
        assert SYSTEM_CONFIG["sync"]["reservation_ttl_s"] == 45.5
        assert isinstance(SYSTEM_CONFIG["sync"]["reservation_ttl_s"], float)
        assert isinstance(SYSTEM_CONFIG["network"]["delay_ms"], float)
        # "" must mean "no channels", not [""] - a channel named "" would
        # partition nothing while looking like it partitioned something.
        assert SYSTEM_CONFIG["network"]["partition"] == []
    finally:
        SYSTEM_CONFIG["sync"].update(original[0])
        SYSTEM_CONFIG["network"].update(original[1])


def test_partition_env_round_trips_through_a_string():
    from edge_system.config import SYSTEM_CONFIG, service_env

    original = list(SYSTEM_CONFIG["network"]["partition"])
    try:
        SYSTEM_CONFIG["network"]["partition"] = ["web", "marketplace"]
        assert service_env()["FYP_PARTITION"] == "web,marketplace"
        SYSTEM_CONFIG["network"]["partition"] = []
        # Must survive the empty case: "".split(",") is [""], not [], and a
        # channel named "" would partition nothing while looking like something.
        assert service_env()["FYP_PARTITION"] == ""
        assert [c for c in "".split(",") if c] == []
    finally:
        SYSTEM_CONFIG["network"]["partition"] = original


def test_configured_ttl_reaches_the_reservation():
    """
    `sync.reservation_ttl_s` used to be dead: the policy never passed it and the
    escrow core's own default happened to be the same 300.0, so the setting
    looked alive and was not. The test uses a value nothing else defaults to.
    """
    import tempfile
    from pathlib import Path

    from edge_system.inventory.policies import make_policy
    from edge_system.inventory.store import SqlitePoolStore

    store = SqlitePoolStore(str(Path(tempfile.mkdtemp()) / "p.db"))
    try:
        policy = make_policy("escrow_quota", store, reservation_ttl_s=42.0,
                             refill_multiple=3.0, low_watermark=0.25)
        policy.stock_sku("A", 100)
        res = policy.reserve("pos", "A", 1)
        assert res is not None and res.ttl_s == 42.0
    finally:
        store.close()


def test_make_policy_drops_options_a_policy_does_not_take():
    """The caller hands over the whole sync config; picking a subset is the bug."""
    import tempfile
    from pathlib import Path

    from edge_system.inventory.policies import make_policy
    from edge_system.inventory.store import SqlitePoolStore

    store = SqlitePoolStore(str(Path(tempfile.mkdtemp()) / "p.db"))
    try:
        # strong_lock accepts neither refill_multiple nor low_watermark.
        policy = make_policy("strong_lock", store, refill_multiple=3.0,
                             low_watermark=0.25, reservation_ttl_s=99.0)
        assert policy.name == "strong_lock"
        assert policy.reservation_ttl_s == 99.0     # but it DOES take the ttl
    finally:
        store.close()


def test_every_scenario_is_well_formed():
    """Scenarios only name real config sections, and only real policies."""
    from edge_system.config import SYSTEM_CONFIG
    from edge_system.run_system import SCENARIOS

    valid = {"strong_lock", "eventual", "escrow_quota"}
    for name, spec in SCENARIOS.items():
        for section, values in spec.items():
            assert section in SYSTEM_CONFIG, f"{name} sets unknown section {section}"
            unknown = set(values) - set(SYSTEM_CONFIG[section])
            assert not unknown, f"{name}.{section} sets unknown keys {unknown}"
        assert spec.get("sync", {}).get("policy", "strong_lock") in valid
        assert spec.get("sim", {}).get("pricing", "node") in {"node", "static"}
