"""Guardrails are the one place an operator's rule changes a price, so every
boundary is pinned here."""

import pytest

from .guardrails import EffectiveRule, GlobalRule, SkuRule, apply, gross_margin, resolve


def rule(cost=7.0, margin=None, lo=None, hi=None):
    return EffectiveRule(unit_cost=cost, min_margin_pct=margin, min_price=lo, max_price=hi)


def test_no_rules_leaves_the_agents_price_untouched():
    r = apply(9.4371, 10.0, rule())
    assert r.final_price == 9.4371          # not even rounded: no rule moved it
    assert r.bound_by == [] and not r.conflict


def test_margin_is_gross_margin_on_price_not_markup():
    # cost 7, min margin 30% -> floor = 7 / 0.7 = 10.00 (markup would say 9.10)
    r = apply(9.0, 10.0, rule(cost=7.0, margin=0.30))
    assert r.final_price == 10.0
    assert r.bound_by == ["min_margin"]
    assert gross_margin(r.final_price, 7.0) == pytest.approx(0.30)


def test_rounding_never_lands_below_a_margin_floor():
    # floor = 7 / 0.65 = 10.769230... -> must round UP to 10.77, not to 10.77-
    r = apply(9.0, 10.0, rule(cost=7.0, margin=0.35))
    assert r.final_price == 10.77
    assert r.margin_pct >= 0.35


def test_max_price_caps_and_rounds_down():
    r = apply(11.0, 10.0, rule(hi=10.499))
    assert r.final_price == 10.49 and r.bound_by == ["max_price"]


def test_min_price_floor():
    r = apply(9.0, 10.0, rule(lo=9.5))
    assert r.final_price == 9.5 and r.bound_by == ["min_price"]


def test_the_higher_floor_wins_and_is_named():
    r = apply(8.0, 10.0, rule(cost=7.0, margin=0.20, lo=9.5))  # margin floor 8.75
    assert r.final_price == 9.5 and r.bound_by == ["min_price"]
    r = apply(8.0, 10.0, rule(cost=7.0, margin=0.30, lo=9.5))  # margin floor 10.0
    assert r.final_price == 10.0 and r.bound_by == ["min_margin"]


def test_conflicting_rules_keep_the_margin_floor_and_say_so():
    # margin floor 10.0 above a max price of 9.0: never sell at a loss
    r = apply(9.5, 10.0, rule(cost=7.0, margin=0.30, hi=9.0))
    assert r.conflict
    assert r.final_price == 10.0
    assert set(r.bound_by) == {"min_margin", "max_price"}


def test_price_inside_the_band_is_not_reported_as_bound():
    r = apply(10.2, 10.0, rule(cost=7.0, margin=0.20, lo=9.0, hi=11.0))
    assert r.final_price == 10.2 and r.bound_by == []


@pytest.mark.parametrize("m", [1.0, 1.5, -0.1])
def test_impossible_margins_are_rejected(m):
    with pytest.raises(ValueError):
        apply(10.0, 10.0, rule(margin=m))


def test_resolve_defaults_cost_from_base_price():
    eff = resolve(10.0, GlobalRule(min_margin_pct=0.2, unit_cost_ratio=0.7))
    assert eff.unit_cost == pytest.approx(7.0)
    assert eff.min_margin_pct == 0.2
    assert eff.min_price is None and eff.max_price is None


def test_sku_override_beats_global_margin_and_cost():
    eff = resolve(10.0, GlobalRule(min_margin_pct=0.2),
                  SkuRule(min_margin_pct=0.05, unit_cost=4.0))
    assert eff.unit_cost == 4.0 and eff.min_margin_pct == 0.05


def test_tighter_limit_wins_between_global_and_sku():
    g = GlobalRule(min_margin_pct=None, max_discount_pct=0.05, max_markup_pct=0.05)
    # global band [9.5, 10.5]; the SKU tries to widen it to [9.0, 11.0]
    eff = resolve(10.0, g, SkuRule(min_price=9.0, max_price=11.0))
    assert eff.min_price == pytest.approx(9.5) and eff.max_price == pytest.approx(10.5)
    # and can narrow it
    eff = resolve(10.0, g, SkuRule(min_price=9.8, max_price=10.2))
    assert eff.min_price == pytest.approx(9.8) and eff.max_price == pytest.approx(10.2)


def test_full_ppo_range_against_default_rules():
    """The default rule (20% margin, cost 70% of base) puts the floor at 0.875 x
    base - below the PPO's -10% tier - so out of the box it never binds. An
    operator rule that does bind must be one they chose."""
    eff = resolve(10.0, GlobalRule())
    for tier in [-0.10, -0.06, 0.0, 0.06, 0.10]:
        raw = 10.0 * (1 + tier)
        assert apply(raw, 10.0, eff).bound_by == [], tier
