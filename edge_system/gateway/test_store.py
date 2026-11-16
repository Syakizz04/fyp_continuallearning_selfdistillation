"""The decision log is an audit trail: these pin that it cannot be rewritten,
double-decided, or decided anonymously."""

import pytest

from .guardrails import GlobalRule, SkuRule
from .store import DecisionError, Store


@pytest.fixture
def store():
    s = Store(":memory:")
    yield s
    s.close()


def rec(sku="FOODS_1_001", channel="pos", day="2013-03-01", raw=9.0, final=9.0, bound=()):
    return {"sku": sku, "channel": channel, "sim_date": day, "model_generation": 0,
            "tier": -0.1, "raw_price": raw, "base_price": 10.0, "final_price": final,
            "unit_cost": 7.0, "margin_pct": (final - 7) / final, "raw_margin_pct": None,
            "floor": None, "ceiling": None, "bound_by": list(bound), "conflict": False,
            "inventory_view": 40, "true_available": 50, "forecast_p50_total": None}


def test_generation_is_idempotent_per_day(store):
    assert store.add_recommendation(rec()) is not None
    assert store.add_recommendation(rec()) is None          # re-polling the same day
    assert store.counts()["pending"] == 1


def test_a_new_day_supersedes_yesterdays_pending(store):
    old = store.add_recommendation(rec(day="2013-03-01"))
    store.add_recommendation(rec(day="2013-03-02"))
    assert store.recommendation(old)["status"] == "superseded"
    assert store.counts() == {"pending": 1, "approved": 0, "rejected": 0, "superseded": 1}


def test_a_new_day_does_not_touch_other_channels_or_decided_rows(store):
    web = store.add_recommendation(rec(channel="web", day="2013-03-01"))
    pos = store.add_recommendation(rec(channel="pos", day="2013-03-01"))
    store.decide(pos, "approve", user="ops")
    store.add_recommendation(rec(channel="pos", day="2013-03-02"))
    assert store.recommendation(web)["status"] == "pending"
    assert store.recommendation(pos)["status"] == "approved"


def test_approve_records_who_when_and_the_applied_price(store):
    rid = store.add_recommendation(rec(raw=8.5, final=9.0, bound=["min_margin"]))
    out = store.decide(rid, "approve", user="  syakir ", note="ok")
    assert out["status"] == "approved"
    assert out["decided_by"] == "syakir"
    assert out["applied_price"] == 9.0               # the guarded price, not the raw one
    assert out["bound_by"] == ["min_margin"]


def test_reject_applies_no_price(store):
    rid = store.add_recommendation(rec())
    out = store.decide(rid, "reject", user="ops", note="too low")
    assert out["status"] == "rejected" and out["applied_price"] is None


def test_cannot_decide_twice(store):
    rid = store.add_recommendation(rec())
    store.decide(rid, "approve", user="a")
    with pytest.raises(DecisionError) as e:
        store.decide(rid, "reject", user="b")
    assert e.value.status == 409
    assert len(store.decisions()) == 1


def test_cannot_decide_a_superseded_recommendation(store):
    old = store.add_recommendation(rec(day="2013-03-01"))
    store.add_recommendation(rec(day="2013-03-02"))
    with pytest.raises(DecisionError) as e:
        store.decide(old, "approve", user="a")
    assert e.value.status == 409


@pytest.mark.parametrize("user", ["", "   "])
def test_a_decision_needs_a_name(store, user):
    rid = store.add_recommendation(rec())
    with pytest.raises(DecisionError) as e:
        store.decide(rid, "approve", user=user)
    assert e.value.status == 400


def test_unknown_recommendation_is_404(store):
    with pytest.raises(DecisionError) as e:
        store.decide(999, "approve", user="a")
    assert e.value.status == 404


def test_decisions_filter_by_user(store):
    a = store.add_recommendation(rec(sku="A"))
    b = store.add_recommendation(rec(sku="B"))
    store.decide(a, "approve", user="alice")
    store.decide(b, "reject", user="bob")
    assert [d["sku"] for d in store.decisions(user="alice")] == ["A"]


def test_rule_changes_are_logged_with_before_and_after(store):
    store.set_global_rule(GlobalRule(min_margin_pct=0.3, unit_cost_ratio=0.6), user="ops")
    store.set_sku_rule("A", SkuRule(min_price=2.0), user="ops")
    store.set_sku_rule("A", SkuRule(min_price=2.5), user="ops")
    store.delete_sku_rule("A", user="ops")
    log = store.rule_changes()
    assert [c["scope"] for c in log] == ["sku", "sku", "sku", "global"]
    assert log[0]["after"] is None and log[0]["before"]["min_price"] == 2.5
    assert log[1]["before"]["min_price"] == 2.0 and log[1]["after"]["min_price"] == 2.5
    assert log[3]["after"]["min_margin_pct"] == 0.3
    assert store.global_rule().unit_cost_ratio == 0.6


@pytest.mark.parametrize("bad", [
    SkuRule(min_price=5.0, max_price=4.0),
    SkuRule(min_price=-1.0),
    SkuRule(min_margin_pct=1.0),
])
def test_invalid_sku_rules_are_refused(store, bad):
    with pytest.raises(ValueError):
        store.set_sku_rule("A", bad, user="ops")
    assert store.sku_rules() == {}


def test_unit_cost_ratio_of_one_is_refused(store):
    with pytest.raises(ValueError):
        store.set_global_rule(GlobalRule(unit_cost_ratio=1.0), user="ops")
