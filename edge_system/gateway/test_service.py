import os
"""The gateway over HTTP, against a fake upstream: every acceptance criterion's
endpoint, with no edge nodes, inventory or control plane running."""

import pytest
from fastapi.testclient import TestClient

from . import service as svc
from .auth import AuthStore
from .store import Store
from .upstream import UpstreamError

SKUS = ["FOODS_1_001", "FOODS_3_090"]
CHANNELS = ["pos", "web", "marketplace"]


class FakeUpstream:
    """Stands in for every other service. `down` names the ones that fail."""

    def __init__(self):
        self.sim_date = "2013-03-01"
        self.down = set()
        self.price_calls = 0
        self.forecast_calls = 0

    def channels(self):
        return CHANNELS

    def _check(self, name):
        if name in self.down:
            raise UpstreamError(name, "unreachable (ConnectError)")

    def sim(self, path, params=None):
        # Not running unless a test says so - the batch path has no simulator.
        raise UpstreamError("simulator", "unreachable (ConnectError)")

    def sim_post(self, path, body=None):
        raise UpstreamError("simulator", "unreachable (ConnectError)")

    def node(self, name, path, params=None, timeout=None):
        self._check(f"edge node {name}")
        params = params or {}
        if path == "/drift":
            return {"status": {"checks": 1}, "events": [
                        {"tick": 7, "sim_date": self.sim_date, "kind": "forecasting",
                         "value": 1.9, "threshold": 1.6, "node": name}],
                    "stream": [{"tick": 7, "date": self.sim_date, "mase": None,
                                "profit_index": 1.01}]}
        if path == "/price":
            self.price_calls += 1
            # web believes stock is lower than pos does: different raw prices
            tier = {"pos": -0.10, "web": 0.04, "marketplace": 0.10}[name]
            return {"sku": params["sku"], "tier": tier, "price": 10.0 * (1 + tier),
                    "base_price": 10.0, "inventory_level_used": 30.0,
                    "model_generation": 2, "node": name}
        if path == "/forecast":
            self.forecast_calls += 1
            daily = [{"date": f"2013-03-{d:02d}", "p10": 1.0, "p50": 2.0, "p90": 4.0}
                     for d in range(2, 16)]
            return {"sku": params["sku"], "as_of": params["sim_date"], "horizon": 14,
                    "daily": daily, "history": [{"date": "2013-03-01", "actual": 3.0}],
                    "total_p50": 28.0, "forecast": 2.0, "model_generation": 2,
                    "forecaster_generation": 1, "forecaster_updated_at": 1.7e9,
                    "forecaster_updated_sim_date": "2013-02-20"}
        if path == "/health":
            return {"ok": True, "node": name, "model_generation": 2, "sdft_alpha": 0.99}
        if path == "/model":
            return {"node": name, "version": {"generation": 2}, "history": []}
        raise AssertionError(f"unexpected node path {path}")

    def inventory(self, path, params=None):
        self._check("inventory service")
        if path == "/skus":
            return SKUS
        if path.startswith("/stock/"):
            sku = path.rsplit("/", 1)[1]
            return {"sku": sku, "true_available": 50, "total": 60, "committed": 10,
                    "reserved": 0, "central_free": 20,
                    "node_views": {"pos": 15, "web": 10, "marketplace": 5},
                    "staleness": {"pos": 35, "web": 40, "marketplace": 45},
                    "stock_estimates": {"pos": 50, "web": 44, "marketplace": 38},
                    "estimate_errors": {"pos": 0, "web": -6, "marketplace": -12}}
        if path == "/health":
            return {"ok": True, "policy": "escrow_quota", "backend": "redis",
                    "effective": {"reservation_ttl_s": 300.0}}
        if path == "/metrics":
            return {"policy": "escrow_quota", "network": {
                "partition": ["marketplace"], "per_node_hops": {"pos": 4, "web": 3}}}
        if path == "/events":
            return [{"kind": "refill", "node": "web"}]
        if path == "/daily":
            return {SKUS[0]: [{"sim_date": "2013-03-01", "sold": 4, "refused": 1, "stock": 46}]}
        raise AssertionError(f"unexpected inventory path {path}")

    def control(self, path, params=None):
        self._check("control plane")
        if path == "/health":
            return {"ok": True}
        if path == "/models":
            return [{"ts": 1.7e9, "node": "pos", "kind": "forecasting", "generation": 1,
                     "sim_date": "2013-02-20", "reason": "drift", "ok": 1,
                     "duration_s": 40.0}]
        raise AssertionError(f"unexpected control path {path}")


@pytest.fixture
def tmp_state(monkeypatch):
    import tempfile
    from pathlib import Path
    d = Path(tempfile.mkdtemp(dir=os.environ.get("FYP_TEST_TMP")))
    monkeypatch.setitem(svc.SYSTEM_CONFIG["paths"], "state_dir", str(d))
    return d


@pytest.fixture
def up():
    return FakeUpstream()


@pytest.fixture
def client(up):
    svc.GW["gw"] = svc.Gateway(Store(":memory:"), up)
    svc.AUTH["auth"] = AuthStore(":memory:")
    with TestClient(svc.app) as c:
        r = c.post("/api/auth/setup", json={"username": "ops", "password": "correct-horse"})
        assert r.status_code == 200, r.text
        yield c
    svc.GW["gw"] = None
    svc.AUTH["auth"] = None


def gen(client):
    r = client.post("/api/recommendations/generate", json={})
    assert r.status_code == 200, r.text
    return r.json()


# ── status / clock ───────────────────────────────────────────────────────────

def test_status_reports_sim_day_display_clock_and_services(client, up):
    up.down.add("edge node marketplace")
    s = client.get("/api/status").json()
    assert s["sim_date"] == "2013-03-01"
    assert s["display_date"] is not None and s["offset_days"] > 0
    assert s["services"]["inventory"] is True
    assert s["services"]["edge-marketplace"] is False


def test_display_clock_advances_one_real_day_per_simulated_day(client, up):
    first = client.get("/api/status").json()["display_date"]
    up.sim_date = "2013-03-04"
    later = client.get("/api/status").json()["display_date"]
    import datetime as dt
    assert (dt.date.fromisoformat(later) - dt.date.fromisoformat(first)).days == 3


# ── criterion 1: forecast ────────────────────────────────────────────────────

def test_forecast_has_14_days_band_and_last_update(client, up):
    f = client.get("/api/products/FOODS_1_001/forecast").json()
    assert f["horizon"] == 14 and len(f["daily"]) == 14
    assert all(r["p10"] <= r["p50"] <= r["p90"] for r in f["daily"])
    assert f["forecaster_updated_sim_date"] == "2013-02-20"
    assert all(r["display_date"] for r in f["daily"])


def test_forecast_is_cached_per_day(client, up):
    client.get("/api/products/FOODS_1_001/forecast")
    client.get("/api/products/FOODS_1_001/forecast")
    assert up.forecast_calls == 1
    up.sim_date = "2013-03-02"
    client.get("/api/status")
    client.get("/api/products/FOODS_1_001/forecast")
    assert up.forecast_calls == 2


def test_forecast_unknown_node_is_404(client):
    assert client.get("/api/products/FOODS_1_001/forecast?node=nope").status_code == 404


# ── criterion 2: rules + recommendations ─────────────────────────────────────

def test_generate_prices_every_sku_on_every_channel(client, up):
    out = gen(client)
    assert out["created"] == len(SKUS) * len(CHANNELS)
    recs = client.get("/api/recommendations?status=pending").json()
    assert {r["channel"] for r in recs} == set(CHANNELS)
    assert gen(client)["created"] == 0              # idempotent per day


def test_rule_binds_and_says_which_rule(client):
    client.put("/api/rules/global", json={"min_margin_pct": 0.30, "unit_cost_ratio": 0.7,
                                          "user": "ops"})
    gen(client)
    pos = client.get("/api/recommendations?channel=pos").json()[0]
    assert pos["raw_price"] == pytest.approx(9.0)     # agent's -10%
    assert pos["final_price"] == 10.0                 # 7 / 0.7
    assert pos["bound_by"] == ["min_margin"]
    mkt = client.get("/api/recommendations?channel=marketplace").json()[0]
    assert mkt["bound_by"] == []                      # +10% is already above the floor


def test_sku_rule_and_delete(client):
    r = client.put("/api/rules/sku/FOODS_1_001",
                   json={"max_price": 10.5, "user": "ops"}).json()
    assert r["skus"][0]["max_price"] == 10.5
    gen(client)
    mkt = client.get("/api/recommendations?channel=marketplace&sku=FOODS_1_001").json()[0]
    assert mkt["final_price"] == 10.5 and mkt["bound_by"] == ["max_price"]
    assert client.delete("/api/rules/sku/FOODS_1_001?user=ops").status_code == 200
    assert client.delete("/api/rules/sku/FOODS_1_001?user=ops").status_code == 404


def test_invalid_rule_is_422(client):
    r = client.put("/api/rules/sku/A", json={"min_price": 5, "max_price": 4, "user": "ops"})
    assert r.status_code == 422
    r = client.put("/api/rules/global", json={"unit_cost_ratio": 1.2, "user": "ops"})
    assert r.status_code == 422


def test_preview_counts_without_saving(client):
    gen(client)
    p = client.post("/api/rules/preview", json={"global_rule": {
        "min_margin_pct": 0.30, "unit_cost_ratio": 0.7, "user": "ops"}}).json()
    assert p["pending"] == 6
    assert p["changed"] == 2                          # the two pos rows at -10%
    assert client.get("/api/rules").json()["global"]["min_margin_pct"] == 0.20


# ── criterion 3: approve / reject / review ───────────────────────────────────

def test_approve_reject_and_review(client):
    gen(client)
    a, b = [r["id"] for r in client.get("/api/recommendations").json()[:2]]
    r = client.post(f"/api/recommendations/{a}/approve", json={"user": "ops"})
    assert r.json()["status"] == "approved"
    client.post(f"/api/recommendations/{b}/reject", json={"user": "ops", "note": "no"})
    again = client.post(f"/api/recommendations/{a}/reject", json={"user": "x"})
    assert again.status_code == 409
    mine = client.get("/api/decisions?user=ops").json()
    assert {d["action"] for d in mine} == {"approve", "reject"}


def test_bulk_reports_partial_failure(client):
    gen(client)
    ids = [r["id"] for r in client.get("/api/recommendations").json()[:3]]
    client.post(f"/api/recommendations/{ids[0]}/approve", json={"user": "a"})
    out = client.post("/api/recommendations/bulk",
                      json={"ids": ids, "action": "approve", "user": "a"}).json()
    assert out["decided"] == ids[1:]
    assert [f["id"] for f in out["failed"]] == [ids[0]]


def test_recommendations_page_has_total_and_disjoint_pages(client):
    gen(client)
    every = client.get("/api/recommendations", params={"limit": 1000}).json()
    p1 = client.get("/api/recommendations/page", params={"limit": 2}).json()
    p2 = client.get("/api/recommendations/page", params={"limit": 2, "offset": 2}).json()
    assert p1["total"] == len(every) and p1["offset"] == 0
    assert [r["id"] for r in p1["items"] + p2["items"]] == [r["id"] for r in every[:4]]
    # Filters are applied before paging, and the total counts what matches.
    sku = every[0]["sku"]
    hit = client.get("/api/recommendations/page", params={"q": sku}).json()
    assert hit["total"] == sum(r["sku"] == sku for r in every)
    assert client.get("/api/recommendations/page", params={"q": "%"}).json()["total"] == 0


# ── criterion 4: stock + sync ────────────────────────────────────────────────

def test_inventory_per_channel_sync_status(client):
    inv = client.get("/api/inventory").json()
    assert inv["policy"] == "escrow_quota"
    by = {c["channel"]: c for c in inv["channels"]}
    assert by["marketplace"]["partitioned"] is True and by["pos"]["partitioned"] is False
    assert by["marketplace"]["mean_abs_estimate_error"] == 12
    assert by["pos"]["skus_with_stale_view"] == 0
    assert len(inv["skus"]) == len(SKUS)


def test_upstream_down_is_a_named_502(client, up):
    up.down.add("inventory service")
    r = client.get("/api/inventory")
    assert r.status_code == 502 and r.json()["service"] == "inventory service"


# ── criterion 5: model health ────────────────────────────────────────────────

def test_models_survive_one_node_offline(client, up):
    up.down.add("edge node web")
    m = client.get("/api/models").json()
    by = {n["node"]: n for n in m["nodes"]}
    assert by["web"]["online"] is False and "error" in by["web"]
    assert by["pos"]["drift"]["events"][0]["kind"] == "forecasting"
    assert m["versions"][0]["kind"] == "forecasting"


# ── criterion 6: history ─────────────────────────────────────────────────────

def test_history_merges_every_kind(client):
    client.put("/api/rules/global", json={"min_margin_pct": 0.25, "unit_cost_ratio": 0.7,
                                          "user": "ops"})
    gen(client)
    rid = client.get("/api/recommendations").json()[0]["id"]
    client.post(f"/api/recommendations/{rid}/approve", json={"user": "ops"})
    kinds = {h["kind"] for h in client.get("/api/history").json()}
    assert kinds == {"decision", "rule", "recommendations", "model"}


# ── routing ──────────────────────────────────────────────────────────────────

def test_unknown_api_path_is_404_not_the_dashboard(client):
    assert client.get("/api/nope").status_code == 404


# ─── The interactive simulator's day cycle ──────────────────────────────────

def _open(client, day="2013-03-01"):
    return client.post("/api/internal/open-day", json={"sim_date": day}).json()


def _live(client, day="2013-03-01", require=False):
    return client.post("/api/internal/live-prices",
                       json={"sim_date": day, "require_approval": require}).json()


def test_open_day_takes_over_the_clock_and_generates(client, up):
    out = _open(client, "2013-03-05")
    assert out["created"] == len(SKUS) * len(CHANNELS)
    up.sim_date = "2013-04-01"          # the nodes' monitors no longer set the day
    assert client.get("/api/status").json()["sim_date"] == "2013-03-05"


def test_live_prices_follow_decisions(client):
    _open(client)
    recs = {(r["sku"], r["channel"]): r for r in
            client.get("/api/recommendations?status=pending").json()}
    a, b = recs[(SKUS[0], "pos")], recs[(SKUS[0], "web")]
    client.post(f"/api/recommendations/{a['id']}/approve", json={"user": "ops"})
    client.post(f"/api/recommendations/{b['id']}/reject", json={"user": "ops"})

    live = {(p["sku"], p["channel"]): p for p in _live(client)["prices"]}
    assert live[(SKUS[0], "pos")]["price"] == a["final_price"]
    assert live[(SKUS[0], "pos")]["source"] == "approved"
    # rejected: the shelf price (base, on day one) stays on sale
    assert live[(SKUS[0], "web")]["price"] == b["base_price"]
    assert live[(SKUS[0], "web")]["source"] == "rejected"
    # undecided, approval not required: the recommendation goes live
    assert live[(SKUS[0], "marketplace")]["source"] == "auto"
    stored = client.get(f"/api/recommendations?sku={SKUS[0]}&channel=web").json()[0]
    assert stored["live_price"] == b["base_price"] and stored["live_source"] == "rejected"


def test_required_approval_keeps_the_shelf_price(client):
    _open(client)
    out = _live(client, require=True)
    assert out["counts"]["unreviewed"] == len(SKUS) * len(CHANNELS)
    assert all(p["price"] == 10.0 for p in out["prices"])      # base price


def test_shelf_price_carries_over_a_rejection(client):
    _open(client, "2013-03-01")
    r = client.get(f"/api/recommendations?sku={SKUS[0]}&channel=pos").json()[0]
    client.post(f"/api/recommendations/{r['id']}/approve", json={"user": "ops"})
    _live(client, "2013-03-01")
    _open(client, "2013-03-02")
    r2 = client.get(f"/api/recommendations?sku={SKUS[0]}&channel=pos&status=pending").json()[0]
    client.post(f"/api/recommendations/{r2['id']}/reject", json={"user": "ops"})
    live = {(p["sku"], p["channel"]): p for p in _live(client, "2013-03-02")["prices"]}
    assert live[(SKUS[0], "pos")]["price"] == r["final_price"]   # yesterday's approval


def test_reset_clears_the_run_but_keeps_rules(client, tmp_state):
    client.put("/api/rules/global", json={"min_margin_pct": 0.25, "unit_cost_ratio": 0.7,
                                          "user": "ops"})
    _open(client)
    r = client.get("/api/recommendations?status=pending").json()[0]
    client.post(f"/api/recommendations/{r['id']}/approve", json={"user": "ops"})
    assert client.post("/api/internal/reset").json()["ok"]
    assert client.get("/api/recommendations").json() == []
    assert client.get("/api/decisions").json() == []
    assert client.get("/api/rules").json()["global"]["min_margin_pct"] == 0.25
    assert client.get("/api/status").json()["sim_date"] is None
    assert list((tmp_state / "archive").glob("decisions.*.db"))   # signed decisions kept


def test_sim_routes_report_the_simulator_down(client):
    r = client.get("/api/sim")
    assert r.status_code == 502 and r.json()["service"] == "simulator"
    assert client.post("/api/sim/teleport").status_code == 404


def test_products_carry_microchart_series(client):
    _open(client)
    _live(client)
    rows = {p["sku"]: p for p in client.get("/api/products").json()}
    assert rows[SKUS[0]]["daily"][0]["sold"] == 4
    assert rows[SKUS[0]]["price_history"][0]["sim_date"] == "2013-03-01"
    assert rows[SKUS[1]]["daily"] == []


def test_products_carry_a_profile_where_the_data_has_one(client, monkeypatch):
    prof = {"price": 0.88, "mean_daily": 7.5, "cv": 1.07, "snap_lift": 1.45,
            "elasticity": -0.79, "tags": ["Value"]}
    monkeypatch.setattr(svc, "product_profiles", lambda: {SKUS[0]: prof})
    rows = {p["sku"]: p for p in client.get("/api/products").json()}
    assert rows[SKUS[0]]["profile"] == prof
    assert rows[SKUS[1]]["profile"] is None


# ── accounts ─────────────────────────────────────────────────────────────────

def test_signed_out_gets_401_and_setup_runs_once(client):
    client.post("/api/auth/logout")
    assert client.get("/api/status").status_code == 401
    assert client.get("/api/health").json()["ok"]           # launcher probe
    st = client.get("/api/auth/status").json()
    assert st == {"setup_required": False, "user": None, "roles": ["operator", "viewer"]}
    again = client.post("/api/auth/setup", json={"username": "eve", "password": "12345678"})
    assert again.status_code == 409


def test_login_wrong_password_and_logout(client):
    client.post("/api/auth/logout")
    bad = client.post("/api/auth/login", json={"username": "ops", "password": "nope-nope"})
    assert bad.status_code == 401 and "shelfsense_session" not in client.cookies
    ok = client.post("/api/auth/login", json={"username": "OPS", "password": "correct-horse"})
    assert ok.status_code == 200 and ok.json()["user"]["username"] == "ops"
    assert client.get("/api/status").status_code == 200
    client.post("/api/auth/logout")
    assert client.get("/api/status").status_code == 401


def test_decision_is_signed_by_the_session_not_the_body(client):
    gen(client)
    rid = client.get("/api/recommendations").json()[0]["id"]
    r = client.post(f"/api/recommendations/{rid}/approve", json={"user": "someone-else"})
    assert r.json()["decided_by"] == "ops"


def test_viewer_reads_but_cannot_change(client):
    gen(client)
    r = client.post("/api/auth/users", json={"username": "vic", "password": "viewer-pw", "role": "viewer"})
    assert r.status_code == 200 and r.json()["created_by"] == "ops"
    client.post("/api/auth/logout")
    client.post("/api/auth/login", json={"username": "vic", "password": "viewer-pw"})
    rid = client.get("/api/recommendations").json()[0]["id"]
    assert client.post(f"/api/recommendations/{rid}/approve", json={}).status_code == 403
    assert client.put("/api/rules/global", json={"unit_cost_ratio": 0.7}).status_code == 403
    assert client.post("/api/sim/play").status_code == 403
    assert client.post("/api/auth/users", json={"username": "x1", "password": "12345678"}).status_code == 403
    assert client.get("/api/auth/users").status_code == 403
    # Previewing a draft rule changes nothing, so a viewer may.
    assert client.post("/api/rules/preview", json={}).status_code == 200


def test_cannot_remove_self_or_last_operator(client):
    assert client.delete("/api/auth/users/ops").status_code == 409
    client.post("/api/auth/users", json={"username": "op2", "password": "op2-password", "role": "operator"})
    assert client.delete("/api/auth/users/op2").status_code == 200


def test_internal_routes_need_the_launch_token(client, monkeypatch):
    monkeypatch.setenv("FYP_INTERNAL_TOKEN", "s3cret")
    client.post("/api/auth/logout")
    body = {"sim_date": "2013-03-01"}
    assert client.post("/api/internal/open-day", json=body).status_code == 403
    ok = client.post("/api/internal/open-day", json=body, headers={"X-Internal-Token": "s3cret"})
    assert ok.status_code == 200
