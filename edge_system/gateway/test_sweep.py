"""
Sweeps over EVERY gateway route, built from the app's own route table so a
route added later is covered without anyone remembering to add it here.

  - the auth matrix: anonymous / viewer / operator get exactly the access the
    middleware promises, on every route
  - no input - junk paging, odd SKUs, malformed bodies - ever produces a 500
  - any one upstream down produces a named 502 or a degraded answer, never a 500
  - paging: pages never overlap, and together they are exactly `total`

The per-feature behaviour lives in test_service.py; this file only asks
questions that must hold for all routes at once.
"""

import json
import random

import pytest
from fastapi.testclient import TestClient

from . import service as svc
from .auth import AuthStore
from .store import Store
from .test_service import CHANNELS, SKUS, FakeUpstream

PATH_VALUES = {"sku": SKUS[0], "rec_id": "1", "username": "nobody", "action": "pause"}
ODD_SKUS = ["%", "_", "a%25b", "FOODS_1_001'--", "ü-sku", "x" * 300, "..%2F..%2Fetc"]


def api_routes():
    """(method, path template) for every /api route the app defines."""
    out = []
    # svc.api, not svc.app: FastAPI wraps an included router in one opaque entry.
    for r in svc.api.routes:
        path = getattr(r, "path", "")
        if not path.startswith("/api/"):
            continue
        for m in sorted(getattr(r, "methods", None) or ()):
            if m != "HEAD":
                out.append((m, path))
    return sorted(set(out))


def fill(path, **over):
    vals = {**PATH_VALUES, **over}
    for k, v in vals.items():
        path = path.replace("{" + k + "}", v)
    assert "{" not in path, f"no sample value for a parameter in {path}"
    return path


def call(c, method, path, body=None, params=None):
    if method in ("GET", "DELETE"):
        return c.request(method, path, params=params)
    return c.request(method, path, json={} if body is None else body, params=params)


@pytest.fixture
def world(monkeypatch, tmp_path):
    """One gateway, three clients sharing it: anonymous, viewer, operator."""
    monkeypatch.setitem(svc.SYSTEM_CONFIG["paths"], "state_dir", str(tmp_path))
    monkeypatch.delenv("FYP_INTERNAL_TOKEN", raising=False)
    up = FakeUpstream()
    svc.GW["gw"] = svc.Gateway(Store(":memory:"), up)
    svc.AUTH["auth"] = AuthStore(":memory:")
    with TestClient(svc.app) as op, TestClient(svc.app) as viewer, \
            TestClient(svc.app) as anon:
        assert op.post("/api/auth/setup", json={"username": "ops", "password": "correct-horse"}).status_code == 200
        assert op.post("/api/auth/users", json={"username": "vic", "password": "viewer-pass",
                                                "role": "viewer"}).status_code == 200
        assert viewer.post("/api/auth/login", json={"username": "vic", "password": "viewer-pass"}).status_code == 200
        assert op.post("/api/recommendations/generate", json={}).status_code == 200
        yield {"up": up, "op": op, "viewer": viewer, "anon": anon}
    svc.GW["gw"] = None
    svc.AUTH["auth"] = None


def no_500(r, what):
    assert r.status_code < 500 or (r.status_code == 502 and "detail" in r.json()), (
        f"{what} -> {r.status_code}: {r.text[:300]}")


# ── the route table itself ──────────────────────────────────────────────────

def test_the_sweep_sees_the_routes_it_should():
    routes = api_routes()
    assert len(routes) >= 32
    for must in [("GET", "/api/products"), ("POST", "/api/recommendations/bulk"),
                 ("DELETE", "/api/auth/users/{username}"), ("POST", "/api/sim/{action}")]:
        assert must in routes


# ── auth matrix ─────────────────────────────────────────────────────────────

# Routes whose job is to change the session itself, so calling them changes
# who the client is for the rest of the sweep.
SESSION_ROUTES = {"/api/auth/logout", "/api/auth/login", "/api/auth/setup"}


@pytest.mark.parametrize("method,path", api_routes())
def test_anonymous_gets_401_except_open_routes(world, method, path):
    r = call(world["anon"], method, fill(path))
    no_500(r, f"anon {method} {path}")
    if path in svc._OPEN:
        assert r.status_code != 401
    elif path.startswith("/api/internal/"):
        pass                               # token-guarded; see test_internal_*
    else:
        assert r.status_code == 401, f"anon {method} {path} -> {r.status_code}"


@pytest.mark.parametrize("method,path", api_routes())
def test_viewer_reads_and_never_writes(world, method, path):
    if path in SESSION_ROUTES or path.startswith("/api/internal/"):
        return
    r = call(world["viewer"], method, fill(path))
    no_500(r, f"viewer {method} {path}")
    writes = method not in ("GET",) and path not in svc._READ_ONLY_POSTS
    if path in svc._OPEN:
        return
    if writes or path in svc._OPERATOR_READS:
        assert r.status_code == 403, f"viewer {method} {path} -> {r.status_code}"
    else:
        assert r.status_code not in (401, 403), f"viewer {method} {path} -> {r.status_code}"


@pytest.mark.parametrize("method,path", api_routes())
def test_operator_is_never_refused_and_never_500s(world, method, path):
    if path in SESSION_ROUTES:
        return
    # Removing the only other account is fine; removing yourself is a 409 by design.
    r = call(world["op"], method, fill(path))
    no_500(r, f"operator {method} {path}")
    assert r.status_code not in (401, 403), f"operator {method} {path} -> {r.status_code}: {r.text[:200]}"


def test_internal_routes_refuse_a_wrong_token(world, monkeypatch):
    monkeypatch.setenv("FYP_INTERNAL_TOKEN", "s3cret")
    for method, path in api_routes():
        if path.startswith("/api/internal/"):
            for c in (world["anon"], world["viewer"], world["op"]):
                r = c.request(method, path, json={}, headers={"X-Internal-Token": "wrong"})
                assert r.status_code == 403, (path, r.status_code)


def test_auth_cannot_be_dodged_by_spelling_the_path_differently(world):
    """The middleware matches on the raw path. Variants must not reach a
    guarded route without a session."""
    anon = world["anon"]
    for p in ["/api/auth/users/", "//api/auth/users", "/api//auth/users",
              "/api/auth/./users", "/api/products/", "/api/rules/../auth/users"]:
        r = anon.get(p, follow_redirects=True)
        assert r.status_code in (401, 404) or not r.headers.get("content-type", "").startswith(
            "application/json"), f"{p} -> {r.status_code} {r.text[:120]}"
        if r.status_code == 200 and r.headers.get("content-type", "").startswith("application/json"):
            pytest.fail(f"{p} served JSON to an anonymous client: {r.text[:120]}")


def test_dashboard_files_cannot_escape_dist(world):
    for p in ["/..%2F..%2Fedge_system%2Fconfig.py", "/../../edge_system/config.py",
              "/%2e%2e/%2e%2e/requirements.txt", "/C:/Windows/win.ini"]:
        r = world["anon"].get(p)
        assert "SYSTEM_CONFIG" not in r.text and "[fonts]" not in r.text, p


# ── junk input ──────────────────────────────────────────────────────────────

JUNK_PARAMS = [
    {"limit": -1}, {"limit": 0}, {"limit": 10**9}, {"limit": 10**20}, {"limit": "abc"},
    {"offset": -5}, {"offset": 10**12}, {"offset": 10**20},
    {"stream": -1}, {"stream": 10**7}, {"days": -3},
    {"status": "'; DROP TABLE recommendations;--"}, {"channel": "nope"},
    {"q": "%"}, {"q": "_"}, {"q": "\\"}, {"q": "ü" * 500}, {"bound": "maybe"},
    {"sim_date": "not-a-date"}, {"sim_date": "2013-02-30"}, {"user": "'"},
    {"kind": "<script>"},
]


@pytest.mark.parametrize("method,path", [r for r in api_routes() if r[0] == "GET"])
def test_get_routes_survive_junk_params(world, method, path):
    for params in JUNK_PARAMS:
        r = world["op"].get(fill(path), params=params)
        no_500(r, f"GET {path} {params}")


@pytest.mark.parametrize("method,path", [r for r in api_routes() if "{sku}" in r[1]])
def test_odd_skus_never_500(world, method, path):
    for sku in ODD_SKUS:
        r = call(world["op"], method, fill(path, sku=sku))
        no_500(r, f"{method} {path} sku={sku!r}")


BAD_BODIES = [None, [], "x", 0, {"ids": "1"}, {"ids": [10**30]}, {"action": "delete"},
              {"global_rule": {"min_pct": "x"}}, {"global_rule": {"min_pct": -999, "max_pct": 999}},
              {"multiplier": float("1e308")}, {"username": "", "password": ""},
              {"username": "a" * 5000, "password": "b" * 5000}, {"note": "n" * 100000}]


@pytest.mark.parametrize("method,path", [r for r in api_routes()
                                         if r[0] in ("POST", "PUT") and r[1] not in SESSION_ROUTES])
def test_write_routes_survive_bad_bodies(world, method, path):
    for body in BAD_BODIES:
        if body is None:
            r = world["op"].request(method, fill(path), content=b"{not json",
                                    headers={"content-type": "application/json"})
        else:
            r = world["op"].request(method, fill(path), content=json.dumps(body),
                                    headers={"content-type": "application/json"})
        no_500(r, f"{method} {path} body={str(body)[:60]}")


def test_login_survives_bad_bodies(world):
    for body in BAD_BODIES[1:]:
        r = world["anon"].post("/api/auth/login", json=body)
        no_500(r, f"login {str(body)[:60]}")
        assert r.status_code in (401, 422), (body, r.status_code)


# ── upstream failures ───────────────────────────────────────────────────────

SERVICES = [f"edge node {c}" for c in CHANNELS] + ["inventory service", "control plane"]


@pytest.mark.parametrize("down", SERVICES + ["everything"])
def test_any_upstream_down_is_never_a_500(world, down):
    up = world["up"]
    up.down = set(SERVICES) if down == "everything" else {down}
    for method, path in api_routes():
        if method != "GET":
            continue
        r = world["op"].get(fill(path))
        no_500(r, f"GET {path} with {down} down")


# ── paging ──────────────────────────────────────────────────────────────────

def test_pages_are_disjoint_and_cover_the_total(world):
    op, up = world["op"], world["up"]
    # Several days, some decided, so every status and channel filter has rows.
    for day in ("2013-03-02", "2013-03-03", "2013-03-04"):
        up.sim_date = day
        assert op.post("/api/recommendations/generate", json={"sim_date": day}).status_code == 200
    every = op.get("/api/recommendations", params={"limit": 5000}).json()
    for r in every[::3]:
        if r["status"] == "pending":
            op.post(f"/api/recommendations/{r['id']}/approve", json={})
    rng = random.Random(7)
    for _ in range(40):
        filt = {k: v for k, v in {
            "status": rng.choice([None, "pending", "approved", "rejected", "superseded"]),
            "channel": rng.choice([None] + CHANNELS),
            "q": rng.choice([None, SKUS[0], "FOODS", "3_0"]),
            "bound": rng.choice([False, True]),
        }.items() if v is not None}
        size = rng.choice([1, 2, 3, 7, 50])
        first = op.get("/api/recommendations/page", params={**filt, "limit": size}).json()
        total, ids, off = first["total"], [], 0
        while off < total + size:
            page = op.get("/api/recommendations/page", params={**filt, "limit": size, "offset": off}).json()
            assert page["total"] == total, filt
            assert len(page["items"]) <= size
            ids += [r["id"] for r in page["items"]]
            off += size
        assert len(ids) == len(set(ids)) == total, (filt, size, len(ids), total)
