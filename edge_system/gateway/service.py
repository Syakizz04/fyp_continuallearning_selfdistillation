"""
Decision-support gateway (FastAPI). The only service the operator dashboard
talks to.

    GET  /status                         simulated day, display clock, services up
    GET  /products                       catalogue with latest stock + pending count
    GET  /products/{sku}/forecast        14-day forecast with p10-p90 band
    GET  /rules                          global rule + per-SKU overrides
    PUT  /rules/global                   set the global rule
    PUT  /rules/sku/{sku}                set a per-SKU override
    DELETE /rules/sku/{sku}              drop it
    POST /rules/preview                  how many pending prices a draft would move
    GET  /recommendations                list, filterable by status / sku / channel
    POST /recommendations/generate       price every SKU x channel for a day
    POST /recommendations/{id}/approve
    POST /recommendations/{id}/reject
    POST /recommendations/bulk           decide several at once
    GET  /decisions                      the decision log, optionally one user's
    GET  /inventory                      stock + sync status per SKU and channel
    GET  /inventory/events               recent pool events
    GET  /models                         per-node health, drift monitor, versions
    GET  /history                        merged audit trail
    GET  /auth/status  POST /auth/login|logout|setup   sessions (see auth.py)
    GET|POST /auth/users  DELETE /auth/users/{name}     accounts (operators)

Run:  uvicorn edge_system.gateway.service:app --port 8020
"""

from __future__ import annotations

import datetime as dt
import hmac
import os
import threading
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Dict, List, Optional

from fastapi import APIRouter, FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, Response
from pydantic import BaseModel, Field

from ..config import APPLIED_OVERRIDES, PROJECT_ROOT, SYSTEM_CONFIG, ensure_dirs
from . import guardrails as gr
from .profiles import profiles as product_profiles
from .auth import COOKIE, ROLES, SESSION_TTL_S, AuthError, AuthStore, default_path
from .store import DecisionError, Store
from .upstream import Upstream, UpstreamError


# ─── Request bodies ──────────────────────────────────────────────────────────

class GlobalRuleIn(BaseModel):
    min_margin_pct: Optional[float] = Field(None, ge=0, lt=1)
    max_discount_pct: Optional[float] = Field(None, ge=0, le=1)
    max_markup_pct: Optional[float] = Field(None, ge=0, le=1)
    unit_cost_ratio: float = Field(..., gt=0, lt=1)
    user: Optional[str] = None      # ignored: the session signs the change


class SkuRuleIn(BaseModel):
    min_price: Optional[float] = Field(None, gt=0)
    max_price: Optional[float] = Field(None, gt=0)
    min_margin_pct: Optional[float] = Field(None, ge=0, lt=1)
    unit_cost: Optional[float] = Field(None, gt=0)
    user: Optional[str] = None      # ignored: the session signs the change


class PreviewIn(BaseModel):
    global_rule: Optional[GlobalRuleIn] = None
    sku: Optional[str] = None
    sku_rule: Optional[SkuRuleIn] = None


class DecisionIn(BaseModel):
    user: Optional[str] = None      # ignored: the session signs the decision
    note: Optional[str] = None


class BulkDecisionIn(BaseModel):
    ids: List[int]
    action: str
    user: Optional[str] = None
    note: Optional[str] = None


class LoginIn(BaseModel):
    username: str
    password: str


class NewUserIn(BaseModel):
    username: str
    password: str
    role: str = "viewer"


class GenerateIn(BaseModel):
    sim_date: Optional[str] = None


# ─── The gateway ─────────────────────────────────────────────────────────────

class Gateway:
    """State and behaviour, separate from the HTTP layer so tests can drive it
    with a fake `Upstream` and no running services."""

    def __init__(self, store: Store, upstream) -> None:
        self.store = store
        self.up = upstream
        self.sim_date: Optional[str] = None
        # Display clock: the first simulated day seen is shown as today's date,
        # and each simulated day after it as one real day later. Internally
        # everything stays in M5 dates - the offset is presentation only.
        self._anchor_sim: Optional[dt.date] = None
        self._anchor_real: Optional[dt.date] = None
        self.last_generated: Optional[str] = None
        self.last_generation: Optional[Dict] = None
        self._forecast_cache: Dict[tuple, Dict] = {}
        self._gen_lock = threading.Lock()
        self._stop = threading.Event()
        self._poller: Optional[threading.Thread] = None
        # Set once the interactive simulator opens a day. From then on the
        # simulator owns the clock and the poller stops inferring it.
        self.driven = False
        # The price each (sku, channel) is currently on sale at - what a
        # rejected or unreviewed recommendation leaves in place.
        self.shelf: Dict[tuple, float] = {}

    # ── simulated time ──────────────────────────────────────────────────────

    def observe_sim_date(self) -> Optional[str]:
        """The latest simulated day any edge node has been checked on.

        Read off the nodes' own drift monitors - the driver posts every day's
        outcome to them - so the gateway needs no hook into the simulator."""
        if self.driven:
            return self.sim_date
        latest = None
        for ch in self.up.channels():
            try:
                stream = self.up.node(ch, "/drift", {"limit": 1}, timeout=2.0).get("stream") or []
            except UpstreamError:
                continue
            if stream and stream[-1].get("date"):
                latest = max(latest or "", stream[-1]["date"])
        if latest:
            self.sim_date = latest
            if self._anchor_sim is None:
                self._anchor_sim = dt.date.fromisoformat(latest)
                self._anchor_real = dt.date.today()
        return self.sim_date

    def display_date(self, sim_date: Optional[str]) -> Optional[str]:
        if not sim_date or self._anchor_sim is None:
            return None
        delta = dt.date.fromisoformat(sim_date) - self._anchor_sim
        return (self._anchor_real + delta).isoformat()

    def offset_days(self) -> Optional[int]:
        if self._anchor_sim is None:
            return None
        return (self._anchor_real - self._anchor_sim).days

    def require_sim_date(self, sim_date: Optional[str] = None) -> str:
        d = sim_date or self.sim_date or self.observe_sim_date()
        if not d:
            raise HTTPException(409, "no simulated day yet: start the simulator "
                                     "(python -m edge_system.run_system --scenario demo)")
        return d

    # ── rules ───────────────────────────────────────────────────────────────

    def effective(self, sku: str, base_price: float, glob=None, sku_rules=None) -> gr.EffectiveRule:
        glob = glob or self.store.global_rule()
        sku_rules = self.store.sku_rules() if sku_rules is None else sku_rules
        return gr.resolve(base_price, glob, sku_rules.get(sku))

    # ── recommendations ─────────────────────────────────────────────────────

    def generate(self, sim_date: str) -> Dict:
        """Price every SKU on every channel for one day, through the guardrails.

        Each channel prices against ITS OWN stock belief, so the same SKU can get
        different recommendations per channel - the sync staleness the system
        studies, made visible to the operator. Idempotent per day."""
        with self._gen_lock:
            started = time.time()
            skus = self.up.inventory("/skus")
            glob, sku_rules = self.store.global_rule(), self.store.sku_rules()
            created, errors = 0, []
            for sku in skus:
                try:
                    stock = self.up.inventory(f"/stock/{sku}")
                except UpstreamError as exc:
                    errors.append(str(exc)); continue
                fc = self._forecast_cache.get((sku, self.primary_node(), sim_date))
                for ch in self.up.channels():
                    try:
                        p = self.up.node(ch, "/price", {"sku": sku, "sim_date": sim_date})
                    except UpstreamError as exc:
                        errors.append(str(exc)); continue
                    res = gr.apply(float(p["price"]), float(p["base_price"]),
                                   self.effective(sku, float(p["base_price"]), glob, sku_rules))
                    rid = self.store.add_recommendation({
                        "sku": sku, "channel": ch, "sim_date": sim_date,
                        "model_generation": p.get("model_generation"),
                        "tier": p.get("tier"),
                        **{k: v for k, v in res.as_dict().items()
                           if k in ("raw_price", "final_price", "base_price", "unit_cost",
                                    "floor", "ceiling", "bound_by", "conflict",
                                    "margin_pct", "raw_margin_pct")},
                        "inventory_view": p.get("inventory_level_used"),
                        "true_available": stock.get("true_available"),
                        "forecast_p50_total": fc.get("total_p50") if fc else None,
                    })
                    created += rid is not None
            self.last_generated = sim_date
            self.last_generation = {"sim_date": sim_date, "created": created,
                                    "skus": len(skus), "errors": errors[:10],
                                    "n_errors": len(errors),
                                    "seconds": round(time.time() - started, 2)}
            return self.last_generation

    # ── the interactive simulator's day cycle ──────────────────────────────

    def open_day(self, sim_date: str) -> Dict:
        """A new trading day starts: recommendations first, trading after the
        operator has had the chance to decide on them."""
        self.driven = True
        self.sim_date = sim_date
        if self._anchor_sim is None:
            self._anchor_sim = dt.date.fromisoformat(sim_date)
            self._anchor_real = dt.date.today()
        return self.generate(sim_date)

    def live_prices(self, sim_date: str, *, require_approval: bool) -> Dict:
        """The price that actually goes on sale for each (sku, channel) today.

            approved            -> the recommendation's price
            rejected            -> the shelf price stays
            not decided         -> the recommendation's price, or - when
                                   approval is required - the shelf price stays

        The shelf starts at base price. Recorded on each recommendation, so the
        audit trail says what was sold at as well as what was proposed."""
        out, counts = [], {"approved": 0, "rejected": 0, "auto": 0, "unreviewed": 0}
        for r in self.store.recommendations(sim_date=sim_date, limit=10000):
            key = (r["sku"], r["channel"])
            shelf = self.shelf.get(key, r["base_price"])
            if r["status"] == "approved":
                price, source = r["final_price"], "approved"
            elif r["status"] == "rejected":
                price, source = shelf, "rejected"
            elif require_approval:
                price, source = shelf, "unreviewed"
            else:
                price, source = r["final_price"], "auto"
            counts[source] += 1
            self.shelf[key] = price
            self.store.set_live(r["id"], price, source)
            out.append({"sku": r["sku"], "channel": r["channel"], "price": price,
                        "tier": r["tier"] or 0.0, "generation": r["model_generation"] or 0,
                        "source": source})
        return {"sim_date": sim_date, "prices": out, "counts": counts}

    def reset_run(self) -> None:
        archive = Path(SYSTEM_CONFIG["paths"]["state_dir"]) / "archive" /             f"decisions.{time.strftime('%Y%m%d-%H%M%S')}.db"
        with self._gen_lock:
            self.store.clear_run(archive_to=archive)
            self.sim_date = self.last_generated = self.last_generation = None
            self._anchor_sim = self._anchor_real = None
            self._forecast_cache.clear()
            self.shelf.clear()

    def primary_node(self) -> str:
        return SYSTEM_CONFIG["gateway"]["primary_node"]

    # ── forecasts ───────────────────────────────────────────────────────────

    def forecast(self, sku: str, node: Optional[str] = None,
                 sim_date: Optional[str] = None) -> Dict:
        node = node or self.primary_node()
        if node not in self.up.channels():
            raise HTTPException(404, f"unknown node {node!r}")
        sim_date = self.require_sim_date(sim_date)
        key = (sku, node, sim_date)
        if key not in self._forecast_cache:
            # A TFT pass is seconds; the simulated day changes far less often
            # than the page refreshes, so cache per (sku, node, day).
            out = self.up.node(node, "/forecast", {"sku": sku, "sim_date": sim_date},
                               timeout=60.0)
            if len(self._forecast_cache) > 500:
                self._forecast_cache.clear()
            self._forecast_cache[key] = out
        out = dict(self._forecast_cache[key])
        out["display_as_of"] = self.display_date(out.get("as_of"))
        out["forecaster_updated_display_date"] = self.display_date(
            out.get("forecaster_updated_sim_date"))
        for row in out.get("daily", []) + out.get("history", []):
            row["display_date"] = self.display_date(row["date"])
        return out

    # ── background poll ─────────────────────────────────────────────────────

    def start(self) -> None:
        self._poller = threading.Thread(target=self._poll, name="gateway-poll", daemon=True)
        self._poller.start()

    def stop(self) -> None:
        self._stop.set()

    def _poll(self) -> None:
        while not self._stop.is_set():
            try:
                d = self.observe_sim_date()
                if d and d != self.last_generated and not self.driven:
                    self.generate(d)
            except Exception:   # noqa: BLE001 - the poller must outlive a bad tick
                pass
            self._stop.wait(float(SYSTEM_CONFIG["gateway"]["poll_interval_s"]))


GW: Dict[str, Optional[Gateway]] = {"gw": None}
AUTH: Dict[str, Optional[AuthStore]] = {"auth": None}


def auth() -> AuthStore:
    a = AUTH["auth"]
    if a is None:
        raise HTTPException(503, "gateway not started")
    return a


def gw() -> Gateway:
    g = GW["gw"]
    if g is None:
        raise HTTPException(503, "gateway not started")
    return g


@asynccontextmanager
async def lifespan(app: FastAPI):
    ensure_dirs()
    if AUTH["auth"] is None:        # tests install their own, like the gateway
        AUTH["auth"] = AuthStore(default_path())
    if GW["gw"] is None:            # tests install their own before startup
        GW["gw"] = Gateway(
            Store(SYSTEM_CONFIG["paths"]["decisions_db"],
                  default_unit_cost_ratio=SYSTEM_CONFIG["gateway"]["unit_cost_ratio"]),
            Upstream())
        GW["gw"].start()
    yield
    if GW["gw"] is not None:
        GW["gw"].stop()


app = FastAPI(title="FYP decision-support gateway", version="1.0", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["http://localhost:5173",
                                                  "http://127.0.0.1:5173"],
                   allow_methods=["*"], allow_headers=["*"])
# Every API route lives under /api, so the dashboard's own client-side routes
# (/products, /models, ...) can never collide with an endpoint on refresh.
api = APIRouter(prefix="/api")


# ─── Who is asking ───────────────────────────────────────────────────────────
# Every /api route needs a signed-in session, except signing in itself. Reads
# are open to any role; anything that changes state needs an operator.
# /api/internal/* is the simulator calling in, authenticated by a token the
# launcher hands both processes rather than by a user session.

# /api/health is the launcher's readiness probe.
_OPEN = {"/api/health", "/api/auth/status", "/api/auth/login", "/api/auth/setup",
         "/api/auth/logout"}
_READ_ONLY_POSTS = {"/api/rules/preview"}
# Reads that are still an operator's: who has an account is admin information.
_OPERATOR_READS = {"/api/auth/users"}


def _internal_ok(request: Request) -> bool:
    token = os.environ.get("FYP_INTERNAL_TOKEN")
    if token:
        return hmac.compare_digest(request.headers.get("x-internal-token", ""), token)
    # No token configured (a gateway started on its own): local callers only.
    return request.client is not None and request.client.host in {
        "127.0.0.1", "::1", "localhost", "testclient"}


@app.middleware("http")
async def _authenticate(request: Request, call_next):
    path = request.url.path
    if not path.startswith("/api/") or path in _OPEN or request.method == "OPTIONS":
        return await call_next(request)
    if path.startswith("/api/internal/"):
        if not _internal_ok(request):
            return JSONResponse(status_code=403, content={"detail": "internal route"})
        return await call_next(request)
    a = AUTH["auth"]
    user = a.session_user(request.cookies.get(COOKIE)) if a is not None else None
    if user is None:
        return JSONResponse(status_code=401, content={"detail": "sign in first"})
    writes = request.method not in ("GET", "HEAD") and path not in _READ_ONLY_POSTS
    if (writes or path in _OPERATOR_READS) and user["role"] != "operator":
        return JSONResponse(status_code=403,
                            content={"detail": "viewers can look but not change anything"})
    request.state.user = user
    return await call_next(request)


def actor(request: Request) -> str:
    """The signed-in user's name - the only identity a change is recorded under."""
    return request.state.user["username"]


@app.exception_handler(AuthError)
async def _auth_error(_: Request, exc: AuthError):
    return JSONResponse(status_code=exc.status, content={"detail": str(exc)})


@app.exception_handler(UpstreamError)
async def _upstream(_: Request, exc: UpstreamError):
    return JSONResponse(status_code=502, content={"detail": str(exc), "service": exc.service})


@app.exception_handler(DecisionError)
async def _decision(_: Request, exc: DecisionError):
    return JSONResponse(status_code=exc.status, content={"detail": str(exc)})


@app.exception_handler(ValueError)
async def _value(_: Request, exc: ValueError):
    return JSONResponse(status_code=422, content={"detail": str(exc)})


# ─── Status ──────────────────────────────────────────────────────────────────

@api.get("/health")
def health() -> Dict:
    return {"ok": GW["gw"] is not None, "applied_overrides": APPLIED_OVERRIDES,
            "unit_cost_ratio_default": SYSTEM_CONFIG["gateway"]["unit_cost_ratio"]}


@api.get("/status")
def status() -> Dict:
    g = gw()
    services = {}
    for name, call in (("inventory", lambda: g.up.inventory("/health")),
                       ("control", lambda: g.up.control("/health"))):
        try:
            services[name] = bool(call().get("ok"))
        except UpstreamError:
            services[name] = False
    for ch in g.up.channels():
        try:
            services[f"edge-{ch}"] = bool(g.up.node(ch, "/health", timeout=2.0).get("ok"))
        except UpstreamError:
            services[f"edge-{ch}"] = False
    sim = None
    try:
        sim = g.up.sim("/state", {"brief": True})
        services["simulator"] = True
    except UpstreamError:
        if g.driven:
            services["simulator"] = False
    d = g.observe_sim_date()
    return {"sim_date": d, "display_date": g.display_date(d),
            "offset_days": g.offset_days(), "services": services, "sim": sim,
            "last_generation": g.last_generation,
            "counts": g.store.counts(), "channels": g.up.channels()}


# ─── Products and forecasts ──────────────────────────────────────────────────

@api.get("/products")
def products() -> List[Dict]:
    g = gw()
    latest = {}
    for r in g.store.recommendations(limit=5000):
        latest.setdefault(r["sku"], r)
    pending: Dict[str, int] = {}
    for r in g.store.recommendations(status="pending", limit=5000):
        pending[r["sku"]] = pending.get(r["sku"], 0) + 1
    try:
        daily = g.up.inventory("/daily", {"days": 28})
    except UpstreamError:
        daily = {}
    # The in-store till's live price per day: the price a shopper actually paid.
    price_hist: Dict[str, List[Dict]] = {}
    for r in g.store.recommendations(channel=g.primary_node(), limit=5000):
        if r.get("live_price") is not None:
            price_hist.setdefault(r["sku"], []).append(
                {"sim_date": r["sim_date"], "price": r["live_price"]})
    prof = product_profiles()
    out = []
    for sku in g.up.inventory("/skus"):
        r = latest.get(sku, {})
        out.append({"sku": sku, "dept": "_".join(sku.split("_")[:2]),
                    "daily": daily.get(sku, []),
                    "price_history": sorted(price_hist.get(sku, []),
                                            key=lambda x: x["sim_date"])[-28:],
                    "base_price": r.get("base_price"),
                    "true_available": r.get("true_available"),
                    "latest_sim_date": r.get("sim_date"),
                    "pending": pending.get(sku, 0),
                    "has_rule": sku in g.store.sku_rules(),
                    "profile": prof.get(sku)})
    return out


@api.get("/products/{sku}/forecast")
def product_forecast(sku: str, node: Optional[str] = None,
                     sim_date: Optional[str] = None) -> Dict:
    return gw().forecast(sku, node, sim_date)


# ─── Accounts ────────────────────────────────────────────────────────────────

def _set_cookie(resp: Response, token: str) -> None:
    resp.set_cookie(COOKIE, token, max_age=SESSION_TTL_S, httponly=True,
                    samesite="strict", path="/")


@api.get("/auth/status")
def auth_status(request: Request) -> Dict:
    a = auth()
    return {"setup_required": a.count() == 0,
            "user": a.session_user(request.cookies.get(COOKIE)), "roles": list(ROLES)}


@api.post("/auth/setup")
def auth_setup(body: LoginIn, response: Response) -> Dict:
    user = auth().setup_first(body.username, body.password)
    _set_cookie(response, auth().new_session(user["username"]))
    return {"user": user}


@api.post("/auth/login")
def auth_login(body: LoginIn, response: Response) -> Dict:
    token, user = auth().login(body.username, body.password)
    _set_cookie(response, token)
    return {"user": user}


@api.post("/auth/logout")
def auth_logout(request: Request, response: Response) -> Dict:
    auth().logout(request.cookies.get(COOKIE))
    response.delete_cookie(COOKIE, path="/")
    return {"ok": True}


@api.get("/auth/users")
def auth_users() -> List[Dict]:
    return auth().users()


@api.post("/auth/users")
def auth_add_user(body: NewUserIn, request: Request) -> Dict:
    return auth().add_user(body.username, body.password, body.role, created_by=actor(request))


@api.delete("/auth/users/{username}")
def auth_remove_user(username: str, request: Request) -> Dict:
    if username.lower() == actor(request).lower():
        raise HTTPException(409, "you cannot remove your own account")
    auth().remove_user(username)
    return {"ok": True}


# ─── Rules ───────────────────────────────────────────────────────────────────

@api.get("/rules")
def rules() -> Dict:
    g = gw()
    return {"global": g.store.global_rule_row(), "skus": g.store.sku_rule_rows(),
            "unit_cost_is_assumed": True}


@api.put("/rules/global")
def put_global(body: GlobalRuleIn, request: Request) -> Dict:
    rule = gr.GlobalRule(**body.model_dump(exclude={"user"}))
    gw().store.set_global_rule(rule, user=actor(request))
    return rules()


@api.put("/rules/sku/{sku}")
def put_sku(sku: str, body: SkuRuleIn, request: Request) -> Dict:
    gw().store.set_sku_rule(sku, gr.SkuRule(**body.model_dump(exclude={"user"})),
                            user=actor(request))
    return rules()


@api.delete("/rules/sku/{sku}")
def delete_sku(sku: str, request: Request, user: Optional[str] = None) -> Dict:
    if not gw().store.delete_sku_rule(sku, user=actor(request)):
        raise HTTPException(404, f"no rule for {sku}")
    return rules()


@api.post("/rules/preview")
def preview(body: PreviewIn) -> Dict:
    """Re-run today's pending recommendations through a DRAFT rule set, without
    saving it, so the operator sees what a change would do before making it."""
    g = gw()
    glob = (gr.GlobalRule(**body.global_rule.model_dump(exclude={"user"}))
            if body.global_rule else g.store.global_rule())
    sku_rules = g.store.sku_rules()
    if body.sku and body.sku_rule:
        sku_rules[body.sku] = gr.SkuRule(**body.sku_rule.model_dump(exclude={"user"}))
    changed, rows = 0, []
    pending = g.store.recommendations(status="pending", limit=5000)
    for r in pending:
        res = gr.apply(r["raw_price"], r["base_price"],
                       gr.resolve(r["base_price"], glob, sku_rules.get(r["sku"])))
        if abs(res.final_price - r["final_price"]) > 1e-9:
            changed += 1
            if len(rows) < 50:
                rows.append({"id": r["id"], "sku": r["sku"], "channel": r["channel"],
                             "current": r["final_price"], "draft": res.final_price,
                             "bound_by": res.bound_by})
    return {"pending": len(pending), "changed": changed, "examples": rows}


# ─── Recommendations and decisions ───────────────────────────────────────────

#: Upper bound on any list a route returns. SQLite reads LIMIT -1 as "no limit"
#: and raises OverflowError past 2**63, so every limit/offset goes through these.
MAX_ROWS = 5000


def _limit(n: int, cap: int = MAX_ROWS) -> int:
    return min(max(n, 1), cap)


def _offset(n: int) -> int:
    return min(max(n, 0), 2**62)


def _with_display(rows: List[Dict]) -> List[Dict]:
    g = gw()
    for r in rows:
        r["display_date"] = g.display_date(r.get("sim_date"))
    return rows


@api.get("/recommendations")
def list_recommendations(status: Optional[str] = None, sku: Optional[str] = None,
                         channel: Optional[str] = None, sim_date: Optional[str] = None,
                         limit: int = 200) -> List[Dict]:
    return _with_display(gw().store.recommendations(
        status=status, sku=sku, channel=channel, sim_date=sim_date, limit=_limit(limit)))


@api.get("/recommendations/page")
def page_recommendations(status: Optional[str] = None, channel: Optional[str] = None,
                         q: Optional[str] = None, bound: bool = False,
                         limit: int = 50, offset: int = 0) -> Dict:
    """One page plus the total, so the table can page server-side: a long run
    accumulates thousands of rows (SKUs x channels x days)."""
    store = gw().store
    limit, offset = _limit(limit, 500), _offset(offset)
    filt = dict(status=status, channel=channel, q=q, bound=bound)
    return {"items": _with_display(store.recommendations(**filt, limit=limit, offset=offset)),
            "total": store.count_recommendations(**filt),
            "limit": limit, "offset": offset}


@api.post("/recommendations/generate")
def generate(body: GenerateIn) -> Dict:
    g = gw()
    return g.generate(g.require_sim_date(body.sim_date))


@api.post("/recommendations/{rec_id}/approve")
def approve(rec_id: int, body: DecisionIn, request: Request) -> Dict:
    return _with_display([gw().store.decide(rec_id, "approve", user=actor(request),
                                            note=body.note)])[0]


@api.post("/recommendations/{rec_id}/reject")
def reject(rec_id: int, body: DecisionIn, request: Request) -> Dict:
    return _with_display([gw().store.decide(rec_id, "reject", user=actor(request),
                                            note=body.note)])[0]


@api.post("/recommendations/bulk")
def bulk(body: BulkDecisionIn, request: Request) -> Dict:
    """Decide each independently: one already-decided row must not block the
    rest, and the response says which ones did not go through and why."""
    done, failed = [], []
    for rid in body.ids:
        try:
            gw().store.decide(rid, body.action, user=actor(request), note=body.note)
            done.append(rid)
        except DecisionError as exc:
            failed.append({"id": rid, "status": exc.status, "detail": str(exc)})
    return {"decided": done, "failed": failed}


@api.get("/decisions")
def decisions(user: Optional[str] = None, limit: int = 200) -> List[Dict]:
    return _with_display(gw().store.decisions(user=user, limit=_limit(limit)))


# ─── Inventory and sync ──────────────────────────────────────────────────────

@api.get("/inventory")
def inventory() -> Dict:
    g = gw()
    health = g.up.inventory("/health")
    metrics = g.up.inventory("/metrics")
    rows = [g.up.inventory(f"/stock/{sku}") for sku in g.up.inventory("/skus")]
    partitioned = set((metrics.get("network") or {}).get("partition") or [])
    per_node_hops = (metrics.get("network") or {}).get("per_node_hops") or {}
    channels = []
    for ch in g.up.channels():
        errs = [abs(r["estimate_errors"].get(ch, 0)) for r in rows]
        channels.append({
            "channel": ch,
            "partitioned": ch in partitioned,
            "quota_units": sum(r["node_views"].get(ch, 0) for r in rows),
            "mean_abs_estimate_error": (sum(errs) / len(errs)) if errs else None,
            "max_abs_estimate_error": max(errs) if errs else None,
            "skus_with_stale_view": sum(e > 0 for e in errs),
            "sync_hops": per_node_hops.get(ch, 0),
        })
    return {"sim_date": g.sim_date, "display_date": g.display_date(g.sim_date),
            "policy": health.get("policy"), "backend": health.get("backend"),
            "effective": health.get("effective"), "metrics": metrics,
            "channels": channels, "skus": rows}


@api.get("/inventory/events")
def inventory_events(limit: int = 100, kind: Optional[str] = None) -> List[Dict]:
    params = {"limit": _limit(limit)}
    if kind:
        params["kind"] = kind
    return gw().up.inventory("/events", params)


# ─── Model health ────────────────────────────────────────────────────────────

@api.get("/models")
def models(stream: int = 120) -> Dict:
    g = gw()
    stream = _limit(stream)
    nodes = []
    for ch in g.up.channels():
        entry: Dict = {"node": ch}
        try:
            entry["health"] = g.up.node(ch, "/health", timeout=3.0)
            entry["model"] = g.up.node(ch, "/model", timeout=3.0)
            entry["drift"] = g.up.node(ch, "/drift", {"limit": stream}, timeout=3.0)
            for row in entry["drift"].get("stream", []):
                row["display_date"] = g.display_date(row.get("date"))
            for ev in entry["drift"].get("events", []):
                ev["display_date"] = g.display_date(ev.get("sim_date"))
            entry["online"] = True
        except UpstreamError as exc:
            entry["online"], entry["error"] = False, str(exc)
        nodes.append(entry)
    try:
        versions = g.up.control("/models", {"limit": 200})
    except UpstreamError:
        versions = []
    for v in versions:
        v["display_date"] = g.display_date(v.get("sim_date"))
    return {"nodes": nodes, "versions": versions,
            "sdft_alpha": SYSTEM_CONFIG["models"]["sdft_alpha"]}


# ─── Audit trail ─────────────────────────────────────────────────────────────

@api.get("/history")
def history(limit: int = 300) -> List[Dict]:
    """Decisions, rule changes, model versions and daily recommendation batches
    on one timeline, newest first."""
    g = gw()
    limit = _limit(limit)
    items: List[Dict] = []
    for d in g.store.decisions(limit=limit):
        items.append({"kind": "decision", "ts": d["ts"], "user": d["user"],
                      "sim_date": d["sim_date"], "sku": d["sku"], "channel": d["channel"],
                      "action": d["action"], "note": d["note"],
                      "price": d["applied_price"] if d["action"] == "approve" else d["final_price"],
                      "recommendation_id": d["recommendation_id"]})
    for c in g.store.rule_changes(limit=limit):
        items.append({"kind": "rule", "ts": c["ts"], "user": c["user"], "scope": c["scope"],
                      "sku": c["sku"], "before": c["before"], "after": c["after"]})
    for b in g.store.batches(limit=limit):
        items.append({"kind": "recommendations", "ts": b["ts"], "sim_date": b["sim_date"],
                      "count": b["count"], "bound": b["bound"]})
    try:
        for v in g.up.control("/models", {"limit": limit}):
            items.append({"kind": "model", "ts": v.get("ts"), "node": v.get("node"),
                          "sim_date": v.get("sim_date"), "model_kind": v.get("kind"),
                          "generation": v.get("generation"), "reason": v.get("reason"),
                          "ok": v.get("ok"), "duration_s": v.get("duration_s")})
    except UpstreamError:
        pass
    for it in items:
        it["display_date"] = g.display_date(it.get("sim_date"))
    items.sort(key=lambda x: x.get("ts") or 0, reverse=True)
    return items[:limit]


# ─── The interactive simulator ───────────────────────────────────────────────
# /internal/* is called BY the simulator; /sim/* is the dashboard driving it.

class OpenDayIn(BaseModel):
    sim_date: str


class LivePricesIn(BaseModel):
    sim_date: str
    require_approval: bool = False


@api.post("/internal/open-day")
def internal_open_day(body: OpenDayIn) -> Dict:
    return gw().open_day(body.sim_date)


@api.post("/internal/live-prices")
def internal_live_prices(body: LivePricesIn) -> Dict:
    return gw().live_prices(body.sim_date, require_approval=body.require_approval)


@api.post("/internal/reset")
def internal_reset() -> Dict:
    gw().reset_run()
    return {"ok": True}


def _with_sim_display(state: Dict) -> Dict:
    g = gw()
    for key in ("sim_date", "last_traded"):
        if state.get(key):
            state[f"display_{key}"] = g.display_date(state[key])
    for row in state.get("days", []):
        row["display_date"] = g.display_date(row.get("sim_date"))
    for ev in state.get("log", []):
        ev["display_date"] = g.display_date(ev.get("sim_date"))
    if state.get("shock"):
        state["shock"]["display_until"] = g.display_date(state["shock"].get("until_date"))
    return state


@api.get("/sim")
def sim_state() -> Dict:
    return _with_sim_display(gw().up.sim("/state"))


@api.post("/sim/{action}")
def sim_action(action: str, body: Optional[Dict] = None) -> Dict:
    if action not in {"play", "pause", "step", "speed", "outage", "latency",
                      "shock", "clear-shock", "restock", "settings", "reset"}:
        raise HTTPException(404, f"unknown simulator action {action!r}")
    return _with_sim_display(gw().up.sim_post(f"/{action}", body or {}))


app.include_router(api)


# ─── The built dashboard, when present ───────────────────────────────────────
# One origin for the demo: `npm run build` in webapp/ and the gateway serves it.

_DIST = PROJECT_ROOT / "webapp" / "dist"


@app.get("/{path:path}", include_in_schema=False)
def spa(path: str):
    if path.startswith("api/"):
        raise HTTPException(404, f"no endpoint /{path}")
    if not _DIST.exists():
        raise HTTPException(404, "dashboard not built: cd webapp && npm run build")
    target = (_DIST / path).resolve()
    if path and target.is_file() and _DIST.resolve() in target.parents:
        return FileResponse(target)
    return FileResponse(_DIST / "index.html")
