"""
The dashboard in a real browser: the built webapp (webapp/dist), served by the
real gateway, over `e2e_world.WorldUpstream` instead of the live services.

Every page, as operator and as viewer, must render with no uncaught error, no
console error, no failed API call and no NaN / undefined on screen. On top of
that: the viewer's restrictions are real in the UI, the main flows work end to
end, a dead backend degrades to error boxes, and nothing scrolls sideways on a
phone.

Opt-in, because it drives a browser (~1 min):
    FYP_E2E=1 python -m pytest edge_system/gateway/test_browser.py
Needs `pip install playwright` and Microsoft Edge (or set FYP_E2E_CHANNEL=chrome);
no browser download. Rebuild the dashboard first: cd webapp && npm run build

Live mode runs the read-only checks against a RUNNING system instead, so real
data - not the fake's - goes through every page:
    FYP_E2E=1 FYP_E2E_URL=http://127.0.0.1:8020 \
    FYP_E2E_OP=user:password FYP_E2E_VIEWER=user:password python -m pytest ...
Anything that would change the run (approving, saving rules, the simulator's
buttons, switching backends off) is skipped there.
"""

import os
import re
import tempfile
from pathlib import Path

import pytest

pytestmark = pytest.mark.skipif(os.environ.get("FYP_E2E") != "1",
                                reason="browser tests are opt-in: set FYP_E2E=1")

playwright = pytest.importorskip("playwright.sync_api")

from types import SimpleNamespace  # noqa: E402

from .e2e_world import Served, WorldUpstream  # noqa: E402
from .service import PROJECT_ROOT  # noqa: E402
from .test_service import SKUS  # noqa: E402

DIST = PROJECT_ROOT / "webapp" / "dist" / "index.html"
LIVE = os.environ.get("FYP_E2E_URL")
OP, OP_PW = (os.environ.get("FYP_E2E_OP") or "ops:correct-horse").split(":", 1)
VIEWER, VIEWER_PW = (os.environ.get("FYP_E2E_VIEWER") or "vic:viewer-pass").split(":", 1)
# A test that changes the run's state only runs against the fake world.
fake_only = pytest.mark.skipif(bool(LIVE), reason="changes state; fake world only")

# "/products/*" stands for the first product the system lists.
SHARED = ["/", "/products", "/products/*", "/recommendations", "/rules",
          "/inventory", "/models", "/history"]
OPERATOR_ONLY = ["/simulation", "/accounts"]
# v2, the business-first design, over the same data.
SHARED_V2 = ["/v2", "/v2/prices", "/v2/stock", "/v2/products", "/v2/products/*", "/v2/rules",
             "/v2/history", "/v2/models", "/v2/sync"]
OPERATOR_ONLY_V2 = ["/v2/simulation", "/v2/accounts"]
ALL = SHARED + OPERATOR_ONLY + SHARED_V2 + OPERATOR_ONLY_V2

# Text that only reaches the screen when a value went missing on the way.
JUNK = re.compile(r"\bNaN\b|\bundefined\b|\[object Object\]|\bInfinity\b|Invalid Date")


# ── fixtures ────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def browser():
    if not DIST.exists():
        pytest.skip("dashboard not built: cd webapp && npm run build")
    with playwright.sync_playwright() as p:
        b = p.chromium.launch(channel=os.environ.get("FYP_E2E_CHANNEL", "msedge"), headless=True)
        yield b
        b.close()


def make_world(**kw):
    up = WorldUpstream(**kw)
    d = Path(tempfile.mkdtemp(dir=os.environ.get("FYP_TEST_TMP")))
    return Served(up, d)


def seed(served):
    """Accounts, and enough recommendations for more than one page."""
    import httpx
    with httpx.Client(base_url=served.url) as c:
        assert c.post("/api/auth/setup", json={"username": OP, "password": OP_PW}).status_code == 200
        assert c.post("/api/auth/users", json={"username": VIEWER, "password": VIEWER_PW,
                                               "role": "viewer"}).status_code == 200
        for d in ("2013-02-27", "2013-02-28", "2013-03-01"):
            assert c.post("/api/recommendations/generate", json={"sim_date": d}).status_code == 200
        assert c.put(f"/api/rules/sku/{SKUS[1]}", json={"min_price": 1.0}).status_code in (200, 422)


@pytest.fixture(scope="module")
def world():
    if LIVE:
        yield SimpleNamespace(url=LIVE.rstrip("/"), up=None)
        return
    with make_world() as served:
        seed(served)
        yield served


def resolve(page, path):
    if not path.endswith("/products/*"):
        return path
    r = page.request.get("/api/products")
    skus = r.json() if r.ok else []          # inventory down: a 502, not a list
    return path[:-1] + (skus[0]["sku"] if skus else SKUS[0])


class Watch:
    """Everything that went wrong on a page while it was open."""

    def __init__(self, page, *, allow_status=()):
        self.errors = []
        self.allow = set(allow_status)
        page.on("pageerror", lambda e: self.errors.append(f"uncaught: {e}"))
        page.on("console", self._console)
        page.on("response", self._response)

    def _console(self, m):
        # The browser logs every failed request itself; `_response` already
        # judges those against `allow`, with the URL.
        if m.type == "error" and "Failed to load resource" not in m.text:
            self.errors.append(f"console: {m.text}")

    def _response(self, r):
        if "/api/" in r.url and r.status >= 400 and r.status not in self.allow:
            self.errors.append(f"HTTP {r.status} {r.request.method} {r.url}")


def new_page(browser, served, *, width=1280, height=900):
    ctx = browser.new_context(base_url=served.url, viewport={"width": width, "height": height})
    return ctx, ctx.new_page()


def sign_in(page, user, pw):
    page.goto("/")
    page.fill("#su", user)
    page.fill("#sp", pw)
    page.click("button.signin-btn")
    page.wait_for_selector("nav.nav", timeout=10000)


def settle(page):
    page.wait_for_load_state("networkidle")
    page.wait_for_timeout(300)


def visible_text(page):
    return page.inner_text("body")


def assert_rendered(page, path, watch):
    settle(page)
    text = visible_text(page)
    assert page.locator("h1").count() >= 1, f"{path}: no page heading - blank or crashed render"
    bad = JUNK.findall(text)
    assert not bad, f"{path}: junk on screen {sorted(set(bad))}"
    assert not watch.errors, f"{path}: " + "\n  ".join(watch.errors)


# ── every page, both roles ──────────────────────────────────────────────────

@pytest.mark.parametrize("path", ALL)
def test_operator_page_renders_clean(browser, world, path):
    ctx, page = new_page(browser, world)
    try:
        sign_in(page, OP, OP_PW)
        w = Watch(page)
        page.goto(resolve(page, path))
        assert_rendered(page, path, w)
    finally:
        ctx.close()


@pytest.mark.parametrize("path", SHARED + SHARED_V2)
def test_viewer_page_renders_clean(browser, world, path):
    ctx, page = new_page(browser, world)
    try:
        sign_in(page, VIEWER, VIEWER_PW)
        w = Watch(page)
        page.goto(resolve(page, path))
        assert_rendered(page, path, w)
        # Nothing a viewer can press changes anything.
        for label in ("Approve", "Reject", "Approve selected", "Save rules",
                      "Save product rule", "Set a price rule", "Approve all", "Edit"):
            n = page.get_by_role("button", name=label, exact=True).count()
            assert n == 0, f"{path}: viewer sees a {label!r} button"
    finally:
        ctx.close()


def test_viewer_cannot_reach_operator_pages(browser, world):
    ctx, page = new_page(browser, world)
    try:
        sign_in(page, VIEWER, VIEWER_PW)
        nav = page.inner_text("nav.nav")
        assert "Simulation" not in nav and "Accounts" not in nav
        assert page.locator('a[href="/simulation"]').count() == 0
        for path in OPERATOR_ONLY:
            page.goto(path)
            settle(page)
            assert page.url.rstrip("/") == world.url, f"viewer stayed on {path}: {page.url}"
        for path in OPERATOR_ONLY_V2:
            page.goto(path)
            settle(page)
            assert page.url.rstrip("/") == world.url + "/v2", f"viewer stayed on {path}: {page.url}"
    finally:
        ctx.close()


# ── the two designs ─────────────────────────────────────────────────────────

@pytest.mark.parametrize("start,expect", [("/products", "/v2/products"), ("/recommendations", "/v2/prices"),
                                          ("/", "/v2"), ("/inventory", "/v2/sync")])
def test_design_switch_keeps_the_page(browser, world, start, expect):
    ctx, page = new_page(browser, world)
    try:
        sign_in(page, OP, OP_PW)
        page.goto(start)
        settle(page)
        page.get_by_role("link", name="v2", exact=True).click()
        settle(page)
        assert page.url == world.url + expect
        page.get_by_role("link", name="v1", exact=True).click()
        settle(page)
        assert page.url.rstrip("/") == (world.url + start).rstrip("/")
    finally:
        ctx.close()


@pytest.mark.parametrize("start,home", [("/history", "/"), ("/v2/history", "/v2")])
def test_logo_goes_home_in_its_own_design(browser, world, start, home):
    ctx, page = new_page(browser, world)
    try:
        sign_in(page, OP, OP_PW)
        page.goto(start)
        settle(page)
        page.click("a.brand")
        settle(page)
        assert page.url.rstrip("/") == (world.url + home).rstrip("/")
    finally:
        ctx.close()


def test_links_inside_v2_stay_in_v2(browser, world):
    ctx, page = new_page(browser, world)
    try:
        sign_in(page, OP, OP_PW)
        page.goto("/v2/products")
        settle(page)
        page.locator("table a.mono").first.click()
        settle(page)
        assert "/v2/products/" in page.url
        assert "Today" in page.inner_text("nav.nav")
    finally:
        ctx.close()


@fake_only
@pytest.mark.parametrize("path", ["/", "/products", "/v2", "/v2/stock"])
def test_offline_banner_shows_in_both_designs(browser, world, path):
    ctx, page = new_page(browser, world)
    try:
        sign_in(page, OP, OP_PW)
        page.goto(path)
        settle(page)
        assert "is offline" not in visible_text(page)
        world.up.partition = ["marketplace"]
        page.wait_for_selector("text=Marketplace is offline", timeout=6000)
    finally:
        world.up.partition = []
        ctx.close()


def test_sidebar_says_when_a_model_update_holds_the_clock(browser, world):
    ctx, page = new_page(browser, world)
    try:
        sign_in(page, OP, OP_PW)
        page.goto("/v2")
        settle(page)
        assert "model update" not in visible_text(page)
        world.up.waiting_for = ["web"]
        page.wait_for_selector("text=Waiting for Web store model update", timeout=6000)
    finally:
        world.up.waiting_for = []
        ctx.close()


def test_profit_sparklines_draw_a_line_not_lone_points(browser, world):
    """A node's history interleaves pricing and forecast checks; a sparkline
    fed both breaks at every forecast row and draws no line at all."""
    ctx, page = new_page(browser, world)
    try:
        sign_in(page, OP, OP_PW)
        for path in ("/models", "/"):
            page.goto(path)
            settle(page)
            paths = page.eval_on_selector_all("svg.spark path", "els => els.map(e => e.getAttribute('d'))")
            lines = [d for d in paths if d and d.count("L") >= 10]
            assert len(lines) >= 3, (path, paths)
    finally:
        ctx.close()


def test_threshold_gap_after_a_retrain_is_shaded_and_named(browser, world):
    ctx, page = new_page(browser, world)
    try:
        sign_in(page, OP, OP_PW)
        page.goto("/v2/models")
        settle(page)
        assert visible_text(page).count("Learning normal level") == 2   # one legend per chart
    finally:
        ctx.close()


@pytest.mark.parametrize("path", ["/models", "/v2/models", "/", "/v2"])
def test_drift_warnings_page_five_at_a_time(browser, world, path):
    """6 warnings (2 per channel in the fake world): 5, then 1 on page 2."""
    ctx, page = new_page(browser, world)
    try:
        sign_in(page, OP, OP_PW)
        page.goto(path)
        settle(page)
        card = page.locator(".card, section").filter(has_text="Drift warnings").filter(has=page.locator(".pager")).last
        pager = card.locator(".pager")
        assert "1–5 of 6" in pager.inner_text()
        rows = card.locator("tbody tr, .list-row")
        assert rows.count() == 5
        pager.get_by_label("Next page").click()
        page.wait_for_function("() => [...document.querySelectorAll('.pager')].some(p => p.innerText.includes('6–6 of 6'))")
        assert rows.count() == 1
    finally:
        ctx.close()


# ── flows ───────────────────────────────────────────────────────────────────

def test_wrong_password_stays_on_sign_in_with_a_message(browser, world):
    ctx, page = new_page(browser, world)
    try:
        w = Watch(page, allow_status={401})
        page.goto("/")
        page.fill("#su", OP)
        page.fill("#sp", "not-the-password")
        page.click("button.signin-btn")
        page.wait_for_timeout(800)
        assert page.locator("nav.nav").count() == 0
        assert re.search(r"(?i)wrong|invalid|incorrect", visible_text(page)), visible_text(page)[:300]
        assert not w.errors, w.errors
    finally:
        ctx.close()


@fake_only
def test_approve_shows_up_in_history_signed_by_the_operator(browser, world):
    ctx, page = new_page(browser, world)
    try:
        sign_in(page, OP, OP_PW)
        w = Watch(page)
        page.goto("/recommendations")
        settle(page)
        row = page.locator("table.t tbody tr").filter(has=page.get_by_role("button", name="Approve", exact=True)).first
        sku = row.locator("td").filter(has_text=re.compile(r"FOODS_")).first.inner_text().strip()
        row.get_by_role("button", name="Approve", exact=True).click()
        settle(page)
        page.goto("/history")
        settle(page)
        text = visible_text(page)
        assert OP in text and sku.split()[0] in text, text[:600]
        assert not w.errors, w.errors
    finally:
        ctx.close()


def test_recommendations_pager_moves_and_stays_consistent(browser, world):
    ctx, page = new_page(browser, world)
    try:
        sign_in(page, OP, OP_PW)
        w = Watch(page)
        page.goto("/recommendations")
        settle(page)
        # The pending tab holds one day; "All" holds every day seeded.
        page.get_by_role("tab", name="All", exact=True).click()
        settle(page)
        nxt = page.get_by_role("button", name="Next page")
        assert nxt.count() == 1 and nxt.is_enabled(), "18 recommendations should not fit one page"
        first = page.inner_text("table.t tbody")
        nxt.click()
        settle(page)
        second = page.inner_text("table.t tbody")
        assert first != second, "next page shows the same rows"
        page.get_by_role("button", name="Previous page").click()
        settle(page)
        assert page.inner_text("table.t tbody") == first, "previous page did not come back"
        assert not w.errors, w.errors
    finally:
        ctx.close()


@fake_only
def test_saving_a_global_rule_is_reflected_and_logged(browser, world):
    ctx, page = new_page(browser, world)
    try:
        sign_in(page, OP, OP_PW)
        w = Watch(page)
        page.goto("/rules")
        settle(page)
        page.get_by_role("button", name="Save rules", exact=True).click()
        settle(page)
        assert "Rules saved" in visible_text(page)
        page.goto("/history")
        settle(page)
        assert re.search(r"(?i)rule", visible_text(page))
        assert not w.errors, w.errors
    finally:
        ctx.close()


@fake_only
def test_simulation_controls_reach_the_simulator(browser, world):
    ctx, page = new_page(browser, world)
    try:
        sign_in(page, OP, OP_PW)
        w = Watch(page)
        page.goto("/simulation")
        settle(page)
        before = len(world.up.posts)
        play = page.get_by_role("button", name=re.compile(r"^(Play|Start|Resume)", re.I))
        if play.count():
            play.first.click()
            settle(page)
            assert len(world.up.posts) > before, "Play sent nothing to the simulator"
        assert not w.errors, w.errors
    finally:
        ctx.close()


def test_sign_out_returns_to_sign_in_and_drops_the_session(browser, world):
    ctx, page = new_page(browser, world)
    try:
        sign_in(page, OP, OP_PW)
        page.get_by_role("button", name="Sign out").click()
        page.wait_for_selector("#su", timeout=5000)
        page.goto("/models")
        page.wait_for_selector("#su", timeout=5000)
        r = page.request.get("/api/status")
        assert r.status == 401
    finally:
        ctx.close()


def test_signed_out_deep_link_lands_on_the_page_after_sign_in(browser, world):
    ctx, page = new_page(browser, world)
    try:
        page.goto("/models")
        page.fill("#su", OP)
        page.fill("#sp", OP_PW)
        page.click("button.signin-btn")
        page.wait_for_selector("nav.nav")
        settle(page)
        assert page.url.endswith("/models"), page.url
    finally:
        ctx.close()


# ── degraded backends ───────────────────────────────────────────────────────

@fake_only
@pytest.mark.parametrize("down", ["simulator", "inventory service", "everything"])
def test_dead_backends_show_error_boxes_not_a_crash(browser, world, down):
    # The same server with backends switched off, then back on. The gateway's
    # state is process-global, so a second server would tear this one down.
    served, up = world, world.up
    if down in ("simulator", "everything"):
        up.sim_up = False
    if down in ("inventory service", "everything"):
        up.down.add("inventory service")
    if down == "everything":
        up.down |= {f"edge node {c}" for c in ("pos", "web", "marketplace")} | {"control plane"}
    try:
        ctx, page = new_page(browser, served)
        try:
            sign_in(page, OP, OP_PW)
            for path in ALL:
                w = Watch(page, allow_status={502, 409})
                page.goto(resolve(page, path))
                settle(page)
                assert page.locator("h1").count() >= 1, f"{path} with {down} down: blank page"
                bad = JUNK.findall(visible_text(page))
                assert not bad, f"{path} with {down} down: {sorted(set(bad))}"
                crashes = w.errors
                assert not crashes, f"{path} with {down} down: " + "\n  ".join(crashes)
        finally:
            ctx.close()
    finally:
        up.sim_up, up.down = True, set()


# ── phone width ─────────────────────────────────────────────────────────────

# The innermost elements that stick out past the right edge - the culprits,
# not every ancestor they stretch.
WIDEST = """() => { const W = document.documentElement.clientWidth; const out = [];
  for (const el of document.querySelectorAll('body *')) {
    const r = el.getBoundingClientRect();
    if (r.right <= W + 1 || r.width === 0) continue;
    if ([...el.children].some(c => c.getBoundingClientRect().right > W + 1)) continue;
    out.push(`${el.tagName.toLowerCase()}.${[...el.classList].join('.')} right=${Math.round(r.right)}`
             + ` "${(el.textContent || '').trim().slice(0, 40)}"`) }
  return out.slice(0, 8) }"""

@pytest.mark.parametrize("path", ALL)
def test_no_sideways_scroll_on_a_phone(browser, world, path):
    ctx, page = new_page(browser, world, width=390, height=844)
    try:
        sign_in(page, OP, OP_PW)
        page.goto(resolve(page, path))
        settle(page)
        sw, cw = page.evaluate("[document.documentElement.scrollWidth, document.documentElement.clientWidth]")
        assert sw <= cw + 1, f"{path}: page is {sw}px wide in a {cw}px viewport; widest: {page.evaluate(WIDEST)}"
    finally:
        ctx.close()
