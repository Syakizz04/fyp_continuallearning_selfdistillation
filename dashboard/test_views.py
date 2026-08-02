"""
Smoke tests: every page renders without raising.

A Streamlit page fails at *render* time, over the websocket, long after the HTTP
route has already returned 200 - so hitting the URL proves nothing. `AppTest`
actually executes the script and collects whatever it raised, which is the only
cheap way to know a page still works.

These matter most for the two deployed-system pages, whose whole job is to
degrade gracefully: they must render when the services are up, when they are
down but a recorded run exists, and when there is nothing at all. The last case
is the one that breaks silently, because it is the one nobody demos until the
day the demo is on someone else's machine.

Run: pytest dashboard/ -q
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

DASHBOARD = Path(__file__).resolve().parent
if str(DASHBOARD) not in sys.path:
    sys.path.insert(0, str(DASHBOARD))

AppTest = pytest.importorskip("streamlit.testing.v1").AppTest

VIEWS = ["view_home", "view_initial", "view_drift", "view_cross",
         "view_live", "view_health"]

#: Long enough for the recorded-mode SQLite reads; these pages touch a 5 MB
#: event log on first render.
TIMEOUT = 90


def run_view(module: str, extra: str = "") -> "AppTest":
    script = (
        "import sys\n"
        f"sys.path.insert(0, {str(DASHBOARD)!r})\n"
        f"{extra}\n"
        f"import {module}\n"
        f"{module}.render()\n"
    )
    at = AppTest.from_string(script, default_timeout=TIMEOUT)
    at.run()
    return at


def assert_clean(at: "AppTest", module: str) -> None:
    if at.exception:
        detail = "\n".join(str(e.value) for e in at.exception)
        pytest.fail(f"{module} raised while rendering:\n{detail}")


@pytest.mark.parametrize("module", VIEWS)
def test_view_renders(module):
    assert_clean(run_view(module), module)


@pytest.mark.parametrize("module", ["view_live", "view_health"])
def test_deployed_pages_render_with_no_data_at_all(module):
    """The state that only shows up on someone else's machine.

    No services, no event log, no results. The pages must say so rather than
    raise on an empty frame or an absent file.
    """
    stub = (
        "import live as LV\n"
        "LV.probe = lambda: {'inventory': False, 'control': False, 'nodes': {}}\n"
        "LV.has_recorded = lambda: False\n"
        "LV.available_modes = lambda: [LV.NONE]\n"
        "LV.drift_arms = lambda: []\n"
        "LV.registry_models = lambda: __import__('pandas').DataFrame()\n"
        "LV.registry_nodes = lambda: __import__('pandas').DataFrame()\n"
        "LV.live_nodes = lambda: __import__('pandas').DataFrame()\n"
    )
    assert_clean(run_view(module, extra=stub), module)


@pytest.mark.parametrize("module", ["view_live", "view_health"])
def test_deployed_pages_render_without_httpx(module):
    """`httpx` ships in requirements-system.txt, NOT requirements.txt.

    The dashboard-deploy branch installs the slim viewer set, so live mode is
    genuinely unavailable there. Importing httpx at module scope would break
    that branch on import; this pins the lazy path.
    """
    stub = (
        "import live as LV\n"
        "LV._httpx = lambda: None\n"
        "LV.http_available = lambda: False\n"
        "LV.probe = lambda: {'inventory': False, 'control': False, 'nodes': {}}\n"
    )
    assert_clean(run_view(module, extra=stub), module)


def test_live_module_imports_no_training_stack():
    """The dashboard must stay installable from the slim viewer requirements.

    Run in a SUBPROCESS. Inspecting this process's `sys.modules` would be
    meaningless in a full-suite run: the drift_pipeline and edge_system tests
    import torch long before this one executes, so the check would pass alone
    and fail together while telling you nothing about the dashboard either way.
    """
    import subprocess

    probe = (
        "import sys\n"
        f"sys.path.insert(0, {str(DASHBOARD)!r})\n"
        "import live\n"
        "banned = {'torch', 'lightning', 'pytorch_forecasting',\n"
        "          'stable_baselines3', 'gymnasium', 'darts'}\n"
        "print(','.join(sorted(banned & set(sys.modules))))\n"
    )
    r = subprocess.run([sys.executable, "-c", probe], capture_output=True,
                       text=True, timeout=180)
    assert r.returncode == 0, f"importing dashboard.live failed:\n{r.stderr}"
    pulled = [m for m in r.stdout.strip().split(",") if m]
    assert not pulled, (
        f"importing dashboard.live pulled in {pulled} - the dashboard must not "
        f"depend on the training stack, or dashboard-deploy cannot install")


def test_httpx_is_not_a_hard_dependency():
    """Module scope must not import httpx, whatever is installed locally."""
    source = (DASHBOARD / "live.py").read_text(encoding="utf-8")
    top_level = [ln for ln in source.splitlines()
                 if ln.startswith(("import httpx", "from httpx"))]
    assert not top_level, (
        "httpx is imported at module scope in live.py; it is absent from the "
        "slim viewer requirements and this breaks dashboard-deploy on import")
