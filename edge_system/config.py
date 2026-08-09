"""
Single source of truth for the runtime system.

Mirrors the convention used by the experiment packages: a module-level dict that
is **mutated in place** after import rather than subclassed, so scenarios and
tests can shrink or repoint the system without a config framework. Helpers
re-read SYSTEM_CONFIG at call time so those mutations take effect.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List

PROJECT_ROOT = Path(__file__).resolve().parents[1]

SYSTEM_CONFIG: Dict = {
    # ── Topology ────────────────────────────────────────────────────────────
    # Three sales channels contending for ONE shared stock pool per SKU - the
    # omnichannel retail setting, where a physical till, an online storefront and
    # a marketplace listing all decrement the same inventory. This is the textbook
    # oversell scenario and needs no retraining: it reuses the existing
    # CA_1 x 100-item checkpoints as-is.
    #
    # M5 is Walmart store data, so a physical channel carrying the largest share
    # is the right shape. The WEIGHTS THEMSELVES ARE AN EXPERIMENTAL CHOICE, NOT
    # AN EMPIRICAL ONE, and the writeup must say so: Walmart's actual online share
    # over the M5 window (2011-2016) was roughly 3%, which would leave the pool
    # effectively single-channel, nothing would contend, and every sync policy
    # would look identical. These weights are set to put the pool under
    # contention, which is the regime under study - the same reasoning as
    # sim.initial_cover_days below.
    #
    # M5 records daily unit sales, not the channel a unit sold through. The split
    # and the discrete order arrivals built from it are a modelling overlay
    # (see sim/order_gen.py), calibrated so aggregating generated orders recovers
    # the observed daily total.
    "channels": [
        {"name": "pos",         "port": 8010, "weight": 0.55},
        {"name": "web",         "port": 8011, "weight": 0.35},
        {"name": "marketplace", "port": 8012, "weight": 0.10},
    ],
    "control_port": 8000,
    "inventory_port": 8001,
    "host": "127.0.0.1",

    # ── Sync ────────────────────────────────────────────────────────────────
    # NOTE: everything a *child service process* reads must appear in
    # _SERVICE_KEYS at the bottom of this file, or setting it has no effect on
    # the running system. See the warning there.
    "sync": {
        # strong_lock | eventual | escrow_quota   (the independent variable in E1)
        "policy": "escrow_quota",
        # Quota a node requests when its local rights run low, as a multiple of
        # its recent per-tick consumption. Larger = fewer round trips but a more
        # conservative (staler) local view.
        "quota_refill_multiple": 3.0,
        # Refill proactively once quota falls below this fraction of the last
        # grant, instead of waiting until an order cannot be covered.
        #
        # DEFAULT 0.0 = purely reactive, which is what the implementation
        # actually does and what the demarcation protocol's basic form
        # describes. It is off by default rather than at some tuned-looking
        # value because it has been measured to be near-inert: `_refill` takes a
        # fixed block, so the coordination rate is set by block size / order
        # size and not by when the top-up fires (75 refills at 0.0 vs 76 at 0.9
        # over 300 orders). A non-zero default would imply a mechanism that does
        # not bite. See EscrowQuotaPolicy.low_watermark.
        "quota_low_watermark": 0.0,
        "reservation_ttl_s": 300.0,      # uncommitted reservations auto-release
    },

    # ── Simulated network (E1/E2 independent variable) ──────────────────────
    # Latency is simulated rather than real BY DESIGN: it is the independent
    # variable, so it has to be a dial we set, not an emergent property of a
    # broker we cannot control. See the plan's Kafka rationale.
    "network": {
        # one-way node <-> inventory service delay
        "delay_ms": 0.0,
        "jitter_ms": 0.0,
        # channel names currently cut off from central
        "partition": [],
        # Conditions applied part-way through a run, e.g.
        #   [{"at_tick": 15, "partition": ["marketplace"]},
        #    {"at_tick": 25, "partition": []}]
        # Partition ONSET is the interesting case and it cannot be expressed by
        # the static setting above: a node partitioned from tick 0 never acquires
        # any escrow in the first place, so escrow_quota degrades exactly like
        # strong_lock and its availability advantage is invisible. The advantage
        # only exists for rights acquired BEFORE the link dropped.
        "schedule": [],
    },

    # ── Storage ─────────────────────────────────────────────────────────────
    "redis": {
        "url": "redis://127.0.0.1:6379/0",
        "key_prefix": "fyp:inv:",
        # redis | sqlite | auto. "auto" is convenient for development but the
        # E1 scripts must set this explicitly: which primitive served the
        # strong_lock arm is part of the result, not something to discover.
        "backend": "auto",
    },
    # SQLite is used for the durable event log and the control-plane registry,
    # off the hot path. It is also a drop-in PoolStore so the unit tests run
    # with no Redis up.
    "paths": {
        "state_dir": str(PROJECT_ROOT / "outputs" / "system" / "state"),
        "results":   str(PROJECT_ROOT / "outputs" / "system" / "results"),
        "events_db": str(PROJECT_ROOT / "outputs" / "system" / "state" / "events.db"),
        "registry_db": str(PROJECT_ROOT / "outputs" / "system" / "state" / "registry.db"),
        # `base_cover` = pricer retrained against realistic (lead-time)
        # inventory, an inventory-constrained reward, and cover-based inventory
        # scaling. The original `base` pricer is kept and is the arm to compare
        # against; it is effectively blind to stock. See
        # experiments/retrain_pricer.py.
        "base_ckpt_dir": str(PROJECT_ROOT / "outputs" / "drift" / "checkpoints" / "base_v4"),
        # v4 = v3 with competitor_price rebuilt as a real cross-store price for
        # the same item (it was the focal store's own department median). v3 was
        # v2 with only the inventory columns regenerated (5.1% stockout days
        # rather than 0.05%). demand_forecasting.csv is byte-identical
        # throughout, so the base TFT checkpoint remains valid; the base PPO
        # does not, because the RL state also went 13 -> 9 dims, and is retrained.
        "data_dir": str(PROJECT_ROOT / "data" / "processed_m5_v4"),
    },

    # ── Simulation ──────────────────────────────────────────────────────────
    "sim": {
        "seed": 42,
        "tick_days": 1,              # 1 tick = 1 M5 day
        "wall_clock_per_tick_s": 0.0,   # 0 = run as fast as possible
        "start_date": "2013-01-01",
        "end_date": "2015-12-31",
        "max_ticks": None,           # None = the whole window
        # Initial stock per SKU as a multiple of mean daily demand. This is an
        # experimental parameter, not a detail: generous cover means nothing ever
        # runs out and every sync policy looks identical.
        "initial_cover_days": 21.0,
        "replenish_cover_days": 21.0,
        "reorder_point_frac": 0.5,
        # None = all 100. Smoke runs and E1 pilots cut this down; the driver
        # makes ~n_skus x n_channels HTTP calls per tick, so it is the main
        # runtime dial.
        "n_skus": None,
        "mean_basket": 1.8,
        # Share of granted reservations abandoned instead of paid for. Exercises
        # the Saga compensation path (release) rather than leaving it dead code.
        "abandon_rate": 0.02,
        # node   -> quote comes from each edge node's PPO, using ITS stale view
        # static -> quote is base_price; no models loaded, no GPU. E1 uses this.
        "pricing": "node",
    },
}


def channel_names() -> List[str]:
    """Re-reads SYSTEM_CONFIG so post-import mutation takes effect."""
    return [c["name"] for c in SYSTEM_CONFIG["channels"]]


def channel(name: str) -> Dict:
    for c in SYSTEM_CONFIG["channels"]:
        if c["name"] == name:
            return c
    raise KeyError(f"unknown channel {name!r}; have {channel_names()}")


def ensure_dirs() -> None:
    for key in ("state_dir", "results"):
        Path(SYSTEM_CONFIG["paths"][key]).mkdir(parents=True, exist_ok=True)


# ─── Crossing the process boundary ───────────────────────────────────────────
#
# **Mutating SYSTEM_CONFIG does not cross a process boundary.** The offline
# pipelines mutate CONFIG in place and everything downstream sees it, because it
# is all one process. Here the inventory service, the control plane and each edge
# node are separate uvicorn children that import their own fresh copy of this
# module. A scenario that sets `sync.policy = "strong_lock"` in the supervisor
# and then launches the service gets a service running the *default* policy - and
# because every policy exposes the same endpoints and returns entirely plausible
# numbers, nothing fails. The run just quietly measures the wrong thing.
#
# This manifest is the single place that knows which settings a child reads. It
# drives BOTH directions - the env vars the supervisor exports, and the overrides
# a child applies at import - so the two cannot drift apart.
#
# **Adding a setting a service reads without adding it here reintroduces the
# bug.** `test_sim.py` scans the service modules for `SYSTEM_CONFIG[...]` reads
# and fails if any section is missing from this list.

_SERVICE_KEYS: List[tuple] = [
    # (section, key, environment variable)
    ("sync",    "policy",                "FYP_SYNC_POLICY"),
    ("sync",    "quota_refill_multiple", "FYP_QUOTA_REFILL"),
    ("sync",    "quota_low_watermark",   "FYP_QUOTA_WATERMARK"),
    ("sync",    "reservation_ttl_s",     "FYP_RESERVATION_TTL"),
    ("network", "delay_ms",              "FYP_DELAY_MS"),
    ("network", "jitter_ms",             "FYP_JITTER_MS"),
    ("network", "partition",             "FYP_PARTITION"),
    ("redis",   "url",                   "FYP_REDIS_URL"),
    ("redis",   "backend",               "FYP_POOL_BACKEND"),
    # Repointed by E3 (old vs re-grounded elasticity). Without this the edge
    # nodes load whatever `data_dir` defaults to and both arms of the ablation
    # read the SAME data - an experiment that can only report "no effect".
    ("paths",   "data_dir",              "FYP_DATA_DIR"),
    ("paths",   "base_ckpt_dir",         "FYP_BASE_CKPT_DIR"),
    ("paths",   "events_db",             "FYP_EVENTS_DB"),
    ("paths",   "registry_db",           "FYP_REGISTRY_DB"),
    ("paths",   "state_dir",             "FYP_STATE_DIR"),
    ("paths",   "results",               "FYP_RESULTS_DIR"),
]

#: Sections whose values a child service process may read.
SERVICE_SECTIONS = {section for section, _, _ in _SERVICE_KEYS}


def _coerce(raw: str, like):
    """Parse an env string back to the type the default value had."""
    if isinstance(like, bool):
        return raw.strip().lower() in {"1", "true", "yes", "on"}
    if isinstance(like, list):
        return [part for part in raw.split(",") if part]
    if isinstance(like, float):
        return float(raw)
    if isinstance(like, int):
        return int(raw)
    return raw


def _format(value) -> str:
    return ",".join(str(v) for v in value) if isinstance(value, list) else str(value)


def service_env() -> Dict[str, str]:
    """The environment a child service must be launched with. Read at call time."""
    return {var: _format(SYSTEM_CONFIG[section][key])
            for section, key, var in _SERVICE_KEYS}


def apply_env_overrides(environ: Dict[str, str] | None = None) -> List[str]:
    """
    Pull the manifest back out of the environment. Run at import.

    Returns the names of the settings that were overridden, which
    `/health` reports so a run can prove what actually arrived rather than
    trusting that it did.
    """
    environ = os.environ if environ is None else environ
    applied = []
    for section, key, var in _SERVICE_KEYS:
        raw = environ.get(var)
        if raw is None:
            continue
        SYSTEM_CONFIG[section][key] = _coerce(raw, SYSTEM_CONFIG[section][key])
        applied.append(f"{section}.{key}")
    return applied


#: Populated at import so a child process reflects its launch environment.
APPLIED_OVERRIDES: List[str] = apply_env_overrides()


def inventory_url() -> str:
    c = SYSTEM_CONFIG
    return f"http://{c['host']}:{c['inventory_port']}"


def control_url() -> str:
    c = SYSTEM_CONFIG
    return f"http://{c['host']}:{c['control_port']}"
