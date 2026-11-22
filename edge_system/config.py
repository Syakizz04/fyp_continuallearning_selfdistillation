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
    # The decision-support gateway: the only service the operator dashboard
    # talks to. It owns price rules, recommendations and decisions, and reads
    # everything else from the services above.
    "gateway_port": 8020,
    "sim_port": 8030,
    "host": "127.0.0.1",

    # ── Models served by the edge nodes ─────────────────────────────────────
    "models": {
        # The deployed SDFT trade-off. `load_base` rebinds the drift CONFIG["cl"]
        # to the checkpoint's base_meta.json, which stores 0.5 - the value E2
        # measured and found does not adapt (adaptation 0.075 under
        # escrow_quota). E5 at 0.99 recovered adaptation 0.641 while halving
        # naive's forgetting. Written into the model's own dict after load, the
        # same in-place injection exp_sdft_alpha.py uses.
        "sdft_alpha": 0.99,
        # A node scores its own forecaster on realised demand every N ticks and
        # feeds that MASE to its drift monitor. Without this the forecasting
        # detector never sees a value - the driver only reports profit_index -
        # so the forecaster could never retrain in the deployed system. 7 = the
        # weekly cadence the drift thresholds were calibrated on. 0 disables.
        "fc_check_every_ticks": 7,
        # How the node decides its pricer has drifted.
        #   "fixed"   - one day's profit_index below drift CONFIG rl_profit_floor
        #               (1.0), twice running. The original behaviour.
        #   "rolling" - profit over the last rl_window_days against its base-price
        #               reference, below a floor the node calibrates on its own
        #               first rl_warmup_days: mu - rl_k_sigma * sigma.
        # "fixed" fires on noise live: a day's index for ten SKUs is a few dozen
        # orders, and the live index does not sit at 1.0 (the reference is
        # base-price revenue, not the training-era profit the offline floor was
        # defined against). Chosen by replaying a year of simulated orders for
        # the demo's SKUs: window 7 / warm-up 28 / k 2.5 gave 7, 0 and 0 false
        # fires a year on pos/web/marketplace, and caught a 40% slump on pos and
        # web every time within 3-6 days. A demo setting; E2 does not use the
        # node monitor.
        "rl_detector": "fixed",
        # What the pricing detector reads (rolling mode).
        #   "revenue" - the channel's revenue against expected demand at base
        #               price. The original signal. It is a revenue alarm: the
        #               pricer's whole action range moves it ~11%, under its
        #               noise floor, while demand slumps, refusals and price
        #               rules move it freely.
        #   "critic"  - the pricer's own expectation against the outcome, like
        #               the forecast check: the day's training reward over what
        #               PPO's critic expected, asked after the day with the
        #               demand that turned up. Needs no true demand curve.
        #               Measured (experiments/critic_compare.py, rebuilt base_v4,
        #               17 non-overlapping trials 2013-2015): catches a pricer
        #               stuck at -10% (pos 14/17, web 9/17; revenue 1/17, 0/17),
        #               but not one stuck at base or +10%, and not a 2x
        #               price-sensitive world (0-1/17, the same as no fault). It
        #               is a one-sided markup gauge. A one-number markup index
        #               matches it (12/17, 11/17) or beats it (marketplace 9/17 vs
        #               3/17), because price is not in V(s)'s inputs. The earlier
        #               "13/15 and 12/15" figures (old pricer, lost script) do
        #               not reproduce. A demo setting; E2 does not use the node
        #               monitor.
        "rl_signal": "critic",
        "rl_window_days": 7,
        "rl_warmup_days": 28,
        "rl_k_sigma": 2.5,
        # What the forecaster check scores.
        #   "dataset"  - MASE against the static M5 frame, on the offline
        #                threshold. The original behaviour. Every node sees the
        #                same series whatever it sold, so in the demo it never
        #                fired: 1.31-1.62 against 1.56, never twice running.
        #   "realised" - MASE of the node's own 14-day forecast against what it
        #                was asked for (ordered / channel share) on the SKUs it
        #                trades, with a threshold calibrated on its first
        #                fc_warmup_days checks: mu + fc_live_k * sigma, and
        #                fc_live_consecutive breaches running. The same orders
        #                become the retrain's data. Re-warms after each retrain.
        # Chosen by replaying a year of simulated orders: warm-up 28 / k 3 / 3
        # running gave 1-2 retrains a year on normal trading (real seasonal M5
        # drift) and caught a 10-day x2.5 surge 7-9 times in 9 within 3-4 days.
        # A slump barely moves it - the median forecast already sits low - so
        # slumps are the pricing detector's to catch. Needs fc_check_every_ticks
        # 1. A demo setting; E2 does not use the node monitor.
        "fc_signal": "dataset",
        # After a model swap the old line is dropped (it belonged to the old
        # model). With this > 0, a provisional line goes up after this many of
        # the new model's own checks: level from those checks, spread carried
        # from the previous calibration (it is demand noise, not the model),
        # refined each check, replaced by a full calibration at the warm-up.
        # 0 = no line and no checking for the whole warm-up (the original
        # rule). 7 = one rolling window, the earliest the pricing value holds
        # only new-model days. Replayed on a 400-day demo run (1 pricer and 7
        # forecaster swaps): blind spell 28 -> 7 days, the early line raised
        # no alarm in any of them, and sat 0.1-2.7 sigma from the line later
        # learned on 28 checks. The first calibration of a run has no
        # previous spread, so it still waits the full warm-up.
        "relearn_min_days": 7,
        "fc_warmup_days": 28,
        "fc_live_k": 3.0,
        "fc_live_consecutive": 3,
        # Where a node runs a retrain.
        #   "thread"     - in the node process, on a background thread. The
        #                  original behaviour, and what every experiment uses.
        #   "subprocess" - in a short-lived child (edge/retrain_child.py), one
        #                  at a time machine-wide. A retrain left ~1.2 GB
        #                  committed (~0.8 GB resident) in the node for good;
        #                  a child gives it back on exit. Costs the child's
        #                  start-up per retrain. SDFT / naive / periodic only -
        #                  replay and EWC keep state in the controller.
        "retrain_process": "thread",
    },

    # ── Decision-support gateway ────────────────────────────────────────────
    "gateway": {
        # M5 records sell prices but no unit costs, so margin needs an assumed
        # cost. Unit cost = base_price x this ratio unless a per-SKU rule sets
        # one. AN ASSUMPTION, NOT DATA - the writeup must say so.
        "unit_cost_ratio": 0.70,
        # Which edge node's forecaster the product page shows by default. Every
        # node forecasts the same SKU-level demand; they differ only once their
        # local retrains diverge.
        "primary_node": "pos",
        # Seconds between the gateway's polls for a new simulated day, on which
        # it generates that day's price recommendations.
        "poll_interval_s": 1.0,
    },

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
        "decisions_db": str(PROJECT_ROOT / "outputs" / "system" / "state" / "decisions.db"),
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
        # Which n_skus: "stable" = first by name (every experiment), "top" =
        # best sellers (the demo, so a day's profit is more than a few orders).
        "sku_select": "stable",
        # Demo-only filters on that choice (None = off, as every experiment
        # runs): drop SKUs with a no-sales run longer than this many days in
        # the run's window, and take at most this many from one department.
        "sku_max_gap_days": None,
        "sku_per_dept": None,
        "mean_basket": 1.8,
        # Share of granted reservations abandoned instead of paid for. Exercises
        # the Saga compensation path (release) rather than leaving it dead code.
        "abandon_rate": 0.02,
        # node   -> quote comes from each edge node's PPO, using ITS stale view
        # static -> quote is base_price; no models loaded, no GPU. E1 uses this.
        "pricing": "node",
        # Live sim service only: a retrain is treated as overnight work. A
        # channel trades out the day its retrain started on the old model, and
        # the clock will not open the next day until the new one is serving -
        # so the retrain lag is one trading day on any machine, at any speed,
        # instead of however many days the clock ran during the real fit.
        # The cap stops a stuck fit from hanging the run; after it the clock
        # carries on and the swap lands whenever the fit finishes.
        "hold_for_retrain": True,
        "retrain_hold_max_s": 900.0,
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
    ("paths",   "decisions_db",          "FYP_DECISIONS_DB"),
    ("models",  "sdft_alpha",            "FYP_SDFT_ALPHA"),
    ("models",  "fc_check_every_ticks",  "FYP_FC_CHECK_EVERY"),
    ("models",  "rl_detector",           "FYP_RL_DETECTOR"),
    ("models",  "rl_window_days",        "FYP_RL_WINDOW_DAYS"),
    ("models",  "rl_warmup_days",        "FYP_RL_WARMUP_DAYS"),
    ("models",  "relearn_min_days",      "FYP_RELEARN_MIN_DAYS"),
    ("models",  "rl_signal",             "FYP_RL_SIGNAL"),
    ("models",  "rl_k_sigma",            "FYP_RL_K_SIGMA"),
    ("models",  "fc_signal",             "FYP_FC_SIGNAL"),
    ("models",  "fc_warmup_days",        "FYP_FC_WARMUP_DAYS"),
    ("models",  "fc_live_k",             "FYP_FC_LIVE_K"),
    ("models",  "fc_live_consecutive",   "FYP_FC_LIVE_CONSECUTIVE"),
    ("models",  "retrain_process",       "FYP_RETRAIN_PROCESS"),
    ("gateway", "unit_cost_ratio",       "FYP_UNIT_COST_RATIO"),
    ("gateway", "primary_node",          "FYP_PRIMARY_NODE"),
    ("gateway", "poll_interval_s",       "FYP_GATEWAY_POLL_S"),
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


def node_url(name: str) -> str:
    return f"http://{SYSTEM_CONFIG['host']}:{channel(name)['port']}"


def sim_url() -> str:
    c = SYSTEM_CONFIG
    return f"http://{c['host']}:{c['sim_port']}"


def gateway_url() -> str:
    c = SYSTEM_CONFIG
    return f"http://{c['host']}:{c['gateway_port']}"
