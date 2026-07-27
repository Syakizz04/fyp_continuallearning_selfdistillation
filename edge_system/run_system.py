"""
Process supervisor: brings the whole system up, runs a scenario, tears it down.

    python -m edge_system.run_system --scenario smoke --ticks 30

Launches, as separate OS processes:

    control plane     :8000
    inventory sync    :8001
    edge node pos     :8010     ) one uvicorn each, models loaded in-process
    edge node web     :8011     ) - this is what "edge = local inference" means
    edge node market  :8012     )

then runs the tick loop against them over HTTP and shuts everything down.

**Edge nodes are native processes, not containers.** They retrain with SDFT on a
single RTX 3050 with 4 GB, and sharing that card across containers on Windows
(WSL2 + the container toolkit + VRAM contention between three PyTorch runtimes)
buys nothing and risks the retrain step OOM-ing for reasons unrelated to the
research. Docker runs Redis and only Redis.

Scenarios are named points in the experiment space. They mutate SYSTEM_CONFIG in
place - the same convention the offline pipelines use for CONFIG - so there is
one source of truth and no parallel settings object to keep in sync.
"""

from __future__ import annotations

import argparse
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import httpx

from .config import (PROJECT_ROOT, SYSTEM_CONFIG, channel_names, control_url,
                     ensure_dirs, inventory_url, service_env)


# ─── Scenarios ───────────────────────────────────────────────────────────────

SCENARIOS: Dict[str, Dict] = {
    # Does the stack stand up and trade? No models, tiny scope, seconds to run.
    "smoke": {
        "sim": {"n_skus": 10, "max_ticks": 30, "pricing": "static"},
        "sync": {"policy": "escrow_quota"},
        "network": {"delay_ms": 0.0},
    },
    # Same, but with the real TFT/PPO loaded - checks the model path end to end.
    "smoke_models": {
        "sim": {"n_skus": 5, "max_ticks": 10, "pricing": "node"},
        "sync": {"policy": "escrow_quota"},
        "network": {"delay_ms": 0.0},
    },
    # E1's shape: one policy x delay cell. exp_sync.py sweeps by overriding
    # --policy and --delay-ms rather than by defining 12 scenarios here.
    "sync_cell": {
        "sim": {"n_skus": 50, "pricing": "static", "initial_cover_days": 7.0},
        "network": {"delay_ms": 0.0},
    },
    # Availability under partition: the escrow argument at its sharpest.
    # The link drops PART WAY THROUGH, once marketplace holds escrow - see
    # SimulationDriver._apply_network_schedule for why onset is the only version
    # of this experiment that distinguishes the policies.
    "partition": {
        "sim": {"n_skus": 25, "max_ticks": 60, "pricing": "static",
                "initial_cover_days": 7.0},
        "network": {"delay_ms": 50.0, "partition": [],
                    "schedule": [{"at_tick": 20, "partition": ["marketplace"]},
                                 {"at_tick": 45, "partition": []}]},
    },
    # E2: models on, staleness graduated through the refill multiple. NOT the
    # default multiple - see the calibration note in the README.
    "staleness_cl": {
        "sim": {"n_skus": 25, "pricing": "node", "initial_cover_days": 10.0},
        "sync": {"policy": "escrow_quota", "quota_refill_multiple": 3.0},
        "network": {"delay_ms": 200.0},
    },
}


def apply_scenario(name: str) -> None:
    """Mutate SYSTEM_CONFIG in place. Unknown keys are a typo, not a feature."""
    if name not in SCENARIOS:
        raise SystemExit(f"unknown scenario {name!r}; have {sorted(SCENARIOS)}")
    for section, values in SCENARIOS[name].items():
        if section not in SYSTEM_CONFIG:
            raise KeyError(f"scenario {name!r} sets unknown section {section!r}")
        SYSTEM_CONFIG[section].update(values)


# ─── Preflight ───────────────────────────────────────────────────────────────

def check_redis() -> str:
    """
    Fail fast and usefully if the pool backend is not up.

    Which primitive served the run is part of the result - `strong_lock`'s claim
    to be a credible control arm rests on its atomicity coming from Redis rather
    than from our own escrow code - so an E1 run silently falling back to SQLite
    would invalidate the comparison it is making.
    """
    backend = SYSTEM_CONFIG["redis"].get("backend", "auto")
    if backend == "sqlite":
        return "sqlite (explicitly requested)"
    try:
        import redis  # noqa: PLC0415
        client = redis.Redis.from_url(SYSTEM_CONFIG["redis"]["url"],
                                      socket_connect_timeout=2.0)
        client.ping()
        client.close()
        return "redis"
    except Exception as exc:  # noqa: BLE001
        if backend == "redis":
            raise SystemExit(
                f"Redis is required for backend='redis' but is unreachable "
                f"({exc}).\n  Start it with:  docker compose up -d redis"
            )
        print(f"[run] Redis unavailable ({exc}); falling back to SQLite. "
              f"Do NOT report E1 numbers from this run.", file=sys.stderr)
        return "sqlite (fallback)"


# ─── Process management ──────────────────────────────────────────────────────

class ServiceProcess:
    """One uvicorn child, with the environment it needs to know who it is."""

    def __init__(self, name: str, app: str, port: int,
                 env: Optional[Dict[str, str]] = None) -> None:
        self.name = name
        self.app = app
        self.port = port
        self.env = env or {}
        self.proc: Optional[subprocess.Popen] = None
        self.log: Optional[Path] = None

    @property
    def url(self) -> str:
        return f"http://{SYSTEM_CONFIG['host']}:{self.port}"

    def preflight(self) -> None:
        """
        Refuse to start if something already holds the port.

        Not fussiness. If a service from an earlier run is still bound, our child
        fails to bind and dies, and `wait_ready` then gets a cheerful `ok: true`
        from the *stranger* - so the run proceeds against a process configured by
        some previous experiment, with a different policy, a different network
        delay, and stock left over from before. Every number it produced would be
        wrong and none of it would look wrong. Fail loudly instead.
        """
        import socket  # noqa: PLC0415

        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            if sock.connect_ex((SYSTEM_CONFIG["host"], self.port)) == 0:
                raise SystemExit(
                    f"port {self.port} ({self.name}) is already in use.\n"
                    f"  A service from an earlier run is probably still up. "
                    f"Stop it before starting a new one --\n"
                    f"  PowerShell:  Get-NetTCPConnection -State Listen "
                    f"-LocalPort {self.port} | %{{ Stop-Process -Id "
                    f"$_.OwningProcess -Force }}"
                )

    def start(self, log_dir: Path) -> None:
        env = os.environ.copy()
        env.update(self.env)
        # rich and the pipeline banners emit box-drawing characters that a
        # cp1252 console cannot encode; without this the child dies on its first
        # log line rather than on anything to do with the system.
        env.setdefault("PYTHONIOENCODING", "utf-8")
        env.setdefault("PYTHONUNBUFFERED", "1")

        self.log = log_dir / f"{self.name}.log"
        handle = self.log.open("w", encoding="utf-8")
        self.proc = subprocess.Popen(
            [sys.executable, "-m", "uvicorn", self.app,
             "--host", SYSTEM_CONFIG["host"], "--port", str(self.port),
             "--log-level", "warning"],
            cwd=str(PROJECT_ROOT), env=env, stdout=handle, stderr=subprocess.STDOUT,
        )

    def wait_ready(self, timeout: float) -> bool:
        deadline = time.time() + timeout
        while time.time() < deadline:
            if self.proc is not None and self.proc.poll() is not None:
                return False          # died during startup; the log says why
            try:
                if httpx.get(f"{self.url}/health", timeout=2.0).json().get("ok"):
                    return True
            except Exception:  # noqa: BLE001
                pass
            time.sleep(0.5)
        return False

    def stop(self, timeout: float = 10.0) -> None:
        if self.proc is None or self.proc.poll() is not None:
            return
        # CTRL_BREAK on Windows / SIGTERM elsewhere, so uvicorn runs its lifespan
        # shutdown and the event log and SQLite connections close cleanly.
        try:
            if os.name == "nt":
                self.proc.terminate()
            else:
                self.proc.send_signal(signal.SIGTERM)
            self.proc.wait(timeout=timeout)
        except Exception:  # noqa: BLE001
            self.proc.kill()

    def tail(self, n: int = 25) -> str:
        if self.log is None or not self.log.exists():
            return "(no log)"
        return "".join(self.log.read_text(encoding="utf-8",
                                          errors="replace").splitlines(True)[-n:])


def build_services(*, with_models: bool, strategy: str) -> List[ServiceProcess]:
    # Scenario settings reach the children through the environment, never through
    # the mutated dict - see `config.service_env`.
    shared = service_env()
    services = [
        ServiceProcess("control", "edge_system.control.service:app",
                       SYSTEM_CONFIG["control_port"], env=dict(shared)),
        ServiceProcess("inventory", "edge_system.inventory.service:app",
                       SYSTEM_CONFIG["inventory_port"], env=dict(shared)),
    ]
    if with_models:
        for ch in SYSTEM_CONFIG["channels"]:
            services.append(ServiceProcess(
                f"edge-{ch['name']}", "edge_system.edge.service:app", ch["port"],
                env={**shared, "FYP_NODE": ch["name"], "FYP_STRATEGY": strategy},
            ))
    return services


def verify_inventory_config() -> Dict:
    """
    Assert the inventory service is running the scenario that was asked for.

    Cheap insurance against the whole class of bug where a setting fails to
    reach a child process: every policy answers the same endpoints and returns
    entirely plausible numbers, so a mis-propagated policy produces a complete,
    well-formed, wrong result set. The only safe assumption is that config did
    NOT arrive until the service says otherwise.
    """
    health = httpx.get(f"{inventory_url()}/health", timeout=5.0).json()
    sync = SYSTEM_CONFIG["sync"]
    effective = health.get("effective", {})

    mismatches = []
    if health.get("policy") != sync["policy"]:
        mismatches.append(f"policy: asked {sync['policy']!r}, "
                          f"running {health.get('policy')!r}")
    if effective.get("reservation_ttl_s") != sync["reservation_ttl_s"]:
        mismatches.append(f"reservation_ttl_s: asked {sync['reservation_ttl_s']}, "
                          f"running {effective.get('reservation_ttl_s')}")
    if sync["policy"] == "escrow_quota":
        for key, want in (("refill_multiple", sync["quota_refill_multiple"]),
                          ("low_watermark", sync["quota_low_watermark"])):
            if effective.get(key) != want:
                mismatches.append(f"{key}: asked {want}, running {effective.get(key)}")

    backend = SYSTEM_CONFIG["redis"].get("backend", "auto")
    if backend == "redis" and health.get("backend") != "RedisPoolStore":
        mismatches.append(f"backend: required redis, running {health.get('backend')!r}")

    if mismatches:
        raise SystemExit(
            "the inventory service is not running the requested configuration:\n"
            + "\n".join(f"  - {m}" for m in mismatches)
            + "\nRefusing to run: the results would be labelled with settings "
              "that never executed.\n"
              "  Every setting a service reads must appear in "
              "`config._SERVICE_KEYS`."
        )
    return health


# ─── Entry point ─────────────────────────────────────────────────────────────

def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description="Run the FYP edge system.")
    p.add_argument("--scenario", default="smoke", choices=sorted(SCENARIOS))
    p.add_argument("--ticks", type=int, help="override sim.max_ticks")
    p.add_argument("--n-skus", type=int, help="override sim.n_skus")
    p.add_argument("--policy", choices=["strong_lock", "eventual", "escrow_quota"])
    p.add_argument("--delay-ms", type=float, help="one-way node<->centre delay")
    p.add_argument("--partition", nargs="*", help="channels cut off from the centre")
    p.add_argument("--strategy", default="sdft",
                   help="CL method the edge nodes retrain with")
    p.add_argument("--run-id", help="results filename suffix (default: scenario)")
    p.add_argument("--keep-up", action="store_true",
                   help="leave services running after the sim (for the dashboard)")
    p.add_argument("--startup-timeout", type=float, default=180.0,
                   help="per-service readiness timeout; nodes load torch, so be "
                        "generous on a cold start")
    args = p.parse_args(argv)

    apply_scenario(args.scenario)
    sim = SYSTEM_CONFIG["sim"]
    if args.ticks is not None:
        sim["max_ticks"] = args.ticks
    if args.n_skus is not None:
        sim["n_skus"] = args.n_skus
    if args.policy:
        SYSTEM_CONFIG["sync"]["policy"] = args.policy
    if args.delay_ms is not None:
        SYSTEM_CONFIG["network"]["delay_ms"] = args.delay_ms
    if args.partition is not None:
        SYSTEM_CONFIG["network"]["partition"] = list(args.partition)

    with_models = sim.get("pricing", "node") == "node"
    run_id = args.run_id or args.scenario

    ensure_dirs()
    log_dir = Path(SYSTEM_CONFIG["paths"]["state_dir"]) / "logs" / run_id
    log_dir.mkdir(parents=True, exist_ok=True)

    backend = check_redis()
    print(f"[run] scenario={args.scenario} policy={SYSTEM_CONFIG['sync']['policy']} "
          f"delay={SYSTEM_CONFIG['network']['delay_ms']}ms backend={backend} "
          f"models={with_models}")

    services = build_services(with_models=with_models, strategy=args.strategy)
    started: List[ServiceProcess] = []
    rc = 0
    try:
        # All ports checked up front, so a clash is reported before anything is
        # launched rather than half way through bringing the stack up.
        for svc in services:
            svc.preflight()

        for svc in services:
            print(f"[run] starting {svc.name} on :{svc.port} ...", flush=True)
            svc.start(log_dir)
            started.append(svc)
            if not svc.wait_ready(args.startup_timeout):
                print(f"[run] {svc.name} failed to become ready:\n{svc.tail()}",
                      file=sys.stderr)
                return 1
            print(f"[run] {svc.name} ready", flush=True)

        # Applied after startup so the service picks up any CLI override, and so
        # a sweep can re-point a live system without restarting it.
        httpx.post(f"{inventory_url()}/admin/network", timeout=10.0, json={
            "delay_ms": SYSTEM_CONFIG["network"]["delay_ms"],
            "jitter_ms": SYSTEM_CONFIG["network"]["jitter_ms"],
            "partition": SYSTEM_CONFIG["network"]["partition"],
        }).raise_for_status()

        health = verify_inventory_config()
        print(f"[run] verified: policy={health['policy']} backend={health['backend']}")

        from .sim.driver import SimulationDriver   # torch-free, but heavy: pandas
        driver = SimulationDriver(run_id=run_id,
                                  pricing=sim.get("pricing", "node"))
        try:
            path = driver.run()
            summary = driver.summary     # taken before the clients close
        finally:
            driver.close()

        print(f"\n[run] {run_id}: {summary['units_committed']} units sold, "
              f"fill rate {summary['fill_rate']:.1%}, "
              f"mean staleness {summary['mean_staleness_units']:.1f} units, "
              f"oversell {summary['inventory_metrics'].get('oversell_units', 0)}")
        print(f"[run] results -> {path}")

        if args.keep_up:
            print("[run] --keep-up: services still running. Ctrl-C to stop.")
            try:
                while True:
                    time.sleep(1.0)
            except KeyboardInterrupt:
                pass
    except KeyboardInterrupt:
        rc = 130
    finally:
        for svc in reversed(started):
            svc.stop()
        print("[run] all services stopped")
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
