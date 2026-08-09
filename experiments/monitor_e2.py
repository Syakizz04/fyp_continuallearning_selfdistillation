"""
Live status table for a running (or resumed) E2 sweep. Read-only - it only
parses the per-cell logs `run_e2.sh` already writes, so it cannot interfere
with the sweep the way attaching to its tmux pane risks doing (a stray
keystroke there sends a signal to the running processes; this never touches
them).

    python -m experiments.monitor_e2                 # refresh every 30s
    python -m experiments.monitor_e2 --interval 10
    python -m experiments.monitor_e2 --once           # single snapshot, no loop

Each cell's own process prints `done in X min (N FC + M RL retrains)` to its
own stdout when it finishes (see exp_staleness_cl.main), and `run_e2.sh`
redirects that stdout into `outputs/drift/logs/e2/<censoring>_<arm>.log` - so
wall time and retrain counts are already sitting in the log the moment a cell
completes, with no changes needed to the sweep itself. `probe_scores_*.json`
is the authoritative "done" signal (same file the sweep's own resume logic
uses), the log line is only read for its timing detail.
"""

from __future__ import annotations

import argparse
import re
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from rich.console import Console  # noqa: E402
from rich.live import Live  # noqa: E402
from rich.table import Table  # noqa: E402

from experiments.exp_staleness_cl import DEFAULT_CENSORING  # noqa: E402
from experiments.exp_staleness_cl import ARMS as ALL_ARMS  # noqa: E402

console = Console()

DONE_RE = re.compile(r"done in ([\d.]+) min \((\d+) FC \+ (\d+) RL retrains\)")
#: KeyboardInterrupt is listed explicitly - that is what actually killed the
#: first attempt (a stray Ctrl+C into the tmux pane), not a code bug, and it is
#: worth surfacing distinctly rather than lumping it under a generic "Error".
FAIL_MARKERS = ("Traceback", "KeyboardInterrupt", "Error")


def cell_status(logdir: Path, resultsdir: Path, censoring: str, arm: str) -> dict:
    log = logdir / f"{censoring}_{arm}.log"
    probe = resultsdir / censoring / f"probe_scores_{arm}.json"

    if probe.exists():
        text = log.read_text(errors="ignore") if log.exists() else ""
        m = DONE_RE.search(text)
        if m:
            return {"status": "[green]done[/green]",
                    "elapsed": f"{float(m.group(1)):.1f} min",
                    "retrains": f"{m.group(2)} FC + {m.group(3)} RL"}
        # Probe scores exist but the log line wasn't found - e.g. a
        # STATIC_ARMS ('frozen') cell copied from another censoring level
        # rather than actually run, so it never printed its own timing.
        return {"status": "[green]done[/green]", "elapsed": "(copied)", "retrains": "-"}

    if not log.exists():
        return {"status": "[dim]pending[/dim]", "elapsed": "-", "retrains": "-"}

    text = log.read_text(errors="ignore")
    age_min = (time.time() - log.stat().st_mtime) / 60
    if any(marker in text for marker in FAIL_MARKERS):
        hit = next(marker for marker in FAIL_MARKERS if marker in text)
        return {"status": f"[red]interrupted ({hit})[/red]",
                "elapsed": f"stalled {age_min:.0f}m ago", "retrains": "-"}
    return {"status": "[yellow]running[/yellow]",
            "elapsed": f"log last wrote {age_min:.0f}m ago", "retrains": "-"}


def render(out: str) -> Table:
    logdir = PROJECT_ROOT / "outputs" / "drift" / "logs" / "e2"
    resultsdir = PROJECT_ROOT / "outputs" / "drift" / "results" / out

    table = Table(title=f"E2 sweep status ({out})")
    table.add_column("censoring")
    table.add_column("arm")
    table.add_column("status")
    table.add_column("elapsed / wall time")
    table.add_column("retrains")

    n_done = 0
    n_total = 0
    for censoring in DEFAULT_CENSORING:
        for arm in ALL_ARMS:
            n_total += 1
            s = cell_status(logdir, resultsdir, censoring, arm)
            table.add_row(censoring, arm, s["status"], s["elapsed"], s["retrains"])
            if "done" in s["status"]:
                n_done += 1
    table.caption = f"{n_done}/{n_total} cells done"
    return table


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Live status table for an E2 sweep.")
    ap.add_argument("--out", default="e2")
    ap.add_argument("--interval", type=float, default=30.0,
                    help="seconds between refreshes")
    ap.add_argument("--once", action="store_true",
                    help="print a single snapshot and exit, no live refresh")
    args = ap.parse_args(argv)

    if args.once:
        console.print(render(args.out))
        return 0

    with Live(render(args.out), console=console, refresh_per_second=1) as live:
        while True:
            time.sleep(args.interval)
            live.update(render(args.out))


if __name__ == "__main__":
    raise SystemExit(main())
