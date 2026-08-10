"""
Live status table for a running (or resumed) sweep. Read-only - it only parses
the per-cell logs the launcher already writes, so it cannot interfere with the
sweep the way attaching to its tmux pane risks doing (a stray keystroke there
sends a signal to the running processes; this never touches them).

    python -m experiments.monitor_e2                    # E2, refresh every 30s
    python -m experiments.monitor_e2 --layout e5        # the alpha sweep
    python -m experiments.monitor_e2 --once             # single snapshot

Each cell's own process prints `done in X min (N FC + M RL retrains)` to its
stdout when it finishes, and the launcher redirects that into
`outputs/drift/logs/<exp>/<cell>.log` - so wall time and retrain counts are
already on disk the moment a cell completes, with no changes to the sweep
itself. `probe_scores_*.json` is the authoritative "done" signal (the same file
the sweeps' own resume logic keys off); the log line is read only for detail.

The two layouts differ in how a cell is addressed, which is why this needs to
know which one it is looking at rather than globbing blindly:

    e2   logs/e2/<censoring>_<arm>.log
         results/e2/<censoring>/probe_scores_<arm>.json
    e5   logs/e5/<censoring>_alpha<a>.log
         results/e5/<censoring>/alpha_<tag>/probe_scores_sdft.json
"""

from __future__ import annotations

import argparse
import re
import sys
import time
from pathlib import Path
from typing import List, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from rich.console import Console  # noqa: E402
from rich.live import Live  # noqa: E402
from rich.table import Table  # noqa: E402

from experiments.exp_staleness_cl import ARMS as E2_ARMS  # noqa: E402
from experiments.exp_staleness_cl import \
    DEFAULT_CENSORING as E2_CENSORING  # noqa: E402
from experiments.exp_sdft_alpha import DEFAULT_ALPHAS as E5_ALPHAS  # noqa: E402
from experiments.exp_sdft_alpha import \
    DEFAULT_CENSORING as E5_CENSORING  # noqa: E402
from experiments.exp_sdft_alpha import alpha_tag  # noqa: E402

console = Console()

DONE_RE = re.compile(r"done in ([\d.]+) min \((\d+) FC \+ (\d+) RL retrains\)")
#: KeyboardInterrupt and CUDA OOM are listed explicitly because both have
#: actually happened: a stray Ctrl-C killed the first E2 attempt, and six
#: concurrent SDFT cells (each holding a student AND a frozen teacher) exhausted
#: a 15 GB card. Neither is a code fault, and both are worth naming rather than
#: flattening into a generic "Error".
FAIL_MARKERS = ("out of memory", "KeyboardInterrupt", "Traceback", "Error")


def cells(layout: str) -> List[Tuple[str, str, str, str]]:
    """(row_label, col_label, log stem, probe path relative to the results root)."""
    out = []
    if layout == "e5":
        for c in E5_CENSORING:
            for a in E5_ALPHAS:
                out.append((c, f"alpha={a}", f"{c}_alpha{a}",
                            f"{c}/{alpha_tag(a)}/probe_scores_sdft.json"))
    else:
        for c in E2_CENSORING:
            for arm in E2_ARMS:
                out.append((c, arm, f"{c}_{arm}",
                            f"{c}/probe_scores_{arm}.json"))
    return out


def cell_status(logdir: Path, resultsdir: Path, stem: str, probe_rel: str) -> dict:
    log = logdir / f"{stem}.log"
    probe = resultsdir / probe_rel

    if probe.exists():
        text = log.read_text(errors="ignore") if log.exists() else ""
        m = DONE_RE.search(text)
        if m:
            return {"status": "[green]done[/green]",
                    "elapsed": f"{float(m.group(1)):.1f} min",
                    "retrains": f"{m.group(2)} FC + {m.group(3)} RL"}
        # Probe scores present without a fresh timing line: the cell was
        # skipped on a resume, or (E2 only) copied from another censoring level.
        return {"status": "[green]done[/green]", "elapsed": "(earlier run)",
                "retrains": "-"}

    if not log.exists():
        return {"status": "[dim]pending[/dim]", "elapsed": "-", "retrains": "-"}

    text = log.read_text(errors="ignore")
    age_min = (time.time() - log.stat().st_mtime) / 60
    hit = next((m for m in FAIL_MARKERS if m in text), None)
    if hit:
        label = "OOM" if hit == "out of memory" else hit
        return {"status": f"[red]failed ({label})[/red]",
                "elapsed": f"stopped {age_min:.0f}m ago", "retrains": "-"}
    n = len(re.findall(r"drift retrain", text))
    return {"status": "[yellow]running[/yellow]",
            "elapsed": f"log wrote {age_min:.0f}m ago",
            "retrains": f"{n} so far"}


def render(layout: str, out: str) -> Table:
    logdir = PROJECT_ROOT / "outputs" / "drift" / "logs" / out
    resultsdir = PROJECT_ROOT / "outputs" / "drift" / "results" / out

    table = Table(title=f"{layout.upper()} sweep status ({out})")
    table.add_column("censoring")
    table.add_column("alpha" if layout == "e5" else "arm")
    table.add_column("status")
    table.add_column("elapsed / wall time")
    table.add_column("retrains")

    grid = cells(layout)
    n_done = 0
    for row_label, col_label, stem, probe_rel in grid:
        s = cell_status(logdir, resultsdir, stem, probe_rel)
        table.add_row(row_label, col_label, s["status"], s["elapsed"], s["retrains"])
        if "done" in s["status"]:
            n_done += 1
    table.caption = f"{n_done}/{len(grid)} cells done"
    return table


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Live status table for a sweep.")
    ap.add_argument("--layout", choices=("e2", "e5"), default="e2")
    ap.add_argument("--out", default=None,
                    help="results/logs subdirectory (defaults to the layout name)")
    ap.add_argument("--interval", type=float, default=30.0)
    ap.add_argument("--once", action="store_true")
    args = ap.parse_args(argv)
    out = args.out or args.layout

    if args.once:
        console.print(render(args.layout, out))
        return 0

    with Live(render(args.layout, out), console=console, refresh_per_second=1) as live:
        while True:
            time.sleep(args.interval)
            live.update(render(args.layout, out))


if __name__ == "__main__":
    raise SystemExit(main())
