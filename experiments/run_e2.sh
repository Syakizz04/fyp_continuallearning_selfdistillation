#!/usr/bin/env bash
# E2 sweep launcher for a rented GPU box.
#
#   bash experiments/run_e2.sh                 # 3 censoring x 4 arms, 4 at a time
#   CONCURRENCY=6 bash experiments/run_e2.sh   # more parallel (needs ~30 GB RAM)
#   ARMS="frozen sdft" bash experiments/run_e2.sh
#
# Every (censoring, arm) cell is independent - separate process, separate results
# directory, separate base load - so the sweep parallelises at cell granularity
# rather than per censoring level. That matters: the models are ~4 MB and much of
# each check is pandas and PPO env rollouts, so one cell leaves most of a 3090
# idle. Running several is what actually uses the card you are paying for.
#
# The driver is resumable (a cell whose probe scores exist is skipped), so
# re-running this after a crash, a disconnect, or an instance restart picks up
# where it stopped. That is the intended recovery path - do not pass --force.

set -uo pipefail
cd "$(dirname "$0")/.."

CENSORING=${CENSORING:-"none strong_lock escrow_quota"}
ARMS=${ARMS:-"frozen naive replay sdft"}
CONCURRENCY=${CONCURRENCY:-4}
OUT=${OUT:-e2}
LOGDIR=${LOGDIR:-outputs/drift/logs/e2}

# Thread caps, checkpointing and the progress display all live in the shared
# launcher plumbing - see experiments/sweep_common.sh for why each is set.
source experiments/sweep_common.sh
sweep_setup_env
mkdir -p "$LOGDIR"

if ! python -m experiments.vast_preflight --concurrency "$CONCURRENCY"; then
    echo "Preflight failed - not starting." >&2
    exit 1
fi

# `frozen` never retrains, so it is genuinely identical at every censoring
# level; it is still run per level because under parallel execution the
# duplicates cost no wall-clock, and copying artifacts between concurrently
# running processes would be a race.
JOBS=$(mktemp)
for c in $CENSORING; do
    for a in $ARMS; do
        echo "$c $a" >> "$JOBS"
    done
done
N=$(wc -l < "$JOBS" | tr -d ' ')

sweep_banner "E2 sweep - censoring x CL arm" "$N"     "censoring:$CENSORING" "arms:$ARMS" "logs:$LOGDIR"
sweep_progress_init "$N"
trap 'rm -f "$JOBS"; sweep_progress_cleanup' EXIT

# The inner shell swallows the exit status deliberately: xargs aborts the whole
# batch if a child returns 255, and one bad cell must not cancel the cells still
# queued. A partial sweep is recoverable by re-running; a cancelled one is not.
xargs -P "$CONCURRENCY" -L1 bash -c '
    set -- $0 $@
    c=$1; a=$2
    source experiments/sweep_common.sh
    sweep_run_cell "$c  x  $a" "'"$LOGDIR"'/${c}_${a}.log"         python -m experiments.exp_staleness_cl             --censoring "$c" --arms "$a" --out "'"$OUT"'"
' < "$JOBS"

# Each cell process wrote only its own arm, so no single run built the
# cross-arm tables. This final pass reads every cell off disk and assembles them.
echo "Building summary across all cells..."
python -m experiments.exp_staleness_cl     --censoring $CENSORING --arms $ARMS --out "$OUT"

sweep_footer "$LOGDIR" "outputs/drift/results/$OUT/${OUT}_summary.csv"
