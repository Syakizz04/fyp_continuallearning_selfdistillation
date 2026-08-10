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

# Default 0 for notebook/fork safety on Windows; a Linux box with spare cores
# should use them, but not so many that N cells x M workers oversubscribes.
export FYP_NUM_WORKERS=${FYP_NUM_WORKERS:-2}

# Torch sizes its intra-op thread pool from the PHYSICAL CORE COUNT of the whole
# box, with no idea that N-1 sibling cells are doing the same. On an 8-core box
# at CONCURRENCY=6 that is 6 x 8 = 48 runnable threads over 8 cores - measured,
# not theorised: load average sat at 48.4 while the GPU idled at 0% and no cell
# finished in 11 hours.
#
# One thread per process is the right cap rather than a smaller multiple,
# because the tensors here are small (the TFT is ~4 MB, PPO acts on batch-1
# observations) and intra-op threading on tensors that size loses more to
# thread-sync overhead than it wins. Capping is therefore expected to make each
# cell faster on its own, not merely stop it fighting its siblings. The real
# parallelism is across cells, which is what CONCURRENCY already controls.
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-1}
export PYTHONIOENCODING=utf-8
export PYTHONUNBUFFERED=1

# Mid-walk checkpointing. Cell-level resume alone means a cell killed at check
# 150 of 157 costs as much as one killed at check 2, which is what an 11-hour
# loss to a stray Ctrl-C actually bought. Every 20 checks is ~8 snapshots per
# cell; the replay arm's is the expensive one (~3 GB) and the rest are small.
# Set 0 to disable. State lives in outputs/drift/results/<cell>/walk_state_<arm>/
# and survives a vast.ai STOP but not a DESTROY - copy it off before destroying.
export FYP_CHECKPOINT_EVERY=${FYP_CHECKPOINT_EVERY:-20}

mkdir -p "$LOGDIR"

echo "E2 sweep"
echo "  censoring   : $CENSORING"
echo "  arms        : $ARMS"
echo "  concurrency : $CONCURRENCY"
echo "  logs        : $LOGDIR"
echo

if ! python -m experiments.vast_preflight --concurrency "$CONCURRENCY"; then
    echo "Preflight failed - not starting." >&2
    exit 1
fi

# Build the job list. `frozen` never retrains, so it is genuinely identical at
# every censoring level; it is still run per level because under parallel
# execution the duplicates cost no wall-clock, and copying artifacts between
# concurrently-running processes would be a race.
JOBS=$(mktemp)
trap 'rm -f "$JOBS"' EXIT
for c in $CENSORING; do
    for a in $ARMS; do
        echo "$c $a" >> "$JOBS"
    done
done
echo "$(wc -l < "$JOBS") cells queued"
echo

START=$(date +%s)

# The inner shell swallows the exit status deliberately: xargs aborts the whole
# batch if a child returns 255, and one bad cell must not cancel the cells still
# queued. A partial sweep is recoverable by re-running; a cancelled one is not.
xargs -P "$CONCURRENCY" -L1 bash -c '
    set -- $0 $@
    c=$1; a=$2
    echo "[start] $c x $a"
    if python -m experiments.exp_staleness_cl \
            --censoring "$c" --arms "$a" --out "'"$OUT"'" \
            > "'"$LOGDIR"'/${c}_${a}.log" 2>&1; then
        echo "[done ] $c x $a"
    else
        echo "[FAIL ] $c x $a  -- see '"$LOGDIR"'/${c}_${a}.log"
    fi
' < "$JOBS"

echo
echo "elapsed: $(( ($(date +%s) - START) / 60 )) min"

# Each cell process wrote only its own arm, so no single run built the
# cross-arm tables. This final pass reads every cell off disk and assembles them.
echo "Building summary across all cells..."
python -m experiments.exp_staleness_cl \
    --censoring $CENSORING --arms $ARMS --out "$OUT"

echo
echo "Results: outputs/drift/results/$OUT/${OUT}_summary.csv"
