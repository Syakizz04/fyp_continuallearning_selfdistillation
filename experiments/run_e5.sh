#!/usr/bin/env bash
# E5 alpha-sweep launcher. Same shape as run_e2.sh - see that file for why the
# thread caps and the resumability matter; both were paid for the hard way.
#
#   bash experiments/run_e5.sh
#   ALPHAS="0.7 0.85 0.95" CENSORING="none escrow_quota" bash experiments/run_e5.sh
#   CONCURRENCY=6 bash experiments/run_e5.sh
#
# Each (censoring, alpha) cell is a separate process with its own results
# directory and its own base load, so the sweep parallelises at cell granularity.

set -uo pipefail
cd "$(dirname "$0")/.."

CENSORING=${CENSORING:-"none escrow_quota"}
ALPHAS=${ALPHAS:-"0.7 0.85 0.95"}
CONCURRENCY=${CONCURRENCY:-6}
OUT=${OUT:-e5}
LOGDIR=${LOGDIR:-outputs/drift/logs/e5}

export FYP_NUM_WORKERS=${FYP_NUM_WORKERS:-2}

# Torch sizes its intra-op pool from the box's physical core count with no idea
# that sibling cells are doing the same; N cells x cores threads thrashed an
# 8-core box to a standstill (load 48, GPU idle, nothing finished in 11 hours).
# The tensors here are small enough that intra-op threading loses more to
# thread-sync overhead than it wins, so one thread per process is both faster
# per cell and contention-free. Parallelism comes from CONCURRENCY.
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-1}
export PYTHONIOENCODING=utf-8
export PYTHONUNBUFFERED=1

# Mid-walk checkpointing: a cell killed at check 150 of 157 otherwise costs as
# much as one killed at check 2.
export FYP_CHECKPOINT_EVERY=${FYP_CHECKPOINT_EVERY:-20}

mkdir -p "$LOGDIR"

echo "E5 alpha sweep"
echo "  censoring   : $CENSORING"
echo "  alphas      : $ALPHAS   (0.5 is E2's baseline, not re-run)"
echo "  concurrency : $CONCURRENCY"
echo "  logs        : $LOGDIR"
echo

if ! python -m experiments.vast_preflight --concurrency "$CONCURRENCY"; then
    echo "Preflight failed - not starting." >&2
    exit 1
fi

# Fail before renting time on it if E2's frozen anchor is not on disk: without
# it every cell would run to completion and then produce no forgetting number.
if ! python -m experiments.exp_sdft_alpha --censoring $CENSORING --alphas $ALPHAS \
        --out "$OUT" --dry-run; then
    echo "Dry run failed - not starting." >&2
    exit 1
fi

JOBS=$(mktemp)
trap 'rm -f "$JOBS"' EXIT
for c in $CENSORING; do
    for a in $ALPHAS; do
        echo "$c $a" >> "$JOBS"
    done
done
echo "$(wc -l < "$JOBS") cells queued"
echo

START=$(date +%s)

# The inner shell swallows the exit status deliberately: xargs aborts the whole
# batch on a 255, and one bad cell must not cancel the cells still queued.
xargs -P "$CONCURRENCY" -L1 bash -c '
    set -- $0 $@
    c=$1; a=$2
    echo "[start] $c x alpha=$a"
    if python -m experiments.exp_sdft_alpha \
            --censoring "$c" --alphas "$a" --out "'"$OUT"'" \
            > "'"$LOGDIR"'/${c}_alpha${a}.log" 2>&1; then
        echo "[done ] $c x alpha=$a"
    else
        echo "[FAIL ] $c x alpha=$a  -- see '"$LOGDIR"'/${c}_alpha${a}.log"
    fi
' < "$JOBS"

echo
echo "elapsed: $(( ($(date +%s) - START) / 60 )) min"

# Each cell process wrote only its own alpha, so no single run built the
# cross-cell table. This final pass reads every cell off disk and assembles it.
echo "Building summary across all cells..."
python -m experiments.exp_sdft_alpha --censoring $CENSORING --alphas $ALPHAS --out "$OUT"

echo
echo "Results: outputs/drift/results/$OUT/${OUT}_summary.csv"
