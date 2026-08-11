#!/usr/bin/env bash
# Seed replicates at ONE censoring level - the run that decides whether an arm
# gap is real.
#
#   bash experiments/run_replicates.sh
#   SEEDS="42 123 7 2024 99" CONCURRENCY=6 bash experiments/run_replicates.sh
#   CENSORING=none bash experiments/run_replicates.sh
#
# Why this exists: retraining fires on a threshold crossing, and a GPU fit is not
# bit-reproducible even against itself (~2e-6 across 533 tensors). A 1e-6
# difference can flip whether MASE breaches mu+2*sigma twice running, changing
# the retrain schedule and everything after it. A single run therefore cannot
# rank arms - E2 and the FYP1 drift run disagreed about whether SDFT adapts
# (walk MASE 1.415 vs 1.074) on a byte-identical forecaster and dataset.
#
# --seed drives per-retrain seeding as well as the censoring draw, so each seed
# is a genuinely independent replicate rather than only a different censor mask.
#
# One level, several seeds - not several levels, one seed. E2 already showed the
# censoring effect is smaller than the run-to-run spread, so spending the same
# ~12 cells on replication answers a question that spending them on doses does
# not.
#
# `frozen` is run at every seed even though it never trains and is therefore
# seed-invariant: it is needed in each seed's directory as the forgetting
# anchor, the duplicates cost no wall-clock under parallel execution, and the
# three results agreeing to six decimals is a free determinism check.

set -uo pipefail
cd "$(dirname "$0")/.."

CENSORING=${CENSORING:-escrow_quota}
SEEDS=${SEEDS:-"42 123 7"}
ARMS=${ARMS:-"frozen naive replay sdft"}
CONCURRENCY=${CONCURRENCY:-6}
LOGDIR=${LOGDIR:-outputs/drift/logs/rep}

export FYP_NUM_WORKERS=${FYP_NUM_WORKERS:-2}
# See run_e2.sh: N cells x (cores) threads thrashed an 8-core box to a standstill.
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-1}
export PYTHONIOENCODING=utf-8
export PYTHONUNBUFFERED=1
export FYP_CHECKPOINT_EVERY=${FYP_CHECKPOINT_EVERY:-20}

mkdir -p "$LOGDIR"

echo "Seed replicates"
echo "  censoring   : $CENSORING   (one level, deliberately)"
echo "  seeds       : $SEEDS"
echo "  arms        : $ARMS"
echo "  concurrency : $CONCURRENCY"
echo

if ! python -m experiments.vast_preflight --concurrency "$CONCURRENCY"; then
    echo "Preflight failed - not starting." >&2
    exit 1
fi

JOBS=$(mktemp)
trap 'rm -f "$JOBS"' EXIT
for s in $SEEDS; do
    for a in $ARMS; do
        echo "$s $a" >> "$JOBS"
    done
done
echo "$(wc -l < "$JOBS") cells queued"
echo

START=$(date +%s)

xargs -P "$CONCURRENCY" -L1 bash -c '
    set -- $0 $@
    s=$1; a=$2
    echo "[start] seed=$s $a"
    if python -m experiments.exp_staleness_cl \
            --censoring "'"$CENSORING"'" --arms "$a" --seed "$s" --out "rep_s$s" \
            > "'"$LOGDIR"'/s${s}_${a}.log" 2>&1; then
        echo "[done ] seed=$s $a"
    else
        echo "[FAIL ] seed=$s $a  -- see '"$LOGDIR"'/s${s}_${a}.log"
    fi
' < "$JOBS"

echo
echo "elapsed: $(( ($(date +%s) - START) / 60 )) min"
echo
echo "Read it with:  python -m experiments.compare_replicates"
