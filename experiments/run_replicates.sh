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

source experiments/sweep_common.sh
sweep_setup_env
mkdir -p "$LOGDIR"

if ! python -m experiments.vast_preflight --concurrency "$CONCURRENCY"; then
    echo "Preflight failed - not starting." >&2
    exit 1
fi

JOBS=$(mktemp)
for s in $SEEDS; do
    for a in $ARMS; do
        echo "$s $a" >> "$JOBS"
    done
done
N=$(wc -l < "$JOBS" | tr -d ' ')

sweep_banner "Seed replicates" "$N" \
    "censoring:$CENSORING (one level, deliberately)" \
    "seeds:$SEEDS" \
    "arms:$ARMS" \
    "logs:$LOGDIR"
sweep_progress_init "$N"
trap 'rm -f "$JOBS"; sweep_progress_cleanup' EXIT

xargs -P "$CONCURRENCY" -L1 bash -c '
    set -- $0 $@
    s=$1; a=$2
    source experiments/sweep_common.sh
    sweep_run_cell "seed=$s  $a" "'"$LOGDIR"'/s${s}_${a}.log" \
        python -m experiments.exp_staleness_cl \
            --censoring "'"$CENSORING"'" --arms "$a" --seed "$s" --out "rep_s$s"
' < "$JOBS"

# Each cell process ran ONE arm, so its build_tables call only ever saw that arm
# and skipped the tables for want of the `frozen` anchor. Without this pass the
# run finishes with every probe score on disk and metrics_efficiency.csv holding
# frozen alone - all the compute done, none of it aggregated.
echo "Building cross-arm tables for each seed..."
for s in $SEEDS; do
    python -m experiments.exp_staleness_cl \
        --censoring "$CENSORING" --arms $ARMS --seed "$s" --out "rep_s$s" \
        > "$LOGDIR/aggregate_s${s}.log" 2>&1 \
        && echo "  seed $s ok" || echo "  seed $s FAILED - see $LOGDIR/aggregate_s${s}.log"
done

sweep_footer "$LOGDIR" "python -m experiments.compare_replicates"
