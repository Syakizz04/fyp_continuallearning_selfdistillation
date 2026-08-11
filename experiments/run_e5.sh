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

source experiments/sweep_common.sh
sweep_setup_env
mkdir -p "$LOGDIR"

if ! python -m experiments.vast_preflight --concurrency "$CONCURRENCY"; then
    echo "Preflight failed - not starting." >&2
    exit 1
fi

# Fail before spending rental time if E2's frozen anchor is not on disk: without
# it every cell would run to completion and then produce no forgetting number.
if ! python -m experiments.exp_sdft_alpha --censoring $CENSORING --alphas $ALPHAS         --out "$OUT" --dry-run; then
    echo "Dry run failed - not starting." >&2
    exit 1
fi

JOBS=$(mktemp)
for c in $CENSORING; do
    for a in $ALPHAS; do
        echo "$c $a" >> "$JOBS"
    done
done
N=$(wc -l < "$JOBS" | tr -d ' ')

sweep_banner "E5 sweep - SDFT alpha" "$N"     "censoring:$CENSORING" "alphas:$ALPHAS (0.5 is E2's baseline)" "logs:$LOGDIR"
sweep_progress_init "$N"
trap 'rm -f "$JOBS"; sweep_progress_cleanup' EXIT

xargs -P "$CONCURRENCY" -L1 bash -c '
    set -- $0 $@
    c=$1; a=$2
    source experiments/sweep_common.sh
    sweep_run_cell "$c  alpha=$a" "'"$LOGDIR"'/${c}_alpha${a}.log"         python -m experiments.exp_sdft_alpha             --censoring "$c" --alphas "$a" --out "'"$OUT"'"
' < "$JOBS"

# Each cell process wrote only its own alpha; this pass assembles the table.
echo "Building summary across all cells..."
python -m experiments.exp_sdft_alpha --censoring $CENSORING --alphas $ALPHAS --out "$OUT"

sweep_footer "$LOGDIR" "outputs/drift/results/$OUT/${OUT}_summary.csv"
