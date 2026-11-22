#!/usr/bin/env bash
# Regenerate every committed result that does NOT depend on which base
# forecaster is the baseline, and copy it into results/ (committed) with a
# run_meta.json. The tables these steps produced before were lost with
# outputs/ in the laptop loss.
#
#   bash experiments/run_model_free.sh            # all steps
#   STEPS="e1 audit" bash experiments/run_model_free.sh
#
# Steps, in order:
#   e1     E1 sync sweep: 3 policies x 4 delays on Redis, static prices, no models
#   audit  synthetic-layer honesty audit (data only)
#   e4     E4 static memory accounting: bytes per CL structure (the architecture
#          is identical for the original and rebuilt TFT, so so are the bytes)
#   tests  the full test suite, including the slow real-data ones CI skips
# E2/E5 are deliberately not here: they depend on the baseline decision.
set -uo pipefail
cd "$(dirname "$0")/.."
PY=${PY:-.venv/Scripts/python.exe}
STEPS=${STEPS:-"e1 audit e4 tests"}
export PYTHONIOENCODING=utf-8 PYTHONPATH=.
LOG=outputs/model_free_logs; mkdir -p "$LOG"

stamp() {  # stamp <results subdir> <json extra>
    "$PY" -c "
import json, sys
from pathlib import Path
from experiments.provenance import write_run_meta
write_run_meta(Path(sys.argv[1]), data_dir=Path(sys.argv[2]), extra=json.loads(sys.argv[3]))
" "$1" "$2" "$3"
}

step() { echo "=== $(date '+%F %T') $1"; }
ok=0; fail=0
for s in $STEPS; do
    case $s in
    e1)
        step "E1 sync sweep"
        if "$PY" -m experiments.exp_sync > "$LOG/e1.log" 2>&1; then
            mkdir -p results/e1_sync
            cp outputs/system/results/e1_sync.csv outputs/system/results/e1_sync_config.json results/e1_sync/
            stamp results/e1_sync data/processed_m5_v4 '{"experiment":"E1","backend":"redis"}'
            ok=$((ok+1)); tail -3 "$LOG/e1.log"
        else fail=$((fail+1)); echo "E1 FAILED"; tail -20 "$LOG/e1.log"; fi ;;
    audit)
        step "synthetic audit"
        if "$PY" -m experiments.exp_synthetic_audit --out results/synthetic_audit > "$LOG/audit.log" 2>&1; then
            stamp results/synthetic_audit data/processed_m5_v3 '{"experiment":"synthetic_audit"}'
            ok=$((ok+1)); tail -12 "$LOG/audit.log"
        else fail=$((fail+1)); echo "audit FAILED"; tail -20 "$LOG/audit.log"; fi ;;
    e4)
        step "E4 static memory"
        if "$PY" -m experiments.exp_memory --static > "$LOG/e4.log" 2>&1; then
            mkdir -p results/e4_memory
            cp outputs/drift/results/e4_memory_static.csv results/e4_memory/
            stamp results/e4_memory data/processed_m5_v4 '{"experiment":"E4","mode":"static"}'
            ok=$((ok+1)); tail -15 "$LOG/e4.log"
        else fail=$((fail+1)); echo "E4 FAILED"; tail -20 "$LOG/e4.log"; fi ;;
    tests)
        step "full test suite (slow, real data)"
        if "$PY" -m pytest edge_system drift_pipeline -q -p no:cacheprovider -rs > "$LOG/tests.log" 2>&1; then
            ok=$((ok+1))
        else fail=$((fail+1)); echo "TESTS FAILED"; grep -E "^FAILED|^ERROR" "$LOG/tests.log" | head -20; fi
        tail -1 "$LOG/tests.log" ;;
    *) echo "unknown step $s"; fail=$((fail+1)) ;;
    esac
done
step "done: $ok ok, $fail failed"
[ "$fail" -eq 0 ]
