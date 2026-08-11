#!/usr/bin/env bash
# Shared launcher plumbing for the sweeps (run_e2.sh, run_e5.sh,
# run_replicates.sh). Sourced, never executed.
#
# The three launchers had drifted into three copies of the same env exports and
# three slightly different progress formats, which mattered more than it sounds:
# the console is the only place you see a sweep as a whole, and "[done] x" told
# you nothing about how long a cell took, how many were left, or - the case that
# actually cost a night - WHY one failed.

# ── Environment every sweep needs ────────────────────────────────────────────
sweep_setup_env() {
    export FYP_NUM_WORKERS=${FYP_NUM_WORKERS:-2}

    # Torch sizes its intra-op pool from the box's physical core count with no
    # idea that sibling cells are doing the same, so N cells claim N x cores
    # threads. On an 8-core box at CONCURRENCY=6 that measured load average 48.4
    # with the GPU idle at 0% and not one cell finishing in 11 hours. These
    # tensors are small enough that intra-op threading loses more to thread-sync
    # overhead than it wins; the parallelism that pays is across cells.
    export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}
    export MKL_NUM_THREADS=${MKL_NUM_THREADS:-1}

    export PYTHONIOENCODING=utf-8
    export PYTHONUNBUFFERED=1

    # Mid-walk checkpointing: without it a cell killed at check 150 of 157 costs
    # exactly as much as one killed at check 2.
    export FYP_CHECKPOINT_EVERY=${FYP_CHECKPOINT_EVERY:-20}
}

sweep_banner() {                       # $1 title, $2 total cells, rest: "k: v"
    local title="$1" total="$2"; shift 2
    printf '\n%s\n' "$title"
    printf '  %-12s %s\n' "cells" "$total"
    printf '  %-12s %s\n' "concurrency" "${CONCURRENCY}"
    printf '  %-12s %s\n' "threads/cell" "${OMP_NUM_THREADS} (torch intra-op)"
    printf '  %-12s %s\n' "checkpoint" "every ${FYP_CHECKPOINT_EVERY} checks"
    while [ $# -gt 0 ]; do printf '  %-12s %s\n' "${1%%:*}" "${1#*:}"; shift; done
    printf '  %-12s %s\n' "started" "$(date '+%Y-%m-%d %H:%M:%S')"
    printf '\n'
}

# Progress is counted through a file rather than a variable because every cell
# runs in its own xargs child - a shell variable would be incremented in a
# subshell and lost. Appends of a few bytes are atomic, so `wc -l` is a safe
# counter under concurrency.
sweep_progress_init() {
    SWEEP_PROGRESS="$(mktemp)"; SWEEP_TOTAL="$1"
    export SWEEP_PROGRESS SWEEP_TOTAL SWEEP_START
    SWEEP_START=$(date +%s)
}

sweep_progress_cleanup() { rm -f "$SWEEP_PROGRESS"; }

#: Named explicitly because both have actually happened: a stray Ctrl-C killed a
#: sweep, and six concurrent SDFT cells (student + frozen teacher each) exhausted
#: a 15 GB card. A bare "FAIL" sent you digging through logs for either.
sweep_failure_reason() {
    local log="$1"
    if   grep -qi "out of memory"      "$log"; then echo "CUDA OOM"
    elif grep -q  "KeyboardInterrupt"  "$log"; then echo "interrupted (Ctrl-C)"
    elif grep -q  "No space left"      "$log"; then echo "disk full"
    else grep -oE "[A-Za-z_.]*(Error|Exception): .*" "$log" | tail -1 | cut -c1-70
    fi
}

# sweep_run_cell <label> <logfile> <command...>
sweep_run_cell() {
    local label="$1" log="$2"; shift 2
    local t0 dt mins k line
    t0=$(date +%s)
    printf '[%s] start   %s\n' "$(date +%H:%M:%S)" "$label"

    if "$@" > "$log" 2>&1; then line="done "; else line="FAIL "; fi

    dt=$(( $(date +%s) - t0 ))
    mins=$(awk "BEGIN{printf \"%.1f\", $dt/60}")
    # Append-then-count under a lock: without it two cells finishing in the same
    # instant both read the same total and print the same "(k/N)".
    if command -v flock >/dev/null 2>&1; then
        { flock 9
          echo x >> "$SWEEP_PROGRESS"
          k=$(wc -l < "$SWEEP_PROGRESS" | tr -d ' ')
        } 9>"${SWEEP_PROGRESS}.lock"
    else
        echo x >> "$SWEEP_PROGRESS"
        k=$(wc -l < "$SWEEP_PROGRESS" | tr -d ' ')
    fi

    if [ "$line" = "done " ]; then
        # The cell's own process prints its retrain counts; surface them here so
        # the console shows what a cell DID, not merely that it stopped.
        local detail
        detail=$(grep -oE '\([0-9]+ FC \+ [0-9]+ RL retrains\)' "$log" | tail -1)
        printf '[%s] done    %-34s %6s min  (%s/%s)  %s\n' \
               "$(date +%H:%M:%S)" "$label" "$mins" "$k" "$SWEEP_TOTAL" "${detail:-}"
    else
        printf '[%s] FAIL    %-34s %6s min  (%s/%s)  %s\n' \
               "$(date +%H:%M:%S)" "$label" "$mins" "$k" "$SWEEP_TOTAL" \
               "$(sweep_failure_reason "$log")"
        printf '                 -> %s\n' "$log"
    fi
}
export -f sweep_run_cell sweep_failure_reason

sweep_footer() {                        # $1 logdir, $2 results path
    local elapsed done_n fail_n
    elapsed=$(( ($(date +%s) - SWEEP_START) / 60 ))
    done_n=$(wc -l < "$SWEEP_PROGRESS" | tr -d ' ')
    fail_n=$(grep -lE "out of memory|KeyboardInterrupt|Error:" "$1"/*.log 2>/dev/null | wc -l | tr -d ' ')
    printf '\n%s\n' "$(printf '%.0s-' {1..72})"
    printf '  %s/%s cells finished · %s failed · elapsed %s min\n' \
           "$done_n" "$SWEEP_TOTAL" "$fail_n" "$elapsed"
    [ "$fail_n" -gt 0 ] && printf '  re-run the same command to resume failed cells (do NOT pass --force)\n'
    printf '  results: %s\n\n' "$2"
}
