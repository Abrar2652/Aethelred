#!/usr/bin/env bash
# =============================================================================
#  run_all.sh — Full Aethelred experiment runner
#
#  Runs every active table and figure sequentially.
#  Streams real-time output to the terminal AND to a timestamped log file.
#  A failed step is logged and skipped — the run continues to the next step.
#
#  Usage:
#    bash run_all.sh              # uses cached checkpoints where available
#    bash run_all.sh --retrain    # force-retrain all models from scratch
#
#  tmux (recommended):
#    tmux new-session -s aethelred
#    bash run_all.sh 2>&1 | tee logs/run_LIVE.log
#    # In a second pane: tail -f logs/run_LIVE.log
#
#  Log location:  logs/run_YYYYMMDD_HHMMSS.log
#  Results:       results/
# =============================================================================
set -uo pipefail   # no -e: step failures are caught individually

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV="${SCRIPT_DIR}/venv/bin/python"
RUNNER="${SCRIPT_DIR}/run_aethelred_comparison.py"
LOG_DIR="${SCRIPT_DIR}/logs"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="${LOG_DIR}/run_${TIMESTAMP}.log"

# ── Parse flags ─────────────────────────────────────────────────────────────
RETRAIN_FLAG=""
for arg in "$@"; do
    case "${arg}" in
        --retrain) RETRAIN_FLAG="--force_retrain" ;;
        *)
            echo "Unknown argument: ${arg}"
            echo "Usage: bash run_all.sh [--retrain]"
            exit 1
            ;;
    esac
done

mkdir -p "${LOG_DIR}"
export PYTHONUNBUFFERED=1   # force line-buffered Python output for real-time tee

# ── Counters ────────────────────────────────────────────────────────────────
PASS=0
FAIL=0
FAILED_STEPS=()
START_EPOCH=$(date +%s)

# ── Helpers ─────────────────────────────────────────────────────────────────
divider() { printf '%0.s═' {1..80}; echo ""; }

header() {
    echo ""
    divider
    printf '  %-64s %s\n' "$1" "$(date '+%H:%M:%S')"
    divider
    echo ""
}

elapsed() {
    local now; now=$(date +%s)
    local secs=$(( now - START_EPOCH ))
    printf '%02dh %02dm %02ds' $(( secs/3600 )) $(( (secs%3600)/60 )) $(( secs%60 ))
}

run_step() {
    local label="$1"; shift
    local step_start; step_start=$(date +%s)

    header "▶  ${label}"

    local exit_code=0
    # shellcheck disable=SC2086
    "${VENV}" "${RUNNER}" "$@" ${RETRAIN_FLAG} || exit_code=$?

    local step_secs=$(( $(date +%s) - step_start ))
    local step_time; step_time=$(printf '%02dm %02ds' $(( step_secs/60 )) $(( step_secs%60 )))

    if [[ ${exit_code} -eq 0 ]]; then
        header "✔  ${label}  [${step_time}]"
        PASS=$(( PASS + 1 ))
    else
        header "✘  ${label}  [${step_time}]  — exit code ${exit_code}"
        FAIL=$(( FAIL + 1 ))
        FAILED_STEPS+=("${label}")
    fi
}

# ── Main run (all output piped through tee) ──────────────────────────────────
{

header "AETHELRED — FULL EXPERIMENT RUN   [${TIMESTAMP}]"
echo "  Log file   : ${LOG_FILE}"
echo "  Results    : ${SCRIPT_DIR}/results/"
echo "  Retrain    : ${RETRAIN_FLAG:-'no (cache respected)'}"
echo ""

# ── Tables ───────────────────────────────────────────────────────────────────
run_step "Table 1.1  Clean GNN Baseline (3 seeds)"        --table 1.1  --n_seeds 3
run_step "Table 1.2  Aethelred GCN Clean Acc (3 seeds)"   --table 1.2  --n_seeds 3
run_step "Table 1.3  Aethelred GSAGE Clean Acc (3 seeds)" --table 1.3  --n_seeds 3
run_step "Table 1.4  Aethelred GAT Clean Acc (3 seeds)"   --table 1.4  --n_seeds 3
run_step "Table 4    PGD Robustness (3 seeds)"            --table 4    --n_seeds 3
run_step "Table 6    Explanation Quality"                 --table 6
run_step "Table 7    Adaptive-Attack Stress Test (3 seeds)" --table 7  --n_seeds 3
run_step "Table 8    Ablation Study (3 seeds)"            --table 8    --ablation_n_seeds 3
run_step "Table 9    MUTAG Explanation Faithfulness"      --table expl_gt

# ── Figures ──────────────────────────────────────────────────────────────────
run_step "Figure 2   Certification Radius Sweep"          --figure 2
run_step "Figure 3   Hyperparameter Sensitivity"          --figure 3
run_step "Figure 4   Causal Visualization"                --figure 4

# ── Final summary ────────────────────────────────────────────────────────────
header "SUMMARY  [total elapsed: $(elapsed)]"
echo "  Passed : ${PASS} / $(( PASS + FAIL ))"
echo "  Failed : ${FAIL} / $(( PASS + FAIL ))"
if [[ ${FAIL} -gt 0 ]]; then
    echo ""
    echo "  Failed steps:"
    for s in "${FAILED_STEPS[@]}"; do
        echo "    ✘  ${s}"
    done
fi
echo ""
echo "  Log    : ${LOG_FILE}"
echo "  Results: ${SCRIPT_DIR}/results/"
divider
echo ""

} 2>&1 | tee "${LOG_FILE}"
