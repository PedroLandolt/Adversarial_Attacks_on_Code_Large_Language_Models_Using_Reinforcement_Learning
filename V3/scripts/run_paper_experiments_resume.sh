#!/usr/bin/env bash
# Resume script for the paper experiment pipeline.
# MBPP seeds 1 and 2 are already complete. This script runs:
#   - MBPP seed 3 (5 epochs training + validation + test)
#   - MBPP baselines (3x random + 3x agent)
#   - HumanEval seeds 1-3 (5 epochs + validation + test each)
#   - HumanEval baselines (3x random + 3x agent)
#   - Aggregate + plot
#
# Usage:
#   bash V3/scripts/run_paper_experiments_resume.sh
#
# Background:
#   nohup bash V3/scripts/run_paper_experiments_resume.sh > /dev/null 2>&1 &

set -euo pipefail

# -----------------------------------------------------------
# Fixed settings (match original run)
# -----------------------------------------------------------
EPOCHS=5
REPEATS=3
MAX_ITERATIONS="${MAX_ITERATIONS:-12}"
MODEL="${MODEL:-ollama/llama3.1:8b}"
JUDGE_MODEL="${JUDGE_MODEL:-$MODEL}"
SELECTOR_MODEL="${SELECTOR_MODEL:-$JUDGE_MODEL}"
EPOCH_TIMEOUT_SECS="${EPOCH_TIMEOUT_SECS:-86400}"

# -----------------------------------------------------------
# Paths — same venv-detection logic as run_paper_experiments.sh
# -----------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJECT_ROOT"

if [[ -f ".venv/Scripts/inspect" ]]; then
    INSPECT_BIN=".venv/Scripts/inspect"
elif [[ -f ".venv/bin/inspect" ]]; then
    INSPECT_BIN=".venv/bin/inspect"
elif [[ -f "../.venv/Scripts/inspect" ]]; then
    INSPECT_BIN="../.venv/Scripts/inspect"
elif [[ -f "../.venv/bin/inspect" ]]; then
    INSPECT_BIN="../.venv/bin/inspect"
else
    INSPECT_BIN="inspect"
fi

if [[ -f ".venv/Scripts/python.exe" ]]; then
    PYTHON_BIN=".venv/Scripts/python.exe"
elif [[ -f ".venv/bin/python" ]]; then
    PYTHON_BIN=".venv/bin/python"
elif [[ -f "../.venv/Scripts/python.exe" ]]; then
    PYTHON_BIN="../.venv/Scripts/python.exe"
elif [[ -f "../.venv/bin/python" ]]; then
    PYTHON_BIN="../.venv/bin/python"
else
    PYTHON_BIN="python"
fi

# -----------------------------------------------------------
# Session log
# -----------------------------------------------------------
SESSION="$(date +%Y%m%d_%H%M%S)"
LOG_DIR="logs/paper_experiment"
mkdir -p "$LOG_DIR"
LOG_FILE="${LOG_DIR}/resume_${SESSION}.log"

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOG_FILE"; }

# -----------------------------------------------------------
# Load .env
# -----------------------------------------------------------
if [[ -f ".env" ]]; then
    set -a; source ".env"; set +a
elif [[ -f "V3/.env" ]]; then
    set -a; source "V3/.env"; set +a
fi

log "========================================================"
log "Paper experiment pipeline — RESUME RUN"
log "  Skipping: mbpp seeds 1 and 2 (already complete)"
log "  Running:  mbpp seed 3 + mbpp baselines"
log "            humaneval seeds 1-3 + humaneval baselines"
log "  epochs/seed: $EPOCHS"
log "  repeats:     $REPEATS (random + agent baselines)"
log "  max iter:    $MAX_ITERATIONS"
log "  model:       $MODEL"
log "  session:     $SESSION"
log "  log:         $LOG_FILE"
log "========================================================"

# -----------------------------------------------------------
# Helpers
# -----------------------------------------------------------

run_inspect() {
    local label="$1"; shift
    log "--- $label ---"
    PYTHONPATH=V3 "$INSPECT_BIN" eval V3/adversarial_attack.py@adversarial_code_llm \
      --model "$MODEL" \
      -T benchmark="$BENCHMARK" \
      -T mutation_strategy=react \
      -T experiment_mode=iterative \
      -T use_llm_judge=True \
      -T judge_model="$JUDGE_MODEL" \
      -T selector_model="$SELECTOR_MODEL" \
      -T max_iterations="$MAX_ITERATIONS" \
      "$@" 2>&1 | tee -a "$LOG_FILE"
}

init_seed_weights() {
    local path="$1"
    "$PYTHON_BIN" -c "
import json, sys
json.dump({'pull_counts': {}, 'cumulative_rewards': {}}, open(sys.argv[1], 'w'), indent=2)
" "$path"
    log "Initialized fresh weights -> $path"
}

run_seed() {
    local BM="$1"
    local SEED="$2"
    local TRAIN_N="$3"
    local TRAIN_DEF="$4"
    local VAL_N="$5"
    local VAL_DEF="$6"
    local VAL_RANGE="$7"
    local TEST_N="$8"
    local TEST_DEF="$9"
    local TEST_RANGE="${10}"

    log "===== $BM  seed $SEED ====="

    SEED_WEIGHTS="$("$PYTHON_BIN" -c "import os; print(os.path.abspath('weights/${BM}_ucb1_seed${SEED}.json'))")"
    init_seed_weights "$SEED_WEIGHTS"

    TRAINED_WEIGHTS="$SEED_WEIGHTS"

    for EPOCH in $(seq 1 "$EPOCHS"); do
        log "--- $BM  seed $SEED  epoch $EPOCH / $EPOCHS ---"
        PYTHONPATH=V3 timeout "$EPOCH_TIMEOUT_SECS" "$INSPECT_BIN" eval \
            V3/adversarial_attack.py@adversarial_code_llm \
            --model "$MODEL" \
            -T benchmark="$BM" \
            -T mutation_strategy=react \
            -T experiment_mode=iterative \
            -T use_llm_judge=True \
            -T judge_model="$JUDGE_MODEL" \
            -T selector_model="$SELECTOR_MODEL" \
            -T max_iterations="$MAX_ITERATIONS" \
            -T policy_mode=rl_bandit \
            -T bandit_algorithm=ucb1 \
            -T bandit_weights_path="$SEED_WEIGHTS" \
            -T bandit_freeze_weights=False \
            -T experiment_split=train \
            -T split_definition="$TRAIN_DEF" \
            --max-samples "$TRAIN_N" \
            --limit "$TRAIN_N" \
            2>&1 | tee -a "$LOG_FILE" \
        || log "WARNING: $BM seed $SEED epoch $EPOCH returned non-zero. Checkpoint saved."

        CKPT="${SEED_WEIGHTS%.json}_epoch_$(printf '%03d' "$EPOCH").json"
        cp "$SEED_WEIGHTS" "$CKPT"
        TRAINED_WEIGHTS="$CKPT"
        log "Checkpoint: $CKPT"
        docker rm -f $(docker ps -q --filter "ancestor=aisiuk/inspect-tool-support") 2>/dev/null || true
        log "Sandbox containers cleaned up after epoch $EPOCH."
    done

    BENCHMARK="$BM"
    run_inspect "$BM seed $SEED — RL validation (frozen)" \
      -T policy_mode=rl_bandit \
      -T bandit_algorithm=ucb1 \
      -T bandit_weights_path="$TRAINED_WEIGHTS" \
      -T bandit_freeze_weights=True \
      -T experiment_split=validation \
      -T split_definition="$VAL_DEF" \
      --max-samples "$VAL_N" \
      --limit "$VAL_RANGE"

    run_inspect "$BM seed $SEED — RL test (frozen)" \
      -T policy_mode=rl_bandit \
      -T bandit_algorithm=ucb1 \
      -T bandit_weights_path="$TRAINED_WEIGHTS" \
      -T bandit_freeze_weights=True \
      -T experiment_split=test \
      -T split_definition="$TEST_DEF" \
      --max-samples "$TEST_N" \
      --limit "$TEST_RANGE"

    log "===== $BM seed $SEED complete ====="
}

# ============================================================
# MBPP — seed 3 only (seeds 1 and 2 already complete)
# ============================================================
BENCHMARK="mbpp"
log "========================================================"
log "Benchmark: mbpp  (resuming from seed 3)"
log "========================================================"

run_seed mbpp 3 \
  179 "mbpp:70_15_15:1-179" \
  39  "mbpp:70_15_15:180-218" "180-218" \
  39  "mbpp:70_15_15:219-257" "219-257"

# MBPP baselines
BENCHMARK="mbpp"
for REP in $(seq 1 "$REPEATS"); do
    run_inspect "mbpp — random choice rep $REP / $REPEATS" \
      -T policy_mode=random_choice \
      -T experiment_split=test \
      -T split_definition="mbpp:70_15_15:219-257" \
      --max-samples 39 \
      --limit "219-257"

    run_inspect "mbpp — agent-based decision (CoT) rep $REP / $REPEATS" \
      -T policy_mode=agent_based_decision \
      -T selector_use_cot=True \
      -T experiment_split=test \
      -T split_definition="mbpp:70_15_15:219-257" \
      --max-samples 39 \
      --limit "219-257"
done

log "===== mbpp all seeds and baselines complete ====="

# ============================================================
# HumanEval — all 3 seeds (none previously run)
# ============================================================
BENCHMARK="humaneval"
log "========================================================"
log "Benchmark: humaneval  (train 115 / val 24 / test 25)"
log "========================================================"

for SEED in 1 2 3; do
    run_seed humaneval "$SEED" \
      115 "humaneval:70_15_15:1-115" \
      24  "humaneval:70_15_15:116-139" "116-139" \
      25  "humaneval:70_15_15:140-164" "140-164"
done

# HumanEval baselines
BENCHMARK="humaneval"
for REP in $(seq 1 "$REPEATS"); do
    run_inspect "humaneval — random choice rep $REP / $REPEATS" \
      -T policy_mode=random_choice \
      -T experiment_split=test \
      -T split_definition="humaneval:70_15_15:140-164" \
      --max-samples 25 \
      --limit "140-164"

    run_inspect "humaneval — agent-based decision (CoT) rep $REP / $REPEATS" \
      -T policy_mode=agent_based_decision \
      -T selector_use_cot=True \
      -T experiment_split=test \
      -T split_definition="humaneval:70_15_15:140-164" \
      --max-samples 25 \
      --limit "140-164"
done

log "===== humaneval all seeds and baselines complete ====="

# ============================================================
# Aggregate results
# ============================================================
log "--- Aggregating results ---"
"$PYTHON_BIN" V3/scripts/aggregate_results.py \
  --results-dir results \
  --output-dir results/aggregates 2>&1 | tee -a "$LOG_FILE"

# ============================================================
# Plots
# ============================================================
PLOT_BASE="plots/${SESSION}"

log "--- Plotting rl_bandit ---"
"$PYTHON_BIN" plot.py \
  --results-dir results \
  --output-dir "${PLOT_BASE}/rl_bandit" \
  --policy-mode rl_bandit 2>&1 | tee -a "$LOG_FILE"

log "--- Plotting random_choice ---"
"$PYTHON_BIN" plot.py \
  --results-dir results \
  --output-dir "${PLOT_BASE}/random_choice" \
  --policy-mode random_choice 2>&1 | tee -a "$LOG_FILE"

log "--- Plotting agent_based_decision ---"
"$PYTHON_BIN" plot.py \
  --results-dir results \
  --output-dir "${PLOT_BASE}/agent_based_decision" \
  --policy-mode agent_based_decision 2>&1 | tee -a "$LOG_FILE"

log "--- Plotting comparison (all policies) ---"
"$PYTHON_BIN" plot.py \
  --results-dir results \
  --output-dir "${PLOT_BASE}/comparison" 2>&1 | tee -a "$LOG_FILE"

log "========================================================"
log "Experiment complete."
log "  Results:    results/"
log "  Aggregates: results/aggregates/"
log "  Plots:      ${PLOT_BASE}/"
log "  Seed weights completed:"
log "    mbpp:      weights/mbpp_ucb1_seed{1,2,3}.json  (seeds 1+2 from prior run)"
log "    humaneval: weights/humaneval_ucb1_seed{1,2,3}.json"
log "========================================================"
