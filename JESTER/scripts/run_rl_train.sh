#!/usr/bin/env bash
# run_rl_train.sh — Bandit training (train split, weight updates enabled)
#
# Usage:
#   bash JESTER/scripts/run_rl_train.sh                          # both benchmarks, UCB1
#   bash JESTER/scripts/run_rl_train.sh adversarial_code_buggy
#   bash JESTER/scripts/run_rl_train.sh cubert_wbo
#   MAX_ITERATIONS=6 bash JESTER/scripts/run_rl_train.sh
#   BANDIT_ALGORITHM=thompson WEIGHTS_PATH=weights/acb_thompson_llama31_qwen.json bash JESTER/scripts/run_rl_train.sh adversarial_code_buggy
#
# Supported algorithms: ucb1, thompson, klucb, exp3
# WEIGHTS_PATH: if set, overrides auto-naming (weights/{benchmark}_{algorithm}.json)
# Output: weights file updated in-place; entropy curve written alongside it.
set -euo pipefail

BENCHMARK="${1:-both}"
MAX_ITERATIONS="${MAX_ITERATIONS:-6}"
MODEL="${MODEL:-ollama/llama3.1:8b}"
TARGET_MODEL="${TARGET_MODEL:-ollama/qwen2.5-coder:7b}"
SELECTOR_MODEL="${SELECTOR_MODEL:-$MODEL}"
EPOCH_TIMEOUT="${EPOCH_TIMEOUT:-86400}"
BANDIT_ALGORITHM="${BANDIT_ALGORITHM:-ucb1}"
WEIGHTS_PATH_OVERRIDE="${WEIGHTS_PATH:-}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$(cd "$SCRIPT_DIR/../.." && pwd)"

[[ -f "../.venv/Scripts/inspect" ]] && INSPECT="../.venv/Scripts/inspect" \
|| { [[ -f "../.venv/bin/inspect" ]] && INSPECT="../.venv/bin/inspect"; } \
|| INSPECT="inspect"

[[ -f "../.venv/Scripts/python.exe" ]] && PYTHON="../.venv/Scripts/python.exe" \
|| { [[ -f "../.venv/bin/python" ]] && PYTHON="../.venv/bin/python"; } \
|| PYTHON="python"

mkdir -p logs/experiment weights
LOG="logs/experiment/rl_train_$(date +%Y%m%d_%H%M%S).log"

train_benchmark() {
    local BM="$1"

    local WEIGHTS_PATH
    if [[ -n "$WEIGHTS_PATH_OVERRIDE" ]]; then
        WEIGHTS_PATH="$("$PYTHON" -c "import os; print(os.path.abspath('$WEIGHTS_PATH_OVERRIDE'))")"
    else
        WEIGHTS_PATH="$("$PYTHON" -c "import os; print(os.path.abspath('weights/${BM}_${BANDIT_ALGORITHM}.json'))")"
    fi

    echo "[$(date '+%H:%M:%S')] rl_bandit TRAIN — $BM  algorithm=$BANDIT_ALGORITHM  max_iter=$MAX_ITERATIONS" | tee -a "$LOG"
    echo "[$(date '+%H:%M:%S')] attacker (selector): $SELECTOR_MODEL  target (judge): $TARGET_MODEL  code gen (LLM): $MODEL" | tee -a "$LOG"
    echo "[$(date '+%H:%M:%S')] weights: $WEIGHTS_PATH" | tee -a "$LOG"

    timeout "$EPOCH_TIMEOUT" "$INSPECT" eval JESTER/adversarial_attack.py@adversarial_code_llm \
        --model "$MODEL" \
        --max-samples 10 \
        --limit 1000 \
        -T benchmark="$BM" \
        -T policy_mode=rl_bandit \
        -T bandit_algorithm="$BANDIT_ALGORITHM" \
        -T bandit_weights_path="$WEIGHTS_PATH" \
        -T bandit_freeze_weights=False \
        -T mutation_strategy=react \
        -T experiment_mode=iterative \
        -T use_llm_judge=True \
        -T target_model="$TARGET_MODEL" \
        -T selector_model="$SELECTOR_MODEL" \
        -T max_iterations="$MAX_ITERATIONS" \
        -T experiment_split=train \
        -T split_definition="${BM}:70_15_15:train" \
        2>&1 | tee -a "$LOG" \
    || echo "[$(date '+%H:%M:%S')] WARNING: training exited non-zero (timeout or failed sample)" | tee -a "$LOG"

    if [[ -f "$WEIGHTS_PATH" ]]; then
        "$PYTHON" JESTER/scripts/log_entropy.py "$WEIGHTS_PATH" "$WEIGHTS_PATH" 2>&1 | tee -a "$LOG"
        echo "[$(date '+%H:%M:%S')] Training complete. Weights: $WEIGHTS_PATH" | tee -a "$LOG"
    fi

    docker rm -f "$(docker ps -q --filter 'ancestor=aisiuk/inspect-tool-support')" 2>/dev/null || true
}

[[ "$BENCHMARK" == "adversarial_code_buggy" || "$BENCHMARK" == "both" ]] && train_benchmark adversarial_code_buggy
[[ "$BENCHMARK" == "cubert_wbo"             || "$BENCHMARK" == "both" ]] && train_benchmark cubert_wbo

echo "[$(date '+%H:%M:%S')] Done. Log: $LOG"
powershell -c "[console]::beep(880,200);[console]::beep(1100,200);[console]::beep(1320,500)" 2>/dev/null || true
