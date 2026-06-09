#!/usr/bin/env bash
# run_rl_train.sh — Bandit training (train split, weight updates enabled)
#
# Usage:
#   bash V3/scripts/run_rl_train.sh                          # both benchmarks, UCB1
#   bash V3/scripts/run_rl_train.sh adversarial_code_buggy
#   bash V3/scripts/run_rl_train.sh cubert_wbo
#   EPOCHS=5 MAX_ITERATIONS=6 bash V3/scripts/run_rl_train.sh
#   BANDIT_ALGORITHM=thompson WEIGHTS_PATH=weights/acb_thompson_llama31_qwen.json bash V3/scripts/run_rl_train.sh adversarial_code_buggy
#
# Supported algorithms: ucb1, thompson, klucb, exp3
# WEIGHTS_PATH: if set, overrides auto-naming (weights/{benchmark}_{algorithm}.json)
# Per-epoch checkpoints: {weights_base}_epoch_001.json, ...
set -euo pipefail

BENCHMARK="${1:-both}"
MAX_ITERATIONS="${MAX_ITERATIONS:-6}"
EPOCHS="${EPOCHS:-5}"
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
    local TRAINED_WEIGHTS="$WEIGHTS_PATH"

    echo "[$(date '+%H:%M:%S')] rl_bandit TRAIN — $BM  algorithm=$BANDIT_ALGORITHM  epochs=$EPOCHS  max_iter=$MAX_ITERATIONS" | tee -a "$LOG"
    echo "[$(date '+%H:%M:%S')] attacker (selector): $SELECTOR_MODEL  target (judge): $TARGET_MODEL  code gen (LLM): $MODEL" | tee -a "$LOG"
    echo "[$(date '+%H:%M:%S')] weights: $WEIGHTS_PATH" | tee -a "$LOG"

    for epoch in $(seq 1 "$EPOCHS"); do
        echo "[$(date '+%H:%M:%S')] --- epoch $epoch / $EPOCHS ---" | tee -a "$LOG"
        PYTHONPATH=V3 timeout "$EPOCH_TIMEOUT" "$INSPECT" eval V3/adversarial_attack.py@adversarial_code_llm \
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
            -T split_definition="${BM}:full:train" \
            2>&1 | tee -a "$LOG" \
        || echo "[$(date '+%H:%M:%S')] WARNING: epoch $epoch exited non-zero (timeout or failed sample)" | tee -a "$LOG"

        local CHECKPOINT="${WEIGHTS_PATH%.json}_epoch_$(printf '%03d' "$epoch").json"
        if [[ -f "$WEIGHTS_PATH" ]]; then
            cp "$WEIGHTS_PATH" "$CHECKPOINT"
            TRAINED_WEIGHTS="$CHECKPOINT"
            echo "[$(date '+%H:%M:%S')] Checkpoint saved: $CHECKPOINT" | tee -a "$LOG"
            "$PYTHON" V3/scripts/log_entropy.py "$CHECKPOINT" "$WEIGHTS_PATH" 2>&1 | tee -a "$LOG"
        fi
        docker rm -f "$(docker ps -q --filter 'ancestor=aisiuk/inspect-tool-support')" 2>/dev/null || true
    done
    echo "[$(date '+%H:%M:%S')] Training complete. Final weights: $TRAINED_WEIGHTS" | tee -a "$LOG"
}

[[ "$BENCHMARK" == "adversarial_code_buggy" || "$BENCHMARK" == "both" ]] && train_benchmark adversarial_code_buggy
[[ "$BENCHMARK" == "cubert_wbo"             || "$BENCHMARK" == "both" ]] && train_benchmark cubert_wbo

echo "[$(date '+%H:%M:%S')] Done. Log: $LOG"
