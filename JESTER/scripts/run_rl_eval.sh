#!/usr/bin/env bash
# run_rl_eval.sh — Bandit evaluation (validation + test, frozen weights)
#
# Run AFTER run_rl_train.sh.
#
# Usage:
#   bash JESTER/scripts/run_rl_eval.sh                          # both benchmarks, UCB1
#   bash JESTER/scripts/run_rl_eval.sh adversarial_code_buggy
#   bash JESTER/scripts/run_rl_eval.sh cubert_wbo
#   BANDIT_ALGORITHM=thompson WEIGHTS_PATH=weights/acb_thompson_llama31_qwen.json bash JESTER/scripts/run_rl_eval.sh adversarial_code_buggy
#
# Supported algorithms: ucb1, thompson, klucb, exp3
# WEIGHTS_PATH: if set, uses that file; otherwise auto-resolves to weights/{benchmark}_{algorithm}.json
set -euo pipefail

BENCHMARK="${1:-both}"
MAX_ITERATIONS="${MAX_ITERATIONS:-6}"
MODEL="${MODEL:-ollama/llama3.1:8b}"
TARGET_MODEL="${TARGET_MODEL:-ollama/qwen2.5-coder:7b}"
SELECTOR_MODEL="${SELECTOR_MODEL:-$MODEL}"
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

mkdir -p logs/experiment
LOG="logs/experiment/rl_eval_$(date +%Y%m%d_%H%M%S).log"

eval_benchmark() {
    local BM="$1"

    local WEIGHTS
    if [[ -n "$WEIGHTS_PATH_OVERRIDE" ]]; then
        WEIGHTS="$("$PYTHON" -c "import os; print(os.path.abspath('$WEIGHTS_PATH_OVERRIDE'))")"
    else
        WEIGHTS="$("$PYTHON" -c "import os; print(os.path.abspath('weights/${BM}_${BANDIT_ALGORITHM}.json'))")"
    fi

    if [[ ! -f "$WEIGHTS" ]]; then
        echo "[$(date '+%H:%M:%S')] ERROR: weights not found: $WEIGHTS — run run_rl_train.sh first" | tee -a "$LOG"
        return 1
    fi

    echo "[$(date '+%H:%M:%S')] rl_bandit EVAL — $BM  algorithm=$BANDIT_ALGORITHM  weights=$(basename "$WEIGHTS")  max_iter=$MAX_ITERATIONS" | tee -a "$LOG"
    echo "[$(date '+%H:%M:%S')] attacker (selector): $SELECTOR_MODEL  target (judge): $TARGET_MODEL  code gen (LLM): $MODEL" | tee -a "$LOG"

    for SPLIT in validation test; do
        echo "[$(date '+%H:%M:%S')] --- $SPLIT split ---" | tee -a "$LOG"
        PYTHONPATH=V3 "$INSPECT" eval JESTER/adversarial_attack.py@adversarial_code_llm \
            --model "$MODEL" \
            --max-samples 10 \
            --limit 1000 \
            -T benchmark="$BM" \
            -T policy_mode=rl_bandit \
            -T bandit_algorithm="$BANDIT_ALGORITHM" \
            -T bandit_weights_path="$WEIGHTS" \
            -T bandit_freeze_weights=True \
            -T mutation_strategy=react \
            -T experiment_mode=iterative \
            -T use_llm_judge=True \
            -T target_model="$TARGET_MODEL" \
            -T selector_model="$SELECTOR_MODEL" \
            -T max_iterations="$MAX_ITERATIONS" \
            -T experiment_split="$SPLIT" \
            -T split_definition="${BM}:70_15_15:${SPLIT}" \
            2>&1 | tee -a "$LOG"
    done
    echo "[$(date '+%H:%M:%S')] Eval complete for $BM / $BANDIT_ALGORITHM" | tee -a "$LOG"
}

[[ "$BENCHMARK" == "adversarial_code_buggy" || "$BENCHMARK" == "both" ]] && eval_benchmark adversarial_code_buggy
[[ "$BENCHMARK" == "cubert_wbo"             || "$BENCHMARK" == "both" ]] && eval_benchmark cubert_wbo

echo "[$(date '+%H:%M:%S')] Done. Log: $LOG"
powershell -c "[console]::beep(880,200);[console]::beep(1100,200);[console]::beep(1320,500)" 2>/dev/null || true
