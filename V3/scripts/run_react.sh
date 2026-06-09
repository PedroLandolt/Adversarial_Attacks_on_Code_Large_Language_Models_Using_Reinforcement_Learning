#!/usr/bin/env bash
# run_react.sh — ReAct agent (CoT) selector
#
# Usage:
#   bash V3/scripts/run_react.sh                          # both benchmarks
#   bash V3/scripts/run_react.sh adversarial_code_buggy
#   bash V3/scripts/run_react.sh cubert_wbo
#   MAX_ITERATIONS=3 bash V3/scripts/run_react.sh
set -euo pipefail

BENCHMARK="${1:-both}"
MAX_ITERATIONS="${MAX_ITERATIONS:-6}"
MODEL="${MODEL:-ollama/llama3.1:8b}"
TARGET_MODEL="${TARGET_MODEL:-ollama/qwen2.5-coder:7b}"
SELECTOR_MODEL="${SELECTOR_MODEL:-$MODEL}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$(cd "$SCRIPT_DIR/../.." && pwd)"

[[ -f "../.venv/Scripts/inspect" ]] && INSPECT="../.venv/Scripts/inspect" \
|| { [[ -f "../.venv/bin/inspect" ]] && INSPECT="../.venv/bin/inspect"; } \
|| INSPECT="inspect"

mkdir -p logs/experiment
LOG="logs/experiment/react_$(date +%Y%m%d_%H%M%S).log"

run() {
    local BM="$1"
    echo "[$(date '+%H:%M:%S')] agent_based_decision — $BM (max_iter=$MAX_ITERATIONS)" | tee -a "$LOG"
    PYTHONPATH=V3 "$INSPECT" eval V3/adversarial_attack.py@adversarial_code_llm \
        --model "$MODEL" \
        --max-samples 10 \
        --limit 1000 \
        -T benchmark="$BM" \
        -T policy_mode=agent_based_decision \
        -T selector_use_cot=True \
        -T mutation_strategy=react \
        -T experiment_mode=iterative \
        -T use_llm_judge=True \
        -T target_model="$TARGET_MODEL" \
        -T selector_model="$SELECTOR_MODEL" \
        -T max_iterations="$MAX_ITERATIONS" \
        -T experiment_split=test \
        -T split_definition="${BM}:full:test" \
        2>&1 | tee -a "$LOG"
}

[[ "$BENCHMARK" == "adversarial_code_buggy" || "$BENCHMARK" == "both" ]] && run adversarial_code_buggy
[[ "$BENCHMARK" == "cubert_wbo"             || "$BENCHMARK" == "both" ]] && run cubert_wbo

echo "[$(date '+%H:%M:%S')] Done. Log: $LOG"
