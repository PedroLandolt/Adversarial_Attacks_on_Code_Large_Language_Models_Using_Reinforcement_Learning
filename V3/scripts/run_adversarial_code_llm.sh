#!/usr/bin/env bash

# Run presets for task: adversarial_code_llm
# Usage:
#   bash V3/scripts/run_adversarial_code_llm.sh
#   bash V3/scripts/run_adversarial_code_llm.sh --quick

set -euo pipefail

MODEL="${MODEL:-ollama/qwen3.5:9b}"
JUDGE_MODEL="${JUDGE_MODEL:-$MODEL}"
SELECTOR_MODEL="${SELECTOR_MODEL:-$JUDGE_MODEL}"
MAX_ITER="${MAX_ITER:-3}"
LIMIT="${LIMIT:-5}"
MAX_SAMPLES="${MAX_SAMPLES:-2}"

for arg in "$@"; do
    case "$arg" in
        --quick)
            LIMIT=2
            ;;
    esac
done

echo "Running adversarial_code_llm preset"
echo "  MODEL=$MODEL"
echo "  JUDGE_MODEL=$JUDGE_MODEL"
echo "  SELECTOR_MODEL=$SELECTOR_MODEL"
echo "  MAX_ITER=$MAX_ITER"
echo "  LIMIT=$LIMIT"
echo "  MAX_SAMPLES=$MAX_SAMPLES"

inspect eval \
    V3/adversarial_attack.py@adversarial_code_llm \
    --model "$MODEL" \
    -T mutation_strategy=react \
    -T use_llm_judge=True \
    -T judge_model="$JUDGE_MODEL" \
    -T selector_model="$SELECTOR_MODEL" \
    -T max_iterations="$MAX_ITER" \
    --max-samples "$MAX_SAMPLES" \
    --limit "$LIMIT"
