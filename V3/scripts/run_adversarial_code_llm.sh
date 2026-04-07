#!/usr/bin/env bash

# Run presets for task: adversarial_code_llm
# Usage:
#   bash V3/scripts/run_adversarial_code_llm.sh
#   bash V3/scripts/run_adversarial_code_llm.sh mbpp
#   bash V3/scripts/run_adversarial_code_llm.sh humaneval
#   bash V3/scripts/run_adversarial_code_llm.sh --quick

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

# Load environment variables from .env if present
ENV_FILE=""
if [[ -f "$PROJECT_ROOT/.env" ]]; then
    ENV_FILE="$PROJECT_ROOT/.env"
elif [[ -f "$PROJECT_ROOT/V3/.env" ]]; then
    ENV_FILE="$PROJECT_ROOT/V3/.env"
fi

if [[ -n "$ENV_FILE" ]]; then
    set -a
    # shellcheck disable=SC1090
    source "$ENV_FILE"
    set +a
fi

MODEL="${MODEL:-ollama/qwen3.5:0.8b}"
JUDGE_MODEL="${JUDGE_MODEL:-$MODEL}"
SELECTOR_MODEL="${SELECTOR_MODEL:-$JUDGE_MODEL}"
BENCHMARK="mbpp"
MAX_ITER="${MAX_ITER:-3}"
LIMIT="${LIMIT:-5}"
MAX_SAMPLES="${MAX_SAMPLES:-2}"
PASSTHROUGH_ARGS=()

for arg in "$@"; do
    case "$arg" in
        mbpp|humaneval)
            BENCHMARK="$arg"
            ;;
        --quick)
            LIMIT=2
            ;;
        *)
            PASSTHROUGH_ARGS+=("$arg")
            ;;
    esac
done

echo "Running adversarial_code_llm preset"
echo "  BENCHMARK=$BENCHMARK"
echo "  MODEL=$MODEL"
echo "  JUDGE_MODEL=$JUDGE_MODEL"
echo "  SELECTOR_MODEL=$SELECTOR_MODEL"
echo "  MAX_ITER=$MAX_ITER"
echo "  LIMIT=$LIMIT"
echo "  MAX_SAMPLES=$MAX_SAMPLES"
if [[ ${#PASSTHROUGH_ARGS[@]} -gt 0 ]]; then
    echo "  PASSTHROUGH_ARGS=${PASSTHROUGH_ARGS[*]}"
fi

inspect eval \
    V3/adversarial_attack.py@adversarial_code_llm \
    --model "$MODEL" \
    -T benchmark="$BENCHMARK" \
    -T mutation_strategy=react \
    -T use_llm_judge=True \
    -T judge_model="$JUDGE_MODEL" \
    -T selector_model="$SELECTOR_MODEL" \
    -T max_iterations="$MAX_ITER" \
    --max-samples "$MAX_SAMPLES" \
    --limit "$LIMIT" \
    "${PASSTHROUGH_ARGS[@]}"
