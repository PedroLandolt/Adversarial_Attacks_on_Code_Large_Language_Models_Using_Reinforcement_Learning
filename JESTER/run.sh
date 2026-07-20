#!/usr/bin/env bash

# Unified launcher for the adversarial_code_llm task.
# Usage:
#   bash JESTER/run.sh                        # defaults to the mbpp benchmark
#   bash JESTER/run.sh adversarial_code_buggy
#   bash JESTER/run.sh cubert_wbo
#   bash JESTER/run.sh humaneval --limit 10

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

BENCHMARK="mbpp"
EXTRA_ARGS=()

# ------------------------------------------------------------
# Load .env automatically if present.
# Prefer the project root .env, fall back to JESTER/.env.
# ------------------------------------------------------------
ENV_FILE=""
if [[ -f "$PROJECT_ROOT/.env" ]]; then
    ENV_FILE="$PROJECT_ROOT/.env"
elif [[ -f "$SCRIPT_DIR/.env" ]]; then
    ENV_FILE="$SCRIPT_DIR/.env"
fi

if [[ -n "$ENV_FILE" ]]; then
    echo "Loading environment from: $ENV_FILE"
    set -a
    # shellcheck disable=SC1090
    source "$ENV_FILE"
    set +a
else
    echo "No .env file found. Using the current shell environment."
fi

# ------------------------------------------------------------
# Parse the benchmark name (first positional argument).
# ------------------------------------------------------------
if [[ $# -ge 1 ]]; then
    case "$1" in
        adversarial_code_buggy|cubert_wbo|mbpp|humaneval|mbpp_pregenerated|humaneval_pregenerated)
            BENCHMARK="$1"
            shift
            ;;
    esac
fi

if [[ $# -gt 0 ]]; then
    EXTRA_ARGS=("$@")
fi

# ------------------------------------------------------------
# Effective config preview.
# ------------------------------------------------------------
EFFECTIVE_MODEL="${MODEL:-ollama/llama3.1:8b}"
EFFECTIVE_TARGET_MODEL="${TARGET_MODEL:-ollama/qwen2.5-coder:7b}"
EFFECTIVE_SELECTOR_MODEL="${SELECTOR_MODEL:-$EFFECTIVE_MODEL}"
EFFECTIVE_MAX_ITER="${MAX_ITER:-3}"
EFFECTIVE_LIMIT="${LIMIT:-5}"
EFFECTIVE_MAX_SAMPLES="${MAX_SAMPLES:-2}"

echo "============================================================"
echo "Adversarial Attack Launcher"
echo "============================================================"
echo "BENCHMARK:          $BENCHMARK"
echo "MODEL (attacker):   $EFFECTIVE_MODEL"
echo "TARGET_MODEL (judge): $EFFECTIVE_TARGET_MODEL"
echo "SELECTOR_MODEL:     $EFFECTIVE_SELECTOR_MODEL"
echo "MAX_ITER:           $EFFECTIVE_MAX_ITER"
echo "LIMIT:              $EFFECTIVE_LIMIT"
echo "MAX_SAMPLES:        $EFFECTIVE_MAX_SAMPLES"
if [[ ${#EXTRA_ARGS[@]} -gt 0 ]]; then
    echo "EXTRA_ARGS:         ${EXTRA_ARGS[*]}"
fi
echo "============================================================"

bash "$SCRIPT_DIR/scripts/run_adversarial_code_llm.sh" "$BENCHMARK" "${EXTRA_ARGS[@]}"
