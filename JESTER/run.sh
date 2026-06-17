#!/usr/bin/env bash

# Unified launcher for task presets in adversarial_attack.py
# Usage:
#   bash JESTER/run.sh
#   bash JESTER/run.sh mbpp
#   bash JESTER/run.sh humaneval
#   bash JESTER/run.sh gitea
#   bash JESTER/run.sh mbpp --quick
#   bash JESTER/run.sh humaneval --quick
#   bash JESTER/run.sh gitea --quick

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

TASK="benchmark"
BENCHMARK="mbpp"
EXTRA_ARGS=()

# ------------------------------------------------------------
# Load .env automatically if present
# Prefer project root .env, fallback to JESTER/.env
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
    echo "No .env file found in project root or V3 directory. Using current shell environment."
fi

# ------------------------------------------------------------
# Parse task preset
# ------------------------------------------------------------
if [[ $# -ge 1 ]]; then
    case "$1" in
        mbpp|humaneval)
            TASK="benchmark"
            BENCHMARK="$1"
            shift
            ;;
        gitea)
            TASK="$1"
            shift
            ;;
    esac
fi

if [[ $# -gt 0 ]]; then
    EXTRA_ARGS=("$@")
fi

# ------------------------------------------------------------
# Effective config preview
# ------------------------------------------------------------
EFFECTIVE_MODEL="${MODEL:-ollama/llama3.1:8b}"
EFFECTIVE_JUDGE_MODEL="${JUDGE_MODEL:-$EFFECTIVE_MODEL}"
EFFECTIVE_SELECTOR_MODEL="${SELECTOR_MODEL:-$EFFECTIVE_JUDGE_MODEL}"
EFFECTIVE_MAX_ITER="${MAX_ITER:-3}"
EFFECTIVE_LIMIT="${LIMIT:-5}"
EFFECTIVE_MAX_SAMPLES="${MAX_SAMPLES:-2}"
EFFECTIVE_BASE_BRANCH="${BASE_BRANCH:-main}"
EFFECTIVE_GITEA_REPO="${GITEA_REPO:-}"

echo "============================================================"
echo "Adversarial Attack Launcher"
echo "============================================================"
echo "Task preset:        $TASK"
if [[ "$TASK" == "benchmark" ]]; then
    echo "BENCHMARK:          $BENCHMARK"
fi
echo "MODEL:              $EFFECTIVE_MODEL"
echo "JUDGE_MODEL:        $EFFECTIVE_JUDGE_MODEL"
echo "SELECTOR_MODEL:     $EFFECTIVE_SELECTOR_MODEL"
echo "MAX_ITER:           $EFFECTIVE_MAX_ITER"
echo "LIMIT:              $EFFECTIVE_LIMIT"
echo "MAX_SAMPLES:        $EFFECTIVE_MAX_SAMPLES"
if [[ "$TASK" == "gitea" ]]; then
    echo "BASE_BRANCH:        $EFFECTIVE_BASE_BRANCH"
    echo "GITEA_REPO:         ${EFFECTIVE_GITEA_REPO:-<not set>}"
fi
if [[ ${#EXTRA_ARGS[@]} -gt 0 ]]; then
    echo "EXTRA_ARGS:         ${EXTRA_ARGS[*]}"
fi
echo "============================================================"

action_benchmark() {
    bash "$SCRIPT_DIR/scripts/run_adversarial_code_llm.sh" "$BENCHMARK" "${EXTRA_ARGS[@]}"
}

action_gitea() {
    bash "$SCRIPT_DIR/scripts/run_adversarial_gitea_react_attack.sh" "${EXTRA_ARGS[@]}"
}

case "$TASK" in
    benchmark)
        action_benchmark
        ;;
    gitea)
        action_gitea
        ;;
    *)
        echo "Unknown task preset: $TASK"
        echo "Valid options: mbpp | humaneval | gitea"
        exit 1
        ;;
esac
