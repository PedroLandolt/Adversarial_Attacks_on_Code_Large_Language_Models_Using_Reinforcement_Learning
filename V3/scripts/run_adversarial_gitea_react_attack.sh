#!/usr/bin/env bash

# Run presets for task: adversarial_gitea_react_attack
# Usage:
#   bash V3/scripts/run_adversarial_gitea_react_attack.sh
#   bash V3/scripts/run_adversarial_gitea_react_attack.sh --quick
#
# Requires:
#   GITEA_REPO
# Optional:
#   GITHUB_API_URL, GITHUB_TOKEN, REVIEWER_TOKEN

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

GITHUB_API_URL="${GITHUB_API_URL:-}"
GITHUB_TOKEN="${GITHUB_TOKEN:-}"
REVIEWER_TOKEN="${REVIEWER_TOKEN:-}"
GITEA_REPO="${GITEA_REPO:-}"

MODEL="${MODEL:-ollama/llama3.1:8b}"
JUDGE_MODEL="${JUDGE_MODEL:-$MODEL}"
SELECTOR_MODEL="${SELECTOR_MODEL:-$JUDGE_MODEL}"
MAX_ITER="${MAX_ITER:-3}"
LIMIT="${LIMIT:-3}"
MAX_SAMPLES="${MAX_SAMPLES:-1}"
BASE_BRANCH="${BASE_BRANCH:-main}"
GITEA_REPO_VALUE="${GITEA_REPO:-}"

for arg in "$@"; do
    case "$arg" in
        --quick)
            LIMIT=1
            MAX_ITER=1
            ;;
    esac
done

if [[ -z "$GITEA_REPO_VALUE" ]]; then
    echo "ERROR: GITEA_REPO is not set."
    echo "Example: GITEA_REPO=gitadmin/test-repo"
    exit 1
fi

echo "Running adversarial_gitea_react_attack preset"
echo "  MODEL=$MODEL"
echo "  JUDGE_MODEL=$JUDGE_MODEL"
echo "  SELECTOR_MODEL=$SELECTOR_MODEL"
echo "  GITEA_REPO=$GITEA_REPO_VALUE"
echo "  BASE_BRANCH=$BASE_BRANCH"
echo "  MAX_ITER=$MAX_ITER"
echo "  LIMIT=$LIMIT"
echo "  MAX_SAMPLES=$MAX_SAMPLES"

inspect eval \
    V3/adversarial_attack.py@adversarial_gitea_react_attack \
    --model "$MODEL" \
    -T judge_model="$JUDGE_MODEL" \
    -T selector_model="$SELECTOR_MODEL" \
    -T repo="$GITEA_REPO_VALUE" \
    -T base_branch="$BASE_BRANCH" \
    -T max_iterations="$MAX_ITER" \
    --max-samples "$MAX_SAMPLES" \
    --limit "$LIMIT"