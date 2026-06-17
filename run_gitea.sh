#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

source venv/Scripts/activate

MODEL="${MODEL:-ollama/llama3.1:8b}"
JUDGE_MODEL="${JUDGE_MODEL:-$MODEL}"
SELECTOR_MODEL="${SELECTOR_MODEL:-$JUDGE_MODEL}"
MAX_ITER="${MAX_ITER:-3}"
LIMIT="${LIMIT:-3}"
MAX_SAMPLES="${MAX_SAMPLES:-1}"
BASE_BRANCH="${BASE_BRANCH:-main}"
GITEA_REPO_VALUE="${GITEA_REPO:-}"

if [[ -z "$GITEA_REPO_VALUE" ]]; then
  echo "ERROR: GITEA_REPO is not set"
  echo "Example: export GITEA_REPO='owner/repo'"
  exit 1
fi

echo "Running Gitea adversarial attack"
echo "  MODEL=$MODEL"
echo "  JUDGE_MODEL=$JUDGE_MODEL"
echo "  SELECTOR_MODEL=$SELECTOR_MODEL"
echo "  GITEA_REPO=$GITEA_REPO_VALUE"
echo "  BASE_BRANCH=$BASE_BRANCH"
echo "  MAX_ITER=$MAX_ITER"
echo "  LIMIT=$LIMIT"
echo "  MAX_SAMPLES=$MAX_SAMPLES"

inspect eval \
  JESTER/adversarial_attack.py@adversarial_gitea_react_attack \
  --model "$MODEL" \
  -T judge_model="$JUDGE_MODEL" \
  -T selector_model="$SELECTOR_MODEL" \
  -T repo="$GITEA_REPO_VALUE" \
  -T base_branch="$BASE_BRANCH" \
  -T max_iterations="$MAX_ITER" \
  --max-samples "$MAX_SAMPLES" \
  --limit "$LIMIT"
