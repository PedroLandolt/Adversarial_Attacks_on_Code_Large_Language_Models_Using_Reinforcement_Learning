#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

source venv/Scripts/activate

MODEL="${MODEL:-ollama/qwen3.5:9b}"
JUDGE_MODEL="${JUDGE_MODEL:-$MODEL}"
SELECTOR_MODEL="${SELECTOR_MODEL:-$JUDGE_MODEL}"
MAX_ITER="${MAX_ITER:-10}"
LIMIT="${LIMIT:-5}"
MAX_SAMPLES="${MAX_SAMPLES:-10}"

echo "Running MBPP adversarial attack"
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
