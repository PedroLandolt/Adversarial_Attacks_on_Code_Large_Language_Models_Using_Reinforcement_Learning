#!/usr/bin/env bash
# Unattended RL bandit training: N training epochs then validation and test with frozen weights.
# Weights are updated incrementally — if interrupted, progress in weights/<benchmark>_ucb1.json is preserved.
#
# Foreground:
#   bash V3/scripts/train_rl_overnight.sh
#   bash V3/scripts/train_rl_overnight.sh --benchmark humaneval --epochs 5
#   bash V3/scripts/train_rl_overnight.sh --samples 50   # cap train samples per epoch (quick test)
#
# Background (log is always written to logs/overnight/):
#   nohup bash V3/scripts/train_rl_overnight.sh --epochs 3 > /dev/null 2>&1 &

set -euo pipefail

# -----------------------------------------------------------
# Defaults — override via env vars or flags
# -----------------------------------------------------------
BENCHMARK="${BENCHMARK:-mbpp}"
EPOCHS="${EPOCHS:-3}"
MAX_ITERATIONS="${MAX_ITERATIONS:-12}"
MODEL="${MODEL:-ollama/qwen3.5:0.8b}"
JUDGE_MODEL="${JUDGE_MODEL:-$MODEL}"
SELECTOR_MODEL="${SELECTOR_MODEL:-$JUDGE_MODEL}"
TRAIN_SAMPLES=""   # if set, caps --limit and --max-samples for training phases
EPOCH_TIMEOUT_SECS="${EPOCH_TIMEOUT_SECS:-86400}"  # 24h safety net — kills hung Ollama calls before they freeze the run forever

while [[ $# -gt 0 ]]; do
    case "$1" in
        --benchmark)      BENCHMARK="$2";      shift 2 ;;
        --epochs)         EPOCHS="$2";         shift 2 ;;
        --max-iterations) MAX_ITERATIONS="$2"; shift 2 ;;
        --samples)        TRAIN_SAMPLES="$2";  shift 2 ;;
        *) echo "Unknown arg: $1" >&2; exit 1 ;;
    esac
done

# -----------------------------------------------------------
# Paths
# -----------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJECT_ROOT"

# Locate inspect binary (works with or without venv activation)
if [[ -f ".venv/Scripts/inspect" ]]; then
    INSPECT_BIN=".venv/Scripts/inspect"
elif [[ -f ".venv/bin/inspect" ]]; then
    INSPECT_BIN=".venv/bin/inspect"
else
    INSPECT_BIN="inspect"
fi

# Locate Python binary (same venv)
if [[ -f ".venv/Scripts/python.exe" ]]; then
    _PYTHON_BIN=".venv/Scripts/python.exe"
elif [[ -f ".venv/bin/python" ]]; then
    _PYTHON_BIN=".venv/bin/python"
else
    _PYTHON_BIN="python"
fi

# Use Python to get the Windows-format absolute path so it resolves correctly
# even if Inspect AI changes the Python process cwd before task setup.
WEIGHTS_PATH="$("$_PYTHON_BIN" -c "import os; print(os.path.abspath('weights/${BENCHMARK}_ucb1.json'))")"
LOG_DIR="logs/overnight"
mkdir -p "$LOG_DIR"
LOG_FILE="${LOG_DIR}/${BENCHMARK}_$(date +%Y%m%d_%H%M%S).log"

# -----------------------------------------------------------
# Load .env
# -----------------------------------------------------------
if [[ -f ".env" ]]; then
    set -a; source ".env"; set +a
elif [[ -f "V3/.env" ]]; then
    set -a; source "V3/.env"; set +a
fi

# -----------------------------------------------------------
# Split definitions
# -----------------------------------------------------------
case "$BENCHMARK" in
    mbpp)
        # inspect_evals.mbpp loads the sanitized HuggingFace split: 257 samples total
        TRAIN_N="${TRAIN_SAMPLES:-179}"; TRAIN_DEF="mbpp:70_15_15:1-179"; VAL_RANGE="180-218"; TEST_RANGE="219-257"
        VAL_N=39;  VAL_DEF="mbpp:70_15_15:180-218"
        TEST_N=39; TEST_DEF="mbpp:70_15_15:219-257"
        ;;
    humaneval)
        # inspect_evals loads standard HumanEval: 164 samples total
        TRAIN_N="${TRAIN_SAMPLES:-115}"; TRAIN_DEF="humaneval:70_15_15:1-115"; VAL_RANGE="116-139"; TEST_RANGE="140-164"
        VAL_N=24;  VAL_DEF="humaneval:70_15_15:116-139"
        TEST_N=25; TEST_DEF="humaneval:70_15_15:140-164"
        ;;
    *)
        echo "Unknown benchmark: $BENCHMARK (expected mbpp or humaneval)" >&2; exit 1 ;;
esac

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOG_FILE"; }

log "========================================================"
log "Overnight RL training session"
log "  benchmark:   $BENCHMARK"
log "  epochs:      $EPOCHS"
log "  train limit: $TRAIN_N samples"
log "  max iter:    $MAX_ITERATIONS"
log "  model:       $MODEL"
log "  weights:     $WEIGHTS_PATH"
log "  log:         $LOG_FILE"
log "========================================================"

run_phase() {
    local label="$1"; shift
    log "--- $label ---"
    PYTHONPATH=V3 "$INSPECT_BIN" eval V3/adversarial_attack.py@adversarial_code_llm \
      --model "$MODEL" \
      -T mutation_strategy=react \
      -T policy_mode=rl_bandit \
      -T bandit_algorithm=ucb1 \
      -T bandit_weights_path="$WEIGHTS_PATH" \
      -T experiment_mode=iterative \
      -T use_llm_judge=True \
      -T judge_model="$JUDGE_MODEL" \
      -T selector_model="$SELECTOR_MODEL" \
      -T max_iterations="$MAX_ITERATIONS" \
      "$@" 2>&1 | tee -a "$LOG_FILE"
}

# -----------------------------------------------------------
# Training epochs
# -----------------------------------------------------------

# TRAINED_WEIGHTS points to the last epoch checkpoint so val/test always read
# from a frozen artifact, never the live training file.
TRAINED_WEIGHTS="$WEIGHTS_PATH"

for epoch in $(seq 1 "$EPOCHS"); do
    log "--- Train epoch $epoch / $EPOCHS ---"
    # timeout kills a hung Ollama call before it freezes the run forever.
    # || log prevents set -e from aborting before the checkpoint cp below.
    PYTHONPATH=V3 timeout "$EPOCH_TIMEOUT_SECS" "$INSPECT_BIN" eval \
        V3/adversarial_attack.py@adversarial_code_llm \
        --model "$MODEL" \
        -T benchmark="$BENCHMARK" \
        -T mutation_strategy=react \
        -T policy_mode=rl_bandit \
        -T bandit_algorithm=ucb1 \
        -T bandit_weights_path="$WEIGHTS_PATH" \
        -T experiment_mode=iterative \
        -T use_llm_judge=True \
        -T judge_model="$JUDGE_MODEL" \
        -T selector_model="$SELECTOR_MODEL" \
        -T max_iterations="$MAX_ITERATIONS" \
        -T experiment_split=train \
        -T split_definition="$TRAIN_DEF" \
        -T bandit_freeze_weights=False \
        --max-samples "$TRAIN_N" \
        --limit "$TRAIN_N" \
        2>&1 | tee -a "$LOG_FILE" \
    || log "WARNING: Train epoch $epoch returned non-zero (exit $?; possibly timeout or a failed sample). Checkpoint saved from current weights."

    EPOCH_CHECKPOINT="${WEIGHTS_PATH%.json}_epoch_$(printf '%03d' "$epoch").json"
    cp "$WEIGHTS_PATH" "$EPOCH_CHECKPOINT"
    TRAINED_WEIGHTS="$EPOCH_CHECKPOINT"
    log "Epoch $epoch complete. Checkpoint: $EPOCH_CHECKPOINT"
    docker rm -f $(docker ps -q --filter "ancestor=aisiuk/inspect-tool-support") 2>/dev/null || true
    log "Sandbox containers cleaned up after epoch $epoch."
done

# Point WEIGHTS_PATH at the frozen checkpoint so run_phase uses it for val/test.
WEIGHTS_PATH="$TRAINED_WEIGHTS"

# -----------------------------------------------------------
# Validation (frozen weights)
# -----------------------------------------------------------
run_phase "Validation (frozen weights)" \
  -T benchmark="$BENCHMARK" \
  -T experiment_split=validation \
  -T split_definition="$VAL_DEF" \
  -T bandit_freeze_weights=True \
  --max-samples "$VAL_N" \
  --limit "$VAL_RANGE"

# -----------------------------------------------------------
# Test (frozen weights)
# -----------------------------------------------------------
run_phase "Test (frozen weights)" \
  -T benchmark="$BENCHMARK" \
  -T experiment_split=test \
  -T split_definition="$TEST_DEF" \
  -T bandit_freeze_weights=True \
  --max-samples "$TEST_N" \
  --limit "$TEST_RANGE"

log "========================================================"
log "Session complete."
log "  Results: results/"
log "  Weights: $WEIGHTS_PATH"
log "========================================================"
