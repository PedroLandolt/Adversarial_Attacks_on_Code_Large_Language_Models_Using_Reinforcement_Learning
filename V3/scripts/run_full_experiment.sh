#!/usr/bin/env bash
# Full experiment pipeline for paper results.
#
# Phases:
#   1. RL bandit  — N training epochs (train split, weight updates)
#   2. RL bandit  — validation (frozen weights)
#   3. RL bandit  — test (frozen weights)
#   4. Random choice — test split
#   5. Agent-based decision (CoT) — test split
#   6. Aggregate all results
#   7. Plot individually: rl_bandit / random_choice / agent_based_decision
#   8. Plot comparison: all 3 policies together
#
# All three policies are evaluated on the same held-out test split for a fair comparison.
# Plots go to plots/<session_timestamp>/{rl_bandit,random_choice,agent_based_decision,comparison}/
#
# Usage (foreground):
#   bash V3/scripts/run_full_experiment.sh
#   bash V3/scripts/run_full_experiment.sh --benchmark humaneval --epochs 5
#
# Smoke test (caps every phase to 5 samples, 1 epoch):
#   bash V3/scripts/run_full_experiment.sh --samples 5 --epochs 1
#
# Background:
#   nohup bash V3/scripts/run_full_experiment.sh > /dev/null 2>&1 &

set -euo pipefail

# -----------------------------------------------------------
# Defaults — override via env vars or flags
# -----------------------------------------------------------
BENCHMARK="${BENCHMARK:-mbpp}"
EPOCHS="${EPOCHS:-3}"
MAX_ITERATIONS="${MAX_ITERATIONS:-12}"
MODEL="${MODEL:-ollama/llama3.1:8b}"
JUDGE_MODEL="${JUDGE_MODEL:-$MODEL}"
SELECTOR_MODEL="${SELECTOR_MODEL:-$JUDGE_MODEL}"
SAMPLES_CAP=""   # if set, caps ALL phases (train + val + test) for quick smoke testing
EPOCH_TIMEOUT_SECS="${EPOCH_TIMEOUT_SECS:-86400}"  # 24h safety net — kills hung Ollama calls before they freeze the run forever

while [[ $# -gt 0 ]]; do
    case "$1" in
        --benchmark)      BENCHMARK="$2";      shift 2 ;;
        --epochs)         EPOCHS="$2";         shift 2 ;;
        --max-iterations) MAX_ITERATIONS="$2"; shift 2 ;;
        --samples)        SAMPLES_CAP="$2";    shift 2 ;;
        *) echo "Unknown arg: $1" >&2; exit 1 ;;
    esac
done

# -----------------------------------------------------------
# Paths
# -----------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJECT_ROOT"

if [[ -f ".venv/Scripts/inspect" ]]; then
    INSPECT_BIN=".venv/Scripts/inspect"
elif [[ -f ".venv/bin/inspect" ]]; then
    INSPECT_BIN=".venv/bin/inspect"
elif [[ -f "../.venv/Scripts/inspect" ]]; then
    INSPECT_BIN="../.venv/Scripts/inspect"
elif [[ -f "../.venv/bin/inspect" ]]; then
    INSPECT_BIN="../.venv/bin/inspect"
else
    INSPECT_BIN="inspect"
fi

if [[ -f ".venv/Scripts/python.exe" ]]; then
    PYTHON_BIN=".venv/Scripts/python.exe"
elif [[ -f ".venv/bin/python" ]]; then
    PYTHON_BIN=".venv/bin/python"
elif [[ -f "../.venv/Scripts/python.exe" ]]; then
    PYTHON_BIN="../.venv/Scripts/python.exe"
elif [[ -f "../.venv/bin/python" ]]; then
    PYTHON_BIN="../.venv/bin/python"
else
    PYTHON_BIN="python"
fi

# Use the venv Python to resolve the weights path to a Windows-format absolute path.
# Inspect AI changes the Python process cwd before task setup, so a relative path
# silently resolves to a non-existent location and load_bandit_weights returns None.
WEIGHTS_PATH="$("$PYTHON_BIN" -c "import os; print(os.path.abspath('weights/${BENCHMARK}_ucb1.json'))")"
SESSION="$(date +%Y%m%d_%H%M%S)"
LOG_DIR="logs/experiment"
mkdir -p "$LOG_DIR"
LOG_FILE="${LOG_DIR}/${BENCHMARK}_${SESSION}.log"

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
        TRAIN_N=179; TRAIN_RANGE="179"
        VAL_N=39;    VAL_DEF="mbpp:70_15_15:180-218"; VAL_RANGE="180-218"
        TEST_N=39;   TEST_DEF="mbpp:70_15_15:219-257"; TEST_RANGE="219-257"
        ;;
    humaneval)
        # inspect_evals loads standard HumanEval: 164 samples total
        TRAIN_N=115; TRAIN_RANGE="115"
        VAL_N=24;    VAL_DEF="humaneval:70_15_15:116-139"; VAL_RANGE="116-139"
        TEST_N=25;   TEST_DEF="humaneval:70_15_15:140-164"; TEST_RANGE="140-164"
        ;;
    *)
        echo "Unknown benchmark: $BENCHMARK (expected mbpp or humaneval)" >&2; exit 1 ;;
esac

# In smoke-test mode (--samples N), cap all phases to N samples.
# Split integrity is relaxed intentionally — use only for pipeline validation.
if [[ -n "$SAMPLES_CAP" ]]; then
    TRAIN_N="$SAMPLES_CAP"; TRAIN_RANGE="$SAMPLES_CAP"
    VAL_N="$SAMPLES_CAP";   VAL_RANGE="$SAMPLES_CAP"
    TEST_N="$SAMPLES_CAP";  TEST_RANGE="$SAMPLES_CAP"
fi

# Derive TRAIN_DEF after TRAIN_N is finalised so --samples overrides are reflected.
TRAIN_DEF="${BENCHMARK}:70_15_15:1-${TRAIN_N}"

# -----------------------------------------------------------
# Logging helper
# -----------------------------------------------------------
log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOG_FILE"; }

log "========================================================"
log "Full experiment pipeline"
log "  benchmark:   $BENCHMARK"
log "  RL epochs:   $EPOCHS"
log "  train:       $TRAIN_N samples ($TRAIN_RANGE)"
log "  val:         $VAL_N samples ($VAL_RANGE)"
log "  test:        $TEST_N samples ($TEST_RANGE)"
log "  max iter:    $MAX_ITERATIONS"
log "  model:       $MODEL"
log "  weights:     $WEIGHTS_PATH"
log "  session:     $SESSION"
log "  log:         $LOG_FILE"
log "========================================================"

run_inspect() {
    local label="$1"; shift
    log "--- $label ---"
    PYTHONPATH=V3 "$INSPECT_BIN" eval V3/adversarial_attack.py@adversarial_code_llm \
      --model "$MODEL" \
      -T benchmark="$BENCHMARK" \
      -T mutation_strategy=react \
      -T experiment_mode=iterative \
      -T use_llm_judge=True \
      -T judge_model="$JUDGE_MODEL" \
      -T max_iterations="$MAX_ITERATIONS" \
      "$@" 2>&1 | tee -a "$LOG_FILE"
}

# ==============================================================
# Phase 1-3: RL bandit (train → validation → test)
# ==============================================================

# TRAINED_WEIGHTS tracks the last epoch checkpoint.
# Validation and test always read from this frozen artifact, never from the
# live training file — so an accidental restart cannot corrupt the trained state.
TRAINED_WEIGHTS="$WEIGHTS_PATH"

for epoch in $(seq 1 "$EPOCHS"); do
    log "--- RL train epoch $epoch / $EPOCHS ---"
    # timeout kills a hung Ollama call before it freezes the run forever.
    # || log prevents set -e from aborting before the checkpoint cp below.
    PYTHONPATH=V3 timeout "$EPOCH_TIMEOUT_SECS" "$INSPECT_BIN" eval \
        V3/adversarial_attack.py@adversarial_code_llm \
        --model "$MODEL" \
        -T benchmark="$BENCHMARK" \
        -T mutation_strategy=react \
        -T experiment_mode=iterative \
        -T use_llm_judge=True \
        -T judge_model="$JUDGE_MODEL" \
        -T max_iterations="$MAX_ITERATIONS" \
        -T policy_mode=rl_bandit \
        -T bandit_algorithm=ucb1 \
        -T bandit_weights_path="$WEIGHTS_PATH" \
        -T bandit_freeze_weights=False \
        -T experiment_split=train \
        -T split_definition="$TRAIN_DEF" \
        -T selector_model="$SELECTOR_MODEL" \
        --max-samples "$TRAIN_N" \
        --limit "$TRAIN_RANGE" \
        2>&1 | tee -a "$LOG_FILE" \
    || log "WARNING: RL train epoch $epoch returned non-zero (exit $?; possibly timeout or a failed sample). Checkpoint saved from current weights."

    EPOCH_CHECKPOINT="${WEIGHTS_PATH%.json}_epoch_$(printf '%03d' "$epoch").json"
    cp "$WEIGHTS_PATH" "$EPOCH_CHECKPOINT"
    TRAINED_WEIGHTS="$EPOCH_CHECKPOINT"
    log "RL epoch $epoch complete. Checkpoint: $EPOCH_CHECKPOINT"
    docker rm -f $(docker ps -q --filter "ancestor=aisiuk/inspect-tool-support") 2>/dev/null || true
    log "Sandbox containers cleaned up after epoch $epoch."
done

run_inspect "RL validation (frozen weights)" \
  -T policy_mode=rl_bandit \
  -T bandit_algorithm=ucb1 \
  -T bandit_weights_path="$TRAINED_WEIGHTS" \
  -T bandit_freeze_weights=True \
  -T experiment_split=validation \
  -T split_definition="$VAL_DEF" \
  -T selector_model="$SELECTOR_MODEL" \
  --max-samples "$VAL_N" \
  --limit "$VAL_RANGE"

run_inspect "RL test (frozen weights)" \
  -T policy_mode=rl_bandit \
  -T bandit_algorithm=ucb1 \
  -T bandit_weights_path="$TRAINED_WEIGHTS" \
  -T bandit_freeze_weights=True \
  -T experiment_split=test \
  -T split_definition="$TEST_DEF" \
  -T selector_model="$SELECTOR_MODEL" \
  --max-samples "$TEST_N" \
  --limit "$TEST_RANGE"

# ==============================================================
# Phase 4: Random choice — test split
# ==============================================================
run_inspect "Random choice — test" \
  -T policy_mode=random_choice \
  -T experiment_split=test \
  -T split_definition="$TEST_DEF" \
  --max-samples "$TEST_N" \
  --limit "$TEST_RANGE"

# ==============================================================
# Phase 5: Agent-based decision (CoT) — test split
# ==============================================================
run_inspect "Agent-based decision (CoT) — test" \
  -T policy_mode=agent_based_decision \
  -T selector_use_cot=True \
  -T experiment_split=test \
  -T split_definition="$TEST_DEF" \
  -T selector_model="$SELECTOR_MODEL" \
  --max-samples "$TEST_N" \
  --limit "$TEST_RANGE"

# ==============================================================
# Phase 6: Aggregate all results
# ==============================================================
log "--- Aggregating results ---"
"$PYTHON_BIN" V3/scripts/aggregate_results.py \
  --results-dir results \
  --output-dir results/aggregates 2>&1 | tee -a "$LOG_FILE"

# ==============================================================
# Phase 7: Individual plots (one directory per policy)
# ==============================================================
PLOT_BASE="plots/${SESSION}"

log "--- Plotting rl_bandit ---"
"$PYTHON_BIN" plot.py \
  --results-dir results \
  --output-dir "${PLOT_BASE}/rl_bandit" \
  --policy-mode rl_bandit 2>&1 | tee -a "$LOG_FILE"

log "--- Plotting random_choice ---"
"$PYTHON_BIN" plot.py \
  --results-dir results \
  --output-dir "${PLOT_BASE}/random_choice" \
  --policy-mode random_choice 2>&1 | tee -a "$LOG_FILE"

log "--- Plotting agent_based_decision ---"
"$PYTHON_BIN" plot.py \
  --results-dir results \
  --output-dir "${PLOT_BASE}/agent_based_decision" \
  --policy-mode agent_based_decision 2>&1 | tee -a "$LOG_FILE"

# ==============================================================
# Phase 8: Comparison plot (all 3 policies together)
# ==============================================================
log "--- Plotting comparison (all policies) ---"
"$PYTHON_BIN" plot.py \
  --results-dir results \
  --output-dir "${PLOT_BASE}/comparison" 2>&1 | tee -a "$LOG_FILE"

log "========================================================"
log "Experiment complete."
log "  Results:     results/"
log "  Aggregates:  results/aggregates/"
log "  Plots:       ${PLOT_BASE}/"
log "    ├── rl_bandit/"
log "    ├── random_choice/"
log "    ├── agent_based_decision/"
log "    └── comparison/"
log "========================================================"
