#!/usr/bin/env bash
# Full paper experiment pipeline — both benchmarks, N seeds, R baseline repeats.
#
# For each benchmark × seed:
#   1. Train RL bandit (fresh weights per seed, EPOCHS epochs on train split)
#   2. Validate with frozen weights (seed's final epoch checkpoint)
#   3. Test with frozen weights (seed's final epoch checkpoint)
# Then per benchmark:
#   4. Random choice   — test split, REPEATS runs
#   5. Agent-based CoT — test split, REPEATS runs
# Finally:
#   6. Aggregate all results
#   7. Plot per-policy and comparison
#
# Existing weights are backed up before anything is touched; seed weight files
# are separate from the live weights/{bm}_ucb1.json so nothing is overwritten.
#
# Usage:
#   bash V3/scripts/run_paper_experiments.sh
#   bash V3/scripts/run_paper_experiments.sh --benchmarks mbpp --seeds 3 --epochs 5
#   bash V3/scripts/run_paper_experiments.sh --benchmarks humaneval --seeds 1 --epochs 3
#
# Smoke test (caps every phase to 2 samples):
#   bash V3/scripts/run_paper_experiments.sh --benchmarks mbpp --seeds 1 --epochs 1 --repeats 1 --samples 2
#
# Background:
#   nohup bash V3/scripts/run_paper_experiments.sh > /dev/null 2>&1 &

set -euo pipefail

# -----------------------------------------------------------
# Defaults — override via flags
# -----------------------------------------------------------
BENCHMARKS="mbpp,humaneval"
SEEDS=3
EPOCHS=5
REPEATS=3
MAX_ITERATIONS="${MAX_ITERATIONS:-12}"
MODEL="${MODEL:-ollama/llama3.1:8b}"
JUDGE_MODEL="${JUDGE_MODEL:-$MODEL}"
SELECTOR_MODEL="${SELECTOR_MODEL:-$JUDGE_MODEL}"
SAMPLES_CAP=""
EPOCH_TIMEOUT_SECS="${EPOCH_TIMEOUT_SECS:-86400}"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --benchmarks)     BENCHMARKS="$2";     shift 2 ;;
        --seeds)          SEEDS="$2";          shift 2 ;;
        --epochs)         EPOCHS="$2";         shift 2 ;;
        --repeats)        REPEATS="$2";        shift 2 ;;
        --max-iterations) MAX_ITERATIONS="$2"; shift 2 ;;
        --samples)        SAMPLES_CAP="$2";    shift 2 ;;
        --model)          MODEL="$2"; JUDGE_MODEL="$MODEL"; SELECTOR_MODEL="$MODEL"; shift 2 ;;
        *) echo "Unknown arg: $1" >&2; exit 1 ;;
    esac
done

# -----------------------------------------------------------
# Paths — same venv-detection logic as run_full_experiment.sh
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

# -----------------------------------------------------------
# Session log
# -----------------------------------------------------------
SESSION="$(date +%Y%m%d_%H%M%S)"
LOG_DIR="logs/paper_experiment"
mkdir -p "$LOG_DIR"
LOG_FILE="${LOG_DIR}/${SESSION}.log"

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOG_FILE"; }

# -----------------------------------------------------------
# Load .env
# -----------------------------------------------------------
if [[ -f ".env" ]]; then
    set -a; source ".env"; set +a
elif [[ -f "V3/.env" ]]; then
    set -a; source "V3/.env"; set +a
fi

log "========================================================"
log "Paper experiment pipeline"
log "  benchmarks:  $BENCHMARKS"
log "  seeds:       $SEEDS"
log "  epochs/seed: $EPOCHS"
log "  repeats:     $REPEATS (random + agent baselines)"
log "  max iter:    $MAX_ITERATIONS"
log "  model:       $MODEL"
log "  session:     $SESSION"
log "  log:         $LOG_FILE"
log "========================================================"

# -----------------------------------------------------------
# Helpers
# -----------------------------------------------------------

# run_inspect LABEL [extra -T flags...]
# Calls inspect eval with fixed args; BENCHMARK and TRAIN_* must be set in caller scope.
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
      -T selector_model="$SELECTOR_MODEL" \
      -T max_iterations="$MAX_ITERATIONS" \
      "$@" 2>&1 | tee -a "$LOG_FILE"
}

# init_seed_weights PATH
# Writes a fresh zero-state weight file. The bandit will start with all arms
# at zero because the registry initialises pull_counts from tactic_registry,
# not from the file, and only updates from matching arm IDs.
init_seed_weights() {
    local path="$1"
    "$PYTHON_BIN" -c "
import json, sys
json.dump({'pull_counts': {}, 'cumulative_rewards': {}}, open(sys.argv[1], 'w'), indent=2)
" "$path"
    log "Initialized fresh weights → $path"
}

# -----------------------------------------------------------
# Backup existing weight files before touching anything
# -----------------------------------------------------------
IFS=',' read -ra BM_LIST <<< "$BENCHMARKS"
for BM in "${BM_LIST[@]}"; do
    LIVE_WEIGHTS="weights/${BM}_ucb1.json"
    if [[ -f "$LIVE_WEIGHTS" ]]; then
        BACKUP="weights/${BM}_ucb1_prerun_backup_${SESSION}.json"
        cp "$LIVE_WEIGHTS" "$BACKUP"
        log "Backed up $LIVE_WEIGHTS → $BACKUP"
    fi
done

# ============================================================
# Main loop: per benchmark × per seed
# ============================================================
for BM in "${BM_LIST[@]}"; do
    BENCHMARK="$BM"

    # Split definitions (70/15/15)
    case "$BM" in
        mbpp)
            TRAIN_N=179
            VAL_N=39;  VAL_DEF="mbpp:70_15_15:180-218"; VAL_RANGE="180-218"
            TEST_N=39; TEST_DEF="mbpp:70_15_15:219-257"; TEST_RANGE="219-257"
            ;;
        humaneval)
            TRAIN_N=115
            VAL_N=24;  VAL_DEF="humaneval:70_15_15:116-139"; VAL_RANGE="116-139"
            TEST_N=25; TEST_DEF="humaneval:70_15_15:140-164"; TEST_RANGE="140-164"
            ;;
        *)
            log "ERROR: Unknown benchmark '$BM' (expected mbpp or humaneval)." >&2
            exit 1 ;;
    esac

    # Apply smoke-test cap
    if [[ -n "$SAMPLES_CAP" ]]; then
        TRAIN_N="$SAMPLES_CAP"
        VAL_N="$SAMPLES_CAP";  VAL_RANGE="$SAMPLES_CAP"
        TEST_N="$SAMPLES_CAP"; TEST_RANGE="$SAMPLES_CAP"
    fi

    TRAIN_DEF="${BM}:70_15_15:1-${TRAIN_N}"

    log "========================================================"
    log "Benchmark: $BM  (train $TRAIN_N / val $VAL_N / test $TEST_N)"
    log "========================================================"

    # ----------------------------------------------------------
    # RL bandit: N seeds
    # ----------------------------------------------------------
    for SEED in $(seq 1 "$SEEDS"); do
        log "===== $BM  seed $SEED / $SEEDS ====="

        # Resolve seed weight path as absolute (Inspect AI changes cwd)
        SEED_WEIGHTS="$("$PYTHON_BIN" -c "import os; print(os.path.abspath('weights/${BM}_ucb1_seed${SEED}.json'))")"
        init_seed_weights "$SEED_WEIGHTS"

        TRAINED_WEIGHTS="$SEED_WEIGHTS"

        # Training epochs
        for EPOCH in $(seq 1 "$EPOCHS"); do
            log "--- $BM  seed $SEED  epoch $EPOCH / $EPOCHS ---"
            PYTHONPATH=V3 timeout "$EPOCH_TIMEOUT_SECS" "$INSPECT_BIN" eval \
                V3/adversarial_attack.py@adversarial_code_llm \
                --model "$MODEL" \
                -T benchmark="$BM" \
                -T mutation_strategy=react \
                -T experiment_mode=iterative \
                -T use_llm_judge=True \
                -T judge_model="$JUDGE_MODEL" \
                -T selector_model="$SELECTOR_MODEL" \
                -T max_iterations="$MAX_ITERATIONS" \
                -T policy_mode=rl_bandit \
                -T bandit_algorithm=ucb1 \
                -T bandit_weights_path="$SEED_WEIGHTS" \
                -T bandit_freeze_weights=False \
                -T experiment_split=train \
                -T split_definition="$TRAIN_DEF" \
                --max-samples "$TRAIN_N" \
                --limit "$TRAIN_N" \
                2>&1 | tee -a "$LOG_FILE" \
            || log "WARNING: $BM seed $SEED epoch $EPOCH returned non-zero (exit $?). Checkpoint saved."

            CKPT="${SEED_WEIGHTS%.json}_epoch_$(printf '%03d' "$EPOCH").json"
            cp "$SEED_WEIGHTS" "$CKPT"
            TRAINED_WEIGHTS="$CKPT"
            log "Checkpoint: $CKPT"
            docker rm -f $(docker ps -q --filter "ancestor=aisiuk/inspect-tool-support") 2>/dev/null || true
            log "Sandbox containers cleaned up after epoch $EPOCH."
        done

        # Validation (frozen weights — last epoch checkpoint)
        run_inspect "$BM seed $SEED — RL validation (frozen)" \
          -T policy_mode=rl_bandit \
          -T bandit_algorithm=ucb1 \
          -T bandit_weights_path="$TRAINED_WEIGHTS" \
          -T bandit_freeze_weights=True \
          -T experiment_split=validation \
          -T split_definition="$VAL_DEF" \
          --max-samples "$VAL_N" \
          --limit "$VAL_RANGE"

        # Test (frozen weights — last epoch checkpoint)
        run_inspect "$BM seed $SEED — RL test (frozen)" \
          -T policy_mode=rl_bandit \
          -T bandit_algorithm=ucb1 \
          -T bandit_weights_path="$TRAINED_WEIGHTS" \
          -T bandit_freeze_weights=True \
          -T experiment_split=test \
          -T split_definition="$TEST_DEF" \
          --max-samples "$TEST_N" \
          --limit "$TEST_RANGE"

        log "===== $BM seed $SEED complete ====="
    done

    # ----------------------------------------------------------
    # Baseline policies: R repeats on test split
    # ----------------------------------------------------------
    for REP in $(seq 1 "$REPEATS"); do
        run_inspect "$BM — random choice rep $REP / $REPEATS" \
          -T policy_mode=random_choice \
          -T experiment_split=test \
          -T split_definition="$TEST_DEF" \
          --max-samples "$TEST_N" \
          --limit "$TEST_RANGE"

        run_inspect "$BM — agent-based decision (CoT) rep $REP / $REPEATS" \
          -T policy_mode=agent_based_decision \
          -T selector_use_cot=True \
          -T experiment_split=test \
          -T split_definition="$TEST_DEF" \
          --max-samples "$TEST_N" \
          --limit "$TEST_RANGE"
    done

    log "===== $BM all seeds and baselines complete ====="
done

# ============================================================
# Aggregate results
# ============================================================
log "--- Aggregating results ---"
"$PYTHON_BIN" V3/scripts/aggregate_results.py \
  --results-dir results \
  --output-dir results/aggregates 2>&1 | tee -a "$LOG_FILE"

# ============================================================
# Plots
# ============================================================
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

log "--- Plotting comparison (all policies) ---"
"$PYTHON_BIN" plot.py \
  --results-dir results \
  --output-dir "${PLOT_BASE}/comparison" 2>&1 | tee -a "$LOG_FILE"

log "========================================================"
log "Experiment complete."
log "  Results:    results/"
log "  Aggregates: results/aggregates/"
log "  Plots:      ${PLOT_BASE}/"
log "    ├── rl_bandit/"
log "    ├── random_choice/"
log "    ├── agent_based_decision/"
log "    └── comparison/"
log "  Seed weights:"
for BM in "${BM_LIST[@]}"; do
    for SEED in $(seq 1 "$SEEDS"); do
        log "    weights/${BM}_ucb1_seed${SEED}.json"
    done
done
log "========================================================"
