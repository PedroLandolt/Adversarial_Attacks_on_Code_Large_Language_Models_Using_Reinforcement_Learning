#!/usr/bin/env bash
# run_section2_auto.sh — Runs all 20 Section 2 experiments sequentially.
# llama3.1:8b (attacker) vs deepseek-coder:6.7b (judge)
#
# - Stores results and marks [X] in RUNBOOK.txt after each run.
# - Skips any run whose stored_results/ folder already exists.
# - Safe to restart: re-running picks up from the first un-stored run.
#
# Usage (from project root, with venv active):
#   bash JESTER/scripts/run_section2_auto.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJECT"

LOG="$PROJECT/section2_auto.log"
echo "=== Section 2 auto-run started at $(date) ===" | tee -a "$LOG"

mark_done() {
    local store_name="$1"
    python3 - "$store_name" <<'PYEOF'
import sys
store_name = sys.argv[1]
with open('RUNBOOK.txt', 'r') as f:
    lines = f.readlines()
new_lines = list(lines)
for i, line in enumerate(lines):
    if '[ ]' in line and store_name in line:
        new_lines[i] = line.replace('[ ]', '[X]', 1)
        # Also mark the preceding [ ] line (the run command)
        for j in range(i - 1, max(i - 4, -1), -1):
            prev = lines[j]
            if '[ ]' in prev:
                new_lines[j] = prev.replace('[ ]', '[X]', 1)
                break
with open('RUNBOOK.txt', 'w') as f:
    f.writelines(new_lines)
print(f'Marked [X]: {store_name}')
PYEOF
}

run_one() {
    # run_one <store_name> [weights_file] -- <cmd...>
    local store_name="$1"; shift
    local weights_arg=""
    if [[ "$1" != "--" ]]; then
        weights_arg="$1"; shift
    fi
    shift  # consume "--"

    if [ -d "stored_results/$store_name" ]; then
        echo "[SKIP] $store_name (already stored)" | tee -a "$LOG"
        return 0
    fi

    echo "" | tee -a "$LOG"
    echo "============================================================" | tee -a "$LOG"
    echo "[START] $store_name" | tee -a "$LOG"
    echo "Time: $(date)" | tee -a "$LOG"
    echo "============================================================" | tee -a "$LOG"

    "$@" 2>&1 | tee -a "$LOG"

    echo "[STORE] $store_name" | tee -a "$LOG"
    if [ -n "$weights_arg" ]; then
        bash JESTER/scripts/store_results.sh "$store_name" "$weights_arg" 2>&1 | tee -a "$LOG"
    else
        bash JESTER/scripts/store_results.sh "$store_name" 2>&1 | tee -a "$LOG"
    fi

    mark_done "$store_name" | tee -a "$LOG"
    echo "[DONE] $store_name at $(date)" | tee -a "$LOG"
}

# ── Non-RL baselines ──────────────────────────────────────────────────────────

run_one "section-2/random-llama31-deepseek-adversarial_code_buggy" -- \
    env TARGET_MODEL=ollama/deepseek-coder:6.7b bash JESTER/scripts/run_random.sh adversarial_code_buggy

run_one "section-2/random-llama31-deepseek-cubert_wbo" -- \
    env TARGET_MODEL=ollama/deepseek-coder:6.7b bash JESTER/scripts/run_random.sh cubert_wbo

run_one "section-2/react-llama31-deepseek-adversarial_code_buggy" -- \
    env TARGET_MODEL=ollama/deepseek-coder:6.7b bash JESTER/scripts/run_react.sh adversarial_code_buggy

run_one "section-2/react-llama31-deepseek-cubert_wbo" -- \
    env TARGET_MODEL=ollama/deepseek-coder:6.7b bash JESTER/scripts/run_react.sh cubert_wbo

# ── UCB1 ──────────────────────────────────────────────────────────────────────

run_one "section-2/ucb1-train-llama31-deepseek-adversarial_code_buggy" -- \
    env TARGET_MODEL=ollama/deepseek-coder:6.7b \
        WEIGHTS_PATH=weights/acb_ucb1_llama31_deepseek.json \
        bash JESTER/scripts/run_rl_train.sh adversarial_code_buggy

run_one "section-2/ucb1-eval-llama31-deepseek-adversarial_code_buggy" \
    "weights/acb_ucb1_llama31_deepseek.json" -- \
    env TARGET_MODEL=ollama/deepseek-coder:6.7b \
        WEIGHTS_PATH=weights/acb_ucb1_llama31_deepseek.json \
        bash JESTER/scripts/run_rl_eval.sh adversarial_code_buggy

run_one "section-2/ucb1-train-llama31-deepseek-cubert_wbo" -- \
    env TARGET_MODEL=ollama/deepseek-coder:6.7b \
        WEIGHTS_PATH=weights/cubert_ucb1_llama31_deepseek.json \
        bash JESTER/scripts/run_rl_train.sh cubert_wbo

run_one "section-2/ucb1-eval-llama31-deepseek-cubert_wbo" \
    "weights/cubert_ucb1_llama31_deepseek.json" -- \
    env TARGET_MODEL=ollama/deepseek-coder:6.7b \
        WEIGHTS_PATH=weights/cubert_ucb1_llama31_deepseek.json \
        bash JESTER/scripts/run_rl_eval.sh cubert_wbo

# ── Thompson Sampling ─────────────────────────────────────────────────────────

run_one "section-2/thompson-train-llama31-deepseek-adversarial_code_buggy" -- \
    env TARGET_MODEL=ollama/deepseek-coder:6.7b \
        BANDIT_ALGORITHM=thompson \
        WEIGHTS_PATH=weights/acb_thompson_llama31_deepseek.json \
        bash JESTER/scripts/run_rl_train.sh adversarial_code_buggy

run_one "section-2/thompson-eval-llama31-deepseek-adversarial_code_buggy" \
    "weights/acb_thompson_llama31_deepseek.json" -- \
    env TARGET_MODEL=ollama/deepseek-coder:6.7b \
        BANDIT_ALGORITHM=thompson \
        WEIGHTS_PATH=weights/acb_thompson_llama31_deepseek.json \
        bash JESTER/scripts/run_rl_eval.sh adversarial_code_buggy

run_one "section-2/thompson-train-llama31-deepseek-cubert_wbo" -- \
    env TARGET_MODEL=ollama/deepseek-coder:6.7b \
        BANDIT_ALGORITHM=thompson \
        WEIGHTS_PATH=weights/cubert_thompson_llama31_deepseek.json \
        bash JESTER/scripts/run_rl_train.sh cubert_wbo

run_one "section-2/thompson-eval-llama31-deepseek-cubert_wbo" \
    "weights/cubert_thompson_llama31_deepseek.json" -- \
    env TARGET_MODEL=ollama/deepseek-coder:6.7b \
        BANDIT_ALGORITHM=thompson \
        WEIGHTS_PATH=weights/cubert_thompson_llama31_deepseek.json \
        bash JESTER/scripts/run_rl_eval.sh cubert_wbo

# ── KL-UCB ────────────────────────────────────────────────────────────────────

run_one "section-2/klucb-train-llama31-deepseek-adversarial_code_buggy" -- \
    env TARGET_MODEL=ollama/deepseek-coder:6.7b \
        BANDIT_ALGORITHM=klucb \
        WEIGHTS_PATH=weights/acb_klucb_llama31_deepseek.json \
        bash JESTER/scripts/run_rl_train.sh adversarial_code_buggy

run_one "section-2/klucb-eval-llama31-deepseek-adversarial_code_buggy" \
    "weights/acb_klucb_llama31_deepseek.json" -- \
    env TARGET_MODEL=ollama/deepseek-coder:6.7b \
        BANDIT_ALGORITHM=klucb \
        WEIGHTS_PATH=weights/acb_klucb_llama31_deepseek.json \
        bash JESTER/scripts/run_rl_eval.sh adversarial_code_buggy

run_one "section-2/klucb-train-llama31-deepseek-cubert_wbo" -- \
    env TARGET_MODEL=ollama/deepseek-coder:6.7b \
        BANDIT_ALGORITHM=klucb \
        WEIGHTS_PATH=weights/cubert_klucb_llama31_deepseek.json \
        bash JESTER/scripts/run_rl_train.sh cubert_wbo

run_one "section-2/klucb-eval-llama31-deepseek-cubert_wbo" \
    "weights/cubert_klucb_llama31_deepseek.json" -- \
    env TARGET_MODEL=ollama/deepseek-coder:6.7b \
        BANDIT_ALGORITHM=klucb \
        WEIGHTS_PATH=weights/cubert_klucb_llama31_deepseek.json \
        bash JESTER/scripts/run_rl_eval.sh cubert_wbo

# ── EXP3 ──────────────────────────────────────────────────────────────────────

run_one "section-2/exp3-train-llama31-deepseek-adversarial_code_buggy" -- \
    env TARGET_MODEL=ollama/deepseek-coder:6.7b \
        BANDIT_ALGORITHM=exp3 \
        WEIGHTS_PATH=weights/acb_exp3_llama31_deepseek.json \
        bash JESTER/scripts/run_rl_train.sh adversarial_code_buggy

run_one "section-2/exp3-eval-llama31-deepseek-adversarial_code_buggy" \
    "weights/acb_exp3_llama31_deepseek.json" -- \
    env TARGET_MODEL=ollama/deepseek-coder:6.7b \
        BANDIT_ALGORITHM=exp3 \
        WEIGHTS_PATH=weights/acb_exp3_llama31_deepseek.json \
        bash JESTER/scripts/run_rl_eval.sh adversarial_code_buggy

run_one "section-2/exp3-train-llama31-deepseek-cubert_wbo" -- \
    env TARGET_MODEL=ollama/deepseek-coder:6.7b \
        BANDIT_ALGORITHM=exp3 \
        WEIGHTS_PATH=weights/cubert_exp3_llama31_deepseek.json \
        bash JESTER/scripts/run_rl_train.sh cubert_wbo

run_one "section-2/exp3-eval-llama31-deepseek-cubert_wbo" \
    "weights/cubert_exp3_llama31_deepseek.json" -- \
    env TARGET_MODEL=ollama/deepseek-coder:6.7b \
        BANDIT_ALGORITHM=exp3 \
        WEIGHTS_PATH=weights/cubert_exp3_llama31_deepseek.json \
        bash JESTER/scripts/run_rl_eval.sh cubert_wbo

echo "" | tee -a "$LOG"
echo "=== Section 2 COMPLETE at $(date) ===" | tee -a "$LOG"
