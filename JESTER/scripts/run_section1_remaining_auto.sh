#!/usr/bin/env bash
# run_section1_remaining_auto.sh — Section 1 missing runs (5 runs).
# klucb-eval/cubert + exp3 x4, qwen judge.
# Waits for GPU to cool below 65°C between runs.

set -euo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/../.."

LOG="$PWD/section1_remaining_auto.log"
echo "=== Section 1 remaining auto-run started at $(date) ===" | tee -a "$LOG"

wait_for_cool() {
    local threshold=65 max_wait=1800 waited=0 temp
    temp=$(nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader,nounits 2>/dev/null | tr -d ' \r')
    [[ -z "$temp" || "$temp" -le "$threshold" ]] && return 0
    echo "[COOLING] GPU at ${temp}°C — waiting for <${threshold}°C..." | tee -a "$LOG"
    while [[ "$temp" -gt "$threshold" && "$waited" -lt "$max_wait" ]]; do
        sleep 30; waited=$((waited+30))
        temp=$(nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader,nounits 2>/dev/null | tr -d ' \r')
        echo "[COOLING] ${temp}°C (${waited}s elapsed)" | tee -a "$LOG"
    done
    echo "[COOLING] GPU at ${temp}°C — proceeding." | tee -a "$LOG"
}

mark_done() {
    python3 - "$1" <<'PYEOF'
import sys
store_name = sys.argv[1]
with open('RUNBOOK.txt', 'r') as f:
    lines = f.readlines()
new_lines = list(lines)
for i, line in enumerate(lines):
    if '[ ]' in line and store_name in line:
        new_lines[i] = line.replace('[ ]', '[X]', 1)
        for j in range(i - 1, max(i - 4, -1), -1):
            if '[ ]' in lines[j]:
                new_lines[j] = lines[j].replace('[ ]', '[X]', 1)
                break
with open('RUNBOOK.txt', 'w') as f:
    f.writelines(new_lines)
print(f'Marked [X]: {store_name}')
PYEOF
}

run_one() {
    local store_name="$1"; shift
    local weights_arg=""
    if [[ "$1" != "--" ]]; then weights_arg="$1"; shift; fi
    shift
    if [ -d "stored_results/$store_name" ]; then
        echo "[SKIP] $store_name" | tee -a "$LOG"; return 0
    fi
    wait_for_cool
    echo "" | tee -a "$LOG"
    echo "============================================================" | tee -a "$LOG"
    echo "[START] $store_name  —  $(date)" | tee -a "$LOG"
    echo "============================================================" | tee -a "$LOG"
    "$@" 2>&1 | tee -a "$LOG"
    if [ -n "$weights_arg" ]; then
        bash JESTER/scripts/store_results.sh "$store_name" "$weights_arg" 2>&1 | tee -a "$LOG"
    else
        bash JESTER/scripts/store_results.sh "$store_name" 2>&1 | tee -a "$LOG"
    fi
    mark_done "$store_name" | tee -a "$LOG"
    echo "[DONE] $store_name  —  $(date)" | tee -a "$LOG"
}

run_one "section-1/klucb-eval-llama31-qwen-cubert_wbo" \
    "weights/cubert_klucb_llama31_qwen.json" -- \
    env BANDIT_ALGORITHM=klucb \
        WEIGHTS_PATH=weights/cubert_klucb_llama31_qwen.json \
        bash JESTER/scripts/run_rl_eval.sh cubert_wbo

run_one "section-1/exp3-train-llama31-qwen-adversarial_code_buggy" -- \
    env BANDIT_ALGORITHM=exp3 \
        WEIGHTS_PATH=weights/acb_exp3_llama31_qwen.json \
        bash JESTER/scripts/run_rl_train.sh adversarial_code_buggy

run_one "section-1/exp3-eval-llama31-qwen-adversarial_code_buggy" \
    "weights/acb_exp3_llama31_qwen.json" -- \
    env BANDIT_ALGORITHM=exp3 \
        WEIGHTS_PATH=weights/acb_exp3_llama31_qwen.json \
        bash JESTER/scripts/run_rl_eval.sh adversarial_code_buggy

run_one "section-1/exp3-train-llama31-qwen-cubert_wbo" -- \
    env BANDIT_ALGORITHM=exp3 \
        WEIGHTS_PATH=weights/cubert_exp3_llama31_qwen.json \
        bash JESTER/scripts/run_rl_train.sh cubert_wbo

run_one "section-1/exp3-eval-llama31-qwen-cubert_wbo" \
    "weights/cubert_exp3_llama31_qwen.json" -- \
    env BANDIT_ALGORITHM=exp3 \
        WEIGHTS_PATH=weights/cubert_exp3_llama31_qwen.json \
        bash JESTER/scripts/run_rl_eval.sh cubert_wbo

echo "" | tee -a "$LOG"
echo "=== Section 1 remaining COMPLETE at $(date) ===" | tee -a "$LOG"
