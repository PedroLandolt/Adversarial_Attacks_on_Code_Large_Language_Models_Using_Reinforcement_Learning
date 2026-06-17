#!/usr/bin/env bash
# run_ablations_auto.sh — Ablation A (3-iter) + Ablation B (1-iter).
# random + react, all 4 judges, both datasets. 32 runs total.
# Waits for GPU to cool below 65°C between runs.
# Run AFTER Sections 1-4 are complete.

set -euo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/../.."

LOG="$PWD/ablations_auto.log"
echo "=== Ablations auto-run started at $(date) ===" | tee -a "$LOG"

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

echo "=== ABLATION A — 3 ITERATIONS ===" | tee -a "$LOG"

# Section 1 judge: qwen2.5-coder:7b
run_one "section-1/random-3iter-llama31-qwen-adversarial_code_buggy" -- \
    env MAX_ITERATIONS=3 bash JESTER/scripts/run_random.sh adversarial_code_buggy
run_one "section-1/random-3iter-llama31-qwen-cubert_wbo" -- \
    env MAX_ITERATIONS=3 bash JESTER/scripts/run_random.sh cubert_wbo
run_one "section-1/react-3iter-llama31-qwen-adversarial_code_buggy" -- \
    env MAX_ITERATIONS=3 bash JESTER/scripts/run_react.sh adversarial_code_buggy
run_one "section-1/react-3iter-llama31-qwen-cubert_wbo" -- \
    env MAX_ITERATIONS=3 bash JESTER/scripts/run_react.sh cubert_wbo

# Section 2 judge: deepseek-coder:6.7b
run_one "section-2/random-3iter-llama31-deepseek-adversarial_code_buggy" -- \
    env MAX_ITERATIONS=3 TARGET_MODEL=ollama/deepseek-coder:6.7b bash JESTER/scripts/run_random.sh adversarial_code_buggy
run_one "section-2/random-3iter-llama31-deepseek-cubert_wbo" -- \
    env MAX_ITERATIONS=3 TARGET_MODEL=ollama/deepseek-coder:6.7b bash JESTER/scripts/run_random.sh cubert_wbo
run_one "section-2/react-3iter-llama31-deepseek-adversarial_code_buggy" -- \
    env MAX_ITERATIONS=3 TARGET_MODEL=ollama/deepseek-coder:6.7b bash JESTER/scripts/run_react.sh adversarial_code_buggy
run_one "section-2/react-3iter-llama31-deepseek-cubert_wbo" -- \
    env MAX_ITERATIONS=3 TARGET_MODEL=ollama/deepseek-coder:6.7b bash JESTER/scripts/run_react.sh cubert_wbo

# Section 3 judge: codellama:7b
run_one "section-3/random-3iter-llama31-codellama-adversarial_code_buggy" -- \
    env MAX_ITERATIONS=3 TARGET_MODEL=ollama/codellama:7b bash JESTER/scripts/run_random.sh adversarial_code_buggy
run_one "section-3/random-3iter-llama31-codellama-cubert_wbo" -- \
    env MAX_ITERATIONS=3 TARGET_MODEL=ollama/codellama:7b bash JESTER/scripts/run_random.sh cubert_wbo
run_one "section-3/react-3iter-llama31-codellama-adversarial_code_buggy" -- \
    env MAX_ITERATIONS=3 TARGET_MODEL=ollama/codellama:7b bash JESTER/scripts/run_react.sh adversarial_code_buggy
run_one "section-3/react-3iter-llama31-codellama-cubert_wbo" -- \
    env MAX_ITERATIONS=3 TARGET_MODEL=ollama/codellama:7b bash JESTER/scripts/run_react.sh cubert_wbo

# Section 4 judge: starcoder2:7b
run_one "section-4/random-3iter-llama31-starcoder-adversarial_code_buggy" -- \
    env MAX_ITERATIONS=3 TARGET_MODEL=ollama/starcoder2:7b bash JESTER/scripts/run_random.sh adversarial_code_buggy
run_one "section-4/random-3iter-llama31-starcoder-cubert_wbo" -- \
    env MAX_ITERATIONS=3 TARGET_MODEL=ollama/starcoder2:7b bash JESTER/scripts/run_random.sh cubert_wbo
run_one "section-4/react-3iter-llama31-starcoder-adversarial_code_buggy" -- \
    env MAX_ITERATIONS=3 TARGET_MODEL=ollama/starcoder2:7b bash JESTER/scripts/run_react.sh adversarial_code_buggy
run_one "section-4/react-3iter-llama31-starcoder-cubert_wbo" -- \
    env MAX_ITERATIONS=3 TARGET_MODEL=ollama/starcoder2:7b bash JESTER/scripts/run_react.sh cubert_wbo

echo "=== ABLATION B — 1 ITERATION ===" | tee -a "$LOG"

# Section 1 judge: qwen2.5-coder:7b
run_one "section-1/random-1iter-llama31-qwen-adversarial_code_buggy" -- \
    env MAX_ITERATIONS=1 bash JESTER/scripts/run_random.sh adversarial_code_buggy
run_one "section-1/random-1iter-llama31-qwen-cubert_wbo" -- \
    env MAX_ITERATIONS=1 bash JESTER/scripts/run_random.sh cubert_wbo
run_one "section-1/react-1iter-llama31-qwen-adversarial_code_buggy" -- \
    env MAX_ITERATIONS=1 bash JESTER/scripts/run_react.sh adversarial_code_buggy
run_one "section-1/react-1iter-llama31-qwen-cubert_wbo" -- \
    env MAX_ITERATIONS=1 bash JESTER/scripts/run_react.sh cubert_wbo

# Section 2 judge: deepseek-coder:6.7b
run_one "section-2/random-1iter-llama31-deepseek-adversarial_code_buggy" -- \
    env MAX_ITERATIONS=1 TARGET_MODEL=ollama/deepseek-coder:6.7b bash JESTER/scripts/run_random.sh adversarial_code_buggy
run_one "section-2/random-1iter-llama31-deepseek-cubert_wbo" -- \
    env MAX_ITERATIONS=1 TARGET_MODEL=ollama/deepseek-coder:6.7b bash JESTER/scripts/run_random.sh cubert_wbo
run_one "section-2/react-1iter-llama31-deepseek-adversarial_code_buggy" -- \
    env MAX_ITERATIONS=1 TARGET_MODEL=ollama/deepseek-coder:6.7b bash JESTER/scripts/run_react.sh adversarial_code_buggy
run_one "section-2/react-1iter-llama31-deepseek-cubert_wbo" -- \
    env MAX_ITERATIONS=1 TARGET_MODEL=ollama/deepseek-coder:6.7b bash JESTER/scripts/run_react.sh cubert_wbo

# Section 3 judge: codellama:7b
run_one "section-3/random-1iter-llama31-codellama-adversarial_code_buggy" -- \
    env MAX_ITERATIONS=1 TARGET_MODEL=ollama/codellama:7b bash JESTER/scripts/run_random.sh adversarial_code_buggy
run_one "section-3/random-1iter-llama31-codellama-cubert_wbo" -- \
    env MAX_ITERATIONS=1 TARGET_MODEL=ollama/codellama:7b bash JESTER/scripts/run_random.sh cubert_wbo
run_one "section-3/react-1iter-llama31-codellama-adversarial_code_buggy" -- \
    env MAX_ITERATIONS=1 TARGET_MODEL=ollama/codellama:7b bash JESTER/scripts/run_react.sh adversarial_code_buggy
run_one "section-3/react-1iter-llama31-codellama-cubert_wbo" -- \
    env MAX_ITERATIONS=1 TARGET_MODEL=ollama/codellama:7b bash JESTER/scripts/run_react.sh cubert_wbo

# Section 4 judge: starcoder2:7b
run_one "section-4/random-1iter-llama31-starcoder-adversarial_code_buggy" -- \
    env MAX_ITERATIONS=1 TARGET_MODEL=ollama/starcoder2:7b bash JESTER/scripts/run_random.sh adversarial_code_buggy
run_one "section-4/random-1iter-llama31-starcoder-cubert_wbo" -- \
    env MAX_ITERATIONS=1 TARGET_MODEL=ollama/starcoder2:7b bash JESTER/scripts/run_random.sh cubert_wbo
run_one "section-4/react-1iter-llama31-starcoder-adversarial_code_buggy" -- \
    env MAX_ITERATIONS=1 TARGET_MODEL=ollama/starcoder2:7b bash JESTER/scripts/run_react.sh adversarial_code_buggy
run_one "section-4/react-1iter-llama31-starcoder-cubert_wbo" -- \
    env MAX_ITERATIONS=1 TARGET_MODEL=ollama/starcoder2:7b bash JESTER/scripts/run_react.sh cubert_wbo

echo "" | tee -a "$LOG"
echo "=== Ablations COMPLETE at $(date) ===" | tee -a "$LOG"
