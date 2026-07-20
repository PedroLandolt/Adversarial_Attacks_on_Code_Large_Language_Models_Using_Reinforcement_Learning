#!/usr/bin/env bash
# run_forced_tactic.sh — Per-tactic isolation runs for RQ2 + Appendix B.
#
# Runs each of the 9 tactics independently, one-shot, test split.
# Covers all 4 judges: qwen (S1), deepseek (S2), codellama (S3), starcoder (S4).
# starcoder is LAST — stop the script after codellama if you want only 3 judges.
#
# Store path: rq2/{tactic}-llama31-{judge}-{corpus}
# Est. time:  qwen ~4h | deepseek ~4h | codellama ~1.5h | starcoder ~8h
#
# Usage:
#   bash JESTER/scripts/run_forced_tactic.sh            # all 4 judges
#   bash JESTER/scripts/run_forced_tactic.sh qwen       # single judge
#   bash JESTER/scripts/run_forced_tactic.sh qwen deepseek codellama   # skip starcoder

set -euo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/../.."

LOG="$PWD/forced_tactic_auto.log"
echo "=== Forced-tactic runs started at $(date) ===" | tee -a "$LOG"

[[ -f "../.venv/Scripts/inspect" ]] && INSPECT="../.venv/Scripts/inspect" \
|| { [[ -f "../.venv/bin/inspect" ]] && INSPECT="../.venv/bin/inspect"; } \
|| INSPECT="inspect"

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
    echo "[DONE] $1"
}

run_one() {
    local store_name="$1" judge_model="$2" corpus="$3" tactic="$4"
    if [ -d "stored_results/$store_name" ]; then
        echo "[SKIP] $store_name" | tee -a "$LOG"
        return 0
    fi
    wait_for_cool
    echo "" | tee -a "$LOG"
    echo "============================================================" | tee -a "$LOG"
    echo "[START] $store_name  --  $(date)" | tee -a "$LOG"
    echo "============================================================" | tee -a "$LOG"
    "$INSPECT" eval JESTER/adversarial_attack.py@adversarial_code_llm \
        --model ollama/llama3.1:8b \
        --max-samples 10 \
        --limit 1000 \
        -T benchmark="$corpus" \
        -T policy_mode=random_choice \
        -T mutation_strategy=react \
        -T experiment_mode=one_shot \
        -T use_llm_judge=True \
        -T target_model="$judge_model" \
        -T selector_model="ollama/llama3.1:8b" \
        -T max_iterations=1 \
        -T forced_tactic="$tactic" \
        -T experiment_split=test \
        -T split_definition="${corpus}:70_15_15:test" \
        2>&1 | tee -a "$LOG"
    bash JESTER/scripts/store_results.sh "$store_name" 2>&1 | tee -a "$LOG"
    mark_done "$store_name" | tee -a "$LOG"
    echo "[DONE] $store_name  --  $(date)" | tee -a "$LOG"
}

TACTICS=(
    legacy_injection
    legacy_output
    legacy_semantic
    legacy_cot
    taxonomy_roleplay
    taxonomy_appeal_to_authority
    taxonomy_formatting_smuggling
    taxonomy_recursion_crescendo
    taxonomy_crowding
)

CORPORA=(adversarial_code_buggy cubert_wbo)

run_judge() {
    local judge_tag="$1" judge_model="$2"
    echo "" | tee -a "$LOG"
    echo "################################################################" | tee -a "$LOG"
    echo "JUDGE: $judge_tag ($judge_model)  --  $(date)" | tee -a "$LOG"
    echo "################################################################" | tee -a "$LOG"
    for corpus in "${CORPORA[@]}"; do
        for tactic in "${TACTICS[@]}"; do
            store="rq2/${tactic}-llama31-${judge_tag}-${corpus}"
            run_one "$store" "$judge_model" "$corpus" "$tactic"
        done
    done
    echo "JUDGE $judge_tag DONE -- $(date)" | tee -a "$LOG"
}

# Determine which judges to run
if [[ $# -gt 0 ]]; then
    JUDGES=("$@")
else
    JUDGES=(qwen deepseek codellama starcoder)
fi

for judge in "${JUDGES[@]}"; do
    case "$judge" in
        qwen)      run_judge "qwen"      "ollama/qwen2.5-coder:7b"    ;;
        deepseek)  run_judge "deepseek"  "ollama/deepseek-coder:6.7b" ;;
        codellama) run_judge "codellama" "ollama/codellama:7b"        ;;
        starcoder) run_judge "starcoder" "ollama/starcoder2:7b"       ;;
        *) echo "Unknown judge: $judge (valid: qwen deepseek codellama starcoder)" | tee -a "$LOG" ;;
    esac
done

echo "" | tee -a "$LOG"
echo "=== Forced-tactic COMPLETE at $(date) ===" | tee -a "$LOG"
powershell -c "[console]::beep(880,200);[console]::beep(1100,200);[console]::beep(1320,500)" 2>/dev/null || true
