#!/usr/bin/env bash
# run_master_auto.sh — Full experiment chain after Section 2.
#
# Waits for Section 2 to finish (20 stored runs), then runs in order:
#   1. Section 1 remaining  (5 runs,  ~9h)
#   2. Section 3 codellama  (20 runs, ~33h)
#   3. Section 4 starcoder  (20 runs, ~33h)
#   4. Ablations A+B        (32 runs, ~23h)
#
# Each sub-script has GPU cooldown (<65°C) between runs and skip logic.
# All output is tee'd to master_auto.log.

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$(cd "$SCRIPT_DIR/../.." && pwd)"

LOG="$PWD/master_auto.log"
echo "=== Master auto-run started at $(date) ===" | tee -a "$LOG"

# ── Wait for Section 2 to finish (20 stored runs) ─────────────────────────────
echo "[WAIT] Waiting for Section 2 to complete (need 20 stored runs)..." | tee -a "$LOG"
while true; do
    count=$(ls stored_results/section-2/ 2>/dev/null | wc -l | tr -d ' ')
    echo "[WAIT] Section 2: ${count}/20 stored  —  $(date '+%H:%M')" | tee -a "$LOG"
    if [[ "$count" -ge 20 ]]; then
        echo "[WAIT] Section 2 complete. Proceeding." | tee -a "$LOG"
        break
    fi
    sleep 300
done

# ── Section 1 remaining ────────────────────────────────────────────────────────
echo "" | tee -a "$LOG"
echo "================================================================" | tee -a "$LOG"
echo "PHASE 1: Section 1 remaining — $(date)" | tee -a "$LOG"
echo "================================================================" | tee -a "$LOG"
bash "$SCRIPT_DIR/run_section1_remaining_auto.sh" 2>&1 | tee -a "$LOG"
echo "PHASE 1 DONE — $(date)" | tee -a "$LOG"

# ── Section 3 ─────────────────────────────────────────────────────────────────
echo "" | tee -a "$LOG"
echo "================================================================" | tee -a "$LOG"
echo "PHASE 2: Section 3 (codellama) — $(date)" | tee -a "$LOG"
echo "================================================================" | tee -a "$LOG"
bash "$SCRIPT_DIR/run_section3_auto.sh" 2>&1 | tee -a "$LOG"
echo "PHASE 2 DONE — $(date)" | tee -a "$LOG"

# ── Section 4 ─────────────────────────────────────────────────────────────────
echo "" | tee -a "$LOG"
echo "================================================================" | tee -a "$LOG"
echo "PHASE 3: Section 4 (starcoder) — $(date)" | tee -a "$LOG"
echo "================================================================" | tee -a "$LOG"
bash "$SCRIPT_DIR/run_section4_auto.sh" 2>&1 | tee -a "$LOG"
echo "PHASE 3 DONE — $(date)" | tee -a "$LOG"

# ── Ablations A + B ───────────────────────────────────────────────────────────
echo "" | tee -a "$LOG"
echo "================================================================" | tee -a "$LOG"
echo "PHASE 4: Ablations A+B — $(date)" | tee -a "$LOG"
echo "================================================================" | tee -a "$LOG"
bash "$SCRIPT_DIR/run_ablations_auto.sh" 2>&1 | tee -a "$LOG"
echo "PHASE 4 DONE — $(date)" | tee -a "$LOG"

echo "" | tee -a "$LOG"
echo "=== ALL EXPERIMENTS COMPLETE at $(date) ===" | tee -a "$LOG"
