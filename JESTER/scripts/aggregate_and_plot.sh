#!/usr/bin/env bash
# aggregate_and_plot.sh — Aggregate results, generate plots, write resume files
#
# Usage:
#   bash JESTER/scripts/aggregate_and_plot.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$(cd "$SCRIPT_DIR/../.." && pwd)"

[[ -f "../.venv/Scripts/python.exe" ]] && PYTHON="../.venv/Scripts/python.exe" \
|| { [[ -f "../.venv/bin/python" ]] && PYTHON="../.venv/bin/python"; } \
|| PYTHON="python"

echo "[$(date '+%H:%M:%S')] Aggregating results..."
"$PYTHON" JESTER/scripts/aggregate_results.py --results-dir results --output-dir results/aggregates

echo "[$(date '+%H:%M:%S')] Generating plots..."
"$PYTHON" plot.py --results-dir results --output-dir "plots/$(date +%Y%m%d_%H%M%S)"

echo "[$(date '+%H:%M:%S')] Writing Resume_adversarial_code_buggy.txt..."
"$PYTHON" JESTER/scripts/write_resume.py \
    --benchmark adversarial_code_buggy \
    --results-dir results \
    --output Resume_adversarial_code_buggy.txt

echo "[$(date '+%H:%M:%S')] Writing Resume_cubert.txt..."
"$PYTHON" JESTER/scripts/write_resume.py \
    --benchmark cubert_wbo \
    --results-dir results \
    --output Resume_cubert.txt

echo "[$(date '+%H:%M:%S')] Done."
echo "  Resume_adversarial_code_buggy.txt"
echo "  Resume_cubert.txt"
