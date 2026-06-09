#!/usr/bin/env bash
# store_results.sh — store one completed run into a named folder.
#
# Creates stored_results/<section>/<run>/ containing:
#   *.eval          <- latest .eval file from logs/
#   results/        <- everything from results/ (excluding archive/)
#
# Run immediately after each experiment command.
#
# Usage:
#   bash V3/scripts/store_results.sh section-1/random-llama31-qwen-adversarial_code_buggy

set -euo pipefail
NAME="${1:?Usage: bash V3/scripts/store_results.sh <section>/<run-name>}"

python - "$NAME" <<'PYEOF'
import os, shutil, glob, sys

name = sys.argv[1]
dest = os.path.join("stored_results", name)

if os.path.exists(dest):
    print(f"ERROR: {dest} already exists. Delete it or pick a different name.")
    sys.exit(1)

os.makedirs(dest, exist_ok=True)

# Move latest .eval file from logs/
evals = sorted(glob.glob("logs/*.eval"))
if evals:
    latest_eval = evals[-1]
    shutil.move(latest_eval, dest)
    print(f"  .eval  -> {dest}/")
else:
    print("  WARNING: no .eval file found in logs/")

# Move results/ contents into dest/results/
dest_results = os.path.join(dest, "results")
os.makedirs(dest_results, exist_ok=True)
moved = 0
if os.path.isdir("results"):
    for item in os.listdir("results"):
        if item == "archive":
            continue
        shutil.move(os.path.join("results", item), dest_results)
        moved += 1

print(f"  results ({moved} run folder(s)) -> {dest}/results/")
print(f"Stored -> {dest}/")
print("results/ is now empty — ready for the next run.")
PYEOF
