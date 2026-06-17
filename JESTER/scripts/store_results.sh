#!/usr/bin/env bash
# store_results.sh — store one completed run into a named folder.
#
# Creates stored_results/<section>/<run>/ containing:
#   *.eval          <- ALL .eval files from logs/ (eval runs produce two: validation + test)
#   *.log           <- latest experiment log from logs/experiment/
#   results/        <- everything from results/ (excluding archive/)
#   *.json          <- weights file + entropy curve (only when WEIGHTS_FILE is given)
#
# Run immediately after each experiment command.
#
# Usage:
#   bash JESTER/scripts/store_results.sh section-1/random-llama31-qwen-adversarial_code_buggy
#   bash JESTER/scripts/store_results.sh section-1/ucb1-eval-llama31-qwen-adversarial_code_buggy weights/acb_ucb1_llama31_qwen.json

set -euo pipefail
NAME="${1:?Usage: bash JESTER/scripts/store_results.sh <section>/<run-name> [weights-file]}"
WEIGHTS_FILE="${2:-}"

python - "$NAME" "$WEIGHTS_FILE" <<'PYEOF'
import os, shutil, glob, sys

name        = sys.argv[1]
weights_file = sys.argv[2]
dest = os.path.join("stored_results", name)

if os.path.exists(dest):
    print(f"ERROR: {dest} already exists. Delete it or pick a different name.")
    sys.exit(1)

os.makedirs(dest, exist_ok=True)

# Move ALL .eval files from logs/ (eval runs produce validation + test, both needed)
evals = sorted(glob.glob("logs/*.eval"))
if evals:
    for ef in evals:
        shutil.move(ef, dest)
    print(f"  {len(evals)} .eval file(s) -> {dest}/")
else:
    print("  WARNING: no .eval file found in logs/")

# Move latest experiment log from logs/experiment/
logs = sorted(glob.glob("logs/experiment/*.log"))
if logs:
    shutil.move(logs[-1], dest)
    print(f"  experiment log ({os.path.basename(logs[-1])}) -> {dest}/")
else:
    print("  WARNING: no experiment log found in logs/experiment/")

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

# Optionally move weights file + entropy curve
if weights_file:
    weights_file = os.path.abspath(weights_file)
    if os.path.isfile(weights_file):
        shutil.move(weights_file, dest)
        print(f"  weights -> {dest}/")
    else:
        print(f"  WARNING: weights file not found: {weights_file}")
    entropy_file = weights_file.replace(".json", "_entropy_curve.json")
    if os.path.isfile(entropy_file):
        shutil.move(entropy_file, dest)
        print(f"  entropy curve -> {dest}/")

print(f"Stored -> {dest}/")
print("results/ is now empty — ready for the next run.")
PYEOF
