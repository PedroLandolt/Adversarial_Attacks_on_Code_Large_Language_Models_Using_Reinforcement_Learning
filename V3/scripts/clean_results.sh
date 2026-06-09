#!/usr/bin/env bash
# clean_results.sh — archive existing results and logs before a fresh experiment run.
#
# Archives (does NOT delete) all current run directories and .eval log files
# to results/archive/<timestamp>/. Safe to re-run multiple times.
#
# Run from project root:
#   bash V3/scripts/clean_results.sh

set -euo pipefail

python - <<'PYEOF'
import os, shutil, glob, datetime

archive_ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
archive_dir = os.path.join("results", "archive", archive_ts)
os.makedirs(archive_dir, exist_ok=True)

moved = 0

# Archive result run directories (everything in results/ except archive/ itself)
if os.path.isdir("results"):
    for item in os.listdir("results"):
        if item == "archive":
            continue
        src = os.path.join("results", item)
        shutil.move(src, archive_dir)
        moved += 1

# Archive .eval log files from logs/
for f in glob.glob(os.path.join("logs", "*.eval")):
    shutil.move(f, archive_dir)
    moved += 1

if moved > 0:
    print(f"Archived {moved} item(s) -> {archive_dir}")
else:
    print("Nothing to archive — results/ and logs/ are already clean.")
PYEOF
