#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
V3_ROOT = PROJECT_ROOT / "V3"
if str(V3_ROOT) not in sys.path:
    sys.path.insert(0, str(V3_ROOT))

from utils.results_aggregation import aggregate_persisted_runs, write_aggregation_artifacts


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Aggregate persisted benchmark runs across benchmarks and policy modes.",
    )
    parser.add_argument(
        "--results-dir",
        default=str(PROJECT_ROOT / "results"),
        help="Directory containing persisted run folders.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(PROJECT_ROOT / "results" / "aggregates"),
        help="Directory where aggregation artifacts will be written.",
    )
    parser.add_argument(
        "--benchmark",
        action="append",
        dest="benchmarks",
        help="Optional benchmark filter (can be repeated).",
    )
    parser.add_argument(
        "--policy-mode",
        action="append",
        dest="policy_modes",
        help="Optional policy mode filter (can be repeated).",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    aggregation = aggregate_persisted_runs(
        results_dir=args.results_dir,
        benchmarks=args.benchmarks,
        policy_modes=args.policy_modes,
    )
    outputs = write_aggregation_artifacts(
        aggregation=aggregation,
        output_dir=args.output_dir,
    )

    metadata = aggregation.get("aggregation_metadata", {})
    print("Aggregation completed")
    print(f"  runs: {metadata.get('run_count', 0)}")
    print(f"  benchmarks: {', '.join(metadata.get('benchmarks', [])) or 'none'}")
    print(f"  policy_modes: {', '.join(metadata.get('policy_modes', [])) or 'none'}")
    for key, value in outputs.items():
        print(f"  {key}: {value}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
