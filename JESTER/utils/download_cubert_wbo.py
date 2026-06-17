"""
download_cubert_wbo — builder for the cubert_wbo local dataset file.

Downloads the CuBERT ETHPy150 wrong_binary_operator subset from HuggingFace,
filters to records where label == "Wrong binary operator", and saves up to
10 000 records to a local JSONL file.

This must be run ONCE before any Inspect experiment uses `benchmark=cubert_wbo`.
Loading from HuggingFace at Inspect task-init time fails due to an httpx/anyio
client conflict; loading from a local file has no such issue.

Usage:
    python JESTER/utils/download_cubert_wbo.py
    python JESTER/utils/download_cubert_wbo.py --limit 100   # smoke test
    python JESTER/utils/download_cubert_wbo.py --output datasets/cubert_wbo.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_HF_PATH = "claudios/cubert_ETHPy150Open"
_HF_CONFIG = "wrong_binary_operator_datasets"
_LABEL_FILTER = "Wrong binary operator"
_DEFAULT_LIMIT = 10_000
_DEFAULT_OUTPUT = "datasets/cubert_wbo.jsonl"


def download_cubert_wbo(
    output_path: str | Path,
    limit: int = _DEFAULT_LIMIT,
    verbose: bool = True,
) -> int:
    """Download and filter the CuBERT WBO subset.

    Returns the number of records saved.
    """
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise ImportError("pip install datasets") from exc

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if verbose:
        print(f"Downloading {_HF_PATH} (config={_HF_CONFIG}) from HuggingFace...")

    ds = load_dataset(
        _HF_PATH,
        name=_HF_CONFIG,
        split="train",
        trust_remote_code=False,
    )

    if verbose:
        print(f"Loaded {len(ds)} total records. Filtering to label='{_LABEL_FILTER}'...")

    wrong_only = [r for r in ds if r.get("label") == _LABEL_FILTER]

    if verbose:
        print(f"Filtered: {len(wrong_only)} wrong_binary_operator records.")

    if limit:
        wrong_only = wrong_only[:limit]

    if verbose:
        print(f"Saving {len(wrong_only)} records to {output_path}...")

    saved = 0
    with open(output_path, "w", encoding="utf-8") as f:
        for record in wrong_only:
            f.write(json.dumps({
                "function": record.get("function", ""),
                "label": record.get("label", ""),
                "info": record.get("info", ""),
            }) + "\n")
            saved += 1

    if verbose:
        print(f"Done. {saved} records saved to {output_path}")
        print("=== cubert_wbo BUILD COMPLETE ===")

    return saved


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Download and save the CuBERT wrong_binary_operator subset."
    )
    parser.add_argument("--output", default=_DEFAULT_OUTPUT)
    parser.add_argument(
        "--limit",
        type=int,
        default=_DEFAULT_LIMIT,
        help=f"Max records to save (default: {_DEFAULT_LIMIT})",
    )
    args = parser.parse_args()

    n = download_cubert_wbo(output_path=args.output, limit=args.limit)
    sys.exit(0 if n > 0 else 1)
