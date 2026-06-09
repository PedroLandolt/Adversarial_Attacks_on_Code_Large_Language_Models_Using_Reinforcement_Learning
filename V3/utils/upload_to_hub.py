"""
upload_to_hub — push pre-generated buggy code datasets to HuggingFace.

Builds a combined adversarial-code-buggy.jsonl from both source files and uploads
it to PedroLandolt/adversarial-code-buggy as data/train.jsonl.

Schema of each combined record (uniform across both benchmarks):
  source                   "mbpp" | "humaneval"
  task_id                  original task identifier
  problem_text             natural-language problem description
  buggy_code               LLM-generated code with a subtle bug
  original_code            reference solution or raw problem code
  entry_point              name of the function under test (always present)
  tests                    runnable Python test string:
                             MBPP:      assert statements joined by newlines
                             HumanEval: full check() harness function
  baseline_judge_decision  "FAIL" = judge correctly caught the bug at baseline
  baseline_judge_confidence  judge confidence score (0.0–1.0)

Usage:
    python V3/utils/upload_to_hub.py               # build combined JSONL + upload
    python V3/utils/upload_to_hub.py --build-only  # build combined JSONL, skip upload
    python V3/utils/upload_to_hub.py --repo MyOrg/my-dataset

Requires HF_TOKEN environment variable or a logged-in huggingface-cli session.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path

try:
    from dotenv import load_dotenv
    for _env in (
        Path(__file__).parent.parent.parent / ".env",
        Path(__file__).parent.parent / ".env",
    ):
        if _env.exists():
            load_dotenv(_env)
            break
except ImportError:
    pass

_HF_REPO_ID = "PedroLandolt/adversarial-code-buggy"
_DATASETS_DIR = Path(__file__).parent.parent.parent / "datasets"
_MBPP_JSONL = _DATASETS_DIR / "mbpp_pregenerated.jsonl"
_HUMANEVAL_JSONL = _DATASETS_DIR / "humaneval_pregenerated.jsonl"
_COMBINED_JSONL = _DATASETS_DIR / "adversarial-code-buggy.jsonl"


def _load_jsonl(path: Path) -> list[dict]:
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def _mbpp_entry_point(test_list: list[str]) -> str | None:
    """Extract the function name from the first MBPP assert statement.

    Handles both plain calls (assert f(...)) and parenthesised calls (assert (f(...))).
    """
    for test in test_list or []:
        m = re.match(r"assert\s+\(?\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*\(", test.strip())
        if m:
            return m.group(1)
    return None


def build_combined(
    mbpp_path: Path = _MBPP_JSONL,
    humaneval_path: Path = _HUMANEVAL_JSONL,
    output_path: Path = _COMBINED_JSONL,
    verbose: bool = True,
) -> Path:
    """Merge both source files into a single normalized JSONL.

    Unified schema — every record has the same fields:
      source       "mbpp" | "humaneval"
      task_id      original problem identifier
      problem_text natural-language description
      buggy_code   LLM-generated code with a subtle bug
      original_code reference / correct solution
      entry_point  name of the function under test (always present)
      tests        runnable Python test string
                     MBPP:      assert statements joined by newlines
                     HumanEval: the full check() harness function
      baseline_judge_decision  "FAIL" = judge correctly caught the bug
      baseline_judge_confidence  judge confidence (0.0–1.0)
    """
    records: list[dict] = []

    if mbpp_path.exists():
        raw = _load_jsonl(mbpp_path)
        for r in raw:
            test_list = r.get("test_list") or []
            records.append({
                "source": "mbpp",
                "task_id": r.get("task_id"),
                "problem_text": r.get("problem_text"),
                "buggy_code": r.get("buggy_code"),
                "original_code": r.get("original_code"),
                "entry_point": _mbpp_entry_point(test_list),
                "tests": "\n".join(test_list),
                "baseline_judge_decision": r.get("baseline_judge_decision"),
                "baseline_judge_confidence": r.get("baseline_judge_confidence"),
            })
        if verbose:
            print(f"Loaded {len(raw)} MBPP records from {mbpp_path.name}")
    else:
        print(f"Warning: {mbpp_path} not found — skipping MBPP", file=sys.stderr)

    if humaneval_path.exists():
        raw = _load_jsonl(humaneval_path)
        for r in raw:
            records.append({
                "source": "humaneval",
                "task_id": r.get("task_id"),
                "problem_text": r.get("problem_text"),
                "buggy_code": r.get("buggy_code"),
                "original_code": r.get("original_code"),
                "entry_point": r.get("entry_point"),
                "tests": r.get("test_harness") or "",
                "baseline_judge_decision": r.get("baseline_judge_decision"),
                "baseline_judge_confidence": r.get("baseline_judge_confidence"),
            })
        if verbose:
            print(f"Loaded {len(raw)} HumanEval records from {humaneval_path.name}")
    else:
        print(f"Warning: {humaneval_path} not found — skipping HumanEval", file=sys.stderr)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    if verbose:
        print(f"Combined {len(records)} records -> {output_path}")

    return output_path


def upload(
    jsonl_path: Path = _COMBINED_JSONL,
    repo_id: str = _HF_REPO_ID,
    verbose: bool = True,
) -> None:
    try:
        from huggingface_hub import HfApi
    except ImportError as exc:
        raise ImportError("pip install huggingface_hub") from exc

    if not jsonl_path.exists():
        raise FileNotFoundError(
            f"Combined JSONL not found: {jsonl_path}\n"
            "Run with --build-only first, or omit --build-only to build and upload together."
        )

    token = os.environ.get("HF_TOKEN")
    records = _load_jsonl(jsonl_path)

    if verbose:
        print(f"Uploading {len(records)} records to {repo_id} ...")

    api = HfApi(token=token)
    api.upload_file(
        path_or_fileobj=str(jsonl_path),
        path_in_repo="data/train.jsonl",
        repo_id=repo_id,
        repo_type="dataset",
        commit_message=f"Upload combined dataset ({len(records)} records: MBPP + HumanEval)",
    )

    if verbose:
        print(f"Done. View at: https://huggingface.co/datasets/{repo_id}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Build combined adversarial-code-buggy.jsonl and upload to HuggingFace."
    )
    parser.add_argument(
        "--build-only",
        action="store_true",
        help="Build the combined JSONL but do not upload.",
    )
    parser.add_argument(
        "--upload-only",
        action="store_true",
        help="Skip build — upload an existing combined JSONL.",
    )
    parser.add_argument(
        "--repo",
        default=_HF_REPO_ID,
        help=f"HuggingFace repo ID (default: {_HF_REPO_ID})",
    )
    parser.add_argument(
        "--combined-jsonl",
        default=str(_COMBINED_JSONL),
        help=f"Output path for the combined JSONL (default: {_COMBINED_JSONL})",
    )
    args = parser.parse_args()

    combined_path = Path(args.combined_jsonl)

    if not args.upload_only:
        build_combined(output_path=combined_path)

    if not args.build_only:
        if not os.environ.get("HF_TOKEN"):
            print("Warning: HF_TOKEN not set. Authentication may fail.", file=sys.stderr)
        upload(jsonl_path=combined_path, repo_id=args.repo)
