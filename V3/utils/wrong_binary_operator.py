"""
wrong_binary_operator — dataset builder and loader.

Builds a benchmark where every sample has pre-known buggy code produced by
substituting one binary or comparison operator with an incorrect counterpart.
Tests are guaranteed to fail; the pipeline enters at Step 3 (attack generation)
and skips Steps 1–2 entirely.

Usage (build dataset, run once):
    python V3/utils/wrong_binary_operator.py

Usage (programmatic):
    from utils.wrong_binary_operator import load_wrong_binary_operator_dataset
    samples = load_wrong_binary_operator_dataset("datasets/wrong_binary_operator.jsonl")
"""

from __future__ import annotations

import ast
import json
import subprocess
import sys
from pathlib import Path
from typing import Iterator

_MBPP_DATASET_PATH = "google-research-datasets/mbpp"
_MBPP_DATASET_REVISION = "4bb6404fdc6cacfda99d4ac4205087b89d32030c"

# ---------------------------------------------------------------------------
# Operator substitution map
# Each entry: AST operator type → "wrong" replacement type.
# Chosen so that the substitution reliably breaks numerical/logical tests.
# ---------------------------------------------------------------------------
_SUBSTITUTION_MAP: dict[type, type] = {
    # Arithmetic
    ast.Add: ast.Sub,
    ast.Sub: ast.Add,
    ast.Mult: ast.Add,
    ast.Div: ast.Mult,
    ast.FloorDiv: ast.Add,
    ast.Mod: ast.Add,
    ast.Pow: ast.Mult,
    # Bitwise
    ast.BitAnd: ast.BitOr,
    ast.BitOr: ast.BitAnd,
    ast.BitXor: ast.BitAnd,
    # Comparison (applied to the first comparator in a Compare node)
    ast.Gt: ast.Lt,
    ast.Lt: ast.Gt,
    ast.GtE: ast.LtE,
    ast.LtE: ast.GtE,
    ast.Eq: ast.NotEq,
    ast.NotEq: ast.Eq,
}

_OP_SYMBOL: dict[type, str] = {
    ast.Add: "+",    ast.Sub: "-",   ast.Mult: "*",   ast.Div: "/",
    ast.FloorDiv: "//", ast.Mod: "%", ast.Pow: "**",
    ast.BitAnd: "&", ast.BitOr: "|", ast.BitXor: "^",
    ast.Gt: ">",   ast.Lt: "<",    ast.GtE: ">=",  ast.LtE: "<=",
    ast.Eq: "==",  ast.NotEq: "!=",
}


class _FirstOpSubstitutor(ast.NodeTransformer):
    """Replaces the first substitutable BinOp or Compare operator."""

    def __init__(self) -> None:
        self.done = False
        self.substitution: dict | None = None

    def visit_BinOp(self, node: ast.BinOp) -> ast.BinOp:
        if not self.done:
            op_type = type(node.op)
            if op_type in _SUBSTITUTION_MAP:
                new_op_type = _SUBSTITUTION_MAP[op_type]
                self.substitution = {
                    "node_type": "BinOp",
                    "original_op": _OP_SYMBOL.get(op_type, op_type.__name__),
                    "new_op": _OP_SYMBOL.get(new_op_type, new_op_type.__name__),
                    "lineno": getattr(node, "lineno", None),
                    "col_offset": getattr(node, "col_offset", None),
                }
                node.op = new_op_type()
                self.done = True
        self.generic_visit(node)
        return node

    def visit_Compare(self, node: ast.Compare) -> ast.Compare:
        if not self.done and node.ops:
            op_type = type(node.ops[0])
            if op_type in _SUBSTITUTION_MAP:
                new_op_type = _SUBSTITUTION_MAP[op_type]
                self.substitution = {
                    "node_type": "Compare",
                    "original_op": _OP_SYMBOL.get(op_type, op_type.__name__),
                    "new_op": _OP_SYMBOL.get(new_op_type, new_op_type.__name__),
                    "lineno": getattr(node, "lineno", None),
                    "col_offset": getattr(node, "col_offset", None),
                }
                node.ops[0] = new_op_type()
                self.done = True
        self.generic_visit(node)
        return node


def substitute_first_operator(code: str) -> tuple[str, dict] | None:
    """
    Parse code and substitute the first binary/comparison operator.
    Returns (modified_code, substitution_info) or None if no operator found.
    """
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return None

    substitutor = _FirstOpSubstitutor()
    new_tree = substitutor.visit(tree)

    if not substitutor.done:
        return None

    ast.fix_missing_locations(new_tree)
    try:
        modified_code = ast.unparse(new_tree)
    except Exception:
        return None

    return modified_code, substitutor.substitution


def _build_verification_program(code: str, test_list: list[str]) -> str:
    """Assemble code + test assertions into a single runnable program."""
    parts = [code]
    for test in test_list:
        assert_expr = str(test)
        message = assert_expr[len("assert "):] if assert_expr.startswith("assert ") else assert_expr
        parts.append(f"{assert_expr}, {repr(message)}")
    return "\n".join(parts)


def _tests_fail(buggy_code: str, test_list: list[str], timeout: float = 5.0) -> bool:
    """Return True if the test assertions fail against the buggy code."""
    if not test_list:
        return False
    program = _build_verification_program(buggy_code, test_list)
    try:
        result = subprocess.run(
            [sys.executable, "-c", program],
            timeout=timeout,
            capture_output=True,
        )
        return result.returncode != 0
    except (subprocess.TimeoutExpired, OSError):
        return False


def build_wrong_binary_operator_dataset(
    output_path: str | Path,
    limit: int | None = None,
    verbose: bool = True,
) -> int:
    """
    Build the wrong_binary_operator dataset from MBPP and write to output_path (JSONL).

    For each MBPP sample:
      1. Find the first binary/comparison operator in the reference solution.
      2. Substitute it with a wrong counterpart.
      3. Run the MBPP test assertions locally (no Docker) to confirm they fail.
      4. Keep the record only if tests fail.

    Returns the number of records saved.
    """
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise ImportError(
            "The 'datasets' package is required to build the dataset. "
            "Install it with: pip install datasets"
        ) from exc

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    all_records: list[dict] = []
    for split in ("train", "validation", "test", "prompt"):
        ds = load_dataset(
            _MBPP_DATASET_PATH,
            name="full",
            split=split,
            revision=_MBPP_DATASET_REVISION,
            trust_remote_code=False,
        )
        all_records.extend(ds)

    if limit:
        all_records = all_records[:limit]

    total = len(all_records)
    saved = 0
    skipped_no_op = 0
    skipped_tests_pass = 0

    with open(output_path, "w", encoding="utf-8") as f:
        for i, record in enumerate(all_records):
            code = record.get("code", "")
            task_id = record.get("task_id")
            problem_text = record.get("text", "")
            test_list = record.get("test_list", [])

            if verbose and i % 50 == 0:
                print(f"  [{i}/{total}] saved={saved} no_op={skipped_no_op} tests_pass={skipped_tests_pass}")

            if not code or not test_list:
                skipped_no_op += 1
                continue

            result = substitute_first_operator(code)
            if result is None:
                skipped_no_op += 1
                continue

            buggy_code, substitution = result

            if not _tests_fail(buggy_code, test_list):
                skipped_tests_pass += 1
                continue

            entry = {
                "task_id": task_id,
                "problem_text": problem_text,
                "original_code": code,
                "buggy_code": buggy_code,
                "test_list": test_list,
                "operator_substitution": substitution,
                "source_benchmark": "mbpp",
            }
            f.write(json.dumps(entry) + "\n")
            saved += 1

    if verbose:
        print(f"\nDataset built: {saved} records saved to {output_path}")
        print(f"  Skipped (no substitutable op): {skipped_no_op}")
        print(f"  Skipped (tests still pass after substitution): {skipped_tests_pass}")

    return saved


def load_wrong_binary_operator_dataset(path: str | Path) -> list[dict]:
    """Load all records from a JSONL file built by build_wrong_binary_operator_dataset."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Wrong binary operator dataset not found at {path}. "
            f"Build it first with: python V3/utils/wrong_binary_operator.py"
        )
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def iter_wrong_binary_operator_dataset(path: str | Path) -> Iterator[dict]:
    """Lazily iterate records from the dataset file."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Wrong binary operator dataset not found at {path}."
        )
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Build the wrong_binary_operator dataset from MBPP."
    )
    parser.add_argument(
        "--output",
        default="datasets/wrong_binary_operator.jsonl",
        help="Output JSONL path (default: datasets/wrong_binary_operator.jsonl)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Cap the number of MBPP records to process (default: all)",
    )
    args = parser.parse_args()

    print(f"Building wrong_binary_operator dataset -> {args.output}")
    n = build_wrong_binary_operator_dataset(
        output_path=args.output,
        limit=args.limit,
        verbose=True,
    )
    print(f"Done. {n} records ready for experiments.")
