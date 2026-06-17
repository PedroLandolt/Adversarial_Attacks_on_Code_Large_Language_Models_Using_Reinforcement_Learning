"""
operator_mutation — builder for Pedro's synthesized_wbo dataset.

Applies one binary/comparison operator substitution to each MBPP and HumanEval
reference solution via AST, runs tests locally to confirm they fail, and saves
the filtered records to a JSONL file.

The resulting file is the source for the `synthesized_wbo` benchmark.

Usage (build dataset — run once):
    python JESTER/utils/operator_mutation.py

Usage (programmatic):
    from utils.operator_mutation import load_synthesized_wbo
    records = load_synthesized_wbo("datasets/synthesized_wbo.jsonl")
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
# ---------------------------------------------------------------------------
_SUBSTITUTION_MAP: dict[type, type] = {
    ast.Add: ast.Sub,
    ast.Sub: ast.Add,
    ast.Mult: ast.Add,
    ast.Div: ast.Mult,
    ast.FloorDiv: ast.Add,
    ast.Mod: ast.Add,
    ast.Pow: ast.Mult,
    ast.BitAnd: ast.BitOr,
    ast.BitOr: ast.BitAnd,
    ast.BitXor: ast.BitAnd,
    ast.Gt: ast.Lt,
    ast.Lt: ast.Gt,
    ast.GtE: ast.LtE,
    ast.LtE: ast.GtE,
    ast.Eq: ast.NotEq,
    ast.NotEq: ast.Eq,
}

_OP_SYMBOL: dict[type, str] = {
    ast.Add: "+",    ast.Sub: "-",    ast.Mult: "*",   ast.Div: "/",
    ast.FloorDiv: "//", ast.Mod: "%", ast.Pow: "**",
    ast.BitAnd: "&", ast.BitOr: "|", ast.BitXor: "^",
    ast.Gt: ">",   ast.Lt: "<",    ast.GtE: ">=",  ast.LtE: "<=",
    ast.Eq: "==",  ast.NotEq: "!=",
}


class _FirstOpSubstitutor(ast.NodeTransformer):
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
    """Substitute the first binary/comparison operator in code.
    Returns (modified_code, substitution_info) or None if no operator found."""
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
        return ast.unparse(new_tree), substitutor.substitution
    except Exception:
        return None


def _run_program(code: str, timeout: float = 5.0) -> bool:
    """Run a Python program string locally. Returns True if exit code is 0."""
    try:
        result = subprocess.run(
            [sys.executable, "-c", code],
            timeout=timeout,
            capture_output=True,
        )
        return result.returncode == 0
    except (subprocess.TimeoutExpired, OSError):
        return False


def _mbpp_tests_fail(buggy_code: str, test_list: list[str]) -> bool:
    if not test_list:
        return False
    parts = [buggy_code]
    for test in test_list:
        expr = str(test)
        msg = expr[len("assert "):] if expr.startswith("assert ") else expr
        parts.append(f"{expr}, {repr(msg)}")
    return not _run_program("\n".join(parts))


def _humaneval_tests_fail(buggy_code: str, test_harness: str, entry_point: str) -> bool:
    if not test_harness:
        return False
    program = "\n".join([
        buggy_code,
        test_harness,
        f"check({entry_point})" if f"check({entry_point})" not in test_harness else "",
    ])
    return not _run_program(program.strip())


# ---------------------------------------------------------------------------
# Dataset builder
# ---------------------------------------------------------------------------

def build_synthesized_wbo_dataset(
    output_path: str | Path,
    include_mbpp: bool = True,
    include_humaneval: bool = True,
    limit: int | None = None,
    verbose: bool = True,
) -> int:
    """Build the synthesized_wbo dataset from MBPP and/or HumanEval.

    For each sample:
      1. Find the first binary/comparison operator in the reference solution.
      2. Substitute it with a wrong counterpart.
      3. Run tests locally to confirm they fail.
      4. Save only records where tests fail.

    Returns the number of records saved.
    """
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise ImportError("pip install datasets") from exc

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    saved = 0
    skipped_no_op = 0
    skipped_tests_pass = 0

    with open(output_path, "w", encoding="utf-8") as f:

        # --- MBPP ---
        if include_mbpp:
            mbpp_records: list[dict] = []
            for split in ("train", "validation", "test", "prompt"):
                ds = load_dataset(
                    _MBPP_DATASET_PATH,
                    name="full",
                    split=split,
                    revision=_MBPP_DATASET_REVISION,
                    trust_remote_code=False,
                )
                mbpp_records.extend(ds)

            if limit:
                mbpp_records = mbpp_records[:limit]

            total_mbpp = len(mbpp_records)
            if verbose:
                print(f"Processing {total_mbpp} MBPP samples...")

            for i, record in enumerate(mbpp_records):
                code = record.get("code", "")
                task_id = record.get("task_id")
                problem_text = record.get("text", "")
                test_list = record.get("test_list", [])

                if verbose and i % 100 == 0:
                    print(f"  MBPP [{i}/{total_mbpp}] saved={saved}")

                if not code or not test_list:
                    skipped_no_op += 1
                    continue

                result = substitute_first_operator(code)
                if result is None:
                    skipped_no_op += 1
                    continue

                buggy_code, substitution = result
                if not _mbpp_tests_fail(buggy_code, test_list):
                    skipped_tests_pass += 1
                    continue

                entry = {
                    "task_id": task_id,
                    "problem_text": problem_text,
                    "original_code": code,
                    "buggy_code": buggy_code,
                    "test_list": test_list,
                    "test_harness": None,
                    "entry_point": None,
                    "operator_substitution": substitution,
                    "source_benchmark": "mbpp",
                }
                f.write(json.dumps(entry) + "\n")
                saved += 1

        # --- HumanEval ---
        if include_humaneval:
            try:
                he_ds = load_dataset("openai/openai_humaneval", split="test", trust_remote_code=False)
            except Exception as exc:
                if verbose:
                    print(f"  Warning: could not load HumanEval: {exc}")
                he_ds = []

            he_records = list(he_ds)
            if limit:
                he_records = he_records[:limit]

            total_he = len(he_records)
            if verbose:
                print(f"Processing {total_he} HumanEval samples...")

            for i, record in enumerate(he_records):
                code = record.get("canonical_solution", "")
                task_id = record.get("task_id", f"HumanEval/{i}")
                problem_text = record.get("prompt", "")
                test_harness = record.get("test", "")
                entry_point = record.get("entry_point", "")

                if verbose and i % 50 == 0:
                    print(f"  HumanEval [{i}/{total_he}] saved={saved}")

                if not code or not test_harness:
                    skipped_no_op += 1
                    continue

                # Prepend the prompt (function signature + docstring) for substitution
                full_code = problem_text + code if not code.startswith("def ") else code
                result = substitute_first_operator(full_code)
                if result is None:
                    # Try just the solution body
                    result = substitute_first_operator(code)
                if result is None:
                    skipped_no_op += 1
                    continue

                buggy_code, substitution = result
                if not _humaneval_tests_fail(buggy_code, test_harness, entry_point):
                    skipped_tests_pass += 1
                    continue

                entry = {
                    "task_id": task_id,
                    "problem_text": problem_text,
                    "original_code": code,
                    "buggy_code": buggy_code,
                    "test_list": [],
                    "test_harness": test_harness,
                    "entry_point": entry_point,
                    "operator_substitution": substitution,
                    "source_benchmark": "humaneval",
                }
                f.write(json.dumps(entry) + "\n")
                saved += 1

    if verbose:
        print(f"\nDone. {saved} records saved to {output_path}")
        print(f"  Skipped (no substitutable op): {skipped_no_op}")
        print(f"  Skipped (tests still pass after substitution): {skipped_tests_pass}")

    return saved


def load_synthesized_wbo(path: str | Path) -> list[dict]:
    """Load all records from the synthesized_wbo JSONL file."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"synthesized_wbo dataset not found at {path}. "
            f"Build it with: python JESTER/utils/operator_mutation.py"
        )
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def iter_synthesized_wbo(path: str | Path) -> Iterator[dict]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"synthesized_wbo dataset not found at {path}.")
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Build the synthesized_wbo dataset from MBPP and HumanEval."
    )
    parser.add_argument(
        "--output",
        default="datasets/synthesized_wbo.jsonl",
        help="Output JSONL path (default: datasets/synthesized_wbo.jsonl)",
    )
    parser.add_argument("--no-mbpp", action="store_true", help="Skip MBPP")
    parser.add_argument("--no-humaneval", action="store_true", help="Skip HumanEval")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    print(f"Building synthesized_wbo dataset -> {args.output}")
    n = build_synthesized_wbo_dataset(
        output_path=args.output,
        include_mbpp=not args.no_mbpp,
        include_humaneval=not args.no_humaneval,
        limit=args.limit,
        verbose=True,
    )
    print(f"Done. {n} records ready for experiments.")
