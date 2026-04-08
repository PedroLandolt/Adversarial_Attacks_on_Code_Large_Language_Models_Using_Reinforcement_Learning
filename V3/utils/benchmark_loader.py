from __future__ import annotations

from dataclasses import dataclass
import inspect
from typing import Any

from inspect_evals.mbpp import mbpp

try:
    from inspect_evals.humaneval import humaneval
except ImportError:  # pragma: no cover - depends on installed Inspect package version
    humaneval = None


@dataclass(frozen=True)
class BenchmarkSpec:
    benchmark_name: str
    problem_text: str
    entry_point: str | None
    test_list: list[str] | None
    test_harness: str | None
    normalization_requirements: dict[str, Any]


def normalize_benchmark_name(benchmark_name: str) -> str:
    normalized = str(benchmark_name).strip().lower()
    if normalized not in {"mbpp", "humaneval"}:
        raise ValueError(f"Unsupported benchmark: {benchmark_name}")
    return normalized


def load_benchmark_task(benchmark_name: str, *, temperature: float):
    normalized = normalize_benchmark_name(benchmark_name)
    if normalized == "mbpp":
        return mbpp(temperature=temperature)

    if humaneval is None:
        raise ImportError("HumanEval benchmark is not available in this Inspect install.")
    try:
        humaneval_signature = inspect.signature(humaneval)
    except (TypeError, ValueError):
        humaneval_signature = None

    if humaneval_signature is not None and "temperature" in humaneval_signature.parameters:
        return humaneval(temperature=temperature)

    try:
        return humaneval()
    except TypeError as exc:
        if "temperature" in str(exc):
            return humaneval()
        raise


def extract_benchmark_spec(task_state, benchmark_name: str) -> BenchmarkSpec:
    normalized = normalize_benchmark_name(benchmark_name)
    metadata = getattr(task_state, "metadata", {}) or {}
    problem_text = _extract_problem_text(task_state)

    if normalized == "mbpp":
        test_list = metadata.get("test_list")
        if not isinstance(test_list, list) or not test_list:
            target = getattr(task_state, "target", None)
            if isinstance(target, list) and target:
                test_list = target
        normalized_tests = [str(test_case) for test_case in (test_list or [])]
        return BenchmarkSpec(
            benchmark_name="mbpp",
            problem_text=problem_text,
            entry_point=str(metadata.get("entry_point")) if metadata.get("entry_point") else None,
            test_list=normalized_tests,
            test_harness=None,
            normalization_requirements={
                "extract_python_code": True,
                "requires_entry_point": False,
                "test_format": "assert_list",
            },
        )

    test_harness = metadata.get("test")
    if test_harness is None:
        target = getattr(task_state, "target", None)
        if isinstance(target, str):
            test_harness = target

    entry_point = metadata.get("entry_point")
    return BenchmarkSpec(
        benchmark_name="humaneval",
        problem_text=problem_text,
        entry_point=str(entry_point) if entry_point else None,
        test_list=None,
        test_harness=str(test_harness) if test_harness is not None else None,
        normalization_requirements={
            "extract_python_code": True,
            "requires_entry_point": True,
            "test_format": "python_harness",
        },
    )


def build_verification_program(executable_code: str | None, benchmark_spec: BenchmarkSpec) -> str | None:
    if executable_code is None:
        return None

    if benchmark_spec.benchmark_name == "mbpp":
        if not benchmark_spec.test_list:
            return None
        program_parts = [str(executable_code)]
        for test_case in benchmark_spec.test_list:
            assert_expr = str(test_case)
            message = assert_expr[len("assert ") :] if assert_expr.startswith("assert ") else assert_expr
            program_parts.append(f"{assert_expr}, {repr(message)}")
        return "\n".join(program_parts)

    if not benchmark_spec.test_harness:
        return None

    program_parts = [str(executable_code), str(benchmark_spec.test_harness)]
    explicit_check_call = (
        benchmark_spec.entry_point is not None
        and f"check({benchmark_spec.entry_point})" in str(benchmark_spec.test_harness)
    )
    if benchmark_spec.entry_point and not explicit_check_call:
        program_parts.append(f"check({benchmark_spec.entry_point})")
    return "\n".join(program_parts)


def _extract_problem_text(task_state) -> str:
    messages = getattr(task_state, "messages", None)
    if messages:
        return str(messages[0].content)
    return str(getattr(task_state, "input_text", ""))
