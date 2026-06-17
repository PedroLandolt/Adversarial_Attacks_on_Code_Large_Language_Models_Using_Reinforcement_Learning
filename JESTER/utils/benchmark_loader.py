from __future__ import annotations

from dataclasses import dataclass
import inspect
import os
from pathlib import Path
from typing import Any

from inspect_evals.mbpp import mbpp
from inspect_evals.mbpp.mbpp import verify as _mbpp_verify

try:
    from inspect_evals.humaneval import humaneval
except ImportError:  # pragma: no cover - depends on installed Inspect package version
    humaneval = None

_MBPP_DATASET_PATH = "google-research-datasets/mbpp"
_MBPP_DATASET_REVISION = "4bb6404fdc6cacfda99d4ac4205087b89d32030c"

# HuggingFace path for the CuBERT wrong binary operator dataset.
_CUBERT_WBO_HF_PATH = "claudios/cubert_ETHPy150Open"
_CUBERT_WBO_HF_CONFIG = "wrong_binary_operator_datasets"
_CUBERT_WBO_LABEL = "Wrong binary operator"
_CUBERT_WBO_LIMIT = 10_000

# ---------------------------------------------------------------------------
# Valid benchmark names
# ---------------------------------------------------------------------------
_VALID_BENCHMARKS = {
    # Standard: LLM generates buggy code at runtime (Steps 1–4 full pipeline)
    "mbpp",
    "humaneval",
    # Pedro's dataset: pre-generated LLM bugs, judge-filtered (enter at Step 3)
    # Published at: https://huggingface.co/datasets/PedroLandolt/adversarial-code-buggy
    "adversarial_code_buggy",        # combined MBPP + HumanEval (primary external name)
    "mbpp_pregenerated",             # MBPP only  (internal alias for validation)
    "humaneval_pregenerated",        # HumanEval only (internal alias for validation)
    # CuBERT wrong binary operator subset — external dataset for generalization experiments
    "cubert_wbo",
    # DEPRECATED: synthesized_wbo (AST operator mutation on MBPP+HumanEval).
    # Redundant given cubert_wbo covers operator substitution from real-world code.
    "synthesized_wbo",
}

# Alternate spellings / aliases resolved before validation
_BENCHMARK_ALIASES: dict[str, str] = {
    "adversarial-code-buggy": "adversarial_code_buggy",
    "adversarial_code_buggy_mbpp": "mbpp_pregenerated",
    "adversarial_code_buggy_humaneval": "humaneval_pregenerated",
}

_DEPRECATED_BENCHMARKS = {"synthesized_wbo"}


def normalize_benchmark_name(benchmark_name: str) -> str:
    normalized = str(benchmark_name).strip().lower().replace("-", "_")
    normalized = _BENCHMARK_ALIASES.get(normalized, normalized)
    if normalized not in _VALID_BENCHMARKS:
        raise ValueError(
            f"Unsupported benchmark: {benchmark_name!r}. "
            f"Valid options: {sorted(_VALID_BENCHMARKS - _DEPRECATED_BENCHMARKS)}"
        )
    if normalized in _DEPRECATED_BENCHMARKS:
        import warnings
        warnings.warn(
            f"benchmark='{normalized}' is deprecated. "
            "Use 'adversarial_code_buggy' (LLM-generated) or 'cubert_wbo' (external) instead.",
            DeprecationWarning,
            stacklevel=2,
        )
    return normalized


# ---------------------------------------------------------------------------
# Helpers for resolving local JSONL dataset paths
# ---------------------------------------------------------------------------

def _project_root() -> Path:
    """Return the project root (parent of JESTER/)."""
    return Path(__file__).resolve().parent.parent.parent


def _resolve_dataset_path(env_var: str, default_relative: str) -> Path:
    override = os.environ.get(env_var)
    if override:
        return Path(override)
    return _project_root() / default_relative


# ---------------------------------------------------------------------------
# MBPP sample converter
# ---------------------------------------------------------------------------

def _mbpp_full_record_to_sample(record: dict[str, Any]):
    from inspect_ai.dataset import Sample
    problem_text = record["text"]
    test_list = record["test_list"]
    return Sample(
        input=problem_text,
        target=test_list,
        id=record["task_id"],
        metadata={
            "prompt": problem_text,
            "test_list": test_list,
            "test_list_str": "\n".join(test_list),
            "source_file": "",
            "code": record["code"],
            "test_imports": record.get("test_setup_code", ""),
            "task_id": record["task_id"],
        },
    )


# ---------------------------------------------------------------------------
# Pre-generated JSONL sample converters
# ---------------------------------------------------------------------------

def _pregenerated_mbpp_record_to_sample(record: dict[str, Any]):
    """Load one record from a pre-generated MBPP buggy-code JSONL file."""
    from inspect_ai.dataset import Sample
    problem_text = record["problem_text"]
    test_list = record.get("test_list", [])
    task_id = record.get("task_id")
    return Sample(
        input=problem_text,
        target=test_list,
        id=f"pregen_mbpp_{task_id}",
        metadata={
            "prompt": problem_text,
            "test_list": test_list,
            "test_list_str": "\n".join(test_list),
            "source_file": "",
            "code": record.get("original_code", ""),
            "pregenerated_buggy_code": record["buggy_code"],
            "test_imports": "",
            "task_id": task_id,
            "source_benchmark": "mbpp",
        },
    )


def _pregenerated_humaneval_record_to_sample(record: dict[str, Any]):
    """Load one record from a pre-generated HumanEval buggy-code JSONL file."""
    from inspect_ai.dataset import Sample
    problem_text = record["problem_text"]
    task_id = record.get("task_id")
    return Sample(
        input=problem_text,
        target=record.get("test_harness", ""),
        id=f"pregen_he_{task_id}",
        metadata={
            "prompt": problem_text,
            "test": record.get("test_harness", ""),
            "entry_point": record.get("entry_point", ""),
            "source_file": "",
            "code": record.get("original_code", ""),
            "pregenerated_buggy_code": record["buggy_code"],
            "test_imports": "",
            "task_id": task_id,
            "source_benchmark": "humaneval",
        },
    )


def _load_pregenerated_jsonl(path: Path, record_to_sample) -> list:
    """Load all records from a pre-generated JSONL file."""
    import json
    if not path.exists():
        raise FileNotFoundError(
            f"Pre-generated dataset not found at {path}.\n"
            f"Build it first — see Task 1.5 in TODO.txt."
        )
    samples = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                samples.append(record_to_sample(json.loads(line)))
    return samples


# ---------------------------------------------------------------------------
# Synthesized WBO (Pedro's MBPP + HumanEval operator mutation dataset)
# ---------------------------------------------------------------------------

def _synthesized_wbo_record_to_sample(record: dict[str, Any]):
    from inspect_ai.dataset import Sample
    problem_text = record["problem_text"]
    test_list = record.get("test_list", [])
    test_harness = record.get("test_harness")
    entry_point = record.get("entry_point")
    task_id = record.get("task_id")
    source = record.get("source_benchmark", "mbpp")
    return Sample(
        input=problem_text,
        target=test_list if source == "mbpp" else (test_harness or ""),
        id=f"swbo_{task_id}",
        metadata={
            "prompt": problem_text,
            "test_list": test_list,
            "test_list_str": "\n".join(test_list),
            "test": test_harness,
            "entry_point": entry_point,
            "source_file": "",
            "code": record.get("original_code", ""),
            "pregenerated_buggy_code": record["buggy_code"],
            "operator_substitution": record.get("operator_substitution"),
            "test_imports": "",
            "task_id": task_id,
            "source_benchmark": source,
        },
    )


def _load_synthesized_wbo(path: Path) -> list:
    import json
    if not path.exists():
        raise FileNotFoundError(
            f"Synthesized WBO dataset not found at {path}.\n"
            f"Build it: python JESTER/utils/operator_mutation.py"
        )
    samples = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                samples.append(_synthesized_wbo_record_to_sample(json.loads(line)))
    return samples


# ---------------------------------------------------------------------------
# CuBERT WBO (HuggingFace subset, no unit tests)
# ---------------------------------------------------------------------------

def _cubert_record_to_sample(record: dict[str, Any], idx: int):
    from inspect_ai.dataset import Sample
    function = record.get("function", "")
    info = record.get("info", "")
    # Use info as problem context if it differs meaningfully from function.
    # If info is empty or identical to function, use a minimal prompt instead.
    if info and info.strip() and info.strip() != function.strip():
        problem_text = info.strip()
    else:
        problem_text = function
    return Sample(
        input=problem_text,
        target=[],
        id=f"cubert_{idx}",
        metadata={
            "prompt": problem_text,
            "test_list": [],
            "test_list_str": "",
            "source_file": "",
            "code": function,
            "pregenerated_buggy_code": function,
            "cubert_info": info,
            "label": record.get("label", _CUBERT_WBO_LABEL),
            "test_imports": "",
            "task_id": f"cubert_{idx}",
            "source_benchmark": "cubert_wbo",
        },
    )


def _load_cubert_wbo(limit: int = _CUBERT_WBO_LIMIT) -> list:
    """Load the CuBERT wrong_binary_operator subset from local JSONL.

    The local file must be built first with:
        python JESTER/utils/download_cubert_wbo.py
    This avoids HuggingFace network calls at Inspect task-init time (which
    conflict with the httpx client inside the anyio async runtime).
    """
    import json

    path = _resolve_dataset_path("CUBERT_WBO_PATH", "datasets/cubert_wbo.jsonl")
    if not path.exists():
        raise FileNotFoundError(
            f"CuBERT WBO dataset not found at {path}.\n"
            "Build it first with:\n"
            "    python JESTER/utils/download_cubert_wbo.py\n"
            f"(downloads from {_CUBERT_WBO_HF_PATH} and saves ~{_CUBERT_WBO_LIMIT} records)"
        )

    records: list[dict] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    if limit:
        records = records[:limit]

    return [_cubert_record_to_sample(r, i) for i, r in enumerate(records)]


# ---------------------------------------------------------------------------
# BenchmarkSpec
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class BenchmarkSpec:
    benchmark_name: str
    problem_text: str
    entry_point: str | None
    test_list: list[str] | None
    test_harness: str | None
    normalization_requirements: dict[str, Any]


# ---------------------------------------------------------------------------
# Main loader
# ---------------------------------------------------------------------------

def load_benchmark_task(benchmark_name: str, *, temperature: float):
    from inspect_ai import Task
    from inspect_ai.dataset import MemoryDataset

    normalized = normalize_benchmark_name(benchmark_name)

    if normalized == "adversarial_code_buggy":
        samples = []
        mbpp_path = _resolve_dataset_path("MBPP_PREGENERATED_PATH", "datasets/mbpp_pregenerated.jsonl")
        he_path = _resolve_dataset_path("HUMANEVAL_PREGENERATED_PATH", "datasets/humaneval_pregenerated.jsonl")
        if mbpp_path.exists():
            samples.extend(_load_pregenerated_jsonl(mbpp_path, _pregenerated_mbpp_record_to_sample))
        if he_path.exists():
            samples.extend(_load_pregenerated_jsonl(he_path, _pregenerated_humaneval_record_to_sample))
        if not samples:
            raise FileNotFoundError(
                "No adversarial_code_buggy records found.\n"
                "Build them first with: python JESTER/utils/pregenerate_buggy_code.py"
            )
        return Task(dataset=MemoryDataset(samples), scorer=[_mbpp_verify()])

    if normalized == "synthesized_wbo":
        from utils.operator_mutation import load_synthesized_wbo as _load_swbo
        path = _resolve_dataset_path("SYNTHESIZED_WBO_PATH", "datasets/synthesized_wbo.jsonl")
        records = _load_swbo(path)
        samples = [_synthesized_wbo_record_to_sample(r) for r in records]
        return Task(dataset=MemoryDataset(samples), scorer=[_mbpp_verify()])

    if normalized == "cubert_wbo":
        samples = _load_cubert_wbo()
        return Task(dataset=MemoryDataset(samples), scorer=[_mbpp_verify()])

    if normalized == "mbpp_pregenerated":
        path = _resolve_dataset_path("MBPP_PREGENERATED_PATH", "datasets/mbpp_pregenerated.jsonl")
        samples = _load_pregenerated_jsonl(path, _pregenerated_mbpp_record_to_sample)
        return Task(dataset=MemoryDataset(samples), scorer=[_mbpp_verify()])

    if normalized == "humaneval_pregenerated":
        if humaneval is None:
            raise ImportError("HumanEval benchmark is not available in this Inspect install.")
        base = humaneval() if callable(humaneval) else None
        path = _resolve_dataset_path("HUMANEVAL_PREGENERATED_PATH", "datasets/humaneval_pregenerated.jsonl")
        samples = _load_pregenerated_jsonl(path, _pregenerated_humaneval_record_to_sample)
        scorer = base.scorer if base is not None else []
        return Task(dataset=MemoryDataset(samples), scorer=scorer)

    if normalized == "mbpp":
        from inspect_ai.dataset import MemoryDataset
        from inspect_evals.utils.huggingface import hf_dataset

        base = mbpp(temperature=temperature)
        all_samples = []
        for split in ("train", "validation", "test", "prompt"):
            all_samples.extend(hf_dataset(
                path=_MBPP_DATASET_PATH,
                name="full",
                sample_fields=_mbpp_full_record_to_sample,
                split=split,
                revision=_MBPP_DATASET_REVISION,
            ))
        return Task(dataset=MemoryDataset(all_samples), scorer=base.scorer)

    # humaneval
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


# ---------------------------------------------------------------------------
# Spec extraction
# ---------------------------------------------------------------------------

def extract_benchmark_spec(task_state, benchmark_name: str) -> BenchmarkSpec:
    normalized = normalize_benchmark_name(benchmark_name)
    metadata = getattr(task_state, "metadata", {}) or {}
    problem_text = _extract_problem_text(task_state)

    # CuBERT WBO: no tests — pre-known buggy, judge-only attack
    if normalized == "cubert_wbo":
        return BenchmarkSpec(
            benchmark_name="cubert_wbo",
            problem_text=problem_text,
            entry_point=None,
            test_list=[],
            test_harness=None,
            normalization_requirements={
                "extract_python_code": True,
                "requires_entry_point": False,
                "test_format": "none",
            },
        )

    # synthesized_wbo with HumanEval-sourced samples uses the harness format
    test_harness = metadata.get("test")
    if test_harness is None:
        target = getattr(task_state, "target", None)
        if isinstance(target, str):
            test_harness = target

    if normalized == "synthesized_wbo" and metadata.get("source_benchmark") == "humaneval":
        entry_point = metadata.get("entry_point")
        return BenchmarkSpec(
            benchmark_name="synthesized_wbo",
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

    # adversarial_code_buggy: mixed MBPP + HumanEval — dispatch by source_benchmark in metadata.
    # Each sample carries "source_benchmark": "mbpp" or "humaneval" set by the JSONL loader.
    if normalized == "adversarial_code_buggy":
        source_bm = metadata.get("source_benchmark", "mbpp")
        if source_bm == "humaneval":
            entry_point = metadata.get("entry_point")
            return BenchmarkSpec(
                benchmark_name="adversarial_code_buggy",
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
        # MBPP source
        test_list = metadata.get("test_list")
        if not isinstance(test_list, list) or not test_list:
            target = getattr(task_state, "target", None)
            if isinstance(target, list) and target:
                test_list = target
        return BenchmarkSpec(
            benchmark_name="adversarial_code_buggy",
            problem_text=problem_text,
            entry_point=None,
            test_list=[str(t) for t in (test_list or [])],
            test_harness=None,
            normalization_requirements={
                "extract_python_code": True,
                "requires_entry_point": False,
                "test_format": "assert_list",
            },
        )

    # MBPP-format benchmarks: assert-list tests (mbpp, synthesized_wbo MBPP samples, mbpp_pregenerated)
    if normalized in {"mbpp", "synthesized_wbo", "mbpp_pregenerated"}:
        test_list = metadata.get("test_list")
        if not isinstance(test_list, list) or not test_list:
            target = getattr(task_state, "target", None)
            if isinstance(target, list) and target:
                test_list = target
        normalized_tests = [str(t) for t in (test_list or [])]
        return BenchmarkSpec(
            benchmark_name=normalized,
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

    # HumanEval-format benchmarks (humaneval, humaneval_pregenerated)
    entry_point = metadata.get("entry_point")
    return BenchmarkSpec(
        benchmark_name=normalized,
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


# ---------------------------------------------------------------------------
# Verification program builder
# ---------------------------------------------------------------------------

def build_verification_program(executable_code: str | None, benchmark_spec: BenchmarkSpec) -> str | None:
    if executable_code is None:
        return None

    # Assert-list format (MBPP-style, including adversarial_code_buggy MBPP records)
    if benchmark_spec.normalization_requirements.get("test_format") == "assert_list":
        if not benchmark_spec.test_list:
            return None
        parts = [str(executable_code)]
        for test_case in benchmark_spec.test_list:
            assert_expr = str(test_case)
            message = assert_expr[len("assert "):] if assert_expr.startswith("assert ") else assert_expr
            parts.append(f"{assert_expr}, {repr(message)}")
        return "\n".join(parts)

    # No tests (cubert_wbo) — pipeline uses pregenerated_code FAIL path
    if benchmark_spec.benchmark_name == "cubert_wbo":
        return None

    # Python harness format (HumanEval-style)
    if not benchmark_spec.test_harness:
        return None

    parts = [str(executable_code), str(benchmark_spec.test_harness)]
    explicit_check = (
        benchmark_spec.entry_point is not None
        and f"check({benchmark_spec.entry_point})" in str(benchmark_spec.test_harness)
    )
    if benchmark_spec.entry_point and not explicit_check:
        parts.append(f"check({benchmark_spec.entry_point})")
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def _extract_problem_text(task_state) -> str:
    messages = getattr(task_state, "messages", None)
    if messages:
        for msg in messages:
            if getattr(msg, "role", None) != "system":
                return str(msg.content)
        return str(messages[0].content)
    return str(getattr(task_state, "input_text", ""))


def get_benchmark_size(benchmark_name: str) -> int:
    """Return the total number of samples available for the benchmark."""
    task = load_benchmark_task(benchmark_name, temperature=0.0)
    return len(task.dataset)
