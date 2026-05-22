from __future__ import annotations

from collections import Counter, defaultdict
from datetime import datetime
import json
from pathlib import Path
from uuid import uuid4

from utils.reward_accounting import (
    REWARD_RULE_VERSION,
    compute_attempt_reward,
    normalize_arm_id,
    summarize_arm_accounting,
)


def _sanitize_model_tag(model_name: str | None) -> str:
    text = str(model_name or "unknown-model")
    return (
        text.replace("/", "-")
        .replace(":", "-")
        .replace("\\", "-")
        .replace(" ", "-")
    )


def _artifact_summary(text: str | None, limit: int = 160) -> dict | None:
    if text is None:
        return None
    normalized = " ".join(str(text).split())
    preview = normalized if len(normalized) <= limit else normalized[: limit - 3] + "..."
    return {
        "chars": len(str(text)),
        "preview": preview,
    }


def _normalize_model_value(model_value) -> str | None:
    if model_value is None:
        return None
    return str(model_value)


def _extract_attempt_record(
    *,
    run_id: str,
    sample_id: str,
    benchmark: str,
    policy_mode: str,
    experiment_split: str | None,
    split_definition: str | None,
    selector_cot_enabled: bool | None,
    experiment_mode: str,
    record: dict,
) -> dict:
    selector_output = ((record.get("trace") or {}).get("selector_output") or {})
    tactic_action = record.get("selected_tactic_action") or record.get("applied_tactic_action") or {}
    llm_judge = record.get("llm_judge") or {}
    test_judge = record.get("test_judge") or {}
    trace_summary = ((record.get("trace") or {}).get("summary") or {})
    failure_stage = record.get("failure_stage") or trace_summary.get("failure_stage")
    arm_id = normalize_arm_id(
        tactic_id=tactic_action.get("tactic_id") or selector_output.get("tactic_id"),
        tactic_family=tactic_action.get("tactic_family") or selector_output.get("tactic_family"),
    )
    reward_info = compute_attempt_reward(record, failure_stage)

    return {
        "run_id": run_id,
        "sample_id": sample_id,
        "benchmark": benchmark,
        "policy_mode": policy_mode,
        "experiment_split": experiment_split,
        "split_definition": split_definition,
        "selector_cot_enabled": (
            selector_output.get("selector_cot_enabled")
            if selector_output.get("selector_cot_enabled") is not None
            else selector_cot_enabled
        ),
        "experiment_mode": experiment_mode,
        "iteration": record.get("iteration"),
        "selected_tactic": record.get("selected_tactic") or record.get("tactic"),
        "arm_id": arm_id,
        "tactic_id": tactic_action.get("tactic_id") or selector_output.get("tactic_id"),
        "tactic_family": tactic_action.get("tactic_family") or selector_output.get("tactic_family"),
        "test_judge_decision": test_judge.get("decision"),
        "llm_judge_decision": llm_judge.get("decision"),
        "llm_judge_confidence": llm_judge.get("confidence"),
        "attack_success": record.get("attack_success"),
        "syntax_valid": record.get("syntax_valid"),
        "failure_stage": failure_stage,
        "reward": reward_info["reward"],
        "reward_components": reward_info["reward_components"],
        "reward_rule": reward_info["reward_rule"],
        "bandit_algorithm": selector_output.get("bandit_algorithm"),
        "bandit_state": selector_output.get("bandit_state"),
        "bandit_state_post_update": record.get("bandit_state_post_update"),
        "selector_reasoning": selector_output.get("selector_reasoning"),
        "stop_reason": record.get("stop_reason"),
        "raw_completion_summary": _artifact_summary(record.get("raw_completion")),
        "executable_code_summary": _artifact_summary(record.get("executable_code")),
        "review_artifact_summary": _artifact_summary(record.get("review_artifact")),
    }


def _build_attempt_rows(metadata: dict, run_id: str, sample_id: str) -> list[dict]:
    benchmark = metadata.get("benchmark")
    policy_mode = metadata.get("policy_mode")
    experiment_split = metadata.get("experiment_split")
    split_definition = metadata.get("split_definition")
    selector_cot_enabled = metadata.get("selector_cot_enabled")
    experiment_mode = metadata.get("experiment_mode")
    rows = []

    baseline = metadata.get("baseline")
    if baseline:
        rows.append(
            _extract_attempt_record(
                run_id=run_id,
                sample_id=sample_id,
                benchmark=benchmark,
                policy_mode=policy_mode,
                experiment_split=experiment_split,
                split_definition=split_definition,
                selector_cot_enabled=selector_cot_enabled,
                experiment_mode=experiment_mode,
                record=baseline,
            )
        )

    for attempt in metadata.get("all_attempts", []):
        rows.append(
            _extract_attempt_record(
                run_id=run_id,
                sample_id=sample_id,
                benchmark=benchmark,
                policy_mode=policy_mode,
                experiment_split=experiment_split,
                split_definition=split_definition,
                selector_cot_enabled=selector_cot_enabled,
                experiment_mode=experiment_mode,
                record=attempt,
            )
        )

    return rows


def _compute_summary(metadata: dict, attempt_rows: list[dict], run_id: str) -> dict:
    stop_reason_counts = Counter()
    llm_confidences = []
    arm_accounting = summarize_arm_accounting(attempt_rows)

    for row in attempt_rows:
        stop_reason = row.get("stop_reason")
        if stop_reason:
            stop_reason_counts[stop_reason] += 1

        confidence = row.get("llm_judge_confidence")
        if confidence is not None:
            llm_confidences.append(confidence)

    syntax_invalid_present = any(row.get("syntax_valid") is False for row in attempt_rows)
    invalid_attempts = [
        row
        for row in attempt_rows
        if row.get("syntax_valid") is False or row.get("failure_stage") in {"attack_application", "iteration_exception"}
    ]

    baseline_success = any(
        row.get("iteration") == 0 and row.get("attack_success") is True
        for row in attempt_rows
    )
    tactic_driven_success = any(
        (row.get("iteration") or 0) > 0 and row.get("attack_success") is True
        for row in attempt_rows
    )

    successful_iteration = metadata.get("successful_iteration")

    return {
        "run_id": run_id,
        "benchmark": metadata.get("benchmark"),
        "policy_mode": metadata.get("policy_mode"),
        "experiment_split": metadata.get("experiment_split"),
        "split_definition": metadata.get("split_definition"),
        "bandit_algorithm": metadata.get("bandit_algorithm"),
        "selector_cot_enabled": metadata.get("selector_cot_enabled"),
        "bandit_freeze_weights_effective": metadata.get("bandit_freeze_weights_effective"),
        "experiment_mode": metadata.get("experiment_mode"),
        "num_samples": 1,
        "attack_success_rate": 1.0 if metadata.get("attack_succeeded") else 0.0,
        "successful_samples": 1 if metadata.get("attack_succeeded") else 0,
        "failed_samples": 0 if metadata.get("attack_succeeded") else 1,
        "baseline_success": baseline_success,
        "tactic_driven_success": tactic_driven_success,
        "syntax_invalid_rate": 1.0 if syntax_invalid_present else 0.0,
        "invalid_attempt_rate": (
            len(invalid_attempts) / len(attempt_rows) if attempt_rows else 0.0
        ),
        "average_iterations_to_success": (
            float(successful_iteration) if successful_iteration is not None else None
        ),
        "average_llm_confidence": (
            sum(llm_confidences) / len(llm_confidences) if llm_confidences else None
        ),
        "reward_rule": REWARD_RULE_VERSION,
        "success_by_arm": arm_accounting["success_by_arm"],
        "pulls_by_arm": arm_accounting["pulls_by_arm"],
        "cumulative_reward_by_arm": arm_accounting["cumulative_reward_by_arm"],
        "average_reward_by_arm": arm_accounting["average_reward_by_arm"],
        "stop_reason_counts": dict(stop_reason_counts),
    }


def persist_run_results(
    *,
    results_dir: str,
    metadata: dict,
    task_name: str,
    benchmark: str,
    policy_mode: str,
    experiment_split: str | None,
    split_definition: str | None,
    bandit_algorithm: str | None,
    selector_cot_enabled: bool | None,
    bandit_freeze_weights_effective: bool | None,
    experiment_mode: str,
    target_model: str | None,
    judge_model: str | None,
    selector_model: str | None,
    max_iterations: int,
    sample_id: str | None = None,
    limit: int | None = None,
    max_samples: int | None = None,
    seed: int | None = None,
    git_commit: str | None = None,
) -> dict:
    timestamp = datetime.now().astimezone()
    target_model = _normalize_model_value(target_model)
    judge_model = _normalize_model_value(judge_model)
    selector_model = _normalize_model_value(selector_model)
    model_tag = _sanitize_model_tag(target_model or judge_model or selector_model)
    run_id = (
        f"{timestamp.strftime('%Y-%m-%d_%H-%M-%S')}_"
        f"{benchmark}_{policy_mode}_{model_tag}_{uuid4().hex[:8]}"
    )
    run_root = Path(results_dir)
    run_path = run_root / run_id
    run_path.mkdir(parents=True, exist_ok=False)

    sample_identifier = str(sample_id or metadata.get("sample_id") or run_id)
    run_config = {
        "run_id": run_id,
        "timestamp": timestamp.isoformat(),
        "benchmark": benchmark,
        "policy_mode": policy_mode,
        "experiment_split": experiment_split,
        "split_definition": split_definition,
        "bandit_algorithm": bandit_algorithm,
        "selector_cot_enabled": selector_cot_enabled,
        "bandit_freeze_weights_effective": bandit_freeze_weights_effective,
        "experiment_mode": experiment_mode,
        "target_model": target_model,
        "judge_model": judge_model,
        "selector_model": selector_model,
        "max_iterations": max_iterations,
        "limit": limit,
        "max_samples": max_samples,
        "seed": seed,
        "task_name": task_name,
        "git_commit": git_commit,
    }
    attempt_rows = _build_attempt_rows(metadata, run_id, sample_identifier)
    run_summary = _compute_summary(metadata, attempt_rows, run_id)

    (run_path / "run_config.json").write_text(
        json.dumps(run_config, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    (run_path / "run_summary.json").write_text(
        json.dumps(run_summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    with (run_path / "attempts.jsonl").open("w", encoding="utf-8") as handle:
        for row in attempt_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    return {
        "run_id": run_id,
        "run_path": str(run_path),
        "run_config": run_config,
        "run_summary": run_summary,
    }


def load_persisted_runs(results_dir: str) -> list[dict]:
    results_root = Path(results_dir)
    if not results_root.exists():
        return []

    loaded_runs = []
    for run_dir in sorted(path for path in results_root.iterdir() if path.is_dir()):
        run_config_path = run_dir / "run_config.json"
        run_summary_path = run_dir / "run_summary.json"
        attempts_path = run_dir / "attempts.jsonl"
        if not (run_config_path.exists() and run_summary_path.exists() and attempts_path.exists()):
            continue

        attempts = []
        for line in attempts_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                attempts.append(json.loads(line))

        loaded_runs.append(
            {
                "run_path": str(run_dir),
                "run_config": json.loads(run_config_path.read_text(encoding="utf-8")),
                "run_summary": json.loads(run_summary_path.read_text(encoding="utf-8")),
                "attempts": attempts,
            }
        )

    return loaded_runs
