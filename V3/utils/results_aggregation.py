from __future__ import annotations

from collections import defaultdict
from datetime import datetime
import csv
import json
from pathlib import Path
from typing import Any

from utils.results_persistence import load_persisted_runs


def _to_bool(value: Any) -> bool | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"true", "1", "yes"}:
        return True
    if text in {"false", "0", "no"}:
        return False
    return None


def _safe_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _parse_timestamp(run_config: dict, run_path: str) -> datetime | None:
    timestamp = run_config.get("timestamp")
    if timestamp:
        try:
            return datetime.fromisoformat(str(timestamp))
        except ValueError:
            pass

    run_id = str(run_config.get("run_id") or Path(run_path).name)
    prefix = "_".join(run_id.split("_")[:2])
    try:
        return datetime.strptime(prefix, "%Y-%m-%d_%H-%M-%S")
    except ValueError:
        return None


def _group_key(row: dict) -> tuple:
    return (
        row.get("benchmark"),
        row.get("policy_mode"),
        row.get("experiment_mode"),
        row.get("experiment_split"),
        row.get("bandit_algorithm"),
        row.get("selector_model"),
        row.get("target_model"),
    )


def _average(values: list[float]) -> float | None:
    return (sum(values) / len(values)) if values else None


def aggregate_persisted_runs(
    *,
    results_dir: str,
    benchmarks: list[str] | None = None,
    policy_modes: list[str] | None = None,
) -> dict:
    loaded_runs = load_persisted_runs(results_dir)
    benchmark_filter = {b.lower() for b in benchmarks} if benchmarks else None
    policy_filter = {p.lower() for p in policy_modes} if policy_modes else None

    run_rows = []
    for run in loaded_runs:
        run_config = run.get("run_config") or {}
        run_summary = run.get("run_summary") or {}
        benchmark = str(run_summary.get("benchmark") or run_config.get("benchmark") or "").lower()
        policy_mode = str(run_summary.get("policy_mode") or run_config.get("policy_mode") or "").lower()

        if benchmark_filter and benchmark not in benchmark_filter:
            continue
        if policy_filter and policy_mode not in policy_filter:
            continue

        timestamp = _parse_timestamp(run_config, run.get("run_path", ""))
        run_rows.append(
            {
                "run_id": run_summary.get("run_id") or run_config.get("run_id"),
                "run_path": run.get("run_path"),
                "timestamp": timestamp.isoformat() if timestamp else None,
                "benchmark": benchmark,
                "policy_mode": policy_mode,
                "experiment_mode": run_summary.get("experiment_mode") or run_config.get("experiment_mode"),
                "experiment_split": run_summary.get("experiment_split") or run_config.get("experiment_split"),
                "split_definition": run_summary.get("split_definition") or run_config.get("split_definition"),
                "target_model": str(run_config.get("target_model") or ""),
                "selector_model": str(run_config.get("selector_model") or ""),
                "code_generation_model": run_config.get("code_generation_model"),
                "max_iterations": run_config.get("max_iterations"),
                "selector_cot_enabled": _to_bool(
                    run_summary.get("selector_cot_enabled")
                    if run_summary.get("selector_cot_enabled") is not None
                    else run_config.get("selector_cot_enabled")
                ),
                "bandit_algorithm": run_summary.get("bandit_algorithm") or run_config.get("bandit_algorithm"),
                "bandit_freeze_weights_effective": _to_bool(
                    run_summary.get("bandit_freeze_weights_effective")
                    if run_summary.get("bandit_freeze_weights_effective") is not None
                    else run_config.get("bandit_freeze_weights_effective")
                ),
                "num_samples": run_summary.get("num_samples"),
                "attack_success_rate": _safe_float(run_summary.get("attack_success_rate")),
                "syntax_invalid_rate": _safe_float(run_summary.get("syntax_invalid_rate")),
                "invalid_attempt_rate": _safe_float(run_summary.get("invalid_attempt_rate")),
                "average_iterations_to_success": _safe_float(run_summary.get("average_iterations_to_success")),
                "average_llm_confidence": _safe_float(run_summary.get("average_llm_confidence")),
                "successful_samples": run_summary.get("successful_samples"),
                "failed_samples": run_summary.get("failed_samples"),
                "baseline_success": run_summary.get("baseline_success"),
                "tactic_driven_success": run_summary.get("tactic_driven_success"),
                "success_by_arm": run_summary.get("success_by_arm") or {},
                "pulls_by_arm": run_summary.get("pulls_by_arm") or {},
                "cumulative_reward_by_arm": run_summary.get("cumulative_reward_by_arm") or {},
                "average_reward_by_arm": run_summary.get("average_reward_by_arm") or {},
                "stop_reason_counts": run_summary.get("stop_reason_counts") or {},
            }
        )

    grouped_acc = defaultdict(
        lambda: {
            "runs": 0,
            "attack_success_rate": [],
            "syntax_invalid_rate": [],
            "invalid_attempt_rate": [],
            "average_iterations_to_success": [],
            "average_llm_confidence": [],
            "successful_samples": 0,
            "failed_samples": 0,
            "baseline_success_count": 0,
            "tactic_driven_success_count": 0,
            "stop_reason_counts": defaultdict(int),
        }
    )

    evolution_acc = defaultdict(list)

    for row in run_rows:
        key = _group_key(row)
        acc = grouped_acc[key]
        acc["runs"] += 1

        for metric_key in (
            "attack_success_rate",
            "syntax_invalid_rate",
            "invalid_attempt_rate",
            "average_iterations_to_success",
            "average_llm_confidence",
        ):
            value = row.get(metric_key)
            if value is not None:
                acc[metric_key].append(value)

        acc["successful_samples"] += int(row.get("successful_samples") or 0)
        acc["failed_samples"] += int(row.get("failed_samples") or 0)
        if row.get("baseline_success"):
            acc["baseline_success_count"] += 1
        if row.get("tactic_driven_success"):
            acc["tactic_driven_success_count"] += 1

        stop_reasons = row.get("stop_reason_counts") or {}
        for reason, count in stop_reasons.items():
            acc["stop_reason_counts"][str(reason)] += int(count)

        evolution_acc[key].append(
            {
                "run_id": row.get("run_id"),
                "timestamp": row.get("timestamp"),
                "attack_success_rate": row.get("attack_success_rate"),
                "syntax_invalid_rate": row.get("syntax_invalid_rate"),
                "invalid_attempt_rate": row.get("invalid_attempt_rate"),
                "average_iterations_to_success": row.get("average_iterations_to_success"),
                "average_llm_confidence": row.get("average_llm_confidence"),
                "experiment_split": row.get("experiment_split"),
                "bandit_algorithm": row.get("bandit_algorithm"),
                "pulls_by_arm": row.get("pulls_by_arm") or {},
                "average_reward_by_arm": row.get("average_reward_by_arm") or {},
                "cumulative_reward_by_arm": row.get("cumulative_reward_by_arm") or {},
            }
        )

    grouped_summary = []
    for key, acc in grouped_acc.items():
        benchmark, policy_mode, experiment_mode, experiment_split, bandit_algorithm, selector_model, target_model = key
        grouped_summary.append(
            {
                "benchmark": benchmark,
                "policy_mode": policy_mode,
                "experiment_mode": experiment_mode,
                "experiment_split": experiment_split,
                "bandit_algorithm": bandit_algorithm,
                "selector_model": selector_model,
                "target_model": target_model,
                "run_count": acc["runs"],
                "mean_attack_success_rate": _average(acc["attack_success_rate"]),
                "mean_syntax_invalid_rate": _average(acc["syntax_invalid_rate"]),
                "mean_invalid_attempt_rate": _average(acc["invalid_attempt_rate"]),
                "mean_iterations_to_success": _average(acc["average_iterations_to_success"]),
                "mean_llm_confidence": _average(acc["average_llm_confidence"]),
                "total_successful_samples": acc["successful_samples"],
                "total_failed_samples": acc["failed_samples"],
                "baseline_success_count": acc["baseline_success_count"],
                "tactic_driven_success_count": acc["tactic_driven_success_count"],
                "stop_reason_counts": dict(acc["stop_reason_counts"]),
            }
        )

    grouped_summary.sort(
        key=lambda row: (
            row.get("benchmark") or "",
            row.get("policy_mode") or "",
            row.get("experiment_mode") or "",
            row.get("experiment_split") or "",
            row.get("bandit_algorithm") or "",
        )
    )

    evolution_by_group = []
    for key, entries in evolution_acc.items():
        benchmark, policy_mode, experiment_mode, experiment_split, bandit_algorithm, selector_model, target_model = key
        entries_sorted = sorted(entries, key=lambda item: item.get("timestamp") or "")
        evolution_by_group.append(
            {
                "benchmark": benchmark,
                "policy_mode": policy_mode,
                "experiment_mode": experiment_mode,
                "experiment_split": experiment_split,
                "bandit_algorithm": bandit_algorithm,
                "selector_model": selector_model,
                "target_model": target_model,
                "runs": entries_sorted,
            }
        )

    evolution_by_group.sort(
        key=lambda row: (
            row.get("benchmark") or "",
            row.get("policy_mode") or "",
            row.get("experiment_mode") or "",
            row.get("experiment_split") or "",
            row.get("bandit_algorithm") or "",
        )
    )

    run_rows.sort(key=lambda row: row.get("timestamp") or "")

    return {
        "aggregation_metadata": {
            "results_dir": str(results_dir),
            "run_count": len(run_rows),
            "benchmarks": sorted({row.get("benchmark") for row in run_rows if row.get("benchmark")}),
            "policy_modes": sorted({row.get("policy_mode") for row in run_rows if row.get("policy_mode")}),
        },
        "runs": run_rows,
        "grouped_summary": grouped_summary,
        "evolution_by_group": evolution_by_group,
    }


def write_aggregation_artifacts(*, aggregation: dict, output_dir: str) -> dict:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    runs_json_path = output_path / "aggregated_runs.json"
    grouped_json_path = output_path / "grouped_summary.json"
    evolution_json_path = output_path / "evolution_by_group.json"
    runs_csv_path = output_path / "aggregated_runs.csv"

    runs_json_path.write_text(
        json.dumps(
            {
                "aggregation_metadata": aggregation.get("aggregation_metadata", {}),
                "runs": aggregation.get("runs", []),
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    grouped_json_path.write_text(
        json.dumps(aggregation.get("grouped_summary", []), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    evolution_json_path.write_text(
        json.dumps(aggregation.get("evolution_by_group", []), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    run_rows = aggregation.get("runs", [])
    csv_fields = [
        "run_id",
        "timestamp",
        "benchmark",
        "policy_mode",
        "experiment_mode",
        "experiment_split",
        "split_definition",
        "target_model",
        "selector_model",
        "code_generation_model",
        "max_iterations",
        "selector_cot_enabled",
        "bandit_algorithm",
        "bandit_freeze_weights_effective",
        "num_samples",
        "attack_success_rate",
        "syntax_invalid_rate",
        "invalid_attempt_rate",
        "average_iterations_to_success",
        "average_llm_confidence",
        "successful_samples",
        "failed_samples",
        "run_path",
    ]
    with runs_csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=csv_fields)
        writer.writeheader()
        for row in run_rows:
            writer.writerow({field: row.get(field) for field in csv_fields})

    # Comparison summary — one row per (benchmark, selector_model, target_model, policy_mode,
    # bandit_algorithm, experiment_split) with mean ASR. Ready for paper tables.
    comparison_csv_path = output_path / "comparison_summary.csv"
    comparison_fields = [
        "benchmark", "selector_model", "target_model", "policy_mode",
        "bandit_algorithm", "experiment_split", "run_count",
        "mean_attack_success_rate", "mean_syntax_invalid_rate",
        "mean_invalid_attempt_rate", "mean_iterations_to_success",
        "mean_llm_confidence",
    ]
    with comparison_csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=comparison_fields)
        writer.writeheader()
        for row in aggregation.get("grouped_summary", []):
            writer.writerow({field: row.get(field) for field in comparison_fields})

    return {
        "runs_json": str(runs_json_path),
        "grouped_summary_json": str(grouped_json_path),
        "evolution_json": str(evolution_json_path),
        "runs_csv": str(runs_csv_path),
        "comparison_summary_csv": str(comparison_csv_path),
    }
