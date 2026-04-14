#!/usr/bin/env python
from __future__ import annotations

from collections import defaultdict
from datetime import datetime
import argparse
import json
from math import ceil
from pathlib import Path
import sys
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parent
V3_ROOT = PROJECT_ROOT / "V3"
if str(V3_ROOT) not in sys.path:
    sys.path.insert(0, str(V3_ROOT))

from utils.results_aggregation import aggregate_persisted_runs


def _sanitize_filename(text: str) -> str:
    safe = [character if character.isalnum() or character in {"-", "_"} else "_" for character in text]
    collapsed = "".join(safe).strip("_")
    return collapsed or "plot"


def _mean(values: list[float]) -> float | None:
    return (sum(values) / len(values)) if values else None


def _ensure_output_dir(output_dir: str | None) -> Path:
    if output_dir:
        path = Path(output_dir)
    else:
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        path = PROJECT_ROOT / "plots" / timestamp
    path.mkdir(parents=True, exist_ok=True)
    return path


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _save_figure(fig: plt.Figure, path: Path) -> str:
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return str(path)


def _group_mean(rows: list[dict], group_fields: tuple[str, ...], value_field: str) -> list[dict]:
    grouped: dict[tuple, list[float]] = defaultdict(list)
    for row in rows:
        value = row.get(value_field)
        if value is None:
            continue
        grouped[tuple(row.get(field) for field in group_fields)].append(float(value))

    result = []
    for key, values in grouped.items():
        item = {field: key[index] for index, field in enumerate(group_fields)}
        item[f"mean_{value_field}"] = _mean(values)
        item["count"] = len(values)
        result.append(item)

    result.sort(key=lambda item: tuple(str(item.get(field) or "") for field in group_fields))
    return result


def _plot_grouped_bar(
    grouped_rows: list[dict],
    *,
    x_field: str,
    hue_field: str,
    value_field: str,
    title: str,
    ylabel: str,
    output_path: Path,
) -> str | None:
    if not grouped_rows:
        return None

    x_labels = sorted({str(row.get(x_field) or "unknown") for row in grouped_rows})
    hues = sorted({str(row.get(hue_field) or "unknown") for row in grouped_rows})
    if not x_labels or not hues:
        return None

    lookups: dict[tuple[str, str], list[float]] = defaultdict(list)
    for row in grouped_rows:
        value = row.get(value_field)
        if value is None:
            continue
        lookups[(str(row.get(x_field) or "unknown"), str(row.get(hue_field) or "unknown"))].append(float(value))

    fig, ax = plt.subplots(figsize=(max(9, len(x_labels) * 1.2), 6))
    bar_width = 0.8 / max(len(hues), 1)
    x_positions = list(range(len(x_labels)))

    for hue_index, hue in enumerate(hues):
        heights = []
        for x_label in x_labels:
            heights.append(_mean(lookups.get((x_label, hue), [])) or 0.0)
        offsets = [position + (hue_index - (len(hues) - 1) / 2) * bar_width for position in x_positions]
        ax.bar(offsets, heights, width=bar_width, label=hue)

    ax.set_xticks(x_positions)
    ax.set_xticklabels(x_labels, rotation=20, ha="right")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(loc="best")
    ax.grid(axis="y", alpha=0.25)
    return _save_figure(fig, output_path)


def _plot_single_bar(
    items: list[dict],
    *,
    label_field: str,
    value_field: str,
    title: str,
    ylabel: str,
    output_path: Path,
    horizontal: bool = False,
) -> str | None:
    if not items:
        return None

    labels = [str(item.get(label_field) or "unknown") for item in items]
    values = [float(item.get(value_field) or 0.0) for item in items]

    fig, ax = plt.subplots(figsize=(max(9, len(labels) * 0.9), 6))
    if horizontal:
        positions = list(range(len(labels)))
        ax.barh(positions, values)
        ax.set_yticks(positions)
        ax.set_yticklabels(labels)
        ax.set_xlabel(ylabel)
    else:
        positions = list(range(len(labels)))
        ax.bar(positions, values)
        ax.set_xticks(positions)
        ax.set_xticklabels(labels, rotation=20, ha="right")
        ax.set_ylabel(ylabel)

    ax.set_title(title)
    ax.grid(axis="y", alpha=0.25)
    return _save_figure(fig, output_path)


def _plot_iterations_distribution(values: list[float], output_path: Path) -> str | None:
    if not values:
        return None

    fig, ax = plt.subplots(figsize=(8, 5))
    bins = min(10, max(3, ceil(len(values) / 2)))
    ax.hist(values, bins=bins, color="#4c78a8", edgecolor="white")
    ax.set_title("Iterations to Success Distribution")
    ax.set_xlabel("Iterations to success")
    ax.set_ylabel("Run count")
    ax.grid(axis="y", alpha=0.25)
    return _save_figure(fig, output_path)


def _plot_time_series(
    evolution_group: dict,
    *,
    series_field: str,
    title: str,
    ylabel: str,
    output_path: Path,
    top_n: int = 5,
) -> str | None:
    runs = evolution_group.get("runs") or []
    if len(runs) < 2:
        return None

    arm_totals: dict[str, float] = defaultdict(float)
    for run in runs:
        pulls_by_arm = run.get("pulls_by_arm") or {}
        for arm_id, count in pulls_by_arm.items():
            arm_totals[str(arm_id)] += float(count)

    if not arm_totals:
        return None

    top_arms = [arm_id for arm_id, _ in sorted(arm_totals.items(), key=lambda item: item[1], reverse=True)[:top_n]]
    x_positions = list(range(1, len(runs) + 1))

    fig, ax = plt.subplots(figsize=(10, 6))
    for arm_id in top_arms:
        y_values = []
        for run in runs:
            if series_field == "pull_share":
                pulls_by_arm = run.get("pulls_by_arm") or {}
                total_pulls = sum(int(value) for value in pulls_by_arm.values())
                pulls = int(pulls_by_arm.get(arm_id, 0))
                y_values.append((pulls / total_pulls) if total_pulls else 0.0)
            else:
                raise ValueError(f"Unsupported series_field: {series_field}")
        ax.plot(x_positions, y_values, marker="o", label=arm_id)

    ax.set_xticks(x_positions)
    ax.set_xticklabels([str(index) for index in x_positions])
    ax.set_xlabel("Run order")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(loc="best")
    ax.grid(alpha=0.25)
    return _save_figure(fig, output_path)


def generate_plots(aggregation: dict, output_dir: str) -> dict:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    run_rows = aggregation.get("runs") or []
    grouped_summary = aggregation.get("grouped_summary") or []
    evolution_by_group = aggregation.get("evolution_by_group") or []

    generated_files: list[str] = []

    attack_by_policy_input = [
        {
            "benchmark": row.get("benchmark"),
            "policy_mode": row.get("policy_mode"),
            "attack_success_rate": row.get("attack_success_rate"),
        }
        for row in run_rows
    ]
    attack_by_policy = _plot_grouped_bar(
        attack_by_policy_input,
        x_field="benchmark",
        hue_field="policy_mode",
        value_field="attack_success_rate",
        title="Attack Success Rate by Policy Mode",
        ylabel="Attack success rate",
        output_path=output_path / "attack_success_rate_by_policy_mode.png",
    )
    if attack_by_policy:
        generated_files.append(attack_by_policy)

    success_by_benchmark = _plot_single_bar(
        _group_mean(run_rows, ("benchmark",), "attack_success_rate"),
        label_field="benchmark",
        value_field="mean_attack_success_rate",
        title="Success by Benchmark",
        ylabel="Average attack success rate",
        output_path=output_path / "success_by_benchmark.png",
    )
    if success_by_benchmark:
        generated_files.append(success_by_benchmark)

    syntax_by_policy = _plot_grouped_bar(
        [
            {
                "benchmark": row.get("benchmark"),
                "policy_mode": row.get("policy_mode"),
                "syntax_invalid_rate": row.get("syntax_invalid_rate"),
            }
            for row in run_rows
        ],
        x_field="benchmark",
        hue_field="policy_mode",
        value_field="syntax_invalid_rate",
        title="Syntax Invalid Rate by Policy Mode",
        ylabel="Syntax invalid rate",
        output_path=output_path / "syntax_invalid_rate_by_policy_mode.png",
    )
    if syntax_by_policy:
        generated_files.append(syntax_by_policy)

    one_shot_vs_iterative = _plot_single_bar(
        _group_mean(run_rows, ("experiment_mode",), "attack_success_rate"),
        label_field="experiment_mode",
        value_field="mean_attack_success_rate",
        title="One-shot vs Iterative Comparison",
        ylabel="Average attack success rate",
        output_path=output_path / "one_shot_vs_iterative_comparison.png",
    )
    if one_shot_vs_iterative:
        generated_files.append(one_shot_vs_iterative)

    split_comparison = _plot_single_bar(
        _group_mean(run_rows, ("experiment_split",), "attack_success_rate"),
        label_field="experiment_split",
        value_field="mean_attack_success_rate",
        title="Train / Validation / Test Comparison",
        ylabel="Average attack success rate",
        output_path=output_path / "train_validation_test_comparison.png",
    )
    if split_comparison:
        generated_files.append(split_comparison)

    iterations_values = [
        float(row.get("average_iterations_to_success"))
        for row in run_rows
        if row.get("average_iterations_to_success") is not None
    ]
    iterations_plot = _plot_iterations_distribution(
        iterations_values,
        output_path / "iterations_to_success_distribution.png",
    )
    if iterations_plot:
        generated_files.append(iterations_plot)

    pull_totals: dict[str, float] = defaultdict(float)
    reward_totals: dict[str, list[float]] = defaultdict(list)
    for row in run_rows:
        for arm_id, pull_count in (row.get("pulls_by_arm") or {}).items():
            pull_totals[str(arm_id)] += float(pull_count)
        for arm_id, average_reward in (row.get("average_reward_by_arm") or {}).items():
            reward_totals[str(arm_id)].append(float(average_reward))

    arm_pull_counts = _plot_single_bar(
        [{"arm": arm_id, "pulls": pulls} for arm_id, pulls in sorted(pull_totals.items(), key=lambda item: item[1], reverse=True)],
        label_field="arm",
        value_field="pulls",
        title="Arm Pull Counts",
        ylabel="Pull count",
        output_path=output_path / "arm_pull_counts.png",
        horizontal=True,
    )
    if arm_pull_counts:
        generated_files.append(arm_pull_counts)

    average_reward_by_arm = _plot_single_bar(
        [{"arm": arm_id, "average_reward": _mean(values) or 0.0} for arm_id, values in sorted(reward_totals.items(), key=lambda item: _mean(item[1]) or 0.0, reverse=True)],
        label_field="arm",
        value_field="average_reward",
        title="Average Reward by Arm",
        ylabel="Average reward",
        output_path=output_path / "average_reward_by_arm.png",
        horizontal=True,
    )
    if average_reward_by_arm:
        generated_files.append(average_reward_by_arm)

    group_artifacts = []
    for group in evolution_by_group:
        benchmark = str(group.get("benchmark") or "unknown")
        policy_mode = str(group.get("policy_mode") or "unknown")
        experiment_mode = str(group.get("experiment_mode") or "unknown")
        experiment_split = str(group.get("experiment_split") or "full")
        bandit_algorithm = group.get("bandit_algorithm") or "none"
        suffix = _sanitize_filename(f"{benchmark}_{policy_mode}_{experiment_mode}_{experiment_split}_{bandit_algorithm}")

        preference_path = output_path / f"arm_preference_over_time_{suffix}.png"
        preference_plot = _plot_time_series(
            group,
            series_field="pull_share",
            title=f"Arm Preference Over Time - {benchmark} / {policy_mode}",
            ylabel="Share of pulls",
            output_path=preference_path,
            top_n=5,
        )
        if preference_plot:
            generated_files.append(preference_plot)
            group_artifacts.append(
                {
                    "benchmark": benchmark,
                    "policy_mode": policy_mode,
                    "experiment_mode": experiment_mode,
                    "experiment_split": experiment_split,
                    "bandit_algorithm": bandit_algorithm,
                    "figure": preference_plot,
                }
            )

        if group.get("bandit_algorithm"):
            bandit_path = output_path / f"rl_bandit_evolution_{suffix}.png"
            runs = group.get("runs") or []
            if len(runs) >= 2:
                fig, ax = plt.subplots(figsize=(10, 6))
                x_positions = list(range(1, len(runs) + 1))
                y_attack = [float(run.get("attack_success_rate") or 0.0) for run in runs]
                y_confidence = [float(run.get("average_llm_confidence") or 0.0) for run in runs]
                y_syntax = [float(run.get("syntax_invalid_rate") or 0.0) for run in runs]
                ax.plot(x_positions, y_attack, marker="o", label="attack_success_rate")
                ax.plot(x_positions, y_confidence, marker="o", label="average_llm_confidence")
                ax.plot(x_positions, y_syntax, marker="o", label="syntax_invalid_rate")
                ax.set_title(f"RL Bandit Evolution - {benchmark} / {policy_mode}")
                ax.set_xlabel("Run order")
                ax.set_ylabel("Rate")
                ax.set_xticks(x_positions)
                ax.set_xticklabels([str(index) for index in x_positions])
                ax.legend(loc="best")
                ax.grid(alpha=0.25)
                saved = _save_figure(fig, bandit_path)
                generated_files.append(saved)
                group_artifacts.append(
                    {
                        "benchmark": benchmark,
                        "policy_mode": policy_mode,
                        "experiment_mode": experiment_mode,
                        "experiment_split": experiment_split,
                        "bandit_algorithm": bandit_algorithm,
                        "figure": saved,
                    }
                )

    manifest = {
        "generated_at": datetime.now().isoformat(),
        "run_count": len(run_rows),
        "group_count": len(grouped_summary),
        "plot_count": len(generated_files),
        "plots": generated_files,
        "group_artifacts": group_artifacts,
        "source_summary": aggregation.get("aggregation_metadata", {}),
    }
    _write_json(output_path / "plot_manifest.json", manifest)
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate offline plots from persisted benchmark runs.",
    )
    parser.add_argument(
        "--results-dir",
        default=str(PROJECT_ROOT / "results"),
        help="Directory containing persisted experiment runs.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory where plot images and manifest will be written (default: plots/<timestamp>).",
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
    output_dir = _ensure_output_dir(args.output_dir)
    aggregation = aggregate_persisted_runs(
        results_dir=args.results_dir,
        benchmarks=args.benchmarks,
        policy_modes=args.policy_modes,
    )
    manifest = generate_plots(aggregation, str(output_dir))

    print("Plot generation completed")
    print(f"  output_dir: {output_dir}")
    print(f"  runs: {manifest['run_count']}")
    print(f"  groups: {manifest['group_count']}")
    print(f"  plots: {manifest['plot_count']}")
    for plot_path in manifest["plots"]:
        print(f"  plot: {plot_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
