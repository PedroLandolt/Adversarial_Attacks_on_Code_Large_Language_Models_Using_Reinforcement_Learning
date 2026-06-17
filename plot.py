#!/usr/bin/env python
from __future__ import annotations

from collections import defaultdict
from datetime import datetime
import argparse
import json
from math import ceil, sqrt
from pathlib import Path
import sys
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent
V3_ROOT = PROJECT_ROOT / "V3"
if str(V3_ROOT) not in sys.path:
    sys.path.insert(0, str(V3_ROOT))

from utils.results_aggregation import aggregate_persisted_runs

# ---------------------------------------------------------------------------
# Global style — paper quality
# ---------------------------------------------------------------------------

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 10,
    "axes.titlesize": 12,
    "axes.titleweight": "bold",
    "axes.labelsize": 11,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "legend.framealpha": 0.85,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "grid.linestyle": "--",
    "figure.autolayout": False,
})

# ---------------------------------------------------------------------------
# Tactic display names and colors
# ---------------------------------------------------------------------------

TACTIC_DISPLAY_NAMES: dict[str, str] = {
    "legacy_injection": "injection",
    "legacy_output": "output",
    "legacy_semantic": "semantic",
    "legacy_cot": "cot",
    "taxonomy_roleplay": "roleplay",
    "taxonomy_appeal_to_authority": "appeal_to_authority",
    "taxonomy_formatting_smuggling": "formatting_smuggling",
    "taxonomy_recursion_crescendo": "recursion_crescendo",
    "taxonomy_crowding": "crowding",
}

TACTIC_FAMILY_COLORS: dict[str, str] = {
    "injection":            "#E53935",
    "formatting_smuggling": "#E53935",
    "output":               "#FB8C00",
    "crowding":             "#FB8C00",
    "semantic":             "#43A047",
    "roleplay":             "#43A047",
    "cot":                  "#1E88E5",
    "recursion_crescendo":  "#1E88E5",
    "appeal_to_authority":  "#8E24AA",
}

TACTIC_FAMILY_LABEL: dict[str, str] = {
    "injection":            "Structural Logic",
    "formatting_smuggling": "Structural Logic",
    "output":               "Obfuscation & Noise",
    "crowding":             "Obfuscation & Noise",
    "semantic":             "Narrative & Context",
    "roleplay":             "Narrative & Context",
    "cot":                  "Strategy & Pacing",
    "recursion_crescendo":  "Strategy & Pacing",
    "appeal_to_authority":  "Pressure & Persuasion",
}

# ---------------------------------------------------------------------------
# Policy / algorithm display
# ---------------------------------------------------------------------------

ALGO_DISPLAY: dict[str, str] = {
    "ucb1":     "UCB1",
    "thompson": "Thompson",
    "klucb":    "KL-UCB",
    "exp3":     "EXP3",
}

ALGO_COLORS: dict[str, str] = {
    "ucb1":     "#1565C0",
    "thompson": "#2E7D32",
    "klucb":    "#E65100",
    "exp3":     "#6A1B9A",
}

# Ordered list for consistent rendering
ALGO_ORDER = ["ucb1", "thompson", "klucb", "exp3"]

POLICY_BASE_COLORS: dict[str, str] = {
    "random_choice":       "#FF9800",
    "agent_based_decision":"#4CAF50",
}


def _clean_tactic_name(raw_id: str) -> str:
    return TACTIC_DISPLAY_NAMES.get(str(raw_id), str(raw_id))


def _short_model(model: str | None) -> str:
    """'ollama/qwen2.5-coder:7b' -> 'qwen2.5-coder'"""
    s = str(model or "unknown")
    s = s.split("/")[-1]   # strip provider prefix
    s = s.rsplit(":", 1)[0] if ":" in s else s  # strip :Xb tag
    return s or "unknown"


def _policy_label(row: dict) -> str:
    """Build a compact display label from policy_mode + bandit_algorithm."""
    pm = row.get("policy_mode") or ""
    if pm == "rl_bandit":
        alg = str(row.get("bandit_algorithm") or "ucb1").lower()
        return f"RL ({ALGO_DISPLAY.get(alg, alg)})"
    return {"random_choice": "Random", "agent_based_decision": "ReAct"}.get(pm, pm)


def _policy_sort_key(label: str) -> int:
    order = ["Random", "ReAct", "RL (UCB1)", "RL (Thompson)", "RL (KL-UCB)", "RL (EXP3)"]
    try:
        return order.index(label)
    except ValueError:
        return 99


def _policy_color(label: str) -> str:
    mapping = {
        "Random":        "#FF9800",
        "ReAct":         "#4CAF50",
        "RL (UCB1)":     "#1565C0",
        "RL (Thompson)": "#2E7D32",
        "RL (KL-UCB)":   "#E65100",
        "RL (EXP3)":     "#6A1B9A",
    }
    return mapping.get(label, "#9E9E9E")


def _sanitize_filename(text: str) -> str:
    safe = [c if c.isalnum() or c in {"-", "_"} else "_" for c in text]
    return "".join(safe).strip("_") or "plot"


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
    fig.savefig(path, dpi=300, bbox_inches="tight")
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
        item = {field: key[i] for i, field in enumerate(group_fields)}
        item[f"mean_{value_field}"] = _mean(values)
        item["count"] = len(values)
        result.append(item)
    result.sort(key=lambda item: tuple(str(item.get(f) or "") for f in group_fields))
    return result


# ---------------------------------------------------------------------------
# Low-level plot helpers
# ---------------------------------------------------------------------------

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
        v = row.get(value_field)
        if v is None:
            continue
        lookups[(str(row.get(x_field) or "unknown"), str(row.get(hue_field) or "unknown"))].append(float(v))
    fig, ax = plt.subplots(figsize=(max(9, len(x_labels) * 1.2), 6), constrained_layout=True)
    bar_width = 0.8 / max(len(hues), 1)
    x_pos = list(range(len(x_labels)))
    for hi, hue in enumerate(hues):
        heights = [_mean(lookups.get((xl, hue), [])) or 0.0 for xl in x_labels]
        offsets = [p + (hi - (len(hues) - 1) / 2) * bar_width for p in x_pos]
        ax.bar(offsets, heights, width=bar_width, label=hue)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(x_labels, rotation=20, ha="right")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(loc="best")
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
    item_colors: list[str] | None = None,
) -> str | None:
    if not items:
        return None
    labels = [str(item.get(label_field) or "unknown") for item in items]
    values = [float(item.get(value_field) or 0.0) for item in items]
    colors = item_colors if item_colors and len(item_colors) == len(labels) else None
    fig, ax = plt.subplots(figsize=(max(9, len(labels) * 0.9), 6), constrained_layout=True)
    if horizontal:
        ax.barh(range(len(labels)), values, color=colors or "#2196F3", alpha=0.85)
        ax.set_yticks(range(len(labels)))
        ax.set_yticklabels(labels)
        ax.set_xlabel(ylabel)
    else:
        ax.bar(range(len(labels)), values, color=colors or "#2196F3", alpha=0.85)
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=20, ha="right")
        ax.set_ylabel(ylabel)
    ax.set_title(title)
    return _save_figure(fig, output_path)


def _plot_iterations_distribution(values: list[float], output_path: Path) -> str | None:
    if not values:
        return None
    fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)
    bins = min(10, max(3, ceil(len(values) / 2)))
    ax.hist(values, bins=bins, color="#4c78a8", edgecolor="white")
    ax.set_title("Iterations to Success Distribution")
    ax.set_xlabel("Iterations to success")
    ax.set_ylabel("Run count")
    return _save_figure(fig, output_path)


# ---------------------------------------------------------------------------
# Tactic / arm plots
# ---------------------------------------------------------------------------

def _plot_tactic_win_rate(run_rows: list[dict], output_path: Path) -> str | None:
    total_success: dict[str, int] = defaultdict(int)
    total_pulls: dict[str, int] = defaultdict(int)
    for row in run_rows:
        if row.get("policy_mode") != "rl_bandit":
            continue
        for arm_id, count in (row.get("success_by_arm") or {}).items():
            total_success[str(arm_id)] += int(count)
        for arm_id, count in (row.get("pulls_by_arm") or {}).items():
            total_pulls[str(arm_id)] += int(count)
    entries = [
        {"arm": _clean_tactic_name(arm_id), "win_rate": total_success[arm_id] / pulls, "pulls": pulls}
        for arm_id, pulls in total_pulls.items() if pulls > 0
    ]
    if not entries:
        return None
    entries.sort(key=lambda e: e["win_rate"], reverse=True)
    labels = [e["arm"] for e in entries]
    values = [e["win_rate"] * 100 for e in entries]
    annotations = [f"n={e['pulls']}" for e in entries]
    colors = [TACTIC_FAMILY_COLORS.get(lbl, "#2196F3") for lbl in labels]
    fig, ax = plt.subplots(figsize=(9, max(5, len(labels) * 0.7)), constrained_layout=True)
    bars = ax.barh(range(len(labels)), values, color=colors, alpha=0.85)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=10)
    ax.set_xlabel("Attack Success Rate (%)", fontsize=11)
    ax.set_title("Tactic Win Rate — RL Bandit (all training)", fontsize=12)
    max_val = max(values) if values else 1.0
    ax.set_xlim(0, max_val * 1.3)
    for bar, ann in zip(bars, annotations):
        ax.text(bar.get_width() + max_val * 0.02, bar.get_y() + bar.get_height() / 2,
                ann, ha="left", va="center", fontsize=8)
    seen: dict[str, str] = {}
    for lbl, col in zip(labels, colors):
        family = TACTIC_FAMILY_LABEL.get(lbl, "")
        if family and family not in seen:
            seen[family] = col
    handles = [plt.Rectangle((0, 0), 1, 1, color=c, alpha=0.85) for c in seen.values()]
    ax.legend(handles, list(seen.keys()), loc="lower right", fontsize=8, title="Family")
    return _save_figure(fig, output_path)


# ---------------------------------------------------------------------------
# Success breakdown
# ---------------------------------------------------------------------------

def _plot_success_breakdown(run_rows: list[dict], output_path: Path) -> str | None:
    policy_order = ["random_choice", "agent_based_decision", "rl_bandit"]
    policy_labels = {"random_choice": "Random", "agent_based_decision": "ReAct", "rl_bandit": "RL Bandit"}
    buckets: dict[str, dict[str, list]] = defaultdict(lambda: {"baseline": [], "tactic": []})
    for row in run_rows:
        pm = row.get("policy_mode")
        if not pm:
            continue
        buckets[pm]["baseline"].append(float(bool(row.get("baseline_success"))))
        buckets[pm]["tactic"].append(float(bool(row.get("tactic_driven_success"))))
    present = [p for p in policy_order if p in buckets]
    if not present:
        return None
    labels = [policy_labels.get(p, p) for p in present]
    baseline_rates = [_mean(buckets[p]["baseline"]) or 0.0 for p in present]
    tactic_rates   = [_mean(buckets[p]["tactic"]) or 0.0 for p in present]
    x = list(range(len(present)))
    fig, ax = plt.subplots(figsize=(max(6, len(present) * 2.5), 6), constrained_layout=True)
    ax.bar(x, baseline_rates, label="Baseline (no tactic)", color="#FF7043", alpha=0.85)
    ax.bar(x, tactic_rates, label="Tactic-driven", color="#2196F3", alpha=0.85, bottom=baseline_rates)
    for xi, (b, t) in enumerate(zip(baseline_rates, tactic_rates)):
        total = b + t
        if total > 0.005:
            ax.text(xi, total + 0.015, f"{total:.2f}", ha="center", va="bottom", fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_ylabel("Rate")
    ax.set_title("Attack Success: Baseline vs. Tactic-Driven")
    ax.set_ylim(0, 1.15)
    ax.legend(loc="upper right")
    return _save_figure(fig, output_path)


# ---------------------------------------------------------------------------
# Training / evolution plots
# ---------------------------------------------------------------------------

def _compute_chunk_std(values: list[float], chunk_size: int) -> float:
    chunks = [values[i:i + chunk_size] for i in range(0, len(values), chunk_size)]
    chunk_means = [sum(c) / len(c) for c in chunks if len(c) >= max(1, chunk_size // 2)]
    if len(chunk_means) < 2:
        return 0.0
    mean = sum(chunk_means) / len(chunk_means)
    return sqrt(sum((x - mean) ** 2 for x in chunk_means) / (len(chunk_means) - 1))


def _plot_training_learning_curve(run_rows: list[dict], output_path: Path) -> str | None:
    train_rows = sorted(
        [r for r in run_rows if r.get("policy_mode") == "rl_bandit" and r.get("experiment_split") == "train"],
        key=lambda r: r.get("timestamp") or "",
    )
    if len(train_rows) < 10:
        return None
    values = [float(r.get("asr_1shot_win") if r.get("asr_1shot_win") is not None
                    else r.get("attack_success_rate") or 0.0) for r in train_rows]
    # Bin into ~50 evenly-spaced windows; show mean ± std per bin
    n_bins = min(50, len(values) // 5)
    bin_size = len(values) // n_bins
    bin_x, bin_mean, bin_lo, bin_hi = [], [], [], []
    for i in range(n_bins):
        chunk = values[i * bin_size: (i + 1) * bin_size]
        if not chunk:
            continue
        m = sum(chunk) / len(chunk)
        s = sqrt(sum((v - m) ** 2 for v in chunk) / max(len(chunk) - 1, 1))
        bin_x.append(i * bin_size + bin_size / 2)
        bin_mean.append(m)
        bin_lo.append(max(0.0, m - s))
        bin_hi.append(min(1.0, m + s))
    overall_mean = sum(values) / len(values)
    fig, ax = plt.subplots(figsize=(11, 5), constrained_layout=True)
    ax.fill_between(bin_x, bin_lo, bin_hi, alpha=0.25, color="#2196F3")
    ax.plot(bin_x, bin_mean, color="#2196F3", linewidth=2, label=f"Mean ± 1 SD (bin={bin_size})")
    ax.axhline(y=overall_mean, color="#555", linestyle="--", linewidth=1,
               label=f"Overall mean = {overall_mean:.3f}")
    ax.set_xlabel("Training sample")
    ax.set_ylabel("1-Shot ASR")
    ax.set_title("RL Bandit Learning Curve — Training (1-Shot ASR)")
    ax.set_ylim(0, 1.05)
    ax.legend()
    return _save_figure(fig, output_path)



def _plot_exploit_explore(evolution_group: dict, output_path: Path) -> str | None:
    from math import log as _log
    from matplotlib.ticker import MaxNLocator
    if not evolution_group.get("bandit_algorithm"):
        return None
    runs = evolution_group.get("runs") or []
    if len(runs) < 5:
        return None
    benchmark = str(evolution_group.get("benchmark") or "unknown")
    experiment_split = str(evolution_group.get("experiment_split") or "train")
    alg = str(evolution_group.get("bandit_algorithm") or "bandit")
    all_arm_ids: set[str] = set()
    for run in runs:
        for arm_id in (run.get("pulls_by_arm") or {}):
            all_arm_ids.add(str(arm_id))
    if not all_arm_ids:
        return None
    arm_ids_sorted = sorted(all_arm_ids)
    num_arms = len(arm_ids_sorted)
    h_uniform = _log(num_arms) if num_arms > 1 else 0.0
    x_pos = list(range(1, len(runs) + 1))
    cum_arm: dict[str, float] = defaultdict(float)
    arm_share_series: dict[str, list[float]] = {a: [] for a in arm_ids_sorted}
    entropy_series: list[float] = []
    for run in runs:
        for arm_id, count in (run.get("pulls_by_arm") or {}).items():
            cum_arm[str(arm_id)] += float(count)
        total = sum(cum_arm.values()) or 1.0
        probs = []
        for arm_id in arm_ids_sorted:
            share = cum_arm[arm_id] / total
            arm_share_series[arm_id].append(share)
            if share > 0:
                probs.append(share)
        entropy = -sum(p * _log(p) for p in probs) if probs else 0.0
        entropy_series.append(entropy)
    fig, (ax_area, ax_ent) = plt.subplots(
        2, 1, figsize=(12, 8), gridspec_kw={"height_ratios": [2, 1]}, sharex=True, constrained_layout=True)
    bottom = [0.0] * len(runs)
    for arm_id in arm_ids_sorted:
        label = _clean_tactic_name(arm_id)
        color = TACTIC_FAMILY_COLORS.get(label, "#9E9E9E")
        values = arm_share_series[arm_id]
        ax_area.fill_between(x_pos, bottom, [b + v for b, v in zip(bottom, values)],
                             label=label, color=color, alpha=0.80)
        bottom = [b + v for b, v in zip(bottom, values)]
    ax_area.set_ylabel("Cumulative pull share")
    ax_area.set_ylim(0, 1.0)
    ax_area.legend(loc="upper right", fontsize=7, ncol=3)
    ax_area.set_title(
        f"{alg.upper()} Explore/Exploit — {benchmark.upper()} ({experiment_split})")
    ax_ent.plot(x_pos, entropy_series, color="#1565C0", linewidth=1.8, label="Arm entropy")
    ax_ent.axhline(h_uniform, color="#999", linestyle="--", linewidth=1.2,
                   label=f"H_uniform = {h_uniform:.2f} nats ({num_arms} arms)")
    ax_ent.set_xlabel("Training problem")
    ax_ent.set_ylabel("Entropy (nats)")
    ax_ent.set_ylim(0, h_uniform * 1.15)
    ax_ent.xaxis.set_major_locator(MaxNLocator(nbins=15, integer=True))
    ax_ent.legend(fontsize=8)
    return _save_figure(fig, output_path)


# ---------------------------------------------------------------------------
# Policy comparison — test split (primary thesis figure)
# ---------------------------------------------------------------------------

def _plot_policy_comparison_test(
    run_rows: list[dict],
    grouped_summary: list[dict],
    output_path: Path,
) -> str | None:
    """Headline figure: all policies (Random, ReAct, UCB1, Thompson, KL-UCB, EXP3)
    on test split, grouped by benchmark. Uses 1-shot ASR as primary metric."""
    test_rows = [r for r in run_rows if (r.get("experiment_split") or "") == "test"]
    if not test_rows:
        return None

    benchmarks = sorted({str(r.get("benchmark") or "") for r in test_rows if r.get("benchmark")})
    if not benchmarks:
        return None

    # Build per-(benchmark, policy_label) 1-shot ASR lists (primary metric)
    buckets: dict[tuple[str, str], list[float]] = defaultdict(list)
    for row in test_rows:
        bm = str(row.get("benchmark") or "")
        label = _policy_label(row)
        asr = row.get("asr_1shot_win") if row.get("asr_1shot_win") is not None else row.get("attack_success_rate")
        if asr is not None:
            buckets[(bm, label)].append(float(asr))

    all_labels = sorted({lb for _, lb in buckets.keys()}, key=_policy_sort_key)
    if not all_labels:
        return None

    n_bm = len(benchmarks)
    n_pol = len(all_labels)
    bar_width = 0.8 / max(n_pol, 1)
    x_pos = list(range(n_bm))

    fig, ax = plt.subplots(figsize=(max(7, n_bm * n_pol * 0.8), 6), constrained_layout=True)

    all_heights: list[float] = []
    for pi, label in enumerate(all_labels):
        heights, errs, annots = [], [], []
        for bm in benchmarks:
            vals = buckets.get((bm, label), [])
            m = _mean(vals) or 0.0
            e = (sqrt(sum((v - m) ** 2 for v in vals) / len(vals)) if len(vals) > 1 else 0.0)
            heights.append(m * 100)
            errs.append(e * 100)
            annots.append(f"{m * 100:.1f}%")
            all_heights.append(m * 100)
        offsets = [x + (pi - (n_pol - 1) / 2) * bar_width for x in x_pos]
        color = _policy_color(label)
        bars = ax.bar(offsets, heights, width=bar_width, label=label, color=color, alpha=0.88,
                      yerr=errs, capsize=4, error_kw={"elinewidth": 1.5, "ecolor": "#333"})
        for bar, ann in zip(bars, annots):
            h = bar.get_height()
            if h > 2:
                ax.text(bar.get_x() + bar.get_width() / 2, h + max(errs) + 0.5,
                        ann, ha="center", va="bottom", fontsize=8, fontweight="bold")

    valid = [h for h in all_heights if h > 0]
    y_min = max(0.0, (min(valid) - 10.0) // 5 * 5) if valid else 0.0
    y_max = min(103.0, max(valid) + 8.0) if valid else 100.0

    ax.set_xticks(x_pos)
    ax.set_xticklabels([b.upper() for b in benchmarks], fontsize=12)
    ax.set_ylabel("Attack Success Rate (%)")
    ax.set_title("Policy Comparison — Test Split")
    ax.set_ylim(y_min, y_max)
    ax.legend(loc="lower right", ncol=2)
    return _save_figure(fig, output_path)


# ---------------------------------------------------------------------------
# Algorithm comparison (RL only) — new
# ---------------------------------------------------------------------------

def _plot_algorithm_comparison(run_rows: list[dict], output_path: Path) -> str | None:
    """Bar chart: UCB1 vs Thompson vs KL-UCB vs EXP3 on test split (frozen weights).
    One group per benchmark, one bar per algorithm."""
    # Only frozen-weight eval rows on test split
    eval_rows = [
        r for r in run_rows
        if r.get("policy_mode") == "rl_bandit"
        and (r.get("experiment_split") or "") == "test"
        and r.get("bandit_freeze_weights_effective") is True
        and r.get("bandit_algorithm")
    ]
    if not eval_rows:
        return None

    benchmarks = sorted({str(r.get("benchmark") or "") for r in eval_rows if r.get("benchmark")})
    algos_present = sorted({str(r.get("bandit_algorithm") or "") for r in eval_rows},
                           key=lambda a: ALGO_ORDER.index(a) if a in ALGO_ORDER else 99)
    if not benchmarks or not algos_present:
        return None

    buckets: dict[tuple[str, str], list[float]] = defaultdict(list)
    for row in eval_rows:
        bm = str(row.get("benchmark") or "")
        alg = str(row.get("bandit_algorithm") or "")
        asr = row.get("asr_1shot_win") if row.get("asr_1shot_win") is not None else row.get("attack_success_rate")
        if asr is not None:
            buckets[(bm, alg)].append(float(asr))

    n_bm = len(benchmarks)
    n_alg = len(algos_present)
    bar_width = 0.8 / max(n_alg, 1)
    x_pos = list(range(n_bm))

    fig, ax = plt.subplots(figsize=(max(7, n_bm * n_alg * 0.9), 6), constrained_layout=True)
    all_heights: list[float] = []
    for ai, alg in enumerate(algos_present):
        heights, errs, annots = [], [], []
        for bm in benchmarks:
            vals = buckets.get((bm, alg), [])
            m = _mean(vals) or 0.0
            e = (sqrt(sum((v - m) ** 2 for v in vals) / len(vals)) if len(vals) > 1 else 0.0)
            heights.append(m * 100)
            errs.append(e * 100)
            annots.append(f"{m * 100:.1f}%")
            all_heights.append(m * 100)
        offsets = [x + (ai - (n_alg - 1) / 2) * bar_width for x in x_pos]
        color = ALGO_COLORS.get(alg, "#9E9E9E")
        bars = ax.bar(offsets, heights, width=bar_width,
                      label=ALGO_DISPLAY.get(alg, alg), color=color, alpha=0.88,
                      yerr=errs, capsize=4, error_kw={"elinewidth": 1.5, "ecolor": "#333"})
        for bar, ann in zip(bars, annots):
            h = bar.get_height()
            if h > 2:
                ax.text(bar.get_x() + bar.get_width() / 2, h + max(errs) + 0.5,
                        ann, ha="center", va="bottom", fontsize=8, fontweight="bold")

    valid = [h for h in all_heights if h > 0]
    y_min = max(0.0, (min(valid) - 10.0) // 5 * 5) if valid else 0.0
    y_max = min(103.0, max(valid) + 8.0) if valid else 100.0
    ax.set_xticks(x_pos)
    ax.set_xticklabels([b.upper() for b in benchmarks], fontsize=12)
    ax.set_ylabel("Attack Success Rate (%)")
    ax.set_title("Bandit Algorithm Comparison — Test Split (Frozen Weights)")
    ax.set_ylim(y_min, y_max)
    ax.legend(loc="lower right", title="Algorithm")
    return _save_figure(fig, output_path)


# ---------------------------------------------------------------------------
# Cross-model heatmap — new
# ---------------------------------------------------------------------------

def _plot_cross_model_heatmap(
    run_rows: list[dict],
    benchmark: str,
    output_path: Path,
) -> str | None:
    """2D heatmap: attacker (columns) × judge (rows), value = best ASR across all
    policies on test split. Shows which (attacker, judge) pair is hardest to fool."""
    rows = [
        r for r in run_rows
        if (r.get("benchmark") or "").lower() == benchmark.lower()
        and (r.get("experiment_split") or "") == "test"
        and r.get("selector_model")
        and r.get("target_model")
    ]
    if not rows:
        return None

    # Best 1-shot ASR per (attacker, judge) across all policies
    best: dict[tuple[str, str], float] = {}
    for row in rows:
        key = (_short_model(row.get("selector_model")), _short_model(row.get("target_model")))
        asr = float(row.get("asr_1shot_win") if row.get("asr_1shot_win") is not None else (row.get("attack_success_rate") or 0.0))
        best[key] = max(best.get(key, 0.0), asr)

    attackers = sorted({k[0] for k in best})
    judges = sorted({k[1] for k in best})
    if len(attackers) < 1 or len(judges) < 1:
        return None

    matrix = np.full((len(judges), len(attackers)), np.nan)
    for (att, jud), asr in best.items():
        if att in attackers and jud in judges:
            ci = attackers.index(att)
            ri = judges.index(jud)
            matrix[ri, ci] = asr * 100

    fig, ax = plt.subplots(figsize=(max(5, len(attackers) * 1.6), max(4, len(judges) * 1.2)),
                           constrained_layout=True)
    cmap = mcolors.LinearSegmentedColormap.from_list(
        "asr", ["#1565C0", "#FFF176", "#B71C1C"])
    masked = np.ma.masked_invalid(matrix)
    im = ax.imshow(masked, cmap=cmap, vmin=0, vmax=100, aspect="auto")
    plt.colorbar(im, ax=ax, label="Best ASR (%)")

    ax.set_xticks(range(len(attackers)))
    ax.set_xticklabels(attackers, rotation=30, ha="right", fontsize=10)
    ax.set_yticks(range(len(judges)))
    ax.set_yticklabels(judges, fontsize=10)
    ax.set_xlabel("Attacker model")
    ax.set_ylabel("Judge model")
    ax.set_title(f"Cross-Model Attack Success — {benchmark.upper()} (Test, Best Policy)")

    for ri in range(len(judges)):
        for ci in range(len(attackers)):
            val = matrix[ri, ci]
            if not np.isnan(val):
                text_color = "white" if val > 60 or val < 20 else "black"
                ax.text(ci, ri, f"{val:.1f}%", ha="center", va="center",
                        fontsize=9, fontweight="bold", color=text_color)
    return _save_figure(fig, output_path)


# ---------------------------------------------------------------------------
# Algorithm × judge heatmap — new
# ---------------------------------------------------------------------------

def _plot_algorithm_judge_heatmap(
    run_rows: list[dict],
    benchmark: str,
    output_path: Path,
) -> str | None:
    """2D heatmap: algorithm (rows) × judge (columns), value = mean test ASR.
    Uses the primary attacker model (most data) or averages across attackers."""
    rows = [
        r for r in run_rows
        if (r.get("benchmark") or "").lower() == benchmark.lower()
        and r.get("policy_mode") == "rl_bandit"
        and (r.get("experiment_split") or "") == "test"
        and r.get("bandit_freeze_weights_effective") is True
        and r.get("bandit_algorithm")
        and r.get("target_model")
    ]
    if not rows:
        return None

    buckets: dict[tuple[str, str], list[float]] = defaultdict(list)
    for row in rows:
        alg = str(row.get("bandit_algorithm") or "")
        jud = _short_model(row.get("target_model"))
        asr = row.get("asr_1shot_win") if row.get("asr_1shot_win") is not None else row.get("attack_success_rate")
        if asr is not None:
            buckets[(alg, jud)].append(float(asr))

    algos = [a for a in ALGO_ORDER if any(k[0] == a for k in buckets)]
    judges = sorted({k[1] for k in buckets})
    if not algos or not judges:
        return None

    matrix = np.full((len(algos), len(judges)), np.nan)
    for (alg, jud), vals in buckets.items():
        if alg in algos and jud in judges:
            ri = algos.index(alg)
            ci = judges.index(jud)
            matrix[ri, ci] = (_mean(vals) or 0.0) * 100

    fig, ax = plt.subplots(figsize=(max(5, len(judges) * 1.6), max(3, len(algos) * 1.1)),
                           constrained_layout=True)
    cmap = mcolors.LinearSegmentedColormap.from_list(
        "asr", ["#1565C0", "#FFF176", "#B71C1C"])
    masked = np.ma.masked_invalid(matrix)
    im = ax.imshow(masked, cmap=cmap, vmin=0, vmax=100, aspect="auto")
    plt.colorbar(im, ax=ax, label="Mean ASR (%)")

    ax.set_xticks(range(len(judges)))
    ax.set_xticklabels(judges, rotation=30, ha="right", fontsize=10)
    ax.set_yticks(range(len(algos)))
    ax.set_yticklabels([ALGO_DISPLAY.get(a, a) for a in algos], fontsize=10)
    ax.set_xlabel("Judge model")
    ax.set_ylabel("Bandit algorithm")
    ax.set_title(f"Algorithm × Judge Success Rate — {benchmark.upper()} (Test)")

    for ri in range(len(algos)):
        for ci in range(len(judges)):
            val = matrix[ri, ci]
            if not np.isnan(val):
                text_color = "white" if val > 60 or val < 20 else "black"
                ax.text(ci, ri, f"{val:.1f}%", ha="center", va="center",
                        fontsize=9, fontweight="bold", color=text_color)
    return _save_figure(fig, output_path)


# ---------------------------------------------------------------------------
# Iteration ablation — new
# ---------------------------------------------------------------------------

def _plot_iteration_ablation(run_rows: list[dict], output_path: Path) -> str | None:
    """Line chart: ASR vs max_iterations (1, 3, 12) for Random and ReAct,
    one subplot per benchmark. Fills the one-shot vs iterative comparison."""
    rows = [
        r for r in run_rows
        if r.get("policy_mode") in ("random_choice", "agent_based_decision")
        and r.get("max_iterations") is not None
        and (r.get("experiment_split") or "") == "test"
    ]
    if not rows:
        return None

    benchmarks = sorted({str(r.get("benchmark") or "") for r in rows if r.get("benchmark")})
    if not benchmarks:
        return None

    # Gather distinct iteration values, cast to int
    iter_values = sorted({int(r["max_iterations"]) for r in rows})
    policies = ["random_choice", "agent_based_decision"]
    policy_labels = {"random_choice": "Random", "agent_based_decision": "ReAct"}
    policy_colors = {"random_choice": "#FF9800", "agent_based_decision": "#4CAF50"}

    n_bm = len(benchmarks)
    fig, axes = plt.subplots(1, n_bm, figsize=(5 * n_bm, 5), sharey=True, constrained_layout=True)
    if n_bm == 1:
        axes = [axes]

    for ax, bm in zip(axes, benchmarks):
        for pm in policies:
            y_vals, y_errs = [], []
            for itr in iter_values:
                vals = [
                    float(r["attack_success_rate"])
                    for r in rows
                    if r.get("policy_mode") == pm
                    and (r.get("benchmark") or "") == bm
                    and int(r["max_iterations"]) == itr
                    and r.get("attack_success_rate") is not None
                ]
                m = _mean(vals) or 0.0
                e = (sqrt(sum((v - m) ** 2 for v in vals) / len(vals)) if len(vals) > 1 else 0.0)
                y_vals.append(m * 100)
                y_errs.append(e * 100)
            if any(v > 0 for v in y_vals):
                ax.errorbar(iter_values, y_vals, yerr=y_errs, marker="o", linewidth=2,
                            capsize=4, label=policy_labels[pm], color=policy_colors[pm])
                for x, y in zip(iter_values, y_vals):
                    ax.annotate(f"{y:.1f}%", (x, y), textcoords="offset points",
                                xytext=(0, 8), ha="center", fontsize=8)
        ax.set_xlabel("Max iterations budget")
        ax.set_title(bm.upper())
        ax.set_xticks(iter_values)
        ax.legend(loc="lower right")

    axes[0].set_ylabel("Attack Success Rate (%)")
    fig.suptitle("Iteration Budget Ablation — Test Split", fontsize=13, fontweight="bold")
    return _save_figure(fig, output_path)



# ---------------------------------------------------------------------------
# One attacker vs multiple judges — professor's request
# ---------------------------------------------------------------------------

def _plot_attacker_vs_judges(
    run_rows: list[dict],
    benchmark: str,
    output_path_dir: Path,
) -> list[str]:
    """For each distinct attacker model in the data, produce one figure:
    X-axis = judge model, grouped bars = policy, Y-axis = ASR (test split).
    Shows how easy/hard each judge is to fool for a given attacker."""
    rows = [
        r for r in run_rows
        if (r.get("benchmark") or "").lower() == benchmark.lower()
        and (r.get("experiment_split") or "") == "test"
        and r.get("selector_model")
        and r.get("target_model")
    ]
    if not rows:
        return []

    attackers = sorted({_short_model(r.get("selector_model")) for r in rows})
    saved: list[str] = []

    for attacker in attackers:
        attacker_rows = [r for r in rows if _short_model(r.get("selector_model")) == attacker]
        judges = sorted({_short_model(r.get("target_model")) for r in attacker_rows})
        if not judges:
            continue

        # Collect per (judge, policy_label) 1-shot ASR values
        buckets: dict[tuple[str, str], list[float]] = defaultdict(list)
        for row in attacker_rows:
            jud = _short_model(row.get("target_model"))
            label = _policy_label(row)
            asr = row.get("asr_1shot_win") if row.get("asr_1shot_win") is not None else row.get("attack_success_rate")
            if asr is not None:
                buckets[(jud, label)].append(float(asr))

        all_labels = sorted({lb for _, lb in buckets.keys()}, key=_policy_sort_key)
        if not all_labels:
            continue

        n_jud = len(judges)
        n_pol = len(all_labels)
        bar_width = 0.8 / max(n_pol, 1)
        x_pos = list(range(n_jud))

        fig, ax = plt.subplots(figsize=(max(7, n_jud * n_pol * 0.85), 6), constrained_layout=True)
        all_heights: list[float] = []

        for pi, label in enumerate(all_labels):
            heights, errs, annots = [], [], []
            for jud in judges:
                vals = buckets.get((jud, label), [])
                m = _mean(vals) or 0.0
                e = (sqrt(sum((v - m) ** 2 for v in vals) / len(vals)) if len(vals) > 1 else 0.0)
                heights.append(m * 100)
                errs.append(e * 100)
                annots.append(f"{m * 100:.1f}%")
                all_heights.append(m * 100)
            offsets = [x + (pi - (n_pol - 1) / 2) * bar_width for x in x_pos]
            color = _policy_color(label)
            bars = ax.bar(offsets, heights, width=bar_width, label=label, color=color, alpha=0.88,
                          yerr=errs, capsize=4, error_kw={"elinewidth": 1.5, "ecolor": "#333"})
            for bar, ann in zip(bars, annots):
                h = bar.get_height()
                if h > 2:
                    ax.text(bar.get_x() + bar.get_width() / 2, h + (max(errs) if errs else 0) + 0.5,
                            ann, ha="center", va="bottom", fontsize=7, fontweight="bold")

        ax.set_xticks(x_pos)
        ax.set_xticklabels(judges, rotation=25, ha="right", fontsize=10)
        valid = [h for h in all_heights if h > 0]
        ax.set_ylim(0, min(105, max(valid) * 1.2 + 5) if valid else 100)
        ax.set_xlabel("Judge model")
        ax.set_ylabel("Attack Success Rate (%)")
        ax.set_title(
            f"Attacker: {attacker}  —  ASR per Judge\n"
            f"Benchmark: {benchmark.upper()}, Test split"
        )
        ax.legend(loc="upper right", ncol=2)

        fname = _sanitize_filename(f"attacker_vs_judges_{benchmark}_{attacker}") + ".png"
        path = _save_figure(fig, output_path_dir / fname)
        saved.append(path)

    return saved


def _plot_iterations_by_judge(
    run_rows: list[dict],
    benchmark: str,
    output_path_dir: Path,
) -> list[str]:
    """For each attacker, bar chart of mean iterations-to-success per judge.
    Shorter bars = judge is easy to fool quickly; longer = judge forces more attempts."""
    rows = [
        r for r in run_rows
        if (r.get("benchmark") or "").lower() == benchmark.lower()
        and (r.get("experiment_split") or "") == "test"
        and r.get("selector_model")
        and r.get("target_model")
        and r.get("average_iterations_to_success") is not None
    ]
    if not rows:
        return []

    attackers = sorted({_short_model(r.get("selector_model")) for r in rows})
    saved: list[str] = []

    for attacker in attackers:
        attacker_rows = [r for r in rows if _short_model(r.get("selector_model")) == attacker]
        judges = sorted({_short_model(r.get("target_model")) for r in attacker_rows})
        if not judges:
            continue

        # Collect per (judge, policy_label) iterations values
        iter_buckets: dict[tuple[str, str], list[float]] = defaultdict(list)
        for row in attacker_rows:
            jud = _short_model(row.get("target_model"))
            label = _policy_label(row)
            itr = row.get("average_iterations_to_success")
            if itr is not None:
                iter_buckets[(jud, label)].append(float(itr))

        all_labels = sorted({lb for _, lb in iter_buckets.keys()}, key=_policy_sort_key)
        if not all_labels:
            continue

        n_jud = len(judges)
        n_pol = len(all_labels)
        bar_width = 0.8 / max(n_pol, 1)
        x_pos = list(range(n_jud))

        fig, ax = plt.subplots(figsize=(max(7, n_jud * n_pol * 0.85), 6), constrained_layout=True)

        for pi, label in enumerate(all_labels):
            heights, errs = [], []
            for jud in judges:
                vals = iter_buckets.get((jud, label), [])
                m = _mean(vals) or 0.0
                e = (sqrt(sum((v - m) ** 2 for v in vals) / len(vals)) if len(vals) > 1 else 0.0)
                heights.append(m)
                errs.append(e)
            offsets = [x + (pi - (n_pol - 1) / 2) * bar_width for x in x_pos]
            color = _policy_color(label)
            bars = ax.bar(offsets, heights, width=bar_width, label=label, color=color, alpha=0.88,
                          yerr=errs, capsize=4, error_kw={"elinewidth": 1.5, "ecolor": "#333"})
            for bar, h in zip(bars, heights):
                if h > 0.05:
                    ax.text(bar.get_x() + bar.get_width() / 2, h + 0.05,
                            f"{h:.2f}", ha="center", va="bottom", fontsize=8)

        ax.set_xticks(x_pos)
        ax.set_xticklabels(judges, rotation=25, ha="right", fontsize=10)
        ax.set_ylim(0, None)
        ax.set_xlabel("Judge model")
        ax.set_ylabel("Mean iterations to success")
        ax.set_title(
            f"Attacker: {attacker}  —  Iterations to Success per Judge\n"
            f"Benchmark: {benchmark.upper()}, Test split"
        )
        ax.legend(loc="upper right", ncol=2)

        fname = _sanitize_filename(f"iterations_by_judge_{benchmark}_{attacker}") + ".png"
        path = _save_figure(fig, output_path_dir / fname)
        saved.append(path)

    return saved


# ---------------------------------------------------------------------------
# Epoch-level convergence plot — reads weights/*_entropy_curve.json
# ---------------------------------------------------------------------------

def _plot_epoch_entropy_curve(weights_dir: Path, output_path: Path) -> str | None:
    """Normalized arm entropy per training epoch, one line per (algorithm, benchmark).
    Reads weights/*_entropy_curve.json files produced by log_entropy.py.
    y=1.0 = full exploration (uniform), y→0 = exploitation settled on one arm."""
    from matplotlib.ticker import MaxNLocator

    curve_files = sorted(weights_dir.glob("*_entropy_curve.json"))
    if not curve_files:
        return None

    series: list[dict] = []
    for f in curve_files:
        stem = f.stem.replace("_entropy_curve", "")
        alg = next((a for a in ALGO_ORDER if f"_{a}_" in f"_{stem}_"), None)
        if alg is None:
            continue
        try:
            curve = json.loads(f.read_text(encoding="utf-8"))
        except Exception:
            continue
        if not curve:
            continue
        bm_part = stem.split(f"_{alg}")[0]
        bm_label = "acb" if bm_part == "acb" else bm_part
        series.append({
            "label": f"{ALGO_DISPLAY.get(alg, alg)} / {bm_label}",
            "alg": alg,
            "bm": bm_label,
            "epochs": [e["epoch"] for e in curve],
            "entropies": [e["normalized_entropy"] for e in curve],
        })

    if not series:
        return None

    fig, ax = plt.subplots(figsize=(9, 5), constrained_layout=True)
    ax.axhline(1.0, color="#ccc", linestyle=":", linewidth=1.2, label="H_uniform (pure exploration)")

    for s in series:
        color = ALGO_COLORS.get(s["alg"], "#9E9E9E")
        linestyle = "-" if s["bm"] == "acb" else "--"
        ax.plot(s["epochs"], s["entropies"], marker="o", linewidth=2,
                linestyle=linestyle, color=color, label=s["label"])

    ax.set_xlabel("Training epoch")
    ax.set_ylabel("Normalized arm entropy  (1 = uniform, 0 = one arm)")
    ax.set_title("Bandit Convergence — Arm Entropy per Epoch")
    ax.set_ylim(0, 1.15)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.legend(loc="lower left", fontsize=8, ncol=2)
    return _save_figure(fig, output_path)


# ---------------------------------------------------------------------------
# Tactic × judge win-rate heatmap
# ---------------------------------------------------------------------------

def _plot_tactic_judge_heatmap(
    run_rows: list[dict],
    benchmark: str,
    output_path: Path,
) -> str | None:
    """9 tactics (rows) × judge models (columns), cell = win rate.
    Only cells with ≥5 pulls are shown. RL bandit runs only."""
    rows = [
        r for r in run_rows
        if r.get("policy_mode") == "rl_bandit"
        and (r.get("benchmark") or "").lower() == benchmark.lower()
        and r.get("target_model")
    ]
    if not rows:
        return None

    judge_arm_success: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    judge_arm_pulls: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    for row in rows:
        jud = _short_model(row.get("target_model"))
        for arm_id, cnt in (row.get("success_by_arm") or {}).items():
            judge_arm_success[jud][str(arm_id)] += float(cnt)
        for arm_id, cnt in (row.get("pulls_by_arm") or {}).items():
            judge_arm_pulls[jud][str(arm_id)] += float(cnt)

    judges = sorted(judge_arm_pulls.keys())
    all_tactics: set[str] = set()
    for d in judge_arm_pulls.values():
        all_tactics.update(d.keys())
    tactics = sorted(all_tactics)
    if not judges or not tactics:
        return None

    tactic_labels = [_clean_tactic_name(t) for t in tactics]
    matrix = np.full((len(tactics), len(judges)), np.nan)
    for ri, tactic_id in enumerate(tactics):
        for ci, jud in enumerate(judges):
            pulls = judge_arm_pulls[jud].get(tactic_id, 0.0)
            if pulls >= 5:
                matrix[ri, ci] = (judge_arm_success[jud].get(tactic_id, 0.0) / pulls) * 100

    fig, ax = plt.subplots(
        figsize=(max(5, len(judges) * 1.8), max(4, len(tactics) * 0.85)),
        constrained_layout=True,
    )
    cmap = mcolors.LinearSegmentedColormap.from_list("asr", ["#1565C0", "#FFF176", "#B71C1C"])
    masked = np.ma.masked_invalid(matrix)
    im = ax.imshow(masked, cmap=cmap, vmin=0, vmax=100, aspect="auto")
    plt.colorbar(im, ax=ax, label="Win rate (%)")

    ax.set_xticks(range(len(judges)))
    ax.set_xticklabels(judges, rotation=30, ha="right", fontsize=10)
    ax.set_yticks(range(len(tactics)))
    ax.set_yticklabels(tactic_labels, fontsize=9)
    ax.set_xlabel("Judge model")
    ax.set_ylabel("Tactic")
    ax.set_title(f"Tactic × Judge Win Rate — {benchmark.upper()} (RL Bandit, all splits)")

    for ri in range(len(tactics)):
        for ci in range(len(judges)):
            val = matrix[ri, ci]
            if not np.isnan(val):
                text_color = "white" if val > 60 or val < 20 else "black"
                ax.text(ci, ri, f"{val:.0f}%", ha="center", va="center",
                        fontsize=8, fontweight="bold", color=text_color)
            else:
                ax.text(ci, ri, "–", ha="center", va="center", fontsize=8, color="#aaa")
    return _save_figure(fig, output_path)


# ---------------------------------------------------------------------------
# Per-epoch ASR learning curve — one line per bandit algorithm
# ---------------------------------------------------------------------------

def _plot_epoch_asr_curve(evolution_by_group: list[dict], output_path: Path) -> str | None:
    """ASR per training epoch, one line per bandit algorithm.
    Averages over all model combinations that share the same algorithm.
    Shaded band = ±1 std across combinations."""
    from matplotlib.ticker import MaxNLocator

    train_groups = [
        g for g in evolution_by_group
        if g.get("policy_mode") == "rl_bandit"
        and (g.get("experiment_split") or "") == "train"
        and g.get("bandit_algorithm")
        and g.get("benchmark")
    ]
    if not train_groups:
        return None

    benchmarks = sorted({str(g["benchmark"]) for g in train_groups})
    if not benchmarks:
        return None

    # (benchmark, algorithm) -> list of per-epoch ASR sequences (one per model combo)
    combo_asr: dict[tuple[str, str], list[list[float]]] = defaultdict(list)
    for group in train_groups:
        bm = str(group["benchmark"])
        alg = str(group["bandit_algorithm"])
        epoch_asrs = [
            float(r["attack_success_rate"]) * 100
            for r in group.get("runs", [])
            if r.get("attack_success_rate") is not None
        ]
        if len(epoch_asrs) >= 2:
            combo_asr[(bm, alg)].append(epoch_asrs)

    if not combo_asr:
        return None

    n_bm = len(benchmarks)
    fig, axes = plt.subplots(1, n_bm, figsize=(6 * n_bm, 5),
                             sharey=False, constrained_layout=True)
    if n_bm == 1:
        axes = [axes]

    for ax, bm in zip(axes, benchmarks):
        for alg in ALGO_ORDER:
            combos = combo_asr.get((bm, alg), [])
            if not combos:
                continue
            min_len = min(len(c) for c in combos)
            aligned = [c[:min_len] for c in combos]
            means = [sum(c[i] for c in aligned) / len(aligned) for i in range(min_len)]
            stds = [
                sqrt(sum((c[i] - means[i]) ** 2 for c in aligned) / max(len(aligned) - 1, 1))
                for i in range(min_len)
            ]
            epochs = list(range(1, min_len + 1))
            color = ALGO_COLORS.get(alg, "#9E9E9E")
            ax.plot(epochs, means, marker="o", linewidth=2, color=color,
                    label=ALGO_DISPLAY.get(alg, alg))
            ax.fill_between(
                epochs,
                [m - s for m, s in zip(means, stds)],
                [m + s for m, s in zip(means, stds)],
                alpha=0.15, color=color,
            )

        ax.set_xlabel("Training epoch")
        ax.set_ylabel("Attack Success Rate (%)")
        ax.set_title(bm.upper())
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.legend(loc="lower right", fontsize=8)

    fig.suptitle("Bandit Training Progress — ASR per Epoch",
                 fontsize=13, fontweight="bold")
    return _save_figure(fig, output_path)


# ---------------------------------------------------------------------------
# Cumulative empirical regret — one line per bandit algorithm
# ---------------------------------------------------------------------------

def _plot_cumulative_regret(evolution_by_group: list[dict], output_path: Path) -> str | None:
    """Cumulative empirical regret per training epoch, one line per algorithm.
    Regret_t = best_arm_rate * N_t - R_t  (hindsight-optimal arm from full history).
    Lower and flatter is better — EXP3 has O(√T) theoretical guarantee."""
    from matplotlib.ticker import MaxNLocator

    train_groups = [
        g for g in evolution_by_group
        if g.get("policy_mode") == "rl_bandit"
        and (g.get("experiment_split") or "") == "train"
        and g.get("bandit_algorithm")
        and g.get("benchmark")
    ]
    if not train_groups:
        return None

    benchmarks = sorted({str(g["benchmark"]) for g in train_groups})
    if not benchmarks:
        return None

    combo_regret: dict[tuple[str, str], list[list[float]]] = defaultdict(list)

    for group in train_groups:
        bm = str(group["benchmark"])
        alg = str(group["bandit_algorithm"])
        runs = [r for r in group.get("runs", [])
                if r.get("cumulative_reward_by_arm") and r.get("pulls_by_arm")]
        if len(runs) < 2:
            continue

        # Identify hindsight-optimal arm from final epoch
        final = runs[-1]
        final_cum_r = final.get("cumulative_reward_by_arm") or {}
        final_pulls = final.get("pulls_by_arm") or {}
        arm_rates = {
            arm: float(final_cum_r.get(arm, 0)) / float(p)
            for arm, p in final_pulls.items()
            if float(p) > 0
        }
        if not arm_rates:
            continue
        best_rate = max(arm_rates.values())

        # Cumulative regret at each epoch checkpoint
        regret_seq: list[float] = []
        for run in runs:
            cum_r = run.get("cumulative_reward_by_arm") or {}
            cum_p = run.get("pulls_by_arm") or {}
            N = sum(float(v) for v in cum_p.values())
            R = sum(float(v) for v in cum_r.values())
            regret_seq.append(max(0.0, best_rate * N - R))

        if regret_seq:
            combo_regret[(bm, alg)].append(regret_seq)

    if not combo_regret:
        return None

    n_bm = len(benchmarks)
    fig, axes = plt.subplots(1, n_bm, figsize=(6 * n_bm, 5),
                             sharey=False, constrained_layout=True)
    if n_bm == 1:
        axes = [axes]

    _N_CHECKPOINTS = 50  # subsample to this many evenly-spaced points for readability

    for ax, bm in zip(axes, benchmarks):
        for alg in ALGO_ORDER:
            combos = combo_regret.get((bm, alg), [])
            if not combos:
                continue
            min_len = min(len(c) for c in combos)
            aligned = [c[:min_len] for c in combos]
            means = [sum(c[i] for c in aligned) / len(aligned) for i in range(min_len)]
            stds = [
                sqrt(sum((c[i] - means[i]) ** 2 for c in aligned) / max(len(aligned) - 1, 1))
                for i in range(min_len)
            ]
            # Subsample to at most _N_CHECKPOINTS evenly-spaced indices
            step = max(1, min_len // _N_CHECKPOINTS)
            idx = list(range(0, min_len, step))
            epochs = [i + 1 for i in idx]
            means_s = [means[i] for i in idx]
            stds_s = [stds[i] for i in idx]
            color = ALGO_COLORS.get(alg, "#9E9E9E")
            ax.plot(epochs, means_s, linewidth=2, color=color,
                    label=ALGO_DISPLAY.get(alg, alg))
            ax.fill_between(
                epochs,
                [max(0.0, m - s) for m, s in zip(means_s, stds_s)],
                [m + s for m, s in zip(means_s, stds_s)],
                alpha=0.15, color=color,
            )

        ax.set_xlabel("Training epoch")
        ax.set_ylabel("Cumulative empirical regret")
        ax.set_title(bm.upper())
        ax.set_ylim(bottom=0)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.legend(loc="upper left", fontsize=8)

    fig.suptitle("Cumulative Empirical Regret per Epoch — Bandit Algorithms",
                 fontsize=13, fontweight="bold")
    return _save_figure(fig, output_path)


# ---------------------------------------------------------------------------
# Train / validation / test ASR per algorithm (generalisation check)
# ---------------------------------------------------------------------------

def _plot_algo_split_comparison(run_rows: list[dict], output_path: Path) -> str | None:
    """Grouped bar: train / validation / test ASR for each bandit algorithm.
    Shows whether weights learned on the train split generalise to held-out splits."""
    SPLITS_ORDER = ["train", "validation", "test"]
    SPLIT_COLORS = {"train": "#42A5F5", "validation": "#FFA726", "test": "#66BB6A"}
    SPLIT_LABELS = {"train": "Train", "validation": "Validation", "test": "Test"}

    rows = [
        r for r in run_rows
        if r.get("policy_mode") == "rl_bandit"
        and r.get("bandit_algorithm")
        and (r.get("experiment_split") or "") in SPLITS_ORDER
        and r.get("attack_success_rate") is not None
    ]
    if not rows:
        return None

    benchmarks = sorted({str(r.get("benchmark") or "") for r in rows if r.get("benchmark")})
    if not benchmarks:
        return None

    n_bm = len(benchmarks)
    fig, axes = plt.subplots(1, n_bm, figsize=(6 * n_bm, 5),
                             sharey=False, constrained_layout=True)
    if n_bm == 1:
        axes = [axes]

    for ax, bm in zip(axes, benchmarks):
        bm_rows = [r for r in rows if (r.get("benchmark") or "") == bm]
        algos_present = [a for a in ALGO_ORDER
                         if any(r.get("bandit_algorithm") == a for r in bm_rows)]
        if not algos_present:
            continue

        buckets: dict[tuple[str, str], list[float]] = defaultdict(list)
        for row in bm_rows:
            buckets[(str(row["bandit_algorithm"]), str(row["experiment_split"]))].append(
                float(row["attack_success_rate"]) * 100
            )

        n_alg = len(algos_present)
        bar_width = 0.8 / 3
        x_pos = list(range(n_alg))
        all_heights: list[float] = []

        for si, split in enumerate(SPLITS_ORDER):
            heights = []
            for alg in algos_present:
                vals = buckets.get((alg, split), [])
                m = _mean(vals) or 0.0
                heights.append(m)
                all_heights.append(m)
            offsets = [x + (si - 1) * bar_width for x in x_pos]
            ax.bar(offsets, heights, width=bar_width,
                   label=SPLIT_LABELS[split], color=SPLIT_COLORS[split], alpha=0.88)

        ax.set_xticks(x_pos)
        ax.set_xticklabels([ALGO_DISPLAY.get(a, a) for a in algos_present], fontsize=10)
        ax.set_ylabel("Attack Success Rate (%)")
        ax.set_title(bm.upper())
        valid = [h for h in all_heights if h > 0]
        ax.set_ylim(0, min(105, max(valid) * 1.2 + 5) if valid else 100)
        ax.legend(loc="upper right", fontsize=8)

    fig.suptitle("Train / Validation / Test ASR per Bandit Algorithm",
                 fontsize=13, fontweight="bold")
    return _save_figure(fig, output_path)


# ---------------------------------------------------------------------------
# Per-tactic ASR across both datasets (transferability check)
# ---------------------------------------------------------------------------

def _plot_tactic_dataset_comparison(run_rows: list[dict], output_path: Path) -> str | None:
    """Grouped bar: per-tactic win rate on each benchmark.
    Shows whether tactic effectiveness transfers across bug types.
    Uses success_by_arm / pulls_by_arm from all RL bandit runs."""
    rows = [r for r in run_rows if r.get("policy_mode") == "rl_bandit"]
    if not rows:
        return None

    benchmarks = sorted({str(r.get("benchmark") or "") for r in rows if r.get("benchmark")})
    if len(benchmarks) < 2:
        return None

    BM_COLORS = [
        "#1565C0", "#E65100", "#2E7D32", "#6A1B9A",
    ]

    arm_success: dict[tuple[str, str], float] = defaultdict(float)
    arm_pulls: dict[tuple[str, str], float] = defaultdict(float)
    for row in rows:
        bm = str(row.get("benchmark") or "")
        for arm_id, cnt in (row.get("success_by_arm") or {}).items():
            arm_success[(str(arm_id), bm)] += float(cnt)
        for arm_id, cnt in (row.get("pulls_by_arm") or {}).items():
            arm_pulls[(str(arm_id), bm)] += float(cnt)

    all_arms = sorted({k[0] for k in arm_pulls})
    arms_with_data = [
        a for a in all_arms
        if any(arm_pulls.get((a, bm), 0) >= 5 for bm in benchmarks)
    ]
    if not arms_with_data:
        return None

    tactic_labels = [_clean_tactic_name(a) for a in arms_with_data]
    n_arms = len(arms_with_data)
    n_bm = len(benchmarks)
    bar_width = 0.8 / n_bm
    x_pos = list(range(n_arms))
    all_heights: list[float] = []

    fig, ax = plt.subplots(figsize=(max(9, n_arms * 1.4), 6), constrained_layout=True)

    for bi, bm in enumerate(benchmarks):
        heights: list[float] = []
        for arm_id in arms_with_data:
            pulls = arm_pulls.get((arm_id, bm), 0)
            success = arm_success.get((arm_id, bm), 0)
            heights.append((success / pulls * 100) if pulls >= 5 else 0.0)
            all_heights.append(heights[-1])
        offsets = [x + (bi - (n_bm - 1) / 2) * bar_width for x in x_pos]
        ax.bar(offsets, heights, width=bar_width, label=bm.upper(),
               color=BM_COLORS[bi % len(BM_COLORS)], alpha=0.88)

    ax.set_xticks(x_pos)
    ax.set_xticklabels(tactic_labels, rotation=20, ha="right", fontsize=9)
    ax.set_ylabel("Attack Success Rate (%)")
    ax.set_title("Tactic Effectiveness Across Datasets — RL Bandit")
    valid = [h for h in all_heights if h > 0]
    ax.set_ylim(0, min(105, max(valid) * 1.2 + 5) if valid else 100)
    ax.legend(title="Dataset", loc="upper right")
    return _save_figure(fig, output_path)


# ---------------------------------------------------------------------------
# LLM judge confidence distribution by policy
# ---------------------------------------------------------------------------

def _plot_confidence_distribution(run_rows: list[dict], output_path: Path) -> str | None:
    """Box plot of mean LLM judge confidence per run, grouped by policy.
    Uses run-level average_llm_confidence from run_summary.json.
    Higher confidence = judge more convinced the (adversarially framed) code is correct."""
    rows = [r for r in run_rows if r.get("average_llm_confidence") is not None]
    if not rows:
        return None

    by_policy: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        by_policy[_policy_label(row)].append(float(row["average_llm_confidence"]))

    policy_order = sorted(by_policy.keys(), key=_policy_sort_key)
    if not policy_order:
        return None

    data = [by_policy[p] for p in policy_order]
    colors = [_policy_color(p) for p in policy_order]

    fig, ax = plt.subplots(figsize=(max(7, len(policy_order) * 1.5), 6),
                           constrained_layout=True)
    bp = ax.boxplot(data, patch_artist=True, notch=False,
                    medianprops={"color": "black", "linewidth": 2},
                    flierprops={"marker": "o", "markersize": 4, "alpha": 0.5})
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.75)

    ax.set_xticks(range(1, len(policy_order) + 1))
    ax.set_xticklabels(policy_order, rotation=15, ha="right", fontsize=10)
    ax.set_ylabel("Mean LLM judge confidence (per run)")
    ax.set_title("LLM Judge Confidence Distribution by Policy")
    ax.set_ylim(0, 1.05)
    ax.axhline(0.5, color="#aaa", linestyle="--", linewidth=1.2, label="Decision threshold (0.5)")
    ax.legend(fontsize=8)
    return _save_figure(fig, output_path)


# ---------------------------------------------------------------------------
# Cumulative ASR curve per iteration budget — key thesis figure
# ---------------------------------------------------------------------------

def _plot_cumulative_asr_curve(
    run_rows: list[dict],
    benchmark: str,
    output_path: Path,
) -> str | None:
    """Cumulative ASR vs iteration budget (1–6), one line per policy.

    Each point (X=N, Y=p) means: p% of test samples had their attack succeed
    within N tactic attempts. Primary metric = 1-shot (X=1); 3-shot and 6-shot
    are secondary convergence checkpoints.

    For RL bandits, only frozen-weight test rows are included so the curve
    reflects the trained policy, not the training run itself.
    """
    from matplotlib.ticker import MaxNLocator

    SHOT_FIELDS = [
        ("baseline_win", 0),
        ("asr_1shot_win", 1),
        ("asr_2shot_win", 2),
        ("asr_3shot_win", 3),
        ("asr_4shot_win", 4),
        ("asr_5shot_win", 5),
        ("asr_6shot_win", 6),
    ]

    bm_rows = [
        r for r in run_rows
        if (r.get("benchmark") or "").lower() == benchmark.lower()
        and (r.get("experiment_split") or "") == "test"
    ]
    if not bm_rows:
        return None

    def _rows_for_label(label: str) -> list[dict]:
        if label in ("Random", "ReAct"):
            return [r for r in bm_rows if _policy_label(r) == label]
        # RL algorithm label — use frozen-weight eval rows only
        return [
            r for r in bm_rows
            if _policy_label(r) == label
            and r.get("bandit_freeze_weights_effective") is True
        ]

    all_labels = sorted({_policy_label(r) for r in bm_rows}, key=_policy_sort_key)
    if not all_labels:
        return None

    fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)

    has_data = False
    for label in all_labels:
        rows = _rows_for_label(label)
        if not rows:
            continue

        xs, ys = [], []
        for field, n in SHOT_FIELDS:
            vals = [float(r[field]) for r in rows if r.get(field) is not None]
            if vals:
                xs.append(n)
                ys.append(sum(vals) / len(vals) * 100)

        if len(xs) < 2:
            continue

        color = _policy_color(label)
        linestyle = "--" if label == "ReAct" else "-"
        marker = "s" if label == "ReAct" else "o"
        ax.plot(xs, ys, marker=marker, linewidth=2.2, linestyle=linestyle,
                color=color, label=f"{label} (n={len(rows)})")
        has_data = True

    if not has_data:
        plt.close(fig)
        return None

    ax.set_xlabel("Iteration budget (max tactic attempts)")
    ax.set_ylabel("Cumulative ASR (%)")
    ax.set_title(
        f"Cumulative Attack Success Rate vs Iteration Budget\n"
        f"Benchmark: {benchmark.upper()}, Test split"
    )
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.set_xlim(left=0)
    ax.set_ylim(0, 103)
    ax.axvline(1, color="#888", linestyle=":", linewidth=1.2, label="1-shot (primary)")
    ax.axvline(3, color="#bbb", linestyle=":", linewidth=1.0, label="3-shot")
    ax.legend(loc="lower right", fontsize=9, ncol=2)
    ax.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.6)
    return _save_figure(fig, output_path)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def generate_plots(aggregation: dict, output_dir: str) -> dict:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    run_rows = aggregation.get("runs") or []
    grouped_summary = aggregation.get("grouped_summary") or []
    evolution_by_group = aggregation.get("evolution_by_group") or []

    generated_files: list[str] = []

    # ---- Standard overview plots -------------------------------------------

    result = _plot_grouped_bar(
        [{"benchmark": r.get("benchmark"), "policy_mode": r.get("policy_mode"),
          "attack_success_rate": r.get("attack_success_rate")} for r in run_rows],
        x_field="benchmark", hue_field="policy_mode", value_field="attack_success_rate",
        title="Attack Success Rate by Policy Mode",
        ylabel="Attack success rate",
        output_path=output_path / "attack_success_rate_by_policy_mode.png",
    )
    if result:
        generated_files.append(result)

    iter_vals = [float(r["average_iterations_to_success"])
                 for r in run_rows if r.get("average_iterations_to_success") is not None]
    result = _plot_iterations_distribution(iter_vals, output_path / "iterations_to_success_distribution.png")
    if result:
        generated_files.append(result)

    # Arm pull / reward
    pull_totals: dict[str, float] = defaultdict(float)
    reward_totals: dict[str, list[float]] = defaultdict(list)
    for row in run_rows:
        for arm_id, c in (row.get("pulls_by_arm") or {}).items():
            pull_totals[str(arm_id)] += float(c)
        for arm_id, v in (row.get("average_reward_by_arm") or {}).items():
            reward_totals[str(arm_id)].append(float(v))

    pull_items = [{"arm": _clean_tactic_name(a), "pulls": p}
                  for a, p in sorted(pull_totals.items(), key=lambda i: i[1], reverse=True)]
    result = _plot_single_bar(
        pull_items, label_field="arm", value_field="pulls",
        title="Arm Pull Counts — RL Bandit", ylabel="Pull count",
        output_path=output_path / "arm_pull_counts.png", horizontal=True,
        item_colors=[TACTIC_FAMILY_COLORS.get(i["arm"], "#2196F3") for i in pull_items],
    )
    if result:
        generated_files.append(result)

    reward_items = [{"arm": _clean_tactic_name(a), "average_reward": _mean(vs) or 0.0}
                    for a, vs in sorted(reward_totals.items(), key=lambda i: _mean(i[1]) or 0.0, reverse=True)]
    result = _plot_single_bar(
        reward_items, label_field="arm", value_field="average_reward",
        title="Average Reward by Arm — RL Bandit", ylabel="Average reward",
        output_path=output_path / "average_reward_by_arm.png", horizontal=True,
        item_colors=[TACTIC_FAMILY_COLORS.get(i["arm"], "#2196F3") for i in reward_items],
    )
    if result:
        generated_files.append(result)

    # ---- Evolution / per-group plots ----------------------------------------

    for group in evolution_by_group:
        bm = str(group.get("benchmark") or "unknown")
        pm = str(group.get("policy_mode") or "unknown")
        em = str(group.get("experiment_mode") or "unknown")
        sp = str(group.get("experiment_split") or "full")
        alg = group.get("bandit_algorithm") or "none"
        suffix = _sanitize_filename(f"{bm}_{pm}_{em}_{sp}_{alg}")

        if group.get("bandit_algorithm"):
            result = _plot_exploit_explore(group, output_path / f"exploit_explore_{suffix}.png")
            if result:
                generated_files.append(result)

    # ---- Key thesis / comparison figures ------------------------------------

    result = _plot_policy_comparison_test(run_rows, grouped_summary, output_path / "policy_comparison_test.png")
    if result:
        generated_files.append(result)

    result = _plot_tactic_win_rate(run_rows, output_path / "tactic_win_rate.png")
    if result:
        generated_files.append(result)

    result = _plot_success_breakdown(run_rows, output_path / "success_breakdown_baseline_vs_tactic.png")
    if result:
        generated_files.append(result)

    result = _plot_training_learning_curve(run_rows, output_path / "training_learning_curve.png")
    if result:
        generated_files.append(result)

    # ---- Cross-model and algorithm comparison figures -----------------------

    result = _plot_algorithm_comparison(run_rows, output_path / "algorithm_comparison_test.png")
    if result:
        generated_files.append(result)

    result = _plot_iteration_ablation(run_rows, output_path / "iteration_ablation.png")
    if result:
        generated_files.append(result)

    # ---- New: RL training quality and generalisation figures ----------------

    result = _plot_epoch_asr_curve(evolution_by_group, output_path / "epoch_asr_curve.png")
    if result:
        generated_files.append(result)

    result = _plot_cumulative_regret(evolution_by_group, output_path / "cumulative_regret.png")
    if result:
        generated_files.append(result)

    result = _plot_algo_split_comparison(run_rows, output_path / "algo_split_comparison.png")
    if result:
        generated_files.append(result)

    result = _plot_tactic_dataset_comparison(run_rows, output_path / "tactic_dataset_comparison.png")
    if result:
        generated_files.append(result)

    result = _plot_confidence_distribution(run_rows, output_path / "confidence_distribution.png")
    if result:
        generated_files.append(result)

    benchmarks_present = sorted({str(r.get("benchmark") or "") for r in run_rows if r.get("benchmark")})
    for bm in benchmarks_present:
        result = _plot_cross_model_heatmap(
            run_rows, bm, output_path / f"cross_model_heatmap_{_sanitize_filename(bm)}.png")
        if result:
            generated_files.append(result)

        result = _plot_algorithm_judge_heatmap(
            run_rows, bm, output_path / f"algorithm_judge_heatmap_{_sanitize_filename(bm)}.png")
        if result:
            generated_files.append(result)

    # ---- Cumulative ASR curve per iteration budget (key thesis figure) -------

    for bm in benchmarks_present:
        result = _plot_cumulative_asr_curve(
            run_rows, bm,
            output_path / f"cumulative_asr_curve_{_sanitize_filename(bm)}.png")
        if result:
            generated_files.append(result)

    # ---- One attacker vs multiple judges (professor's request) --------------

    for bm in benchmarks_present:
        attacker_judge_plots = _plot_attacker_vs_judges(run_rows, bm, output_path)
        generated_files.extend(attacker_judge_plots)

        iter_judge_plots = _plot_iterations_by_judge(run_rows, bm, output_path)
        generated_files.extend(iter_judge_plots)

    # ---- Epoch-level convergence and tactic × judge heatmap ----------------

    result = _plot_epoch_entropy_curve(
        PROJECT_ROOT / "weights", output_path / "epoch_entropy_curve.png")
    if result:
        generated_files.append(result)

    for bm in benchmarks_present:
        result = _plot_tactic_judge_heatmap(
            run_rows, bm,
            output_path / f"tactic_judge_heatmap_{_sanitize_filename(bm)}.png")
        if result:
            generated_files.append(result)

    # ---- Manifest -----------------------------------------------------------

    manifest = {
        "generated_at": datetime.now().isoformat(),
        "run_count": len(run_rows),
        "group_count": len(grouped_summary),
        "plot_count": len(generated_files),
        "plots": generated_files,
        "source_summary": aggregation.get("aggregation_metadata", {}),
    }
    _write_json(output_path / "plot_manifest.json", manifest)
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate offline plots from persisted benchmark runs.",
    )
    parser.add_argument("--results-dir", default=str(PROJECT_ROOT / "results"))
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--benchmark", action="append", dest="benchmarks")
    parser.add_argument("--policy-mode", action="append", dest="policy_modes")
    parser.add_argument("--split", action="append", dest="splits")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_dir = _ensure_output_dir(args.output_dir)
    aggregation = aggregate_persisted_runs(
        results_dir=args.results_dir,
        benchmarks=args.benchmarks,
        policy_modes=args.policy_modes,
    )
    if args.splits:
        split_filter = {s.lower() for s in args.splits}
        aggregation["runs"] = [
            r for r in aggregation.get("runs", [])
            if (r.get("experiment_split") or "").lower() in split_filter
        ]
        aggregation["grouped_summary"] = [
            g for g in aggregation.get("grouped_summary", [])
            if (g.get("experiment_split") or "").lower() in split_filter
        ]
    manifest = generate_plots(aggregation, str(output_dir))

    print("Plot generation completed")
    print(f"  output_dir: {output_dir}")
    print(f"  runs: {manifest['run_count']}")
    print(f"  plots: {manifest['plot_count']}")
    for p in manifest["plots"]:
        print(f"  -> {Path(p).name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
