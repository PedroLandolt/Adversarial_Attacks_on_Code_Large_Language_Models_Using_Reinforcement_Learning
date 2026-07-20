#!/usr/bin/env python
"""
thesis_plots.py
Generate all thesis figures directly from results_csv/*.csv.

Produces figures referenced in:
  Pedro_MSc/body/chapter7_Results_and_Discussion.tex
  Pedro_MSc/backmatter/appendix3.tex

Usage (from Adversarial_Attacks/ directory):
  python thesis_plots.py [--output-dir PATH]
  Default output: ../Pedro_MSc/plots
"""
from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
CSV_DIR = SCRIPT_DIR / "results_csv"
DEFAULT_OUT = SCRIPT_DIR.parent / "Pedro_MSc" / "plots"

# ── Global style ──────────────────────────────────────────────────────────────
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
})

# ── Display constants ─────────────────────────────────────────────────────────
POLICY_ORDER = ["random", "react", "ucb1", "thompson", "kl-ucb", "exp3"]
BANDIT_ORDER = ["ucb1", "thompson", "kl-ucb", "exp3"]
JUDGE_ORDER  = ["qwen", "deepseek", "codellama", "starcoder"]
CORPUS_ORDER = ["adversarial-code-buggy", "cubert_wbo"]

POLICY_LABEL = {
    "random": "Random", "react": "ReAct", "ucb1": "UCB1",
    "thompson": "Thompson", "kl-ucb": "KL-UCB", "exp3": "EXP3",
}
POLICY_COLOR = {
    "Random":   "#FF9800", "ReAct":    "#4CAF50", "UCB1":     "#1565C0",
    "Thompson": "#2E7D32", "KL-UCB":   "#E65100", "EXP3":     "#6A1B9A",
}
JUDGE_LABEL = {
    "qwen":      "Qwen2.5-Coder 7B",
    "deepseek":  "DeepSeek-Coder 6.7B",
    "codellama": "CodeLlama 7B",
    "starcoder": "StarCoder2 7B",
}
CORPUS_LABEL = {
    "adversarial-code-buggy": "ACB",
    "cubert_wbo":             "CuBERT-WBO",
}
ALGO_LABEL = {"ucb1": "UCB1", "thompson": "Thompson", "kl-ucb": "KL-UCB", "exp3": "EXP3"}
ALGO_COLOR = {
    "ucb1": "#1565C0", "thompson": "#2E7D32", "kl-ucb": "#E65100", "exp3": "#6A1B9A",
}
TACTIC_ORDER = [
    "legacy_injection", "legacy_output", "legacy_semantic", "legacy_cot",
    "taxonomy_roleplay", "taxonomy_appeal_to_authority",
    "taxonomy_formatting_smuggling", "taxonomy_recursion_crescendo",
    "taxonomy_crowding",
]
TACTIC_DISPLAY = {
    "legacy_injection":              "prompt injection",
    "legacy_output":                 "output manipulation",
    "legacy_semantic":               "semantic framing",
    "legacy_cot":                    "CoT poisoning",
    "taxonomy_roleplay":             "roleplay",
    "taxonomy_appeal_to_authority":  "appeal to authority",
    "taxonomy_formatting_smuggling": "formatting smuggling",
    "taxonomy_recursion_crescendo":  "recursion crescendo",
    "taxonomy_crowding":             "crowding",
}
TACTIC_FAMILY_COLOR = {
    "injection":           "#E53935", "formatting_smuggling": "#E53935",
    "output":              "#FB8C00", "crowding":             "#FB8C00",
    "semantic":            "#43A047", "roleplay":             "#43A047",
    "cot":                 "#1E88E5", "recursion_crescendo":  "#1E88E5",
    "appeal_to_authority": "#8E24AA",
}


def _binomial_ci(p: float, n: int) -> float:
    """95% Wald binomial CI half-width in percentage points."""
    p_frac = p / 100.0
    return 1.96 * math.sqrt(max(p_frac * (1.0 - p_frac) / max(n, 1), 0.0)) * 100.0


# ── CSV readers ───────────────────────────────────────────────────────────────
def _read_asr() -> list[dict]:
    rows = []
    with open(CSV_DIR / "asr_by_policy_judge_corpus.csv", newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            rows.append({
                "judge_tag": r["judge_tag"],
                "policy":    r["policy"],
                "corpus":    r["corpus"],
                "n":         int(r["n"]),
                "baseline":  float(r["baseline_asr"]),
                "s1":        float(r["asr_1shot"]),
                "s3":        float(r["asr_3shot"]),
                "s6":        float(r["asr_6shot"]),
            })
    return rows


def _read_rq2() -> list[dict]:
    rows = []
    with open(CSV_DIR / "rq2_tactic_effectiveness.csv", newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if not r.get("tactic_asr"):
                continue
            rows.append({
                "tactic":     r["tactic"],
                "judge_tag":  r["judge_tag"],
                "corpus":     r["corpus"],
                "n":          int(r["n"]) if r.get("n") else 0,
                "tactic_asr": float(r["tactic_asr"]),
            })
    return rows


def _read_entropy() -> list[dict]:
    rows = []
    with open(CSV_DIR / "entropy_curves.csv", newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            rows.append({
                "algorithm":          r["algorithm"],
                "corpus":             r["corpus"],
                "normalized_entropy": float(r["normalized_entropy"]),
            })
    return rows


# ── Save helper ───────────────────────────────────────────────────────────────
def _save(fig: plt.Figure, path: Path) -> str:
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  {path.name}")
    return str(path)


def _bar_label(ax: plt.Axes, bar, text: str, err: float, y_max: float, fs: int = 8) -> None:
    """Place text above the error-bar cap; fall back to inside when space is tight."""
    h = bar.get_height()
    y_above = h + err + 1.5
    if y_above + 2 > y_max:
        ax.text(bar.get_x() + bar.get_width() / 2, max(h - 3.5, 2),
                text, ha="center", va="top", fontsize=fs, fontweight="bold", color="white")
    else:
        ax.text(bar.get_x() + bar.get_width() / 2, y_above,
                text, ha="center", va="bottom", fontsize=fs, fontweight="bold")


# ══════════════════════════════════════════════════════════════════════════════
# Figure 1 — Policy comparison  (ACB, primary judge)
# Ch7 fig:rq1_acb_bar
# ══════════════════════════════════════════════════════════════════════════════
def plot_policy_comparison(rows: list[dict], out: Path) -> str:
    asr = {r["policy"]: r
           for r in rows if r["judge_tag"] == "qwen" and r["corpus"] == "adversarial-code-buggy"}
    if not asr:
        return ""

    n       = asr["random"]["n"]
    heights = [asr[p]["s1"] for p in POLICY_ORDER]
    errs    = [_binomial_ci(h, n) for h in heights]
    labels  = [POLICY_LABEL[p] for p in POLICY_ORDER]
    colors  = [POLICY_COLOR[l] for l in labels]

    y_max = max(h + e for h, e in zip(heights, errs)) + 12

    fig, ax = plt.subplots(figsize=(9, 6), constrained_layout=True)
    bars = ax.bar(range(len(labels)), heights, color=colors, alpha=0.88,
                  yerr=errs, capsize=5, error_kw={"elinewidth": 1.6, "ecolor": "#333"})
    for bar, h, e in zip(bars, heights, errs):
        _bar_label(ax, bar, f"{h:.1f}%", e, y_max)

    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_ylabel("1-Shot Attack Success Rate (%)")
    ax.set_ylim(0, y_max)
    ax.set_title(
        "Policy Comparison — Test Split\n"
        "adversarial-code-buggy  |  Qwen2.5-Coder 7B judge  |  n = 148"
    )
    return _save(fig, out)


# ══════════════════════════════════════════════════════════════════════════════
# Figures 2 / 3 — Cumulative ASR vs iteration budget
# Ch7 fig:rq3_iteration, fig:rq3_iteration_cubert
# ══════════════════════════════════════════════════════════════════════════════
def plot_cumulative_asr(rows: list[dict], corpus: str, out: Path) -> str:
    data = {r["policy"]: r
            for r in rows if r["corpus"] == corpus and r["judge_tag"] == "qwen"}
    if not data:
        return ""

    xs = [0, 1, 3, 6]
    fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)

    for policy in POLICY_ORDER:
        if policy not in data:
            continue
        r   = data[policy]
        ys  = [r["baseline"], r["s1"], r["s3"], r["s6"]]
        lbl = POLICY_LABEL[policy]
        ax.plot(xs, ys,
                marker="s" if policy == "react" else "o",
                linewidth=2.2,
                linestyle="--" if policy == "react" else "-",
                color=POLICY_COLOR[lbl],
                label=lbl)

    ax.axvline(1, color="#888", linestyle=":", linewidth=1.2)
    ax.axvline(3, color="#bbb", linestyle=":", linewidth=1.0)
    ax.text(1.08, 3, "1-shot", color="#888", fontsize=7.5, va="bottom")
    ax.text(3.08, 3, "3-shot", color="#bbb", fontsize=7.5, va="bottom")
    ax.set_xlabel("Iteration budget (max tactic attempts)")
    ax.set_ylabel("Cumulative ASR (%)")
    ax.set_title(
        f"Cumulative Attack Success Rate vs. Iteration Budget — "
        f"{CORPUS_LABEL.get(corpus, corpus)}\n"
        f"Qwen2.5-Coder 7B judge, test split  |  n = {data['random']['n']}"
    )
    ax.set_xticks(xs)
    ax.set_xlim(-0.3, 6.5)
    ax.set_ylim(0, 103)
    ax.legend(loc="lower right", fontsize=9, ncol=2)
    return _save(fig, out)


# ══════════════════════════════════════════════════════════════════════════════
# Figure 4 — 1-shot ASR per judge model
# Ch7 fig:rq4_heatmap
# ══════════════════════════════════════════════════════════════════════════════
def plot_attacker_vs_judges(rows: list[dict], out: Path) -> str:
    corpus  = "adversarial-code-buggy"
    lookup  = {(r["judge_tag"], r["policy"]): r["s1"]
               for r in rows if r["corpus"] == corpus}
    judges  = [j for j in JUDGE_ORDER
               if any(r["judge_tag"] == j for r in rows if r["corpus"] == corpus)]

    n_jud = len(judges)
    n_pol = len(POLICY_ORDER)
    bar_w = 0.76 / n_pol
    x     = list(range(n_jud))

    all_h = [v for v in lookup.values()]
    y_max = (max(all_h) if all_h else 100) + 12

    fig, ax = plt.subplots(figsize=(max(10, n_jud * n_pol * 0.85), 6), constrained_layout=True)

    for pi, policy in enumerate(POLICY_ORDER):
        lbl     = POLICY_LABEL[policy]
        heights = [lookup.get((j, policy), 0.0) for j in judges]
        offsets = [xi + (pi - (n_pol - 1) / 2) * bar_w for xi in x]
        bars    = ax.bar(offsets, heights, width=bar_w, label=lbl,
                         color=POLICY_COLOR[lbl], alpha=0.88)
        for bar, h in zip(bars, heights):
            if h > 1.0:
                _bar_label(ax, bar, f"{h:.1f}%", 0, y_max, fs=7)

    if "starcoder" in judges:
        ax.annotate(
            "bandit not run",
            xy=(judges.index("starcoder"), 8),
            ha="center", va="bottom", fontsize=7.5, color="#666", style="italic",
        )

    ax.set_xticks(x)
    ax.set_xticklabels([JUDGE_LABEL[j] for j in judges], rotation=12, ha="right", fontsize=9)
    ax.set_ylabel("1-Shot Attack Success Rate (%)")
    ax.set_ylim(0, y_max)
    ax.set_title(
        "1-Shot ASR per Judge Model — adversarial-code-buggy, Test split\n"
        "Attacker: Llama 3.1 8B (Generator)"
    )
    ax.legend(loc="upper right", ncol=2, fontsize=8)
    return _save(fig, out)


# ══════════════════════════════════════════════════════════════════════════════
# Figure 5 — Bandit algorithm comparison (primary judge, both corpora)
# Ch7 fig:rq5_algorithm_bar
# ══════════════════════════════════════════════════════════════════════════════
def plot_algorithm_comparison(rows: list[dict], out: Path) -> str:
    lookup = {(r["policy"], r["corpus"]): r
              for r in rows if r["judge_tag"] == "qwen" and r["policy"] in BANDIT_ORDER}

    n_alg  = len(BANDIT_ORDER)
    bar_w  = 0.76 / n_alg
    x      = list(range(len(CORPUS_ORDER)))

    all_h_e = [(lookup[(p, c)]["s1"],
                _binomial_ci(lookup[(p, c)]["s1"], lookup[(p, c)]["n"]))
               for p in BANDIT_ORDER for c in CORPUS_ORDER if (p, c) in lookup]

    y_min = max(0.0, min(h for h, _ in all_h_e) - 10)
    y_max = max(h + e for h, e in all_h_e) + 12

    fig, ax = plt.subplots(figsize=(8, 6), constrained_layout=True)

    for ai, policy in enumerate(BANDIT_ORDER):
        lbl     = ALGO_LABEL[policy]
        heights = []
        errs    = []
        for corpus in CORPUS_ORDER:
            r = lookup.get((policy, corpus))
            h = r["s1"] if r else 0.0
            e = _binomial_ci(h, r["n"]) if r else 0.0
            heights.append(h)
            errs.append(e)
        offsets = [xi + (ai - (n_alg - 1) / 2) * bar_w for xi in x]
        bars = ax.bar(offsets, heights, width=bar_w, label=lbl,
                      color=ALGO_COLOR[policy], alpha=0.88,
                      yerr=errs, capsize=4, error_kw={"elinewidth": 1.5, "ecolor": "#333"})
        for bar, h, e in zip(bars, heights, errs):
            if h > 0.5:
                _bar_label(ax, bar, f"{h:.1f}%", e, y_max, fs=8)

    ax.set_xticks(x)
    ax.set_xticklabels([CORPUS_LABEL[c] for c in CORPUS_ORDER], fontsize=12)
    ax.set_ylabel("1-Shot Attack Success Rate (%)")
    ax.set_ylim(y_min, y_max)
    ax.set_title(
        "Bandit Algorithm Comparison — Test Split\n"
        "Qwen2.5-Coder 7B judge  |  frozen weights after training"
    )
    ax.legend(loc="lower right", title="Algorithm")
    return _save(fig, out)


# ══════════════════════════════════════════════════════════════════════════════
# Appendix C Figure 1 — Per-tactic ASR (forced-tactic, Qwen, ACB + CuBERT-WBO)
# ══════════════════════════════════════════════════════════════════════════════
def plot_tactic_win_rate(rows: list[dict], out: Path) -> str:
    vals: dict[tuple, float] = {
        (r["tactic"], r["corpus"]): r["tactic_asr"]
        for r in rows if r["judge_tag"] == "qwen"
    }

    acb_vals     = {t: vals.get((t, "adversarial-code-buggy"), 0.0) for t in TACTIC_ORDER}
    tactics_asc  = sorted(TACTIC_ORDER, key=lambda t: acb_vals[t])  # lowest at bottom
    labels       = [TACTIC_DISPLAY[t] for t in tactics_asc]
    family_key   = [t.replace("legacy_", "").replace("taxonomy_", "") for t in tactics_asc]
    colors       = [TACTIC_FAMILY_COLOR.get(k, "#9E9E9E") for k in family_key]

    n_tac  = len(tactics_asc)
    y_pos  = list(range(n_tac))
    bar_h  = 0.38

    all_vals = [vals.get((t, c), 0.0) for t in tactics_asc for c in CORPUS_ORDER]
    x_max   = max(all_vals, default=0) + 22

    HATCH = {"adversarial-code-buggy": "", "cubert_wbo": "///"}
    ALPHA = {"adversarial-code-buggy": 0.88, "cubert_wbo": 0.60}

    fig, ax = plt.subplots(figsize=(10, max(5, n_tac * 0.72)), constrained_layout=True)

    for bi, corpus in enumerate(CORPUS_ORDER):
        heights = [vals.get((t, corpus), 0.0) for t in tactics_asc]
        offsets = [yi + (bi - 0.5) * bar_h for yi in y_pos]
        ax.barh(offsets, heights, height=bar_h,
                color=colors, alpha=ALPHA[corpus], hatch=HATCH[corpus],
                label=CORPUS_LABEL[corpus], edgecolor="white")

    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel("Tactic Attack Success Rate (%)  —  Qwen2.5-Coder 7B judge")
    ax.set_title("Per-Tactic ASR — Forced-Tactic Evaluation, Test Split")
    ax.set_xlim(0, x_max)
    ax.legend(loc="lower right", fontsize=9)
    return _save(fig, out)


# ══════════════════════════════════════════════════════════════════════════════
# Appendix C Figure 2 — Tactic × judge heatmap (ACB, forced-tactic)
# ══════════════════════════════════════════════════════════════════════════════
def plot_tactic_judge_heatmap(rows: list[dict], out: Path) -> str:
    corpus = "adversarial-code-buggy"
    vals: dict[tuple, float] = {
        (r["tactic"], r["judge_tag"]): r["tactic_asr"]
        for r in rows if r["corpus"] == corpus
    }
    judges = [j for j in JUDGE_ORDER
              if any(r["judge_tag"] == j and r["corpus"] == corpus for r in rows)]
    tactic_labels = [TACTIC_DISPLAY[t] for t in TACTIC_ORDER]

    matrix = np.full((len(TACTIC_ORDER), len(judges)), np.nan)
    for ri, tactic in enumerate(TACTIC_ORDER):
        for ci, judge in enumerate(judges):
            v = vals.get((tactic, judge))
            if v is not None:
                matrix[ri, ci] = v

    fig, ax = plt.subplots(
        figsize=(max(6, len(judges) * 2.1), max(4, len(TACTIC_ORDER) * 0.78)),
        constrained_layout=True,
    )
    cmap   = mcolors.LinearSegmentedColormap.from_list("asr", ["#1565C0", "#FFF176", "#B71C1C"])
    masked = np.ma.masked_invalid(matrix)
    im     = ax.imshow(masked, cmap=cmap, vmin=0, vmax=100, aspect="auto")
    plt.colorbar(im, ax=ax, label="Tactic ASR (%)")

    ax.set_xticks(range(len(judges)))
    ax.set_xticklabels([JUDGE_LABEL[j] for j in judges], rotation=20, ha="right", fontsize=9)
    ax.set_yticks(range(len(TACTIC_ORDER)))
    ax.set_yticklabels(tactic_labels, fontsize=9)
    ax.set_xlabel("Judge model")
    ax.set_ylabel("Tactic")
    ax.set_title(
        "Per-Tactic ASR per Judge — adversarial-code-buggy\n"
        "Forced-Tactic Evaluation, Test split"
    )

    for ri in range(len(TACTIC_ORDER)):
        for ci in range(len(judges)):
            val = matrix[ri, ci]
            if not np.isnan(val):
                tc = "white" if val > 60 or val < 20 else "black"
                ax.text(ci, ri, f"{val:.0f}%", ha="center", va="center",
                        fontsize=8, fontweight="bold", color=tc)
            else:
                ax.text(ci, ri, "—", ha="center", va="center", fontsize=8, color="#aaa")
    return _save(fig, out)


# ══════════════════════════════════════════════════════════════════════════════
# Appendix C Figure 3 — Post-training arm entropy (bar chart, single epoch)
# Referenced: Ch7 RQ5 arm-entropy subsection → Appendix app:bandit-arms
# ══════════════════════════════════════════════════════════════════════════════
def plot_entropy_bar(rows: list[dict], out: Path) -> str:
    ALGO_MAP = {"ucb1": "ucb1", "thompson": "thompson", "klucb": "kl-ucb", "exp3": "exp3"}
    vals: dict[tuple, float] = {
        (ALGO_MAP.get(r["algorithm"], r["algorithm"]), r["corpus"]): r["normalized_entropy"]
        for r in rows
    }

    n_alg  = len(BANDIT_ORDER)
    bar_w  = 0.38
    x      = list(range(n_alg))
    CORPUS_COLORS = {"adversarial-code-buggy": "#1565C0", "cubert_wbo": "#E65100"}

    fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)

    for ci, corpus in enumerate(CORPUS_ORDER):
        heights = [vals.get((alg, corpus), 0.0) for alg in BANDIT_ORDER]
        offsets = [xi + (ci - 0.5) * bar_w for xi in x]
        bars = ax.bar(offsets, heights, width=bar_w,
                      label=CORPUS_LABEL[corpus],
                      color=CORPUS_COLORS[corpus], alpha=0.85)
        for bar, h in zip(bars, heights):
            if h > 0.01:
                ax.text(bar.get_x() + bar.get_width() / 2, h + 0.013,
                        f"{h:.3f}", ha="center", va="bottom",
                        fontsize=8, fontweight="bold")

    ax.axhline(1.0, color="#aaa", linestyle=":", linewidth=1.5,
               label="Uniform entropy  (1 = all arms equally likely)")
    ax.set_xticks(x)
    ax.set_xticklabels([ALGO_LABEL[a] for a in BANDIT_ORDER], fontsize=11)
    ax.set_ylabel("Normalized arm entropy  (1 = uniform,  0 = one arm)")
    ax.set_ylim(0, 1.25)
    ax.set_title("Post-Training Arm Entropy by Algorithm")
    ax.legend(loc="upper right", fontsize=9)
    return _save(fig, out)


# ── Main ──────────────────────────────────────────────────────────────────────
def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate thesis figures from results_csv/*.csv"
    )
    parser.add_argument("--output-dir", default=str(DEFAULT_OUT),
                        help=f"Output directory (default: {DEFAULT_OUT})")
    args = parser.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    print(f"Output: {out}\n")

    asr  = _read_asr()
    rq2  = _read_rq2()
    ent  = _read_entropy()
    done: list[str] = []

    print("Ch7 figures:")
    r = plot_policy_comparison(asr, out / "policy_comparison_test.png")
    if r: done.append(r)

    for corpus in CORPUS_ORDER:
        fname = f"cumulative_asr_curve_{corpus.replace('-', '_')}.png"
        r = plot_cumulative_asr(asr, corpus, out / fname)
        if r: done.append(r)

    r = plot_attacker_vs_judges(asr, out / "attacker_vs_judges_adversarial_code_buggy_llama3_1.png")
    if r: done.append(r)

    r = plot_algorithm_comparison(asr, out / "algorithm_comparison_test.png")
    if r: done.append(r)

    print("\nAppendix C figures:")
    r = plot_tactic_win_rate(rq2, out / "tactic_win_rate.png")
    if r: done.append(r)

    r = plot_tactic_judge_heatmap(rq2, out / "tactic_judge_heatmap_adversarial_code_buggy.png")
    if r: done.append(r)

    r = plot_entropy_bar(ent, out / "entropy_bar.png")
    if r: done.append(r)

    print(f"\n{len(done)} figures generated.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
