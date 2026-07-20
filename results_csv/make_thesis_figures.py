#!/usr/bin/env python
"""Regenerate the thesis figures in Pedro_MSc/plots/ from the curated CSVs.

Single source of truth for the figures included in Chapter 7 and Appendix 3 of
the thesis. Every figure uses one shared style: one policy/algorithm color
palette, one global font size, and 95% binomial confidence intervals on all bar
charts (error bars drawn above the bars). Run from anywhere:

    python results_csv/make_thesis_figures.py
"""
import csv, math, json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np

CSV_DIR = Path(__file__).resolve().parent
TESE = CSV_DIR.parent.parent                     # .../Tese
OUT = TESE / "Pedro_MSc" / "plots"
OUT.mkdir(parents=True, exist_ok=True)

# ---- one global style (no per-figure font overrides) ----
plt.rcParams.update({
    "font.family": "serif", "font.size": 12,
    "axes.titlesize": 14, "axes.titleweight": "bold", "axes.labelsize": 13,
    "xtick.labelsize": 11, "ytick.labelsize": 11, "legend.fontsize": 11,
    "legend.framealpha": 0.9, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "axes.axisbelow": True, "grid.alpha": 0.3, "grid.linestyle": "--",
})

POLICY_COLOR = {"random": "#FF9800", "react": "#4CAF50", "ucb1": "#1565C0",
                "thompson": "#2E7D32", "kl-ucb": "#E65100", "exp3": "#6A1B9A"}
POLICY_LABEL = {"random": "Random", "react": "ReAct", "ucb1": "UCB1",
                "thompson": "Thompson", "kl-ucb": "KL-UCB", "exp3": "EXP3"}
POLICY_ORDER = ["random", "react", "ucb1", "thompson", "kl-ucb", "exp3"]
TACTIC_NAME = {"legacy_injection": "Prompt Injection", "legacy_semantic": "Semantic Framing",
               "legacy_output": "Output Manipulation", "taxonomy_appeal_to_authority": "Appeal to Authority",
               "legacy_cot": "CoT Poisoning", "taxonomy_crowding": "Crowding", "taxonomy_roleplay": "Roleplay",
               "taxonomy_recursion_crescendo": "Recursion Crescendo",
               "taxonomy_formatting_smuggling": "Formatting Smuggling"}
FAMILY_COLOR = {"legacy_injection": "#E53935", "taxonomy_formatting_smuggling": "#E53935",
                "legacy_output": "#FB8C00", "taxonomy_crowding": "#FB8C00", "legacy_semantic": "#43A047",
                "taxonomy_roleplay": "#43A047", "legacy_cot": "#1E88E5",
                "taxonomy_recursion_crescendo": "#1E88E5", "taxonomy_appeal_to_authority": "#8E24AA"}
FAMILY_NAME = {"#E53935": "Structural Logic", "#FB8C00": "Obfuscation & Noise",
               "#43A047": "Narrative & Context", "#1E88E5": "Strategy & Pacing",
               "#8E24AA": "Pressure & Persuasion"}
JUDGE_ORDER = ["qwen", "deepseek", "codellama", "starcoder"]
JUDGE_LABEL = {"qwen": "Qwen2.5", "deepseek": "DeepSeek", "codellama": "CodeLlama", "starcoder": "StarCoder2"}
ALGO_COLOR = {"ucb1": "#1565C0", "thompson": "#2E7D32", "klucb": "#E65100", "exp3": "#6A1B9A"}
ALGO_DISP = {"ucb1": "UCB1", "thompson": "Thompson", "klucb": "KL-UCB", "exp3": "EXP3"}


def rd(name):
    return list(csv.DictReader(open(CSV_DIR / name, encoding="utf-8")))

def f(x):
    try: return float(x)
    except (TypeError, ValueError): return None

def ci95(p_pct, n):
    p = p_pct / 100.0
    return 1.96 * math.sqrt(p * (1 - p) / n) * 100.0

def save(fig, name):
    fig.savefig(OUT / name, dpi=300, bbox_inches="tight")
    plt.close(fig)


ASR = rd("asr_by_policy_judge_corpus.csv")
def asr_sel(judge, corpus):
    return {r["policy"]: r for r in ASR if r["judge_tag"] == judge and r["corpus"] == corpus}


def fig_policy_comparison():
    sel = asr_sel("qwen", "adversarial-code-buggy")
    x = list(range(len(POLICY_ORDER)))
    h = [f(sel[p]["asr_1shot"]) for p in POLICY_ORDER]
    ci = [ci95(hh, int(sel[p]["n"])) for hh, p in zip(h, POLICY_ORDER)]
    col = [POLICY_COLOR[p] for p in POLICY_ORDER]
    fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)
    ax.bar(x, h, width=0.62, color=col, alpha=0.9, edgecolor="white", linewidth=0.6, zorder=2)
    ax.errorbar(x, h, yerr=ci, fmt="none", ecolor="#222", elinewidth=1.5, capsize=5, capthick=1.5, zorder=6)
    for xi, hh, cc in zip(x, h, ci):
        ax.text(xi, hh + cc + 1.2, f"{hh:.1f}", ha="center", va="bottom", fontweight="bold")
    ax.set_xticks(x); ax.set_xticklabels([POLICY_LABEL[p] for p in POLICY_ORDER])
    ax.set_ylabel("1-shot Attack Success Rate (%)"); ax.set_ylim(0, 85)
    ax.set_title("Policy Comparison — adversarial-code-buggy (Qwen judge, test)")
    ax.grid(axis="x", visible=False)
    save(fig, "policy_comparison_test.png")


def fig_cumulative(corpus, name):
    sel = asr_sel("qwen", corpus); xs = [1, 3, 6]
    fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)
    for p in POLICY_ORDER:
        if p not in sel: continue
        ys = [f(sel[p]["asr_1shot"]), f(sel[p]["asr_3shot"]), f(sel[p]["asr_6shot"])]
        n = int(sel[p]["n"]); cis = [ci95(y, n) for y in ys]
        ax.errorbar(xs, ys, yerr=cis, marker="o", markersize=6, linewidth=2, capsize=4,
                    color=POLICY_COLOR[p], label=POLICY_LABEL[p], zorder=3)
    ax.set_xticks(xs); ax.set_xlabel("Attempt budget"); ax.set_ylabel("Attack Success Rate (%)")
    ax.set_ylim(0, 105)
    disp = "adversarial-code-buggy" if "adver" in corpus else "cubert_wbo"
    ax.set_title(f"Cumulative ASR by Budget — {disp} (Qwen judge, test)")
    ax.legend(loc="lower right", ncol=2)
    save(fig, name)


def fig_attacker_vs_judges():
    fig, ax = plt.subplots(figsize=(11, 5.5), constrained_layout=True)
    ns = len(POLICY_ORDER); w = 0.8 / ns
    for si, p in enumerate(POLICY_ORDER):
        for gi, jt in enumerate(JUDGE_ORDER):
            sel = asr_sel(jt, "adversarial-code-buggy")
            if p not in sel: continue
            v = f(sel[p]["asr_1shot"])
            if v is None: continue
            c = ci95(v, int(sel[p]["n"])); xo = gi + (si - (ns - 1) / 2) * w
            ax.bar(xo, v, width=w, color=POLICY_COLOR[p], alpha=0.9, edgecolor="white",
                   linewidth=0.5, zorder=2, label=POLICY_LABEL[p] if gi == 0 else None)
            ax.errorbar(xo, v, yerr=c, fmt="none", ecolor="#222", elinewidth=1.2, capsize=3, capthick=1.2, zorder=6)
    ax.set_xticks(range(len(JUDGE_ORDER))); ax.set_xticklabels([JUDGE_LABEL[j] for j in JUDGE_ORDER])
    ax.set_ylabel("1-shot Attack Success Rate (%)"); ax.set_xlabel("Judge model"); ax.set_ylim(0, 105)
    ax.set_title("Policy ASR Across Judge Models — adversarial-code-buggy (test)")
    ax.grid(axis="x", visible=False)
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), ncol=1)
    save(fig, "attacker_vs_judges_adversarial_code_buggy_llama3_1.png")


def fig_algorithm_comparison():
    algos = ["ucb1", "thompson", "kl-ucb", "exp3"]
    corpora = ["adversarial-code-buggy", "cubert_wbo"]
    fig, ax = plt.subplots(figsize=(9, 5.5), constrained_layout=True)
    ns = len(algos); w = 0.8 / ns
    for si, alg in enumerate(algos):
        for gi, ck in enumerate(corpora):
            sel = asr_sel("qwen", ck); v = f(sel[alg]["asr_1shot"]); c = ci95(v, int(sel[alg]["n"]))
            xo = gi + (si - (ns - 1) / 2) * w
            ax.bar(xo, v, width=w, color=POLICY_COLOR[alg], alpha=0.9, edgecolor="white",
                   linewidth=0.5, zorder=2, label=POLICY_LABEL[alg] if gi == 0 else None)
            ax.errorbar(xo, v, yerr=c, fmt="none", ecolor="#222", elinewidth=1.3, capsize=4, capthick=1.3, zorder=6)
            ax.text(xo, v + c + 1.0, f"{v:.1f}", ha="center", va="bottom", fontsize=9)
    ax.set_xticks(range(len(corpora))); ax.set_xticklabels(corpora)
    ax.set_ylabel("1-shot Attack Success Rate (%)"); ax.set_ylim(0, 85)
    ax.set_title("Bandit Algorithm Comparison — Qwen judge (test, frozen weights)")
    ax.grid(axis="x", visible=False)
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), title="Algorithm", ncol=1)
    save(fig, "algorithm_comparison_test.png")


def fig_tactic_win_rate():
    rq2 = rd("rq2_tactic_effectiveness.csv")
    acb = {r["tactic"]: r for r in rq2 if r["judge_tag"] == "qwen" and r["corpus"] == "adversarial-code-buggy"}
    cub = {r["tactic"]: r for r in rq2 if r["judge_tag"] == "qwen" and r["corpus"] == "cubert_wbo"}
    order = sorted(acb, key=lambda t: f(acb[t]["tactic_asr"]) or 0, reverse=True)
    w = 0.4
    fig, ax = plt.subplots(figsize=(11, 5.5), constrained_layout=True)
    for i, t in enumerate(order):
        c = FAMILY_COLOR[t]
        va = f(acb[t]["tactic_asr"])
        ax.bar(i - w / 2, va, width=w, color=c, alpha=0.9, edgecolor="white", zorder=2)
        ax.errorbar(i - w / 2, va, yerr=ci95(va, int(acb[t]["n"])), fmt="none", ecolor="#333",
                    elinewidth=1, capsize=3, zorder=6)
        vc = f(cub[t]["tactic_asr"])
        if vc is not None:
            ax.bar(i + w / 2, vc, width=w, color=c, alpha=0.9, hatch="///", edgecolor="white", zorder=2)
            ax.errorbar(i + w / 2, vc, yerr=ci95(vc, int(cub[t]["n"])), fmt="none", ecolor="#333",
                        elinewidth=1, capsize=3, zorder=6)
    ax.set_xticks(range(len(order))); ax.set_xticklabels([TACTIC_NAME[t] for t in order], rotation=30, ha="right")
    ax.set_ylabel("Tactic-only Attack Success Rate (%)"); ax.set_ylim(0, 75)
    ax.set_title("Per-Tactic Effectiveness — Qwen judge (test)"); ax.grid(axis="x", visible=False)
    fam = [plt.Rectangle((0, 0), 1, 1, color=c) for c in FAMILY_NAME]
    leg1 = ax.legend(fam, list(FAMILY_NAME.values()), loc="upper right", title="Family")
    ax.add_artist(leg1)
    sty = [plt.Rectangle((0, 0), 1, 1, facecolor="#777", alpha=0.9),
           plt.Rectangle((0, 0), 1, 1, facecolor="#777", alpha=0.9, hatch="///")]
    ax.legend(sty, ["adversarial-code-buggy", "cubert_wbo"], loc="upper center")
    save(fig, "tactic_win_rate.png")


def fig_tactic_judge_heatmap():
    rq2 = rd("rq2_tactic_effectiveness.csv")
    acb = {r["tactic"]: r for r in rq2 if r["judge_tag"] == "qwen" and r["corpus"] == "adversarial-code-buggy"}
    tactics = sorted(acb, key=lambda t: f(acb[t]["tactic_asr"]) or 0, reverse=True)
    mat = np.full((len(tactics), len(JUDGE_ORDER)), np.nan)
    for ri, t in enumerate(tactics):
        for ci_, jt in enumerate(JUDGE_ORDER):
            row = next((r for r in rq2 if r["tactic"] == t and r["judge_tag"] == jt
                        and r["corpus"] == "adversarial-code-buggy"), None)
            if row and f(row["tactic_asr"]) is not None:
                mat[ri, ci_] = f(row["tactic_asr"])
    fig, ax = plt.subplots(figsize=(7, 6.5), constrained_layout=True)
    cmap = mcolors.LinearSegmentedColormap.from_list("asr", ["#F5F5F5", "#FFB74D", "#B71C1C"])
    im = ax.imshow(np.ma.masked_invalid(mat), cmap=cmap, vmin=0, vmax=75, aspect="auto")
    fig.colorbar(im, ax=ax, label="Tactic-only ASR (%)")
    ax.set_xticks(range(len(JUDGE_ORDER))); ax.set_xticklabels([JUDGE_LABEL[j] for j in JUDGE_ORDER])
    ax.set_yticks(range(len(tactics))); ax.set_yticklabels([TACTIC_NAME[t] for t in tactics])
    ax.set_xlabel("Judge model"); ax.set_ylabel("Tactic")
    ax.set_title("Per-Tactic ASR by Judge — adversarial-code-buggy (test)"); ax.grid(False)
    for ri in range(len(tactics)):
        for ci_ in range(len(JUDGE_ORDER)):
            v = mat[ri, ci_]
            if not np.isnan(v):
                ax.text(ci_, ri, f"{v:.0f}", ha="center", va="center",
                        color="white" if v > 45 else "#222", fontweight="bold")
    save(fig, "tactic_judge_heatmap_adversarial_code_buggy.png")


def fig_entropy_bar():
    ent = rd("entropy_curves.csv"); algos = ["ucb1", "thompson", "klucb", "exp3"]; w = 0.4
    def ev(a, c):
        r = next((r for r in ent if r["algorithm"] == a and r["corpus"] == c), None)
        return float(r["normalized_entropy"]) if r else None
    fig, ax = plt.subplots(figsize=(8.5, 5), constrained_layout=True)
    for i, a in enumerate(algos):
        va, vc = ev(a, "adversarial-code-buggy"), ev(a, "cubert_wbo")
        ax.bar(i - w / 2, va, width=w, color=ALGO_COLOR[a], alpha=0.9, edgecolor="white", zorder=2)
        ax.text(i - w / 2, va + 0.02, f"{va:.2f}", ha="center", va="bottom", fontsize=9)
        ax.bar(i + w / 2, vc, width=w, color=ALGO_COLOR[a], alpha=0.9, hatch="///", edgecolor="white", zorder=2)
        ax.text(i + w / 2, vc + 0.02, f"{vc:.2f}", ha="center", va="bottom", fontsize=9)
    ax.axhline(1.0, color="#999", linestyle=":", linewidth=1.2)
    ax.text(len(algos) - 0.5, 1.01, "uniform", ha="right", va="bottom", fontsize=9, color="#666")
    ax.set_xticks(range(len(algos))); ax.set_xticklabels([ALGO_DISP[a] for a in algos])
    ax.set_ylabel("Normalized arm entropy"); ax.set_ylim(0, 1.12)
    ax.set_title("Post-Training Arm Entropy by Algorithm"); ax.grid(axis="x", visible=False)
    sty = [plt.Rectangle((0, 0), 1, 1, facecolor="#777", alpha=0.9),
           plt.Rectangle((0, 0), 1, 1, facecolor="#777", alpha=0.9, hatch="///")]
    ax.legend(sty, ["adversarial-code-buggy", "cubert_wbo"], loc="upper left")
    save(fig, "entropy_bar.png")


def fig_training_convergence():
    data = json.load(open(CSV_DIR / "training_analysis.json", encoding="utf-8"))
    fig, ax = plt.subplots(figsize=(8.5, 5), constrained_layout=True)
    for alg in ["ucb1", "thompson", "klucb", "exp3"]:
        rec = next((r for r in data if r["judge"] == "qwen"
                    and r["corpus"] == "adversarial_code_buggy" and r["algo"] == alg), None)
        if not rec: continue
        traj = rec["entropy_trajectory_norm"]
        ax.plot([p[0] for p in traj], [p[1] for p in traj], marker="o", markersize=6,
                linewidth=2, color=ALGO_COLOR[alg], label=ALGO_DISP[alg], zorder=3)
    ax.set_xlabel("Cumulative tactic pulls"); ax.set_ylabel("Normalized arm entropy"); ax.set_ylim(0, 1.05)
    ax.set_title("Bandit Convergence During Training — adversarial-code-buggy (Qwen judge)")
    ax.legend(loc="lower left", title="Algorithm", ncol=2)
    save(fig, "training_convergence.png")


if __name__ == "__main__":
    fig_policy_comparison()
    fig_cumulative("adversarial-code-buggy", "cumulative_asr_curve_adversarial_code_buggy.png")
    fig_cumulative("cubert_wbo", "cumulative_asr_curve_cubert_wbo.png")
    fig_attacker_vs_judges()
    fig_algorithm_comparison()
    fig_tactic_win_rate()
    fig_tactic_judge_heatmap()
    fig_entropy_bar()
    fig_training_convergence()
    print(f"Wrote 9 figures to {OUT}")
