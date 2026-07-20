# -*- coding: utf-8 -*-
"""Regenerate the training-convergence figure in the presentation's brown palette,
with larger fonts for projection. Same data as the thesis figure
(training_analysis.json), so it stays faithful. Output goes to the deck scratchpad."""
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

CSV_DIR = Path(__file__).resolve().parent
OUT = Path(r"C:\Users\pedro\AppData\Local\Temp\claude\C--Users-pedro-Desktop-Pedro-Tese\37a6a5e9-895b-4eb3-a4e5-972724c1a430\scratchpad\convergence_brown.png")

data = json.load(open(CSV_DIR / "training_analysis.json", encoding="utf-8"))

# deck brown-coordinated palette + distinct line styles/markers
STYLE = {
    "thompson": dict(color="#C6791E", ls="-",  marker="o", lw=2.8, label="Thompson"),
    "klucb":    dict(color="#965004", ls="-",  marker="D", lw=3.8, label="KL-UCB"),
    "ucb1":     dict(color="#8A6E42", ls="--", marker="s", lw=2.8, label="UCB1"),
    "exp3":     dict(color="#A9905F", ls=":",  marker="^", lw=2.8, label="EXP3"),
}

plt.rcParams.update({"font.family": "sans-serif", "axes.spines.top": False,
                     "axes.spines.right": False, "axes.grid": True,
                     "axes.axisbelow": True, "grid.alpha": 0.3, "grid.linestyle": "--"})

fig, ax = plt.subplots(figsize=(9.2, 5.2), constrained_layout=True)
for alg in ["thompson", "klucb", "ucb1", "exp3"]:
    rec = next(r for r in data if r["judge"] == "qwen"
               and r["corpus"] == "adversarial_code_buggy" and r["algo"] == alg)
    traj = rec["entropy_trajectory_norm"]
    st = STYLE[alg]
    ax.plot([p[0] for p in traj], [p[1] for p in traj], markersize=8,
            markeredgecolor="white", markeredgewidth=0.8, zorder=4, **st)

ax.set_xlabel("Cumulative tactic pulls", fontsize=17)
ax.set_ylabel("Normalized arm entropy", fontsize=17)
ax.set_ylim(0, 1.05)
ax.tick_params(labelsize=14)
ax.axhline(1.0, color="#999", ls=":", lw=1.0)
ax.text(ax.get_xlim()[1], 1.01, "uniform", ha="right", va="bottom", fontsize=12, color="#777")
leg = ax.legend(loc="lower left", title="Algorithm", ncol=2, fontsize=15,
                title_fontsize=15, framealpha=0.95)
fig.savefig(OUT, dpi=200, facecolor="white", bbox_inches="tight")
plt.close(fig)
print("wrote", OUT)
