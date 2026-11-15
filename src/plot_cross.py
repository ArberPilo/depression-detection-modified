"""Figure for Task 6: F1 per test set, in-domain vs cross-domain model, with the one-word rule as reference."""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from config import RESULTS_DIR

D = os.path.join(RESULTS_DIR, "cross_dataset")
runs = pd.read_csv(os.path.join(D, "cross_all_runs.csv"))
rule = pd.read_csv(os.path.join(D, "keyword_rule.csv")).set_index("test_on")
g = runs.groupby(["test_on", "train_on"])["f1"].agg(["mean", "std"])

fig, ax = plt.subplots(figsize=(6.2, 3.6))
for i, tgt in enumerate(["Twitter", "Reddit"]):
    other = "Reddit" if tgt == "Twitter" else "Twitter"
    for j, (src, col, lab) in enumerate([(tgt, "#4f81bd", "model trained on the same dataset"),
                                          (other, "#c0504d", "model trained on the other dataset")]):
        m, s = g.loc[(tgt, src), "mean"], g.loc[(tgt, src), "std"]
        x = i * 3 + j
        ax.bar(x, m, yerr=s, capsize=4, color=col, width=0.8, label=lab if i == 0 else None)
        ax.text(x, m + s + 0.03, f"{m:.2f}", ha="center", fontsize=10)
    r = rule.loc[tgt, "f1"]
    ax.hlines(r, i * 3 - 0.5, i * 3 + 1.5, colors="k", linestyles="--", lw=1.2,
              label="one-word rule: contains 'depress'" if i == 0 else None)
    ax.text(i * 3 + 1.55, r, f"{r:.2f}", va="center", fontsize=9)
ax.set_xticks([0.5, 3.5]); ax.set_xticklabels(["tested on Twitter", "tested on Reddit"])
ax.set_ylim(0, 1.18); ax.set_ylabel("F1, depression-indicative class")
ax.set_title("Cross-dataset F1 (mean ± sd, 3 seeds)", fontsize=10)
ax.legend(fontsize=8, loc="lower center", ncol=1, bbox_to_anchor=(0.5, 0.02), framealpha=0.95)
fig.tight_layout(); fig.savefig(os.path.join(D, "f1_in_vs_cross.png"), dpi=150); plt.close(fig)
print("saved")
