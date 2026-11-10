import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score

root, out = sys.argv[1], sys.argv[2]
z = np.load(f"{root}/outputs/avance5_final/sph_predictions.npz")
held = z["evaluation"]
y = z["target"][held]
budgets = pd.read_csv(f"{root}/reports/avance5_final/presupuestos_derivacion.csv")
INK, MUTED, GRID = "#1a1a19", "#6b6a63", "#e4e3dd"
series = [("hgb", "Boosting on measurements", "#1baf7a", 1.3),
          ("xecg", "Frozen xECG + logistic head", "#eb6834", 1.3),
          ("blend", "Blend (selected)", "#2a78d6", 2.2)]
plt.rcParams.update({"font.size": 9, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
                     "xtick.color": MUTED, "ytick.color": MUTED, "axes.spines.top": False,
                     "axes.spines.right": False, "font.family": "DejaVu Sans"})
fig, (a, b) = plt.subplots(1, 2, figsize=(9.2, 3.7), gridspec_kw={"width_ratios": [1, 1]})
for key, label, color, width in series:
    s = z[key][held]
    fpr, tpr, _ = roc_curve(y, s)
    auc = roc_auc_score(y, s)
    print(key, round(auc, 4))
    a.plot(fpr, tpr, color=color, lw=width, label=f"{label}, AUROC {auc:.3f}")
a.plot([0, 1], [0, 1], ":", color=MUTED, lw=1)
fp_rate = 1 - budgets.specificity
a.scatter(fp_rate, budgets.sensitivity, s=34, color="#2a78d6", edgecolor="white", linewidth=1.2, zorder=5)
for rate, sens, bud in zip(fp_rate, budgets.sensitivity, budgets.normal_budget):
    a.annotate(f"{bud * 100:g}%", (rate, sens), xytext=(7, -11), textcoords="offset points", color=INK, fontsize=8)
a.set(xlabel="Fraction of normal ECGs flagged", ylabel="Sensitivity", xlim=(-0.01, 1), ylim=(0, 1.01))
a.set_title("(a) ROC curves on the SPH evaluation ECGs", loc="left", fontsize=9.5, color=INK)
a.legend(frameon=False, loc="lower right", fontsize=8)
a.grid(color=GRID, lw=0.6)
x = budgets.normal_budget * 100
b.plot(x, budgets.sensitivity * 100, "o-", color="#2a78d6", lw=2, ms=6, mec="white", label="Sensitivity")
b.plot(x, fp_rate * 100, "s-", color="#eb6834", lw=2, ms=6, mec="white", label="Normal ECGs flagged (evaluation)")
b.plot([0, 20], [0, 20], ":", color=MUTED, lw=1, label="Budget (target)")
for xi, sens in zip(x, budgets.sensitivity * 100):
    b.annotate(f"{sens:.1f}", (xi, sens), xytext=(0, 7), textcoords="offset points", ha="center", color=INK, fontsize=8)
for xi, rate in zip(x, fp_rate * 100):
    b.annotate(f"{rate:.1f}", (xi, rate), xytext=(0, 7), textcoords="offset points", ha="center", color=INK, fontsize=8)
b.set(xlabel="Budget: calibration normals allowed above the threshold (%)", ylabel="Percent of evaluation ECGs in the class",
      xlim=(0, 21.5), ylim=(0, 100), xticks=[1, 2.5, 5, 10, 20], xticklabels=["1", "2.5", "5", "10", "20"])
b.set_title("(b) Operating points of the blend", loc="left", fontsize=9.5, color=INK)
b.legend(frameon=False, loc="center right", fontsize=8)
b.grid(color=GRID, lw=0.6)
fig.tight_layout()
fig.savefig(out, dpi=300)
