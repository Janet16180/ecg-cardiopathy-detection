import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

out = sys.argv[1]
INK, MUTED, BLUE, ORANGE = "#1a1a19", "#6b6a63", "#2a78d6", "#eb6834"
plt.rcParams.update({"font.size": 9, "font.family": "DejaVu Sans"})


def bump(t, center, width, height):
    return height * np.exp(-0.5 * ((t - center) / width) ** 2)


def beat(t, start):
    return (bump(t, start + 0.10, 0.025, 0.15) + bump(t, start + 0.235, 0.008, -0.12)
            + bump(t, start + 0.26, 0.010, 1.0) + bump(t, start + 0.285, 0.009, -0.22)
            + bump(t, start + 0.50, 0.050, 0.30))


t = np.linspace(0, 1.75, 3500)
signal = beat(t, 0.10) + beat(t, 0.95)
fig, ax = plt.subplots(figsize=(7.4, 3.0))
ax.plot(t, signal, color=INK, lw=1.6)
for x, y, label in ((0.20, 0.22, "P wave"), (0.50, 0.80, "QRS complex"), (0.60, 0.37, "T wave")):
    ax.text(x, y, label, ha="center", va="bottom", color=BLUE, fontsize=9, fontweight="bold")
ax.text(0.345, 0.97, "R", color=MUTED, fontsize=8, ha="right")
ax.text(0.318, -0.20, "Q", color=MUTED, fontsize=8, ha="right")
ax.text(0.385, -0.25, "S", color=MUTED, fontsize=8, ha="center", va="top")


def span(x0, x1, y, label):
    ax.annotate("", (x0, y), (x1, y), arrowprops={"arrowstyle": "<->", "color": ORANGE, "lw": 1.2})
    ax.text((x0 + x1) / 2, y - 0.035, label, ha="center", va="top", color=INK, fontsize=8.5)
    for x in (x0, x1):
        ax.plot([x, x], [y - 0.03, 0.0 if y < 0 else y], color=ORANGE, lw=0.6, ls=":")


span(0.145, 0.335, -0.42, "PR interval")
span(0.335, 0.405, -0.62, "QRS duration")
span(0.335, 0.715, -0.82, "QT interval")
ax.annotate("", (0.36, 1.13), (1.21, 1.13), arrowprops={"arrowstyle": "<->", "color": ORANGE, "lw": 1.2})
ax.text(0.785, 1.16, "Time between beats (sets the heart rate)", ha="center", va="bottom", color=INK, fontsize=8.5)
ax.text(0.46, -0.05, "ST", color=MUTED, fontsize=8, ha="center", va="top")
ax.plot([1.5, 1.7], [-0.9, -0.9], color=MUTED, lw=1.2)
ax.text(1.6, -0.94, "0.2 s", ha="center", va="top", color=MUTED, fontsize=8)
ax.set(xlim=(0, 1.75), ylim=(-1.05, 1.32))
ax.axis("off")
fig.tight_layout(pad=0.2)
fig.savefig(f"{out}/fig_ecg_beat.png", dpi=300)

leads = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]
drops = np.array([[0.000128, -0.000165, -0.000021], [0.000515, 0.006108, 0.000003],
                  [-0.000160, 0.001237, -0.000001], [-0.000260, 0.001120, 0.003988],
                  [-0.000005, -0.000182, -0.000277], [0.000006, 0.003704, -0.000373],
                  [0.000033, -0.000887, 0.000171], [-0.000234, 0.000877, -0.001866],
                  [-0.000272, 0.002151, -0.000966], [0.000133, 0.000440, -0.000698],
                  [0.000129, 0.006358, 0.001526], [0.000095, 0.000358, 0.001194]])
fig, ax = plt.subplots(figsize=(7.4, 2.0))
limit = np.abs(drops).max()
image = ax.imshow(drops.T * 1000, cmap="RdBu_r", vmin=-limit * 1000, vmax=limit * 1000, aspect="auto")
ax.set_xticks(range(12), leads)
ax.set_yticks(range(3), ["P", "QRS", "ST-T"])
ax.tick_params(length=0, colors=INK)
for row in range(3):
    for col in range(12):
        value = drops[col, row] * 1000
        ax.text(col, row, f"{value + 0.0:.1f}".replace("-0.0", "0.0"), ha="center", va="center", fontsize=7.5,
                color="white" if abs(value) > 3.5 else INK)
for edge in ax.spines.values():
    edge.set_visible(False)
ax.set_xticks(np.arange(-0.5, 12), minor=True)
ax.set_yticks(np.arange(-0.5, 3), minor=True)
ax.grid(which="minor", color="white", lw=2)
ax.tick_params(which="minor", length=0)
bar = fig.colorbar(image, ax=ax, pad=0.02, fraction=0.035)
bar.set_label("Fall in score (×0.001)", fontsize=8, color=INK)
bar.outline.set_visible(False)
bar.ax.tick_params(labelsize=7.5, length=0)
fig.tight_layout(pad=0.3)
fig.savefig(f"{out}/fig_where.png", dpi=300)
