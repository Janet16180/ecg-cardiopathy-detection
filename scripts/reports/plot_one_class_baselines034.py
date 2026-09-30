"""Plot the Experiment 034 figure: each one-class method minus Mahalanobis at SPH and PTB-XL development."""

import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
RESULT = ROOT / "outputs/experiment034_one_class_baselines_v1/result.json"
FIGURES = ROOT / "docs/figures/experiment-034"
SERIES = ("#2a78d6", "#eb6834", "#1baf7a")
MARKERS = ("o", "s", "^")
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
ENCODERS = {"xecg": "xECG (primary)", "jepa": "ECG-JEPA", "cpc": "CPC"}
PANELS = {"sph": "SPH, unseen hospital (primary)",
          "development": "PTB-XL development (dotted: the -0.01 margin of the rule)"}
OFFSET = 0.22


def style(axis: plt.Axes) -> None:
    """
    Apply the recessive grid and axis style.

    Parameters
    ----------
    axis : plt.Axes
        Axis to style.
    """
    axis.grid(axis="x", color=GRID, linewidth=0.8)
    axis.set_axisbelow(True)
    for side in ("top", "right", "left"):
        axis.spines[side].set_visible(False)
    axis.spines["bottom"].set_color(MUTED)
    axis.tick_params(colors=MUTED, labelcolor=INK, left=False)


def plot(result: dict[str, Any]) -> None:
    """
    Write the two-panel dot-and-interval figure.

    Parameters
    ----------
    result : dict[str, Any]
        Experiment 034 ``result.json``.
    """
    methods = [method for method in result["methods"] if method != "mahalanobis"]
    figure, axes = plt.subplots(1, 2, figsize=(12, 5.2), sharey=True)
    for axis, (set_name, title) in zip(axes, PANELS.items(), strict=True):
        for index, (encoder, label) in enumerate(ENCODERS.items()):
            contrasts = result["encoders"][encoder]["evaluations"][set_name]["contrasts"]
            for row, method in enumerate(methods):
                values = contrasts.get(f"{method}_minus_mahalanobis")
                if values is None:
                    continue
                y = row + (index - 1) * OFFSET
                axis.plot([values["ci_low"], values["ci_high"]], [y, y], color=SERIES[index], linewidth=2)
                axis.plot(values["difference"], y, marker=MARKERS[index], color=SERIES[index], markersize=8,
                          markeredgecolor="white", markeredgewidth=1,
                          label=label if row == 0 else None, linestyle="none")
        axis.axvline(0, color=MUTED, linewidth=1)
        if set_name == "development":
            axis.axvline(-0.01, color=MUTED, linewidth=1, linestyle=":")
        axis.set_title(title, color=INK, fontsize=10, loc="left")
        axis.set_xlabel("AUROC minus Mahalanobis (95% patient-bootstrap interval)", color=INK)
        style(axis)
    axes[0].set_yticks(range(len(methods)), methods)
    axes[0].invert_yaxis()
    axes[0].legend(frameon=False, fontsize=8, loc="lower left")
    figure.suptitle("One-class scores on the same frozen embeddings and 10,846 normals as 026b", color=INK,
                    x=0.01, ha="left")
    figure.tight_layout()
    figure.savefig(FIGURES / "one_class_baselines.png", dpi=150)
    plt.close(figure)


def main() -> None:
    """Write the figure from the saved result."""
    FIGURES.mkdir(parents=True, exist_ok=True)
    plot(json.loads(RESULT.read_text()))


if __name__ == "__main__":
    main()
