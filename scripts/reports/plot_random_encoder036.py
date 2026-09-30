"""Plot the Experiment 036 figure: AUROC of each xECG arm with the normal reference and the readout."""

import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
RESULT = ROOT / "outputs/experiment036_random_encoder_v1/result.json"
FIGURES = ROOT / "docs/figures/experiment-036"
SERIES = {"mahalanobis": "#2a78d6", "readout": "#eb6834"}
MARKERS = {"mahalanobis": "o", "readout": "s"}
LABELS = {"mahalanobis": "Normal reference (Mahalanobis, no labels)", "readout": "Supervised readout (022b)"}
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
ARMS = {"pretrained": "pretrained", "random_36001": "random, seed 36001",
        "random_36002": "random, seed 36002", "random_36003": "random, seed 36003",
        "finetuned016": "fine-tuned (016)"}
PANELS = {"sph": "SPH, unseen hospital (primary)", "development": "PTB-XL development"}
OFFSET = 0.15


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
    Write the two-panel dot figure of AUROC by arm and score.

    Parameters
    ----------
    result : dict[str, Any]
        Experiment 036 ``result.json``.
    """
    arms = [arm for arm in ARMS if arm in result["arms"]]
    figure, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    for axis, (set_name, title) in zip(axes, PANELS.items(), strict=True):
        auroc = result["sets"][set_name]["auroc"]
        for index, score in enumerate(SERIES):
            for row, arm in enumerate(arms):
                value = auroc.get(f"{score}:{arm}")
                if value is None:
                    continue
                axis.plot(value, row + (index - 0.5) * 2 * OFFSET, marker=MARKERS[score], color=SERIES[score],
                          markersize=8, markeredgecolor="white", markeredgewidth=1, linestyle="none",
                          label=LABELS[score] if row == 0 else None)
        axis.set_title(title, color=INK, fontsize=10, loc="left")
        axis.set_xlabel("AUROC, standard label", color=INK)
        style(axis)
    axes[0].set_yticks(range(len(arms)), [ARMS[arm] for arm in arms])
    axes[0].invert_yaxis()
    axes[0].legend(frameon=False, fontsize=8, loc="upper left")
    figure.suptitle("The same xECG architecture, input path and pooling; only the weights differ", color=INK,
                    x=0.01, ha="left")
    figure.tight_layout()
    figure.savefig(FIGURES / "random_encoder.png", dpi=150)
    plt.close(figure)


def main() -> None:
    """Read the Experiment 036 result and write its figure."""
    FIGURES.mkdir(parents=True, exist_ok=True)
    plot(json.loads(RESULT.read_text()))


if __name__ == "__main__":
    main()
