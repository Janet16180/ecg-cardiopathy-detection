"""Plot the Experiment 031 figure: sensitivity against budget per rule, and the held-out superclasses."""

import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
RESULT = ROOT / "outputs/experiment031_hybrid_score_v1/result.json"
FIGURES = ROOT / "docs/figures/experiment-031"
SERIES = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#8b5cc4")
MARKERS = ("o", "s", "^", "D", "v")
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
RULES = {"readout": "Readout alone (022b)", "distance": "Distance from normal (026b)",
         "zmean": "z-mean hybrid (primary)", "stack": "Logistic stack", "either": "OR rule"}
SUPERCLASSES = ("MI", "STTC", "CD", "HYP")
SIZE, BUDGET = 200, 0.05


def style(axis: plt.Axes) -> None:
    """
    Apply the recessive grid and axis style.

    Parameters
    ----------
    axis : plt.Axes
        Axis to style.
    """
    axis.grid(axis="y", color=GRID, linewidth=0.8)
    axis.set_axisbelow(True)
    for side in ("top", "right"):
        axis.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        axis.spines[side].set_color(MUTED)
    axis.tick_params(colors=MUTED, labelcolor=INK)


def rows_of(summary: list[dict[str, Any]], **select: Any) -> list[dict[str, Any]]:
    """
    Summary rows matching every selected field, sorted by budget.

    Parameters
    ----------
    summary : list[dict[str, Any]]
        Summary rows.
    **select : Any
        Field values to match.

    Returns
    -------
    list[dict[str, Any]]
        Matching rows.
    """
    matching = [row for row in summary if all(row[key] == value for key, value in select.items())]
    return sorted(matching, key=lambda row: row["budget"])


def plot(result: dict[str, Any]) -> None:
    """
    Write the two-panel figure for xECG with 200 local normals.

    Parameters
    ----------
    result : dict[str, Any]
        Experiment 031 ``result.json``.
    """
    summary = result["summary"]
    figure, axes = plt.subplots(1, 2, figsize=(13, 4.6))
    budgets = [100 * budget for budget in result["budgets"]]

    axis = axes[0]
    for (rule, label), color, marker in zip(RULES.items(), SERIES, MARKERS, strict=True):
        rows = rows_of(summary, score=rule, encoder="xecg", m=SIZE)
        axis.plot(budgets, [row["sensitivity_mean"] for row in rows], color=color, marker=marker,
                  linewidth=2, markersize=5, label=label, linestyle="--" if rule == "stack" else "-")
    axis.set_xscale("log")
    axis.set_xticks(budgets, [f"{value:g}%" for value in budgets])
    axis.minorticks_off()
    axis.set_xlabel("Referral budget (share of normals referred)", color=INK)
    axis.set_ylabel("Sensitivity (abnormal ECGs referred)", color=INK)
    axis.set_title("All abnormal ECGs", color=INK, loc="left")
    axis.legend(frameon=False, fontsize=9)
    style(axis)

    axis = axes[1]
    arms = (("readout", "Full readout", "#9a9994"), ("readout_without", "Readout without S", SERIES[0]),
            ("zmean_without", "z-mean hybrid without S", SERIES[2]))
    positions = np.arange(len(SUPERCLASSES))
    width = 0.26
    for offset, (arm, label, color) in enumerate(arms):
        means, lows, highs = [], [], []
        for name in SUPERCLASSES:
            score = arm if arm == "readout" else f"{arm}_{name}"
            (row,) = [row for row in summary if row["score"] == score and row["encoder"] == "xecg"
                      and row["m"] == SIZE and row["budget"] == BUDGET]
            mean = row[f"sensitivity_{name}_mean"]
            low, high = row[f"sensitivity_{name}_ci"]
            means.append(mean)
            lows.append(mean - low)
            highs.append(high - mean)
        axis.bar(positions + (offset - 1) * width, means, width, color=color, label=label,
                 yerr=[lows, highs], ecolor=INK, capsize=2, error_kw={"linewidth": 0.8})
    axis.set_xticks(positions, SUPERCLASSES)
    axis.set_ylim(0, 1.12)
    axis.set_yticks(np.arange(0, 1.01, 0.2))
    axis.set_ylabel("Share of the superclass referred", color=INK)
    axis.set_title("Held-out superclass S, 5% budget", color=INK, loc="left")
    axis.legend(frameon=False, fontsize=9, loc="upper center", ncol=3)
    style(axis)

    figure.tight_layout()
    FIGURES.mkdir(parents=True, exist_ok=True)
    figure.savefig(FIGURES / "hybrid_score.png", dpi=150)


if __name__ == "__main__":
    plot(json.loads(RESULT.read_text()))
