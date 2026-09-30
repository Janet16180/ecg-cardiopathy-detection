"""Plot the Experiment 030 figure: sensitivity against budget, and achieved rate against local normals."""

import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
RESULT = ROOT / "outputs/experiment030_referral_budget_v1/result.json"
FIGURES = ROOT / "docs/figures/experiment-030"
SERIES = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100")
MARKERS = ("o", "s", "^", "D")
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
SCORES = {"pooled": "Pooled readout (022b)", "ptbxl": "PTB-XL-only readout",
          "normal_ref": "Distance from normal (026b)"}
SUPERCLASSES = ("MI", "STTC", "CD", "HYP")
SIZE = 200


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
    Summary rows matching every selected field, sorted by budget and size.

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
    return sorted(matching, key=lambda row: (row["budget"], row["m"]))


def plot(result: dict[str, Any]) -> None:
    """
    Write the three-panel figure for xECG.

    Parameters
    ----------
    result : dict[str, Any]
        Experiment 030 ``result.json``.
    """
    summary = result["summary"]
    figure, axes = plt.subplots(1, 3, figsize=(15, 4.4))
    budgets = [100 * budget for budget in result["budgets"]]

    axis = axes[0]
    for (score, label), color, marker in zip(SCORES.items(), SERIES, MARKERS, strict=False):
        rows = rows_of(summary, score=score, encoder="xecg", m=SIZE)
        axis.fill_between(budgets, [row["sensitivity_p5"] for row in rows],
                          [row["sensitivity_p95"] for row in rows], color=color, alpha=0.15, linewidth=0)
        axis.plot(budgets, [row["sensitivity_mean"] for row in rows], color=color, marker=marker,
                  markersize=7, linewidth=2, label=label)
    axis.set_title("Abnormal ECGs referred, 200 local normals", color=INK, fontsize=10, loc="left")
    axis.set_ylabel("Sensitivity on the evaluation half", color=INK)
    axis.legend(frameon=False, fontsize=8, loc="lower right")

    axis = axes[1]
    for group, color, marker in zip(SUPERCLASSES, SERIES, MARKERS, strict=True):
        rows = rows_of(summary, score="pooled", encoder="xecg", m=SIZE)
        axis.plot(budgets, [row[f"sensitivity_{group}_mean"] for row in rows], color=color, marker=marker,
                  markersize=7, linewidth=2, label=group)
    axis.set_title("By superclass, pooled readout, 200 local normals", color=INK, fontsize=10, loc="left")
    axis.set_ylabel("Share of the superclass referred", color=INK)
    axis.legend(frameon=False, fontsize=8, loc="lower right")
    for axis in axes[:2]:
        axis.set_xticks(budgets, [f"{budget:g}%" for budget in budgets])
        axis.set_xlabel("Referral budget (share of normal ECGs flagged)", color=INK)
        axis.set_ylim(0, 1)
        style(axis)

    axis = axes[2]
    sizes = [0, *result["sizes"], *result["extra_sizes"]]
    positions = range(len(sizes))
    for budget, color, marker in zip((0.02, 0.05, 0.10), SERIES, MARKERS, strict=False):
        rows = rows_of(summary, score="pooled", encoder="xecg", budget=budget)
        axis.fill_between(positions, [100 * row["rate_p5"] for row in rows],
                          [100 * row["rate_p95"] for row in rows], color=color, alpha=0.15, linewidth=0)
        axis.plot(positions, [100 * row["rate_mean"] for row in rows], color=color, marker=marker,
                  markersize=7, linewidth=2, label=f"{100 * budget:g}% budget")
        axis.axhline(100 * budget, color=color, linewidth=1, linestyle=":")
    axis.set_xticks(positions, ["source"] + [f"{size:,}" for size in sizes[1:]])
    axis.set_xlabel("Local normal ECGs (m)", color=INK)
    axis.set_ylabel("Normal ECGs referred (%)", color=INK)
    axis.set_title("Achieved false-referral rate (mean, 5th-95th pct)", color=INK, fontsize=10, loc="left")
    axis.legend(frameon=False, fontsize=8, loc="upper right")
    style(axis)

    figure.suptitle("xECG at SPH: a threshold set by a referral budget on normal ECGs", color=INK, x=0.01,
                    ha="left")
    figure.tight_layout()
    figure.savefig(FIGURES / "referral_budget.png", dpi=150)
    plt.close(figure)


def main() -> None:
    """Write the figure from the saved result."""
    FIGURES.mkdir(parents=True, exist_ok=True)
    plot(json.loads(RESULT.read_text()))


if __name__ == "__main__":
    main()
