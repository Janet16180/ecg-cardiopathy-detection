"""Plot the Experiment 033 figure: composite gain against binary cost, and sensitivity per finding."""

import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
RESULT = ROOT / "outputs/experiment033_finding_heads_screen_v1/result.json"
FIGURES = ROOT / "docs/figures/experiment-033"
BLUE, ORANGE, GREY = "#2a78d6", "#eb6834", "#9a9994"
INK, MUTED, GRID, REGION = "#0b0b0b", "#52514e", "#e4e3df", "#e3f1e9"
SIZE, BUDGET, MAX_COST = 200, 0.05, 0.010
OUTCOMES = {"binary_positive": "Binary\nlabel", "composite": "Athlete\ncomposite", "pvc": "PVC",
            "frequent_pvc": "Frequent\nPVC", "wpw": "WPW", "af_flutter": "AF/\nflutter",
            "high_grade_av_block": "HG AV\nblock", "long_qt": "Long\nQT"}


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


def primary_rows(result: dict[str, Any], table: str) -> list[dict[str, Any]]:
    """
    Rows of a result table at the primary budget and size.

    Parameters
    ----------
    result : dict[str, Any]
        Experiment 033 ``result.json``.
    table : str
        ``summary`` or ``contrasts``.

    Returns
    -------
    list[dict[str, Any]]
        Matching rows.
    """
    return [row for row in result[table] if row["m"] == SIZE and row["budget"] == BUDGET]


def plot(result: dict[str, Any]) -> None:
    """
    Write the two-panel figure at a 5% budget with 200 local normals.

    Parameters
    ----------
    result : dict[str, Any]
        Experiment 033 ``result.json``.
    """
    selected = result["reading"]["selected"]
    contrasts = primary_rows(result, "contrasts")
    figure, axes = plt.subplots(1, 2, figsize=(13, 4.8), gridspec_kw={"width_ratios": [1, 1.25]})

    axis = axes[0]
    axis.axvspan(-0.002, MAX_COST, color=REGION, zorder=0)
    axis.axhline(0, color=MUTED, linewidth=0.8)
    for rule in [name for name in result["rules"] if name != "binary"]:
        gain = next(row for row in contrasts if row["rule"] == rule and row["outcome"] == "composite")
        binary = next(row for row in contrasts if row["rule"] == rule and row["outcome"] == "binary_positive")
        cost, cost_low, cost_high = -binary["difference"], -binary["ci"][1], -binary["ci"][0]
        color = ORANGE if rule == selected else BLUE
        marker = "o" if rule.startswith("combined") else "s"
        axis.errorbar(cost, gain["difference"], xerr=[[cost - cost_low], [cost_high - cost]],
                      yerr=[[gain["difference"] - gain["ci"][0]], [gain["ci"][1] - gain["difference"]]],
                      fmt=marker, color=color, ecolor=color, elinewidth=1, capsize=2, markersize=6)
        axis.annotate(rule, (cost, gain["difference"]), textcoords="offset points",
                      xytext=(6, 4) if rule.startswith("combined") else (6, -11),
                      fontsize=8, color=INK)
    axis.set_xlim(-0.002, 0.042)
    axis.set_ylim(-0.005, 0.05)
    axis.set_xlabel("Binary-label sensitivity cost (binary alone minus rule)", color=INK)
    axis.set_ylabel("Composite sensitivity gain over binary alone", color=INK)
    axis.set_title("Gain and cost per rule (shaded: cost at most 0.01)", color=INK, loc="left")
    style(axis)

    axis = axes[1]
    summary = primary_rows(result, "summary")
    arms = (("binary", "Binary readout alone", GREY), (selected, f"Selected rule ({selected})", ORANGE))
    positions = np.arange(len(OUTCOMES))
    width = 0.38
    for offset, (rule, label, color) in enumerate(arms):
        row = next(item for item in summary if item["rule"] == rule)
        means = np.array([row[name]["mean"] for name in OUTCOMES])
        lows = means - np.array([row[name]["ci"][0] for name in OUTCOMES])
        highs = np.array([row[name]["ci"][1] for name in OUTCOMES]) - means
        axis.bar(positions + (offset - 0.5) * width, means, width, color=color, label=label,
                 yerr=[lows, highs], ecolor=INK, capsize=2, error_kw={"linewidth": 0.8})
    axis.set_xticks(positions, list(OUTCOMES.values()), fontsize=9)
    axis.set_ylim(0, 1.15)
    axis.set_yticks(np.arange(0, 1.01, 0.2))
    axis.set_ylabel("Share referred (evaluation half)", color=INK)
    axis.set_title("5% budget, 200 local normals, xECG", color=INK, loc="left")
    axis.legend(frameon=False, fontsize=9, loc="upper center", ncol=2)
    style(axis)

    figure.tight_layout()
    FIGURES.mkdir(parents=True, exist_ok=True)
    figure.savefig(FIGURES / "finding_screen.png", dpi=150)


if __name__ == "__main__":
    plot(json.loads(RESULT.read_text()))
