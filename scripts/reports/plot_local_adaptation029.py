"""Plot the Experiment 029 figures: sensitivity against local ECGs, AUROC against local normals."""

import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
RESULT = ROOT / "outputs/experiment029_local_adaptation_v1/result.json"
FIGURES = ROOT / "docs/figures/experiment-029"
SERIES = ("#2a78d6", "#eb6834", "#1baf7a")
MARKERS = ("o", "s", "^")
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
OPTIONS_A = {"platt_threshold": "Platt + threshold (primary)", "blend": "Blend with source data",
             "conservative": "Conservative threshold"}
OPTIONS_B = {"local_plus_pooled": "Pooled + local normals (primary)", "local_only": "Local normals only",
             "recentered": "Pooled, recentred on local mean"}


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
    Summary rows matching every selected field, sorted by size.

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
    return sorted(matching, key=lambda row: row.get("n", row.get("m")))


def plot_part_a(result: dict[str, Any]) -> None:
    """
    Mean and 5th-percentile evaluation sensitivity against local ECGs, xECG pooled readout.

    Parameters
    ----------
    result : dict[str, Any]
        Experiment 029 ``result.json``.
    """
    summary = result["part_a"]["summary"]
    figure, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    for axis, plan, title in ((axes[0], "site", "Local draws at SPH prevalence (34%)"),
                              (axes[1], "low", "Local draws at 5% prevalence")):
        source = rows_of(summary, plan=plan, readout="pooled", encoder="xecg", option="source")[0]
        for (option, label), color, marker in zip(OPTIONS_A.items(), SERIES, MARKERS, strict=True):
            rows = rows_of(summary, plan=plan, readout="pooled", encoder="xecg", option=option)
            sizes = [0] + [row["n"] for row in rows]
            means = [source["sensitivity_mean"]] + [row["sensitivity_mean"] for row in rows]
            lows = [source["sensitivity_p5"]] + [row["sensitivity_p5"] for row in rows]
            positions = range(len(sizes))
            axis.plot(positions, means, color=color, marker=marker, markersize=6, linewidth=2, label=label)
            axis.plot(positions, lows, color=color, linewidth=1.2, linestyle="--")
        axis.axhline(0.95, color=INK, linewidth=1, linestyle=":")
        axis.set_xticks(range(len(sizes)), [f"{size:,}" for size in sizes])
        axis.set_xlabel("Local ECGs read by the cardiologist (n)", color=INK)
        axis.set_title(title, color=INK, fontsize=11, loc="left")
        style(axis)
    axes[0].set_ylabel("Sensitivity on the evaluation half", color=INK)
    axes[0].text(0.05, 0.952, "target 0.95", color=MUTED, fontsize=8)
    axes[1].text(0.05, 0.10, "solid: mean of 200 draws\ndashed: 5th percentile", transform=axes[1].transAxes,
                 color=MUTED, fontsize=8)
    axes[0].legend(frameon=False, fontsize=8, loc="lower right")
    figure.suptitle("xECG, pooled readout: sensitivity after local recalibration", color=INK, x=0.01,
                    ha="left")
    figure.tight_layout()
    figure.savefig(FIGURES / "sensitivity_by_local_ecgs.png", dpi=150)
    plt.close(figure)


def plot_part_b(result: dict[str, Any]) -> None:
    """
    Mean evaluation AUROC with 5th-95th percentiles against local normals, xECG.

    Parameters
    ----------
    result : dict[str, Any]
        Experiment 029 ``result.json``.
    """
    summary = result["part_b"]["summary"]
    references = result["part_b"]["references"]["xecg"]
    figure, axis = plt.subplots(figsize=(7, 4.2))
    for (option, label), color, marker in zip(OPTIONS_B.items(), SERIES, MARKERS, strict=True):
        rows = rows_of(summary, encoder="xecg", option=option)
        positions = range(len(rows))
        axis.fill_between(positions, [row["auroc_p5"] for row in rows], [row["auroc_p95"] for row in rows],
                          color=color, alpha=0.15, linewidth=0)
        axis.plot(positions, [row["auroc_mean"] for row in rows], color=color, marker=marker, markersize=6,
                  linewidth=2, label=label)
    for key, label, style_ in (("nonlocal_pooled", "Non-local normal reference (026b)", "--"),
                               ("supervised_pooled", "Supervised pooled readout (022b)", ":")):
        value = references[key]["auroc"]
        axis.axhline(value, color=MUTED, linewidth=1.2, linestyle=style_)
        axis.text(len(rows) - 1, value + 0.002, f"{label} {value:.3f}", color=MUTED, fontsize=8, ha="right")
    axis.set_xticks(range(len(rows)), [f"{row['m']:,}" for row in rows])
    axis.set_xlabel("Local normal ECGs (m)", color=INK)
    axis.set_ylabel("AUROC on the evaluation half", color=INK)
    axis.set_title("xECG distance from normal with local normals (mean, 5th-95th percentile of 50 draws)",
                   color=INK, fontsize=10, loc="left")
    axis.legend(frameon=False, fontsize=8, loc="lower right")
    style(axis)
    figure.tight_layout()
    figure.savefig(FIGURES / "auroc_by_local_normals.png", dpi=150)
    plt.close(figure)


def main() -> None:
    """Write both figures from the saved result."""
    FIGURES.mkdir(parents=True, exist_ok=True)
    result = json.loads(RESULT.read_text())
    plot_part_a(result)
    plot_part_b(result)


if __name__ == "__main__":
    main()
