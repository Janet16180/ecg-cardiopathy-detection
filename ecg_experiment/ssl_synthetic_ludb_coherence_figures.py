"""Render every fixed activity-synchrony comparison without clinical-quality ranking."""

from pathlib import Path
from typing import Any

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from ecg_experiment.ssl_synthetic_ludb_coherence_bridge import VARIABLES

LEADS = ("I", "II", "V1", "V2", "V3", "V4", "V5", "V6")


def comparison_figures(destination: Path, cohorts: dict[str, list[dict[str, Any]]]) -> None:
    """Show all-record scatter, ECDFs and prospectively fixed-bin distributions."""
    figure, axes = plt.subplots(1, 2, figsize=(12, 6), constrained_layout=True)
    for axis, (name, rows) in zip(axes, cohorts.items(), strict=True):
        axis.scatter([row["coherence"]["shifted_jaccard"] for row in rows],
                     [row["coherence"]["native_jaccard"] for row in rows], s=9, alpha=.6)
        axis.plot([0, 1], [0, 1], color="gray", linewidth=.6)
        axis.set(xlim=(0, 1), ylim=(0, 1), xlabel="Shifted activity Jaccard",
                 ylabel="Native activity Jaccard",
                 title=f"{name}: all{len(rows)} records")
    figure.suptitle("Cross-lead activity synchrony is not clinical ECG quality")
    figure.savefig(destination / "all_record_coherence_scatter.png", dpi=120)
    plt.close(figure)
    figure, axes = plt.subplots(1, 3, figsize=(15, 5), constrained_layout=True)
    histogram, bins = plt.subplots(1, 3, figsize=(15, 5), constrained_layout=True)
    for axis, hist_axis, variable in zip(axes, bins, VARIABLES, strict=True):
        for name, rows in cohorts.items():
            values = np.sort([row["coherence"][variable] for row in rows])
            axis.step(values, np.arange(1, len(values) + 1) / len(values), where="post", label=name)
            edges = np.linspace(-1, 1, 21) if variable == "difference" else np.linspace(0, 1, 21)
            hist_axis.hist(values, bins=edges, histtype="step", density=False, label=name)
        limits = (-1, 1) if variable == "difference" else (0, 1)
        axis.set(xlim=limits, ylim=(0, 1), xlabel=variable, ylabel="Empirical cumulative fraction")
        hist_axis.set(xlim=limits, xlabel=variable, ylabel="Record count")
        axis.legend()
        hist_axis.legend()
    figure.suptitle("Fixed activity descriptors; no realism or physiological-quality threshold")
    histogram.suptitle("Prospective fixed-bin distributions; unequal cohort sizes retained")
    figure.savefig(destination / "coherence_ECDFs.png", dpi=120)
    histogram.savefig(destination / "coherence_histograms.png", dpi=120)
    plt.close(figure)
    plt.close(histogram)


def raster_figure(destination: Path, generated: np.ndarray, real: np.ndarray) -> None:
    """Show all eight activity masks for only the first four prospectively fixed records per cohort."""
    figure, axes = plt.subplots(4, 2, figsize=(16, 10), constrained_layout=True)
    for index in range(4):
        for column, (name, masks) in enumerate([("generated", generated), ("LUDB", real)]):
            axes[index, column].imshow(masks[index], aspect="auto", interpolation="nearest", cmap="Greys",
                                        vmin=0, vmax=1, extent=(0, 10, 7.5, -.5))
            identifier = index if name == "generated" else index + 1
            axes[index, column].set(yticks=np.arange(8), yticklabels=LEADS, xlabel="Seconds",
                                    title=f"Fixed {name} record/index{identifier}; full5000samples")
    figure.suptitle("Fixed activity masks; synchrony does not establish physiological fidelity")
    figure.savefig(destination / "fixed_activity_rasters.png", dpi=120)
    plt.close(figure)
