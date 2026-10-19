"""Fixed all-arm figures for the prospective paired/background factorial."""

from pathlib import Path
from typing import Any

import numpy as np

from ecg_experiment.ssl_synthetic_generated_quality import LEADS

ARMS = ("B", "P", "K", "PK")


def mode_ap(metrics: dict[int, dict[str, Any]], fit: int, arm: str, mode: str) -> float:
    """Read the saved equal-background AP mean for one fixed fit, arm and mode."""
    return float(np.mean([row[mode]["edited_ap"] for row in metrics[fit][arm]["cases"]]))


def summary_figure(destination: Path, metrics: dict[int, dict[str, Any]]) -> None:
    """Plot every fixed arm/fit, primary point differences and control distributions.

    Parameters
    ----------
    destination : Path
        New local output directory.
    metrics : dict
        Fixed fit IDs 1–3, each with B/P/K/PK case, control and top600 summaries.
    """
    import matplotlib.pyplot as plt

    fits = [1, 2, 3]
    fig, axes = plt.subplots(2, 3, figsize=(18, 9))
    for color, arm in enumerate(ARMS):
        ap = [np.mean([mode_ap(metrics, fit, arm, mode) for mode in ["focal", "repeated"]]) for fit in fits]
        peaks = [metrics[fit][arm]["top600_control"]["pooled"]["mean"] for fit in fits]
        axes[0, 0].plot(fits, ap, "o-", color=f"C{color}", label=arm)
        axes[0, 1].plot(fits, peaks, "o-", color=f"C{color}", label=arm)
    for mode in ["focal", "repeated"]:
        values = [mode_ap(metrics, fit, "PK", mode) - mode_ap(metrics, fit, "B", mode) for fit in fits]
        axes[0, 2].plot(fits, values, "o-", label=mode)
    pooled = [
        np.mean(
            [
                mode_ap(metrics, fit, "PK", mode) - mode_ap(metrics, fit, "B", mode)
                for mode in ["focal", "repeated"]
            ]
        )
        for fit in fits
    ]
    axes[0, 2].plot(fits, pooled, "o--", label="equal-mode pooled")
    axes[0, 2].axhline(0, color="black", linewidth=0.6)
    axes[0, 2].axhline(0.02, color="gray", linestyle="--", linewidth=0.6)
    for mode in ["sham", "nuisance", "pooled"]:
        values = [
            metrics[fit]["B"]["top600_control"][mode]["mean"]
            - metrics[fit]["PK"]["top600_control"][mode]["mean"]
            for fit in fits
        ]
        axes[1, 0].plot(fits, values, "o-", label=mode)
    axes[1, 0].axhline(0, color="black", linewidth=0.6)
    axes[1, 0].axhline(0.02, color="gray", linestyle="--", linewidth=0.6)
    for axis, mode in zip(axes[1, 1:], ["sham", "nuisance"], strict=True):
        for color, arm in enumerate(ARMS):
            values = [
                value
                for fit in fits
                for value in metrics[fit][arm]["controls"][mode]["record_maximum_probabilities"]
            ]
            axis.hist(values, bins=np.linspace(0, 1, 21), histtype="step", color=f"C{color}", label=arm)
        axis.axvline(0.5, color="black", linewidth=0.6)
        axis.set(
            title=f"{mode}: all three fits, descriptive",
            xlabel="Uncalibrated maximum",
            ylabel="Fit/background observations",
        )
    for axis, title, ylabel in [
        (axes[0, 0], "All-arm equal-mode AP", "AP"),
        (axes[0, 1], "All-arm control top600 mean", "Probability"),
        (axes[0, 2], "Primary PK minus B AP gain", "AP difference"),
        (axes[1, 0], "Primary B minus PK control reduction", "Probability difference"),
    ]:
        axis.set(title=title, xlabel="Fixed fit ID", ylabel=ylabel)
        axis.set_xticks(fits)
    for axis in axes.ravel():
        axis.legend(fontsize=7)
    fig.suptitle("Fixed factorial: B=BCE, P=paired, K=control penalty, PK=both", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(destination / "factorial_diagnostics.png", dpi=120)
    plt.close(fig)


def example_figure(
    destination: Path,
    observed: np.ndarray,
    support: np.ndarray,
    maps: dict[str, np.ndarray],
    mode: str,
    position: int,
) -> None:
    """Show the same first eight future backgrounds and all four fixed fit-1 models.

    Parameters
    ----------
    destination : Path
        New local output directory.
    observed, support : np.ndarray
        Fresh 64-background four-variant banks and introduced-change masks.
    maps : dict
        Saved fit-1 B/P/K/PK probability maps.
    mode : str
        Focal or repeated mode name.
    position : int
        Corresponding four-variant index, two or three.
    """
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(8, 6, figsize=(28, 20))
    times = np.arange(5000) / 500
    for index in range(8):
        lead = int(np.flatnonzero(support[index, position].any(axis=1))[0])
        axes[index, 0].plot(times, observed[index, 1, lead], color="gray", linewidth=0.5)
        axes[index, 0].plot(times, observed[index, position, lead], color="black", linewidth=0.5)
        axes[index, 0].set_ylabel(f"Seed {2026105000 + index}; {LEADS[lead]}")
        axes[index, 1].imshow(
            support[index, position],
            aspect="auto",
            interpolation="nearest",
            extent=(0, 10, 12, 0),
            vmin=0,
            vmax=1,
            cmap="gray_r",
        )
        for column, arm in enumerate(ARMS, start=2):
            shown = axes[index, column].imshow(
                maps[arm][index, position],
                aspect="auto",
                interpolation="nearest",
                extent=(0, 10, 12, 0),
                vmin=0,
                vmax=1,
                cmap="viridis",
            )
        for axis in axes[index, 1:]:
            axis.set_yticks([0.5, 2.5, 6.5, 11.5], ["I", "III", "V1", "V6"])
    for column, title in enumerate(["Observed and nuisance (robust units)", "Known added support", *ARMS]):
        axes[0, column].set_title(title, fontsize=9)
        axes[-1, column].set_xlabel("Seconds")
    fig.suptitle(f"Fixed fit 1, {mode}, global indices 2000–2007; all maps use scale 0–1", fontsize=12)
    fig.tight_layout(rect=(0, 0, 0.94, 0.97))
    color_axis = fig.add_axes((0.95, 0.2, 0.01, 0.6))
    fig.colorbar(shown, cax=color_axis, label="Uncalibrated probability")
    fig.savefig(destination / f"fixed_{mode}_factorial.png", dpi=100)
    plt.close(fig)


def make_figures(
    destination: Path,
    observed: np.ndarray,
    support: np.ndarray,
    maps: dict[int, dict[str, np.ndarray]],
    metrics: dict[int, dict[str, Any]],
    aggregate: dict[str, Any],
) -> list[str]:
    """Render the fixed all-fit summary and both first-eight all-arm galleries.

    Parameters
    ----------
    destination : Path
        Fresh output directory; existing plot files are prohibited.
    observed, support : np.ndarray
        Future64 waveform/target banks.
    maps, metrics : dict
        Fixed fit IDs 1–3 and B/P/K/PK saved predictions/measurements.
    aggregate : dict
        Saved primary/secondary factorial summaries; no arm selection occurs here.

    Returns
    -------
    list[str]
        Ordered fixed filenames.
    """
    import matplotlib

    matplotlib.use("Agg")
    names = ["factorial_diagnostics.png", "fixed_focal_factorial.png", "fixed_repeated_factorial.png"]
    if any((destination / name).exists() for name in names):
        raise FileExistsError("Fixed factorial figures already exist")
    summary_figure(destination, metrics)
    for mode, position in [("focal", 2), ("repeated", 3)]:
        example_figure(destination, observed, support, maps[1], mode, position)
    return names
