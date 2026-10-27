"""Fixed examples and all-fit summaries for generated-background replication."""

from pathlib import Path
from typing import Any

import numpy as np

from ecg_experiment.ssl_synthetic_generated_quality import LEADS


def contrast_panel(axis: Any, aggregate: dict[str, Any], mode: str) -> None:
    """Plot every new fit's paired estimate and conditional background interval."""
    for color, offset, comparator in [("C0", -0.08, "absolute_amplitude"), ("C1", 0.08, "permuted")]:
        key = f"supervised_minus_{comparator}_{mode}"
        rows = [fit[key] for fit in aggregate["per_fit_contrasts"]]
        means = np.array([row["mean"] for row in rows])
        positions = np.arange(1, 4) + offset
        axis.vlines(positions, [row["ci_low"] for row in rows], [row["ci_high"] for row in rows], color=color)
        axis.plot(positions, means, "o", color=color, label=comparator)
    axis.axhline(0, color="black", linewidth=0.6)
    if mode == "pooled":
        axis.axhline(0.05, color="gray", linewidth=0.6, linestyle="--")
    axis.set(title=f"{mode}: supervised AP gain", xlabel="Fixed fit ID", ylabel="AP difference")
    axis.set_xticks([1, 2, 3])
    axis.legend(fontsize=6)


def diagnostic_figure(
    destination: Path, metrics: dict[int, dict[str, Any]], aggregate: dict[str, Any]
) -> None:
    """Render all fit estimates, both modes and all fresh-panel control distributions.

    Parameters
    ----------
    destination : Path
        Fresh local experiment directory.
    metrics : dict
        Fit ID to arm to saved case/control metrics on all 64 fresh backgrounds.
    aggregate : dict
        Frozen primary aggregate, including conditional intervals for fits 1–3.
    """
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 3, figsize=(18, 9))
    for arm in ["supervised", "permuted"]:
        values = [np.mean([row["mean_positive_ap"] for row in metrics[i][arm]["cases"]]) for i in range(4)]
        axes[0, 0].plot(range(4), values, marker="o", label=arm)
    for field in ["amplitude_ap", "spatial_prior_ap"]:
        cases = metrics[0]["supervised"]["cases"]
        value = np.mean([row[mode][field] for row in cases for mode in ["focal", "repeated"]])
        axes[0, 0].axhline(value, linestyle="--", linewidth=0.7, label=field)
    axes[0, 0].set(title="Fresh panel: equal-mode AP", xlabel="Fit ID (0: replay reference)", ylabel="AP")
    axes[0, 0].set_xticks(range(4))
    axes[0, 0].legend(fontsize=6)
    for axis, mode in zip([axes[0, 1], axes[0, 2], axes[1, 0]], ["pooled", "focal", "repeated"], strict=True):
        contrast_panel(axis, aggregate, mode)
    for axis, arm in zip(axes[1, 1:], ["supervised", "permuted"], strict=True):
        for fit in range(4):
            for mode, linestyle in [("sham", "-"), ("nuisance", "--")]:
                values = metrics[fit][arm]["controls"][mode]["record_maximum_probabilities"]
                axis.hist(
                    values,
                    bins=np.linspace(0, 1, 21),
                    histtype="step",
                    linestyle=linestyle,
                    color=f"C{fit}",
                    label=f"fit{fit} {mode}",
                )
        axis.axvline(0.5, color="black", linewidth=0.6)
        axis.set(title=f"{arm}: uncalibrated control maxima", xlabel="Maximum probability", ylabel="Records")
        axis.legend(fontsize=5)
    fig.suptitle("All fixed fits and fresh backgrounds; intervals condition on each fit", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(destination / "fit_diagnostics.png", dpi=120)
    plt.close(fig)


def example_figure(
    destination: Path,
    observed: np.ndarray,
    support: np.ndarray,
    maps: dict[str, np.ndarray],
    position: int,
    mode: str,
) -> None:
    """Render the prospectively fixed first eight fresh cases using fit 1 only.

    Parameters
    ----------
    destination : Path
        Fresh local experiment directory.
    observed, support : np.ndarray
        All fresh variants and their separately known introduced-change masks.
    maps : dict
        Fit 1 supervised/permuted saved maps, without recomputation or selection.
    position : int
        Positive variant position, two for focal or three for repeated.
    mode : str
        Corresponding positive mode name used in the fixed filename.
    """
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(8, 4, figsize=(20, 20))
    times = np.arange(5000) / 500
    for index in range(8):
        lead = int(np.flatnonzero(support[index, position].any(axis=1))[0])
        axes[index, 0].plot(times, observed[index, 1, lead], color="gray", linewidth=0.5)
        axes[index, 0].plot(times, observed[index, position, lead], color="black", linewidth=0.5)
        axes[index, 0].set_ylabel(f"Seed {2026104000 + index}; {LEADS[lead]}")
        axes[index, 1].imshow(
            support[index, position],
            aspect="auto",
            interpolation="nearest",
            extent=(0, 10, 12, 0),
            vmin=0,
            vmax=1,
            cmap="gray_r",
        )
        for column, arm in [(2, "supervised"), (3, "permuted")]:
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
    for column, title in enumerate(
        [
            "Observed signal and nuisance (robust units)",
            "Known added support",
            "Supervised output",
            "Permuted output",
        ]
    ):
        axes[0, column].set_title(title, fontsize=9)
        axes[-1, column].set_xlabel("Seconds")
    fig.suptitle(f"Fixed fit 1, {mode}, global indices 1000–1007; probability scale 0–1", fontsize=12)
    fig.tight_layout(rect=(0, 0, 0.94, 0.97))
    color_axis = fig.add_axes((0.95, 0.2, 0.01, 0.6))
    fig.colorbar(shown, cax=color_axis, label="Uncalibrated probability")
    fig.savefig(destination / f"fixed_{mode}_maps.png", dpi=110)
    plt.close(fig)


def make_figures(
    destination: Path,
    observed: np.ndarray,
    support: np.ndarray,
    maps: dict[int, dict[str, np.ndarray]],
    metrics: dict[int, dict[str, Any]],
    aggregate: dict[str, Any],
) -> list[str]:
    """Save all prospectively fixed replication figures from already scored outputs.

    Parameters
    ----------
    destination : Path
        New local output directory; existing figure filenames are prohibited.
    observed, support : np.ndarray
        Fresh 64-background waveform/target banks.
    maps, metrics : dict
        Every fit's fresh saved maps and case/control metrics.
    aggregate : dict
        Frozen primary aggregate for fits 1–3.

    Returns
    -------
    list[str]
        Ordered fixed figure filenames.
    """
    import matplotlib

    matplotlib.use("Agg")
    names = ["fit_diagnostics.png", "fixed_focal_maps.png", "fixed_repeated_maps.png"]
    if any((destination / name).exists() for name in names):
        raise FileExistsError("Fixed replication figures already exist")
    diagnostic_figure(destination, metrics, aggregate)
    for position, mode in [(2, "focal"), (3, "repeated")]:
        example_figure(destination, observed, support, maps[1], position, mode)
    return names
