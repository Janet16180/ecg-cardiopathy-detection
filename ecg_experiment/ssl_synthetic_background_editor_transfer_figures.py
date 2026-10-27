"""Render fixed saved numerical interventions and all four frozen model maps locally."""

import json
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np

from ecg_experiment.waveforms import LEADS, SAMPLE_RATE

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

SOURCE_FILES = (
    "ecg_experiment/ssl_synthetic_background_editor_transfer_figures.py",
    "scripts/reports/ssl_synthetic_background_editor_transfer_figures.py",
    "ecg_experiment/waveforms.py",
    "ecg_experiment/files.py",
)
COHORTS = ("generated", "real")
VARIANTS = ("Hann focal", "Hann repeated", "Biphasic focal", "Biphasic repeated")
ARMS = ("B", "P", "K", "PK")
NATIVE_INDICES = (0, 1, 6, 7, 8, 9, 10, 11)


def geometry_row(root: Path, source: str) -> dict[str, Any]:
    """Select only the prospectively fixed slot-zero geometry without inspecting predictions.

    Parameters
    ----------
    root : Path
        Immutable completed transfer-study output directory.
    source : str
        Generated or real source identity.

    Returns
    -------
    dict
        Exactly one slot-zero geometry declaration with the four fixed positive editors.
    """
    with (root / "geometry.jsonl").open() as stream:
        rows = [json.loads(line) for line in stream if line.strip()]
    selected = [row for row in rows if row["source"] == source and row["slot"] == 0]
    if len(selected) != 1 or selected[0]["geometry_index"] != 2000 or len(selected[0]["edits"]) != 4:
        raise ValueError("Fixed source/slot-zero geometry is incomplete or duplicated")
    return selected[0]


def source_figure(
    root: Path, destination: Path, source_position: int, observed: np.ndarray, support: np.ndarray
) -> str:
    """Display every fixed positive intervention on its declared native canonical lead.

    Parameters
    ----------
    root : Path
        Immutable main-study directory with all input and probability banks.
    destination : Path
        Local output directory for a new aggregate-free diagnostic image.
    source_position : int
        Fixed generated-zero or real-one tensor source position.
    observed : np.ndarray
        Read-only full canonical input bank.
    support : np.ndarray
        Read-only four-positive final-difference threshold masks.

    Returns
    -------
    str
        Newly created local PNG basename.
    """
    source = COHORTS[source_position]
    row = geometry_row(root, source)
    name = f"{source}_slot0_all_editors_and_maps.png"
    output = destination / name
    if output.exists():
        raise FileExistsError(output)
    maps = {
        arm: np.load(root / "probabilities" / f"fit_1_{arm}.npy", mmap_mode="r", allow_pickle=False)
        for arm in ARMS
    }
    if any(bank.shape != (2, 64, 6, 12, 5000) or bank.dtype != np.float32 for bank in maps.values()):
        raise ValueError("Required fixed-fit full probability geometry differs")
    figure, axes = plt.subplots(4, 6, figsize=(25, 14), constrained_layout=True)
    times = np.arange(5000) / SAMPLE_RATE
    titles = ["Observed, matched nuisance and actual difference", "Exact scoring support >0.01", *ARMS]
    image = None
    for position, label in enumerate(VARIANTS):
        variant = position + 2
        declaration = row["edits"][position]
        canonical = NATIVE_INDICES[declaration["independent_lead"]]
        axis = axes[position, 0]
        actual = observed[source_position, 0, variant, canonical]
        control = observed[source_position, 0, 1, canonical]
        difference = actual.astype(np.float64) - control.astype(np.float64)
        axis.plot(times, control, color="0.65", linewidth=0.7, label="Matched nuisance")
        axis.plot(times, actual, color="black", linewidth=0.6, label="Observed")
        axis.plot(times, difference, color="#187d7c", linewidth=1.0, label="Actual difference")
        axis.set(xlim=(0, 10), xlabel="Seconds", ylabel=f"{label}\n{LEADS[canonical]} · dimensionless")
        if position == 0:
            axis.legend(fontsize=7, loc="upper right")
        axes[position, 1].imshow(
            support[source_position, 0, position],
            aspect="auto",
            origin="upper",
            extent=(0, 10, 11.5, -0.5),
            vmin=0,
            vmax=1,
            interpolation="nearest",
            cmap="Greys",
        )
        for column, arm in enumerate(ARMS, start=2):
            values = maps[arm][source_position, 0, variant]
            if not np.isfinite(values).all() or np.any(values < 0) or np.any(values > 1):
                raise ValueError("Saved fixed probability map is invalid")
            image = axes[position, column].imshow(
                values,
                aspect="auto",
                origin="upper",
                extent=(0, 10, 11.5, -0.5),
                vmin=0,
                vmax=1,
                interpolation="nearest",
                cmap="magma",
            )
        for column in range(1, 6):
            axes[position, column].set(xlabel="Seconds", yticks=np.arange(12), yticklabels=LEADS)
            axes[position, column].tick_params(axis="y", labelsize=7)
        for column, title in enumerate(titles):
            if position == 0:
                axes[position, column].set_title(title, fontsize=10)
    figure.colorbar(
        image,
        ax=axes[:, 2:].ravel().tolist(),
        label="Uncalibrated model probability · common scale 0–1",
        shrink=0.75,
    )
    figure.suptitle(
        f"{source.capitalize()} background · fixed slot0 / geometry2000 / fit1\n"
        "Introduced numerical differences and frozen maps; these are not disease annotations",
        fontsize=15,
    )
    try:
        with output.open("xb") as stream:
            figure.savefig(stream, format="png", dpi=110)
    finally:
        plt.close(figure)
        del maps
    return name


def plot(root: Path, destination: Path) -> list[str]:
    """Write two fixed local figures without new scores, forwards or outcome-based selection.

    Parameters
    ----------
    root : Path
        Completed immutable main-study output directory, containing inputs.npy and probabilities/.
    destination : Path
        Local figure directory; existing PNG outputs are never overwritten.

    Returns
    -------
    list[str]
        Generated and real PNG basenames, in fixed tensor source order.
    """
    destination.mkdir(parents=True, exist_ok=True)
    observed = np.load(root / "inputs.npy", mmap_mode="r", allow_pickle=False)
    support = np.load(root / "support.npy", mmap_mode="r", allow_pickle=False)
    if observed.shape != (2, 64, 6, 12, 5000) or observed.dtype != np.float32:
        raise ValueError("Saved canonical observation geometry differs")
    if support.shape != (2, 64, 4, 12, 5000) or support.dtype != np.bool_:
        raise ValueError("Saved final-input truth geometry differs")
    return [source_figure(root, destination, source, observed, support) for source in range(2)]
