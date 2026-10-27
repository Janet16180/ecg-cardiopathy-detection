"""Scale-free quality and exact intervention checks for public generated ECGs."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import numpy as np
from scipy.signal import periodogram
from sklearn.metrics import average_precision_score

from ecg_experiment.intervals import patient_groups, patient_resample

LEADS = ("I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6")


def derive_leads(independent: np.ndarray) -> np.ndarray:
    """Derive twelve leads in the input's numerical units.

    Parameters
    ----------
    independent : np.ndarray
        Eight independent float32 leads in I, II, V1 through V6 order.

    Returns
    -------
    np.ndarray
        Twelve float32 leads in canonical order.
    """
    first, second = independent[:2]
    return np.stack(
        [
            first,
            second,
            second - first,
            -(first + second) / 2,
            first - second / 2,
            second - first / 2,
            *independent[2:],
        ]
    ).astype(np.float32)


def standardize(raw: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
    """Center and scale native independent leads without asserting physical units.

    Parameters
    ----------
    raw : np.ndarray
        Native eight-lead output.

    Returns
    -------
    tuple[np.ndarray, dict[str, Any]]
        Standardized eight leads and retained transformation parameters.
    """
    if raw.shape != (8, 5000) or raw.dtype != np.float32 or not np.isfinite(raw).all():
        raise ValueError("Expected finite native float32 (8,5000) output")
    center = np.median(raw.astype(np.float64), axis=1)
    quantiles = np.quantile(raw.astype(np.float64), [0.05, 0.95], axis=1, method="linear")
    scale = quantiles[1] - quantiles[0]
    if np.any(scale <= 1e-6):
        raise ValueError("At least one native lead has an inadmissibly small robust range")
    values = ((raw.astype(np.float64) - center[:, None]) / scale[:, None]).astype(np.float32)
    return values, {"median": center.tolist(), "q95_minus_q05": scale.tolist()}


def activity_masks(values: np.ndarray) -> np.ndarray:
    """Locate high derivative-energy cells using a fixed scale-free rule.

    Parameters
    ----------
    values : np.ndarray
        Eight independent standardized leads.

    Returns
    -------
    np.ndarray
        Strict upper-decile masks for 40 ms smoothed squared derivatives.
    """
    differences = np.diff(values.astype(np.float64), axis=1, prepend=values[:, :1])
    envelopes = np.stack([np.convolve(row**2, np.ones(20) / 20, mode="same") for row in differences])
    return envelopes > np.quantile(envelopes, 0.9, axis=1, method="linear")[:, None]


def mean_jaccard(masks: np.ndarray) -> float:
    """Average temporal Jaccard coincidence across independent lead pairs.

    Parameters
    ----------
    masks : np.ndarray
        Boolean lead-time activity masks.

    Returns
    -------
    float
        Mean over 28 pairs; an empty union contributes zero.
    """
    scores = []
    for first in range(8):
        for second in range(first):
            union = np.count_nonzero(masks[first] | masks[second])
            scores.append(np.count_nonzero(masks[first] & masks[second]) / union if union else 0.0)
    return float(np.mean(scores))


def coherence(values: np.ndarray, index: int) -> dict[str, Any]:
    """Compare native timing coincidence against independent circular shifts.

    Parameters
    ----------
    values : np.ndarray
        Eight independent standardized leads.
    index : int
        Fixed position in the prospective bank.

    Returns
    -------
    dict[str, Any]
        Native/control scores, their paired difference and control shifts.
    """
    masks = activity_masks(values)
    shifts = np.random.default_rng(2026103200 + index).integers(1, 5000, size=8)
    shifted = np.stack([np.roll(row, shift) for row, shift in zip(masks, shifts, strict=True)])
    native, control = mean_jaccard(masks), mean_jaccard(shifted)
    return {
        "native_jaccard": native,
        "shifted_jaccard": control,
        "difference": native - control,
        "shifts": shifts.tolist(),
    }


def morphology(values: np.ndarray, raw: np.ndarray) -> dict[str, Any]:
    """Describe spectral content and derivative-energy activity without clinical interpretation.

    Parameters
    ----------
    values : np.ndarray
        Eight standardized leads.
    raw : np.ndarray
        Native output for saturation counts.

    Returns
    -------
    dict[str, Any]
        Scale-free spectra, activity and numerical shape diagnostics.
    """
    frequencies, power = periodogram(values.astype(np.float64), fs=500, axis=1)
    denominator = power[:, (frequencies >= 0.5) & (frequencies < 100)].sum(axis=1)
    bands = {
        "0p5_to_5": (frequencies >= 0.5) & (frequencies < 5),
        "5_to_25": (frequencies >= 5) & (frequencies < 25),
        "25_to_100": (frequencies >= 25) & (frequencies < 100),
    }
    ratios = {
        key: [
            float(numerator / total) if total > 0 else None
            for numerator, total in zip(power[:, selected].sum(axis=1), denominator, strict=True)
        ]
        for key, selected in bands.items()
    }
    return {
        "spectral_power_ratios": ratios,
        "zero_power_denominator": (denominator == 0).tolist(),
        "raw_saturation_fraction": float(np.mean(np.abs(raw) >= 0.99)),
        "standardized_peak_to_peak": np.ptp(values, axis=1).tolist(),
        "native_mean": raw.mean(axis=1).tolist(),
        "native_rms": np.sqrt(np.mean(raw.astype(np.float64) ** 2, axis=1)).tolist(),
        "native_std": np.std(raw.astype(np.float64), axis=1).tolist(),
        "standardized_rms": np.sqrt(np.mean(values.astype(np.float64) ** 2, axis=1)).tolist(),
        "standardized_std": np.std(values.astype(np.float64), axis=1).tolist(),
        "activity_occupancy": activity_masks(values).mean(axis=1).tolist(),
    }


def intervention_bank(values: np.ndarray, index: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Make four paired variants and retain exact and thresholded numerical supports.

    Parameters
    ----------
    values : np.ndarray
        Eight standardized independent leads.
    index : int
        Fixed bank position defining the nuisance phase.

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray]
        Four canonical variants, two strict masks and two thresholded masks.
    """
    phase = np.random.default_rng(2026103100 + index).uniform(0, 2 * np.pi)
    times = np.arange(5000) / 500
    nuisance = (0.1 * np.sin(2 * np.pi * 0.33 * times + phase)).astype(np.float32)
    baseline = (values + nuisance[None, :]).astype(np.float32)
    variants = [derive_leads(values.copy()), derive_leads(baseline)]
    for centers in [[1250], list(range(750, 4251, 500))]:
        changed = baseline.copy()
        for center in centers:
            changed[1, center - 40 : center + 41] += (0.5 * np.hanning(81)).astype(np.float32)
        variants.append(derive_leads(changed))
    bank = np.stack(variants)
    difference = bank[2:].astype(np.float64) - bank[1].astype(np.float64)
    return bank, difference != 0, np.abs(difference) > 0.01


def intervention_metrics(bank: np.ndarray, strict: np.ndarray, support: np.ndarray) -> dict[str, Any]:
    """Check exact intervention closure and report two support-ranking controls.

    Parameters
    ----------
    bank : np.ndarray
        Sham, nuisance, focal and repeated canonical variants.
    strict : np.ndarray
        Exact nonzero-difference masks for the two edit modes.
    support : np.ndarray
        Masks thresholded at 0.01 standardized units.

    Returns
    -------
    dict[str, Any]
        Per-mode support, closure, energy and descriptive AP results.
    """
    result = {}
    for position, name in enumerate(["focal", "repeated"]):
        difference = bank[position + 2].astype(np.float64) - bank[1].astype(np.float64)
        hull = np.zeros((12, 5000), dtype=bool)
        centers = [1250] if position == 0 else list(range(750, 4251, 500))
        for center in centers:
            hull[[1, 2, 3, 4, 5], center - 40 : center + 41] = True
        mask = support[position]
        energy = np.square(difference)
        result[name] = {
            "support_cells": int(mask.sum()),
            "support_fraction": float(mask.mean()),
            "strict_cells": int(strict[position].sum()),
            "strict_support_fraction": float(strict[position].mean()),
            "outside_intended_hull_cells": int(np.count_nonzero(strict[position] & ~hull)),
            "strict_mask_exact": bool(np.array_equal(strict[position], difference != 0)),
            "threshold_mask_exact": bool(np.array_equal(mask, np.abs(difference) > 0.01)),
            "retained_energy_fraction": float(energy[mask].sum() / energy.sum()),
            "below_threshold_energy_fraction": float(energy[~mask].sum() / energy.sum()),
            "below_threshold_maximum_residual": float(np.abs(difference[~mask]).max()),
            "absolute_amplitude_ap": float(
                average_precision_score(mask.ravel(), np.abs(bank[position + 2]).ravel())
            ),
            "paired_difference_oracle_ap": float(
                average_precision_score(mask.ravel(), np.abs(difference).ravel())
            ),
        }
    return result


def primary_interval(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Bootstrap the fixed-seed paired coherence comparison.

    Parameters
    ----------
    rows : list[dict[str, Any]]
        One complete record per generator seed.

    Returns
    -------
    dict[str, Any]
        Mean difference and percentile interval conditional on this generator.
    """
    values = np.array([row["coherence"]["difference"] for row in rows])
    groups = patient_groups(np.arange(len(values)))
    rng = np.random.default_rng(2026103301)
    draws = [float(values[patient_resample(groups, rng)].mean()) for _ in range(2000)]
    low, high = np.percentile(draws, [2.5, 97.5])
    return {
        "mean_difference": float(values.mean()),
        "ci_low": float(low),
        "ci_high": float(high),
        "draws": 2000,
        "seed": 2026103301,
        "skipped_draws": 0,
        "unit": "generator_seed",
    }


def make_figures(
    destination: Path, raw: np.ndarray, variants: np.ndarray, masks: np.ndarray, rows: list[dict[str, Any]]
) -> list[str]:
    """Save predetermined waveform, intervention and quality figures locally.

    Parameters
    ----------
    destination : Path
        New local output directory.
    raw : np.ndarray
        Native fixed-seed bank.
    variants : np.ndarray
        All four standardized variants per seed.
    masks : np.ndarray
        Thresholded focal/repeated support masks.
    rows : list[dict[str, Any]]
        Saved per-seed measurements.

    Returns
    -------
    list[str]
        Figure filenames.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    paths = []
    times = np.arange(5000) / 500
    fig, axes = plt.subplots(8, 2, figsize=(20, 16), sharex=True)
    for index, pair in enumerate(axes):
        for lead in range(8):
            pair[0].plot(times, raw[index, lead] + lead * 0.25, linewidth=0.5)
        for lead in range(12):
            pair[1].plot(times, variants[index, 0, lead] + lead * 1.5, linewidth=0.5)
        pair[0].set_ylabel(str(rows[index]["seed"]))
    axes[-1, 0].set_xlabel("Seconds; native dimensionless values + offsets")
    axes[-1, 1].set_xlabel("Seconds; standardized robust scale units + offsets")
    fig.suptitle("Pulse2Pulse generated signals: no clinical labels")
    fig.tight_layout()
    fig.savefig(destination / "fixed_first_eight.png", dpi=130)
    plt.close(fig)
    paths.append("fixed_first_eight.png")
    fig, axes = plt.subplots(8, 3, figsize=(22, 20), sharex=True)
    for index in range(8):
        for mode, label in enumerate(["Focal", "Repeated"]):
            axis = axes[index, mode]
            axis.plot(times, variants[index, 1, 1], label="Nuisance baseline", linewidth=0.7)
            axis.plot(times, variants[index, mode + 2, 1], label=label + " edit", linewidth=0.7)
            axis.fill_between(times, -1, 1, where=masks[index, mode, 1], alpha=0.15)
            axis.set_ylabel("Robust scale units")
        axes[index, 2].imshow(masks[index, 1], aspect="auto", extent=(0, 10, 12, 0), interpolation="nearest")
        axes[index, 2].set_yticks(np.arange(12) + 0.5, LEADS)
    axes[0, 0].legend()
    axes[0, 1].legend()
    axes[-1, 2].set_xlabel("Seconds; repeated-edit support >0.01 units")
    fig.tight_layout()
    fig.savefig(destination / "known_support.png", dpi=130)
    plt.close(fig)
    paths.append("known_support.png")
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    native = [r["coherence"]["native_jaccard"] for r in rows]
    shifted = [r["coherence"]["shifted_jaccard"] for r in rows]
    axes[0].scatter(shifted, native, s=10)
    axes[0].plot([0, 1], [0, 1], color="black", linewidth=0.5)
    axes[0].set(xlabel="Shifted activity Jaccard", ylabel="Native activity Jaccard")
    axes[1].hist(np.array([r["standardization"]["q95_minus_q05"] for r in rows]).ravel(), bins=25)
    axes[1].set_xlabel("Native robust scales, independent leads")
    fractions = [
        v for r in rows for v in r["morphology"]["spectral_power_ratios"]["25_to_100"] if v is not None
    ]
    axes[2].hist(fractions, bins=25)
    axes[2].set_xlabel("Independent-lead power fraction 25–100 Hz / 0.5–100 Hz")
    fig.tight_layout()
    fig.savefig(destination / "quality_diagnostics.png", dpi=130)
    plt.close(fig)
    paths.append("quality_diagnostics.png")
    return paths


def save_array(path: Path, values: np.ndarray) -> None:
    """Save a numeric array while refusing overwrite.

    Parameters
    ----------
    path : Path
        New local artifact path.
    values : np.ndarray
        Non-object numerical values.
    """
    with path.open("xb") as stream:
        np.save(stream, values, allow_pickle=False)


def write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write one immutable JSON row per seed.

    Parameters
    ----------
    path : Path
        New local artifact path.
    rows : list[dict[str, Any]]
        Per-seed measurements.
    """
    with path.open("x") as stream:
        for row in rows:
            stream.write(json.dumps(row, allow_nan=False) + "\n")


def elapsed(clock: dict[str, float]) -> dict[str, float]:
    """Measure scientific stage CPU and monotonic wall costs.

    Parameters
    ----------
    clock : dict[str, float]
        Process CPU and monotonic wall start values.

    Returns
    -------
    dict[str, float]
        Elapsed seconds since the recorded starts.
    """
    return {
        "cpu_seconds": time.process_time() - clock["cpu"],
        "wall_seconds": time.perf_counter() - clock["wall"],
    }
