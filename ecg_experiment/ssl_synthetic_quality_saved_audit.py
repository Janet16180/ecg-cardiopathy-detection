"""Independent numerical reconstruction of dimensionless generator quality fixtures."""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.signal import periodogram

from ecg_experiment.intervals import patient_groups, patient_resample
from ecg_experiment.ssl_synthetic_pretrained_saved_audit import require


def canonical(values: np.ndarray) -> np.ndarray:
    """Build canonical leads from independent numerical values.

    Parameters
    ----------
    values : np.ndarray
        Eight float32 leads.

    Returns
    -------
    np.ndarray
        Twelve canonical float32 leads.
    """
    result = np.empty((12, 5000), dtype=np.float32)
    result[[0, 1, 6, 7, 8, 9, 10, 11]] = values
    result[2] = values[1] - values[0]
    result[3] = -(values[0] + values[1]) * np.float32(0.5)
    result[4] = values[0] - values[1] * np.float32(0.5)
    result[5] = values[1] - values[0] * np.float32(0.5)
    return result


def reconstruct(raw: np.ndarray, index: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    """Independently construct standardized variants and exact supports.

    Parameters
    ----------
    raw : np.ndarray
        Native float32 eight-lead sample.
    index : int
        Frozen seed-bank index.

    Returns
    -------
    tuple
        Four variants, strict masks, thresholded masks and standardization metadata.
    """
    require(raw.dtype == np.float32 and raw.shape == (8, 5000), "native geometry")
    require(bool(np.isfinite(raw).all()), "finite native values")
    native = raw.astype(np.float64)
    center = np.quantile(native, 0.5, axis=1, method="linear")
    quantiles = np.quantile(native, [0.05, 0.95], axis=1, method="linear")
    scale = quantiles[1] - quantiles[0]
    require(bool(np.all(scale > 1e-6)), "raw robust scales")
    standardized = ((native - center[:, None]) / scale[:, None]).astype(np.float32)
    phase = np.random.default_rng(2026103100 + index).uniform(0, 2 * np.pi)
    nuisance = (0.10 * np.sin(2 * np.pi * 0.33 * (np.arange(5000) / 500) + phase)).astype(np.float32)
    baseline = standardized + nuisance
    result = [canonical(standardized), canonical(baseline)]
    for centers in ((1250,), tuple(range(750, 4251, 500))):
        independent = baseline.copy()
        for center_sample in centers:
            independent[1, center_sample - 40 : center_sample + 41] += (np.hanning(81) * 0.5).astype(
                np.float32
            )
        result.append(canonical(independent))
    variants = np.stack(result)
    difference = variants[2:].astype(np.float64) - variants[1].astype(np.float64)
    return (
        variants,
        difference != 0,
        np.abs(difference) > 0.01,
        {"median": center.tolist(), "q95_minus_q05": scale.tolist()},
    )


def average_precision(mask: np.ndarray, scores: np.ndarray) -> float:
    """Compute full-grid average precision using complete equal-score blocks.

    Parameters
    ----------
    mask : np.ndarray
        Boolean support.
    scores : np.ndarray
        Finite ranking scores.

    Returns
    -------
    float
        Average precision without interpolation.
    """
    labels = mask.ravel()
    values = scores.ravel()
    require(bool(np.isfinite(values).all()) and bool(labels.any()), "AP inputs")
    order = np.argsort(-values, kind="stable")
    sorted_scores = values[order]
    ends = np.r_[np.flatnonzero(sorted_scores[1:] != sorted_scores[:-1]), len(order) - 1]
    hits = np.cumsum(labels[order], dtype=np.int64)[ends]
    positives_per_block = np.diff(np.r_[0, hits])
    return float(np.sum(positives_per_block * hits / (ends + 1)) / labels.sum())


def intervention_values(bank: np.ndarray, strict: np.ndarray, support: np.ndarray) -> dict[str, Any]:
    """Recompute exact-support and amplitude-overlap measurements.

    Parameters
    ----------
    bank, strict, support : np.ndarray
        Four variants and two corresponding mask banks.

    Returns
    -------
    dict[str, Any]
        Both intervention-mode measurements.
    """
    rows = {}
    for index, mode in enumerate(("focal", "repeated")):
        difference = bank[index + 2].astype(np.float64) - bank[1].astype(np.float64)
        actual_strict = difference != 0
        actual_support = np.abs(difference) > 0.01
        require(np.array_equal(strict[index], actual_strict), "strict numerical support")
        require(np.array_equal(support[index], actual_support), "threshold numerical support")
        require(bool(actual_support.any()) and bool(actual_strict.any()), "nonempty supports")
        require(not bool(np.any(actual_support & ~actual_strict)), "support subset")
        hull = np.zeros_like(actual_strict)
        centers = [1250] if index == 0 else list(range(750, 4251, 500))
        for center in centers:
            for lead in (1, 2, 3, 4, 5):
                hull[lead, center - 40 : center + 41] = True
        outside = np.count_nonzero(actual_strict & ~hull)
        require(outside == 0, "changes outside constructed hull")
        squared = difference * difference
        energy = squared.sum()
        retained = float(squared[actual_support].sum() / energy)
        require(retained >= 0.99, "retained intervention energy")
        rows[mode] = {
            "support_cells": int(actual_support.sum()),
            "support_fraction": float(actual_support.mean()),
            "strict_cells": int(actual_strict.sum()),
            "strict_support_fraction": float(actual_strict.mean()),
            "outside_intended_hull_cells": int(outside),
            "strict_mask_exact": True,
            "threshold_mask_exact": True,
            "retained_energy_fraction": retained,
            "below_threshold_energy_fraction": float(squared[~actual_support].sum() / energy),
            "below_threshold_maximum_residual": float(np.max(np.abs(difference[~actual_support]))),
            "absolute_amplitude_ap": average_precision(actual_support, np.abs(bank[index + 2])),
            "paired_difference_oracle_ap": average_precision(actual_support, np.abs(difference)),
        }
    return rows


def coherence_values(independent: np.ndarray, index: int) -> dict[str, Any]:
    """Recompute paired native and circular-shift activity coincidence.

    Parameters
    ----------
    independent : np.ndarray
        Eight standardized independent leads.
    index : int
        Frozen bank position.

    Returns
    -------
    dict[str, Any]
        Descriptive paired timing-coincidence measurements.
    """
    derivative = np.diff(independent.astype(np.float64), axis=1, prepend=independent[:, :1])
    energy = np.array([np.convolve(row * row, np.full(20, 1 / 20), "same") for row in derivative])
    masks = energy > np.quantile(energy, 0.9, axis=1, method="linear")[:, None]
    shifts = np.random.default_rng(2026103200 + index).integers(1, 5000, 8)
    shifted = np.array([np.roll(masks[lead], shifts[lead]) for lead in range(8)])
    means = []
    for values in (masks, shifted):
        ratios = []
        for first in range(8):
            for second in range(first):
                union = np.logical_or(values[first], values[second]).sum()
                intersection = np.logical_and(values[first], values[second]).sum()
                ratios.append(float(intersection / union) if union else 0.0)
        means.append(float(np.mean(ratios)))
    return {
        "native_jaccard": means[0],
        "shifted_jaccard": means[1],
        "difference": means[0] - means[1],
        "shifts": shifts.tolist(),
    }


def primary_values(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Independently aggregate and seed-bootstrap the paired comparison.

    Parameters
    ----------
    rows : list[dict]
        Recomputed per-seed coherence rows.

    Returns
    -------
    dict[str, Any]
        Mean and percentile interval.
    """
    values = np.array([row["difference"] for row in rows])
    groups = patient_groups(np.arange(len(values)))
    rng = np.random.default_rng(2026103301)
    estimates = [values[patient_resample(groups, rng)].mean() for _ in range(2000)]
    low, high = np.quantile(estimates, [0.025, 0.975], method="linear")
    return {
        "mean_difference": float(values.mean()),
        "ci_low": float(low),
        "ci_high": float(high),
        "draws": 2000,
        "seed": 2026103301,
        "skipped_draws": 0,
        "unit": "generator_seed",
    }


def spectral_values(values: np.ndarray) -> tuple[dict[str, list[Any]], list[bool]]:
    """Recompute half-open periodogram power fractions and zero denominators.

    Parameters
    ----------
    values : np.ndarray
        Eight independent standardized leads.

    Returns
    -------
    tuple
        Power fractions by band and zero-denominator flags.
    """
    frequencies, spectrum = periodogram(
        values.astype(np.float64), fs=500, window="boxcar", detrend="constant", scaling="density", axis=1
    )
    denominator = spectrum[:, (frequencies >= 0.5) & (frequencies < 100)].sum(axis=1)
    bands = {}
    for name, low, high in (("0p5_to_5", 0.5, 5), ("5_to_25", 5, 25), ("25_to_100", 25, 100)):
        numerator = spectrum[:, (frequencies >= low) & (frequencies < high)].sum(axis=1)
        bands[name] = [
            float(top / bottom) if bottom else None
            for top, bottom in zip(numerator, denominator, strict=True)
        ]
    return bands, (denominator == 0).tolist()


def morphology_values(values: np.ndarray, raw: np.ndarray) -> dict[str, Any]:
    """Recompute the complete prospective numerical shape summaries.

    Parameters
    ----------
    values, raw : np.ndarray
        Independent standardized and native leads.

    Returns
    -------
    dict[str, Any]
        Spectral, activity and amplitude descriptions.
    """
    bands, zero = spectral_values(values)
    derivative = np.diff(values.astype(np.float64), axis=1, prepend=values[:, :1])
    energy = np.array([np.convolve(row * row, np.full(20, 1 / 20), "same") for row in derivative])
    active = energy > np.quantile(energy, 0.9, axis=1, method="linear")[:, None]
    return {
        "spectral_power_ratios": bands,
        "zero_power_denominator": zero,
        "raw_saturation_fraction": float(np.mean(np.abs(raw) >= 0.99)),
        "standardized_peak_to_peak": np.ptp(values, axis=1).tolist(),
        "native_mean": raw.mean(axis=1).tolist(),
        "native_rms": np.sqrt(np.mean(raw.astype(np.float64) ** 2, axis=1)).tolist(),
        "native_std": np.std(raw.astype(np.float64), axis=1).tolist(),
        "standardized_rms": np.sqrt(np.mean(values.astype(np.float64) ** 2, axis=1)).tolist(),
        "standardized_std": np.std(values.astype(np.float64), axis=1).tolist(),
        "activity_occupancy": active.mean(axis=1).tolist(),
    }


def diversity_values(variants: np.ndarray) -> dict[str, Any]:
    """Recompute all standardized independent-lead pairwise RMS distances.

    Parameters
    ----------
    variants : np.ndarray
        Four canonical variants per bank member.

    Returns
    -------
    dict[str, Any]
        Pair count and five fixed percentiles, without a rejection cutoff.
    """
    independent = variants[:, 0][:, [0, 1, 6, 7, 8, 9, 10, 11]].astype(np.float64)
    distances = []
    for first in range(len(independent)):
        for second in range(first):
            difference = independent[first] - independent[second]
            distances.append(float(np.sqrt(np.mean(difference * difference))))
    return {
        "pairs": len(distances),
        "percentiles_0_25_50_75_100": np.quantile(distances, [0, 0.25, 0.5, 0.75, 1]).tolist(),
        "near_duplicate_cutoff": None,
    }
