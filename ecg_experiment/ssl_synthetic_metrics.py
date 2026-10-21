"""Frozen full-map engineering metrics and paired latent-subject intervals."""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import average_precision_score

from ecg_experiment.intervals import patient_groups, patient_resample


def positive_map_metrics(support: np.ndarray, scores: np.ndarray) -> dict[str, float]:
    """
    Score exact intervention support over every native lead/time cell.

    Parameters
    ----------
    support : np.ndarray
        Boolean full-map intervention support with both classes present.
    scores : np.ndarray
        Finite scores aligned with the support map.

    Returns
    -------
    dict
        AP, prevalence, and expected-tie precision/recall in the highest-scoring 1% area.
    """
    truth = np.asarray(support, dtype=bool).ravel()
    values = np.asarray(scores).ravel()
    if support.shape != scores.shape or not np.isfinite(values).all():
        raise ValueError("Support and finite scores must share a native grid")
    positives = int(truth.sum())
    if not 0 < positives < len(truth):
        raise ValueError("Positive cases must have both support and background cells")
    area = max(1, int(np.ceil(0.01 * len(values))))
    cutoff = np.partition(values, len(values) - area)[-area]
    above = values > cutoff
    tied = values == cutoff
    fraction = (area - int(above.sum())) / int(tied.sum())
    captured = float(truth[above].sum() + fraction * truth[tied].sum())
    return {
        "average_precision": float(average_precision_score(truth, values)),
        "prevalence": positives / len(truth),
        "top_one_percent_precision": captured / area,
        "top_one_percent_recall": captured / positives,
    }


def paired_subject_difference(
    subject_ids: np.ndarray,
    first: np.ndarray,
    second: np.ndarray,
    draws: int = 2000,
    seed: int = 20261019,
) -> dict[str, Any]:
    """
    Bootstrap paired differences after equally weighting variants within latent subjects.

    Parameters
    ----------
    subject_ids : np.ndarray
        Subject identity per positive case, shared across methods.
    first, second : np.ndarray
        Finite case-level metrics, such as AP or control flags.
    draws : int
        Number of whole-subject bootstrap draws.
    seed : int
        Dedicated paired resampling seed.

    Returns
    -------
    dict
        Subject-macro mean difference and percentile interval.
    """
    ids = np.asarray(subject_ids)
    first, second = np.asarray(first), np.asarray(second)
    if ids.ndim != 1 or first.shape != ids.shape or second.shape != ids.shape or draws < 1:
        raise ValueError("Expected aligned nonempty case vectors and positive draws")
    if not np.isfinite(first).all() or not np.isfinite(second).all():
        raise ValueError("Unscorable cases cannot be silently removed")
    groups = patient_groups(ids)
    differences = np.array([np.mean(first[group] - second[group]) for group in groups])
    unique_groups = patient_groups(np.arange(len(groups)))
    rng = np.random.default_rng(seed)
    values = [float(differences[patient_resample(unique_groups, rng)].mean()) for _ in range(draws)]
    low, high = np.percentile(values, [2.5, 97.5], method="linear")
    return {
        "difference": float(differences.mean()),
        "ci_low": float(low),
        "ci_high": float(high),
        "subjects": len(groups),
        "cases": len(ids),
        "draws": draws,
        "skipped_draws": 0,
        "seed": seed,
    }


def control_threshold(sham: np.ndarray, nuisance: np.ndarray) -> float:
    """
    Fit the conservative validation-only maximum-map control threshold.

    Parameters
    ----------
    sham, nuisance : np.ndarray
        Maximum full-map probabilities on paired control variants for each subject.

    Returns
    -------
    float
        Higher empirical 95th percentile; evaluation flags use strict greater-than.
    """
    sham, nuisance = np.asarray(sham), np.asarray(nuisance)
    if sham.ndim != 1 or not len(sham) or sham.shape != nuisance.shape:
        raise ValueError("Expected aligned validation control vectors")
    if not np.isfinite(sham).all() or not np.isfinite(nuisance).all():
        raise ValueError("Nonfinite validation control score")
    return float(np.quantile(np.maximum(sham, nuisance), 0.95, method="higher"))
