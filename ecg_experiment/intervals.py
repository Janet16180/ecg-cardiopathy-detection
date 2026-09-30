"""
Whole-patient bootstrap intervals shared by new experiments.

Convention, fixed for every function here:

- patients are sorted by ``np.unique`` and one draw is ``rng.integers(0, P, P)`` on a fresh
  ``numpy.random.default_rng(seed)``, keeping every ECG of a drawn patient, in draw order;
- draws whose ECGs hold a single class are skipped and counted as ``skipped_draws``, never
  silently dropped; if every draw is skipped a ``ValueError`` is raised;
- intervals are the 2.5 and 97.5 percentiles (``np.percentile``, linear) of the valid draws;
- the default is 2,000 draws.

For the same seed and draws, ``paired_auroc_difference`` reproduces
``full_development.patient_bootstrap`` bit for bit (its ``invalid_draws`` is ``skipped_draws``
here), and ``two_class_resamples`` reproduces ``normal_manifold.patient_resamples``.
``metric_intervals`` reproduces the matching bounds of ``evaluation.patient_bootstrap``, which
defaults to 500 draws, also reports precision and does not count skipped draws.
``screening_threshold.patient_rates`` gets the same sensitivity and specificity draws from
weighted patient counts.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

DRAWS = 2000
SEED = 2026
METRICS = ("auroc", "average_precision", "sensitivity", "specificity", "brier")


def patient_groups(patient_ids: np.ndarray) -> list[np.ndarray]:
    """
    Record positions of each patient, in O(n log n).

    Parameters
    ----------
    patient_ids : np.ndarray
        One-dimensional patient ID of each record.

    Returns
    -------
    list[np.ndarray]
        One ascending array of record positions per patient, in ``np.unique`` patient order.

    Raises
    ------
    ValueError
        If ``patient_ids`` is empty or not one-dimensional.
    """
    patient_ids = np.asarray(patient_ids)
    if patient_ids.ndim != 1 or not len(patient_ids):
        raise ValueError("patient_ids must be a nonempty one-dimensional array")
    _, codes = np.unique(patient_ids, return_inverse=True)
    order = np.argsort(codes, kind="stable")
    return np.split(order, np.cumsum(np.bincount(codes))[:-1])


def patient_resample(groups: list[np.ndarray], rng: np.random.Generator) -> np.ndarray:
    """
    Record positions of one whole-patient bootstrap draw.

    Parameters
    ----------
    groups : list[np.ndarray]
        Output of ``patient_groups``.
    rng : np.random.Generator
        Generator; one ``integers(0, P, P)`` call is consumed.

    Returns
    -------
    np.ndarray
        Positions of every record of each drawn patient, in draw order.
    """
    drawn = rng.integers(0, len(groups), len(groups))
    return np.concatenate([groups[index] for index in drawn])


def two_class_resamples(
    groups: list[np.ndarray], y: np.ndarray, draws: int, rng: np.random.Generator
) -> Iterator[np.ndarray]:
    """
    Whole-patient draws that contain both classes.

    Single-class draws are skipped; the caller counts them as ``draws`` minus the number yielded.

    Parameters
    ----------
    groups : list[np.ndarray]
        Output of ``patient_groups``.
    y : np.ndarray
        Binary targets of each record.
    draws : int
        Number of draws taken from ``rng``, including skipped ones.
    rng : np.random.Generator
        Generator of the draws.

    Yields
    ------
    np.ndarray
        Record positions of each valid draw.
    """
    for _ in range(draws):
        rows = patient_resample(groups, rng)
        positives = np.count_nonzero(y[rows])
        if 0 < positives < len(rows):
            yield rows


def _aligned(*arrays: np.ndarray) -> list[np.ndarray]:
    """
    Convert inputs to arrays and check that they are aligned.

    Parameters
    ----------
    *arrays : np.ndarray
        Arrays that describe the same ECGs.

    Returns
    -------
    list[np.ndarray]
        The inputs as NumPy arrays.

    Raises
    ------
    ValueError
        If any input is not one-dimensional or the lengths differ.
    """
    arrays = [np.asarray(array) for array in arrays]
    if any(array.ndim != 1 or len(array) != len(arrays[0]) for array in arrays):
        raise ValueError("inputs must be one-dimensional arrays of equal length")
    return arrays


def _percentiles(values: list[float], draws: int) -> tuple[float, float]:
    """
    Return the 2.5 and 97.5 percentiles of the valid draws.

    Parameters
    ----------
    values : list[float]
        One value per valid draw.
    draws : int
        Number of draws taken, for the error message.

    Returns
    -------
    tuple[float, float]
        Lower and upper bounds.

    Raises
    ------
    ValueError
        If no draw was valid.
    """
    if not values:
        raise ValueError(f"all {draws} bootstrap draws had a single class")
    low, high = np.percentile(values, [2.5, 97.5])
    return float(low), float(high)


def paired_auroc_difference(
    patients: np.ndarray,
    y: np.ndarray,
    first: np.ndarray,
    second: np.ndarray,
    draws: int = DRAWS,
    seed: int = SEED,
) -> dict[str, float | int]:
    """
    Paired whole-patient bootstrap of an AUROC difference, first minus second.

    Parameters
    ----------
    patients : np.ndarray
        Patient ID of each ECG.
    y : np.ndarray
        Binary targets.
    first, second : np.ndarray
        Scores of the two models on the same ECGs.
    draws : int
        Number of bootstrap draws, including skipped ones.
    seed : int
        Seed of ``numpy.random.default_rng``.

    Returns
    -------
    dict[str, float | int]
        Observed ``difference``, ``ci_low`` and ``ci_high`` (2.5 and 97.5 percentiles),
        ``draws``, ``skipped_draws`` (single-class draws) and ``seed``.

    Raises
    ------
    ValueError
        If the inputs are misaligned, ``y`` has a single class, or every draw is skipped.
    """
    patients, y, first, second = _aligned(patients, y, first, second)
    rng = np.random.default_rng(seed)
    differences = [
        roc_auc_score(y[rows], first[rows]) - roc_auc_score(y[rows], second[rows])
        for rows in two_class_resamples(patient_groups(patients), y, draws, rng)
    ]
    low, high = _percentiles(differences, draws)
    return {
        "difference": float(roc_auc_score(y, first) - roc_auc_score(y, second)),
        "ci_low": low,
        "ci_high": high,
        "draws": draws,
        "skipped_draws": draws - len(differences),
        "seed": seed,
    }


def threshold_metrics(y: np.ndarray, prob: np.ndarray, threshold: float) -> dict[str, float]:
    """
    AUROC, average precision, sensitivity, specificity and Brier score.

    Parameters
    ----------
    y : np.ndarray
        Binary targets with both classes.
    prob : np.ndarray
        Predicted probabilities.
    threshold : float
        ECGs with ``prob >= threshold`` are predicted positive.

    Returns
    -------
    dict[str, float]
        One value per name in ``METRICS``.
    """
    positive = y == 1
    predicted = prob >= threshold
    return {
        "auroc": float(roc_auc_score(y, prob)),
        "average_precision": float(average_precision_score(y, prob)),
        "sensitivity": float(np.count_nonzero(predicted & positive) / np.count_nonzero(positive)),
        "specificity": float(np.count_nonzero(~predicted & ~positive) / np.count_nonzero(~positive)),
        "brier": float(np.mean((prob - y) ** 2)),
    }


def metric_intervals(
    y: np.ndarray,
    prob: np.ndarray,
    patients: np.ndarray,
    threshold: float,
    draws: int = DRAWS,
    seed: int = SEED,
) -> dict[str, Any]:
    """
    Whole-patient bootstrap intervals of the metrics at a fixed threshold.

    Parameters
    ----------
    y : np.ndarray
        Binary targets.
    prob : np.ndarray
        Predicted probabilities.
    patients : np.ndarray
        Patient ID of each ECG.
    threshold : float
        ECGs with ``prob >= threshold`` are predicted positive.
    draws : int
        Number of bootstrap draws, including skipped ones.
    seed : int
        Seed of ``numpy.random.default_rng``.

    Returns
    -------
    dict[str, Any]
        ``metrics`` maps each name in ``METRICS`` to its observed ``value``, ``ci_low`` and
        ``ci_high``; also ``threshold``, ``draws``, ``skipped_draws`` and ``seed``.

    Raises
    ------
    ValueError
        If the inputs are misaligned, ``y`` has a single class, or every draw is skipped.
    """
    y, prob, patients = _aligned(y, prob, patients)
    observed = threshold_metrics(y, prob, threshold)
    rng = np.random.default_rng(seed)
    resampled = [
        threshold_metrics(y[rows], prob[rows], threshold)
        for rows in two_class_resamples(patient_groups(patients), y, draws, rng)
    ]
    summary = {}
    for name in METRICS:
        low, high = _percentiles([values[name] for values in resampled], draws)
        summary[name] = {"value": observed[name], "ci_low": low, "ci_high": high}
    return {
        "metrics": summary,
        "threshold": float(threshold),
        "draws": draws,
        "skipped_draws": draws - len(resampled),
        "seed": seed,
    }
