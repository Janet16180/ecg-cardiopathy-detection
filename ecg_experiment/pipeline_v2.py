"""Saved heads and paired patient-bootstrap summaries of the candidate screening pipeline, for Experiment 037.

A head is a fitted ``StandardScaler`` and binary ``LogisticRegression``; its parameters are stored as plain
arrays so that a later final test can score new ECGs without refitting. Intervals follow
``ecg_experiment.intervals``: whole-patient draws with patients sorted by ``np.unique``.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.special import expit

from .intervals import patient_groups, patient_resample

PARAMETERS = ("mean", "scale", "coef", "intercept")


def head_parameters(head: tuple[Any, Any], prefix: str) -> dict[str, np.ndarray]:
    """
    Plain arrays that define a fitted scaler and binary logistic head.

    Parameters
    ----------
    head : tuple[Any, Any]
        Fitted ``StandardScaler`` and ``LogisticRegression``.
    prefix : str
        Name of the head, prefixed to every key.

    Returns
    -------
    dict[str, np.ndarray]
        ``{prefix}_mean``, ``{prefix}_scale``, ``{prefix}_coef`` (one row) and ``{prefix}_intercept``.
    """
    scaler, model = head
    values = {"mean": scaler.mean_, "scale": scaler.scale_, "coef": model.coef_[0],
              "intercept": model.intercept_}
    return {f"{prefix}_{name}": np.asarray(values[name], dtype=np.float64).copy() for name in PARAMETERS}


def score_parameters(parameters: dict[str, np.ndarray], prefix: str, x: np.ndarray) -> np.ndarray:
    """
    Positive-class probabilities of a head from its saved parameters.

    Parameters
    ----------
    parameters : dict[str, np.ndarray]
        Arrays written by ``head_parameters`` (possibly with other heads).
    prefix : str
        Name of the head.
    x : np.ndarray
        ``(n, d)`` features.

    Returns
    -------
    np.ndarray
        One probability per row.

    Raises
    ------
    ValueError
        If the feature width differs from the head's.
    """
    mean, scale = parameters[f"{prefix}_mean"], parameters[f"{prefix}_scale"]
    x = np.asarray(x, dtype=np.float64)
    if x.ndim != 2 or x.shape[1] != len(mean):
        raise ValueError(f"Features need {len(mean)} columns")
    standardized = (x - mean) / scale
    return expit(standardized @ parameters[f"{prefix}_coef"] + parameters[f"{prefix}_intercept"][0])


def resample_counts(patients: np.ndarray, draws: int, seed: int) -> np.ndarray:
    """
    How often each ECG enters each whole-patient bootstrap draw.

    Parameters
    ----------
    patients : np.ndarray
        Patient ID of each ECG.
    draws : int
        Number of draws.
    seed : int
        Seed of ``numpy.random.default_rng``; one ``intervals.patient_resample`` per draw, in order.

    Returns
    -------
    np.ndarray
        ``(draws, n)`` float counts; an ECG enters as often as its patient is drawn.
    """
    groups = patient_groups(patients)
    rng = np.random.default_rng(seed)
    counts = np.empty((draws, len(patients)), dtype=np.float64)
    for draw in range(draws):
        counts[draw] = np.bincount(patient_resample(groups, rng), minlength=len(patients))
    return counts


def resampled_rates(counts: np.ndarray, shares: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """
    Referral rate of one outcome in every bootstrap draw, for several screens at once.

    Parameters
    ----------
    counts : np.ndarray
        ``(draws, n)`` output of ``resample_counts``.
    shares : np.ndarray
        ``(n, keys)`` share of the local-normal draws that refer each ECG, per screen.
    mask : np.ndarray
        ECGs with the outcome.

    Returns
    -------
    np.ndarray
        ``(draws, keys)`` rates; NaN in a draw without an ECG of the outcome.
    """
    mask = np.asarray(mask, dtype=np.float64)
    denominator = counts @ mask
    numerator = counts @ (shares * mask[:, None])
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(denominator[:, None] > 0, numerator / denominator[:, None], np.nan)


def percentile_interval(values: np.ndarray) -> list[float]:
    """
    Return the 2.5th and 97.5th percentiles over the draws where a value is defined.

    Parameters
    ----------
    values : np.ndarray
        One value per draw, NaN where undefined.

    Returns
    -------
    list[float]
        Lower and upper bounds.

    Raises
    ------
    ValueError
        If no draw is defined.
    """
    values = np.asarray(values, dtype=np.float64)
    if np.isnan(values).all():
        raise ValueError("No bootstrap draw is defined")
    return [float(value) for value in np.nanpercentile(values, [2.5, 97.5])]


def v2_is_worse(interval: list[float], margin: float) -> bool:
    """
    Whether a v2 minus v1 interval lies entirely below ``-margin``.

    Parameters
    ----------
    interval : list[float]
        Lower and upper bounds of the difference.
    margin : float
        Positive margin.

    Returns
    -------
    bool
        True when the upper bound is below ``-margin``.
    """
    return bool(interval[1] < -margin)
