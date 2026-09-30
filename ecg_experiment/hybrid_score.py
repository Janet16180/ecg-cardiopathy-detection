"""Hybrid screening scores from a supervised readout and a distance from normal, for Experiment 031.

Three rules combine a readout logit R and a log distance from normal D. ``zmean`` averages the two after
standardizing each on source normal ECGs. ``stack`` is a logistic regression on the standardized pair. The
OR rule refers an ECG when either score passes its own threshold, both thresholds taken at the same rank of
the same normal ECGs so that at most ``floor(b m)`` of ``m`` normals are referred.
"""

from __future__ import annotations

import warnings

import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression

from .referral_budget import budget_threshold

MAX_ITERATIONS = 5000


def normal_standardizer(normal_values: np.ndarray) -> tuple[float, float]:
    """
    Mean and standard deviation (ddof 0) of a score on normal ECGs.

    Parameters
    ----------
    normal_values : np.ndarray
        Score of each source normal ECG.

    Returns
    -------
    tuple[float, float]
        Mean and standard deviation.

    Raises
    ------
    ValueError
        If the standard deviation is not positive.
    """
    values = np.asarray(normal_values, dtype=np.float64)
    spread = float(values.std())
    if not spread > 0:
        raise ValueError("Normal scores have no spread")
    return float(values.mean()), spread


def standardize(values: np.ndarray, stats: tuple[float, float]) -> np.ndarray:
    """
    Standardize a score with fixed statistics.

    Parameters
    ----------
    values : np.ndarray
        Scores.
    stats : tuple[float, float]
        Mean and standard deviation from ``normal_standardizer``.

    Returns
    -------
    np.ndarray
        ``(values - mean) / sd``.
    """
    mean, spread = stats
    return (np.asarray(values, dtype=np.float64) - mean) / spread


def fit_stack(features: np.ndarray, y: np.ndarray) -> LogisticRegression:
    """
    Fit an unpenalized logistic regression on the standardized score pair.

    Parameters
    ----------
    features : np.ndarray
        ``(rows, 2)`` standardized readout and distance scores.
    y : np.ndarray
        Binary labels.

    Returns
    -------
    LogisticRegression
        The fitted model; its ``decision_function`` is the stack score.

    Raises
    ------
    RuntimeError
        If the solver does not converge.
    """
    model = LogisticRegression(penalty=None, solver="lbfgs", tol=1e-8, max_iter=MAX_ITERATIONS)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ConvergenceWarning)
        model.fit(np.asarray(features, dtype=np.float64), y)
    if any(issubclass(item.category, ConvergenceWarning) for item in caught) or \
            model.n_iter_[0] >= MAX_ITERATIONS:
        raise RuntimeError("Stack fit did not converge")
    return model


def either_thresholds(readout_normals: np.ndarray, distance_normals: np.ndarray,
                      per_mille: int) -> tuple[float, float]:
    """
    Thresholds of the OR rule: the same rank of both scores, referring at most ``floor(b m)`` normals.

    With ``k = floor(b m)``, the thresholds are the ``(j + 1)``-th highest readout and distance scores of
    the ``m`` normals, for the largest ``j`` in ``0..k`` at which at most ``k`` normals lie strictly above
    either threshold.

    Parameters
    ----------
    readout_normals : np.ndarray
        Readout scores of the normal ECGs.
    distance_normals : np.ndarray
        Distance scores of the same normal ECGs, in the same order.
    per_mille : int
        Budget ``b`` in thousandths.

    Returns
    -------
    tuple[float, float]
        Readout and distance thresholds.

    Raises
    ------
    ValueError
        If the two score arrays differ in length or the budget is out of range.
    """
    readout = np.asarray(readout_normals, dtype=np.float64)
    distance = np.asarray(distance_normals, dtype=np.float64)
    if len(readout) != len(distance):
        raise ValueError("Readout and distance normals differ in length")
    if not 0 <= per_mille < 1000:
        raise ValueError(f"Budget out of range: {per_mille}")
    limit = per_mille * len(readout) // 1000
    readout_sorted = np.sort(readout)[::-1]
    distance_sorted = np.sort(distance)[::-1]
    rank = limit
    while ((readout > readout_sorted[rank]) | (distance > distance_sorted[rank])).sum() > limit:
        rank -= 1
    return float(readout_sorted[rank]), float(distance_sorted[rank])


def rule_thresholds(normals: np.ndarray, per_mille: int) -> np.ndarray:
    """
    Thresholds of a single score (one column) or of the OR rule (two columns) on normal ECGs.

    Parameters
    ----------
    normals : np.ndarray
        ``(m,)`` scores of one rule, or ``(m, 2)`` readout and distance scores for the OR rule.
    per_mille : int
        Budget ``b`` in thousandths.

    Returns
    -------
    np.ndarray
        One threshold, or the readout and distance thresholds.
    """
    if normals.ndim == 1:
        return np.array([budget_threshold(normals, per_mille)])
    return np.array(either_thresholds(normals[:, 0], normals[:, 1], per_mille))


def referred(scores: np.ndarray, thresholds: np.ndarray) -> np.ndarray:
    """
    Which ECGs a rule refers: strictly above its threshold, or above either threshold for the OR rule.

    Parameters
    ----------
    scores : np.ndarray
        ``(n,)`` scores, or ``(n, 2)`` readout and distance scores.
    thresholds : np.ndarray
        Output of ``rule_thresholds``.

    Returns
    -------
    np.ndarray
        Boolean referral per ECG.
    """
    if scores.ndim == 1:
        return scores > thresholds[0]
    return (scores > thresholds).any(axis=1)
