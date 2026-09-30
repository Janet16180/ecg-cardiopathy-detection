"""Thresholds set by a referral budget on normal ECG scores, for Experiment 030.

A budget b flags a fixed share of normal ECGs: the threshold is the empirical (1 - b) quantile of a set of
normal scores, without interpolation, and an ECG is referred when its score is strictly above it.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import norm


def budget_threshold(normal_scores: np.ndarray, per_mille: int) -> float:
    """
    Threshold that leaves ``floor(b m)`` of ``m`` normal scores strictly above it.

    This is the empirical ``1 - b`` quantile without interpolation (Hyndman and Fan type 1): the
    ``(k + 1)``-th highest score with ``k = floor(b m)``, computed in integers.

    Parameters
    ----------
    normal_scores : np.ndarray
        Scores of the normal ECGs; higher means more likely abnormal.
    per_mille : int
        Budget ``b`` in thousandths (50 for 5%).

    Returns
    -------
    float
        The threshold.

    Raises
    ------
    ValueError
        If the budget is outside ``[0, 1000)`` or there are no scores.
    """
    if not 0 <= per_mille < 1000:
        raise ValueError(f"Budget out of range: {per_mille}")
    scores = np.sort(np.asarray(normal_scores, dtype=np.float64))[::-1]
    if not len(scores):
        raise ValueError("No normal scores")
    return float(scores[per_mille * len(scores) // 1000])


def count_above(sorted_scores: np.ndarray, thresholds: np.ndarray) -> np.ndarray:
    """
    Count the scores strictly above each threshold.

    Parameters
    ----------
    sorted_scores : np.ndarray
        Scores sorted in increasing order.
    thresholds : np.ndarray
        Thresholds.

    Returns
    -------
    np.ndarray
        Count of scores above each threshold.
    """
    return len(sorted_scores) - np.searchsorted(sorted_scores, thresholds, side="right")


def referral_share(scores: np.ndarray, thresholds: np.ndarray) -> np.ndarray:
    """
    Share of the thresholds each score lies strictly above: how often each ECG is referred over draws.

    Parameters
    ----------
    scores : np.ndarray
        Scores of the ECGs.
    thresholds : np.ndarray
        One threshold per draw.

    Returns
    -------
    np.ndarray
        Referral share of each ECG.
    """
    return np.searchsorted(np.sort(thresholds), scores, side="left") / len(thresholds)


def bootstrap_counts(size: int, resamples: int, seed: int) -> np.ndarray:
    """
    How often each of ``size`` groups is drawn in each resample of the groups with replacement.

    Parameters
    ----------
    size : int
        Number of groups ``G`` (patients).
    resamples : int
        Number of resamples.
    seed : int
        Seed of ``numpy.random.default_rng``; one ``integers(0, G, G)`` call per resample, in order.

    Returns
    -------
    np.ndarray
        ``(resamples, G)`` array of counts; each row sums to ``G``.
    """
    rng = np.random.default_rng(seed)
    counts = np.empty((resamples, size), dtype=np.float64)
    for resample in range(resamples):
        counts[resample] = np.bincount(rng.integers(0, size, size), minlength=size)
    return counts


def group_sums(codes: np.ndarray, values: np.ndarray, groups: int) -> np.ndarray:
    """
    Sum of the rows of ``values`` within each group.

    Parameters
    ----------
    codes : np.ndarray
        Group code of each row.
    values : np.ndarray
        ``(rows, columns)`` values.
    groups : int
        Number of groups.

    Returns
    -------
    np.ndarray
        ``(groups, columns)`` sums.
    """
    sums = np.zeros((groups, values.shape[1]), dtype=np.float64)
    np.add.at(sums, codes, values)
    return sums


def wilson_interval(successes: int, total: int, level: float = 0.95) -> tuple[float, float]:
    """
    Wilson score interval of a binomial proportion.

    Parameters
    ----------
    successes : int
        Number of successes.
    total : int
        Number of trials, at least one.
    level : float
        Coverage.

    Returns
    -------
    tuple[float, float]
        Lower and upper bounds.
    """
    z = norm.ppf(0.5 + level / 2)
    share = successes / total
    centre = (share + z * z / (2 * total)) / (1 + z * z / total)
    half = z * np.sqrt(share * (1 - share) / total + z * z / (4 * total * total)) / (1 + z * z / total)
    return float(centre - half), float(centre + half)


def referrals_per_1000(sensitivity: np.ndarray, rate: np.ndarray, prevalence: float) -> np.ndarray:
    """
    Referrals per 1,000 ECGs at an assumed prevalence.

    Parameters
    ----------
    sensitivity : np.ndarray
        Share of abnormal ECGs referred.
    rate : np.ndarray
        Share of normal ECGs referred.
    prevalence : float
        Assumed share of abnormal ECGs.

    Returns
    -------
    np.ndarray
        ``1000 (prevalence sensitivity + (1 - prevalence) rate)``.
    """
    return 1000 * (prevalence * np.asarray(sensitivity) + (1 - prevalence) * np.asarray(rate))
