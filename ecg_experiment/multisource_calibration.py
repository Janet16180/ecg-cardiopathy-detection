"""Weighted Platt calibration, weighted thresholds and paired bootstrap rates for Experiment 027b.

Every arm of 027b fits a Platt mapping and an at-least-95%-sensitivity threshold per head on a different
calibration pool. Unweighted arms use the 027 functions unchanged; the family-balanced arm uses the
weighted versions here, which reduce to them with unit weights.
"""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression

from .evaluation import CALIBRATION_BINS, TARGET_SENSITIVITY
from .screening_threshold import calibrate, fit_platt, interval, screening_threshold

RATES = ("sensitivity", "specificity", "brier", "ece_10_bins", "deviation")


def family_weights(families: np.ndarray) -> np.ndarray:
    """
    Record weights that give every family the same total weight and average 1.

    Parameters
    ----------
    families : np.ndarray
        Family name of each record.

    Returns
    -------
    np.ndarray
        ``N / (F * n_f)`` for a record of family ``f``, with ``N`` records and ``F`` families.
    """
    names, codes, sizes = np.unique(families, return_inverse=True, return_counts=True)
    return len(families) / (len(names) * sizes[codes])


def fit_weighted_platt(logits: np.ndarray, y: np.ndarray, weights: np.ndarray) -> LogisticRegression:
    """
    Fit Platt scaling with sample weights, otherwise as ``screening_threshold.fit_platt``.

    Parameters
    ----------
    logits : np.ndarray
        Head logits of the calibration ECGs.
    y : np.ndarray
        Binary labels.
    weights : np.ndarray
        Sample weight of each ECG.

    Returns
    -------
    LogisticRegression
        Fitted one-feature calibrator.

    Raises
    ------
    RuntimeError
        If the fitted slope is not positive.
    """
    calibrator = LogisticRegression(C=1e6, solver="lbfgs", max_iter=1000)
    calibrator.fit(np.asarray(logits, dtype=np.float64).reshape(-1, 1), y, sample_weight=weights)
    if calibrator.coef_[0, 0] <= 0:
        raise RuntimeError("Nonpositive calibration slope")
    return calibrator


def weighted_threshold(y: np.ndarray, calibrated: np.ndarray, weights: np.ndarray,
                       target: float = TARGET_SENSITIVITY) -> float:
    """
    Highest calibrated probability at which the weighted share of positives referred reaches the target.

    Parameters
    ----------
    y : np.ndarray
        Binary labels.
    calibrated : np.ndarray
        Calibrated probabilities.
    weights : np.ndarray
        Weight of each ECG.
    target : float
        Required weighted share of positives with a probability at or above the threshold.

    Returns
    -------
    float
        Threshold; with unit weights it equals ``evaluation.select_threshold``.
    """
    positive = np.asarray(y) == 1
    order = np.argsort(calibrated[positive], kind="stable")
    values = calibrated[positive][order]
    share_above = np.cumsum(weights[positive][order][::-1])[::-1] / weights[positive].sum()
    return float(values[np.flatnonzero(share_above >= target - 1e-12)[-1]])


def fit_arm(logits: np.ndarray, y: np.ndarray, weights: np.ndarray | None = None,
            ) -> tuple[LogisticRegression, float]:
    """
    Fit one arm's Platt mapping and threshold for one head.

    Parameters
    ----------
    logits : np.ndarray
        Head logits of the arm's calibration ECGs.
    y : np.ndarray
        Their binary labels.
    weights : np.ndarray | None
        Sample weights, or ``None`` for the unweighted 027 functions.

    Returns
    -------
    tuple[LogisticRegression, float]
        Calibrator and threshold on the calibrated probability.
    """
    if weights is None:
        calibrator = fit_platt(logits, y)
        return calibrator, screening_threshold(y, calibrate(calibrator, logits))
    calibrator = fit_weighted_platt(logits, y, weights)
    return calibrator, weighted_threshold(y, calibrate(calibrator, logits), weights)


def weighted_rates(weights: np.ndarray, y: np.ndarray, probabilities: np.ndarray,
                   thresholds: np.ndarray) -> dict[str, np.ndarray]:
    """
    Weighted sensitivity, specificity, Brier score, 10-bin ECE and deviation of several columns.

    Parameters
    ----------
    weights : np.ndarray
        Weight of each ECG, shape ``(n,)``.
    y : np.ndarray
        Binary labels, shape ``(n,)``.
    probabilities : np.ndarray
        Calibrated probabilities, shape ``(n, k)``, one column per arm and head.
    thresholds : np.ndarray
        Threshold of each column, shape ``(k,)``.

    Returns
    -------
    dict[str, np.ndarray]
        One ``(k,)`` array per name in ``RATES``; ``deviation`` is ``|sensitivity - 0.95|``. With unit
        weights the values equal ``screening_threshold.operating_point``'s.
    """
    positive = np.asarray(y) == 1
    referred = probabilities >= thresholds
    total = weights.sum()
    sensitivity = (weights * positive) @ referred / (weights @ positive)
    specificity = (weights * ~positive) @ ~referred / (weights @ ~positive)
    brier = weights @ (probabilities - positive[:, None]) ** 2 / total
    columns = probabilities.shape[1]
    bins = np.minimum((probabilities * CALIBRATION_BINS).astype(int), CALIBRATION_BINS - 1)
    slots = (bins + CALIBRATION_BINS * np.arange(columns)).ravel()
    repeated = np.repeat(weights, columns)
    size = CALIBRATION_BINS * columns
    observed = np.bincount(slots, weights=repeated * np.repeat(positive, columns), minlength=size)
    predicted = np.bincount(slots, weights=repeated * probabilities.ravel(), minlength=size)
    ece = np.abs(observed - predicted).reshape(columns, CALIBRATION_BINS).sum(axis=1) / total
    return {"sensitivity": sensitivity, "specificity": specificity, "brier": brier, "ece_10_bins": ece,
            "deviation": np.abs(sensitivity - TARGET_SENSITIVITY)}


def bootstrap_rates(units: np.ndarray, y: np.ndarray, probabilities: np.ndarray, thresholds: np.ndarray,
                    draws: int, seed: int) -> dict[str, np.ndarray]:
    """
    Rates of every column in paired bootstrap draws over units.

    Each draw resamples units (patients, or records) with replacement and gives every ECG its unit's count,
    exactly as ``screening_threshold.patient_rates`` draws. The same draws serve every column. Draws
    without a positive or without a negative are skipped.

    Parameters
    ----------
    units : np.ndarray
        Unit of each ECG.
    y : np.ndarray
        Binary labels.
    probabilities : np.ndarray
        Calibrated probabilities, shape ``(n, k)``.
    thresholds : np.ndarray
        Threshold per column.
    draws : int
        Number of draws.
    seed : int
        Seed of a fresh generator.

    Returns
    -------
    dict[str, np.ndarray]
        One ``(valid_draws, k)`` array per name in ``RATES``.
    """
    unique, codes = np.unique(units, return_inverse=True)
    positive = np.asarray(y) == 1
    rng = np.random.default_rng(seed)
    values = {name: [] for name in RATES}
    for _ in range(draws):
        counts = np.bincount(rng.integers(0, len(unique), len(unique)), minlength=len(unique))
        weights = counts[codes].astype(np.float64)
        if not weights @ positive or not weights @ ~positive:
            continue
        for name, rate in weighted_rates(weights, y, probabilities, thresholds).items():
            values[name].append(rate)
    return {name: np.asarray(rates) for name, rates in values.items()}


def summarize(columns: list[tuple[str, str]], observed: dict[str, np.ndarray], drawn: dict[str, np.ndarray],
              reference: str) -> dict[str, object]:
    """
    Observed rates with intervals per column, and every arm's paired difference from the reference arm.

    Parameters
    ----------
    columns : list[tuple[str, str]]
        ``(arm, head)`` of each column.
    observed : dict[str, np.ndarray]
        ``weighted_rates`` with unit weights.
    drawn : dict[str, np.ndarray]
        ``bootstrap_rates`` output.
    reference : str
        Arm the differences are taken from.

    Returns
    -------
    dict[str, object]
        ``rates[arm][head][rate]`` and ``differences[arm][head][rate]``, each with ``value`` (or
        ``difference``), ``ci_low`` and ``ci_high``, and ``valid_draws``.
    """
    position = {column: index for index, column in enumerate(columns)}
    rates, differences = {}, {}
    for (arm, head), index in position.items():
        rates.setdefault(arm, {})[head] = {
            name: {"value": float(observed[name][index]), **interval(drawn[name][:, index])}
            for name in RATES}
        if arm == reference:
            continue
        base = position[(reference, head)]
        differences.setdefault(arm, {})[head] = {
            name: {"difference": float(observed[name][index] - observed[name][base]),
                   **interval(drawn[name][:, index] - drawn[name][:, base])}
            for name in RATES}
    return {"rates": rates, "differences": differences, "valid_draws": len(drawn["sensitivity"])}


def verdict(difference: dict[str, float]) -> str:
    """
    Read a deviation difference by its interval.

    Parameters
    ----------
    difference : dict[str, float]
        ``ci_low`` and ``ci_high`` of an arm's deviation minus the reference arm's.

    Returns
    -------
    str
        ``better`` if the interval lies below 0, ``worse`` if above 0, otherwise ``not_distinguished``.
    """
    result = "not_distinguished"
    if difference["ci_high"] < 0:
        result = "better"
    elif difference["ci_low"] > 0:
        result = "worse"
    return result

