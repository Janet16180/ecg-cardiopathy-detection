"""Platt calibration, the at-least-95%-sensitivity threshold and screening metrics for Experiment 027."""

from __future__ import annotations

import numpy as np
from scipy.optimize import brentq
from scipy.special import expit, logit
from sklearn.linear_model import LogisticRegression

from .evaluation import TARGET_SENSITIVITY, metrics, scenario_ppv, select_threshold

PREVALENCES = (0.05, 0.01)


def head_logits(probabilities: np.ndarray) -> np.ndarray:
    """
    Logit of a head's positive-class probabilities.

    Parameters
    ----------
    probabilities : np.ndarray
        Probabilities in (0, 1).

    Returns
    -------
    np.ndarray
        Float64 logits.

    Raises
    ------
    ValueError
        If a probability is 0, 1 or not finite.
    """
    values = logit(np.asarray(probabilities, dtype=np.float64))
    if not np.isfinite(values).all():
        raise ValueError("Probabilities must lie strictly between 0 and 1")
    return values


def fit_platt(logits: np.ndarray, y: np.ndarray) -> LogisticRegression:
    """
    Fit Platt scaling as ``evaluation._fit_calibrator`` does.

    Parameters
    ----------
    logits : np.ndarray
        Head logits of the calibration ECGs.
    y : np.ndarray
        Binary labels of the calibration ECGs.

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
    calibrator.fit(np.asarray(logits, dtype=np.float64).reshape(-1, 1), y)
    if calibrator.coef_[0, 0] <= 0:
        raise RuntimeError("Nonpositive calibration slope")
    return calibrator


def calibrate(calibrator: LogisticRegression, logits: np.ndarray) -> np.ndarray:
    """
    Calibrated probabilities of head logits.

    Parameters
    ----------
    calibrator : LogisticRegression
        Output of ``fit_platt``.
    logits : np.ndarray
        Head logits.

    Returns
    -------
    np.ndarray
        Calibrated probabilities.
    """
    return calibrator.predict_proba(np.asarray(logits, dtype=np.float64).reshape(-1, 1))[:, 1]


def screening_threshold(y: np.ndarray, calibrated: np.ndarray) -> float:
    """
    Highest calibrated probability that refers at least 95% of the positives.

    Parameters
    ----------
    y : np.ndarray
        Binary labels of the calibration ECGs.
    calibrated : np.ndarray
        Their calibrated probabilities.

    Returns
    -------
    float
        Threshold; an ECG is referred when its calibrated probability is at or above it.
    """
    return select_threshold(y, calibrated, TARGET_SENSITIVITY)


def calibration_fit(y: np.ndarray, probabilities: np.ndarray) -> dict[str, float]:
    """
    Calibration intercept and slope of probabilities against labels.

    The intercept is ``a`` in ``logit P(y=1) = a + logit(p)`` and the slope is ``b`` in
    ``logit P(y=1) = c + b logit(p)``, both by maximum likelihood.

    Parameters
    ----------
    y : np.ndarray
        Binary labels with both classes.
    probabilities : np.ndarray
        Probabilities in (0, 1).

    Returns
    -------
    dict[str, float]
        ``intercept`` (ideal 0) and ``slope`` (ideal 1).
    """
    offset = head_logits(probabilities)
    intercept = brentq(lambda shift: float(np.sum(y - expit(shift + offset))), -50.0, 50.0, xtol=1e-12)
    model = LogisticRegression(C=np.inf, solver="lbfgs", max_iter=1000, tol=1e-10)
    model.fit(offset.reshape(-1, 1), y)
    return {"intercept": float(intercept), "slope": float(model.coef_[0, 0])}


def operating_point(y: np.ndarray, calibrated: np.ndarray, threshold: float) -> dict[str, object]:
    """
    Screening metrics of calibrated probabilities at a fixed threshold.

    Parameters
    ----------
    y : np.ndarray
        Binary labels with both classes.
    calibrated : np.ndarray
        Calibrated probabilities.
    threshold : float
        Referral threshold.

    Returns
    -------
    dict[str, object]
        Counts, sensitivity, specificity, PPV and NPV at the observed prevalence, false referrals per 1,000
        negatives, Brier score, 10-bin ECE, calibration intercept and slope, and the expected outcomes per
        1,000 people screened at each assumed prevalence in ``PREVALENCES``.
    """
    point = metrics(y, calibrated, threshold)
    return {
        "records": point["n"], "positives": point["tp"] + point["fn"], "prevalence": point["prevalence"],
        "tp": point["tp"], "fp": point["fp"], "tn": point["tn"], "fn": point["fn"],
        "sensitivity": point["sensitivity"], "specificity": point["specificity"],
        "ppv": point["precision"], "npv": point["npv"],
        "false_referrals_per_1000_negatives": 1000 * (1 - point["specificity"]),
        "referred_fraction": point["predicted_positive_rate"],
        "brier": point["brier"], "ece_10_bins": point["ece_10_bins"],
        "calibration": calibration_fit(y, calibrated),
        "assumed_prevalence": {str(prevalence): scenario_ppv(point["sensitivity"], point["specificity"],
                                                             prevalence)
                               for prevalence in PREVALENCES},
    }


def patient_rates(patients: np.ndarray, y: np.ndarray, referred: dict[str, np.ndarray], draws: int,
                  seed: int) -> dict[str, dict[str, np.ndarray]]:
    """
    Sensitivity and specificity of each head in paired whole-patient bootstrap draws.

    Every draw resamples patients with replacement and keeps all ECGs of a sampled patient; the same draws
    are used for every head. Draws without a positive or without a negative are skipped.

    Parameters
    ----------
    patients : np.ndarray
        Patient ID of each ECG.
    y : np.ndarray
        Binary labels.
    referred : dict[str, np.ndarray]
        Boolean referral decision of each ECG, keyed by head.
    draws : int
        Number of bootstrap draws.
    seed : int
        Seed of a fresh generator.

    Returns
    -------
    dict[str, dict[str, np.ndarray]]
        ``sensitivity`` and ``specificity`` per head, one value per valid draw.
    """
    unique, codes = np.unique(patients, return_inverse=True)
    positive = np.asarray(y) == 1
    positives = np.bincount(codes, weights=positive, minlength=len(unique))
    negatives = np.bincount(codes, weights=~positive, minlength=len(unique))
    hits = {name: np.bincount(codes, weights=flags & positive, minlength=len(unique))
            for name, flags in referred.items()}
    passes = {name: np.bincount(codes, weights=~flags & ~positive, minlength=len(unique))
              for name, flags in referred.items()}
    rng = np.random.default_rng(seed)
    rates = {"sensitivity": {name: [] for name in referred}, "specificity": {name: [] for name in referred}}
    for _ in range(draws):
        weights = np.bincount(rng.integers(0, len(unique), len(unique)), minlength=len(unique))
        total_positive, total_negative = weights @ positives, weights @ negatives
        if not total_positive or not total_negative:
            continue
        for name in referred:
            rates["sensitivity"][name].append(weights @ hits[name] / total_positive)
            rates["specificity"][name].append(weights @ passes[name] / total_negative)
    return {metric: {name: np.asarray(values) for name, values in heads.items()}
            for metric, heads in rates.items()}


def interval(values: np.ndarray) -> dict[str, float]:
    """
    Percentile 95% interval.

    Parameters
    ----------
    values : np.ndarray
        Bootstrap values.

    Returns
    -------
    dict[str, float]
        ``ci_low`` and ``ci_high``.
    """
    low, high = np.percentile(values, [2.5, 97.5])
    return {"ci_low": float(low), "ci_high": float(high)}


def bootstrap_summary(patients: np.ndarray, y: np.ndarray, referred: dict[str, np.ndarray],
                      contrasts: dict[str, tuple[str, str]], draws: int, seed: int) -> dict[str, object]:
    """
    Observed values and bootstrap intervals of each head's rates and of specificity differences.

    Parameters
    ----------
    patients : np.ndarray
        Patient ID of each ECG.
    y : np.ndarray
        Binary labels.
    referred : dict[str, np.ndarray]
        Boolean referral decision of each ECG, keyed by head.
    contrasts : dict[str, tuple[str, str]]
        Specificity differences to report, as first minus second head.
    draws : int
        Number of bootstrap draws.
    seed : int
        Seed of a fresh generator.

    Returns
    -------
    dict[str, object]
        ``heads`` (sensitivity and specificity with intervals), ``specificity_contrasts`` and the number of
        valid draws.
    """
    positive = np.asarray(y) == 1
    observed = {"sensitivity": {name: float(np.mean(flags[positive])) for name, flags in referred.items()},
                "specificity": {name: float(np.mean(~flags[~positive])) for name, flags in referred.items()}}
    rates = patient_rates(patients, y, referred, draws, seed)
    heads = {name: {metric: {"value": observed[metric][name], **interval(rates[metric][name])}
                    for metric in observed}
             for name in referred}
    differences = {
        name: {"difference": observed["specificity"][first] - observed["specificity"][second],
               **interval(rates["specificity"][first] - rates["specificity"][second])}
        for name, (first, second) in contrasts.items()}
    valid = len(next(iter(rates["sensitivity"].values())))
    return {"heads": heads, "specificity_contrasts": differences, "draws": draws, "valid_draws": valid,
            "seed": seed}
