"""Arm membership and the family-weighted logistic readout for Experiment 022b.

Every arm of 022b fits 022's readout (a train-only scaler and an L2 logistic head, ``C=0.01``) on PTB-XL
training plus a different set of Challenge training rows. Unweighted arms use
``external_readout.fit_logistic_c`` unchanged; the family-balanced arm uses the weighted fit here, which
reduces to it with unit weights. The same membership rule picks each arm's calibration ECGs.
"""

from __future__ import annotations

import warnings

import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from .external_readout import Head, fit_logistic_c
from .screening_threshold import calibration_fit

READOUT_C = 0.01
MAX_ITERATIONS = 5000
CHALLENGE_FAMILIES = ("chapman_ningbo", "georgia", "cpsc")
ARMS = ("ptbxl", "pooled", "balanced", "ptbxl_chapman_ningbo", "ptbxl_ningbo",
        *(f"loso_{family}" for family in CHALLENGE_FAMILIES))
WEIGHTED_ARMS = ("balanced",)


def arm_members(families: np.ndarray, sources: np.ndarray) -> dict[str, np.ndarray]:
    """
    Rows of each arm, for training or calibration alike.

    Parameters
    ----------
    families : np.ndarray
        Family of each row: ``ptbxl`` or a Challenge family.
    sources : np.ndarray
        Source of each row, such as ``ningbo``.

    Returns
    -------
    dict[str, np.ndarray]
        Boolean selection per name in ``ARMS``.
    """
    ptbxl = families == "ptbxl"
    everything = np.ones(len(families), dtype=bool)
    members = {"ptbxl": ptbxl, "pooled": everything, "balanced": everything,
               "ptbxl_chapman_ningbo": ptbxl | (families == "chapman_ningbo"),
               "ptbxl_ningbo": ptbxl | (sources == "ningbo")}
    for family in CHALLENGE_FAMILIES:
        members[f"loso_{family}"] = families != family
    return members


def fit_weighted_logistic(x: np.ndarray, y: np.ndarray, weights: np.ndarray) -> Head:
    """
    Fit 022's scaler and L2 logistic head with sample weights in both.

    Parameters
    ----------
    x : np.ndarray
        Training inputs.
    y : np.ndarray
        Binary targets.
    weights : np.ndarray
        Sample weight of each row.

    Returns
    -------
    Head
        Fitted scaler and classifier.

    Raises
    ------
    RuntimeError
        If the solver does not converge.
    """
    x = np.asarray(x, dtype=np.float64)
    scaler = StandardScaler().fit(x, sample_weight=weights)
    model = LogisticRegression(C=READOUT_C, penalty="l2", fit_intercept=True, solver="lbfgs",
                               max_iter=MAX_ITERATIONS, tol=1e-8, class_weight=None, random_state=42)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ConvergenceWarning)
        model.fit(scaler.transform(x), y, sample_weight=weights)
    if any(issubclass(item.category, ConvergenceWarning) for item in caught) or \
            model.n_iter_[0] >= MAX_ITERATIONS:
        raise RuntimeError("Logistic fit did not converge")
    return scaler, model


def fit_readout(x: np.ndarray, y: np.ndarray, weights: np.ndarray | None = None) -> Head:
    """
    Fit one arm's readout for one encoder.

    Parameters
    ----------
    x : np.ndarray
        Training inputs.
    y : np.ndarray
        Binary targets.
    weights : np.ndarray | None
        Sample weights, or ``None`` for 022's unweighted fit.

    Returns
    -------
    Head
        Fitted scaler and classifier.
    """
    if weights is None:
        return fit_logistic_c(x, y, READOUT_C)
    return fit_weighted_logistic(x, y, weights)


def probability_quality(y: np.ndarray, probabilities: np.ndarray) -> dict[str, float]:
    """
    Brier score and calibration intercept and slope of probabilities.

    Parameters
    ----------
    y : np.ndarray
        Binary labels with both classes.
    probabilities : np.ndarray
        Probabilities in (0, 1).

    Returns
    -------
    dict[str, float]
        ``brier``, ``intercept`` and ``slope``.
    """
    brier = float(np.mean((probabilities - y) ** 2))
    return {"brier": brier, **calibration_fit(y, probabilities)}
