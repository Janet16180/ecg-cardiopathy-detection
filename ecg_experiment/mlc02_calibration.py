"""
Logistic recalibration fits of MLC-02 by damped Newton, beside the frozen scoring library.

``mlc02_evaluation.logistic_fit`` takes full Newton steps from the null model (slope 0). On scores that
separate the classes well, the first steps overshoot and the fit is reported as not converged (NaN) although
the maximum-likelihood estimate exists: on the MLC-02 validation predictions this happened for 13 of the 140
pooled candidate-condition Platt fits. That library is pinned by the Stage 2 receipts and is not changed.

``damped_logistic_fit`` maximizes the same likelihood, ``logit P(y = 1) = a + b x``, with Newton steps
halved until the log-likelihood does not decrease, starting from the identity ``(a, b) = (0, 1)``.
Where the library converges, both give the same estimate (tested to 1e-8). It returns NaN only when no step
improves the fit within ``MAX_ITERATIONS``, as under perfect separation, where no finite estimate exists.

- ``fit_platt_damped``: per-condition Platt calibration on validation (Stage 3 ``fix``).
- ``calibration_report``: joint-fit calibration intercept and slope of each condition with patient-cluster
  bootstrap intervals on the shared resamples of ``mlc02_bootstrap.cluster_resamples``; reported beside the
  library's values, which stay as computed.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from ecg_experiment.mlc02_bootstrap import CHUNK, labels_digest, percentile_interval
from ecg_experiment.mlc02_data import CONDITIONS
from ecg_experiment.mlc02_evaluation import check_scores, logit, sigmoid_inplace

MAX_ITERATIONS = 200
MAX_HALVINGS = 40
TOLERANCE = 1e-10
IDENTITY = (0.0, 1.0)


def log_likelihood(
    y: np.ndarray, x: np.ndarray, weights: np.ndarray, a: np.ndarray, b: np.ndarray
) -> np.ndarray:
    """
    Weighted logistic log-likelihood of ``a + b x`` for many fits.

    Parameters
    ----------
    y : np.ndarray
        Binary targets of ``m`` records.
    x : np.ndarray
        Predictor of the same records.
    weights : np.ndarray
        ``[fits, m]`` record weights.
    a, b : np.ndarray
        Intercepts and slopes, one per fit.

    Returns
    -------
    np.ndarray
        One log-likelihood per fit.
    """
    eta = b[:, None] * x + a[:, None]
    return np.einsum("ij,ij->i", weights, y * eta - np.logaddexp(0.0, eta))


def damped_logistic_fit(
    y: np.ndarray, x: np.ndarray, weights: np.ndarray | None = None, start: tuple[float, float] = IDENTITY
) -> tuple[np.ndarray, np.ndarray]:
    """
    Unpenalized logistic regression of ``y`` on ``a + b x`` by Newton's method with step halving.

    Parameters
    ----------
    y : np.ndarray
        Binary targets of ``m`` records.
    x : np.ndarray
        Predictor of the same records.
    weights : np.ndarray | None
        ``[fits, m]`` nonnegative weights (bootstrap multiplicities); None is one fit with unit weights.
    start : tuple[float, float]
        Starting intercept and slope of every fit.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        Intercepts and slopes, one per fit; NaN where the full Newton step did not fall below ``TOLERANCE``
        within ``MAX_ITERATIONS`` iterations, or no halving of a larger step improved the likelihood.
    """
    y, x = np.asarray(y, dtype=np.float64), np.asarray(x, dtype=np.float64)
    weights = np.ones((1, len(y))) if weights is None else np.asarray(weights, dtype=np.float64)
    fits = len(weights)
    a, b = np.full(fits, float(start[0])), np.full(fits, float(start[1]))
    converged = np.zeros(fits, dtype=bool)
    rows = np.arange(fits)
    with np.errstate(all="ignore"):
        current = log_likelihood(y, x, weights, a, b)
        for _ in range(MAX_ITERATIONS):
            part = weights[rows]
            mu = np.multiply(b[rows, None], x)
            mu += a[rows, None]
            sigmoid_inplace(mu)
            curvature = mu * (1.0 - mu) * part
            residual = (y - mu) * part
            g0, g1 = residual.sum(axis=1), residual @ x
            h00, h01, h11 = curvature.sum(axis=1), curvature @ x, curvature @ (x * x)
            determinant = h00 * h11 - h01 * h01
            step_a = (h11 * g0 - h01 * g1) / determinant
            step_b = (h00 * g1 - h01 * g0) / determinant
            scale = np.ones(len(rows))
            accepted = np.zeros(len(rows), dtype=bool)
            for _ in range(MAX_HALVINGS):
                pending = ~accepted
                trial = log_likelihood(
                    y,
                    x,
                    part[pending],
                    a[rows[pending]] + scale[pending] * step_a[pending],
                    b[rows[pending]] + scale[pending] * step_b[pending],
                )
                better = np.isfinite(trial) & (
                    trial >= current[rows[pending]] - 1e-12 * np.abs(current[rows[pending]])
                )
                index = np.flatnonzero(pending)
                accepted[index[better]] = True
                current[rows[index[better]]] = trial[better]
                scale[index[~better]] *= 0.5
                if accepted.all():
                    break
            a[rows[accepted]] += (scale * step_a)[accepted]
            b[rows[accepted]] += (scale * step_b)[accepted]
            finite = np.isfinite(step_a) & np.isfinite(step_b)
            done = finite & (np.maximum(np.abs(step_a), np.abs(step_b)) < TOLERANCE)
            converged[rows[done]] = True
            rows = rows[finite & accepted & ~done]
            if not len(rows):
                break
    a[~converged], b[~converged] = np.nan, np.nan
    return a, b


def fit_platt_damped(labels: np.ndarray, probability: np.ndarray) -> np.ndarray:
    """
    Per-condition Platt calibration ``sigmoid(a + b logit(p))`` by maximum likelihood, damped Newton.

    Parameters
    ----------
    labels : np.ndarray
        ``[records, conditions]`` states; each condition's defined entries are used.
    probability : np.ndarray
        Aligned probabilities.

    Returns
    -------
    np.ndarray
        ``[conditions, 2]`` intercepts and slopes, applicable with ``mlc02_evaluation.apply_platt``.

    Raises
    ------
    ValueError
        If a fit does not converge.
    """
    labels, probability = check_scores(labels, probability)
    params = np.empty((labels.shape[1], 2))
    for column in range(labels.shape[1]):
        defined = ~np.isnan(labels[:, column])
        intercept, slope = damped_logistic_fit(labels[defined, column], logit(probability[defined, column]))
        params[column] = intercept[0], slope[0]
    if not np.isfinite(params).all():
        raise ValueError("a Platt fit did not converge")
    return params


def calibration_report(
    labels: np.ndarray,
    probability: np.ndarray,
    resamples: dict[str, Any] | None = None,
    conditions: tuple[str, ...] = CONDITIONS,
) -> tuple[dict[str, Any], np.ndarray | None]:
    """
    Joint-fit calibration intercept and slope per condition, damped Newton, with bootstrap intervals.

    Resampled fits start at the observed estimate. Intervals are given for the conditions that enter the
    set's macro, as in ``mlc02_bootstrap.interval_report``; resamples whose fit did not converge are left out
    and counted.

    Parameters
    ----------
    labels : np.ndarray
        ``[records, conditions]`` states of one set.
    probability : np.ndarray
        Aligned probabilities.
    resamples : dict[str, Any] | None
        Output of ``mlc02_bootstrap.cluster_resamples`` for this set; None gives point estimates only.
    conditions : tuple[str, ...]
        Condition names in column order.

    Returns
    -------
    tuple[dict[str, Any], np.ndarray | None]
        The report: ``conditions`` (per condition ``calibration_intercept``, ``calibration_slope``, and with
        resamples their ``_ci_low``, ``_ci_high`` and ``_nan_draws``) and the resampling ``fingerprint``; and
        the resampled values ``[draws, conditions, 2]`` (intercept, slope; NaN for conditions that do not
        enter), or None without resamples.

    Raises
    ------
    ValueError
        If the resamples were drawn for another set or other labels.
    """
    labels, probability = check_scores(labels, probability)
    if resamples is not None and (
        resamples["weights"].shape[1] != len(labels) or labels_digest(labels) != resamples["labels_sha256"]
    ):
        raise ValueError("resamples were drawn for another set or other labels")
    rows = []
    samples = None if resamples is None else np.full((len(resamples["weights"]), len(conditions), 2), np.nan)
    for column, name in enumerate(conditions):
        defined = ~np.isnan(labels[:, column])
        y, x = labels[defined, column], logit(probability[defined, column])
        row: dict[str, Any] = {
            "condition": name,
            "calibration_intercept": math.nan,
            "calibration_slope": math.nan,
        }
        if 0 < y.sum() < len(y):
            intercept, slope = damped_logistic_fit(y, x)
            row |= {"calibration_intercept": float(intercept[0]), "calibration_slope": float(slope[0])}
        if resamples is not None and resamples["entering"][column]:
            start = (
                (row["calibration_intercept"], row["calibration_slope"])
                if math.isfinite(row["calibration_slope"])
                else IDENTITY
            )
            weights = resamples["weights"]
            for low in range(0, len(weights), CHUNK):
                block = weights[low : low + CHUNK][:, defined].astype(np.float64)
                fitted = damped_logistic_fit(y, x, block, start)
                samples[low : low + CHUNK, column, 0], samples[low : low + CHUNK, column, 1] = fitted
            for index, metric in enumerate(("calibration_intercept", "calibration_slope")):
                lower, upper, missing = percentile_interval(samples[:, column, index])
                row |= {
                    f"{metric}_ci_low": float(lower),
                    f"{metric}_ci_high": float(upper),
                    f"{metric}_nan_draws": int(missing),
                }
        rows.append(row)
    return {
        "conditions": rows,
        "fingerprint": None if resamples is None else resamples["fingerprint"],
    }, samples
