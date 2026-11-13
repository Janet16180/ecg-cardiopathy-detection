"""
Point metrics, validation-fixed operating points, derived views, audits and the selection record of MLC-02.

The rules are those of the MLC-01 protocol (section "Evaluation rules (binding for MLC-02)") and of the
MLC-02 protocol (Stages 2 and 3). Labels are ``[records, conditions]`` matrices of 1.0, 0.0 and NaN
(undefined); probabilities are aligned ``[records, conditions]`` matrices.

- An undefined entry is left out of that condition's metrics and is never a negative.
- A condition enters a set's macro AUROC if the set has at least ``MIN_SUPPORT`` positives and as many
  negatives of it. Fewer than ``IMPRECISE_BELOW`` positives marks a condition imprecise.
- Metrics are computed within one evaluation set. The operating points are the exception the MLC-02
  protocol names: thresholds and Platt calibration are fitted on all validation records where the condition
  is defined.
- Calibration is the logistic recalibration ``logit P(y = 1) = a + b logit(p)``: ``calibration_intercept``
  is ``a`` and ``calibration_slope`` is ``b`` of that joint fit; ``calibration_in_the_large`` is ``a`` with
  ``b`` fixed at 1. Probabilities are clipped to ``[PROBABILITY_CLIP, 1 - PROBABILITY_CLIP]`` before the
  logit. A fit that does not converge (for example under perfect separation) gives NaN.
- A threshold ``t`` predicts positive when ``p >= t``.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np
from scipy.special import expit
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    log_loss,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler

from ecg_experiment.files import write_json_atomic, write_text_atomic
from ecg_experiment.mlc02_data import CONDITIONS, VALIDATION_SOURCES, check_probabilities, check_states

MIN_SUPPORT = 20
IMPRECISE_BELOW = 100
USABLE_AUROC_LOW = 0.80
TARGET_SPECIFICITY = 0.90
TIE_MARGIN = 0.001
CANDIDATES = ("P0", "L-J", "L-C", "G-M", "H-X", "H-J", "H-C", "H-All", "S-R", "F", "E")
PROBABILITY_CLIP = 1e-7
NEWTON_ITERATIONS = 50
NEWTON_TOLERANCE = 1e-10
SEED = 20261110
RANK_METRICS = ("auroc", "average_precision", "calibration_intercept", "calibration_slope",
                "calibration_in_the_large")


def check_scores(labels: np.ndarray, probability: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Check a label matrix and its aligned probabilities.

    Parameters
    ----------
    labels : np.ndarray
        ``[records, conditions]`` states 1.0, 0.0 or NaN.
    probability : np.ndarray
        ``[records, conditions]`` probabilities.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        Both as float64.

    Raises
    ------
    ValueError
        If the labels are not two-dimensional or hold other states, or the probabilities fail
        ``check_probabilities``.
    """
    labels = np.asarray(labels, dtype=np.float64)
    if labels.ndim != 2:
        raise ValueError("labels must be a [records, conditions] matrix")
    check_states(labels, "labels")
    return labels, check_probabilities(probability, *labels.shape)


def support(labels: np.ndarray) -> dict[str, np.ndarray]:
    """
    Positives, negatives and undefined entries per condition.

    Parameters
    ----------
    labels : np.ndarray
        ``[records, conditions]`` states.

    Returns
    -------
    dict[str, np.ndarray]
        ``positives``, ``negatives`` and ``undefined``, one count per condition.
    """
    labels = np.asarray(labels, dtype=np.float64)
    return {"positives": (labels == 1).sum(axis=0), "negatives": (labels == 0).sum(axis=0),
            "undefined": np.isnan(labels).sum(axis=0)}


def enters_macro(labels: np.ndarray, minimum: int = MIN_SUPPORT) -> np.ndarray:
    """
    Conditions that enter a set's macro AUROC.

    Parameters
    ----------
    labels : np.ndarray
        ``[records, conditions]`` states of one evaluation set.
    minimum : int
        Required positives and required negatives.

    Returns
    -------
    np.ndarray
        Boolean per condition: at least ``minimum`` positives and ``minimum`` negatives defined.
    """
    counts = support(labels)
    return (counts["positives"] >= minimum) & (counts["negatives"] >= minimum)


def auroc_by_condition(labels: np.ndarray, probability: np.ndarray) -> np.ndarray:
    """
    AUROC of each condition on its defined entries.

    Parameters
    ----------
    labels : np.ndarray
        ``[records, conditions]`` states.
    probability : np.ndarray
        Aligned probabilities.

    Returns
    -------
    np.ndarray
        One AUROC per condition; NaN where the defined entries hold a single class.
    """
    labels, probability = check_scores(labels, probability)
    result = np.full(labels.shape[1], np.nan)
    for column in range(labels.shape[1]):
        defined = ~np.isnan(labels[:, column])
        y = labels[defined, column]
        if 0 < y.sum() < len(y):
            result[column] = roc_auc_score(y, probability[defined, column])
    return result


def macro_auroc(labels: np.ndarray, probability: np.ndarray, minimum: int = MIN_SUPPORT) -> float:
    """
    Mean AUROC over the conditions that enter the macro.

    Parameters
    ----------
    labels : np.ndarray
        ``[records, conditions]`` states of one evaluation set.
    probability : np.ndarray
        Aligned probabilities.
    minimum : int
        Entry rule of ``enters_macro``.

    Returns
    -------
    float
        Macro AUROC; NaN if no condition enters.
    """
    entering = enters_macro(labels, minimum)
    if not entering.any():
        return math.nan
    return float(auroc_by_condition(labels, probability)[entering].mean())


def logit(probability: np.ndarray) -> np.ndarray:
    """
    Logit of clipped probabilities.

    Parameters
    ----------
    probability : np.ndarray
        Probabilities.

    Returns
    -------
    np.ndarray
        ``log(p / (1 - p))`` after clipping to ``[PROBABILITY_CLIP, 1 - PROBABILITY_CLIP]``.
    """
    p = np.clip(np.asarray(probability, dtype=np.float64), PROBABILITY_CLIP, 1 - PROBABILITY_CLIP)
    return np.log(p) - np.log1p(-p)


def sigmoid_inplace(z: np.ndarray) -> np.ndarray:
    """
    Logistic function ``1 / (1 + exp(-z))``, overwriting ``z``.

    Faster than ``scipy.special.expit`` on large blocks and equal to it within 1e-15 for float64.

    Parameters
    ----------
    z : np.ndarray
        Float64 array, overwritten.

    Returns
    -------
    np.ndarray
        ``z``, now holding the probabilities.
    """
    with np.errstate(over="ignore"):
        np.negative(z, out=z)
        np.exp(z, out=z)
    z += 1.0
    return np.reciprocal(z, out=z)


def start_values(y: np.ndarray, weights: np.ndarray,
                 start: tuple[float, float] | None) -> tuple[np.ndarray, ...]:
    """
    Return the starting intercept and slope of ``logistic_fit``.

    Parameters
    ----------
    y : np.ndarray
        Binary targets.
    weights : np.ndarray
        ``[fits, records]`` weights.
    start : tuple[float, float] | None
        Intercept and slope, or None for the null model (logit of the weighted prevalence, slope 0).

    Returns
    -------
    tuple[np.ndarray, ...]
        Intercepts and slopes, one per fit.
    """
    if start is not None:
        return np.full(len(weights), float(start[0])), np.full(len(weights), float(start[1]))
    with np.errstate(all="ignore"):
        prevalence = (weights @ y) / weights.sum(axis=1)
        return np.log(prevalence) - np.log1p(-prevalence), np.zeros(len(weights))


def logistic_fit(y: np.ndarray, x: np.ndarray, weights: np.ndarray | None = None,
                 start: tuple[float, float] | None = None) -> tuple[np.ndarray, np.ndarray]:
    """
    Unpenalized logistic regression of ``y`` on ``a + b x`` by Newton's method, for many weightings at once.

    Parameters
    ----------
    y : np.ndarray
        Binary targets of ``m`` records.
    x : np.ndarray
        Predictor of the same records.
    weights : np.ndarray | None
        ``[fits, m]`` nonnegative record weights (bootstrap multiplicities); None is one fit with unit
        weights.
    start : tuple[float, float] | None
        Starting intercept and slope; None starts at the null model.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        Intercepts ``a`` and slopes ``b``, one per fit; NaN where Newton's method did not converge within
        ``NEWTON_ITERATIONS`` steps to a step below ``NEWTON_TOLERANCE``.
    """
    y, x = np.asarray(y, dtype=np.float64), np.asarray(x, dtype=np.float64)
    weights = np.ones((1, len(y))) if weights is None else np.asarray(weights, dtype=np.float64)
    a, b = start_values(y, weights, start)
    design = np.column_stack([np.ones_like(x), x, x * x])
    rows = np.arange(len(weights))
    converged = np.zeros(len(weights), dtype=bool)
    with np.errstate(all="ignore"):
        for _ in range(NEWTON_ITERATIONS):
            part = weights if len(rows) == len(weights) else weights[rows]
            mu = np.multiply(b[rows, None], x)
            mu += a[rows, None]
            sigmoid_inplace(mu)
            curvature = np.subtract(1.0, mu)
            curvature *= mu
            curvature *= part
            residual = np.subtract(y, mu, out=mu)
            residual *= part
            (g0, g1), (h00, h01, h11) = (residual @ design[:, :2]).T, (curvature @ design).T
            determinant = h00 * h11 - h01 * h01
            step_a = (h11 * g0 - h01 * g1) / determinant
            step_b = (h00 * g1 - h01 * g0) / determinant
            a[rows] += step_a
            b[rows] += step_b
            finite = np.isfinite(step_a) & np.isfinite(step_b)
            done = finite & (np.maximum(np.abs(step_a), np.abs(step_b)) < NEWTON_TOLERANCE)
            converged[rows[done]] = True
            rows = rows[finite & ~done]
            if not len(rows):
                break
    a[~converged], b[~converged] = np.nan, np.nan
    return a, b


def offset_intercept(y: np.ndarray, x: np.ndarray, weights: np.ndarray | None = None,
                     start: float = 0.0) -> np.ndarray:
    """
    Calibration in the large: intercept of ``logit P(y = 1) = a + x``, slope fixed at 1.

    Parameters
    ----------
    y : np.ndarray
        Binary targets of ``m`` records.
    x : np.ndarray
        Logits of the predicted probabilities.
    weights : np.ndarray | None
        ``[fits, m]`` record weights; None is one fit with unit weights.
    start : float
        Starting intercept.

    Returns
    -------
    np.ndarray
        One intercept per fit; NaN where Newton's method did not converge.
    """
    y, x = np.asarray(y, dtype=np.float64), np.asarray(x, dtype=np.float64)
    weights = np.ones((1, len(y))) if weights is None else np.asarray(weights, dtype=np.float64)
    a = np.full(len(weights), float(start))
    rows = np.arange(len(weights))
    converged = np.zeros(len(weights), dtype=bool)
    with np.errstate(all="ignore"):
        for _ in range(NEWTON_ITERATIONS):
            part = weights if len(rows) == len(weights) else weights[rows]
            mu = np.add(a[rows, None], x)
            sigmoid_inplace(mu)
            curvature = np.subtract(1.0, mu)
            curvature *= mu
            residual = np.subtract(y, mu, out=mu)
            step = np.einsum("ij,ij->i", part, residual) / np.einsum("ij,ij->i", part, curvature)
            a[rows] += step
            finite = np.isfinite(step)
            done = finite & (np.abs(step) < NEWTON_TOLERANCE)
            converged[rows[done]] = True
            rows = rows[finite & ~done]
            if not len(rows):
                break
    a[~converged] = np.nan
    return a


def ranking_and_calibration(y: np.ndarray, p: np.ndarray) -> dict[str, float]:
    """
    AUROC, average precision and calibration of one condition with both classes defined.

    Parameters
    ----------
    y : np.ndarray
        Defined binary states.
    p : np.ndarray
        Probabilities of the same records.

    Returns
    -------
    dict[str, float]
        One value per name in ``RANK_METRICS``.
    """
    x = logit(p)
    intercept, slope = logistic_fit(y, x)
    return {"auroc": float(roc_auc_score(y, p)), "average_precision": float(average_precision_score(y, p)),
            "calibration_intercept": float(intercept[0]), "calibration_slope": float(slope[0]),
            "calibration_in_the_large": float(offset_intercept(y, x)[0])}


def threshold_rates(y: np.ndarray, p: np.ndarray, threshold: float) -> dict[str, float]:
    """
    Sensitivity and specificity at a fixed threshold.

    Parameters
    ----------
    y : np.ndarray
        Defined binary states.
    p : np.ndarray
        Probabilities of the same records.
    threshold : float
        ``p >= threshold`` is predicted positive.

    Returns
    -------
    dict[str, float]
        ``threshold``, ``sensitivity`` and ``specificity``; NaN where the class is absent.
    """
    predicted = p >= threshold
    positive, negative = y == 1, y == 0
    return {"threshold": float(threshold),
            "sensitivity": float(predicted[positive].mean()) if positive.any() else math.nan,
            "specificity": float((~predicted[negative]).mean()) if negative.any() else math.nan}


def condition_metrics(y: np.ndarray, p: np.ndarray, threshold: float | None = None) -> dict[str, Any]:
    """
    Every point metric of one condition within one evaluation set.

    Parameters
    ----------
    y : np.ndarray
        States 1.0, 0.0 or NaN of each record.
    p : np.ndarray
        Probabilities of the same records.
    threshold : float | None
        Validation-fixed threshold; None leaves sensitivity and specificity out.

    Returns
    -------
    dict[str, Any]
        ``positives``, ``negatives``, ``undefined``, ``prevalence`` (positives over defined records),
        ``enters_macro``, ``imprecise``, ``brier`` and the ``RANK_METRICS`` (NaN when a class is absent),
        and with a threshold ``threshold``, ``sensitivity`` and ``specificity``.
    """
    y, p = np.asarray(y, dtype=np.float64), np.asarray(p, dtype=np.float64)
    defined = ~np.isnan(y)
    y, p = y[defined], p[defined]
    positives = int(y.sum())
    negatives = len(y) - positives
    row: dict[str, Any] = {"positives": positives, "negatives": negatives, "undefined": int((~defined).sum()),
                           "prevalence": positives / len(y) if len(y) else math.nan,
                           "enters_macro": bool(positives >= MIN_SUPPORT and negatives >= MIN_SUPPORT),
                           "imprecise": bool(positives < IMPRECISE_BELOW),
                           "brier": float(np.mean((p - y) ** 2)) if len(y) else math.nan}
    row |= ranking_and_calibration(y, p) if positives and negatives else dict.fromkeys(RANK_METRICS, math.nan)
    if threshold is not None:
        row |= threshold_rates(y, p, threshold)
    return row


def metrics_table(labels: np.ndarray, probability: np.ndarray, thresholds: np.ndarray | None = None,
                  conditions: tuple[str, ...] = CONDITIONS) -> dict[str, Any]:
    """
    Per-condition metrics and the macro AUROC of one evaluation set.

    Parameters
    ----------
    labels : np.ndarray
        ``[records, conditions]`` states of one set.
    probability : np.ndarray
        Aligned probabilities.
    thresholds : np.ndarray | None
        Validation-fixed thresholds, one per condition.
    conditions : tuple[str, ...]
        Condition names in column order.

    Returns
    -------
    dict[str, Any]
        ``records``, ``conditions`` (one ``condition_metrics`` row per condition, with its name),
        ``entering`` (names) and ``macro_auroc`` (NaN if none enters).
    """
    labels, probability = check_scores(labels, probability)
    if len(conditions) != labels.shape[1]:
        raise ValueError("condition names do not match the label columns")
    cuts = [None] * len(conditions) if thresholds is None else [float(value) for value in thresholds]
    rows = [{"condition": name, **condition_metrics(labels[:, column], probability[:, column], cuts[column])}
            for column, name in enumerate(conditions)]
    entering = [row for row in rows if row["enters_macro"]]
    return {"records": len(labels), "conditions": rows, "entering": [row["condition"] for row in entering],
            "macro_auroc": float(np.mean([row["auroc"] for row in entering])) if entering else math.nan}


def selection_metric(labels: np.ndarray, probability: np.ndarray, sources: np.ndarray,
                     conditions: tuple[str, ...] = CONDITIONS) -> dict[str, Any]:
    """
    Compute the MLC-02 selection metric: mean over validation sources of the within-source macro AUROC.

    A source where no condition enters the macro is left out of the mean and listed in
    ``excluded_sources``; entry depends on labels only, so the same sources count for every model. The
    per-condition variant averages a condition's within-source AUROC over the sources where it enters.

    Parameters
    ----------
    labels : np.ndarray
        ``[records, conditions]`` states of validation records.
    probability : np.ndarray
        Aligned probabilities.
    sources : np.ndarray
        Manifest source of each record; every one must be a validation source.
    conditions : tuple[str, ...]
        Condition names in column order.

    Returns
    -------
    dict[str, Any]
        ``selection_metric``, ``sources``, ``macro_auroc_by_source``, ``entering_by_source``,
        ``excluded_sources``, ``per_condition`` (name to mean AUROC, NaN if it enters no source) and
        ``per_condition_sources``.

    Raises
    ------
    ValueError
        If a source is not a validation source or no source has an entering condition.
    """
    labels, probability = check_scores(labels, probability)
    sources = np.asarray(sources, dtype=str)
    unknown = set(sources) - set(VALIDATION_SOURCES)
    if unknown:
        raise ValueError(f"selection uses validation sources only, got {sorted(unknown)}")
    names = [source for source in VALIDATION_SOURCES if source in set(sources)]
    aurocs = {source: auroc_by_condition(labels[sources == source], probability[sources == source])
              for source in names}
    entering = {source: enters_macro(labels[sources == source]) for source in names}
    used = [source for source in names if entering[source].any()]
    if not used:
        raise ValueError("no validation source has a condition that enters the macro")
    by_source = {source: float(aurocs[source][entering[source]].mean()) for source in used}
    per_condition_sources = {name: [source for source in names if entering[source][column]]
                             for column, name in enumerate(conditions)}
    per_condition = {name: float(np.mean([aurocs[source][column] for source in per_condition_sources[name]]))
                     if per_condition_sources[name] else math.nan for column, name in enumerate(conditions)}
    return {"selection_metric": float(np.mean(list(by_source.values()))), "sources": names,
            "macro_auroc_by_source": by_source,
            "entering_by_source": {source: [conditions[c] for c in np.flatnonzero(entering[source])]
                                   for source in names},
            "excluded_sources": [source for source in names if source not in used],
            "per_condition": per_condition, "per_condition_sources": per_condition_sources}


def specificity_threshold(y: np.ndarray, p: np.ndarray, target: float = TARGET_SPECIFICITY) -> float:
    """
    Lowest observed score whose specificity on these records is at least ``target``.

    This is the operating point of highest sensitivity among thresholds at the observed scores with
    specificity at least ``target``; with ``p >= t`` positive, at least ``ceil(target * negatives)``
    negatives lie strictly below ``t``.

    Parameters
    ----------
    y : np.ndarray
        States 1.0, 0.0 or NaN; undefined entries are left out.
    p : np.ndarray
        Probabilities of the same records.
    target : float
        Required specificity.

    Returns
    -------
    float
        The threshold; ``inf`` if only predicting every record negative reaches the target.

    Raises
    ------
    ValueError
        If no negative is defined.
    """
    y, p = np.asarray(y, dtype=np.float64), np.asarray(p, dtype=np.float64)
    defined = ~np.isnan(y)
    y, p = y[defined], p[defined]
    negatives = np.sort(p[y == 0])
    if not len(negatives):
        raise ValueError("a specificity threshold needs defined negatives")
    needed = math.ceil(round(target * len(negatives), 9))
    if needed == 0:
        return float(p.min())
    above = p[p > negatives[needed - 1]]
    return float(above.min()) if len(above) else math.inf


def specificity_thresholds(labels: np.ndarray, probability: np.ndarray,
                           target: float = TARGET_SPECIFICITY) -> np.ndarray:
    """
    Per-condition ``specificity_threshold`` on validation records, pooled over validation sources.

    Parameters
    ----------
    labels : np.ndarray
        ``[records, conditions]`` states of all validation records.
    probability : np.ndarray
        Aligned probabilities.
    target : float
        Required specificity.

    Returns
    -------
    np.ndarray
        One threshold per condition.
    """
    labels, probability = check_scores(labels, probability)
    return np.array([specificity_threshold(labels[:, c], probability[:, c], target)
                     for c in range(labels.shape[1])])


def fit_platt(labels: np.ndarray, probability: np.ndarray) -> np.ndarray:
    """
    Per-condition Platt calibration ``sigmoid(a + b logit(p))`` fitted by maximum likelihood on validation.

    Parameters
    ----------
    labels : np.ndarray
        ``[records, conditions]`` states of all validation records; each condition's defined entries are used.
    probability : np.ndarray
        Aligned probabilities.

    Returns
    -------
    np.ndarray
        ``[conditions, 2]`` intercepts ``a`` and slopes ``b``.

    Raises
    ------
    ValueError
        If a fit does not converge.
    """
    labels, probability = check_scores(labels, probability)
    params = np.empty((labels.shape[1], 2))
    for column in range(labels.shape[1]):
        defined = ~np.isnan(labels[:, column])
        intercept, slope = logistic_fit(labels[defined, column], logit(probability[defined, column]))
        params[column] = intercept[0], slope[0]
    if not np.isfinite(params).all():
        raise ValueError("a Platt fit did not converge")
    return params


def apply_platt(probability: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Apply validation-fitted Platt calibration to another set.

    Parameters
    ----------
    probability : np.ndarray
        ``[records, conditions]`` probabilities.
    params : np.ndarray
        Output of ``fit_platt``.

    Returns
    -------
    np.ndarray
        Calibrated probabilities.
    """
    probability = check_probabilities(probability, len(probability), len(params))
    return expit(params[:, 0] + params[:, 1] * logit(probability))


def apply_thresholds(probability: np.ndarray, thresholds: np.ndarray) -> np.ndarray:
    """
    Predicted condition sets at validation-fixed thresholds.

    Parameters
    ----------
    probability : np.ndarray
        ``[records, conditions]`` probabilities.
    thresholds : np.ndarray
        One threshold per condition.

    Returns
    -------
    np.ndarray
        Boolean ``[records, conditions]``, ``p >= threshold``.
    """
    probability = check_probabilities(probability, len(probability), len(thresholds))
    return probability >= np.asarray(thresholds, dtype=np.float64)[None, :]


def single_condition_mask(labels: np.ndarray, require_all_defined: bool = True) -> np.ndarray:
    """
    Mark records with exactly one positive condition.

    Parameters
    ----------
    labels : np.ndarray
        ``[records, conditions]`` states.
    require_all_defined : bool
        Also require every condition to be defined (the conservative reading).

    Returns
    -------
    np.ndarray
        Boolean per record.
    """
    labels = np.asarray(labels, dtype=np.float64)
    mask = (labels == 1).sum(axis=1) == 1
    return mask & ~np.isnan(labels).any(axis=1) if require_all_defined else mask


def top_k_accuracy(labels: np.ndarray, probability: np.ndarray, k: int, require_all_defined: bool = True,
                   conditions: tuple[str, ...] = CONDITIONS) -> dict[str, Any]:
    """
    Share of single-condition records whose condition is among the ``k`` highest probabilities.

    All conditions are ranked, undefined ones included; ties go to the earlier condition.

    Parameters
    ----------
    labels : np.ndarray
        ``[records, conditions]`` states.
    probability : np.ndarray
        Aligned probabilities.
    k : int
        Number of named conditions.
    require_all_defined : bool
        Reading of ``single_condition_mask``.
    conditions : tuple[str, ...]
        Condition names in column order.

    Returns
    -------
    dict[str, Any]
        ``records``, ``accuracy`` (NaN without records), and ``records_by_condition`` and
        ``accuracy_by_condition`` keyed by the named condition.
    """
    labels, probability = check_scores(labels, probability)
    mask = single_condition_mask(labels, require_all_defined)
    named = np.argmax(labels[mask] == 1, axis=1)
    ranked = np.argsort(-probability[mask], axis=1, kind="stable")[:, :k]
    hit = (ranked == named[:, None]).any(axis=1)
    by_condition = {name: hit[named == column] for column, name in enumerate(conditions)}
    return {"records": int(mask.sum()), "accuracy": float(hit.mean()) if len(hit) else math.nan,
            "records_by_condition": {name: int(len(values)) for name, values in by_condition.items()},
            "accuracy_by_condition": {name: float(values.mean()) if len(values) else math.nan
                                      for name, values in by_condition.items()}}


def exact_set_accuracy(labels: np.ndarray, probability: np.ndarray, thresholds: np.ndarray) -> dict[str, Any]:
    """
    Share of records whose predicted condition set equals the labelled one.

    Parameters
    ----------
    labels : np.ndarray
        ``[records, conditions]`` states.
    probability : np.ndarray
        Aligned probabilities.
    thresholds : np.ndarray
        Validation-fixed thresholds.

    Returns
    -------
    dict[str, Any]
        ``records`` and ``accuracy`` over all records, where a record matches if prediction and label agree
        on every defined condition (undefined entries ignored); ``records_all_defined`` and
        ``accuracy_all_defined`` over records with all conditions defined.
    """
    labels, probability = check_scores(labels, probability)
    defined = ~np.isnan(labels)
    match = ((apply_thresholds(probability, thresholds) == (labels == 1)) | ~defined).all(axis=1)
    complete = defined.all(axis=1)
    return {"records": len(labels), "accuracy": float(match.mean()) if len(match) else math.nan,
            "records_all_defined": int(complete.sum()),
            "accuracy_all_defined": float(match[complete].mean()) if complete.any() else math.nan}


def multiclass_view(labels: np.ndarray, probability: np.ndarray, thresholds: np.ndarray,
                    conditions: tuple[str, ...] = CONDITIONS) -> dict[str, Any]:
    """
    Compute the derived multiclass view: top-1 and top-2 under both readings, and exact-set accuracy.

    Parameters
    ----------
    labels : np.ndarray
        ``[records, conditions]`` states of one set.
    probability : np.ndarray
        Aligned probabilities.
    thresholds : np.ndarray
        Validation-fixed thresholds.
    conditions : tuple[str, ...]
        Condition names in column order.

    Returns
    -------
    dict[str, Any]
        ``top_1`` and ``top_2``, each with ``all_defined`` and ``undefined_allowed`` readings, and
        ``exact_set``.
    """
    view = {f"top_{k}": {reading: top_k_accuracy(labels, probability, k, require, conditions)
                         for reading, require in (("all_defined", True), ("undefined_allowed", False))}
            for k in (1, 2)}
    return {**view, "exact_set": exact_set_accuracy(labels, probability, thresholds)}


def sex_masks(sex: np.ndarray) -> dict[str, np.ndarray]:
    """
    Subgroup masks by sex.

    Parameters
    ----------
    sex : np.ndarray
        ``F``, ``M`` or ``unknown`` per record.

    Returns
    -------
    dict[str, np.ndarray]
        ``female``, ``male`` and ``unknown`` masks.
    """
    sex = np.asarray(sex, dtype=str)
    return {"female": sex == "F", "male": sex == "M", "unknown": ~np.isin(sex, ["F", "M"])}


def lead_masks(lead_flag: np.ndarray) -> dict[str, np.ndarray]:
    """
    Subgroup masks by the flat-or-missing-lead flag.

    Parameters
    ----------
    lead_flag : np.ndarray
        True where a lead is flat or has missing samples in the first 10 s.

    Returns
    -------
    dict[str, np.ndarray]
        ``flat_or_missing_lead`` and ``all_leads_present`` masks.
    """
    lead_flag = np.asarray(lead_flag, dtype=bool)
    return {"flat_or_missing_lead": lead_flag, "all_leads_present": ~lead_flag}


def subgroup_tables(labels: np.ndarray, probability: np.ndarray, masks: dict[str, np.ndarray],
                    thresholds: np.ndarray | None = None,
                    conditions: tuple[str, ...] = CONDITIONS) -> dict[str, dict[str, Any]]:
    """
    ``metrics_table`` within each nonempty subgroup of one evaluation set.

    Parameters
    ----------
    labels : np.ndarray
        ``[records, conditions]`` states of one set.
    probability : np.ndarray
        Aligned probabilities.
    masks : dict[str, np.ndarray]
        Subgroup name to record mask.
    thresholds : np.ndarray | None
        Validation-fixed thresholds.
    conditions : tuple[str, ...]
        Condition names in column order.

    Returns
    -------
    dict[str, dict[str, Any]]
        One table per nonempty subgroup; the macro entry rule is applied within the subgroup.
    """
    labels, probability = check_scores(labels, probability)
    return {name: metrics_table(labels[mask], probability[mask], thresholds, conditions)
            for name, mask in masks.items() if np.asarray(mask, dtype=bool).any()}


def cooccurrence_labels(labels: np.ndarray) -> dict[str, np.ndarray]:
    """
    Label matrices whose positives are split by the record's number of positive conditions.

    Each condition keeps all its negatives in both matrices. ``one_condition`` keeps the positives of
    records with no other positive condition; ``several_conditions`` keeps those with at least one other.
    The other positives become undefined. Undefined conditions do not count as positives.

    Parameters
    ----------
    labels : np.ndarray
        ``[records, conditions]`` states.

    Returns
    -------
    dict[str, np.ndarray]
        ``one_condition`` and ``several_conditions`` label matrices.
    """
    labels = np.asarray(labels, dtype=np.float64)
    positive = labels == 1
    several = (positive.sum(axis=1) >= 2)[:, None]
    return {"one_condition": np.where(positive & several, np.nan, labels),
            "several_conditions": np.where(positive & ~several, np.nan, labels)}


def cooccurrence_tables(labels: np.ndarray, probability: np.ndarray, thresholds: np.ndarray | None = None,
                        conditions: tuple[str, ...] = CONDITIONS) -> dict[str, dict[str, Any]]:
    """
    Metrics on positives with one condition against positives with several, beside the same negatives.

    Parameters
    ----------
    labels : np.ndarray
        ``[records, conditions]`` states of one set.
    probability : np.ndarray
        Aligned probabilities.
    thresholds : np.ndarray | None
        Validation-fixed thresholds.
    conditions : tuple[str, ...]
        Condition names in column order.

    Returns
    -------
    dict[str, dict[str, Any]]
        ``metrics_table`` of each matrix of ``cooccurrence_labels``.
    """
    return {name: metrics_table(split, probability, thresholds, conditions)
            for name, split in cooccurrence_labels(labels).items()}


def source_identification(features: np.ndarray, families: np.ndarray, groups: np.ndarray, folds: int = 5,
                          seed: int = SEED, c: float = 1.0) -> dict[str, Any]:
    """
    How well a feature matrix identifies the source family: cross-validated multinomial logistic regression.

    Folds keep each group (patient or record) whole and are stratified by family; the standardizer is fitted
    on the training folds only.

    Parameters
    ----------
    features : np.ndarray
        Finite ``[records, features]`` matrix; impute missing values before calling.
    families : np.ndarray
        Source family of each record, at least two families.
    groups : np.ndarray
        Bootstrap group of each record.
    folds : int
        Number of folds.
    seed : int
        Seed of the fold assignment.
    c : float
        Inverse L2 regularization strength.

    Returns
    -------
    dict[str, Any]
        Out-of-fold ``accuracy``, ``balanced_accuracy``, ``macro_ovr_auroc`` and ``log_loss``; the chance
        references ``majority_accuracy`` and ``chance_balanced_accuracy``; ``classes``, ``confusion``
        (rows true, columns predicted), ``records``, ``folds`` and ``seed``.

    Raises
    ------
    ValueError
        If features are not finite or fewer than two families occur.
    """
    features = np.asarray(features, dtype=np.float64)
    families, groups = np.asarray(families, dtype=str), np.asarray(groups, dtype=str)
    if not np.isfinite(features).all():
        raise ValueError("source identification needs finite features")
    classes = np.unique(families)
    if len(classes) < 2:
        raise ValueError("source identification needs at least two families")
    probability = np.zeros((len(features), len(classes)))
    split = StratifiedGroupKFold(n_splits=folds, shuffle=True, random_state=seed)
    for train, held in split.split(features, families, groups):
        scaler = StandardScaler().fit(features[train])
        model = LogisticRegression(C=c, max_iter=5000).fit(scaler.transform(features[train]), families[train])
        probability[np.ix_(held, np.searchsorted(classes, model.classes_))] = model.predict_proba(
            scaler.transform(features[held]))
    predicted = classes[probability.argmax(axis=1)]
    counts = np.array([np.count_nonzero(families == name) for name in classes])
    return {"accuracy": float(accuracy_score(families, predicted)),
            "balanced_accuracy": float(balanced_accuracy_score(families, predicted)),
            "macro_ovr_auroc": float(roc_auc_score(families, probability, multi_class="ovr", labels=classes))
            if len(classes) > 2 else float(roc_auc_score(families == classes[1], probability[:, 1])),
            "log_loss": float(log_loss(families, probability, labels=classes)),
            "majority_accuracy": float(counts.max() / counts.sum()),
            "chance_balanced_accuracy": 1 / len(classes), "classes": classes.tolist(),
            "confusion": confusion_matrix(families, predicted, labels=classes).tolist(),
            "records": len(features), "folds": folds, "seed": seed}


def select_candidate(metrics: dict[str, float], margin: float = TIE_MARGIN) -> dict[str, Any]:
    """
    Select the model with the highest selection metric, ties within ``margin`` going to the simpler model.

    "Simpler" is earlier in the MLC-02 protocol table (``CANDIDATES``). Every candidate within ``margin`` of
    the highest value (inclusive) is tied with it; the earliest of them is selected.

    Parameters
    ----------
    metrics : dict[str, float]
        Candidate id to selection metric.
    margin : float
        Tie margin.

    Returns
    -------
    dict[str, Any]
        ``selected``, ``highest`` (candidate with the highest value, earliest on equality),
        ``highest_metric``, ``tied`` (in table order) and ``margin``.

    Raises
    ------
    ValueError
        If a candidate is not in the protocol table or a value is not finite.
    """
    unknown = set(metrics) - set(CANDIDATES)
    if unknown or not metrics:
        raise ValueError(f"candidates must come from the protocol table, got {sorted(unknown)}")
    if not all(math.isfinite(value) for value in metrics.values()):
        raise ValueError("selection metrics must be finite")
    order = [name for name in CANDIDATES if name in metrics]
    highest = max(metrics.values())
    tied = [name for name in order if highest - metrics[name] <= margin + 1e-12]
    return {"selected": tied[0], "highest": next(name for name in order if metrics[name] == highest),
            "highest_metric": float(highest), "tied": tied, "margin": margin}


def selection_record(tables: dict[str, dict[str, Any]],
                     settings: dict[str, dict[str, Any]] | None = None) -> dict[str, Any]:
    """
    Build the selection record of Stage 3: every candidate's validation table and the selected model.

    Parameters
    ----------
    tables : dict[str, dict[str, Any]]
        Candidate id to its ``selection_metric`` output on validation.
    settings : dict[str, dict[str, Any]] | None
        Candidate id to its chosen settings (grid values, epochs).

    Returns
    -------
    dict[str, Any]
        ``rule``, ``sources``, ``entering_by_source``, ``excluded_sources``, ``candidates`` (table order, each
        with ``id``, ``selection_metric``, ``macro_auroc_by_source`` and ``settings``) and the fields of
        ``select_candidate``.

    Raises
    ------
    ValueError
        If the candidates do not share sources and entering conditions, which would mean different labels.
    """
    settings = settings or {}
    reference = next(iter(tables.values()))
    for name, table in tables.items():
        if table["entering_by_source"] != reference["entering_by_source"]:
            raise ValueError(f"candidate {name} was scored on other validation labels")
    choice = select_candidate({name: table["selection_metric"] for name, table in tables.items()})
    candidates = [{"id": name, "selection_metric": tables[name]["selection_metric"],
                   "macro_auroc_by_source": tables[name]["macro_auroc_by_source"],
                   "settings": settings.get(name, {})} for name in CANDIDATES if name in tables]
    rule = ("Highest mean over validation sources of the within-source macro AUROC; candidates within "
            f"{TIE_MARGIN} of the highest are tied and the earliest in the protocol table is selected.")
    return {"rule": rule, "sources": reference["sources"],
            "entering_by_source": reference["entering_by_source"],
            "excluded_sources": reference["excluded_sources"], "candidates": candidates, **choice}


def selection_markdown(record: dict[str, Any]) -> str:
    """
    Markdown form of a selection record.

    Parameters
    ----------
    record : dict[str, Any]
        Output of ``selection_record``.

    Returns
    -------
    str
        Rule, table of candidates by source, and the selected model.
    """
    sources = [source for source in record["sources"] if source not in record["excluded_sources"]]
    lines = ["# MLC-02 selection record", "", record["rule"], "",
             "| candidate | " + " | ".join(sources) + " | mean | tied | settings |",
             "|---|" + "---:|" * (len(sources) + 1) + "---|---|"]
    for row in record["candidates"]:
        cells = [f"{row['macro_auroc_by_source'][source]:.4f}" for source in sources]
        tied = "yes" if row["id"] in record["tied"] else ""
        lines.append(f"| {row['id']} | " + " | ".join(cells) + f" | {row['selection_metric']:.4f} | {tied} | "
                     + ", ".join(f"{key}={value}" for key, value in row["settings"].items()) + " |")
    lines += ["", f"Highest: {record['highest']} ({record['highest_metric']:.4f}). "
              f"Selected: {record['selected']}.", ""]
    if record["excluded_sources"]:
        lines += [f"Sources without an entering condition: {', '.join(record['excluded_sources'])}.", ""]
    return "\n".join(lines)


def write_selection_record(directory: Path, record: dict[str, Any]) -> dict[str, Path]:
    """
    Write ``selection_record.json`` and ``selection_record.md``.

    Parameters
    ----------
    directory : Path
        Output directory.
    record : dict[str, Any]
        Output of ``selection_record``.

    Returns
    -------
    dict[str, Path]
        ``json`` and ``markdown`` paths.
    """
    directory = Path(directory)
    paths = {"json": directory / "selection_record.json", "markdown": directory / "selection_record.md"}
    write_json_atomic(paths["json"], record)
    write_text_atomic(paths["markdown"], selection_markdown(record))
    return paths
