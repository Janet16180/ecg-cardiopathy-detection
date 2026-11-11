"""
Patient-cluster bootstrap of MLC-02: shared resamples, intervals and paired model comparisons.

Convention:

- Groups are patients where the manifest has them (PTB-XL, SPH) and records otherwise. One draw takes
  ``P`` groups with replacement from the ``P`` groups of the set and keeps every record of a drawn group,
  through ``intervals.patient_groups`` and ``intervals.patient_resample`` on one
  ``numpy.random.default_rng(seed)``.
- A draw is redrawn when any condition that enters the set's macro (``enters_macro`` on the full set) has no
  positive or no negative among its defined records. Redrawing continues the same generator, so the
  resamples depend only on the groups, the labels and the seed, never on a model; ``cluster_resamples`` is
  computed once per set and passed to every model. Its ``fingerprint`` is the SHA-256 of the weights.
- A resample is stored as integer record weights (how often each record was drawn). Weighted metrics on
  those weights equal the metrics on the concatenated resample.
- Intervals are the 2.5 and 97.5 percentiles (``np.nanpercentile``, linear) of the resampled values. Intervals
  are reported for conditions that enter the macro only; resamples whose calibration fit did not converge
  are left out of that metric's interval and counted.
- Paired comparisons use the same resamples for both models. Per-condition p-values are two-sided
  bootstrap p-values, ``min(1, 2 min((#{d <= 0} + 1) / (B + 1), (#{d >= 0} + 1) / (B + 1)))``, Holm-adjusted
  over the conditions that enter the macro.
"""

from __future__ import annotations

import hashlib
import math
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import numpy as np

from ecg_experiment.intervals import patient_groups, patient_resample
from ecg_experiment.mlc02_data import CONDITIONS
from ecg_experiment.mlc02_evaluation import (
    IMPRECISE_BELOW,
    MIN_SUPPORT,
    USABLE_AUROC_LOW,
    auroc_by_condition,
    check_scores,
    enters_macro,
    logistic_fit,
    logit,
    metrics_table,
    offset_intercept,
)

DRAWS = 2000
SEED = 20261110
CHUNK = 50
WORKERS = 8
MAX_ATTEMPTS_PER_DRAW = 50
METRICS = ("auroc", "average_precision", "brier", "sensitivity", "specificity", "calibration_intercept",
           "calibration_slope", "calibration_in_the_large")
CALIBRATION = ("calibration_intercept", "calibration_slope", "calibration_in_the_large")


def labels_digest(labels: np.ndarray) -> str:
    """
    Return the SHA-256 of a label matrix, undefined entries written as -1.

    Parameters
    ----------
    labels : np.ndarray
        ``[records, conditions]`` states.

    Returns
    -------
    str
        Hexadecimal digest of the float64 bytes and the shape.
    """
    labels = np.asarray(labels, dtype=np.float64)
    return hashlib.sha256(repr(labels.shape).encode() + np.nan_to_num(labels, nan=-1.0).tobytes()).hexdigest()


def valid_draw(weights: np.ndarray, positives: np.ndarray, negatives: np.ndarray) -> bool:
    """
    Whether every entering condition keeps a positive and a negative.

    Parameters
    ----------
    weights : np.ndarray
        Record weights of one draw.
    positives, negatives : np.ndarray
        ``[records, entering conditions]`` indicators of defined positives and negatives.

    Returns
    -------
    bool
        True if the draw is kept.
    """
    return bool((weights @ positives > 0).all() and (weights @ negatives > 0).all())


def cluster_resamples(groups: np.ndarray, labels: np.ndarray, draws: int = DRAWS, seed: int = SEED,
                      minimum: int = MIN_SUPPORT) -> dict[str, Any]:
    """
    Patient-cluster bootstrap resamples of one evaluation set, shared by every model.

    Parameters
    ----------
    groups : np.ndarray
        Bootstrap group of each record.
    labels : np.ndarray
        ``[records, conditions]`` states of the set.
    draws : int
        Number of kept resamples.
    seed : int
        Seed of ``numpy.random.default_rng``.
    minimum : int
        Entry rule of ``enters_macro``.

    Returns
    -------
    dict[str, Any]
        ``weights`` (uint16 ``[draws, records]``), ``entering`` (boolean per condition), ``draws``,
        ``redraws``, ``seed``, ``groups`` (number of groups), ``records``, ``labels_sha256`` and
        ``fingerprint`` (SHA-256 of the weights).

    Raises
    ------
    ValueError
        If the inputs are misaligned or no condition enters the macro.
    RuntimeError
        If more than ``MAX_ATTEMPTS_PER_DRAW`` attempts per kept draw are needed.
    """
    labels = np.asarray(labels, dtype=np.float64)
    groups = np.asarray(groups)
    if labels.ndim != 2 or len(groups) != len(labels):
        raise ValueError("groups and labels must describe the same records")
    entering = enters_macro(labels, minimum)
    if not entering.any():
        raise ValueError("no condition enters the macro; nothing to bootstrap")
    positives = (labels[:, entering] == 1).astype(np.float64)
    negatives = (labels[:, entering] == 0).astype(np.float64)
    members = patient_groups(groups)
    rng = np.random.default_rng(seed)
    weights = np.empty((draws, len(labels)), dtype=np.uint16)
    kept = attempts = 0
    while kept < draws:
        if attempts >= MAX_ATTEMPTS_PER_DRAW * draws:
            raise RuntimeError(f"{attempts} attempts for {kept} valid draws")
        attempts += 1
        drawn = np.bincount(patient_resample(members, rng), minlength=len(labels))
        if valid_draw(drawn, positives, negatives):
            weights[kept] = drawn
            kept += 1
    return {"weights": weights, "entering": entering, "draws": draws, "redraws": attempts - draws,
            "seed": seed, "groups": len(members), "records": len(labels),
            "labels_sha256": labels_digest(labels),
            "fingerprint": hashlib.sha256(weights.tobytes()).hexdigest()}


def group_sums(values: np.ndarray, starts: np.ndarray | None) -> np.ndarray:
    """
    Sum over runs of tied scores along the last axis.

    Parameters
    ----------
    values : np.ndarray
        ``[fits, records]`` values in score order.
    starts : np.ndarray | None
        First position of each tie run, or None when no score is tied.

    Returns
    -------
    np.ndarray
        ``[fits, runs]`` sums.
    """
    return values if starts is None else np.add.reduceat(values, starts, axis=1)


def weighted_ranking(y: np.ndarray, score: np.ndarray, weights: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Weighted AUROC and average precision for many weightings of the same records.

    Tied scores count one half in the AUROC, as in ``roc_auc_score``; average precision is
    ``sum_k (R_k - R_{k-1}) P_k`` over distinct thresholds, as in ``average_precision_score``.

    Parameters
    ----------
    y : np.ndarray
        Defined binary states of ``m`` records.
    score : np.ndarray
        Scores of the same records.
    weights : np.ndarray
        ``[fits, m]`` record weights.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        AUROC and average precision, one per fit.
    """
    order = np.argsort(score, kind="stable")
    sorted_score = score[order]
    tied = np.flatnonzero(np.r_[True, sorted_score[1:] != sorted_score[:-1]])
    starts = None if len(tied) == len(sorted_score) else tied
    weight = np.asarray(weights, dtype=np.float64)[:, order]
    positive_weight = weight * y[order]
    positive = group_sums(positive_weight, starts)
    negative = group_sums(np.subtract(weight, positive_weight, out=weight), starts)
    total_positive, total_negative = positive.sum(axis=1), negative.sum(axis=1)
    negative_up_to = np.cumsum(negative, axis=1)
    pairs = np.einsum("ij,ij->i", positive, negative_up_to)
    if starts is not None:
        pairs -= 0.5 * np.einsum("ij,ij->i", positive, negative)
    flagged_positive = np.cumsum(positive, axis=1)
    np.subtract(total_positive[:, None], flagged_positive, out=flagged_positive)
    flagged_positive += positive
    flagged = np.subtract(total_negative[:, None], negative_up_to, out=negative_up_to)
    flagged += negative
    flagged += flagged_positive
    with np.errstate(all="ignore"):
        precision = np.divide(flagged_positive, flagged, out=flagged_positive, where=flagged > 0)
        auroc = pairs / (total_positive * total_negative)
        average_precision = np.einsum("ij,ij->i", positive, precision) / total_positive
    return auroc, average_precision


def weighted_rates(y: np.ndarray, p: np.ndarray, weights: np.ndarray,
                   threshold: float) -> tuple[np.ndarray, ...]:
    """
    Weighted sensitivity and specificity at a fixed threshold.

    Parameters
    ----------
    y : np.ndarray
        Defined binary states.
    p : np.ndarray
        Probabilities.
    weights : np.ndarray
        ``[fits, records]`` weights.
    threshold : float
        ``p >= threshold`` is positive.

    Returns
    -------
    tuple[np.ndarray, ...]
        Sensitivity and specificity, one per fit.
    """
    predicted = p >= threshold
    positive = y == 1
    hits = np.column_stack([predicted & positive, positive, ~predicted & ~positive, ~positive])
    counts = weights @ hits.astype(np.float64)
    with np.errstate(all="ignore"):
        return counts[:, 0] / counts[:, 1], counts[:, 2] / counts[:, 3]


def chunk_metrics(y: np.ndarray, p: np.ndarray, weights: np.ndarray, threshold: float | None,
                  start: dict[str, Any] | None) -> dict[str, np.ndarray]:
    """
    Bootstrap metrics of one condition for one block of resamples.

    Parameters
    ----------
    y : np.ndarray
        Defined binary states.
    p : np.ndarray
        Probabilities.
    weights : np.ndarray
        ``[fits, records]`` weights.
    threshold : float | None
        Validation-fixed threshold.
    start : dict[str, Any] | None
        Output of ``observed_start``, Newton's starting point; None skips calibration.

    Returns
    -------
    dict[str, np.ndarray]
        One array per computed metric name of ``METRICS``.
    """
    with np.errstate(all="ignore"):
        result = {"brier": weights @ ((p - y) ** 2) / weights.sum(axis=1)}
    if threshold is not None:
        result["sensitivity"], result["specificity"] = weighted_rates(y, p, weights, threshold)
    if start is not None:
        x = logit(p)
        intercept, slope = logistic_fit(y, x, weights, start["joint"])
        result["calibration_intercept"], result["calibration_slope"] = intercept, slope
        result["calibration_in_the_large"] = offset_intercept(y, x, weights, start["offset"])
    result["auroc"], result["average_precision"] = weighted_ranking(y, p, weights)
    return result


def observed_start(y: np.ndarray, p: np.ndarray) -> dict[str, Any]:
    """
    Observed calibration fits, the starting points of the resampled fits.

    Parameters
    ----------
    y : np.ndarray
        Defined binary states.
    p : np.ndarray
        Probabilities.

    Returns
    -------
    dict[str, Any]
        ``joint`` (intercept and slope, or None for the null-model start if the observed fit did not
        converge) and ``offset`` (calibration in the large, 0 if it did not converge).
    """
    x = logit(p)
    intercept, slope = logistic_fit(y, x)
    offset = offset_intercept(y, x)[0]
    joint = (float(intercept[0]), float(slope[0])) if np.isfinite([intercept[0], slope[0]]).all() else None
    return {"joint": joint, "offset": float(offset) if np.isfinite(offset) else 0.0}


def bootstrap_metrics(labels: np.ndarray, probability: np.ndarray, resamples: dict[str, Any],
                      thresholds: np.ndarray | None = None, calibration: bool = True) -> dict[str, Any]:
    """
    Resampled per-condition metrics and macro AUROC of one model on one set.

    Parameters
    ----------
    labels : np.ndarray
        ``[records, conditions]`` states of the set.
    probability : np.ndarray
        Aligned probabilities.
    resamples : dict[str, Any]
        Output of ``cluster_resamples`` for this set.
    thresholds : np.ndarray | None
        Validation-fixed thresholds; None leaves sensitivity and specificity out.
    calibration : bool
        Also resample the calibration fits (the slowest part).

    Returns
    -------
    dict[str, Any]
        ``samples`` (metric name to ``[draws, conditions]``, NaN where undefined), ``macro_auroc``
        (``[draws]``), ``observed_auroc``, ``observed_macro_auroc``, ``entering`` and ``fingerprint``.

    Raises
    ------
    ValueError
        If the resamples were drawn for another set or other labels.
    """
    labels, probability = check_scores(labels, probability)
    weights = resamples["weights"]
    if weights.shape[1] != len(labels) or labels_digest(labels) != resamples["labels_sha256"]:
        raise ValueError("resamples were drawn for another set or other labels")
    skipped = ((() if thresholds is not None else ("sensitivity", "specificity"))
               + (() if calibration else CALIBRATION))
    names = [name for name in METRICS if name not in skipped]
    samples = {name: np.full((len(weights), labels.shape[1]), np.nan) for name in names}
    for column in np.flatnonzero(resamples["entering"]):
        defined = ~np.isnan(labels[:, column])
        y, p = labels[defined, column], probability[defined, column]
        start = observed_start(y, p) if calibration else None
        threshold = None if thresholds is None else float(thresholds[column])
        for low in range(0, len(weights), CHUNK):
            block = weights[low:low + CHUNK][:, defined].astype(np.float64)
            for name, values in chunk_metrics(y, p, block, threshold, start).items():
                samples[name][low:low + CHUNK, column] = values
    entering = resamples["entering"]
    observed = auroc_by_condition(labels, probability)
    return {"samples": samples, "macro_auroc": samples["auroc"][:, entering].mean(axis=1),
            "observed_auroc": observed, "observed_macro_auroc": float(observed[entering].mean()),
            "entering": entering, "fingerprint": resamples["fingerprint"]}


def bootstrap_models(labels: np.ndarray, probabilities: dict[str, np.ndarray], resamples: dict[str, Any],
                     thresholds: dict[str, np.ndarray] | None = None, calibration: bool = True,
                     workers: int = WORKERS) -> dict[str, dict[str, Any]]:
    """
    ``bootstrap_metrics`` of several models on one set, on the same resamples, in parallel threads.

    NumPy releases the GIL in the heavy steps, so threads share the resamples without copying them. Limit
    BLAS threads (``OPENBLAS_NUM_THREADS=1``) when ``workers`` is above 1.

    Parameters
    ----------
    labels : np.ndarray
        ``[records, conditions]`` states of the set.
    probabilities : dict[str, np.ndarray]
        Model id to aligned probabilities.
    resamples : dict[str, Any]
        Output of ``cluster_resamples`` for this set.
    thresholds : dict[str, np.ndarray] | None
        Model id to its validation-fixed thresholds.
    calibration : bool
        Also resample the calibration fits.
    workers : int
        Number of threads.

    Returns
    -------
    dict[str, dict[str, Any]]
        Model id to its ``bootstrap_metrics`` output.
    """
    def run(name: str) -> dict[str, Any]:
        own = None if thresholds is None else thresholds[name]
        return bootstrap_metrics(labels, probabilities[name], resamples, own, calibration)

    with ThreadPoolExecutor(max_workers=workers) as pool:
        return dict(zip(probabilities, pool.map(run, probabilities), strict=True))


def percentile_interval(values: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    2.5 and 97.5 percentiles over resamples, ignoring NaN.

    Parameters
    ----------
    values : np.ndarray
        ``[draws]`` or ``[draws, conditions]`` resampled values.

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray]
        Lower bounds, upper bounds and the number of NaN resamples.
    """
    values = np.asarray(values, dtype=np.float64)
    missing = np.isnan(values).sum(axis=0)
    with np.errstate(all="ignore"):
        if values.ndim == 1:
            if missing == len(values):
                return np.asarray(math.nan), np.asarray(math.nan), np.asarray(missing)
            low, high = np.nanpercentile(values, [2.5, 97.5])
            return np.asarray(low), np.asarray(high), np.asarray(missing)
        low, high = np.full(values.shape[1], np.nan), np.full(values.shape[1], np.nan)
        usable = missing < len(values)
        if usable.any():
            low[usable], high[usable] = np.nanpercentile(values[:, usable], [2.5, 97.5], axis=0)
    return low, high, missing


def interval_report(labels: np.ndarray, probability: np.ndarray, resamples: dict[str, Any],
                    thresholds: np.ndarray | None = None, calibration: bool = True,
                    conditions: tuple[str, ...] = CONDITIONS) -> tuple[dict[str, Any], dict[str, Any]]:
    """
    Observed metrics with percentile 95% intervals for one model on one set.

    Parameters
    ----------
    labels : np.ndarray
        ``[records, conditions]`` states of the set.
    probability : np.ndarray
        Aligned probabilities.
    resamples : dict[str, Any]
        Output of ``cluster_resamples`` for this set.
    thresholds : np.ndarray | None
        Validation-fixed thresholds.
    calibration : bool
        Also give intervals of the calibration fits.
    conditions : tuple[str, ...]
        Condition names in column order.

    Returns
    -------
    tuple[dict[str, Any], dict[str, Any]]
        The report (``metrics_table`` with ``<metric>_ci_low``, ``<metric>_ci_high`` and
        ``<metric>_nan_draws`` per condition, ``usable`` per condition, the macro interval, and the
        resampling identity) and the ``bootstrap_metrics`` output for paired comparisons.
    """
    table = metrics_table(labels, probability, thresholds, conditions)
    boot = bootstrap_metrics(labels, probability, resamples, thresholds, calibration)
    for name, values in boot["samples"].items():
        low, high, missing = percentile_interval(values)
        for column, row in enumerate(table["conditions"]):
            reported = bool(boot["entering"][column])
            row[f"{name}_ci_low"] = float(low[column]) if reported else math.nan
            row[f"{name}_ci_high"] = float(high[column]) if reported else math.nan
            row[f"{name}_nan_draws"] = int(missing[column]) if reported else None
    for row in table["conditions"]:
        row["usable"] = bool(row["enters_macro"] and row["auroc_ci_low"] >= USABLE_AUROC_LOW
                             and row["positives"] >= IMPRECISE_BELOW)
    low, high, _ = percentile_interval(boot["macro_auroc"])
    table |= {"macro_auroc_ci_low": float(low), "macro_auroc_ci_high": float(high),
              "draws": resamples["draws"], "redraws": resamples["redraws"], "seed": resamples["seed"],
              "groups": resamples["groups"],
              "fingerprint": resamples["fingerprint"]}
    return table, boot


def bootstrap_p_value(differences: np.ndarray) -> float:
    """
    Two-sided bootstrap p-value of a zero difference.

    Parameters
    ----------
    differences : np.ndarray
        Resampled differences.

    Returns
    -------
    float
        ``min(1, 2 min((#{d <= 0} + 1) / (B + 1), (#{d >= 0} + 1) / (B + 1)))`` over the non-NaN values.
    """
    values = np.asarray(differences, dtype=np.float64)
    values = values[~np.isnan(values)]
    count = len(values) + 1
    tail = min((np.count_nonzero(values <= 0) + 1) / count, (np.count_nonzero(values >= 0) + 1) / count)
    return float(min(1.0, 2 * tail))


def holm_adjust(p_values: np.ndarray) -> np.ndarray:
    """
    Holm step-down adjusted p-values.

    Parameters
    ----------
    p_values : np.ndarray
        Unadjusted p-values; NaN entries are not part of the family and stay NaN.

    Returns
    -------
    np.ndarray
        Adjusted p-values in the input order.
    """
    p_values = np.asarray(p_values, dtype=np.float64)
    adjusted = np.full(p_values.shape, np.nan)
    family = np.flatnonzero(~np.isnan(p_values))
    order = family[np.argsort(p_values[family], kind="stable")]
    scaled = (len(order) - np.arange(len(order))) * p_values[order]
    adjusted[order] = np.minimum(1.0, np.maximum.accumulate(scaled))
    return adjusted


def paired_comparison(first: dict[str, Any], second: dict[str, Any],
                      conditions: tuple[str, ...] = CONDITIONS) -> dict[str, Any]:
    """
    Paired bootstrap of AUROC differences, first model minus second, on the same resamples.

    Parameters
    ----------
    first, second : dict[str, Any]
        ``bootstrap_metrics`` outputs of the two models on the same set.
    conditions : tuple[str, ...]
        Condition names in column order.

    Returns
    -------
    dict[str, Any]
        ``macro`` (``difference``, ``ci_low``, ``ci_high``, ``better`` when the lower end is above zero) and
        ``conditions`` (per entering condition: ``difference``, ``ci_low``, ``ci_high``, ``p_value``,
        ``p_holm``), with ``fingerprint``.

    Raises
    ------
    ValueError
        If the two outputs come from different resamples.
    """
    if first["fingerprint"] != second["fingerprint"]:
        raise ValueError("paired comparisons need the same resamples")
    entering = first["entering"]
    macro = first["macro_auroc"] - second["macro_auroc"]
    low, high, _ = percentile_interval(macro)
    differences = first["samples"]["auroc"] - second["samples"]["auroc"]
    observed = first["observed_auroc"] - second["observed_auroc"]
    p_values = np.array([bootstrap_p_value(differences[:, c]) if entering[c] else np.nan
                         for c in range(len(entering))])
    holm = holm_adjust(p_values)
    lows, highs, _ = percentile_interval(differences)
    rows = [{"condition": name, "difference": float(observed[c]), "ci_low": float(lows[c]),
             "ci_high": float(highs[c]), "p_value": float(p_values[c]), "p_holm": float(holm[c])}
            for c, name in enumerate(conditions) if entering[c]]
    difference = first["observed_macro_auroc"] - second["observed_macro_auroc"]
    return {"macro": {"difference": float(difference), "ci_low": float(low), "ci_high": float(high),
                      "better": bool(low > 0)},
            "conditions": rows, "holm_family": int(entering.sum()), "fingerprint": first["fingerprint"]}
