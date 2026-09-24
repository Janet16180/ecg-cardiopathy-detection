"""Patient-aware evaluation helpers for binary ECG experiments."""

from __future__ import annotations

import csv
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import GroupShuffleSplit

from .files import read_csv, write_json_atomic


def _arrays(y, prob):
    y = np.asarray(y)
    prob = np.asarray(prob, dtype=float)
    if y.ndim != 1 or prob.ndim != 1 or len(y) != len(prob) or not len(y):
        raise ValueError("y and prob must be nonempty one-dimensional arrays of equal length")
    if not np.all(np.isin(y, [0, 1])):
        raise ValueError("y must contain only binary labels 0 and 1")
    if not np.all(np.isfinite(prob)) or np.any((prob < 0) | (prob > 1)):
        raise ValueError("prob must contain finite probabilities in [0, 1]")
    return y.astype(np.int8), prob


def _ratio(numerator, denominator):
    return float(numerator / denominator) if denominator else None


def select_threshold(y, prob, target_sensitivity=0.95):
    """Highest observed probability whose inclusive decision meets target sensitivity."""
    y, prob = _arrays(y, prob)
    if len(np.unique(y)) != 2:
        raise ValueError("threshold selection requires both classes")
    if not np.isfinite(target_sensitivity) or not 0 < target_sensitivity <= 1:
        raise ValueError("target_sensitivity must be in (0, 1]")
    positive = np.sort(prob[y == 1])
    required = int(np.ceil(target_sensitivity * len(positive) - 1e-12))
    # At this positive order statistic, at least `required` positives score >= threshold.
    return float(positive[len(positive) - required])


def metrics(y, prob, threshold):
    """Return JSON-safe point metrics; undefined ratios are None."""
    y, prob = _arrays(y, prob)
    if not np.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError("threshold must be finite and in [0, 1]")
    pred = prob >= threshold
    tp = int(np.count_nonzero((y == 1) & pred))
    tn = int(np.count_nonzero((y == 0) & ~pred))
    fp = int(np.count_nonzero((y == 0) & pred))
    fn = int(np.count_nonzero((y == 1) & ~pred))
    n = len(y)

    eps = np.finfo(float).eps
    clipped = np.clip(prob, eps, 1 - eps)
    brier = float(np.mean((prob - y) ** 2))
    log_loss = float(-np.mean(y * np.log(clipped) + (1 - y) * np.log1p(-clipped)))

    # Ten equal-width probability bins; p=1 belongs in the last bin.
    bin_ids = np.minimum((prob * 10).astype(int), 9)
    ece = 0.0
    for bin_id in range(10):
        in_bin = bin_ids == bin_id
        if np.any(in_bin):
            ece += (np.count_nonzero(in_bin) / n) * abs(
                float(np.mean(y[in_bin])) - float(np.mean(prob[in_bin]))
            )

    return {
        "auroc": float(roc_auc_score(y, prob)) if len(np.unique(y)) == 2 else None,
        "average_precision": float(average_precision_score(y, prob)) if np.any(y == 1) else None,
        "sensitivity": _ratio(tp, tp + fn),
        "specificity": _ratio(tn, tn + fp),
        "precision": _ratio(tp, tp + fp),
        "npv": _ratio(tn, tn + fn),
        "f1": _ratio(2 * tp, 2 * tp + fp + fn),
        "accuracy": float((tp + tn) / n),
        "brier": brier,
        "log_loss": log_loss,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "n": n,
        "prevalence": float(np.mean(y)),
        "predicted_positive_rate": float(np.mean(pred)),
        "ece_10_bins": float(ece),
    }


def patient_bootstrap(y, prob, patient_ids, threshold, repeats=500, seed=2026):
    """Percentile CIs from patient-cluster resampling at a fixed threshold."""
    y, prob = _arrays(y, prob)
    patient_ids = np.asarray(patient_ids)
    if patient_ids.ndim != 1 or len(patient_ids) != len(y):
        raise ValueError("patient_ids must match y")
    if not isinstance(repeats, int) or repeats < 1:
        raise ValueError("repeats must be a positive integer")
    if not np.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError("threshold must be finite and in [0, 1]")
    _, patient_index = np.unique(patient_ids, return_inverse=True)
    groups = [np.flatnonzero(patient_index == i) for i in range(patient_index.max() + 1)]
    rng = np.random.default_rng(seed)
    names = ("auroc", "average_precision", "sensitivity", "specificity", "precision", "brier")
    values = {name: [] for name in names}
    for _ in range(repeats):
        drawn = rng.integers(len(groups), size=len(groups))
        indices = np.concatenate([groups[i] for i in drawn])
        if len(np.unique(y[indices])) != 2:
            continue
        result = metrics(y[indices], prob[indices], threshold)
        for name in names:
            if result[name] is not None:
                values[name].append(result[name])
    return {
        name: [float(x) for x in np.quantile(sample, [0.025, 0.975])]
        if sample else [None, None]
        for name, sample in values.items()
    }


def scenario_ppv(sensitivity, specificity, prevalence):
    """Expected screening outcomes per 1,000 people at an assumed prevalence."""
    for name, value in (("sensitivity", sensitivity), ("specificity", specificity),
                        ("prevalence", prevalence)):
        if not np.isfinite(value) or not 0 <= value <= 1:
            raise ValueError(f"{name} must be finite and in [0, 1]")
    tp = float(1000 * prevalence * sensitivity)
    fn = float(1000 * prevalence * (1 - sensitivity))
    tn = float(1000 * (1 - prevalence) * specificity)
    fp = float(1000 * (1 - prevalence) * (1 - specificity))
    return {
        "expected_tp_per_1000": tp,
        "expected_fp_per_1000": fp,
        "expected_fn_per_1000": fn,
        "expected_tn_per_1000": tn,
        "expected_referrals_per_1000": tp + fp,
        "ppv": _ratio(tp, tp + fp),
    }


def partition_validation(rows: list[dict[str, str]]) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    """
    Split validation rows into development and calibration patients.

    The split is a fixed 70/30 patient-grouped partition (seed 9001) shared
    by every experiment.

    Parameters
    ----------
    rows : list[dict[str, str]]
        Validation manifest rows with ``patient_id`` and ``target``.

    Returns
    -------
    tuple[list[dict[str, str]], list[dict[str, str]]]
        Development rows and calibration rows.

    Raises
    ------
    ValueError
        If either partition lacks one of the two classes.
    """
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.3, random_state=9001)
    development, calibration = next(splitter.split(rows, groups=[r["patient_id"] for r in rows]))
    partitions = [rows[i] for i in development], [rows[i] for i in calibration]
    for partition in partitions:
        if {r["target"] for r in partition} != {"0", "1"}:
            raise ValueError("Development and calibration partitions must each contain both classes")
    return partitions


def evaluate_predictions(name: str, calibration_logits: Sequence[float] | np.ndarray,
                         test_logits: Sequence[float] | np.ndarray,
                         calibration_rows: list[dict[str, str]], test_rows: list[dict[str, str]],
                         output_dir: str | Path, seed: int, bootstrap: int = 500) -> dict[str, Any]:
    """
    Calibrate and choose the operating threshold using calibration patients only.

    Writes ``metrics.json``, ``test_predictions.csv`` and
    ``calibration_predictions.npz`` to ``output_dir``.

    Parameters
    ----------
    name : str
        Model name recorded in the metrics.
    calibration_logits : Sequence[float] | np.ndarray
        Raw logits for ``calibration_rows``.
    test_logits : Sequence[float] | np.ndarray
        Raw logits for ``test_rows``.
    calibration_rows : list[dict[str, str]]
        Calibration manifest rows with ``ecg_id`` and ``target``.
    test_rows : list[dict[str, str]]
        Test manifest rows with ``ecg_id``, ``patient_id`` and ``target``.
    output_dir : str | Path
        Directory for the written artifacts; created if missing.
    seed : int
        Label seed recorded in the metrics.
    bootstrap : int
        Patient bootstrap repeats for test confidence intervals.

    Returns
    -------
    dict[str, Any]
        The contents written to ``metrics.json``.

    Raises
    ------
    RuntimeError
        If the fitted Platt slope is not positive.
    """
    calibration_y = np.array([int(r["target"]) for r in calibration_rows])
    test_y = np.array([int(r["target"]) for r in test_rows])
    calibrator = LogisticRegression(C=1e6, solver="lbfgs", max_iter=1000)
    calibrator.fit(np.asarray(calibration_logits).reshape(-1, 1), calibration_y)
    if calibrator.coef_[0, 0] <= 0:
        raise RuntimeError("Nonpositive calibration slope: investigate the model before evaluation")
    calibration_p = calibrator.predict_proba(np.asarray(calibration_logits).reshape(-1, 1))[:, 1]
    test_p = calibrator.predict_proba(np.asarray(test_logits).reshape(-1, 1))[:, 1]
    threshold = select_threshold(calibration_y, calibration_p, 0.95)
    test_metrics = metrics(test_y, test_p, threshold)
    result = {"model": name, "label_seed": seed, "threshold": threshold,
              "threshold_selection": "Maximum threshold attaining >=95% sensitivity on calibration patients",
              "calibration": {"method": "Platt logistic scaling", "records": len(calibration_rows),
                              "slope": float(calibrator.coef_[0, 0]), "intercept": float(calibrator.intercept_[0])},
              "test": test_metrics,
              "test_ci95_patient_bootstrap": patient_bootstrap(test_y, test_p,
                    [r["patient_id"] for r in test_rows], threshold, repeats=bootstrap, seed=2026),
              "hypothetical_1pct_prevalence": scenario_ppv(test_metrics["sensitivity"],
                    test_metrics["specificity"], 0.01)}
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json_atomic(output_dir / "metrics.json", result)
    with (output_dir / "test_predictions.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["ecg_id", "patient_id", "target", "raw_logit", "probability", "prediction"])
        for row, logit, probability in zip(test_rows, test_logits, test_p):
            writer.writerow([row["ecg_id"], row["patient_id"], row["target"], float(logit),
                             float(probability), int(probability >= threshold)])
    np.savez(output_dir / "calibration_predictions.npz", logits=calibration_logits,
             targets=calibration_y, probabilities=calibration_p,
             ecg_ids=np.array([int(r["ecg_id"]) for r in calibration_rows]))
    print(json.dumps({"model": name, "label_seed": seed, "test": test_metrics}), flush=True)
    return result


def paired_comparison(left: Path, right: Path, repeats: int = 500) -> dict[str, dict[str, Any]]:
    """
    Compare two evaluated models on the same test rows with a patient bootstrap.

    Each model keeps its own calibrated threshold from ``metrics.json``.

    Parameters
    ----------
    left : Path
        Result directory of the first model, as written by
        ``evaluate_predictions``.
    right : Path
        Result directory of the second model.
    repeats : int
        Patient bootstrap repeats.

    Returns
    -------
    dict[str, dict[str, Any]]
        For AUROC, average precision, sensitivity and specificity: the point
        difference (left minus right) and its paired 95% interval.

    Raises
    ------
    ValueError
        If the two prediction files cover different rows.
    """
    left_rows, right_rows = read_csv(left / "test_predictions.csv"), read_csv(right / "test_predictions.csv")
    if [(r["ecg_id"], r["patient_id"], r["target"]) for r in left_rows] != [
            (r["ecg_id"], r["patient_id"], r["target"]) for r in right_rows]:
        raise ValueError("Cannot pair test predictions with different rows")
    y = np.array([int(r["target"]) for r in left_rows])
    ids = np.array([r["patient_id"] for r in left_rows])
    left_p = np.array([float(r["probability"]) for r in left_rows])
    right_p = np.array([float(r["probability"]) for r in right_rows])
    left_t = json.loads((left / "metrics.json").read_text())["threshold"]
    right_t = json.loads((right / "metrics.json").read_text())["threshold"]
    groups = [np.flatnonzero(ids == patient) for patient in np.unique(ids)]
    rng = np.random.default_rng(2026)
    names = ("auroc", "average_precision", "sensitivity", "specificity")
    point_left, point_right = metrics(y, left_p, left_t), metrics(y, right_p, right_t)
    samples = {name: [] for name in names}
    for _ in range(repeats):
        sampled = np.concatenate([groups[i] for i in rng.integers(len(groups), size=len(groups))])
        if len(np.unique(y[sampled])) < 2:
            continue
        a, b = metrics(y[sampled], left_p[sampled], left_t), metrics(y[sampled], right_p[sampled], right_t)
        for name in names:
            samples[name].append(a[name] - b[name])
    return {name: {"difference": point_left[name] - point_right[name],
                   "ci95": np.quantile(samples[name], [0.025, 0.975]).tolist()}
            for name in names}
