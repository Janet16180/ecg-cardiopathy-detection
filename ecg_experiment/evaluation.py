"""Patient-aware evaluation helpers for binary ECG experiments."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score


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
