"""Fixed-recipe linear readouts and patient-paired analysis for xECG v7."""

from __future__ import annotations

import warnings

import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

from scripts.experiments.run_xecg_probe_finetune016 import affine_from_standardized

FEATURE_WIDTH = 1024
TRAIN_RECORDS = 15359
DEVELOPMENT_RECORDS = 1306
TOTAL_RECORDS = TRAIN_RECORDS + DEVELOPMENT_RECORDS
PROBE_C = 0.01
BOOTSTRAP_DRAWS = 2000
BOOTSTRAP_SEED = 16017


def selected_indices(all_ids: np.ndarray, requested_ids: list[int]) -> np.ndarray:
    """Join requested clean ECG identifiers to a unique frozen cache order."""
    if len(np.unique(all_ids)) != len(all_ids) or len(set(requested_ids)) != len(requested_ids):
        raise ValueError("Duplicate ECG identifier in cache or requested rows")
    index = {int(ecg_id): position for position, ecg_id in enumerate(all_ids)}
    if any(ecg_id not in index for ecg_id in requested_ids):
        raise ValueError("Requested ECG is absent from frozen feature cache")
    return np.asarray([index[ecg_id] for ecg_id in requested_ids], dtype=np.int64)


def affine_logits(features: np.ndarray, weight: np.ndarray, bias: float) -> np.ndarray:
    """Apply a raw-feature affine head, without standardizing a saved head."""
    if features.ndim != 2 or features.shape[1] != FEATURE_WIDTH or weight.shape != (FEATURE_WIDTH,):
        raise ValueError("Raw pooled feature or affine-head dimensions differ")
    logits = features.astype(np.float64) @ weight.astype(np.float64) + float(bias)
    if not np.isfinite(logits).all():
        raise ValueError("Nonfinite affine logits")
    return logits


def fixed_probe(
    train_features: np.ndarray,
    train_labels: np.ndarray,
    development_features: np.ndarray,
) -> dict:
    """Fit the fixed C=.01 scaler/probe solely on clean training features."""
    if (
        train_features.shape != (TRAIN_RECORDS, FEATURE_WIDTH)
        or development_features.shape != (DEVELOPMENT_RECORDS, FEATURE_WIDTH)
        or train_labels.shape != (TRAIN_RECORDS,)
    ):
        raise ValueError("Fixed probe rows or feature width differ")
    if set(np.unique(train_labels)) != {0, 1}:
        raise ValueError("Training labels must contain both classes")
    train = train_features.astype(np.float64)
    development = development_features.astype(np.float64)
    if not np.isfinite(train).all() or not np.isfinite(development).all():
        raise ValueError("Nonfinite probe input features")
    scaler = StandardScaler().fit(train)
    with warnings.catch_warnings():
        warnings.simplefilter("error", ConvergenceWarning)
        model = LogisticRegression(C=PROBE_C, solver="lbfgs", max_iter=3000, random_state=42)
        model.fit(scaler.transform(train), train_labels)
    if int(model.n_iter_[0]) >= 3000:
        raise RuntimeError("Fixed probe reached the nonconvergence limit")
    native = model.decision_function(scaler.transform(development))
    weight, bias = affine_from_standardized(
        model.coef_[0], float(model.intercept_[0]), scaler.mean_, scaler.scale_
    )
    folded = affine_logits(development_features, weight, bias)
    if not np.allclose(native, folded, atol=1e-9, rtol=1e-10):
        raise RuntimeError("Raw-feature folded head differs from standardized probe")
    return {
        "weight": weight,
        "bias": float(bias),
        "mean": scaler.mean_,
        "scale": scaler.scale_,
        "coefficient": model.coef_[0],
        "intercept": float(model.intercept_[0]),
        "iterations": int(model.n_iter_[0]),
        "development_logits": native,
    }


def paired_bootstrap(
    labels: np.ndarray,
    patients: np.ndarray,
    logits: dict[str, np.ndarray],
    contrasts: dict[str, tuple[str, str]],
) -> dict:
    """Use common seed-16017 whole-patient draws for every declared contrast."""
    unique, inverse = np.unique(patients, return_inverse=True)
    members = [np.flatnonzero(inverse == index) for index in range(len(unique))]
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    draws = {name: [] for name in contrasts}
    invalid = 0
    for _ in range(BOOTSTRAP_DRAWS):
        sample = np.concatenate([members[index] for index in rng.integers(len(members), size=len(members))])
        if len(np.unique(labels[sample])) < 2:
            invalid += 1
            continue
        auc = {name: roc_auc_score(labels[sample], values[sample]) for name, values in logits.items()}
        for name, (left, right) in contrasts.items():
            draws[name].append(float(auc[left] - auc[right]))
    return {
        "seed": BOOTSTRAP_SEED,
        "requested": BOOTSTRAP_DRAWS,
        "valid": BOOTSTRAP_DRAWS - invalid,
        "single_class_invalid": invalid,
        "contrasts": {
            name: {
                "observed": float(roc_auc_score(labels, logits[left]) - roc_auc_score(labels, logits[right])),
                "interval_95": np.quantile(draws[name], [0.025, 0.975]).tolist(),
            }
            for name, (left, right) in contrasts.items()
        },
    }


def decisions(metrics: dict[str, dict], bootstrap: dict) -> dict:
    """Apply the frozen Off-primary recovery rules without selecting controls."""
    probe = metrics["released_refit"]
    off = metrics["off_refit"]
    joint = metrics["off_joint"]
    gain = off["auroc"] - joint["auroc"]
    delta_probe = off["auroc"] - probe["auroc"]
    sensitivity_ok = probe["mean_fold_sensitivity"] - off["mean_fold_sensitivity"] <= 0.005
    interval = bootstrap["contrasts"]["off_refit_minus_off_joint"]["interval_95"]
    recoverable = gain >= 0.005
    return {
        "off_refit_minus_joint": gain,
        "off_refit_minus_released_probe": delta_probe,
        "off_sensitivity_harm_vs_probe": probe["mean_fold_sensitivity"] - off["mean_fold_sensitivity"],
        "recoverable_readout_point_screen": recoverable,
        "recoverable_interval_lower_above_zero": interval[0] > 0,
        "near_complete_practical_recovery": recoverable and delta_probe >= -0.002 and sensitivity_ok,
        "potential_useful_adaptation": delta_probe >= 0.002 and sensitivity_ok,
        "all_adapted_refits_below_probe_by_more_than_0_002": all(
            metrics[f"{arm}_refit"]["auroc"] < probe["auroc"] - 0.002 for arm in ("legacy", "residual", "off")
        ),
        "retain_released_probe": True,
    }
