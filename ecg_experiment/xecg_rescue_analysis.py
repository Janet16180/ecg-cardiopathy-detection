"""Development-only analysis for the fixed Experiment 016 DropPath rescue."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score, log_loss, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold

from .evaluation import select_threshold


def development_metrics(labels: np.ndarray, patients: np.ndarray, logits: np.ndarray) -> dict:
    """Report discrimination and patient-separated 95%-sensitivity policy."""
    probabilities = 1 / (1 + np.exp(-np.clip(logits, -80, 80)))
    folds = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    rows = []
    for train, held in folds.split(probabilities, labels, patients):
        threshold = select_threshold(labels[train], probabilities[train], 0.95)
        decision = probabilities[held] >= threshold
        positive = labels[held] == 1
        rows.append(
            {
                "sensitivity": float(np.mean(decision[positive])),
                "specificity": float(np.mean(~decision[~positive])),
                "threshold": threshold,
                "records": len(held),
                "patients": len(np.unique(patients[held])),
            }
        )
    return {
        "auroc": float(roc_auc_score(labels, logits)),
        "average_precision": float(average_precision_score(labels, logits)),
        "bce": float(log_loss(labels, probabilities)),
        "patient_folds": rows,
        "mean_fold_sensitivity": float(np.mean([row["sensitivity"] for row in rows])),
        "mean_fold_specificity": float(np.mean([row["specificity"] for row in rows])),
    }


def paired_patient_bootstrap(
    labels: np.ndarray, patients: np.ndarray, predictions: dict[str, np.ndarray]
) -> dict:
    """Resample whole patients with paired arm predictions, seed 16016."""
    unique, inverse = np.unique(patients, return_inverse=True)
    members = [np.flatnonzero(inverse == i) for i in range(len(unique))]
    pairs = {
        "residual_minus_legacy": ("residual", "legacy"),
        "off_minus_legacy": ("off", "legacy"),
        "residual_minus_off": ("residual", "off"),
    }
    draws = {name: [] for name in pairs}
    invalid = 0
    rng = np.random.default_rng(16016)
    for _ in range(2000):
        index = np.concatenate([members[i] for i in rng.integers(len(members), size=len(members))])
        if len(np.unique(labels[index])) < 2:
            invalid += 1
            continue
        for name, (left, right) in pairs.items():
            draws[name].append(
                float(
                    roc_auc_score(labels[index], predictions[left][index])
                    - roc_auc_score(labels[index], predictions[right][index])
                )
            )
    return {
        "requested": 2000,
        "valid": 2000 - invalid,
        "single_class_invalid": invalid,
        "contrasts": {
            name: {
                "observed": float(
                    roc_auc_score(labels, predictions[left]) - roc_auc_score(labels, predictions[right])
                ),
                "interval_95": np.quantile(draws[name], [0.025, 0.975]).tolist(),
            }
            for name, (left, right) in pairs.items()
        },
    }
