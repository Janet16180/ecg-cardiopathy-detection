"""Prespecified paired patient development contrasts for Experiment 013."""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import roc_auc_score

ARMS = ("gru", "mamba2", "mamba3")


def paired_patient_auc(
    targets: np.ndarray,
    patients: np.ndarray,
    predictions: dict[str, np.ndarray],
    *,
    seed: int = 13045,
    draws: int = 2000,
) -> dict[str, Any]:
    """Compare Mamba-3/Mamba-2 and each Mamba arm/GRU by paired patient bootstrap.

    Parameters
    ----------
    targets : np.ndarray
        Binary development annotations in fixed record order.
    patients : np.ndarray
        Development patient identities in the same order.
    predictions : dict[str, np.ndarray]
        Probabilities for all three frozen architecture arms.
    seed : int
        Bootstrap random seed.
    draws : int
        Number of patient resamples.

    Returns
    -------
    dict[str, Any]
        Point differences, percentile intervals, and valid draw counts.
    """
    if set(predictions) != set(ARMS):
        raise ValueError("Expected GRU, Mamba-2, and Mamba-3 predictions")
    if len(targets) != len(patients) or np.unique(targets).size != 2:
        raise ValueError("Malformed development labels or patient IDs")
    if any(len(value) != len(targets) or not np.isfinite(value).all()
           for value in predictions.values()):
        raise ValueError("Malformed development probabilities")
    pairs = (("mamba3_minus_mamba2", "mamba3", "mamba2"),
             ("mamba2_minus_gru", "mamba2", "gru"),
             ("mamba3_minus_gru", "mamba3", "gru"))
    unique, inverse = np.unique(patients, return_inverse=True)
    positions = [np.flatnonzero(inverse == index) for index in range(len(unique))]
    rng = np.random.default_rng(seed)
    samples: list[np.ndarray] = []
    for _ in range(draws):
        sampled = rng.integers(len(unique), size=len(unique))
        index = np.concatenate([positions[item] for item in sampled])
        if np.unique(targets[index]).size == 2:
            samples.append(index)
    if not samples:
        raise ValueError("No valid paired bootstrap draws")
    result = {}
    for name, left, right in pairs:
        point = float(roc_auc_score(targets, predictions[left])
                      - roc_auc_score(targets, predictions[right]))
        differences = np.asarray([
            roc_auc_score(targets[index], predictions[left][index])
            - roc_auc_score(targets[index], predictions[right][index])
            for index in samples
        ])
        result[name] = {
            "auroc_difference": point,
            "ci95": np.quantile(differences, [0.025, 0.975]).tolist(),
            "valid_draws": len(samples),
            "invalid_draws": draws - len(samples),
        }
    return result
