"""Frozen cohort and paired development statistics for Experiment 011."""

from __future__ import annotations

from collections import Counter
from typing import Any

import numpy as np
from sklearn.metrics import roc_auc_score

from .cpc_subset25 import exposure_order, select_indices
from .files import sha256_json

SELECTION_SEED = 18046
ORDER_SEED = 18047
SUBSET_SIZE = 25_000
EXPOSURES = 115_359
EXPECTED_SELECTION_HASH = "43f405bfbddbb415f5072480744cebe6427afbeed2474a7be0e625cf1deac791"
ARMS = ("gru", "kda", "ckda")


def frozen_stream(sources: list[str]) -> tuple[np.ndarray, np.ndarray, dict[str, int]]:
    """Replay Experiment 019's exact selected rows and exposure stream.

    Parameters
    ----------
    sources : list[str]
        Sources in the verified 115,359-row manifest order.

    Returns
    -------
    tuple[np.ndarray, np.ndarray, dict[str, int]]
        Selected cache indices, full exposure order, and source counts.
    """
    if len(sources) != EXPOSURES:
        raise ValueError("Unexpected frozen cohort size")
    selected = select_indices(sources, SUBSET_SIZE, SELECTION_SEED)
    if sha256_json(selected.tolist()) != EXPECTED_SELECTION_HASH:
        raise ValueError("Experiment 019 selected-index identity changed")
    order = exposure_order(selected, EXPOSURES, ORDER_SEED)
    if len(np.unique(order)) != SUBSET_SIZE or len(order) != EXPOSURES:
        raise ValueError("Frozen exposure stream is malformed")
    counts = dict(sorted(Counter(sources[index] for index in selected).items()))
    return selected, order, counts


def paired_patient_auc(
    targets: np.ndarray,
    patients: np.ndarray,
    predictions: dict[str, np.ndarray],
    *,
    seed: int = 11045,
    draws: int = 2000,
) -> dict[str, Any]:
    """Compare CKDA/KDA and each delta arm/GRU by paired patient bootstrap.

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
        raise ValueError("Expected GRU, KDA, and CKDA predictions")
    if len(targets) != len(patients) or np.unique(targets).size != 2:
        raise ValueError("Malformed development labels or patient IDs")
    if any(len(value) != len(targets) or not np.isfinite(value).all()
           for value in predictions.values()):
        raise ValueError("Malformed development probabilities")
    pairs = (("ckda_minus_kda", "ckda", "kda"),
             ("kda_minus_gru", "kda", "gru"),
             ("ckda_minus_gru", "ckda", "gru"))
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
