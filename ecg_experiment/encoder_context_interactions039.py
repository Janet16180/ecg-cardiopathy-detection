"""Exploratory paired encoder-by-context interactions for Experiment 039."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import roc_auc_score

from ecg_experiment import ROOT
from ecg_experiment.encoder_context_analysis039 import (
    BUDGETS,
    CONTEXTS,
    ENCODERS,
    OUTPUT,
    SEEDS,
    TIERS,
    load_cells,
)
from ecg_experiment.files import sha256_file, write_json_atomic
from ecg_experiment.intervals import patient_groups, two_class_resamples
from ecg_experiment.paths import to_stored

SOURCE_FILES = (
    "ecg_experiment/encoder_context_interactions039.py",
    "tests/test_encoder_context_interactions039.py",
    "ecg_experiment/encoder_context_analysis039.py",
    "ecg_experiment/intervals.py",
    "ecg_experiment/files.py",
    "ecg_experiment/paths.py",
)


def _interaction_keys() -> dict[str, tuple[str, str, str, str]]:
    """Map interaction names to encoder/CNN xLSTM then encoder/CNN GRU keys.

    Returns
    -------
    dict[str, tuple[str, str, str, str]]
        Four score keys for each prespecified interaction.
    """
    return {
        f"{tier}k_{budget}_{encoder}_by_context": (
            f"{tier}_{encoder}_{budget}_xlstm",
            f"{tier}_cnn_{budget}_xlstm",
            f"{tier}_{encoder}_{budget}_gru",
            f"{tier}_cnn_{budget}_gru",
        )
        for tier in TIERS
        for budget in BUDGETS
        for encoder in ENCODERS[1:]
    }


def _validated(arrays: dict[str, np.ndarray], names: list[str]) -> dict[str, np.ndarray]:
    """Require aligned identities, binary targets and finite three-seed scores.

    Parameters
    ----------
    arrays : dict[str, np.ndarray]
        Development identities, targets and candidate score matrices.
    names : list[str]
        Score keys required by the interactions.

    Returns
    -------
    dict[str, np.ndarray]
        Validated targets, patients and score arrays.

    Raises
    ------
    ValueError
        If targets, identities or scores violate alignment or finiteness.
    """
    y = np.asarray(arrays["targets"])
    patients = np.asarray(arrays["patient_ids"])
    if y.ndim != 1 or not len(y) or patients.ndim != 1 or len(patients) != len(y):
        raise ValueError("Targets and patient IDs must be aligned nonempty vectors")
    if not np.array_equal(np.unique(y), [0, 1]):
        raise ValueError("Both binary target classes are required")
    if np.issubdtype(patients.dtype, np.number) and not np.isfinite(patients).all():
        raise ValueError("Patient IDs must be finite")
    if "record_ids" in arrays:
        records = np.asarray(arrays["record_ids"])
        if records.ndim != 1 or len(records) != len(y) or len(np.unique(records)) != len(y):
            raise ValueError("Record IDs must be aligned and unique")
    scores = {name: np.asarray(arrays[name]) for name in names}
    if any(values.shape != (len(SEEDS), len(y)) for values in scores.values()):
        raise ValueError("Scores must be aligned [three seeds, records] matrices")
    if any(not np.isfinite(values).all() for values in scores.values()):
        raise ValueError("Scores must be finite")
    return {"targets": y, "patient_ids": patients, **scores}


def _seed_scores(arrays: dict[str, np.ndarray], names: list[str], rows: np.ndarray) -> dict[str, np.ndarray]:
    """Score each realized seed independently on one shared record draw.

    Parameters
    ----------
    arrays : dict[str, np.ndarray]
        Validated targets and three-seed score matrices.
    names : list[str]
        Matrices whose AUROCs are needed.
    rows : np.ndarray
        Shared sampled record positions, including bootstrap duplicates.

    Returns
    -------
    dict[str, np.ndarray]
        Three AUROCs for every score key.
    """
    y = arrays["targets"][rows]
    return {
        name: np.asarray([roc_auc_score(y, probability[rows]) for probability in arrays[name]])
        for name in names
    }


def _contrasts(
    scores: dict[str, np.ndarray], keys: dict[str, tuple[str, str, str, str]]
) -> dict[str, np.ndarray]:
    """Compute per-seed encoder advantage in xLSTM minus its GRU advantage.

    Parameters
    ----------
    scores : dict[str, np.ndarray]
        AUROCs in fixed training-seed order.
    keys : dict[str, tuple[str, str, str, str]]
        Four score keys for each interaction.

    Returns
    -------
    dict[str, np.ndarray]
        Three difference-of-differences values per interaction.
    """
    return {
        name: (scores[first] - scores[second]) - (scores[third] - scores[fourth])
        for name, (first, second, third, fourth) in keys.items()
    }


def analyze_interactions(
    arrays: dict[str, np.ndarray], draws: int = 2000, seed: int = 39045
) -> dict[str, Any]:
    """Estimate eight exploratory AUROC interactions on shared patient draws.

    Each interaction averages three realized-seed values of
    (encoder xLSTM - CNN xLSTM) - (encoder GRU - CNN GRU). Scores are never
    ensembled. Uncertainty conditions on the realized training seeds.

    Parameters
    ----------
    arrays : dict[str, np.ndarray]
        Aligned development targets, patient identities and [3, records] scores
        from the primary ``load_cells`` function.
    draws : int
        Number of whole-patient bootstrap draws, including skipped draws.
    seed : int
        Seed of the one shared ``numpy.random.default_rng`` draw stream.

    Returns
    -------
    dict[str, Any]
        Mean seed interactions, percentile 95% intervals and skipped-draw count.
        Positive values favor the encoder's relative advantage under xLSTM.

    Raises
    ------
    ValueError
        If arrays are invalid, draws is not positive, or all draws are skipped.
    """
    if not isinstance(draws, (int, np.integer)) or isinstance(draws, bool) or draws <= 0:
        raise ValueError("draws must be a positive integer")
    keys = _interaction_keys()
    names = sorted({key for values in keys.values() for key in values})
    aligned = _validated(arrays, names)
    observed = _contrasts(_seed_scores(aligned, names, np.arange(len(aligned["targets"]))), keys)
    bootstrap: dict[str, list[float]] = {name: [] for name in keys}
    rng = np.random.default_rng(seed)
    groups = patient_groups(aligned["patient_ids"])
    for rows in two_class_resamples(groups, aligned["targets"], draws, rng):
        contrasts = _contrasts(_seed_scores(aligned, names, rows), keys)
        for name, values in contrasts.items():
            bootstrap[name].append(float(values.mean()))
    accepted = len(next(iter(bootstrap.values())))
    if not accepted:
        raise ValueError("Every interaction bootstrap draw had a single class")
    interactions = {}
    for name, values in observed.items():
        low, high = np.percentile(bootstrap[name], [2.5, 97.5])
        interactions[name] = {
            "difference": float(values.mean()),
            "seed_differences": values.tolist(),
            "seed_sd": float(values.std(ddof=1)),
            "ci_low": float(low),
            "ci_high": float(high),
        }
    return {
        "interactions": interactions,
        "draws": int(draws),
        "seed": int(seed),
        "accepted_draws": accepted,
        "skipped_draws": int(draws) - accepted,
        "training_seeds": list(SEEDS),
        "status": "exploratory",
        "primary_decision": False,
        "formula": "mean_seed[(encoder_xlstm - cnn_xlstm) - (encoder_gru - cnn_gru)]",
        "interval_method": "paired whole-patient bootstrap, percentile 95%",
        "uncertainty_scope": "patient sampling conditional on three realized training seeds",
    }


def write_interactions(root: Path = ROOT) -> dict[str, Any]:
    """Write source-bound supplementary interactions from eighteen audited cells.

    Existing receipts are returned only if all cell and executable source hashes
    are unchanged. This analysis does not select models or alter primary results.

    Parameters
    ----------
    root : Path
        Repository root containing all eighteen audited original study cells.

    Returns
    -------
    dict[str, Any]
        The immutable supplementary development-only interaction receipt.

    Raises
    ------
    ValueError
        If audited cells are incomplete or an existing receipt's inputs or
        executable sources changed.
    """
    arrays, hashes = load_cells(root)
    code_root = Path(__file__).parent.parent
    sources = {to_stored(code_root / name): sha256_file(code_root / name) for name in SOURCE_FILES}
    destination = root / OUTPUT / "interactions.json"
    if destination.exists():
        saved = json.loads(destination.read_text())
        if saved["input_sha256"] != hashes or saved["source_sha256"] != sources:
            raise ValueError("Completed interaction receipt inputs or sources changed")
        return saved
    result = {
        "status": "complete_exploratory_interactions_development_only",
        "analysis_role": "exploratory",
        "input_sha256": hashes,
        "source_sha256": sources,
        "tiers": list(TIERS),
        "encoders": list(ENCODERS),
        "contexts": list(CONTEXTS),
        "budgets": list(BUDGETS),
        "training_seeds": list(SEEDS),
        "audited_cells": 18,
        "primary_decision": False,
        "model_selection": False,
        "calibration_or_test_scored": False,
        "inference": analyze_interactions(arrays),
    }
    write_json_atomic(destination, result, sort_keys=True)
    return result
