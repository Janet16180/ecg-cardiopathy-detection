#!/usr/bin/env python3
"""Compare adaptation AUROCs with paired resampling of test patients.

Intervals are conditional on the fitted models. These exploratory comparisons
do not include retraining variation or a multiplicity adjustment.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import roc_auc_score

from ecg_experiment.files import read_csv, write_json_atomic

PREDICTION_FIELDS = {"ecg_id", "patient_id", "target", "probability"}
DEFAULT_REPEATS = 500
DEFAULT_SEED = 2026


def predictions(path: Path) -> list[dict[str, str]]:
    """
    Read and validate one model's test predictions.

    Parameters
    ----------
    path : Path
        ``test_predictions.csv`` of a model.

    Returns
    -------
    list[dict[str, str]]
        Prediction rows in file order.

    Raises
    ------
    ValueError
        If fields, identifiers, classes or probabilities are missing or invalid.
    """
    rows = read_csv(path, PREDICTION_FIELDS)
    if not rows or any(not r["ecg_id"] or not r["patient_id"] for r in rows):
        raise ValueError(f"Empty predictions or missing identifiers: {path}")
    if len({r["ecg_id"] for r in rows}) != len(rows):
        raise ValueError(f"Duplicate ECG identifiers: {path}")
    targets = np.array([int(r["target"]) for r in rows])
    probabilities = np.array([float(r["probability"]) for r in rows])
    if set(targets) != {0, 1}:
        raise ValueError(f"Both binary classes are required: {path}")
    if not np.isfinite(probabilities).all() or np.any((probabilities < 0) | (probabilities > 1)):
        raise ValueError(f"Invalid probabilities: {path}")
    return rows


def aligned_probability(base: list[dict[str, str]], rows: list[dict[str, str]], name: str,
                        reference: str) -> np.ndarray:
    """
    Order a model's probabilities like the reference rows, checking patients and targets.

    Parameters
    ----------
    base : list[dict[str, str]]
        Reference prediction rows.
    rows : list[dict[str, str]]
        Compared model's prediction rows.
    name, reference : str
        Model names, for error messages.

    Returns
    -------
    np.ndarray
        Probabilities aligned with ``base``.

    Raises
    ------
    ValueError
        If the ECG sets, patients or targets differ.
    """
    by_id = {r["ecg_id"]: r for r in rows}
    if set(by_id) != {r["ecg_id"] for r in base}:
        raise ValueError(f"Test ECG sets differ between {name} and {reference}")
    aligned = [by_id[r["ecg_id"]] for r in base]
    for baseline_row, model_row in zip(base, aligned, strict=True):
        if (baseline_row["patient_id"] != model_row["patient_id"]
                or int(baseline_row["target"]) != int(model_row["target"])):
            raise ValueError(f"Patient identity or target differs for ECG {baseline_row['ecg_id']}")
    return np.array([float(r["probability"]) for r in aligned])


def bootstrap_differences(targets: np.ndarray, probability: np.ndarray, base_probability: np.ndarray,
                          groups: Sequence[np.ndarray], repeats: int, seed: int) -> list[float]:
    """
    Resample patients and collect AUROC differences, skipping single-class draws.

    Parameters
    ----------
    targets : np.ndarray
        Binary targets.
    probability, base_probability : np.ndarray
        Compared and reference probabilities.
    groups : Sequence[np.ndarray]
        Row indices of each patient.
    repeats : int
        Number of patient resamples.
    seed : int
        Generator seed.

    Returns
    -------
    list[float]
        Model minus reference AUROC for each valid draw.

    Raises
    ------
    ValueError
        If no draw contains both classes.
    """
    # Resetting the seed gives every comparison the same patient draws.
    rng = np.random.default_rng(seed)
    differences = []
    for _ in range(repeats):
        indices = np.concatenate([groups[k] for k in rng.integers(len(groups), size=len(groups))])
        if len(np.unique(targets[indices])) < 2:
            continue
        differences.append(roc_auc_score(targets[indices], probability[indices])
                           - roc_auc_score(targets[indices], base_probability[indices]))
    if not differences:
        raise ValueError("No bootstrap draw contains both classes")
    return differences


def compare(input_dir: Path, reference: str, models: Sequence[str], repeats: int = DEFAULT_REPEATS,
            seed: int = DEFAULT_SEED) -> dict[str, Any]:
    """
    Compare each model's test AUROC with a reference by paired patient bootstrap.

    Parameters
    ----------
    input_dir : Path
        Directory whose model subdirectories hold ``test_predictions.csv``.
    reference : str
        Reference model directory name.
    models : Sequence[str]
        Compared model directory names.
    repeats : int
        Number of patient resamples.
    seed : int
        Generator seed, shared by all comparisons.

    Returns
    -------
    dict[str, Any]
        Method description, cohort size and one entry per comparison.

    Raises
    ------
    ValueError
        If ``repeats`` is not positive or predictions cannot be paired.
    """
    if repeats < 1:
        raise ValueError("Bootstrap repeats must be positive")
    base = predictions(input_dir / reference / "test_predictions.csv")
    targets = np.array([int(r["target"]) for r in base])
    base_probability = np.array([float(r["probability"]) for r in base])
    patients = np.array([r["patient_id"] for r in base])
    groups = [np.flatnonzero(patients == value) for value in np.unique(patients)]
    comparisons = []
    for name in models:
        rows = predictions(input_dir / name / "test_predictions.csv")
        probability = aligned_probability(base, rows, name, reference)
        differences = bootstrap_differences(targets, probability, base_probability, groups, repeats, seed)
        comparisons.append({
            "model": name, "reference": reference,
            "auroc_difference": float(roc_auc_score(targets, probability)
                                      - roc_auc_score(targets, base_probability)),
            "paired_patient_bootstrap_ci95": [float(v) for v in np.quantile(differences, [0.025, 0.975])],
            "valid_bootstrap_resamples": len(differences),
        })
    return {
        "method": (f"{repeats} paired resamples of test patients, retaining every ECG in a sampled patient; "
                   "percentile intervals conditional on the fitted models. ECG IDs, patient IDs, and targets "
                   "are aligned and verified. Single-class draws are omitted. Exploratory contrasts, "
                   "no multiplicity adjustment."),
        "seed": seed, "test_records": len(base), "test_patients": len(groups),
        "comparisons": comparisons,
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """
    Parse the command line.

    Parameters
    ----------
    argv : list[str] | None
        Arguments, or ``None`` for ``sys.argv``.

    Returns
    -------
    argparse.Namespace
        Parsed arguments.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=Path("outputs/experiment001"))
    parser.add_argument("--reference", default="ecg-fm_finetuned_seed42")
    parser.add_argument("--models", nargs="+", default=["ecg-fm_adapted_finetuned_seed42",
                                                        "ecg-fm_pooled_adapted_finetuned_seed42"])
    parser.add_argument("--repeats", type=int, default=DEFAULT_REPEATS)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--output-json", type=Path)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    """
    Write paired adaptation comparisons as JSON and print them.

    Parameters
    ----------
    argv : list[str] | None
        Arguments, or ``None`` for ``sys.argv``.
    """
    args = parse_args(argv)
    result = compare(args.input_dir, args.reference, args.models, args.repeats, args.seed)
    destination = args.output_json or args.input_dir / "paired_adaptation_comparisons.json"
    write_json_atomic(destination, result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
