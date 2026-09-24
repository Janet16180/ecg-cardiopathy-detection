#!/usr/bin/env python3
"""Compare adaptation AUROCs with paired resampling of test patients.

Intervals are conditional on the fitted models. These exploratory comparisons
do not include retraining variation or a multiplicity adjustment.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score


def predictions(path: Path):
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        if not {"ecg_id", "patient_id", "target", "probability"}.issubset(reader.fieldnames or []):
            raise ValueError(f"Missing prediction fields: {path}")
        rows = list(reader)
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


def compare(input_dir: Path, reference: str, models, repeats=500, seed=2026):
    if repeats < 1:
        raise ValueError("Bootstrap repeats must be positive")
    base = predictions(input_dir / reference / "test_predictions.csv")
    ids = [r["ecg_id"] for r in base]
    targets = np.array([int(r["target"]) for r in base])
    base_probability = np.array([float(r["probability"]) for r in base])
    patients = np.array([r["patient_id"] for r in base])
    groups = [np.flatnonzero(patients == value) for value in np.unique(patients)]
    comparisons = []
    for name in models:
        rows = predictions(input_dir / name / "test_predictions.csv")
        by_id = {r["ecg_id"]: r for r in rows}
        if set(by_id) != set(ids):
            raise ValueError(f"Test ECG sets differ between {name} and {reference}")
        aligned = [by_id[key] for key in ids]
        for baseline_row, model_row in zip(base, aligned):
            if (baseline_row["patient_id"] != model_row["patient_id"]
                    or int(baseline_row["target"]) != int(model_row["target"])):
                raise ValueError(f"Patient identity or target differs for ECG {baseline_row['ecg_id']}")
        probability = np.array([float(r["probability"]) for r in aligned])
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
        comparisons.append({
            "model": name, "reference": reference,
            "auroc_difference": float(roc_auc_score(targets, probability) - roc_auc_score(targets, base_probability)),
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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=Path("outputs/experiment001"))
    parser.add_argument("--reference", default="ecg-fm_finetuned_seed42")
    parser.add_argument("--models", nargs="+", default=["ecg-fm_adapted_finetuned_seed42",
                                                        "ecg-fm_pooled_adapted_finetuned_seed42"])
    parser.add_argument("--repeats", type=int, default=500)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--output-json", type=Path)
    args = parser.parse_args()
    result = compare(args.input_dir, args.reference, args.models, args.repeats, args.seed)
    destination = args.output_json or args.input_dir / "paired_adaptation_comparisons.json"
    destination.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
