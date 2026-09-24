"""Experiment 014: patient-aligned cached JEPA/ordinary-CPC logit fusion."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold

from ecg_experiment.data import read_manifest
from ecg_experiment.evaluation import metrics, partition_validation, patient_bootstrap, select_threshold
from ecg_experiment.files import read_csv, sha256_file

ROOT = Path(__file__).resolve().parents[2]
PROTOCOL = ROOT / "docs/experiment-014-fusion.md"
FULL = ROOT / "data/processed/pretrained/ecg-jepa-full-public"
LIMITED = ROOT / "data/processed/ptbxl/features_jepa_multiblock_union_seeds42_43_44"
CPC = ROOT / "outputs/experiment009_cpc_prediction_mismatch/features"
OUTPUT = ROOT / "outputs/experiment014_jepa_cpc_fusion"
LIMITED_UNION_MANIFEST = ROOT / "data/processed/ptbxl/probe_union_seeds42_43_44/labeled_train.csv"
GRID = (0.0, 0.25, 0.5, 0.75, 1.0)
FOLDS = 5
FOLD_SEED = 14042
BOOTSTRAP_DRAWS = 300
TARGET_SENSITIVITY = 0.95
AUROC_TOLERANCE = 1e-7
JEPA_DIMENSION = 768
CPC_SHAPE = (19126, 3, 512)
TEST_BOOTSTRAP_REPEATS = 500
TEST_BOOTSTRAP_SEED = 2026
CALIBRATION_C = 1e6
REQUIRED_SPECIFICITY_GAIN = 0.02
AUROC_LOSS_LIMIT = 0.002


def utc_now() -> str:
    """
    Return the current UTC time for coordination records.

    Returns
    -------
    str
        ISO 8601 timestamp.
    """
    return datetime.now(UTC).isoformat()


def atomic_json(path: str | Path, value: Any) -> None:
    """
    Write sorted, indented JSON through a per-process temporary file.

    Parameters
    ----------
    path : str | Path
        Destination file; parent directories are created.
    value : Any
        JSON-serializable value without NaN or infinity.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    try:
        temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def rows_by_id(rows: list[dict[str, str]], name: str) -> dict[int, dict[str, str]]:
    """
    Index rows by integer ECG ID.

    Parameters
    ----------
    rows : list[dict[str, str]]
        Rows with ``ecg_id``.
    name : str
        Row source, used in error messages.

    Returns
    -------
    dict[int, dict[str, str]]
        Rows keyed by ECG ID.

    Raises
    ------
    ValueError
        If an ECG ID repeats.
    """
    index = {}
    for row in rows:
        identifier = int(row["ecg_id"])
        if identifier in index:
            raise ValueError(f"Duplicate ECG ID in {name}: {identifier}")
        index[identifier] = row
    return index


def patient_sets(parts: dict[str, list[dict[str, str]]]) -> None:
    """
    Require disjoint patients across partitions.

    Parameters
    ----------
    parts : dict[str, list[dict[str, str]]]
        Rows keyed by partition name.

    Raises
    ------
    ValueError
        If two partitions share a patient.
    """
    sets = {name: {str(row["patient_id"]) for row in rows} for name, rows in parts.items()}
    for name, values in sets.items():
        for other, other_values in sets.items():
            if name < other and values & other_values:
                raise ValueError(f"Patient overlap between {name} and {other}")


def partition_ids(name: str, rows: list[dict[str, str]], jepa_index: dict[int, int],
                  cpc_index: dict[int, dict[str, str]], budget: str) -> list[int]:
    """
    Check one partition against both feature caches and return its ECG IDs.

    Parameters
    ----------
    name : str
        Partition name; ``train`` and ``test`` map to the same CPC split.
    rows : list[dict[str, str]]
        Partition rows.
    jepa_index : dict[int, int]
        JEPA feature row of each ECG ID.
    cpc_index : dict[int, dict[str, str]]
        CPC feature-row identity of each ECG ID.
    budget : str
        Label budget, used in error messages.

    Returns
    -------
    list[int]
        ECG IDs in row order.

    Raises
    ------
    ValueError
        If an ECG is missing, misaligned, unlabeled or duplicated.
    """
    expected = "train" if name == "train" else "test" if name == "test" else "validation"
    ids = []
    for row in rows:
        identifier = int(row["ecg_id"])
        if identifier not in jepa_index or identifier not in cpc_index:
            raise ValueError(f"Missing {name} ECG {identifier} in {budget} features")
        cached = cpc_index[identifier]
        if str(cached["patient_id"]) != str(row["patient_id"]) or cached["split"] != expected:
            raise ValueError(f"CPC patient/split mismatch for ECG {identifier}")
        if row["target"] not in ("0", "1"):
            raise ValueError(f"Nonbinary {name} label for ECG {identifier}")
        ids.append(identifier)
    if len(set(ids)) != len(ids):
        raise ValueError(f"Duplicate ECG ID in {name}")
    return ids


def check_alignment(parts: dict[str, list[dict[str, str]]], jepa_index: dict[int, int],
                    cpc_rows: list[dict[str, str]], budget: str) -> dict[int, tuple[int, dict[str, str]]]:
    """
    Join partitions to both feature caches by ECG ID, patient and split, not row order.

    Parameters
    ----------
    parts : dict[str, list[dict[str, str]]]
        Rows keyed by partition name.
    jepa_index : dict[int, int]
        JEPA feature row of each ECG ID.
    cpc_rows : list[dict[str, str]]
        CPC feature-row identities in cache order.
    budget : str
        Label budget, used in error messages.

    Returns
    -------
    dict[int, tuple[int, dict[str, str]]]
        CPC cache position and row of each ECG ID.

    Raises
    ------
    ValueError
        If partitions overlap or disagree with the caches.
    """
    cpc_index = rows_by_id(cpc_rows, "CPC feature rows")
    if len(jepa_index) != len(set(jepa_index)):
        raise ValueError("Duplicate JEPA feature ECG IDs")
    patient_sets(parts)
    ids_by_partition = {name: partition_ids(name, rows, jepa_index, cpc_index, budget)
                        for name, rows in parts.items()}
    names = list(ids_by_partition)
    for i, left in enumerate(names):
        for right in names[i + 1:]:
            if set(ids_by_partition[left]) & set(ids_by_partition[right]):
                raise ValueError(f"ECG overlap between {left} and {right}")
    return {int(row["ecg_id"]): (i, row) for i, row in enumerate(cpc_rows)}


def linear_logits(x: np.ndarray, path: Path) -> np.ndarray:
    """
    Apply a saved standardized linear probe after checking its parameters.

    Parameters
    ----------
    x : np.ndarray
        Raw features.
    path : Path
        ``.npz`` with ``mean``, ``scale``, ``coefficient`` and ``intercept``.

    Returns
    -------
    np.ndarray
        Float64 logits.

    Raises
    ------
    ValueError
        If the probe's dimensions disagree with the features or it is invalid.
    """
    with np.load(path) as saved:
        mean, scale = saved["mean"], saved["scale"]
        coef, intercept = saved["coefficient"].reshape(-1), float(saved["intercept"].reshape(-1)[0])
    if x.shape[1] != len(mean) or len(mean) != len(scale) or len(coef) != len(mean):
        raise ValueError(f"Probe dimensions disagree with features: {path}")
    if np.any(scale <= 0) or not np.isfinite(mean).all() or not np.isfinite(coef).all():
        raise ValueError(f"Invalid saved probe: {path}")
    return ((np.asarray(x, dtype=np.float64) - mean) / scale) @ coef + intercept


def normalized_train_logits(logits: np.ndarray | list[float],
                            train_count: int) -> tuple[np.ndarray, dict[str, float | int]]:
    """
    Standardize logits with the mean and std of the labeled training rows only.

    Parameters
    ----------
    logits : np.ndarray | list[float]
        Training logits followed by development logits.
    train_count : int
        Number of leading training logits.

    Returns
    -------
    tuple[np.ndarray, dict[str, float | int]]
        Standardized logits and the training statistics.

    Raises
    ------
    ValueError
        If training logits are nonfinite or constant.
    """
    train = np.asarray(logits[:train_count], dtype=np.float64)
    mean, std = float(train.mean()), float(train.std(ddof=0))
    if not np.isfinite(mean) or not np.isfinite(std) or std <= 0:
        raise ValueError("Nonfinite or constant labeled-training logits")
    return (np.asarray(logits, dtype=np.float64) - mean) / std, {"mean": mean, "std": std,
                                                                 "training_records": train_count}


def development_folds(y: np.ndarray, patients: list[str], seed: int = FOLD_SEED) -> np.ndarray:
    """
    Assign development rows to stratified patient-grouped folds.

    Parameters
    ----------
    y : np.ndarray
        Binary labels.
    patients : list[str]
        Patient of each row.
    seed : int
        Fold shuffling seed.

    Returns
    -------
    np.ndarray
        Fold index of each row.

    Raises
    ------
    ValueError
        If a fold lacks a class or a patient spans folds.
    """
    groups = np.asarray([str(value) for value in patients])
    y = np.asarray(y, dtype=int)
    splitter = StratifiedGroupKFold(n_splits=FOLDS, shuffle=True, random_state=seed)
    fold = np.full(len(y), -1, dtype=int)
    for index, (training, heldout) in enumerate(splitter.split(np.zeros(len(y)), y, groups)):
        if len(set(y[training])) != 2 or len(set(y[heldout])) != 2:
            raise ValueError("Development fold lacks a class")
        fold[heldout] = index
    if np.any(fold < 0) or any(len(set(fold[groups == patient])) != 1 for patient in set(groups)):
        raise ValueError("Malformed patient-group folds")
    return fold


def dev_score(y: np.ndarray, logit: np.ndarray, fold: np.ndarray) -> dict[str, Any]:
    """
    Development AUROC and cross-fitted 95%-sensitivity operating point.

    Each fold's inclusive threshold comes from the other folds' positives.

    Parameters
    ----------
    y : np.ndarray
        Binary labels.
    logit : np.ndarray
        Scores.
    fold : np.ndarray
        Fold index of each row.

    Returns
    -------
    dict[str, Any]
        AUROC, pooled sensitivity and specificity, and each fold's threshold.

    Raises
    ------
    ValueError
        If labels or scores are invalid or a fold lacks a class.
    """
    y = np.asarray(y, dtype=int)
    logit = np.asarray(logit, dtype=np.float64)
    fold = np.asarray(fold, dtype=int)
    if set(y) != {0, 1} or not np.isfinite(logit).all() or len(fold) != len(y):
        raise ValueError("Development labels or logits invalid")
    pred = np.zeros(len(y), dtype=bool)
    thresholds = {}
    for index in sorted(set(fold)):
        heldout = fold == index
        training = ~heldout
        if set(y[training]) != {0, 1} or set(y[heldout]) != {0, 1}:
            raise ValueError("Cross-fitted fold lacks a class")
        positive = np.sort(logit[training & (y == 1)])
        required = int(np.ceil(TARGET_SENSITIVITY * len(positive) - 1e-12))
        threshold = float(positive[len(positive) - required])
        thresholds[str(index)] = threshold
        pred[heldout] = logit[heldout] >= threshold
    return {"auroc": float(roc_auc_score(y, logit)),
            "sensitivity": float(np.mean(pred[y == 1])),
            "specificity": float(np.mean(~pred[y == 0])),
            "crossfit_thresholds": thresholds}


def select(scores: dict[float, dict[str, Any]]) -> float:
    """
    Choose the fusion weight by specificity, then AUROC, then closeness to equal weighting.

    Parameters
    ----------
    scores : dict[float, dict[str, Any]]
        Development score of each grid weight.

    Returns
    -------
    float
        Selected JEPA weight.
    """
    return max(GRID, key=lambda a: (scores[a]["specificity"], scores[a]["auroc"], -abs(a - .5), -a))


def crossfit_has_both_classes(y: np.ndarray, fold: np.ndarray) -> bool:
    """
    Check that every cross-fitting split of a sample has both classes.

    Parameters
    ----------
    y : np.ndarray
        Binary labels.
    fold : np.ndarray
        Fold index of each row.

    Returns
    -------
    bool
        True when ``dev_score`` can score the sample.
    """
    both = {0, 1}
    return set(y) == both and all(set(y[fold == index]) == both and set(y[fold != index]) == both
                                  for index in set(fold))


def bootstrap_screen(y: np.ndarray, je: np.ndarray, cp: np.ndarray, patients: list[str], fold: np.ndarray,
                     chosen: float, draws: int = BOOTSTRAP_DRAWS, seed: int = FOLD_SEED) -> dict[str, Any]:
    """
    Resample development patients to test how stable the selected weight and its gain are.

    Draws in which a cross-fitting split lacks a class are skipped.

    Parameters
    ----------
    y : np.ndarray
        Binary labels.
    je : np.ndarray
        Standardized JEPA logits.
    cp : np.ndarray
        Standardized CPC logits.
    patients : list[str]
        Patient of each row.
    fold : np.ndarray
        Fold index of each row.
    chosen : float
        Weight selected on the full development set.
    draws : int
        Bootstrap draws.
    seed : int
        Resampling seed.

    Returns
    -------
    dict[str, Any]
        Selection frequencies and the chosen weight's specificity gain.

    Raises
    ------
    ValueError
        If no draw could be scored.
    """
    groups = {}
    for i, patient in enumerate(patients):
        groups.setdefault(str(patient), []).append(i)
    group_indices = list(groups.values())
    rng = np.random.default_rng(seed)
    chosen_weights, gains = [], []
    for _ in range(draws):
        drawn = rng.integers(len(group_indices), size=len(group_indices))
        sample = np.concatenate([group_indices[i] for i in drawn])
        if not crossfit_has_both_classes(np.asarray(y[sample], dtype=int),
                                         np.asarray(fold[sample], dtype=int)):
            continue
        scores = {a: dev_score(y[sample], a * je[sample] + (1 - a) * cp[sample], fold[sample]) for a in GRID}
        chosen_weights.append(select(scores))
        endpoint = max(scores[0.0]["specificity"], scores[1.0]["specificity"])
        gains.append(scores[chosen]["specificity"] - endpoint)
    if not chosen_weights:
        raise ValueError("No valid patient bootstrap draws")
    return {"seed": seed, "valid_draws": len(chosen_weights),
            "selected_weight_counts": {str(a): chosen_weights.count(a) for a in GRID},
            "interior_selection_fraction": float(np.mean(np.isin(chosen_weights, GRID[1:-1]))),
            "point_weight_positive_gain_fraction": float(np.mean(np.asarray(gains) > 0)),
            "point_weight_gain_ci95": [float(v) for v in np.quantile(gains, [.025, .975])]}


def gate(scores: dict[float, dict[str, Any]], chosen: float, bootstrap: dict[str, Any]) -> dict[str, Any]:
    """
    Apply the prespecified development gate before calibration or test data are opened.

    Parameters
    ----------
    scores : dict[float, dict[str, Any]]
        Development score of each grid weight.
    chosen : float
        Selected weight.
    bootstrap : dict[str, Any]
        Output of ``bootstrap_screen``.

    Returns
    -------
    dict[str, Any]
        Pass flag, individual checks, and differences from the best single model.
    """
    endpoint = max(scores[0.0]["specificity"], scores[1.0]["specificity"])
    best_auc = max(scores[0.0]["auroc"], scores[1.0]["auroc"])
    adjacent = [a for a in GRID[1:-1] if abs(a - chosen) == .25]
    checks = {"interior_weight": chosen in GRID[1:-1],
              "specificity_gain_ge_0.02":
                  scores[chosen]["specificity"] - endpoint >= REQUIRED_SPECIFICITY_GAIN,
              "auroc_loss_le_0.002": scores[chosen]["auroc"] >= best_auc - AUROC_LOSS_LIMIT,
              "adjacent_interior_gain": any(scores[a]["specificity"] > endpoint for a in adjacent),
              "bootstrap_interior_ge_half": bootstrap["interior_selection_fraction"] >= .5,
              "bootstrap_gain_fraction_ge_0.75": bootstrap["point_weight_positive_gain_fraction"] >= .75}
    return {"pass": all(checks.values()), "checks": checks,
            "specificity_gain_vs_best_endpoint": scores[chosen]["specificity"] - endpoint,
            "auroc_difference_vs_best_endpoint": scores[chosen]["auroc"] - best_auc}


def verify_feature_hashes(jepa_dir: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    """
    Check both feature caches against their extraction receipts.

    Parameters
    ----------
    jepa_dir : Path
        JEPA feature cache.

    Returns
    -------
    tuple[dict[str, Any], dict[str, Any]]
        JEPA and CPC cache metadata.

    Raises
    ------
    ValueError
        If either cache differs from its receipt.
    """
    metadata = json.loads((jepa_dir / "metadata.json").read_text())
    cpc_metadata = json.loads((CPC / "metadata.json").read_text())
    if cpc_metadata["sha256"] != {name: sha256_file(CPC / name) for name in ("features.npy", "rows.csv")}:
        raise ValueError("CPC cache checksum differs from extraction receipt")
    if metadata["model"] != "ecg-jepa-multiblock" or metadata["feature_dimension"] != JEPA_DIMENSION:
        raise ValueError("JEPA cache provenance differs")
    return metadata, cpc_metadata


def budget_paths(budget: str) -> dict[str, Path]:
    """
    Input locations of one label budget.

    Parameters
    ----------
    budget : str
        ``"full"`` or ``"ten_percent"``.

    Returns
    -------
    dict[str, Path]
        Manifest directory, JEPA cache, probes and configurations.
    """
    full = budget == "full"
    jepa_probe = ROOT / ("outputs/experiment002_public_labels/ecg-jepa_linear_seed42" if full
                         else "outputs/experiment001/ecg-jepa_linear_seed42")
    manifest = "seed42_fraction1" if full else "seed42_fraction0.1"
    return {"manifest": ROOT / "data/processed/ptbxl" / manifest,
            "jepa_dir": FULL if full else LIMITED, "jepa_probe": jepa_probe,
            "jepa_model": (jepa_probe / "linear_model.npz" if full
                           else OUTPUT / "matched_jepa_ten_percent.npz"),
            "jepa_config": jepa_probe / "config.json" if full else OUTPUT / "matched_jepa_ten_percent.json",
            "cpc_probe": ROOT / f"outputs/experiment009_cpc_prediction_mismatch/ordinary_{budget}_seed42"}


def check_budget_provenance(budget: str, paths: dict[str, Path], jepa_meta: dict[str, Any]) -> dict[str, Any]:
    """
    Check that both probes were trained on this budget's exact label manifests.

    Parameters
    ----------
    budget : str
        ``"full"`` or ``"ten_percent"``.
    paths : dict[str, Path]
        Output of ``budget_paths``.
    jepa_meta : dict[str, Any]
        JEPA cache metadata.

    Returns
    -------
    dict[str, Any]
        The CPC probe configuration.

    Raises
    ------
    ValueError
        If a manifest or probe differs from its recorded provenance.
    """
    manifest = paths["manifest"]
    if budget == "full":
        if sha256_file(manifest / "labeled_train.csv") != jepa_meta["manifest_sha256"]["labeled_train.csv"]:
            raise ValueError("Full JEPA probe label IDs differ from designated budget manifest")
    else:
        if sha256_file(LIMITED_UNION_MANIFEST) != jepa_meta["manifest_sha256"]["labeled_train.csv"]:
            raise ValueError("Limited JEPA union extraction manifest changed")
        matched = json.loads(paths["jepa_config"].read_text())
        if (matched["exact_labeled_manifest_sha256"] != sha256_file(manifest / "labeled_train.csv")
                or matched["model_sha256"] != sha256_file(paths["jepa_model"])):
            raise ValueError("Matched limited JEPA probe provenance differs")
    if sha256_file(manifest / "validation.csv") != jepa_meta["manifest_sha256"]["validation.csv"]:
        raise ValueError("JEPA validation manifest changed")
    if sha256_file(manifest / "test.csv") != jepa_meta["manifest_sha256"]["test.csv"]:
        raise ValueError("JEPA test manifest changed")
    config = json.loads((paths["cpc_probe"] / "config.json").read_text())
    if (config["fingerprint"]["inputs"]["manifests"][budget]["labeled_train"]
            != sha256_file(manifest / "labeled_train.csv")):
        raise ValueError("CPC probe label IDs differ from designated budget manifest")
    if config["arm"] != "ordinary" or config["fingerprint"]["budget"] != budget:
        raise ValueError("Wrong CPC probe")
    return config


def load_caches(jepa_dir: Path, jepa_meta: dict[str, Any]) -> tuple[dict[int, int], np.ndarray, np.ndarray,
                                                                     list[dict[str, str]]]:
    """
    Open both feature caches and check their shapes.

    Parameters
    ----------
    jepa_dir : Path
        JEPA feature cache.
    jepa_meta : dict[str, Any]
        JEPA cache metadata.

    Returns
    -------
    tuple[dict[int, int], np.ndarray, np.ndarray, list[dict[str, str]]]
        JEPA row of each ECG ID, JEPA features, CPC features and CPC row identities.

    Raises
    ------
    ValueError
        If a cache is malformed or has duplicate IDs.
    """
    cpc_rows = read_csv(CPC / "rows.csv")
    jepa_ids = np.load(jepa_dir / "ecg_ids.npy")
    if len(set(map(int, jepa_ids))) != len(jepa_ids):
        raise ValueError("Duplicate JEPA ECG IDs")
    jepa_index = {int(value): i for i, value in enumerate(jepa_ids)}
    jepa_x = np.load(jepa_dir / "features.npy", mmap_mode="r")
    cpc_x = np.load(CPC / "features.npy", mmap_mode="r")
    if jepa_x.shape != (jepa_meta["record_count"], JEPA_DIMENSION) or len(jepa_ids) != len(jepa_x):
        raise ValueError("Malformed JEPA cache")
    if cpc_x.shape != CPC_SHAPE or len(cpc_rows) != len(cpc_x):
        raise ValueError("Malformed CPC cache")
    return jepa_index, jepa_x, cpc_x, cpc_rows


def reproduce_auroc(name: str, y: np.ndarray, logits: np.ndarray, expected: float) -> float:
    """
    Recompute a saved probe's development AUROC and require it to match.

    Parameters
    ----------
    name : str
        Probe name, used in the error message.
    y : np.ndarray
        Development labels.
    logits : np.ndarray
        Development logits.
    expected : float
        Recorded development AUROC.

    Returns
    -------
    float
        Reproduced AUROC.

    Raises
    ------
    ValueError
        If the AUROC differs beyond ``AUROC_TOLERANCE``.
    """
    auc = float(roc_auc_score(y, logits))
    if abs(auc - expected) > AUROC_TOLERANCE:
        raise ValueError(f"{name} saved probe does not reproduce development AUROC: {auc}")
    return auc


def load_budget(budget: str) -> dict[str, Any]:
    """
    Load one budget's training and development logits after every provenance check.

    Only training and development labels are loaded before the development gate.

    Parameters
    ----------
    budget : str
        ``"full"`` or ``"ten_percent"``.

    Returns
    -------
    dict[str, Any]
        Paths, rows, caches, standardized development logits, input digests
        and reproduced probe scores.
    """
    paths = budget_paths(budget)
    manifest = paths["manifest"]
    jepa_meta, _ = verify_feature_hashes(paths["jepa_dir"])
    train = read_manifest(manifest / "labeled_train.csv")
    validation = read_manifest(manifest / "validation.csv")
    development, calibration = partition_validation(validation)
    config = check_budget_provenance(budget, paths, jepa_meta)
    jepa_index, jepa_x, cpc_x, cpc_rows = load_caches(paths["jepa_dir"], jepa_meta)
    parts = {"train": train, "development": development, "calibration": calibration}
    cpc_index = check_alignment(parts, jepa_index, cpc_rows, budget)
    ordered = train + development
    ji = [jepa_index[int(r["ecg_id"])] for r in ordered]
    ci = [cpc_index[int(r["ecg_id"])][0] for r in ordered]
    jepa_logit = linear_logits(jepa_x[ji], paths["jepa_model"])
    cpc_logit = linear_logits(np.concatenate((cpc_x[ci, 0], cpc_x[ci, 1]), axis=1),
                              paths["cpc_probe"] / "linear_model.npz")
    jepa_config = json.loads(paths["jepa_config"].read_text())
    split = len(train)
    y = np.array([int(r["target"]) for r in development])
    expected_jepa = jepa_config["best_development_auroc" if budget == "full" else "development_auroc"]
    jepa_auc = reproduce_auroc("JEPA", y, jepa_logit[split:], expected_jepa)
    cpc_auc = reproduce_auroc("CPC", y, cpc_logit[split:], config["best_development_auroc"])
    jepa_norm, js = normalized_train_logits(jepa_logit, split)
    cpc_norm, cs = normalized_train_logits(cpc_logit, split)
    jepa_dir, cpc_probe = paths["jepa_dir"], paths["cpc_probe"]
    files = [PROTOCOL, Path(__file__),
             jepa_dir / "features.npy", jepa_dir / "ecg_ids.npy", jepa_dir / "metadata.json",
             CPC / "features.npy", CPC / "rows.csv", CPC / "metadata.json",
             paths["jepa_model"], paths["jepa_config"],
             cpc_probe / "linear_model.npz", cpc_probe / "selection.json", cpc_probe / "config.json",
             manifest / "labeled_train.csv", manifest / "validation.csv", manifest / "test.csv"]
    if budget == "ten_percent":
        files.append(LIMITED_UNION_MANIFEST)
    return {"budget": budget, "manifest": manifest, "jepa_dir": jepa_dir, "jepa_probe": paths["jepa_probe"],
            "jepa_model": paths["jepa_model"], "cpc_probe": cpc_probe, "train": train,
            "development": development, "calibration": calibration,
            "jepa_index": jepa_index, "cpc_rows": cpc_rows, "jepa_x": jepa_x, "cpc_x": cpc_x,
            "jepa_dev": jepa_norm[split:], "cpc_dev": cpc_norm[split:], "jepa_stats": js, "cpc_stats": cs,
            "development_y": y, "hashes": {str(path.relative_to(ROOT)): sha256_file(path) for path in files},
            "reproduced_probe_dev_auroc": {"jepa": jepa_auc, "cpc": cpc_auc},
            "records": {"train": len(train), "development": len(development),
                        "calibration": len(calibration)}}


def evaluate_if_pass(data: dict[str, Any], chosen: float) -> dict[str, Any]:
    """
    Calibrate the fused score and evaluate it on test patients.

    Opening test rows and logits occurs only after the development gate passes.

    Parameters
    ----------
    data : dict[str, Any]
        Output of ``load_budget``.
    chosen : float
        Selected JEPA weight.

    Returns
    -------
    dict[str, Any]
        Calibration summary, test metrics and patient-bootstrap intervals.

    Raises
    ------
    ValueError
        If the Platt calibration slope is not positive.
    """
    test = read_manifest(data["manifest"] / "test.csv")
    parts = {"train": data["train"], "development": data["development"],
             "calibration": data["calibration"], "test": test}
    cpc_index = check_alignment(parts, data["jepa_index"], data["cpc_rows"], data["budget"])
    rows = data["calibration"] + test
    ji = [data["jepa_index"][int(r["ecg_id"])] for r in rows]
    ci = [cpc_index[int(r["ecg_id"])][0] for r in rows]
    je = linear_logits(data["jepa_x"][ji], data["jepa_model"])
    cp_x = np.concatenate((data["cpc_x"][ci, 0], data["cpc_x"][ci, 1]), axis=1)
    cp = linear_logits(cp_x, data["cpc_probe"] / "linear_model.npz")
    je = (je - data["jepa_stats"]["mean"]) / data["jepa_stats"]["std"]
    cp = (cp - data["cpc_stats"]["mean"]) / data["cpc_stats"]["std"]
    fused = chosen * je + (1 - chosen) * cp
    ncal = len(data["calibration"])
    cal_y = np.array([int(r["target"]) for r in data["calibration"]])
    test_y = np.array([int(r["target"]) for r in test])
    calibrator = LogisticRegression(C=CALIBRATION_C, solver="lbfgs", max_iter=1000)
    calibrator.fit(fused[:ncal, None], cal_y)
    slope = float(calibrator.coef_[0, 0])
    if slope <= 0:
        raise ValueError("Nonpositive calibration slope")
    cal_p = calibrator.predict_proba(fused[:ncal, None])[:, 1]
    test_p = calibrator.predict_proba(fused[ncal:, None])[:, 1]
    threshold = select_threshold(cal_y, cal_p, TARGET_SENSITIVITY)
    return {"records": {"calibration": ncal, "test": len(test)},
            "calibration": {"method": "Platt logistic", "slope": slope,
                            "intercept": float(calibrator.intercept_[0]), "threshold": threshold,
                            "sensitivity": metrics(cal_y, cal_p, threshold)["sensitivity"]},
            "test": metrics(test_y, test_p, threshold),
            "test_ci95_patient_bootstrap": patient_bootstrap(test_y, test_p, [r["patient_id"] for r in test],
                                                             threshold, repeats=TEST_BOOTSTRAP_REPEATS,
                                                             seed=TEST_BOOTSTRAP_SEED)}


def screen_budget(data: dict[str, Any], fingerprint: dict[str, Any]) -> dict[str, Any]:
    """
    Score the fusion grid on development patients, apply the gate, and evaluate on pass.

    Parameters
    ----------
    data : dict[str, Any]
        Output of ``load_budget``.
    fingerprint : dict[str, Any]
        Receipt fingerprint.

    Returns
    -------
    dict[str, Any]
        Completed budget receipt.
    """
    y = data["development_y"]
    je, cp = data["jepa_dev"], data["cpc_dev"]
    patients = [r["patient_id"] for r in data["development"]]
    fold = development_folds(y, patients)
    fold_hash = hashlib.sha256(np.asarray(fold, dtype=np.int8).tobytes()).hexdigest()
    scores = {a: dev_score(y, a * je + (1 - a) * cp, fold) for a in GRID}
    chosen = select(scores)
    boot = bootstrap_screen(y, je, cp, patients, fold, chosen)
    decision = gate(scores, chosen, boot)
    result = {"completed": True, "fingerprint": fingerprint, "budget": data["budget"],
              "model_description": "released ECG-JEPA frozen linear probe plus Experiment 009 frozen "
                                   "ordinary local/context CPC probe",
              "records": data["records"], "train_logit_normalization": {"jepa": data["jepa_stats"],
                                                                        "cpc": data["cpc_stats"]},
              "reproduced_probe_dev_auroc": data["reproduced_probe_dev_auroc"],
              "development": {"grid": {str(a): scores[a] for a in GRID}, "selected_alpha": chosen,
                              "folds": FOLDS, "fold_seed": FOLD_SEED, "fold_assignment_sha256": fold_hash,
                              "patient_bootstrap": boot, "gate": decision},
              "calibration_test_opened": decision["pass"]}
    if decision["pass"]:
        result["evaluation"] = evaluate_if_pass(data, chosen)
    return result


def main() -> None:
    """Screen both label budgets, reusing verified completed receipts."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT)
    args = parser.parse_args()
    start = time.monotonic()
    atomic_json(args.output_dir / "coordination.json", {"state": "running", "pid": os.getpid(),
                "started_at_utc": utc_now(), "command": " ".join(sys.argv), "returncode": None})
    receipts = {}
    for budget in ("full", "ten_percent"):
        data = load_budget(budget)
        output = args.output_dir / f"{budget}.json"
        fingerprint = {"input_sha256": data["hashes"], "budget": budget,
                       "normalization": "labeled-training logits only", "alpha_grid": list(GRID)}
        if output.exists():
            saved = json.loads(output.read_text())
            if saved["fingerprint"] != fingerprint or saved.get("completed") is not True:
                raise ValueError(f"Existing {budget} receipt fingerprint differs")
            receipt = {"path": str(output), "sha256": sha256_file(output)}
            print(json.dumps({"budget": budget, "status": "verified_completed", "sha256": receipt["sha256"]}),
                  flush=True)
            receipts[budget] = receipt
            continue
        result = screen_budget(data, fingerprint)
        atomic_json(output, result)
        receipts[budget] = {"path": str(output), "sha256": sha256_file(output)}
        decision = result["development"]["gate"]
        print(json.dumps({"budget": budget, "gate": decision,
                          "selected_alpha": result["development"]["selected_alpha"],
                          "seconds_elapsed": time.monotonic() - start, "sha256": sha256_file(output)}),
              flush=True)
    atomic_json(args.output_dir / "coordination.json", {"state": "complete", "pid": os.getpid(),
                "completed_at_utc": utc_now(), "command": " ".join(sys.argv), "returncode": 0,
                "elapsed_seconds": time.monotonic() - start, "receipts": receipts})


if __name__ == "__main__":
    main()
