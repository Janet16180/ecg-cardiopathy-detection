"""Exact interventional linear-logit explanations on reserved-safe physical measurements."""

from __future__ import annotations

import ast
import csv
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

from ecg_experiment import ROOT
from ecg_experiment.clustering01 import FEATURES, select_cohort, train_patient
from ecg_experiment.eda.ptbxl import diagnostic_classes, project_label, superclass_label
from ecg_experiment.eda.signals import LEADS
from ecg_experiment.files import sha256_file
from ecg_experiment.intervals import patient_groups, two_class_resamples
from ecg_experiment.paths import to_stored

SEED = 20261018
PREDECESSOR_SHA256 = "74f0c82918f158a61368d308acdab5d5e98fe759852c966de76867e5c99bc8e7"
BACKGROUNDS = ("all_training", "strict_normal_training")


def predecessor_hashes(predecessor: Path, metadata: Path) -> list[Path]:
    """Bind the frozen predecessor artifacts and helpers before any content analysis."""
    result_path = predecessor / "result.json"
    if sha256_file(result_path) != PREDECESSOR_SHA256:
        raise ValueError("Unexpected predecessor result bytes")
    previous = json.loads(result_path.read_text())
    paths = [result_path, metadata, metadata.with_name("scp_statements.csv")]
    names = (
        "morphology_features.csv",
        "analyzed_training_patients.csv",
        "selected_training_patients.csv",
        "reservation_manifest.json",
        "reservation_policy.json",
    )
    for name in names:
        path = predecessor / name
        if sha256_file(path) != previous["outputs_sha256"][to_stored(path)]:
            raise ValueError(f"Predecessor artifact changed: {name}")
        paths.append(path)
    for name, digest in previous["input_sha256"].items():
        if sha256_file(ROOT / name) != digest:
            raise ValueError("Predecessor metadata input changed")
    if (
        sha256_file(ROOT / "ecg_experiment/clustering01.py")
        != previous["source_sha256"]["ecg_experiment/clustering01.py"]
    ):
        raise ValueError("Predecessor cohort helper bytes changed")
    return paths


def verified_cohort(predecessor: Path, metadata: Path) -> tuple[pd.DataFrame, pd.DataFrame, set[int], dict]:
    """Verify predecessor bytes, identities and reservations before returning features.

    Parameters
    ----------
    predecessor : Path
        Completed CLUSTER-01 output directory.
    metadata : Path
        Original PTB-XL metadata, read as identities only here.

    Returns
    -------
    tuple
        Features, aligned identities, permitted train ECG IDs and integrity receipt.
    """
    paths = predecessor_hashes(predecessor, metadata)
    identifiers = pd.read_csv(metadata, usecols=["ecg_id", "patient_id", "strat_fold", "filename_hr"])
    selected, counts = select_cohort(identifiers)
    saved_selected = pd.read_csv(predecessor / "selected_training_patients.csv")
    if not selected.equals(saved_selected):
        raise ValueError("Predecessor patient sample does not reproduce")
    manifest = json.loads((predecessor / "reservation_manifest.json").read_text())
    if manifest["metadata_sha256"] != sha256_file(metadata):
        raise ValueError("Reservation metadata bytes differ")
    permitted = {int(row["ecg_id"]) for row in manifest["records"] if row["role"] == "train"}
    closed = set(identifiers.loc[identifiers.strat_fold > 8, "patient_id"].astype(int))
    safe = identifiers.loc[identifiers.strat_fold.between(1, 8)]
    train = safe.loc[~safe.patient_id.astype(int).isin(closed) & safe.patient_id.map(train_patient)]
    if set(train.ecg_id) != permitted or len(permitted) != 13938:
        raise ValueError("Training identities differ from the exact PR67 reservation")
    features = pd.read_csv(predecessor / "morphology_features.csv")
    identities = pd.read_csv(predecessor / "analyzed_training_patients.csv")
    if (
        features.shape != (1999, 53)
        or list(features) != FEATURES
        or len(identities) != 1999
        or not identities.patient_id.is_unique
        or not set(identities.ecg_id) <= permitted
        or not np.isfinite(features.to_numpy()).all()
    ):
        raise ValueError("Unexpected predecessor feature/identity denominator")
    expected = saved_selected.loc[saved_selected.ecg_id.isin(identities.ecg_id)].reset_index(drop=True)
    if not identities.equals(expected) or len(saved_selected) - len(identities) != 1:
        raise ValueError("Predecessor feature rows or single exclusion differ")
    return (
        features,
        identities,
        permitted,
        {
            "predecessor_result_sha256": PREDECESSOR_SHA256,
            "identity_counts": counts,
            "analyzed_patients": 1999,
            "features": 53,
            "excluded_without_replacement": 1,
            "all_train_ecg_identities_exact": True,
            "one_record_per_patient": True,
            "input_sha256": {to_stored(path): sha256_file(path) for path in paths},
        },
    )


def read_labels(
    identities: pd.DataFrame, permitted: set[int], metadata: Path
) -> tuple[np.ndarray, np.ndarray]:
    """Parse diagnoses only after every requested record has a verified train role."""
    allowed = set(identities.ecg_id)
    if not allowed or not allowed <= permitted or not identities.patient_id.is_unique:
        raise ValueError("Diagnosis access requires verified unique training records")
    classes = diagnostic_classes(pd.read_csv(metadata.with_name("scp_statements.csv"), index_col=0))
    labels, strict = {}, {}
    with metadata.open() as stream:
        for row in csv.DictReader(stream):
            ecg = int(row["ecg_id"])
            if ecg not in allowed:
                continue
            codes = ast.literal_eval(row["scp_codes"])
            labels[ecg] = superclass_label(codes, classes)
            strict[ecg] = project_label(codes, classes)
    if set(labels) != allowed:
        raise ValueError("Missing permitted diagnostic annotation")
    return (
        np.array([labels[ecg] for ecg in identities.ecg_id]),
        np.array([strict[ecg] for ecg in identities.ecg_id]),
    )


def training_role(patient: int) -> bool:
    """Return the frozen internal patient allocation without interpreting labels."""
    digest = hashlib.sha256(f"shap01_split_v1:{SEED}:{int(patient)}".encode()).hexdigest()
    return int(digest, 16) % 10000 < 8000


def background_indices(train: np.ndarray, target: np.ndarray, strict: np.ndarray) -> dict[str, np.ndarray]:
    """Select complete model-training and strict-normal subsets, excluding evaluation."""
    eligible = np.asarray(train, bool) & np.isfinite(target)
    indices = {
        "all_training": np.flatnonzero(eligible),
        "strict_normal_training": np.flatnonzero(eligible & (strict == 0)),
    }
    if any(not len(rows) for rows in indices.values()):
        raise ValueError("Both training-only backgrounds must be nonempty")
    return indices


def fit_fixed_model(
    features: np.ndarray,
    target: np.ndarray,
    patients: np.ndarray,
    strict: np.ndarray,
) -> tuple[StandardScaler, LogisticRegression, np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    """Fit one prespecified scaler/classifier on internal training patients only."""
    train_role = np.array([training_role(patient) for patient in patients])
    train = np.flatnonzero(train_role & np.isfinite(target))
    evaluation = np.flatnonzero(~train_role & np.isfinite(target))
    if any(len(np.unique(target[rows])) != 2 for rows in (train, evaluation)):
        raise ValueError("Both internal roles must contain both target classes")
    backgrounds = background_indices(train_role, target, strict)
    scaler = StandardScaler().fit(features[train])
    model = LogisticRegression(
        C=1,
        l1_ratio=0.0,
        class_weight="balanced",
        solver="lbfgs",
        max_iter=2000,
        tol=1e-8,
        random_state=SEED,
    )
    model.fit(scaler.transform(features[train]), target[train].astype(int))
    if int(model.n_iter_.max()) >= model.max_iter:
        raise ValueError("Fixed logistic fit did not converge")
    return scaler, model, train, evaluation, backgrounds


def linear_shap(
    values: np.ndarray,
    coefficient: np.ndarray,
    intercept: float,
    background: np.ndarray,
) -> tuple[float, np.ndarray]:
    """Compute exact Shapley values for the linear-logit independent-intervention game.

    Parameters
    ----------
    values : np.ndarray
        Explained standardized feature rows.
    coefficient : np.ndarray
        One fitted linear logit coefficient per feature.
    intercept : float
        Fitted logit intercept.
    background : np.ndarray
        Nonempty training-only standardized background rows.

    Returns
    -------
    tuple
        Background base logit and exact signed feature contributions.
    """
    if background.ndim != 2 or not len(background) or values.ndim != 2:
        raise ValueError("Explanation and background must be nonempty two-dimensional arrays")
    if values.shape[1] != len(coefficient) or background.shape[1] != len(coefficient):
        raise ValueError("Linear explanation dimensions disagree")
    mean = background.mean(axis=0)
    return float(intercept + coefficient @ mean), (values - mean) * coefficient


def group_matrix(feature_names: list[str]) -> tuple[list[str], np.ndarray]:
    """Assign each canonical lead amplitude once, with global duration/rhythm/axis last."""
    groups = [*LEADS, "global_duration_rhythm_axis"]
    matrix = np.zeros((len(feature_names), len(groups)))
    mapping = {
        f"{kind}_{lead}": index
        for index, lead in enumerate(LEADS)
        for kind in ("r_amp", "s_amp", "st", "t_amp")
    }
    mapping.update(dict.fromkeys(FEATURES[:5], len(groups) - 1))
    if len(set(feature_names)) != 53 or set(feature_names) != set(mapping):
        raise ValueError("Expected exactly the canonical 53 physical features")
    for index, name in enumerate(feature_names):
        matrix[index, mapping[name]] = 1
    return groups, matrix


def reconstruction(base: float, grouped: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Rebuild logits from the five largest positive groups and every other group."""
    top = np.argsort(-grouped, axis=1, kind="stable")[:, :5]
    selected = np.take_along_axis(grouped, top, axis=1)
    positive = np.maximum(selected, 0).sum(axis=1)
    remainder = grouped.sum(axis=1) - positive
    return base + positive + remainder, positive, remainder


def sensitivity(first: np.ndarray, second: np.ndarray, groups: list[str]) -> tuple[dict, pd.DataFrame]:
    """Measure patient-level rank/sign changes without treating them as clinical truth."""
    first_order = np.argsort(-np.abs(first), axis=1, kind="stable")
    second_order = np.argsort(-np.abs(second), axis=1, kind="stable")
    top_first = np.where(first.max(axis=1) > 0, np.argmax(first, axis=1), -1)
    top_second = np.where(second.max(axis=1) > 0, np.argmax(second, axis=1), -1)
    correlations = []
    for left, right in zip(first, second, strict=True):
        a, b = rankdata(np.abs(left)), rankdata(np.abs(right))
        correlations.append(float(np.corrcoef(a, b)[0, 1]) if np.std(a) and np.std(b) else np.nan)
    frame = pd.DataFrame(
        {
            "top_absolute_changed": first_order[:, 0] != second_order[:, 0],
            "top_positive_changed": top_first != top_second,
            "top_absolute_all": [groups[i] for i in first_order[:, 0]],
            "top_absolute_normal": [groups[i] for i in second_order[:, 0]],
            "top_positive_all": [groups[i] if i >= 0 else "none" for i in top_first],
            "top_positive_normal": [groups[i] if i >= 0 else "none" for i in top_second],
            "top3_overlap_fraction": [
                len(set(a[:3]) & set(b[:3])) / 3 for a, b in zip(first_order, second_order, strict=True)
            ],
            "absolute_rank_spearman": correlations,
            "group_sign_changed_fraction": np.mean(first * second < 0, axis=1),
            "group_l1_background_shift": np.sum(np.abs(second - first), axis=1),
        }
    )
    summary = {
        name: float(frame[name].mean())
        for name in (
            "top_absolute_changed",
            "top_positive_changed",
            "top3_overlap_fraction",
            "group_sign_changed_fraction",
            "group_l1_background_shift",
        )
    }
    summary["mean_absolute_rank_spearman"] = (
        float(frame.absolute_rank_spearman.mean()) if frame.absolute_rank_spearman.notna().any() else None
    )
    summary["undefined_rank_correlations"] = int(frame.absolute_rank_spearman.isna().sum())
    summary["group_l1_shift_min_max"] = [
        float(frame.group_l1_background_shift.min()),
        float(frame.group_l1_background_shift.max()),
    ]
    return summary, frame


def auc_interval(patients: np.ndarray, target: np.ndarray, probability: np.ndarray) -> dict[str, Any]:
    """Describe AUROC with the shared whole-patient bootstrap and no operating point."""
    draws = [
        roc_auc_score(target[rows], probability[rows])
        for rows in two_class_resamples(patient_groups(patients), target, 2000, np.random.default_rng(SEED))
    ]
    if not draws:
        raise ValueError("No two-class patient bootstrap draw")
    low, high = np.percentile(draws, [2.5, 97.5])
    return {
        "auroc": float(roc_auc_score(target, probability)),
        "ci_low": float(low),
        "ci_high": float(high),
        "draws": 2000,
        "skipped_draws": 2000 - len(draws),
        "seed": SEED,
    }
