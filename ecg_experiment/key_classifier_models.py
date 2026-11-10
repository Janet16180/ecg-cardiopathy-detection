"""Matched multiclass and independent-label readouts for KEY-04."""

from __future__ import annotations

import time
import warnings
from typing import Any

import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, log_loss
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

from .condition_keys import GROUPS, unit
from .intervals import patient_groups, patient_resample

FAMILIES = ("joint_logistic", "independent_logistic")
CS = (0.1, 1.0, 10.0)
STATE_BITS = ((np.arange(16)[:, None] & (1 << np.arange(4))) != 0).astype(int)


def encode_states(flags: np.ndarray) -> np.ndarray:
    """
    Encode the existing four memberships without losing co-occurrence.

    Parameters
    ----------
    flags : np.ndarray
        Binary [records, four groups] annotations.

    Returns
    -------
    np.ndarray
        State integers from zero through fifteen.
    """
    flags = np.asarray(flags)
    if flags.ndim != 2 or flags.shape[1] != 4 or not np.isin(flags, (0, 1)).all():
        raise ValueError("Expected four exactly binary memberships")
    return flags.astype(int) @ (1 << np.arange(4))


def marginal_scores(probabilities: np.ndarray) -> np.ndarray:
    """
    Sum disjoint joint-state probabilities into overlapping group probabilities.

    Parameters
    ----------
    probabilities : np.ndarray
        Probability matrix with sixteen aligned state columns.

    Returns
    -------
    np.ndarray
        Four marginal probabilities; their row sum need not equal one.
    """
    p = np.asarray(probabilities)
    if p.ndim != 2 or p.shape[1] != 16 or not np.isfinite(p).all() or np.any(p < 0):
        raise ValueError("Invalid state probability matrix")
    if not np.allclose(p.sum(axis=1), 1, atol=1e-12, rtol=0):
        raise ValueError("Joint-state probabilities do not sum to one")
    return p @ STATE_BITS


def record_groups(frame: Any) -> np.ndarray:
    """
    Retain known-patient grouping and identify unknown-patient records explicitly.

    Parameters
    ----------
    frame : DataFrame
        Frozen training metadata with identity evidence and source fields.

    Returns
    -------
    np.ndarray
        Group strings; unknown records do not imply distinct biological people.
    """
    known = frame["patient_identity_known"].eq("True")
    return np.where(
        known, "known:" + frame["source"] + ":" + frame["patient_id"], "unknown:" + frame["record_id"]
    ).astype(str)


def folds(groups: np.ndarray) -> np.ndarray:
    """
    Assign three deterministic training-only folds without known-patient overlap.

    Parameters
    ----------
    groups : np.ndarray
        Known-patient or unknown-record group per row.

    Returns
    -------
    np.ndarray
        Fold index per row, covering each row once.
    """
    result = np.full(len(groups), -1, dtype=int)
    split = GroupKFold(n_splits=3, shuffle=True, random_state=20261106)
    for fold, (train, held) in enumerate(split.split(np.zeros(len(groups)), groups=groups)):
        if set(groups[train]) & set(groups[held]):
            raise ValueError("Known-patient fold leakage")
        result[held] = fold
    if (result < 0).any():
        raise ValueError("Unassigned training fold")
    return result


def resample_references(groups: np.ndarray, families: np.ndarray, seed: int) -> np.ndarray:
    """
    Resample patient/record groups within each source family, retaining entire groups.

    Parameters
    ----------
    groups, families : np.ndarray
        Aligned identity groups and source families of training records.
    seed : int
        Frozen resampling seed.

    Returns
    -------
    np.ndarray
        Original row positions, with bootstrap multiplicities.
    """
    rng = np.random.default_rng(seed)
    selected = []
    for family in np.unique(families):
        rows = np.flatnonzero(families == family)
        selected.append(rows[patient_resample(patient_groups(groups[rows]), rng)])
    return np.concatenate(selected)


def fit_model(z: np.ndarray, target: np.ndarray, c: float) -> LogisticRegression:
    """
    Fit a converged L2 logistic model with the frozen numerical recipe.

    Parameters
    ----------
    z : np.ndarray
        Unit training vectors in the fold's standardized space.
    target : np.ndarray
        Binary memberships or joint-state targets.
    c : float
        Registered regularization inverse strength.

    Returns
    -------
    LogisticRegression
        Converged fitted model; warnings at the iteration cap stop execution.
    """
    model = LogisticRegression(
        C=c, solver="lbfgs", tol=1e-8, max_iter=5000, class_weight=None, random_state=42
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ConvergenceWarning)
        model.fit(z, target)
    if any(issubclass(w.category, ConvergenceWarning) for w in caught) or np.max(model.n_iter_) >= 5000:
        raise RuntimeError("Logistic solver did not converge")
    return model


def fit_package(raw: np.ndarray, flags: np.ndarray, family: str, c: float) -> dict[str, Any]:
    """
    Fit a scaler and classifier family using only supplied training records.

    Parameters
    ----------
    raw : np.ndarray
        Float64 pooled encoder features of permitted training rows.
    flags : np.ndarray
        Four binary membership targets.
    family : str
        Joint multiclass or independent-label logistic family.
    c : float
        Frozen or training-selected regularization strength.

    Returns
    -------
    dict[str, Any]
        Scaler, models, family and measured fitting time.
    """
    started = time.perf_counter()
    target = encode_states(flags)
    scaler = StandardScaler().fit(raw)
    z = unit(scaler.transform(raw))
    if family == "joint_logistic":
        models = [fit_model(z, target, c)]
    elif family == "independent_logistic":
        models = [fit_model(z, flags[:, column], c) for column in range(4)]
    else:
        raise ValueError("Unknown classifier family")
    return {
        "family": family,
        "C": c,
        "scaler": scaler,
        "models": models,
        "fit_seconds": time.perf_counter() - started,
        "iterations": [model.n_iter_.tolist() for model in models],
    }


def predict_package(package: dict[str, Any], raw: np.ndarray) -> tuple[np.ndarray, np.ndarray | None]:
    """
    Score fixed features with a training-fitted transformation and readout.

    Parameters
    ----------
    package : dict[str, Any]
        Fitted model family and its training scaler.
    raw : np.ndarray
        Query features; no fitting occurs here.

    Returns
    -------
    tuple[np.ndarray, np.ndarray or None]
        Four marginal scores and, for the multiclass model, sixteen-state probabilities.
    """
    z = unit(package["scaler"].transform(np.asarray(raw, dtype=np.float64)))
    if package["family"] == "joint_logistic":
        model = package["models"][0]
        probability = np.zeros((len(raw), 16))
        probability[:, model.classes_.astype(int)] = model.predict_proba(z)
        return marginal_scores(probability), probability
    scores = np.column_stack([model.predict_proba(z)[:, 1] for model in package["models"]])
    return scores, None


def joint_metrics(flags: np.ndarray, probability: np.ndarray, majority: int) -> dict[str, Any]:
    """
    Describe genuine multiclass accuracy and confusion without a referral threshold.

    Parameters
    ----------
    flags : np.ndarray
        True four-group annotation memberships.
    probability : np.ndarray
        Sixteen-state model probabilities.
    majority : int
        Most common joint state in training only.

    Returns
    -------
    dict[str, Any]
        Confusion, supported-state recall, proper loss and majority-class context.
    """
    truth = encode_states(flags)
    predicted = probability.argmax(axis=1)
    matrix = confusion_matrix(truth, predicted, labels=np.arange(16))
    support, predicted_count = matrix.sum(axis=1), matrix.sum(axis=0)
    tp = np.diag(matrix)
    recall = np.divide(tp, support, out=np.zeros(16, float), where=support != 0)
    precision = np.divide(tp, predicted_count, out=np.zeros(16, float), where=predicted_count != 0)
    f1 = np.divide(
        2 * precision * recall, precision + recall, out=np.zeros(16, float), where=precision + recall != 0
    )
    top2 = np.argsort(-probability, axis=1, kind="stable")[:, :2]
    names = [
        "+".join(g for c, g in enumerate(GROUPS) if STATE_BITS[s, c]) or "normal_proxy" for s in range(16)
    ]
    return {
        "state_names": names,
        "confusion": matrix.tolist(),
        "support": support.tolist(),
        "predicted_count": predicted_count.tolist(),
        "precision": [float(precision[s]) if predicted_count[s] else None for s in range(16)],
        "recall": [float(recall[s]) if support[s] else None for s in range(16)],
        "f1": [float(f1[s]) if support[s] else None for s in range(16)],
        "supported_states": np.flatnonzero(support).tolist(),
        "exact_state_accuracy": float(np.mean(truth == predicted)),
        "macro_recall_supported_states": float(recall[support > 0].mean()),
        "top_two_accuracy": float((top2 == truth[:, None]).any(axis=1).mean()),
        "log_loss": float(log_loss(truth, probability, labels=np.arange(16))),
        "training_majority_state": int(majority),
        "training_majority_accuracy": float(np.mean(truth == majority)),
    }
