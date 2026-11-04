"""Whole-record and weakly supervised region prototypes for KEY-01."""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

from .embedding_geometry import prototype
from .intervals import patient_groups, patient_resample

GROUPS = ("MI", "STTC", "CD", "HYP")
SEED = 20261103
BATCH = 128


def unit(x: np.ndarray) -> np.ndarray:
    """
    Normalize vectors along the last axis, rejecting undefined directions.

    Parameters
    ----------
    x : np.ndarray
        Finite vectors with a nonzero last-axis norm.

    Returns
    -------
    np.ndarray
        Unit vectors with unchanged shape.
    """
    norms = np.linalg.norm(x, axis=-1, keepdims=True)
    if not np.isfinite(x).all() or np.any(norms == 0):
        raise ValueError("Nonfinite or zero-norm feature vector")
    return x / norms


def spans(fixed: bool = False) -> np.ndarray:
    """
    Enumerate fixed or variable half-open token spans in deterministic order.

    Parameters
    ----------
    fixed : bool
        Use ten disjoint four-token windows instead of all 2/4/8-token windows.

    Returns
    -------
    np.ndarray
        Integer array shaped [windows, 2], with start and exclusive stop.
    """
    if fixed:
        return np.array([(start, start + 4) for start in range(0, 40, 4)])
    return np.array([(start, start + size) for size in (2, 4, 8)
                     for start in range(41 - size)])


def region_vectors(tokens: np.ndarray, scaler: StandardScaler, windows: np.ndarray) -> np.ndarray:
    """
    Average raw contextual tokens within each span, standardize and normalize.

    Parameters
    ----------
    tokens : np.ndarray
        Raw [records, 40, features] cached tokens.
    scaler : StandardScaler
        Fitted only on whole-record training vectors.
    windows : np.ndarray
        Half-open token-index spans.

    Returns
    -------
    np.ndarray
        Unit vectors shaped [records, windows, features].
    """
    tokens = np.asarray(tokens, dtype=np.float64)
    if tokens.ndim != 3 or tokens.shape[1] != 40 or not np.isfinite(tokens).all():
        raise ValueError("Expected finite forty-token records")
    prefix = np.concatenate([np.zeros_like(tokens[:, :1]), np.cumsum(tokens, axis=1)], axis=1)
    means = (prefix[:, windows[:, 1]] - prefix[:, windows[:, 0]])
    means /= (windows[:, 1] - windows[:, 0])[None, :, None]
    return unit((means - scaler.mean_) / scaler.scale_)


def whole_bank(vectors: np.ndarray, flags: np.ndarray) -> tuple[StandardScaler, dict[str, np.ndarray]]:
    """
    Fit the training scaler and initial positive/normal prototype pairs.

    Parameters
    ----------
    vectors : np.ndarray
        Whole-record training features.
    flags : np.ndarray
        Binary [records, four groups] annotation memberships.

    Returns
    -------
    tuple[StandardScaler, dict[str, np.ndarray]]
        Training scaler and positive/normal keys shaped [groups, features].
    """
    flags = np.asarray(flags)
    if flags.shape != (len(vectors), 4) or not np.isin(flags, (0, 1)).all():
        raise ValueError("Expected four binary annotation memberships")
    normals = ~flags.any(axis=1)
    if not normals.any() or not flags.any(axis=0).all():
        raise ValueError("Each group and the normal pool need support examples")
    scaler = StandardScaler().fit(np.asarray(vectors, dtype=np.float64))
    z = unit(scaler.transform(np.asarray(vectors, dtype=np.float64)))
    positive = np.stack([prototype(z, flags[:, column] == 1) for column in range(4)])
    normal = np.repeat(prototype(z, normals)[None, :], 4, axis=0)
    return scaler, {"positive": positive, "normal": normal}


def selected_vectors(z: np.ndarray, flags: np.ndarray,
                     initial: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    """
    Select maximal-margin positive and minimal-margin normal regions once.

    Parameters
    ----------
    z : np.ndarray
        Unit candidate vectors [records, windows, features].
    flags : np.ndarray
        Binary group memberships for these records.
    initial : dict[str, np.ndarray]
        Whole-record positive and normal prototype pairs.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        Selected [records, groups, features] vectors and [records, groups] window indices.
        Other-group negative selections are unused when building the bank.
    """
    margins = z @ (initial["positive"] - initial["normal"]).T
    choices = np.where(flags == 1, margins.argmax(axis=1), margins.argmin(axis=1))
    return z[np.arange(len(z))[:, None], choices], choices


def refine(tokens: np.ndarray, positions: np.ndarray, flags: np.ndarray,
           scaler: StandardScaler, initial: dict[str, np.ndarray],
           windows: np.ndarray) -> tuple[dict[str, np.ndarray], np.ndarray]:
    """
    Construct a region bank from a single reference-selection pass.

    Parameters
    ----------
    tokens : np.ndarray
        Memory-mapped raw token cache.
    positions : np.ndarray
        Cache positions of the permitted training cohort, in flag order.
    flags : np.ndarray
        Binary training group memberships.
    scaler : StandardScaler
        Training-only whole-record scaler.
    initial : dict[str, np.ndarray]
        Initial whole-record bank.
    windows : np.ndarray
        Candidate region spans.

    Returns
    -------
    tuple[dict[str, np.ndarray], np.ndarray]
        Refined bank and training selections [records, groups].
    """
    positive = np.zeros_like(initial["positive"])
    normal = np.zeros_like(positive)
    choices = np.empty_like(flags, dtype=np.int16)
    for start in range(0, len(positions), BATCH):
        stop = min(start + BATCH, len(positions))
        z = region_vectors(tokens[positions[start:stop]], scaler, windows)
        part = flags[start:stop]
        selected, choices[start:stop] = selected_vectors(z, part, initial)
        for column in range(4):
            positive[column] += selected[part[:, column] == 1, column].sum(axis=0)
            normal[column] += selected[~part.any(axis=1), column].sum(axis=0)
    return {"positive": unit(positive), "normal": unit(normal)}, choices


def region_scores(z: np.ndarray, bank: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    """
    Maximize prototype margins across regions independently for each group.

    Parameters
    ----------
    z : np.ndarray
        Unit [records, candidate windows, features] vectors.
    bank : dict[str, np.ndarray]
        Positive and normal keys for each group.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        [records, groups] scores and maximizing window indices.
    """
    margins = z @ (bank["positive"] - bank["normal"]).T
    return margins.max(axis=1), margins.argmax(axis=1)


def ranking_metrics(flags: np.ndarray, scores: np.ndarray, normals_only: bool = False) -> dict[str, Any]:
    """
    Report group ranking metrics on all rows or condition-versus-normal subsets.

    Parameters
    ----------
    flags : np.ndarray
        Binary [records, groups] memberships.
    scores : np.ndarray
        Aligned [records, groups] similarity margins.
    normals_only : bool
        Exclude other abnormalities from each group's negatives.

    Returns
    -------
    dict[str, Any]
        Per-group denominators/AUROC/AP and their unweighted means.
    """
    groups = {}
    for column, name in enumerate(GROUPS):
        keep = (flags[:, column] == 1) | ~flags.any(axis=1) if normals_only else np.ones(len(flags), bool)
        y, s = flags[keep, column], scores[keep, column]
        groups[name] = {"records": len(y), "positives": int(y.sum()),
                        "negatives": int(len(y) - y.sum()),
                        "auroc": float(roc_auc_score(y, s)),
                        "average_precision": float(average_precision_score(y, s))}
    return {"groups": groups, **{f"macro_{metric}": float(np.mean([x[metric] for x in groups.values()]))
                                for metric in ("auroc", "average_precision")}}


def macro_interval(patients: np.ndarray, flags: np.ndarray, first: np.ndarray,
                   second: np.ndarray, draws: int = 2000) -> dict[str, Any]:
    """
    Bootstrap a paired macro-AUROC difference with shared whole-patient draws.

    Parameters
    ----------
    patients : np.ndarray
        Patient identity per evaluated ECG.
    flags : np.ndarray
        Four binary group memberships per record.
    first, second : np.ndarray
        Fixed candidate and comparator scores.
    draws : int
        Number of attempted whole-patient bootstrap draws.

    Returns
    -------
    dict[str, Any]
        Difference, percentile interval and valid/skipped draw accounting.
    """
    rng = np.random.default_rng(SEED)
    groups = patient_groups(patients)
    deltas = []
    for _ in range(draws):
        rows = patient_resample(groups, rng)
        y = flags[rows]
        if np.any(y.sum(axis=0) == 0) or np.any(y.sum(axis=0) == len(y)):
            continue
        deltas.append(np.mean([roc_auc_score(y[:, c], first[rows, c])
                               - roc_auc_score(y[:, c], second[rows, c]) for c in range(4)]))
    if not deltas:
        raise ValueError("No valid paired patient bootstrap draws")
    low, high = np.percentile(deltas, [2.5, 97.5])
    difference = np.mean([roc_auc_score(flags[:, c], first[:, c])
                          - roc_auc_score(flags[:, c], second[:, c]) for c in range(4)])
    return {"difference": float(difference), "ci_low": float(low), "ci_high": float(high),
            "draws": draws, "skipped_draws": draws - len(deltas), "seed": SEED}
