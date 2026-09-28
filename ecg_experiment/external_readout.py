"""Experiment 020 heads, frozen-encoder heads and scores shared by the external readouts 022 and 023."""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

from .cpc import CPCEncoder
from .files import read_json, sha256_file
from .full_development import demographics, fit_logistic, predict

ROOT = Path(__file__).resolve().parents[1]
ENCODER = ROOT / "outputs/experiment004_cpc_40k/cpc_ssl/encoder.pt"
PRIOR = ROOT / "outputs/experiment020_full_development_v2"
HEAD_TOLERANCE = 1e-9
PTB_HEADS = ("cpc_project_full", "cpc_standard", "age_sex_standard", "cpc_age_sex_standard")
ENCODER_HEADS = {"jepa_standard": ("jepa", 0.01), "xecg_standard": ("xecg", 0.01),
                 "jepa_standard_c01": ("jepa", 0.1)}
ALL_HEADS = PTB_HEADS + tuple(ENCODER_HEADS)

Head = tuple[StandardScaler, LogisticRegression]


def fit_logistic_c(x: np.ndarray, y: np.ndarray, c: float) -> Head:
    """
    Fit ``full_development.fit_logistic``'s scaler and L2 logistic head with another inverse strength.

    Parameters
    ----------
    x : np.ndarray
        Training inputs.
    y : np.ndarray
        Binary targets.
    c : float
        Inverse L2 regularization strength; 0.01 reproduces ``fit_logistic``.

    Returns
    -------
    Head
        Fitted scaler and classifier.

    Raises
    ------
    RuntimeError
        If the solver does not converge.
    """
    x = np.asarray(x, dtype=np.float64)
    scaler = StandardScaler().fit(x)
    model = LogisticRegression(C=c, penalty="l2", fit_intercept=True, solver="lbfgs", max_iter=5000,
                               tol=1e-8, class_weight=None, random_state=42)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ConvergenceWarning)
        model.fit(scaler.transform(x), y)
    if any(issubclass(item.category, ConvergenceWarning) for item in caught) or model.n_iter_[0] >= 5000:
        raise RuntimeError("Logistic fit did not converge")
    return scaler, model


def cache_positions(cache_ids: np.ndarray, ecg_ids: np.ndarray) -> np.ndarray:
    """
    Row of each requested ECG in a feature cache, found by ECG ID only.

    Parameters
    ----------
    cache_ids : np.ndarray
        Integer ECG ID of every cache row.
    ecg_ids : np.ndarray
        Integer ECG IDs to look up.

    Returns
    -------
    np.ndarray
        Cache row of each requested ECG, or -1 where the cache lacks it.

    Raises
    ------
    ValueError
        If the cache repeats an ECG ID.
    """
    if len(np.unique(cache_ids)) != len(cache_ids):
        raise ValueError("Feature cache repeats an ECG ID")
    lookup = pd.Series(np.arange(len(cache_ids)), index=np.asarray(cache_ids, dtype=np.int64))
    return lookup.reindex(np.asarray(ecg_ids, dtype=np.int64)).fillna(-1).to_numpy(dtype=np.int64)


def combine_features(ecg_ids: np.ndarray, cache: np.ndarray, cache_ids: np.ndarray,
                     extracted_ids: np.ndarray, extracted: np.ndarray) -> np.ndarray:
    """
    Features in the requested order, from the cache where present and from new extraction otherwise.

    Only the requested cache rows are read.

    Parameters
    ----------
    ecg_ids : np.ndarray
        Integer ECG IDs in output order.
    cache : np.ndarray
        Cached features, possibly memory-mapped.
    cache_ids : np.ndarray
        Integer ECG ID of every cache row.
    extracted_ids : np.ndarray
        Integer ECG IDs of the newly extracted rows.
    extracted : np.ndarray
        Newly extracted features.

    Returns
    -------
    np.ndarray
        Float32 features of shape ``(len(ecg_ids), width)``.

    Raises
    ------
    ValueError
        If the extracted rows are not exactly the requested rows the cache lacks, in order.
    """
    positions = cache_positions(cache_ids, ecg_ids)
    missing = positions < 0
    if not np.array_equal(np.asarray(extracted_ids, dtype=np.int64), np.asarray(ecg_ids)[missing]):
        raise ValueError("Extracted rows differ from the rows missing from the cache")
    result = np.empty((len(ecg_ids), cache.shape[1]), dtype=np.float32)
    result[~missing] = cache[positions[~missing]]
    result[missing] = extracted
    return result


def prior_identity() -> dict[str, str]:
    """
    Hash the Experiment 020 artifacts the heads are refitted from.

    Returns
    -------
    dict[str, str]
        SHA-256 of the saved features, predictions and result.

    Raises
    ------
    ValueError
        If the saved features or predictions differ from the hashes in the 020 result.
    """
    result = read_json(PRIOR / "result.json")
    hashes = {name: sha256_file(PRIOR / f"{name}.npz") for name in ("features", "development_predictions")}
    if hashes["features"] != result["features_sha256"] or \
            hashes["development_predictions"] != result["predictions_sha256"]:
        raise ValueError("Experiment 020 artifacts differ from its result receipt")
    return {**hashes, "result": sha256_file(PRIOR / "result.json")}


def prior_cpc_features() -> dict[str, np.ndarray]:
    """
    Experiment 020's saved CPC features.

    Returns
    -------
    dict[str, np.ndarray]
        ``train`` and ``development`` features in the ``full_development.cohorts`` row order.
    """
    with np.load(PRIOR / "features.npz") as saved:
        return {"train": saved["train"], "development": saved["development"]}


def ptb_heads(train: pd.DataFrame, development: pd.DataFrame,
              cpc: dict[str, np.ndarray]) -> tuple[dict[str, Head], float, dict[str, float]]:
    """
    Refit the Experiment 020 heads and require them to reproduce its development probabilities.

    Parameters
    ----------
    train, development : pd.DataFrame
        ``full_development.cohorts`` training and development rows.
    cpc : dict[str, np.ndarray]
        Output of ``prior_cpc_features``.

    Returns
    -------
    tuple[dict[str, Head], float, dict[str, float]]
        Heads named as in ``PTB_HEADS``, the PTB-XL training median age used by the demographic inputs,
        and each head's largest absolute difference from Experiment 020.

    Raises
    ------
    ValueError
        If a refitted head differs from Experiment 020 by more than ``HEAD_TOLERANCE``.
    """
    median_age = float(train["age"].median())
    age_sex = {"train": demographics(train, median_age), "development": demographics(development, median_age)}
    inputs = {
        "cpc": cpc, "age_sex": age_sex,
        "cpc_age_sex": {name: np.hstack([cpc[name], age_sex[name]]) for name in cpc},
    }
    project = train["target"].notna().to_numpy()
    standard = train["standard"].notna().to_numpy()
    plans = {
        "cpc_project_full": ("cpc", project, "target"),
        "cpc_standard": ("cpc", standard, "standard"),
        "age_sex_standard": ("age_sex", standard, "standard"),
        "cpc_age_sex_standard": ("cpc_age_sex", standard, "standard"),
    }
    heads, differences = {}, {}
    with np.load(PRIOR / "development_predictions.npz", allow_pickle=True) as saved:
        if not np.array_equal(saved["record_ids"], development.index.to_numpy()):
            raise ValueError("Development rows differ from Experiment 020")
        for name, (kind, mask, label) in plans.items():
            heads[name] = fit_logistic(inputs[kind]["train"][mask],
                                       train.loc[mask, label].to_numpy(dtype=np.int64))
            differences[name] = float(np.abs(predict(heads[name], inputs[kind]["development"])
                                             - saved[name]).max())
            if differences[name] > HEAD_TOLERANCE:
                raise ValueError(f"Refitted {name} differs from Experiment 020 by {differences[name]}")
    return heads, median_age, differences


def encoder_heads(train: pd.DataFrame, features: dict[str, np.ndarray]) -> dict[str, Head]:
    """
    Fit the frozen ECG-JEPA and xECG heads on the standard-label training rows.

    Parameters
    ----------
    train : pd.DataFrame
        ``full_development.cohorts`` training rows.
    features : dict[str, np.ndarray]
        Training features keyed by encoder (``jepa``, ``xecg``), in the training row order.

    Returns
    -------
    dict[str, Head]
        Heads named as in ``ENCODER_HEADS``.
    """
    standard = train["standard"].notna().to_numpy()
    y = train.loc[standard, "standard"].to_numpy(dtype=np.int64)
    return {name: fit_logistic_c(features[encoder][standard], y, c)
            for name, (encoder, c) in ENCODER_HEADS.items()}


def head_inputs(frame: pd.DataFrame, median_age: float,
                features: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """
    Input matrix of every head for one set of ECGs.

    Parameters
    ----------
    frame : pd.DataFrame
        Rows with ``age`` and ``male``.
    median_age : float
        PTB-XL training median age.
    features : dict[str, np.ndarray]
        Features keyed by encoder (``cpc``, ``jepa``, ``xecg``), in the row order of ``frame``.

    Returns
    -------
    dict[str, np.ndarray]
        Inputs keyed by head name as in ``ALL_HEADS``.
    """
    age_sex = demographics(frame, median_age)
    inputs = {"cpc_project_full": features["cpc"], "cpc_standard": features["cpc"],
              "age_sex_standard": age_sex, "cpc_age_sex_standard": np.hstack([features["cpc"], age_sex])}
    return {**inputs, **{name: features[encoder] for name, (encoder, _) in ENCODER_HEADS.items()}}


def load_encoder() -> CPCEncoder:
    """
    Load the unchanged starting CPC encoder on the GPU with no trainable parameters.

    Returns
    -------
    CPCEncoder
        Encoder in evaluation mode.
    """
    saved = torch.load(ENCODER, map_location="cpu", weights_only=True)
    model = CPCEncoder().cuda().eval()
    model.load_state_dict(saved["encoder"], strict=True)
    model.requires_grad_(False)
    return model


def score(y: np.ndarray, probabilities: np.ndarray) -> dict[str, float | int]:
    """
    AUROC, average precision and class counts.

    Parameters
    ----------
    y : np.ndarray
        Binary targets.
    probabilities : np.ndarray
        Scores of the positive class.

    Returns
    -------
    dict[str, float | int]
        ``auroc``, ``average_precision``, ``records`` and ``positives``.
    """
    return {"auroc": float(roc_auc_score(y, probabilities)),
            "average_precision": float(average_precision_score(y, probabilities)),
            "records": len(y), "positives": int(y.sum())}


def evenly_spaced(frame: pd.DataFrame, count: int) -> pd.DataFrame:
    """
    Deterministic, evenly spaced rows including the first and last.

    Parameters
    ----------
    frame : pd.DataFrame
        Rows to sample from.
    count : int
        Number of rows.

    Returns
    -------
    pd.DataFrame
        The selected rows in their original order.
    """
    return frame.iloc[np.unique(np.linspace(0, len(frame) - 1, count, dtype=np.int64))]
