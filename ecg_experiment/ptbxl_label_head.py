"""Experiment 048: use PTB-XL labels on top of the frozen SimDINOv2 encoder of Experiment 047.

Experiment 047 used labels only inside the memory bank. This module adds the two uses of labels that the
bank did not have: a linear head (one logistic regression per diagnosis) trained on the frozen embeddings,
and, for telling normal from abnormal ECGs, a bank that holds only normal references. Everything works on the
embeddings saved by Experiment 047, so nothing is retrained.
"""

from __future__ import annotations

import ast
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from .ptbxl_simdino_memory import DIAGNOSES, class_aurocs, normalize


def load_embeddings(path: Path, ecg_ids: np.ndarray, encoders: tuple[str, ...]) -> dict[str, np.ndarray]:
    """
    Read the embeddings saved by Experiment 047 and check they follow the metadata order.

    Parameters
    ----------
    path : Path
        ``embeddings.npz`` with an ``ecg_id`` array and one array per encoder.
    ecg_ids : np.ndarray
        ``ecg_id`` of every metadata row, in order.
    encoders : tuple[str, ...]
        Names of the arrays to read.

    Returns
    -------
    dict[str, np.ndarray]
        One float32 array ``(records, width)`` per encoder.

    Raises
    ------
    ValueError
        If the saved ``ecg_id`` order differs from ``ecg_ids``.
    """
    saved = np.load(path)
    if not np.array_equal(saved["ecg_id"], ecg_ids):
        raise ValueError(f"{path} does not follow the metadata order")
    return {name: saved[name].astype(np.float32) for name in encoders}


def label_matrix(metadata: pd.DataFrame, codes: tuple[str, ...]) -> np.ndarray:
    """
    Presence of each PTB-XL statement, whatever its likelihood.

    Parameters
    ----------
    metadata : pd.DataFrame
        PTB-XL metadata with the ``scp_codes`` column.
    codes : tuple[str, ...]
        Statements to read.

    Returns
    -------
    np.ndarray
        Boolean ``(records, len(codes))``.
    """
    statements = metadata.scp_codes.apply(ast.literal_eval)
    return np.column_stack([statements.apply(lambda found: code in found).to_numpy() for code in codes])


def only_statements(metadata: pd.DataFrame, allowed: tuple[str, ...], required: str) -> np.ndarray:
    """
    ECGs whose statements are all in ``allowed`` and include ``required``.

    Parameters
    ----------
    metadata : pd.DataFrame
        PTB-XL metadata with the ``scp_codes`` column.
    allowed : tuple[str, ...]
        Statements permitted, for example ``("NORM", "SR")``.
    required : str
        Statement that must be present.

    Returns
    -------
    np.ndarray
        Boolean, one value per record.
    """
    statements = metadata.scp_codes.apply(ast.literal_eval)
    return statements.apply(lambda found: set(found) <= set(allowed) and required in found).to_numpy()


def diagnosis_names(codes: tuple[str, ...]) -> list[str]:
    """
    Spanish name of each code from Experiment 047, or the code itself when it has none.

    Parameters
    ----------
    codes : tuple[str, ...]
        PTB-XL statements.

    Returns
    -------
    list[str]
        One name per code.
    """
    return [DIAGNOSES.get(code, (code,))[0] for code in codes]


def reference_rows(train_rows: np.ndarray, size: int | None, seed: int) -> np.ndarray:
    """
    Labelled references drawn from ``train``, the way Experiment 047 drew its bank.

    Parameters
    ----------
    train_rows : np.ndarray
        Positions of the ``train`` ECGs.
    size : int or None
        Number of references; ``None`` keeps every ``train`` ECG.
    seed : int
        Seed of ``numpy.random.default_rng``.

    Returns
    -------
    np.ndarray
        Sorted positions.
    """
    if size is None:
        return train_rows
    return np.sort(np.random.default_rng(seed).choice(train_rows, size, replace=False))


def fit_head(features: np.ndarray, labels: np.ndarray, regularization: float
             ) -> tuple[StandardScaler, list[LogisticRegression]]:
    """
    Train one L2-regularized logistic regression per diagnosis on standardized embeddings.

    Parameters
    ----------
    features : np.ndarray
        Frozen embeddings of the references, ``(n, width)``.
    labels : np.ndarray
        Boolean ``(n, classes)``.
    regularization : float
        ``C`` of scikit-learn: smaller means simpler weights.

    Returns
    -------
    tuple[StandardScaler, list[LogisticRegression]]
        The scaler fitted on ``features`` and one model per class.
    """
    scaler = StandardScaler().fit(features)
    scaled = scaler.transform(features)
    models = [LogisticRegression(C=regularization, max_iter=5000).fit(scaled, labels[:, column])
              for column in range(labels.shape[1])]
    return scaler, models


def head_scores(head: tuple[StandardScaler, list[LogisticRegression]], features: np.ndarray) -> np.ndarray:
    """
    Probability of every diagnosis from a fitted head.

    Parameters
    ----------
    head : tuple[StandardScaler, list[LogisticRegression]]
        Output of ``fit_head``.
    features : np.ndarray
        Embeddings ``(n, width)``.

    Returns
    -------
    np.ndarray
        ``(n, classes)`` between 0 and 1.
    """
    scaler, models = head
    scaled = scaler.transform(features)
    return np.column_stack([model.predict_proba(scaled)[:, 1] for model in models])


def choose_regularization(train_features: np.ndarray, train_labels: np.ndarray, val_features: np.ndarray,
                          val_labels: np.ndarray, candidates: tuple[float, ...]) -> pd.Series:
    """
    Validation macro AUROC of the head for every candidate ``C``.

    Parameters
    ----------
    train_features, train_labels : np.ndarray
        References used to fit.
    val_features, val_labels : np.ndarray
        ECGs used to compare candidates.
    candidates : tuple[float, ...]
        Values of ``C``.

    Returns
    -------
    pd.Series
        Macro AUROC indexed by ``C``.
    """
    return pd.Series({value: class_aurocs(val_labels, head_scores(
        fit_head(train_features, train_labels, value), val_features)).mean() for value in candidates},
        name="auroc_macro_val")


def normal_bank_scores(query: np.ndarray, normal_bank: np.ndarray, centre: np.ndarray, k: int) -> np.ndarray:
    """
    How far each ECG is from normal ECGs: 1 minus its mean cosine to the ``k`` most similar normal references.

    Embeddings are centred on ``centre`` first; without it every cosine is close to 1 and differences shrink.

    Parameters
    ----------
    query : np.ndarray
        Embeddings to score, ``(n, width)``.
    normal_bank : np.ndarray
        Embeddings of normal references, ``(m, width)``.
    centre : np.ndarray
        Mean embedding of the labelled references, ``(width,)``.
    k : int
        Normal references compared.

    Returns
    -------
    np.ndarray
        One score per ECG; higher means less like normal ECGs.
    """
    similarity = normalize(query - centre) @ normalize(normal_bank - centre).T
    return 1 - np.sort(similarity, axis=1)[:, -k:].mean(axis=1)
