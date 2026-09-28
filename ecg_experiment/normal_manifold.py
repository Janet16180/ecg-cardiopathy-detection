"""One-class scores fitted on normal ECGs and paired bootstrap contrasts for Experiment 026."""

from __future__ import annotations

import numpy as np
from sklearn.covariance import LedoitWolf
from sklearn.decomposition import PCA
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.preprocessing import StandardScaler, normalize

COMPONENTS = 64
NEIGHBORS = 25
CHUNK = 2048
METRICS = ("auroc", "average_precision")


def fit_mahalanobis(x: np.ndarray, components: int = COMPONENTS) -> tuple[StandardScaler, PCA, LedoitWolf]:
    """
    Fit a scaler, PCA and Ledoit-Wolf covariance on normal ECG features.

    Parameters
    ----------
    x : np.ndarray
        Features of the fit set (normal ECGs only).
    components : int
        Number of principal components kept.

    Returns
    -------
    tuple[StandardScaler, PCA, LedoitWolf]
        The fitted scaler, PCA and covariance estimator.
    """
    x = np.asarray(x, dtype=np.float64)
    scaler = StandardScaler().fit(x)
    pca = PCA(n_components=components, svd_solver="full").fit(scaler.transform(x))
    covariance = LedoitWolf().fit(pca.transform(scaler.transform(x)))
    return scaler, pca, covariance


def mahalanobis_scores(model: tuple[StandardScaler, PCA, LedoitWolf], x: np.ndarray) -> np.ndarray:
    """
    Squared Mahalanobis distance of each row from the fitted normal distribution.

    Parameters
    ----------
    model : tuple[StandardScaler, PCA, LedoitWolf]
        Output of ``fit_mahalanobis``.
    x : np.ndarray
        Features to score.

    Returns
    -------
    np.ndarray
        One distance per row; larger is farther from normal.
    """
    scaler, pca, covariance = model
    return covariance.mahalanobis(pca.transform(scaler.transform(np.asarray(x, dtype=np.float64))))


def fit_knn(x: np.ndarray) -> tuple[StandardScaler, np.ndarray]:
    """
    Fit a scaler on normal ECG features and keep their unit vectors as the reference set.

    Parameters
    ----------
    x : np.ndarray
        Features of the fit set (normal ECGs only).

    Returns
    -------
    tuple[StandardScaler, np.ndarray]
        The fitted scaler and the standardized, unit-length fit-set rows.
    """
    x = np.asarray(x, dtype=np.float64)
    scaler = StandardScaler().fit(x)
    return scaler, normalize(scaler.transform(x))


def knn_scores(model: tuple[StandardScaler, np.ndarray], x: np.ndarray, k: int = NEIGHBORS) -> np.ndarray:
    """
    Mean cosine distance of each row to its ``k`` nearest fit-set ECGs.

    Parameters
    ----------
    model : tuple[StandardScaler, np.ndarray]
        Output of ``fit_knn``.
    x : np.ndarray
        Features to score. They must not belong to the fit set, since no self-match is removed.
    k : int
        Number of neighbors.

    Returns
    -------
    np.ndarray
        One distance per row; larger is farther from normal.
    """
    scaler, reference = model
    query = normalize(scaler.transform(np.asarray(x, dtype=np.float64)))
    chunks = []
    for start in range(0, len(query), CHUNK):
        distance = 1 - query[start:start + CHUNK] @ reference.T
        chunks.append(np.partition(distance, k - 1, axis=1)[:, :k].mean(axis=1))
    return np.concatenate(chunks)


def metrics(y: np.ndarray, scores: np.ndarray) -> dict[str, float]:
    """
    AUROC and average precision of a score, or their mean over several scores.

    Parameters
    ----------
    y : np.ndarray
        Binary targets.
    scores : np.ndarray
        One score per row, or a 2-D array with one score vector per row (for example one per label
        draw), in which case each metric is the mean over the vectors.

    Returns
    -------
    dict[str, float]
        ``auroc`` and ``average_precision``.
    """
    vectors = np.atleast_2d(scores)
    return {"auroc": float(np.mean([roc_auc_score(y, values) for values in vectors])),
            "average_precision": float(np.mean([average_precision_score(y, values) for values in vectors]))}


def patient_resamples(patients: np.ndarray, y: np.ndarray, draws: int,
                      seed: int) -> tuple[list[np.ndarray], int]:
    """
    Whole-patient bootstrap resamples, skipping those with a single class.

    Patients are drawn with replacement and all their ECGs are kept, as in
    ``full_development.patient_bootstrap``.

    Parameters
    ----------
    patients : np.ndarray
        Patient ID of each ECG.
    y : np.ndarray
        Binary targets.
    draws : int
        Number of bootstrap draws.
    seed : int
        Seed of ``numpy.random.default_rng``.

    Returns
    -------
    tuple[list[np.ndarray], int]
        Row positions of each valid resample and the number of single-class draws skipped.
    """
    unique, codes = np.unique(patients, return_inverse=True)
    members = [np.flatnonzero(codes == index) for index in range(len(unique))]
    rng = np.random.default_rng(seed)
    resamples, invalid = [], 0
    for _ in range(draws):
        rows = np.concatenate([members[i] for i in rng.integers(0, len(unique), len(unique))])
        if len(np.unique(y[rows])) < 2:
            invalid += 1
            continue
        resamples.append(rows)
    return resamples, invalid


def bootstrap_metrics(y: np.ndarray, scores: dict[str, np.ndarray],
                      resamples: list[np.ndarray]) -> dict[str, dict[str, np.ndarray]]:
    """
    Metrics of every score on the same resamples.

    Parameters
    ----------
    y : np.ndarray
        Binary targets.
    scores : dict[str, np.ndarray]
        Scores by name, each 1-D or 2-D as accepted by ``metrics``.
    resamples : list[np.ndarray]
        Output of ``patient_resamples``.

    Returns
    -------
    dict[str, dict[str, np.ndarray]]
        Per score and metric, one value per resample.
    """
    values = {name: {metric: np.empty(len(resamples)) for metric in METRICS} for name in scores}
    for draw, rows in enumerate(resamples):
        for name, score in scores.items():
            result = metrics(y[rows], np.atleast_2d(score)[:, rows])
            for metric in METRICS:
                values[name][metric][draw] = result[metric]
    return values


def contrast(observed: dict[str, dict[str, float]], resampled: dict[str, dict[str, np.ndarray]],
             first: str, second: str) -> dict[str, dict[str, float]]:
    """
    Paired difference, first minus second, with its 95% percentile interval.

    Parameters
    ----------
    observed : dict[str, dict[str, float]]
        ``metrics`` of every score on all rows.
    resampled : dict[str, dict[str, np.ndarray]]
        Output of ``bootstrap_metrics``.
    first, second : str
        Score names.

    Returns
    -------
    dict[str, dict[str, float]]
        Per metric, the observed ``difference``, ``ci_low`` and ``ci_high``.
    """
    result = {}
    for metric in METRICS:
        low, high = np.percentile(resampled[first][metric] - resampled[second][metric], [2.5, 97.5])
        result[metric] = {"difference": observed[first][metric] - observed[second][metric],
                          "ci_low": float(low), "ci_high": float(high)}
    return result
