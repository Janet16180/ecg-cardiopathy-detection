"""Prototypes, cosine neighbors, partitions and audits for the Experiment 024 embedding geometry study."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.linear_model import Ridge
from sklearn.metrics import adjusted_mutual_info_score, r2_score
from sklearn.preprocessing import StandardScaler, normalize

SEED = 24024
NEIGHBORS = 25
MULTI_PROTOTYPE_K = (2, 4, 8)
DEVICE_CLASS_MINIMUM = 20
CHUNK = 2048


def unit_space(scaler: StandardScaler, x: np.ndarray) -> np.ndarray:
    """
    Standardize features with a fitted training scaler and scale each row to unit length.

    Parameters
    ----------
    scaler : StandardScaler
        Scaler fitted on training rows only.
    x : np.ndarray
        Raw features.

    Returns
    -------
    np.ndarray
        Float64 unit vectors; cosine similarity is their dot product.
    """
    return normalize(scaler.transform(np.asarray(x, dtype=np.float64)))


def prototype(x: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """
    Re-normalized mean of the selected unit vectors.

    Parameters
    ----------
    x : np.ndarray
        Unit vectors.
    mask : np.ndarray
        Boolean row selection.

    Returns
    -------
    np.ndarray
        Unit-length class mean.
    """
    return normalize(x[mask].mean(axis=0, keepdims=True))[0]


def prototype_scores(train: np.ndarray, y: np.ndarray, query: np.ndarray) -> np.ndarray:
    """
    Cosine similarity to the positive prototype minus that to the negative prototype.

    Parameters
    ----------
    train : np.ndarray
        Training unit vectors.
    y : np.ndarray
        Binary training labels.
    query : np.ndarray
        Query unit vectors.

    Returns
    -------
    np.ndarray
        One score per query.
    """
    return query @ prototype(train, y == 1) - query @ prototype(train, y == 0)


def kmeans_centroids(x: np.ndarray, k: int, seed: int = SEED) -> np.ndarray:
    """
    Re-normalized k-means centroids of unit vectors.

    Parameters
    ----------
    x : np.ndarray
        Unit vectors.
    k : int
        Number of clusters.
    seed : int
        k-means random state.

    Returns
    -------
    np.ndarray
        Array of shape ``(k, features)`` with unit-length rows.
    """
    model = KMeans(n_clusters=k, n_init=10, random_state=seed).fit(x)
    return normalize(model.cluster_centers_)


def multi_prototype_scores(train: np.ndarray, y: np.ndarray, query: np.ndarray, k: int,
                           seed: int = SEED) -> np.ndarray:
    """
    Highest cosine similarity to a positive centroid minus the highest to a negative centroid.

    Parameters
    ----------
    train : np.ndarray
        Training unit vectors.
    y : np.ndarray
        Binary training labels.
    query : np.ndarray
        Query unit vectors.
    k : int
        Centroids per class.
    seed : int
        k-means random state.

    Returns
    -------
    np.ndarray
        One score per query.
    """
    positive = kmeans_centroids(train[y == 1], k, seed)
    negative = kmeans_centroids(train[y == 0], k, seed)
    return (query @ positive.T).max(axis=1) - (query @ negative.T).max(axis=1)


def device_balanced_prototype(x: np.ndarray, mask: np.ndarray, devices: np.ndarray,
                              minimum: int = DEVICE_CLASS_MINIMUM) -> np.ndarray:
    """
    Unweighted mean of per-device class prototypes, re-normalized.

    Parameters
    ----------
    x : np.ndarray
        Unit vectors.
    mask : np.ndarray
        Boolean selection of the class.
    devices : np.ndarray
        Device of each row.
    minimum : int
        Minimum class rows for a device to contribute.

    Returns
    -------
    np.ndarray
        Unit-length prototype.

    Raises
    ------
    ValueError
        If no device has enough rows of the class.
    """
    names, counts = np.unique(devices[mask], return_counts=True)
    eligible = names[counts >= minimum]
    if len(eligible) == 0:
        raise ValueError("No device has enough rows for a balanced prototype")
    means = np.stack([prototype(x, mask & (devices == name)) for name in eligible])
    return normalize(means.mean(axis=0, keepdims=True))[0]


def device_balanced_scores(train: np.ndarray, y: np.ndarray, devices: np.ndarray,
                           query: np.ndarray) -> np.ndarray:
    """
    Prototype score with device-balanced positive and negative prototypes.

    Parameters
    ----------
    train : np.ndarray
        Training unit vectors.
    y : np.ndarray
        Binary training labels.
    devices : np.ndarray
        Device of each training row.
    query : np.ndarray
        Query unit vectors.

    Returns
    -------
    np.ndarray
        One score per query.
    """
    positive = device_balanced_prototype(train, y == 1, devices)
    negative = device_balanced_prototype(train, y == 0, devices)
    return query @ positive - query @ negative


def neighbor_indices(train: np.ndarray, query: np.ndarray, k: int = NEIGHBORS,
                     train_groups: np.ndarray | None = None,
                     query_groups: np.ndarray | None = None) -> np.ndarray:
    """
    Find the ``k`` most cosine-similar training rows for each query, most similar first.

    Training rows in the same group as the query are excluded. Passing row positions as the
    groups of both sides excludes each query itself.

    Parameters
    ----------
    train : np.ndarray
        Training unit vectors.
    query : np.ndarray
        Query unit vectors.
    k : int
        Number of neighbors.
    train_groups, query_groups : np.ndarray | None
        Optional group of each training row and each query.

    Returns
    -------
    np.ndarray
        Integer array of shape ``(queries, k)``.
    """
    chunks = []
    for start in range(0, len(query), CHUNK):
        similarity = query[start:start + CHUNK] @ train.T
        if train_groups is not None:
            blocked = query_groups[start:start + CHUNK, None] == train_groups[None, :]
            similarity[blocked] = -np.inf
        top = np.argpartition(-similarity, k - 1, axis=1)[:, :k]
        order = np.argsort(-np.take_along_axis(similarity, top, axis=1), axis=1, kind="stable")
        chunks.append(np.take_along_axis(top, order, axis=1))
    return np.concatenate(chunks)


def method_scores(train: np.ndarray, y: np.ndarray, train_devices: np.ndarray, query: np.ndarray,
                  query_devices: np.ndarray) -> dict[str, np.ndarray]:
    """
    Every distance-based score of the protocol for one binary task.

    Parameters
    ----------
    train : np.ndarray
        Training unit vectors.
    y : np.ndarray
        Binary training labels.
    train_devices : np.ndarray
        Device of each training row.
    query : np.ndarray
        Query unit vectors.
    query_devices : np.ndarray
        Device of each query.

    Returns
    -------
    dict[str, np.ndarray]
        ``prototype``, ``prototype_device_balanced``, ``knn``, ``knn_other_device`` and
        ``multi_prototype_<k>`` scores.
    """
    neighbors = neighbor_indices(train, query)
    other = neighbor_indices(train, query, train_groups=train_devices, query_groups=query_devices)
    scores = {
        "prototype": prototype_scores(train, y, query),
        "prototype_device_balanced": device_balanced_scores(train, y, train_devices, query),
        "knn": y[neighbors].mean(axis=1),
        "knn_other_device": y[other].mean(axis=1),
    }
    for k in MULTI_PROTOTYPE_K:
        scores[f"multi_prototype_{k}"] = multi_prototype_scores(train, y, query, k)
    return scores


def same_device_summary(neighbors: np.ndarray, train_devices: np.ndarray,
                        query_devices: np.ndarray) -> dict[str, object]:
    """
    Share of each query's neighbors from its own device, against that device's training share.

    Parameters
    ----------
    neighbors : np.ndarray
        Output of ``neighbor_indices``.
    train_devices : np.ndarray
        Device of each training row.
    query_devices : np.ndarray
        Device of each query.

    Returns
    -------
    dict[str, object]
        Overall mean same-device fraction and expected share, and the same per query device.
    """
    fraction = (train_devices[neighbors] == query_devices[:, None]).mean(axis=1)
    share = pd.Series(train_devices).value_counts(normalize=True)
    expected = share.reindex(query_devices).fillna(0.0).to_numpy()
    frame = pd.DataFrame({"device": query_devices, "fraction": fraction, "expected": expected})
    per_device = {name: {"queries": len(rows), "same_device_fraction": float(rows["fraction"].mean()),
                         "training_share": float(rows["expected"].iloc[0])}
                  for name, rows in frame.groupby("device")}
    return {"same_device_fraction": float(fraction.mean()), "expected_share": float(expected.mean()),
            "per_device": per_device}


def label_disagreement(neighbors: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    Fraction of each row's neighbors whose label differs from its own.

    Parameters
    ----------
    neighbors : np.ndarray
        Neighbor indices into ``y``, self excluded, one row per element of ``y``.
    y : np.ndarray
        Binary labels.

    Returns
    -------
    np.ndarray
        One fraction per row.
    """
    return (y[neighbors] != y[:, None]).mean(axis=1)


def audit_candidates(first: np.ndarray, second: np.ndarray, threshold: float = 0.8) -> np.ndarray:
    """
    Rows whose neighbor disagreement reaches the threshold in both spaces.

    Parameters
    ----------
    first, second : np.ndarray
        Disagreement fractions of the same rows in two embedding spaces.
    threshold : float
        Minimum fraction in each space.

    Returns
    -------
    np.ndarray
        Boolean mask.
    """
    return (first >= threshold) & (second >= threshold)


def medoids(x: np.ndarray, mask: np.ndarray, count: int = 5) -> tuple[np.ndarray, np.ndarray]:
    """
    Group members most cosine-similar to the group prototype.

    Parameters
    ----------
    x : np.ndarray
        Unit vectors of all rows.
    mask : np.ndarray
        Boolean selection of the group.
    count : int
        Number of members to return.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        Row positions in ``x``, most similar first, and their similarities.
    """
    members = np.flatnonzero(mask)
    similarity = x[members] @ prototype(x, mask)
    order = np.argsort(-similarity, kind="stable")[:count]
    return members[order], similarity[order]


def ridge_r2(x_train: np.ndarray, y_train: np.ndarray, x_test: np.ndarray, y_test: np.ndarray,
             alpha: float = 1.0) -> float:
    """
    Held-out R² of a training-only scaler and ridge regression.

    Parameters
    ----------
    x_train, y_train : np.ndarray
        Training inputs and targets.
    x_test, y_test : np.ndarray
        Held-out inputs and targets.
    alpha : float
        Ridge penalty.

    Returns
    -------
    float
        Coefficient of determination on the held-out rows.
    """
    scaler = StandardScaler().fit(np.asarray(x_train, dtype=np.float64))
    model = Ridge(alpha=alpha).fit(scaler.transform(np.asarray(x_train, dtype=np.float64)), y_train)
    return float(r2_score(y_test, model.predict(scaler.transform(np.asarray(x_test, dtype=np.float64)))))


def age_decade(age: np.ndarray) -> np.ndarray:
    """
    Age decade labels, with missing ages as their own category.

    Parameters
    ----------
    age : np.ndarray
        Ages in years, NaN when missing.

    Returns
    -------
    np.ndarray
        Strings such as ``"60"`` or ``"missing"``.
    """
    decades = np.floor(np.nan_to_num(age, nan=0.0) / 10).astype(int) * 10
    return np.where(np.isnan(age), "missing", decades.astype(str))


def heart_rate_band(rate: np.ndarray) -> np.ndarray:
    """
    Heart-rate band: below 60, 60 to 100, or above 100 bpm.

    Parameters
    ----------
    rate : np.ndarray
        Heart rate in beats per minute.

    Returns
    -------
    np.ndarray
        ``"<60"``, ``"60-100"`` or ``">100"`` per row.
    """
    return np.select([rate < 60, rate > 100], ["<60", ">100"], default="60-100")


def ami_summary(partition: np.ndarray, factors: dict[str, np.ndarray]) -> dict[str, float]:
    """
    Compute the adjusted mutual information of one partition with each factor.

    Parameters
    ----------
    partition : np.ndarray
        Cluster label of each row.
    factors : dict[str, np.ndarray]
        Categorical factors of the same rows.

    Returns
    -------
    dict[str, float]
        AMI per factor.
    """
    return {name: float(adjusted_mutual_info_score(values, partition)) for name, values in factors.items()}


def patient_subset(patients: np.ndarray, size: int, seed: int = SEED) -> np.ndarray:
    """
    One row per patient in a seeded order, first ``size`` patients.

    Parameters
    ----------
    patients : np.ndarray
        Patient ID of each row.
    size : int
        Number of patients.
    seed : int
        Seed of ``numpy.random.default_rng``.

    Returns
    -------
    np.ndarray
        Sorted row positions.

    Raises
    ------
    ValueError
        If there are fewer than ``size`` patients.
    """
    order = np.random.default_rng(seed).permutation(len(patients))
    _, first = np.unique(patients[order], return_index=True)
    chosen = order[np.sort(first)][:size]
    if len(chosen) != size:
        raise ValueError(f"Fewer than {size} patients")
    return np.sort(chosen)
