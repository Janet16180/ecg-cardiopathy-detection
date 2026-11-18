"""
Condition keys of KEY-05.

A key is one or a few vectors per condition, built from train records without iterative fitting.

``docs/key05-condition-keys-protocol-2026-11-11.md`` is the contract. Features are pooled xECG embeddings
standardized on train. For one condition, ``y`` holds the states 1.0, 0.0 and NaN of the records; undefined
records never enter a key. Every construction accepts record weights (bootstrap counts, or 0/1 for a drawn
subset); unit weights give the plain key.

- K1: weighted mean of the positives, scored by cosine.
- K2: mean of the positives minus mean of the negatives, scored by dot product.
- K3: mean of the positives minus mean of the records with no positive label, dot product.
- K4: the K2 difference whitened by the pooled within-class covariance with Ledoit-Wolf shrinkage
  (``within_class_covariance``), the linear discriminant direction, dot product.
- K5: K2 inside each source with at least ``SOURCE_MINIMUM`` positives and negatives, averaged with equal
  source weights, dot product.
- K6: K4 on source-centred features (``centre_by_source``).
- K7: k-means centres of the positives, each minus the negative mean and whitened as K4; the score is the
  maximum over the centres.
- N1: share of positives among the ``NEIGHBOURS`` nearest train records (cosine) where the condition is
  defined; a bound for reference-set methods, not a key.
"""

from __future__ import annotations

from typing import Any

import numpy as np

SOURCE_MINIMUM = 20
NEIGHBOURS = 20
CLUSTER_COUNTS = (2, 4, 8)
KMEANS_SEED = 20261111
KMEANS_INIT = 10
SINGLE_KEYS = ("K1", "K2", "K3", "K4", "K5", "K6")


def weighted_mean(x: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """
    Weighted mean of rows.

    Parameters
    ----------
    x : np.ndarray
        ``[records, features]``.
    weights : np.ndarray
        Nonnegative weight per record, with a positive sum.

    Returns
    -------
    np.ndarray
        ``[features]``.

    Raises
    ------
    ValueError
        If the weights sum to zero.
    """
    total = float(np.sum(weights))
    if total <= 0:
        raise ValueError("a mean needs records with positive weight")
    return np.asarray(weights, dtype=np.float64) @ x / total


def class_weights(y: np.ndarray, weights: np.ndarray | None) -> tuple[np.ndarray, np.ndarray]:
    """
    Weights of the positives and of the negatives of one condition.

    Parameters
    ----------
    y : np.ndarray
        States 1.0, 0.0 or NaN.
    weights : np.ndarray | None
        Record weights; None gives one per record.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        Positive and negative weights, zero elsewhere (undefined records included).
    """
    w = np.ones(len(y)) if weights is None else np.asarray(weights, dtype=np.float64)
    return np.where(y == 1, w, 0.0), np.where(y == 0, w, 0.0)


def ledoit_wolf(residuals: np.ndarray, weights: np.ndarray) -> dict[str, Any]:
    """
    Ledoit-Wolf shrunk covariance of centred rows with record weights.

    Integer weights give exactly ``sklearn.covariance.ledoit_wolf(..., assume_centered=True)`` on the matrix
    with each row repeated as often as its weight.

    Parameters
    ----------
    residuals : np.ndarray
        ``[records, features]`` centred rows.
    weights : np.ndarray
        Nonnegative weight per row.

    Returns
    -------
    dict[str, Any]
        ``covariance`` ``[features, features]`` and ``shrinkage``.
    """
    keep = np.asarray(weights) > 0
    r = np.asarray(residuals, dtype=np.float64)[keep]
    w = np.asarray(weights, dtype=np.float64)[keep]
    n, p = float(w.sum()), r.shape[1]
    gram = (r * w[:, None]).T @ r
    empirical = gram / n
    trace = float(np.trace(empirical))
    mu = trace / p
    delta_raw = float(np.sum(empirical**2))
    beta_raw = float(w @ np.sum(r**2, axis=1) ** 2)
    beta = (beta_raw / n - delta_raw) / (p * n)
    delta = (delta_raw - 2.0 * mu * trace + p * mu**2) / p
    beta = min(beta, delta)
    shrinkage = 0.0 if beta == 0 else beta / delta
    covariance = (1.0 - shrinkage) * empirical
    covariance[np.diag_indices(p)] += shrinkage * mu
    return {"covariance": covariance, "shrinkage": float(shrinkage)}


def within_class_covariance(
    x: np.ndarray, y: np.ndarray, weights: np.ndarray | None = None
) -> dict[str, Any]:
    """
    Pooled within-class covariance of one condition's defined records, with Ledoit-Wolf shrinkage.

    Each defined record is centred on its class mean (weighted); the shrunk covariance of those residuals
    pools both classes.

    Parameters
    ----------
    x : np.ndarray
        ``[records, features]``.
    y : np.ndarray
        States 1.0, 0.0 or NaN.
    weights : np.ndarray | None
        Record weights.

    Returns
    -------
    dict[str, Any]
        ``covariance``, ``shrinkage``, ``positive_mean`` and ``negative_mean``.
    """
    positive, negative = class_weights(y, weights)
    mean_positive, mean_negative = weighted_mean(x, positive), weighted_mean(x, negative)
    defined = (positive + negative) > 0
    centred = np.where((y[defined] == 1)[:, None], x[defined] - mean_positive, x[defined] - mean_negative)
    fit = ledoit_wolf(centred, (positive + negative)[defined])
    return {**fit, "positive_mean": mean_positive, "negative_mean": mean_negative}


def source_means(
    x: np.ndarray, sources: np.ndarray, weights: np.ndarray | None = None
) -> dict[str, np.ndarray]:
    """
    Weighted mean feature vector of each source, over all its records whatever their labels.

    Parameters
    ----------
    x : np.ndarray
        ``[records, features]``.
    sources : np.ndarray
        Source of each record.
    weights : np.ndarray | None
        Record weights.

    Returns
    -------
    dict[str, np.ndarray]
        Source to mean, for sources with positive weight.
    """
    w = np.ones(len(x)) if weights is None else np.asarray(weights, dtype=np.float64)
    names = [name for name in np.unique(sources) if w[sources == name].sum() > 0]
    return {name: weighted_mean(x, np.where(sources == name, w, 0.0)) for name in names}


def centre_by_source(x: np.ndarray, sources: np.ndarray, means: dict[str, np.ndarray]) -> np.ndarray:
    """
    Features minus the mean of their source.

    Parameters
    ----------
    x : np.ndarray
        ``[records, features]``.
    sources : np.ndarray
        Source of each record.
    means : dict[str, np.ndarray]
        Source to mean; every source of ``sources`` must be present.

    Returns
    -------
    np.ndarray
        Centred copy.

    Raises
    ------
    KeyError
        If a source has no mean.
    """
    out = np.array(x, dtype=np.float64, copy=True)
    for name in np.unique(sources):
        out[sources == name] -= means[str(name)]
    return out


def key_k1(x: np.ndarray, y: np.ndarray, weights: np.ndarray | None = None) -> np.ndarray:
    """
    K1: mean of the positives.

    Parameters
    ----------
    x : np.ndarray
        ``[records, features]``.
    y : np.ndarray
        States of one condition.
    weights : np.ndarray | None
        Record weights.

    Returns
    -------
    np.ndarray
        ``[features]``.
    """
    return weighted_mean(x, class_weights(y, weights)[0])


def key_k2(x: np.ndarray, y: np.ndarray, weights: np.ndarray | None = None) -> np.ndarray:
    """
    K2: mean of the positives minus mean of the negatives.

    Parameters
    ----------
    x : np.ndarray
        ``[records, features]``.
    y : np.ndarray
        States of one condition.
    weights : np.ndarray | None
        Record weights.

    Returns
    -------
    np.ndarray
        ``[features]``.
    """
    positive, negative = class_weights(y, weights)
    return weighted_mean(x, positive) - weighted_mean(x, negative)


def key_k3(
    x: np.ndarray, y: np.ndarray, no_positive: np.ndarray, weights: np.ndarray | None = None
) -> np.ndarray:
    """
    K3: mean of the positives minus mean of the records with no positive label.

    Parameters
    ----------
    x : np.ndarray
        ``[records, features]``.
    y : np.ndarray
        States of one condition.
    no_positive : np.ndarray
        Boolean per record: no positive among the 14 conditions.
    weights : np.ndarray | None
        Record weights.

    Returns
    -------
    np.ndarray
        ``[features]``.
    """
    positive, _ = class_weights(y, weights)
    w = np.ones(len(x)) if weights is None else np.asarray(weights, dtype=np.float64)
    return weighted_mean(x, positive) - weighted_mean(x, np.where(no_positive, w, 0.0))


def key_k4(x: np.ndarray, y: np.ndarray, weights: np.ndarray | None = None) -> dict[str, Any]:
    """
    K4: the K2 difference whitened by the shrunk pooled within-class covariance.

    Parameters
    ----------
    x : np.ndarray
        ``[records, features]``.
    y : np.ndarray
        States of one condition.
    weights : np.ndarray | None
        Record weights.

    Returns
    -------
    dict[str, Any]
        ``key``, ``covariance``, ``shrinkage``, ``positive_mean`` and ``negative_mean``.
    """
    fit = within_class_covariance(x, y, weights)
    key = np.linalg.solve(fit["covariance"], fit["positive_mean"] - fit["negative_mean"])
    return {**fit, "key": key}


def key_k5(
    x: np.ndarray,
    y: np.ndarray,
    sources: np.ndarray,
    weights: np.ndarray | None = None,
    minimum: int = SOURCE_MINIMUM,
) -> dict[str, Any]:
    """
    K5: K2 inside each source with enough positives and negatives, averaged with equal source weights.

    Parameters
    ----------
    x : np.ndarray
        ``[records, features]``.
    y : np.ndarray
        States of one condition.
    sources : np.ndarray
        Source of each record.
    weights : np.ndarray | None
        Record weights; the minimum counts records with positive weight.
    minimum : int
        Required positives and required negatives in a source.

    Returns
    -------
    dict[str, Any]
        ``key`` and ``sources`` (the sources averaged).

    Raises
    ------
    ValueError
        If no source has enough positives and negatives.
    """
    positive, negative = class_weights(y, weights)
    keys, used = [], []
    for name in np.unique(sources):
        inside = sources == name
        if (positive[inside] > 0).sum() >= minimum and (negative[inside] > 0).sum() >= minimum:
            keys.append(
                weighted_mean(x, np.where(inside, positive, 0.0))
                - weighted_mean(x, np.where(inside, negative, 0.0))
            )
            used.append(str(name))
    if not keys:
        raise ValueError("no source has enough positives and negatives for K5")
    return {"key": np.mean(keys, axis=0), "sources": used}


def key_k7(
    x: np.ndarray,
    y: np.ndarray,
    clusters: int,
    covariance: np.ndarray,
    negative_mean: np.ndarray,
    seed: int = KMEANS_SEED,
) -> dict[str, Any]:
    """
    K7: k-means centres of the positives, each minus the negative mean, whitened by K4's covariance.

    Parameters
    ----------
    x : np.ndarray
        ``[records, features]``.
    y : np.ndarray
        States of one condition.
    clusters : int
        Number of centres.
    covariance : np.ndarray
        K4's shrunk within-class covariance.
    negative_mean : np.ndarray
        Mean of the negatives.
    seed : int
        k-means seed.

    Returns
    -------
    dict[str, Any]
        ``keys`` ``[clusters, features]``, ``sizes`` (positives per centre) and ``inertia``.
    """
    from sklearn.cluster import KMeans

    positives = x[y == 1]
    model = KMeans(n_clusters=clusters, n_init=KMEANS_INIT, random_state=seed).fit(positives)
    keys = np.linalg.solve(covariance, (model.cluster_centers_ - negative_mean).T).T
    return {
        "keys": keys,
        "sizes": np.bincount(model.labels_, minlength=clusters),
        "inertia": float(model.inertia_),
    }


def unit_rows(x: np.ndarray) -> np.ndarray:
    """
    Rows scaled to unit Euclidean norm; zero rows stay zero.

    Parameters
    ----------
    x : np.ndarray
        ``[records, features]``.

    Returns
    -------
    np.ndarray
        Unit rows.
    """
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    return np.divide(x, norms, out=np.zeros_like(x, dtype=np.float64), where=norms > 0)


def cosine_score(x: np.ndarray, key: np.ndarray) -> np.ndarray:
    """
    Cosine similarity of each row with a key.

    Parameters
    ----------
    x : np.ndarray
        ``[records, features]``.
    key : np.ndarray
        ``[features]``.

    Returns
    -------
    np.ndarray
        ``[records]``.
    """
    return unit_rows(x) @ (key / np.linalg.norm(key))


def max_score(x: np.ndarray, keys: np.ndarray) -> np.ndarray:
    """
    Maximum dot product over several keys.

    Parameters
    ----------
    x : np.ndarray
        ``[records, features]``.
    keys : np.ndarray
        ``[keys, features]``.

    Returns
    -------
    np.ndarray
        ``[records]``.
    """
    return np.max(x @ keys.T, axis=1)


def neighbour_share(
    query: np.ndarray,
    reference: np.ndarray,
    labels: np.ndarray,
    neighbours: int = NEIGHBOURS,
    chunk: int = 512,
) -> np.ndarray:
    """
    N1: share of positives among the nearest reference records (cosine) where each condition is defined.

    Parameters
    ----------
    query : np.ndarray
        ``[queries, features]`` records to score.
    reference : np.ndarray
        ``[references, features]`` train records.
    labels : np.ndarray
        ``[references, conditions]`` states of the references.
    neighbours : int
        Neighbours per query and condition.
    chunk : int
        Queries per block.

    Returns
    -------
    np.ndarray
        ``[queries, conditions]`` shares in [0, 1].
    """
    reference_unit = unit_rows(reference).astype(np.float32)
    query_unit = unit_rows(query).astype(np.float32)
    defined = ~np.isnan(labels)
    positive = labels == 1
    out = np.empty((len(query), labels.shape[1]))
    for low in range(0, len(query), chunk):
        similarity = query_unit[low : low + chunk] @ reference_unit.T
        for column in range(labels.shape[1]):
            masked = np.where(defined[:, column][None, :], similarity, -np.inf)
            nearest = np.argpartition(-masked, neighbours - 1, axis=1)[:, :neighbours]
            out[low : low + chunk, column] = positive[nearest, column].mean(axis=1)
    return out


def project_out(key: np.ndarray, others: np.ndarray) -> np.ndarray:
    """
    Remove from a key its projection on the span of other keys.

    Parameters
    ----------
    key : np.ndarray
        ``[features]``.
    others : np.ndarray
        ``[keys, features]``.

    Returns
    -------
    np.ndarray
        ``key`` minus its least-squares projection on the rows of ``others``.
    """
    basis, _ = np.linalg.qr(np.asarray(others, dtype=np.float64).T)
    return key - basis @ (basis.T @ key)


def cosine(first: np.ndarray, second: np.ndarray) -> float:
    """
    Cosine of two vectors.

    Parameters
    ----------
    first, second : np.ndarray
        ``[features]``.

    Returns
    -------
    float
        Cosine.
    """
    return float(first @ second / (np.linalg.norm(first) * np.linalg.norm(second)))


def single_keys(
    x: np.ndarray,
    labels: np.ndarray,
    sources: np.ndarray,
    weights: np.ndarray | None = None,
    constructions: tuple[str, ...] = SINGLE_KEYS,
) -> dict[str, Any]:
    """
    K1 to K6 for every condition, on one weighting of the train records.

    Parameters
    ----------
    x : np.ndarray
        ``[records, features]`` standardized train features.
    labels : np.ndarray
        ``[records, conditions]`` states.
    sources : np.ndarray
        Source of each record.
    weights : np.ndarray | None
        Record weights.
    constructions : tuple[str, ...]
        Constructions to build.

    Returns
    -------
    dict[str, Any]
        ``keys`` (construction to ``[conditions, features]``), ``shrinkage`` (K4 and K6 to one value per
        condition), ``k5_sources`` (per condition), ``source_means`` (for K6) and, for K7, ``k4_fits``
        (per condition: covariance and negative mean).
    """
    conditions = labels.shape[1]
    no_positive = ~(labels == 1).any(axis=1)
    keys = {name: np.empty((conditions, x.shape[1])) for name in constructions}
    shrinkage = {name: np.full(conditions, np.nan) for name in ("K4", "K6") if name in constructions}
    out: dict[str, Any] = {"keys": keys, "shrinkage": shrinkage, "k5_sources": [], "k4_fits": []}
    centred = None
    if "K6" in constructions:
        out["source_means"] = source_means(x, sources, weights)
        centred = centre_by_source(x, sources, out["source_means"])
    for column in range(conditions):
        y = labels[:, column]
        if "K1" in keys:
            keys["K1"][column] = key_k1(x, y, weights)
        if "K2" in keys:
            keys["K2"][column] = key_k2(x, y, weights)
        if "K3" in keys:
            keys["K3"][column] = key_k3(x, y, no_positive, weights)
        if "K4" in keys:
            fit = key_k4(x, y, weights)
            keys["K4"][column], shrinkage["K4"][column] = fit["key"], fit["shrinkage"]
            out["k4_fits"].append({"covariance": fit["covariance"], "negative_mean": fit["negative_mean"]})
        if "K5" in keys:
            fit5 = key_k5(x, y, sources, weights)
            keys["K5"][column] = fit5["key"]
            out["k5_sources"].append(fit5["sources"])
        if "K6" in keys:
            fit6 = key_k4(centred, y, weights)
            keys["K6"][column], shrinkage["K6"][column] = fit6["key"], fit6["shrinkage"]
    return out


def score_single(
    construction: str,
    x: np.ndarray,
    keys: np.ndarray,
    sources: np.ndarray | None = None,
    means: dict[str, np.ndarray] | None = None,
) -> np.ndarray:
    """
    Scores of one single-key construction for every condition.

    Parameters
    ----------
    construction : str
        K1 to K6.
    x : np.ndarray
        ``[records, features]`` standardized features.
    keys : np.ndarray
        ``[conditions, features]``.
    sources : np.ndarray | None
        Source of each record (K6).
    means : dict[str, np.ndarray] | None
        Source means used to centre the records (K6).

    Returns
    -------
    np.ndarray
        ``[records, conditions]``: cosine for K1, dot product otherwise.
    """
    if construction == "K1":
        return unit_rows(x) @ unit_rows(keys).T
    if construction == "K6":
        return centre_by_source(x, sources, means) @ keys.T
    return x @ keys.T
