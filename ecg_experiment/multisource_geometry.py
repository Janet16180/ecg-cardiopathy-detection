"""Partition bootstraps, neighbor shares and source probes for Experiment 024b (geometry across sources).

The prototypes, neighbors, medoids and AMI summaries themselves are Experiment 024's
(``embedding_geometry``); these helpers add what a pooled, multi-hospital analysis needs.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_mutual_info_score, homogeneity_score, roc_auc_score

from .embedding_geometry import NEIGHBORS, neighbor_indices
from .full_development import fit_logistic, predict

SUPERCLASSES = ("CD", "HYP", "MI", "STTC")
QUERY_CHUNK = 512


def challenge_combination(frame: pd.DataFrame) -> np.ndarray:
    """
    Superclass combination of Challenge rows in PTB-XL's format.

    Parameters
    ----------
    frame : pd.DataFrame
        Rows with ``primary`` and 0/1 columns ``MI``, ``STTC``, ``CD`` and ``HYP``.

    Returns
    -------
    np.ndarray
        ``NORM`` for a primary negative, otherwise the present superclasses in alphabetical order joined
        by ``+``.
    """
    present = frame[list(SUPERCLASSES)].to_numpy() == 1
    joined = ["+".join(name for name, found in zip(SUPERCLASSES, row, strict=True) if found)
              for row in present]
    return np.where(frame["primary"].to_numpy() == 0, "NORM", np.array(joined, dtype=object)).astype(str)


def equal_family_draw(families: np.ndarray, size: int, seed: int) -> np.ndarray:
    """
    Draw ``size`` records from every family without replacement, visiting families in sorted order.

    Parameters
    ----------
    families : np.ndarray
        Family of each record.
    size : int
        Records per family.
    seed : int
        Seed of ``numpy.random.default_rng``.

    Returns
    -------
    np.ndarray
        Sorted positions of the selected records.
    """
    rng = np.random.default_rng(seed)
    chosen = [rng.choice(np.flatnonzero(families == name), size=size, replace=False)
              for name in np.unique(families)]
    return np.sort(np.concatenate(chosen))


def explained_entropy(partition: np.ndarray, factors: dict[str, np.ndarray]) -> dict[str, float]:
    """
    Fraction of each factor's entropy explained by a partition.

    Parameters
    ----------
    partition : np.ndarray
        Cluster label of each row.
    factors : dict[str, np.ndarray]
        Categorical factors of the same rows.

    Returns
    -------
    dict[str, float]
        ``homogeneity_score`` with the factor as the true labels, per factor.
    """
    return {name: float(homogeneity_score(values, partition)) for name, values in factors.items()}


def bootstrap_partition(partition: np.ndarray, factors: dict[str, np.ndarray], draws: int, seed: int,
                        measure: Callable[[np.ndarray, np.ndarray], float] = adjusted_mutual_info_score
                        ) -> dict[str, np.ndarray]:
    """
    Record-level bootstrap of a partition-factor agreement measure, with the partition held fixed.

    Parameters
    ----------
    partition : np.ndarray
        Cluster label of each row.
    factors : dict[str, np.ndarray]
        Categorical factors of the same rows.
    draws : int
        Number of resamples.
    seed : int
        Seed of ``numpy.random.default_rng``; the same seed gives the same resamples.
    measure : Callable[[np.ndarray, np.ndarray], float]
        Agreement of factor values (first) and cluster labels (second).

    Returns
    -------
    dict[str, np.ndarray]
        One value per resample for each factor.
    """
    rng = np.random.default_rng(seed)
    values = {name: np.empty(draws) for name in factors}
    for draw in range(draws):
        rows = rng.integers(0, len(partition), len(partition))
        for name, factor in factors.items():
            values[name][draw] = measure(factor[rows], partition[rows])
    return values


def paired_interval(observed: float, first: np.ndarray, second: np.ndarray) -> dict[str, float]:
    """
    Observed difference with the 95% percentile interval of paired resampled differences.

    Parameters
    ----------
    observed : float
        Difference on all rows, first minus second.
    first, second : np.ndarray
        Paired resampled values.

    Returns
    -------
    dict[str, float]
        ``difference``, ``ci_low`` and ``ci_high``.
    """
    low, high = np.percentile(first - second, [2.5, 97.5])
    return {"difference": float(observed), "ci_low": float(low), "ci_high": float(high)}


def chunked_neighbors(train: np.ndarray, query: np.ndarray, train_groups: np.ndarray | None = None,
                      query_groups: np.ndarray | None = None, k: int = NEIGHBORS) -> np.ndarray:
    """
    024's ``neighbor_indices`` over query blocks small enough for a large training pool.

    Parameters
    ----------
    train, query : np.ndarray
        Unit vectors.
    train_groups, query_groups : np.ndarray | None
        Optional group of each row; neighbors in the query's own group are excluded.
    k : int
        Number of neighbors.

    Returns
    -------
    np.ndarray
        Integer array of shape ``(queries, k)``, most similar first.
    """
    blocks = []
    for start in range(0, len(query), QUERY_CHUNK):
        stop = start + QUERY_CHUNK
        groups = None if query_groups is None else query_groups[start:stop]
        blocks.append(neighbor_indices(train, query[start:stop], k, train_groups, groups))
    return np.concatenate(blocks)


def neighbor_shares(neighbors: np.ndarray, train_groups: np.ndarray, names: list[str]) -> dict[str, float]:
    """
    Mean share of the neighbors that belong to each group.

    Parameters
    ----------
    neighbors : np.ndarray
        Neighbor indices into ``train_groups``.
    train_groups : np.ndarray
        Group of each training row.
    names : list[str]
        Groups to report.

    Returns
    -------
    dict[str, float]
        Mean fraction per group; the fractions of one query sum to 1 over all groups.
    """
    found = train_groups[neighbors]
    return {name: float((found == name).mean()) for name in names}


def one_vs_rest_probabilities(x_train: np.ndarray, classes: np.ndarray, x_query: np.ndarray,
                              names: list[str]) -> np.ndarray:
    """
    Probabilities of the fixed logistic readout fitted for each class against the rest.

    Parameters
    ----------
    x_train : np.ndarray
        Training features.
    classes : np.ndarray
        Class of each training row.
    x_query : np.ndarray
        Query features.
    names : list[str]
        Classes, one readout each, in column order.

    Returns
    -------
    np.ndarray
        Array of shape ``(queries, len(names))``.
    """
    return np.column_stack([predict(fit_logistic(x_train, (classes == name).astype(np.int64)), x_query)
                            for name in names])


def source_probe_metrics(classes: np.ndarray, probabilities: np.ndarray,
                         names: list[str]) -> dict[str, float]:
    """
    One-vs-rest AUROC per class, their mean, and balanced accuracy of the highest-probability class.

    Parameters
    ----------
    classes : np.ndarray
        True class of each query; every class in ``names`` must occur.
    probabilities : np.ndarray
        Output of ``one_vs_rest_probabilities``.
    names : list[str]
        Classes in column order.

    Returns
    -------
    dict[str, float]
        ``auroc:<name>`` per class, ``mean_auroc`` and ``balanced_accuracy``.
    """
    assigned = np.asarray(names)[probabilities.argmax(axis=1)]
    result = {f"auroc:{name}": float(roc_auc_score(classes == name, probabilities[:, column]))
              for column, name in enumerate(names)}
    result["mean_auroc"] = float(np.mean([result[f"auroc:{name}"] for name in names]))
    recalls = [(assigned[classes == name] == name).mean() for name in names]
    result["balanced_accuracy"] = float(np.mean(recalls))
    return result


def bootstrap_source_probe(classes: np.ndarray, probabilities: np.ndarray, names: list[str], draws: int,
                           seed: int) -> dict[str, dict[str, float]]:
    """
    Record-level bootstrap intervals of the source-probe metrics.

    Parameters
    ----------
    classes : np.ndarray
        True class of each query.
    probabilities : np.ndarray
        Output of ``one_vs_rest_probabilities``.
    names : list[str]
        Classes in column order.
    draws : int
        Number of resamples; those missing a class are skipped and counted.
    seed : int
        Seed of ``numpy.random.default_rng``.

    Returns
    -------
    dict[str, dict[str, float]]
        Per metric, the observed ``value`` and its ``ci_low`` and ``ci_high``, and ``invalid_draws``.
    """
    rng = np.random.default_rng(seed)
    drawn, invalid = [], 0
    for _ in range(draws):
        rows = rng.integers(0, len(classes), len(classes))
        if len(np.unique(classes[rows])) < len(names):
            invalid += 1
            continue
        drawn.append(source_probe_metrics(classes[rows], probabilities[rows], names))
    observed = source_probe_metrics(classes, probabilities, names)
    result: dict[str, dict[str, float]] = {}
    for metric, value in observed.items():
        low, high = np.percentile([values[metric] for values in drawn], [2.5, 97.5])
        result[metric] = {"value": value, "ci_low": float(low), "ci_high": float(high)}
    result["invalid_draws"] = {"value": float(invalid)}
    return result
