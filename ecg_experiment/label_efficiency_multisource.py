"""Pooled label draws and the draw-mean AUROC bootstrap for Experiment 025b.

025b repeats 025's label draws on a pool of PTB-XL and Challenge training labels. The bootstrap here resamples
evaluation patients once per set and scores every readout of every label draw on the same resamples, so a
contrast between two arms or encoders is paired over both the label draws and the evaluation patients. AUROC
on a resample is computed from row multiplicities, which equals ``roc_auc_score`` on the repeated rows.
"""

from __future__ import annotations

import numpy as np

from .label_efficiency import reading
from .normal_manifold import patient_resamples

SortedScores = tuple[np.ndarray, np.ndarray, np.ndarray]


def challenge_units(sources: np.ndarray, records: np.ndarray) -> np.ndarray:
    """
    Patient unit of each Challenge record, which has no patient ID.

    Parameters
    ----------
    sources, records : np.ndarray
        Source and record name of each row.

    Returns
    -------
    np.ndarray
        ``challenge:<source>:<record>`` per row.
    """
    return np.array([f"challenge:{source}:{record}" for source, record in zip(sources, records, strict=True)])


def plus_positions(ptb_positions: np.ndarray, ptb_count: int, total: int) -> np.ndarray:
    """
    Stacked positions of a PTB-XL draw followed by every Challenge row.

    Parameters
    ----------
    ptb_positions : np.ndarray
        Positions of the drawn PTB-XL rows, which come first in the stacked table.
    ptb_count : int
        Number of PTB-XL rows in the stacked table.
    total : int
        Number of stacked rows.

    Returns
    -------
    np.ndarray
        Sorted positions.
    """
    return np.concatenate([np.sort(ptb_positions), np.arange(ptb_count, total)])


def resample_counts(units: np.ndarray, y: np.ndarray, draws: int, seed: int) -> tuple[np.ndarray, int]:
    """
    Row multiplicities of whole-unit bootstrap resamples, skipping those with one class.

    Parameters
    ----------
    units : np.ndarray
        Bootstrap unit (patient or record) of each row.
    y : np.ndarray
        Binary labels.
    draws : int
        Number of resamples.
    seed : int
        Seed of ``numpy.random.default_rng``.

    Returns
    -------
    tuple[np.ndarray, int]
        One row of counts per valid resample, and the number of resamples skipped.
    """
    resamples, invalid = patient_resamples(units, y, draws, seed)
    counts = np.zeros((len(resamples), len(y)), dtype=np.int32)
    for index, rows in enumerate(resamples):
        counts[index] = np.bincount(rows, minlength=len(y))
    return counts, invalid


def sort_scores(y: np.ndarray, scores: np.ndarray) -> SortedScores:
    """
    Sort order, tie groups and labels of one score vector, reused for every resample.

    Parameters
    ----------
    y : np.ndarray
        Binary labels.
    scores : np.ndarray
        One score per row.

    Returns
    -------
    SortedScores
        Row order by score, the tie-group index of each sorted row, and the sorted labels as booleans.
    """
    order = np.argsort(scores, kind="mergesort")
    ordered = scores[order]
    groups = np.concatenate([[0], np.cumsum(ordered[1:] != ordered[:-1])])
    return order, groups, np.asarray(y)[order] == 1


def weighted_auroc(prepared: SortedScores, counts: np.ndarray) -> float:
    """
    AUROC of rows repeated by their counts, with ties counted as one half.

    Parameters
    ----------
    prepared : SortedScores
        Output of ``sort_scores``.
    counts : np.ndarray
        Multiplicity of each row.

    Returns
    -------
    float
        The AUROC ``roc_auc_score`` gives on the repeated rows.
    """
    order, groups, positive = prepared
    weights = counts[order].astype(np.float64)
    size = int(groups[-1]) + 1
    positives = np.bincount(groups, np.where(positive, weights, 0.0), minlength=size)
    negatives = np.bincount(groups, np.where(positive, 0.0, weights), minlength=size)
    below = np.cumsum(negatives) - negatives
    return float(np.sum(positives * (below + 0.5 * negatives)) / (positives.sum() * negatives.sum()))


def bootstrap_mean_auroc(y: np.ndarray, scores: np.ndarray, counts: np.ndarray) -> np.ndarray:
    """
    Mean AUROC over label draws on each bootstrap resample.

    Parameters
    ----------
    y : np.ndarray
        Binary labels.
    scores : np.ndarray
        One score vector, or a 2-D array with one score vector per label draw.
    counts : np.ndarray
        Output of ``resample_counts``.

    Returns
    -------
    np.ndarray
        One draw-mean AUROC per resample.
    """
    prepared = [sort_scores(y, vector) for vector in np.atleast_2d(scores)]
    return np.array([np.mean([weighted_auroc(item, row) for item in prepared]) for row in counts])


def interval(observed: float, first: np.ndarray, second: np.ndarray) -> dict[str, float]:
    """
    Observed paired difference with the 95% percentile interval of its resampled values.

    Parameters
    ----------
    observed : float
        Observed difference, first minus second.
    first, second : np.ndarray
        Resampled values of the two readouts on the same resamples.

    Returns
    -------
    dict[str, float]
        ``difference``, ``ci_low`` and ``ci_high``.
    """
    low, high = np.percentile(np.asarray(first) - np.asarray(second), [2.5, 97.5])
    return {"difference": float(observed), "ci_low": float(low), "ci_high": float(high)}


def verdict(first: str, second: str, summary: dict[str, float | int], bounds: dict[str, float]) -> str:
    """
    Combine 025's 90% draw rule with the side of the bootstrap interval.

    Parameters
    ----------
    first, second : str
        Readout names of the contrast ``first - second``.
    summary : dict[str, float | int]
        Output of ``label_efficiency.paired_summary`` on the per-draw AUROCs.
    bounds : dict[str, float]
        Output of ``interval``.

    Returns
    -------
    str
        ``"<first> better"``, ``"<second> better"`` or ``"not distinguished"``.
    """
    rule = reading(first, second, summary)
    agrees = {f"{first} better": bounds["ci_low"] > 0, f"{second} better": bounds["ci_high"] < 0}
    return rule if agrees.get(rule, False) else "not distinguished"


def decision(verdicts: list[str]) -> str:
    """
    Apply the prespecified decision to the primary verdicts at the read budgets.

    Parameters
    ----------
    verdicts : list[str]
        Output of ``verdict`` for ``pooled - ptbxl`` at each read budget.

    Returns
    -------
    str
        ``adopt_pooled``, ``pooling_hurts`` or ``mixed``.
    """
    result = "mixed"
    if all(item == "pooled better" for item in verdicts):
        result = "adopt_pooled"
    elif all(item == "ptbxl better" for item in verdicts):
        result = "pooling_hurts"
    return result
