"""Local-normal draws, screen outcomes and the adoption reading of a pipeline comparison, for Experiment 044.

The draw plan and threshold logic are Experiment 030's and 033's (``numpy.random.default_rng([seed, m,
draw])`` over the local normals, thresholds from ``finding_screen.split_thresholds``), written here once for
any number of pipelines.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .finding_screen import referred_matrix, split_thresholds


def local_normal_plans(local_normal: np.ndarray, sizes: tuple[int, ...], draws: int, seed: int
                       ) -> dict[int, list[np.ndarray]]:
    """
    Draw the local normals of every size, as Experiment 030 does.

    Parameters
    ----------
    local_normal : np.ndarray
        Row positions of the local normals, in their fixed order.
    sizes : tuple[int, ...]
        Numbers of normals per draw.
    draws : int
        Draws per size.
    seed : int
        First entry of the ``numpy.random.default_rng([seed, m, draw])`` seed.

    Returns
    -------
    dict[int, list[np.ndarray]]
        Per size, one sorted array of row positions per draw.
    """
    return {m: [local_normal[np.sort(np.random.default_rng([seed, m, draw])
                                     .choice(len(local_normal), m, replace=False))]
                for draw in range(draws)] for m in sizes}


def screen_draws(matrix: np.ndarray, plans: list[np.ndarray], per_mille: int, shares: tuple[int, ...],
                 evaluation: np.ndarray, masks: dict[str, np.ndarray], local_masks: dict[str, np.ndarray]
                 ) -> dict[str, Any]:
    """
    Compute the thresholds and outcomes of one screen over every draw of local normals.

    Parameters
    ----------
    matrix : np.ndarray
        ``(n, 1 + F)`` scores of every row: the binary readout first, then the finding scores.
    plans : list[np.ndarray]
        Row positions of the normals of each draw.
    per_mille : int
        Budget in thousandths.
    shares : tuple[int, ...]
        Finding shares in thousandths (``split_thresholds``).
    evaluation : np.ndarray
        Boolean mask of the evaluation rows; the other rows form the local pool.
    masks : dict[str, np.ndarray]
        Outcome masks over the evaluation rows.
    local_masks : dict[str, np.ndarray]
        Outcome masks over the local-pool rows.

    Returns
    -------
    dict[str, Any]
        ``thresholds`` ``(draws, 1 + F)``, ``shares`` (share of draws that refer each evaluation row),
        ``outcomes`` and ``local`` (referral rate per draw of each outcome).
    """
    thresholds = np.array([split_thresholds(matrix[positions], per_mille, shares) for positions in plans])
    referred = referred_matrix(matrix[evaluation], thresholds)
    local = referred_matrix(matrix[~evaluation], thresholds)
    return {"thresholds": thresholds, "shares": referred.mean(axis=1),
            "outcomes": {name: referred[mask].mean(axis=0) for name, mask in masks.items()},
            "local": {name: local[mask].mean(axis=0) for name, mask in local_masks.items()}}


def pipeline_reading(interval: list[float], margin: float) -> str:
    """
    Read a candidate minus current difference by Experiment 044's rule.

    Parameters
    ----------
    interval : list[float]
        Lower and upper bounds of the difference.
    margin : float
        Positive no-worse margin.

    Returns
    -------
    str
        ``adopt_v3`` when the lower bound is above 0, ``no_worse_keep_v2`` when it is above ``-margin``,
        otherwise ``keep_v2``.
    """
    if interval[0] > 0:
        return "adopt_v3"
    if interval[0] > -margin:
        return "no_worse_keep_v2"
    return "keep_v2"
