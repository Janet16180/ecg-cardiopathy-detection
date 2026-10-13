"""Two-layer explanation maps for Experiment 045: one quantile for both layers' red thresholds."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .lead_wave_maps import UnitMap

GRID_LOW = 0.9
GRID_HIGH = 1.0
GRID_STEPS = 1001


@dataclass
class Explanation:
    """Explanation of one ECG: which layer supplied it, whether each layer is red, and that layer's units."""

    layer: int
    first_red: bool
    second_red: bool
    units: UnitMap


def quantile_grid(low: float = GRID_LOW, high: float = GRID_HIGH, steps: int = GRID_STEPS) -> np.ndarray:
    """Return the evenly spaced quantile grid, rounded to four decimals."""
    return np.round(np.linspace(low, high, steps), 4)


def layer_thresholds(first_normal: np.ndarray, second_normal: np.ndarray, q: float) -> tuple[float, float]:
    """
    Return each layer's red threshold, the same quantile of its worst-unit scores on normal ECGs.

    Parameters
    ----------
    first_normal, second_normal : np.ndarray
        Worst-unit score of each normal ECG in each layer.
    q : float
        Quantile, with NumPy's default linear interpolation.

    Returns
    -------
    tuple[float, float]
        The two thresholds.
    """
    return float(np.quantile(first_normal, q)), float(np.quantile(second_normal, q))


def union_share(first_normal: np.ndarray, second_normal: np.ndarray, q: float) -> float:
    """Return the share of normal ECGs with either layer strictly above its quantile-q threshold."""
    first, second = layer_thresholds(first_normal, second_normal, q)
    return float(np.mean((first_normal > first) | (second_normal > second)))


def choose_quantile(first_normal: np.ndarray, second_normal: np.ndarray, target: float,
                    grid: np.ndarray | None = None) -> dict[str, float | int]:
    """
    Choose the grid quantile whose union red share on normal ECGs is nearest the target.

    Ties are broken by the median of the tied grid values (the lower middle one if their number is even).

    Parameters
    ----------
    first_normal, second_normal : np.ndarray
        Worst-unit score of each normal ECG in each layer.
    target : float
        Target share of normal ECGs with either layer red.
    grid : np.ndarray | None
        Candidate quantiles in increasing order; ``quantile_grid()`` if omitted.

    Returns
    -------
    dict[str, float | int]
        ``q``, both thresholds, the achieved ``share`` and ``red_normals``, and the number of ``tied`` values.
    """
    grid = quantile_grid() if grid is None else np.asarray(grid, dtype=np.float64)
    shares = np.array([union_share(first_normal, second_normal, q) for q in grid])
    distance = np.abs(shares - target)
    tied = np.flatnonzero(np.isclose(distance, distance.min(), rtol=0.0, atol=1e-12))
    q = float(grid[tied[(len(tied) - 1) // 2]])
    first, second = layer_thresholds(first_normal, second_normal, q)
    share = union_share(first_normal, second_normal, q)
    return {"q": q, "first_threshold": first, "second_threshold": second, "share": share,
            "red_normals": int(round(share * len(first_normal))), "tied": int(len(tied))}


def explain(first: UnitMap, second: UnitMap, first_threshold: float, second_threshold: float) -> Explanation:
    """
    Explain one ECG with the first layer's units if that layer is red, otherwise with the second layer's.

    Parameters
    ----------
    first, second : UnitMap
        The ECG's units in each layer.
    first_threshold, second_threshold : float
        Red thresholds; a layer is red when its worst unit is strictly above its threshold.

    Returns
    -------
    Explanation
        The supplying layer (1 or 2), both red flags and the supplying layer's units, whose top unit is the
        explanation.
    """
    first_red = bool(first.scores.max() > first_threshold)
    second_red = bool(second.scores.max() > second_threshold)
    return Explanation(1 if first_red else 2, first_red, second_red, first if first_red else second)


def red_unit_count(first: UnitMap, second: UnitMap, first_threshold: float, second_threshold: float) -> int:
    """Return the number of units above their layer's threshold, both layers added."""
    return int((first.scores > first_threshold).sum() + (second.scores > second_threshold).sum())


def mean_logit(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    """Return the elementwise mean of two logit vectors."""
    return (np.asarray(first, dtype=np.float64) + np.asarray(second, dtype=np.float64)) / 2
