"""Strict referral-budget calibration and descriptive confusion matrices for corrective evaluation."""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

from .cpu_screening041 import calibrate_threshold


def aligned(y: np.ndarray, scores: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Require aligned finite scores and a defined binary target for every record."""
    y, scores = np.asarray(y), np.asarray(scores, dtype=float)
    if y.ndim != 1 or scores.shape != y.shape or not len(y):
        raise ValueError("Targets and scores must be nonempty aligned vectors")
    if not np.isin(y, [0, 1]).all() or not np.isfinite(scores).all():
        raise ValueError("Undefined targets or nonfinite scores")
    return y.astype(int), scores


def budget_threshold(y: np.ndarray, scores: np.ndarray) -> dict[str, Any]:
    """Reuse the frozen strict five-percent normal calibration rule, preserving ties."""
    y, scores = aligned(y, scores)
    result = calibrate_threshold(y, scores)
    normals = scores[y == 0]
    return {
        "threshold": result["threshold"], "comparison": ">", "normals": len(normals),
        "flagged_normals": result["referred_normal_count"], "allowed_normals": result["allowed_count"],
        "tied_normals": int(np.count_nonzero(normals == result["threshold"])), "normal_budget": 0.05,
    }


def matrix(y: np.ndarray, scores: np.ndarray, threshold: float) -> dict[str, int | float | None]:
    """Count every evaluated record under score greater than threshold and report exact denominators."""
    y, scores = aligned(y, scores)
    if not np.isfinite(threshold):
        raise ValueError("Threshold must be finite")
    flagged, positive = scores > threshold, y == 1
    tp = int(np.count_nonzero(flagged & positive))
    tn = int(np.count_nonzero(~flagged & ~positive))
    fp = int(np.count_nonzero(flagged & ~positive))
    fn = int(np.count_nonzero(~flagged & positive))
    return {
        "tp": tp, "tn": tn, "fp": fp, "fn": fn, "ecgs": len(y),
        "positives": tp + fn, "negatives": tn + fp,
        "sensitivity": tp / (tp + fn) if tp + fn else None,
        "specificity": tn / (tn + fp) if tn + fp else None,
        "precision": tp / (tp + fp) if tp + fp else None,
        "referral_fraction": (tp + fp) / len(y),
    }


def point_metrics(y: np.ndarray, scores: np.ndarray, threshold: float) -> dict[str, Any]:
    """Add ranking metrics without confusing AUROC with threshold accuracy."""
    y, scores = aligned(y, scores)
    return {
        **matrix(y, scores, threshold),
        "auroc": float(roc_auc_score(y, scores)) if len(np.unique(y)) == 2 else None,
        "average_precision": float(average_precision_score(y, scores)) if np.any(y == 1) else None,
    }


def threshold_sweep(y: np.ndarray, scores: np.ndarray, frozen_threshold: float) -> dict[str, list]:
    """Export an ordered descriptive score sweep without refitting the frozen operating point."""
    y, scores = aligned(y, scores)
    grid = np.unique(np.r_[np.nextafter(scores.min(), -np.inf),
                           np.quantile(scores, np.linspace(0, 1, 201)), frozen_threshold])
    rows = [matrix(y, scores, float(value)) for value in grid]
    keys = ("tp", "tn", "fp", "fn", "sensitivity", "specificity", "precision", "referral_fraction")
    return {"threshold": grid.tolist(), **{key: [row[key] for row in rows] for key in keys}}
