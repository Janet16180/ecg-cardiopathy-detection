"""Fit-set helpers for Experiment 026b: the normal-ECG manifold fitted on several hospitals.

The distance score itself is Experiment 026's (``normal_manifold.fit_mahalanobis``); only the normal ECGs it
is fitted on change. These helpers pick those ECGs: the training quality policy on the exact window the
frozen features were computed from, and an equal-count draw per hospital family.
"""

from __future__ import annotations

import numpy as np

from .challenge_features import canonical_window
from .ecg_quality import assess


def window_reasons(signal: np.ndarray, names: list[str], start: int) -> list[str]:
    """
    Quality-policy exclusion reasons of the ten-second window a feature was computed from.

    Parameters
    ----------
    signal : np.ndarray
        Time-major ``(samples, 12)`` record in mV at 500 Hz.
    names : list[str]
        Stored lead name of each column.
    start : int
        First sample of the window, as saved with the features.

    Returns
    -------
    list[str]
        ``ecg_quality.assess`` exclusion reasons; empty when the window passes.
    """
    window = canonical_window(signal, names, start)
    reasons, _ = assess(window)
    return reasons


def equal_family_subsample(families: np.ndarray, seed: int) -> np.ndarray:
    """
    Draw the same number of records from every family, without replacement.

    Each family gives as many records as the smallest one has. Families are visited in sorted order with
    one generator, so the draw depends only on the family labels and the seed.

    Parameters
    ----------
    families : np.ndarray
        Family name of each record.
    seed : int
        Seed of ``numpy.random.default_rng``.

    Returns
    -------
    np.ndarray
        Sorted positions of the selected records.
    """
    names, counts = np.unique(families, return_counts=True)
    size = counts.min()
    rng = np.random.default_rng(seed)
    chosen = [rng.choice(np.flatnonzero(families == name), size=size, replace=False) for name in names]
    return np.sort(np.concatenate(chosen))
