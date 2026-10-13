"""Focal-versus-diffuse explanation switch for Experiment 049, from beat-aligned wave units."""

from __future__ import annotations

import numpy as np

from .finding_screen import referred_matrix, split_thresholds
from .lead_wave_maps import UnitMap

LEADS_PER_BEAT = 12
WAVES_PER_LEAD = 4
UNITS_PER_BEAT = LEADS_PER_BEAT * WAVES_PER_LEAD


def beat_layout_ok(unit_map: UnitMap) -> bool:
    """
    Return whether units follow ``beat_unit_map``'s order: index = (beat * 12 + lead) * 4 + wave.

    Parameters
    ----------
    unit_map : UnitMap
        Units of one ECG.

    Returns
    -------
    bool
        True if the count is a multiple of 48, the lead of unit u is (u // 4) mod 12, and every lead of a
        beat has the same four wave start times.
    """
    count = len(unit_map.scores)
    if count == 0 or count % UNITS_PER_BEAT:
        return False
    expected = (np.arange(count) // WAVES_PER_LEAD) % LEADS_PER_BEAT
    starts = np.asarray(unit_map.starts).reshape(-1, LEADS_PER_BEAT, WAVES_PER_LEAD)
    return bool(np.array_equal(unit_map.leads, expected) and (starts == starts[:, :1, :]).all())


def beat_scores(unit_map: UnitMap) -> np.ndarray:
    """Return each beat's score, the maximum over its 48 lead-and-wave units."""
    return np.asarray(unit_map.scores, dtype=np.float64).reshape(-1, UNITS_PER_BEAT).max(axis=1)


def focal_ratio(unit_map: UnitMap) -> float:
    """Return the ECG's maximum beat score divided by its median beat score."""
    scores = beat_scores(unit_map)
    return float(scores.max() / np.median(scores))


def combined_referral(binary: np.ndarray, finding: np.ndarray, normal: np.ndarray, per_mille: int = 50,
                      share: int = 50) -> tuple[np.ndarray, np.ndarray]:
    """
    Refer by the binary readout or a finding score, with thresholds split on normal ECGs (033's rule).

    Parameters
    ----------
    binary : np.ndarray
        Binary readout score of every ECG.
    finding : np.ndarray
        Finding score of every ECG.
    normal : np.ndarray
        Boolean mask of the normal ECGs that set the thresholds.
    per_mille : int
        Total budget in thousandths.
    share : int
        The finding score's share of the budget in thousandths.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        Whether each ECG is referred, and the two thresholds (binary, finding).
    """
    matrix = np.column_stack([np.asarray(binary, dtype=np.float64), np.asarray(finding, dtype=np.float64)])
    thresholds = split_thresholds(matrix[np.asarray(normal, dtype=bool)], per_mille, (share,))
    return referred_matrix(matrix, thresholds[None, :])[:, 0], thresholds
