"""Training-normal empirical calibration of fixed-offset ECG localization units."""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.stats import chi2

from ecg_experiment.intervals import patient_groups, patient_resample


def calibrated_scores(reference: np.ndarray, observed: np.ndarray, presorted: bool = False) -> np.ndarray:
    """
    Compare distances with conservative held-out empirical right tails.

    Parameters
    ----------
    reference : np.ndarray
        Calibration beat distances, shape (beats, leads, blocks).
    observed : np.ndarray
        Observed distances with identical channel shape.
    presorted : bool
        Reference channel distances are already ascending on the beat axis.

    Returns
    -------
    np.ndarray
        Negative log right-tail probabilities, retaining observed shape.
    """
    if reference.ndim != 3 or observed.shape[1:] != reference.shape[1:] or not len(reference):
        raise ValueError("Calibration and observed channel shapes must match")
    if not np.isfinite(reference).all() or not np.isfinite(observed).all():
        raise ValueError("Distances must be finite")
    result = np.empty_like(observed, dtype=np.float64)
    for lead in range(reference.shape[1]):
        for block in range(reference.shape[2]):
            ordered = reference[:, lead, block] if presorted else np.sort(reference[:, lead, block])
            below = np.searchsorted(ordered, observed[:, lead, block], side="left")
            result[:, lead, block] = -np.log((len(ordered) - below + 1) / (len(ordered) + 1))
    return result


def equal_width_pieces(pieces: dict[str, np.ndarray], width: int = 35) -> dict[str, np.ndarray]:
    """
    Resample fixed-offset pieces to a common coordinate count.

    Parameters
    ----------
    pieces : dict[str, np.ndarray]
        Beat pieces keyed by fixed-offset block.
    width : int
        Common sampled coordinate count.

    Returns
    -------
    dict[str, np.ndarray]
        Interpolated pieces retaining original block time support.
    """
    result = {}
    for name, values in pieces.items():
        original = np.linspace(0, 1, values.shape[-1])
        target = np.linspace(0, 1, width)
        flat = values.reshape(-1, values.shape[-1])
        result[name] = np.stack([np.interp(target, original, row) for row in flat]).reshape(
            *values.shape[:-1], width
        )
    return result


def chi_square_scores(distances: np.ndarray, dimensions: np.ndarray) -> np.ndarray:
    """
    Normalize raw Mahalanobis distances by the chi-square coordinate-count model.

    Parameters
    ----------
    distances : np.ndarray
        Beat, lead and block distances.
    dimensions : np.ndarray
        Coordinate count of each block.

    Returns
    -------
    np.ndarray
        Finite negative log model tail probabilities.
    """
    survival = chi2.sf(distances, dimensions[None, None, :])
    return -np.log(np.maximum(survival, np.finfo(float).tiny))


def mean_interval(
    patients: np.ndarray, values: np.ndarray, draws: int = 2000, seed: int = 53053
) -> dict[str, float]:
    """
    Bootstrap a scalar mean while keeping records of patients together.

    Parameters
    ----------
    patients : np.ndarray
        Patient identifiers of records.
    values : np.ndarray
        One value per record.
    draws : int
        Number of whole-patient draws.
    seed : int
        Random seed.

    Returns
    -------
    dict[str, float]
        Point mean and percentile confidence bounds.
    """
    groups = patient_groups(patients)
    rng = np.random.default_rng(seed)
    resampled = [float(values[patient_resample(groups, rng)].mean()) for _ in range(draws)]
    low, high = np.percentile(resampled, [2.5, 97.5])
    return {"estimate": float(values.mean()), "ci_low": float(low), "ci_high": float(high)}


def region_difference(
    patients_a: np.ndarray,
    values_a: np.ndarray,
    patients_b: np.ndarray,
    values_b: np.ndarray,
    draws: int = 2000,
    seed: int = 53053,
) -> dict[str, Any]:
    """
    Bootstrap a two-group contrast, retaining paired map differences within records.

    Parameters
    ----------
    patients_a, patients_b : np.ndarray
        Patient identifiers in the two disjoint infarct groups.
    values_a, values_b : np.ndarray
        Per-record values or paired differences.
    draws : int
        Number of whole-patient draws.
    seed : int
        Random seed.

    Returns
    -------
    dict[str, Any]
        Contrast and whole-patient percentile confidence bounds.
    """
    if set(patients_a) & set(patients_b):
        raise ValueError("Infarct groups must have disjoint patients")
    groups_a, groups_b = patient_groups(patients_a), patient_groups(patients_b)
    rng = np.random.default_rng(seed)
    estimates = [
        float(
            values_a[patient_resample(groups_a, rng)].mean()
            - values_b[patient_resample(groups_b, rng)].mean()
        )
        for _ in range(draws)
    ]
    low, high = np.percentile(estimates, [2.5, 97.5])
    return {
        "estimate": float(values_a.mean() - values_b.mean()),
        "ci_low": float(low),
        "ci_high": float(high),
        "first_mean": float(values_a.mean()),
        "second_mean": float(values_b.mean()),
    }


def tied_lead_membership(
    scores: np.ndarray, leads: np.ndarray, region: np.ndarray, tolerance: float = 1e-12
) -> float:
    """
    Average regional membership over all tied maximal leads.

    Parameters
    ----------
    scores : np.ndarray
        Flattened unit scores.
    leads : np.ndarray
        Corresponding canonical lead indices.
    region : np.ndarray
        Lead indices in the region.
    tolerance : float
        Absolute tolerance for a maximal score tie.

    Returns
    -------
    float
        Fraction of unique tied top leads that are in the region.
    """
    selected = np.unique(leads[np.abs(scores - scores.max()) <= tolerance])
    return float(np.isin(selected, region).mean())


def tied_premature_hit(
    scores: np.ndarray, starts: np.ndarray, ends: np.ndarray, windows: list[tuple[float, float]]
) -> tuple[float, float]:
    """
    Average premature-window overlap over tied maximal units.

    Parameters
    ----------
    scores, starts, ends : np.ndarray
        Aligned scores and unit time support.
    windows : list[tuple[float, float]]
        Automatic premature-beat proxy windows.

    Returns
    -------
    tuple[float, float]
        Tied top-unit overlap share and all-unit chance share.
    """
    overlap = np.zeros(len(scores), dtype=bool)
    for low, high in windows:
        overlap |= (starts < high) & (ends > low)
    selected = np.abs(scores - scores.max()) <= 1e-12
    return float(overlap[selected].mean()), float(overlap.mean())
