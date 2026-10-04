"""Unsupervised, equal-support within-record raw morphology localization for Experiment 052."""

from __future__ import annotations

import numpy as np
from matplotlib.figure import Figure

from .lead_wave_maps import UnitMap
from .waveforms import LEADS

FS = 500
WIDTH = 70
STRIDE = 5
BEFORE = 125
AFTER = 225
SCALE_FLOOR = 0.02


def complete_beats(signal: np.ndarray, peaks: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Extract baseline-subtracted complete beats.

    Parameters
    ----------
    signal : np.ndarray
        Twelve leads by samples, in mV at 500 Hz.
    peaks : np.ndarray
        R-peak sample positions, used only for alignment.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        Kept peak positions and beat-by-lead-by-sample segments.
    """
    kept = peaks[(peaks >= BEFORE) & (peaks + AFTER <= signal.shape[1])]
    if len(kept) < 3:
        raise ValueError("Fewer than three complete beats")
    beats = np.stack([signal[:, peak - BEFORE : peak + AFTER] for peak in kept])
    beats -= np.median(beats[:, :, BEFORE - 50 : BEFORE - 30], axis=2, keepdims=True)
    return kept, beats


def residual_field(signal: np.ndarray, peaks: np.ndarray) -> np.ndarray:
    """Construct squared leave-one-out robust residuals with midpoint beat ownership.

    Parameters
    ----------
    signal : np.ndarray
        Twelve leads by samples, in mV.
    peaks : np.ndarray
        Existing R-peak sample positions; no RR anomaly feature is computed.

    Returns
    -------
    np.ndarray
        Lead-by-sample residual energy; zero outside complete beat support.
    """
    kept, beats = complete_beats(signal, peaks)
    field = np.zeros_like(signal, dtype=np.float64)
    for index, peak in enumerate(kept):
        others = np.delete(beats, index, axis=0)
        template = np.median(others, axis=0)
        scale = np.maximum(1.4826 * np.median(np.abs(others - template), axis=0), SCALE_FLOOR)
        energy = ((beats[index] - template) / scale) ** 2
        left = max(peak - BEFORE, 0 if index == 0 else (kept[index - 1] + peak) // 2)
        right = min(
            peak + AFTER, signal.shape[1] if index == len(kept) - 1 else (peak + kept[index + 1]) // 2
        )
        field[:, left:right] = energy[:, left - peak + BEFORE : right - peak + BEFORE]
    return field


def support_mask(samples: int, peaks: np.ndarray) -> np.ndarray:
    """Return the exact complete-beat sample support eligible for both maps.

    Parameters
    ----------
    samples : int
        Recording length.
    peaks : np.ndarray
        R sample positions.

    Returns
    -------
    np.ndarray
        Boolean sample coverage, including midpoint-owned beats only.
    """
    kept = peaks[(peaks >= BEFORE) & (peaks + AFTER <= samples)]
    mask = np.zeros(samples, dtype=bool)
    for index, peak in enumerate(kept):
        left = max(peak - BEFORE, 0 if index == 0 else (kept[index - 1] + peak) // 2)
        right = min(peak + AFTER, samples if index == len(kept) - 1 else (peak + kept[index + 1]) // 2)
        mask[left:right] = True
    return mask


def fixed_window_map(field: np.ndarray, mask: np.ndarray | None = None) -> UnitMap:
    """Average a dense field in identical 140 ms windows at 10 ms stride.

    Parameters
    ----------
    field : np.ndarray
        Lead-by-sample scores at 500 Hz.
    mask : np.ndarray or None
        Shared eligible sample support; every sample of a candidate must be covered.

    Returns
    -------
    UnitMap
        Candidate windows in lead-major order.
    """
    starts = np.arange(0, field.shape[1] - WIDTH + 1, STRIDE)
    if mask is not None:
        count = np.pad(np.cumsum(mask), (1, 0))
        starts = starts[count[starts + WIDTH] - count[starts] == WIDTH]
    integral = np.pad(np.cumsum(field, axis=1), ((0, 0), (1, 0)))
    scores = (integral[:, starts + WIDTH] - integral[:, starts]) / WIDTH
    return UnitMap(
        scores.ravel(),
        np.repeat(np.arange(field.shape[0]), len(starts)),
        np.tile(starts / FS, field.shape[0]),
        np.tile((starts + WIDTH) / FS, field.shape[0]),
    )


def baseline_field(unit_map: UnitMap, samples: int) -> np.ndarray:
    """Broadcast saved U_B scores onto their exact time and lead supports.

    Parameters
    ----------
    unit_map : UnitMap
        Historical U_B pieces.
    samples : int
        Original recording length at 500 Hz.

    Returns
    -------
    np.ndarray
        Dense field, taking the maximum on overlapping pieces.
    """
    field = np.zeros((len(LEADS), samples), dtype=np.float64)
    for score, lead, start, end in zip(
        unit_map.scores, unit_map.leads, unit_map.starts, unit_map.ends, strict=True
    ):
        left, right = max(0, round(start * FS)), min(samples, round(end * FS))
        field[int(lead), left:right] = np.maximum(field[int(lead), left:right], score)
    return field


def baseline_support(unit_map: UnitMap, samples: int) -> np.ndarray:
    """Return the actual historical piece coverage independently of its scores.

    Parameters
    ----------
    unit_map : UnitMap
        Saved U_B pieces, with stricter historical edge eligibility.
    samples : int
        Original recording length.

    Returns
    -------
    np.ndarray
        Sample coverage shared by all leads of the saved complete beats.
    """
    mask = np.zeros(samples, dtype=bool)
    for start, end in zip(unit_map.starts, unit_map.ends, strict=True):
        mask[max(0, round(start * FS)) : min(samples, round(end * FS))] = True
    return mask


def tied_top(unit_map: UnitMap) -> np.ndarray:
    """Return all tied maximum indices with the frozen numerical tolerance.

    Parameters
    ----------
    unit_map : UnitMap
        Candidate scores.

    Returns
    -------
    np.ndarray
        Indices within rtol and atol 1e-10 of the maximum score.
    """
    return np.flatnonzero(np.isclose(unit_map.scores, unit_map.scores.max(), rtol=1e-10, atol=1e-10))


def display_top(unit_map: UnitMap) -> int:
    """Choose the median-time tied maximum for a deterministic descriptive display.

    Parameters
    ----------
    unit_map : UnitMap
        Candidate map.

    Returns
    -------
    int
        Unit index; tied lead ordering is stable.
    """
    indices = tied_top(unit_map)
    indices = indices[np.argsort(unit_map.starts[indices], kind="stable")]
    return int(indices[len(indices) // 2])


def location_metrics(
    unit_map: UnitMap, supports: list[tuple[float, float]], leads: list[int] | None = None
) -> dict[str, float | int]:
    """Measure exact-center localization, uniform-window chance and overlap.

    Parameters
    ----------
    unit_map : UnitMap
        Fixed-duration candidates.
    supports : list[tuple[float, float]]
        Exact target time supports in seconds.
    leads : list[int] or None
        Exact target leads, if known; PVC targets have no lead truth.

    Returns
    -------
    dict
        Top location, temporal/joint hit, uniform chance, distance and overlap fraction.
    """
    tops = tied_top(unit_map)
    top = display_top(unit_map)
    centers = (unit_map.starts + unit_map.ends) / 2
    inside = np.zeros(len(centers), dtype=bool)
    for left, right in supports:
        inside |= (centers >= left) & (centers < right)
    correct_lead = np.ones(len(tops), dtype=bool) if leads is None else np.isin(unit_map.leads[tops], leads)
    overlap = np.zeros(len(tops))
    distances = []
    for left, right in supports:
        overlap += np.maximum(
            0.0, np.minimum(unit_map.ends[tops], right) - np.maximum(unit_map.starts[tops], left)
        )
        distances.append(np.abs(centers[tops] - (left + right) / 2))
    distance = np.min(distances, axis=0).mean()
    return {
        "score": float(unit_map.scores[top]),
        "lead": int(unit_map.leads[top]),
        "start": float(unit_map.starts[top]),
        "end": float(unit_map.ends[top]),
        "center": float(centers[top]),
        "hit": float(inside[tops].mean()),
        "joint_hit": float((inside[tops] & correct_lead).mean()),
        "chance": float(inside.mean()),
        "distance_seconds": float(distance),
        "overlap_fraction": float(overlap.mean() / (WIDTH / FS)),
        "tie_count": len(tops),
    }


def perturb(
    signal: np.ndarray, peaks: np.ndarray, peak: int, leads: list[int], kind: str
) -> tuple[np.ndarray, list[tuple[float, float]]]:
    """Apply a prespecified ST bump or stretched median QRS with exact support.

    Parameters
    ----------
    signal : np.ndarray
        Original twelve-lead ECG in mV.
    peaks : np.ndarray
        Fixed original alignment peaks.
    peak : int
        Chosen complete target beat.
    leads : list[int]
        Leads edited, one for ST and three for QRS.
    kind : str
        ``ST`` or ``QRS``.

    Returns
    -------
    tuple
        Edited copy and exact support in seconds.
    """
    edited = signal.copy()
    if kind == "ST":
        left, right = peak + 50, peak + 100
        shape = 0.15 * np.sin(np.linspace(0, np.pi, right - left)) ** 2
        edited[np.asarray(leads), left:right] += shape
    elif kind == "QRS":
        kept, beats = complete_beats(signal, peaks)
        template = np.median(beats[kept != peak], axis=0)
        left, right = peak - 48, peak + 64
        position = np.arange(-48, 64) / 1.6 + BEFORE
        for lead in leads:
            stretched = np.interp(position, np.arange(template.shape[1]), template[lead])
            baseline = np.median(signal[lead, peak - 50 : peak - 30])
            taper = np.ones(right - left)
            taper[:10] = np.linspace(0, 1, 10)
            taper[-10:] = np.linspace(1, 0, 10)
            edited[lead, left:right] += taper * (stretched + baseline - signal[lead, left:right])
    else:
        raise ValueError(f"Unknown perturbation {kind}")
    return edited, [(left / FS, right / FS)]


def review_figure(
    signal: np.ndarray,
    maps: dict[str, UnitMap],
    title: str,
    supports: list[tuple[float, float]] | None = None,
) -> Figure:
    """Draw raw ECGs with equal-duration map marks for local review.

    Parameters
    ----------
    signal : np.ndarray
        Twelve-lead ECG in mV at 500 Hz.
    maps : dict
        Named fixed-duration maps.
    title : str
        Honest record/experiment title.
    supports : list or None
        Known synthetic or automatic target supports.

    Returns
    -------
    Figure
        Local-only comparison figure.
    """
    figure = Figure(figsize=(8 * len(maps), 12), layout="constrained")
    axes = figure.subplots(len(LEADS), len(maps), sharex=True, squeeze=False)
    seconds = np.arange(signal.shape[1]) / FS
    for column, (name, unit_map) in enumerate(maps.items()):
        top = display_top(unit_map)
        axes[0, column].set_title(f"{name}: {unit_map.starts[top]:.2f}–{unit_map.ends[top]:.2f} s")
        for row, lead in enumerate(LEADS):
            axis = axes[row, column]
            axis.plot(seconds, signal[row], color="black", linewidth=0.6)
            axis.set_ylabel(lead, rotation=0)
            axis.set_yticks([])
            for low, high in supports or []:
                axis.axvspan(low, high, color="green", alpha=0.15)
            if row == unit_map.leads[top]:
                axis.axvspan(unit_map.starts[top], unit_map.ends[top], color="red", alpha=0.35)
        axes[-1, column].set_xlabel("seconds")
    figure.suptitle(title)
    return figure
