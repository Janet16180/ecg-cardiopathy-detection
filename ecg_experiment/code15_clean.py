"""Clean CODE-15% exams for the quality-first cohorts: trim padding, keep 10 s, resample, score, deduplicate.

Every tracing is stored as 4,096 samples at 400 Hz with zero padding at both ends. The contiguous leading and
trailing rows where all twelve leads are exactly zero are trimmed, and only exams with at least 10 s (4,000
samples) of observed signal are kept. The whole observed span is resampled to 500 Hz with a polyphase filter,
and the first 10 s form the canonical ``(12, 5000)`` window.

The amplitude unit is not documented consistently (``docs/clean-code15-v1.md``), and stored values are about
1.6 to 2 times the millivolt amplitudes of the other sources. The quality policy is therefore applied at both
plausible scales, the stored values read as mV and halved, and an exam is usable only if it passes at both.
Manifests keep the stored units.
"""

from __future__ import annotations

import h5py
import numpy as np
import pandas as pd
from scipy.signal import resample_poly

from .ecg_quality import EXCLUSION_REASONS, REVIEW_FLAGS, assess
from .eda.code15 import highpass_range, zero_padding
from .eda.signals import LEADS
from .public_sources import signal_sha256

SAMPLING_RATE = 400
TEN_SECONDS = 4000
WINDOW = 5000
SCALES = {"stored": 1.0, "halved": 0.5}


def window_500hz(active: np.ndarray) -> np.ndarray:
    """
    Resample an observed span to 500 Hz and take its first 10 s.

    Parameters
    ----------
    active : np.ndarray
        Trimmed tracing of shape ``(samples, 12)`` at 400 Hz with at least 4,000 samples.

    Returns
    -------
    np.ndarray
        Float32 array of shape ``(12, 5000)`` in stored units.
    """
    resampled = resample_poly(active.astype(np.float64), 5, 4, axis=0)
    return np.ascontiguousarray(resampled[:WINDOW].T, dtype=np.float32)


def score_tracing(tracing: np.ndarray) -> dict[str, object]:
    """
    Trim, hash and score one stored tracing.

    Parameters
    ----------
    tracing : np.ndarray
        Stored array of shape ``(4096, 12)``.

    Returns
    -------
    dict[str, object]
        ``native_sha256`` of the stored tracing, ``edge_zero_left``, ``edge_zero_right`` and
        ``active_samples``. For exams with 10 s of signal also ``window_sha256`` of the 500 Hz window in
        stored units, semicolon-separated ``exclusion_<scale>`` and ``review_<scale>`` for each scale, and
        the high-passed range per lead of the first 10 s at 400 Hz (``range_<lead>``).
    """
    left, right = zero_padding(tracing)
    active = tracing[left:len(tracing) - right]
    row: dict[str, object] = {"native_sha256": signal_sha256(tracing), "edge_zero_left": left,
                              "edge_zero_right": right, "active_samples": len(active)}
    if len(active) < TEN_SECONDS:
        return row
    window = window_500hz(active)
    row["window_sha256"] = signal_sha256(window)
    for name, scale in SCALES.items():
        reasons, flags = assess(window * np.float32(scale))
        row[f"exclusion_{name}"] = ";".join(reasons)
        row[f"review_{name}"] = ";".join(flags)
    ranges = highpass_range(active[:TEN_SECONDS], SAMPLING_RATE)
    row |= {f"range_{lead}": float(value) for lead, value in zip(LEADS, ranges, strict=True)}
    return row


def score_chunk(item: tuple[str, int, int, int]) -> list[dict[str, object]]:
    """
    Score a contiguous block of tracings of one extracted part.

    Parameters
    ----------
    item : tuple[str, int, int, int]
        HDF5 path, part number, first and last-plus-one storage index.

    Returns
    -------
    list[dict[str, object]]
        ``score_tracing`` rows with ``exam_id``, ``part`` and ``storage_index``.
    """
    path, part, start, stop = item
    with h5py.File(path, "r") as handle:
        ids = handle["exam_id"][start:stop]
        tracings = handle["tracings"][start:stop]
    return [{"exam_id": int(exam_id), "part": part, "storage_index": start + offset, **score_tracing(tracing)}
            for offset, (exam_id, tracing) in enumerate(zip(ids, tracings, strict=True))]


def exclusion_reasons(rows: pd.DataFrame) -> pd.Series:
    """
    Union of the exclusion reasons at both scales, in policy order.

    Parameters
    ----------
    rows : pd.DataFrame
        Scored rows with ``exclusion_stored`` and ``exclusion_halved`` (empty text when passing, NaN when not
        scored).

    Returns
    -------
    pd.Series
        Semicolon-separated reasons, empty when the exam passes at both scales.
    """
    stored = rows["exclusion_stored"].fillna("").str.split(";")
    halved = rows["exclusion_halved"].fillna("").str.split(";")
    return pd.Series([";".join(name for name in EXCLUSION_REASONS if name in set(a) | set(b))
                      for a, b in zip(stored, halved, strict=True)], index=rows.index)


def review_flags(rows: pd.DataFrame) -> pd.Series:
    """
    Union of the review flags at both scales, in policy order.

    Parameters
    ----------
    rows : pd.DataFrame
        Scored rows with ``review_stored`` and ``review_halved``.

    Returns
    -------
    pd.Series
        Semicolon-separated flags.
    """
    stored = rows["review_stored"].fillna("").str.split(";")
    halved = rows["review_halved"].fillna("").str.split(";")
    return pd.Series([";".join(name for name in REVIEW_FLAGS if name in set(a) | set(b))
                      for a, b in zip(stored, halved, strict=True)], index=rows.index)


def duplicate_status(rows: pd.DataFrame) -> pd.Series:
    """
    Resolve exams whose stored tracings are bit-identical, grouped by patient.

    Within an identical group, the lowest exam ID is kept when every copy belongs to one patient; copies
    from different patients cannot all be right about who was recorded, so every copy is dropped.

    Parameters
    ----------
    rows : pd.DataFrame
        One row per exam, indexed by ``exam_id``, with ``native_sha256`` and ``patient_id``.

    Returns
    -------
    pd.Series
        ``unique``, ``kept``, ``dropped_copy`` or ``dropped_conflict`` per exam.
    """
    ordered = rows.sort_index()
    grouped = ordered.groupby("native_sha256")
    size = grouped["patient_id"].transform("size")
    conflict = grouped["patient_id"].transform("nunique") > 1
    first = ~ordered["native_sha256"].duplicated(keep="first")
    status = pd.Series("unique", index=ordered.index)
    status[(size > 1) & first & ~conflict] = "kept"
    status[(size > 1) & ~first & ~conflict] = "dropped_copy"
    status[(size > 1) & conflict] = "dropped_conflict"
    return status.reindex(rows.index)
