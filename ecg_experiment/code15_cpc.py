"""Convert CODE-15% tracings to the 250 Hz CPC input and read them back for supervised training."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.signal import resample_poly
from torch.utils.data import Dataset

from .ecg_quality import longest_constant_runs

NATIVE_RATE = 400
WINDOW = 4000
HALF = WINDOW // 2
OUTPUT_SAMPLES = 2500
FLAT_SAMPLES = NATIVE_RATE
MAX_AMPLITUDE_MV = 20.0
LABELS = ("1dAVb", "RBBB", "LBBB", "SB", "ST", "AF", "normal_ecg")


def active_bounds(tracing: np.ndarray) -> tuple[int, int]:
    """
    First and one-past-last sample that is not all-lead zero padding.

    Parameters
    ----------
    tracing : np.ndarray
        Array of shape ``(samples, 12)``.

    Returns
    -------
    tuple[int, int]
        ``(start, stop)``; ``(0, 0)`` for an all-zero tracing.
    """
    active = np.flatnonzero(np.any(tracing != 0, axis=1))
    if len(active) == 0:
        return 0, 0
    return int(active[0]), int(active[-1]) + 1


def central_window(tracing: np.ndarray) -> np.ndarray | None:
    """
    Cut the central 10 s of the active part, as ``(12, 4000)``.

    Parameters
    ----------
    tracing : np.ndarray
        Native 400 Hz array of shape ``(samples, 12)``.

    Returns
    -------
    np.ndarray | None
        Lead-major window, or None when fewer than 4,000 samples are active.
    """
    start, stop = active_bounds(tracing)
    if stop - start < WINDOW:
        return None
    offset = start + (stop - start - WINDOW) // 2
    return np.ascontiguousarray(tracing[offset:offset + WINDOW].T, dtype=np.float32)


def to_cpc(window: np.ndarray) -> np.ndarray:
    """
    Resample each five-second half from 400 to 250 Hz.

    Parameters
    ----------
    window : np.ndarray
        Array of shape ``(12, 4000)``.

    Returns
    -------
    np.ndarray
        Float32 array of shape ``(12, 2500)`` in native CODE-15 units.
    """
    halves = [resample_poly(window[:, start:start + HALF], 5, 8, axis=1) for start in (0, HALF)]
    result = np.concatenate(halves, axis=1).astype(np.float32)
    if result.shape != (12, OUTPUT_SAMPLES):
        raise ValueError(f"Unexpected CPC shape {result.shape}")
    return result


def window_quality(window: np.ndarray) -> dict[str, float | bool]:
    """
    Quality measurements of one native window.

    Parameters
    ----------
    window : np.ndarray
        Array of shape ``(12, 4000)``.

    Returns
    -------
    dict[str, float | bool]
        ``nonfinite``, ``constant_lead``, ``flat_segment`` and the native
        ``peak`` absolute value.
    """
    finite = bool(np.isfinite(window).all())
    safe = np.nan_to_num(window)
    return {
        "nonfinite": not finite,
        "constant_lead": bool(np.any(np.ptp(safe, axis=1) == 0)),
        "flat_segment": bool(np.any(longest_constant_runs(safe) >= FLAT_SAMPLES)),
        "peak": float(np.abs(safe).max()),
    }


def patient_split(patients: pd.Series, monitoring_fraction: float, seed: int) -> pd.Series:
    """
    Assign whole patients to ``train`` or ``monitoring``.

    Parameters
    ----------
    patients : pd.Series
        Patient ID of each row.
    monitoring_fraction : float
        Share of patients held out for monitoring.
    seed : int
        Permutation seed.

    Returns
    -------
    pd.Series
        Split name of each row, aligned with ``patients``.
    """
    unique = np.sort(patients.unique())
    order = np.random.default_rng(seed).permutation(len(unique))
    monitoring = set(unique[order[:round(len(unique) * monitoring_fraction)]])
    return patients.map(lambda patient: "monitoring" if patient in monitoring else "train")


def lead_std(signals: np.ndarray, chunk: int = 2000) -> np.ndarray:
    """
    Compute the standard deviation of every lead of every record, in chunks.

    Parameters
    ----------
    signals : np.ndarray
        Array (or memory map) of shape ``(records, 12, samples)``.
    chunk : int
        Records read at a time.

    Returns
    -------
    np.ndarray
        Array of shape ``(records, 12)``.
    """
    return np.concatenate([np.asarray(signals[start:start + chunk], dtype=np.float64).std(axis=2)
                           for start in range(0, len(signals), chunk)])


def amplitude_factor(ptb_std: np.ndarray, code_std: np.ndarray) -> float:
    """
    Scale that matches CODE-15 lead amplitudes to PTB-XL.

    Parameters
    ----------
    ptb_std, code_std : np.ndarray
        Per-record lead standard deviations of shape ``(records, 12)``.

    Returns
    -------
    float
        Median over leads of the ratio of median lead standard deviations.
    """
    return float(np.median(np.median(ptb_std, axis=0) / np.median(code_std, axis=0)))


class Code15Dataset(Dataset):
    """
    Scaled, normalized CODE-15 CPC inputs with their seven labels.

    Parameters
    ----------
    directory : Path
        Cache written by ``scripts.data.build_code15_cpc_cache``.
    rows : pd.DataFrame
        Rows of ``rows.csv`` to serve.
    factor : float
        Amplitude factor applied before normalization.
    normalization : Path
        Historical CPC ``normalization.json`` with per-lead ``mean`` and ``std``.
    """

    def __init__(self, directory: Path, rows: pd.DataFrame, factor: float, normalization: Path) -> None:
        self.directory, self.rows, self.factor = directory, rows.reset_index(drop=True), factor
        statistics = json.loads(normalization.read_text())
        self.mean = np.asarray(statistics["mean"], dtype=np.float32)[:, None]
        self.std = np.asarray(statistics["std"], dtype=np.float32)[:, None]
        self.shards: dict[str, np.ndarray] = {}

    def __len__(self) -> int:
        """Return the number of rows."""
        return len(self.rows)

    def __getstate__(self) -> dict[str, object]:
        """Drop open memory maps when copied into a worker."""
        return {**self.__dict__, "shards": {}}

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Return the normalized signal and the float label vector."""
        row = self.rows.iloc[index]
        if row["shard"] not in self.shards:
            self.shards[row["shard"]] = np.load(self.directory / row["shard"], mmap_mode="r")
        signal = np.array(self.shards[row["shard"]][int(row["index"])], dtype=np.float32) * self.factor
        signal = (signal - self.mean) / self.std
        labels = row[list(LABELS)].to_numpy(dtype=np.float32)
        return torch.from_numpy(signal), torch.from_numpy(labels)
