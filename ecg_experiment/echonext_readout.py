"""Tabular inputs, 250 Hz encoder inputs and component label masks of the Experiment 023 EchoNext readout.

EchoNext is credentialed: these functions return arrays for local use, and callers write only aggregates
to documents.
"""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd
from scipy.signal import resample, resample_poly

from .eda.echonext import COMPONENTS, MEASUREMENTS
from .external_encoders import JEPA_LEADS, JEPA_SAMPLES
from .xecg import MODEL_SAMPLES

MISSING_INDICATORS = ("atrial_rate", "pr_interval")
SEXES = ("female", "male")


def male(frame: pd.DataFrame) -> np.ndarray:
    """
    Male indicator from the EchoNext ``sex`` column.

    Parameters
    ----------
    frame : pd.DataFrame
        Rows with ``sex``.

    Returns
    -------
    np.ndarray
        Float64 array, 1 for male and 0 for female.

    Raises
    ------
    ValueError
        If a value is neither ``male`` nor ``female``.
    """
    if not frame["sex"].isin(SEXES).all():
        raise ValueError("Unexpected EchoNext sex value")
    return (frame["sex"] == "male").to_numpy(dtype=np.float64)


def age_sex_inputs(frame: pd.DataFrame) -> np.ndarray:
    """
    Age per decade and male.

    Parameters
    ----------
    frame : pd.DataFrame
        Rows with ``age_at_ecg`` and ``sex``.

    Returns
    -------
    np.ndarray
        Float64 array of shape ``(rows, 2)``.

    Raises
    ------
    ValueError
        If an age is missing.
    """
    if frame["age_at_ecg"].isna().any():
        raise ValueError("EchoNext age missing")
    return np.column_stack([frame["age_at_ecg"].to_numpy(dtype=np.float64) / 10, male(frame)])


def train_medians(train: pd.DataFrame) -> dict[str, float]:
    """
    Training median of each cart measurement, ignoring missing values.

    Parameters
    ----------
    train : pd.DataFrame
        Usable training rows.

    Returns
    -------
    dict[str, float]
        Median keyed by measurement name.
    """
    return {name: float(train[name].median()) for name in MEASUREMENTS}


def tabular_inputs(frame: pd.DataFrame, medians: dict[str, float]) -> np.ndarray:
    """
    Age per decade, male, the imputed cart measurements and missing indicators for atrial rate and PR.

    Parameters
    ----------
    frame : pd.DataFrame
        Rows with ``age_at_ecg``, ``sex`` and the measurements.
    medians : dict[str, float]
        Output of ``train_medians``.

    Returns
    -------
    np.ndarray
        Float64 array of shape ``(rows, 2 + len(MEASUREMENTS) + len(MISSING_INDICATORS))``.
    """
    measured = [frame[name].fillna(medians[name]).to_numpy(dtype=np.float64) for name in MEASUREMENTS]
    missing = [frame[name].isna().to_numpy(dtype=np.float64) for name in MISSING_INDICATORS]
    return np.column_stack([age_sex_inputs(frame), *measured, *missing])


def downsample_250(signal: np.ndarray) -> np.ndarray:
    """
    Convert the first 10 s of a 500 Hz twelve-lead waveform to 250 Hz with ``resample_poly``.

    Parameters
    ----------
    signal : np.ndarray
        Array of shape ``(12, samples)`` with at least 5,000 samples, in mV.

    Returns
    -------
    np.ndarray
        Float64 array of shape ``(12, 2500)``.

    Raises
    ------
    ValueError
        If the waveform is shorter than 10 s or nonfinite.
    """
    if signal.ndim != 2 or signal.shape[0] != 12 or signal.shape[1] < 5000:
        raise ValueError(f"Expected at least 10 s of twelve leads at 500 Hz, got {signal.shape}")
    result = resample_poly(np.asarray(signal[:, :5000], dtype=np.float64), 1, 2, axis=1)
    if not np.isfinite(result).all():
        raise ValueError("Nonfinite downsampled waveform")
    return result


def pooled_lead_statistics(signals: Iterable[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    """
    Per-lead mean and population standard deviation over all samples of all waveforms.

    Parameters
    ----------
    signals : Iterable[np.ndarray]
        Waveforms of shape ``(12, samples)``.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        Float64 mean and standard deviation, shape ``(12,)`` each.
    """
    total, squares, samples = np.zeros(12), np.zeros(12), 0
    for signal in signals:
        values = np.asarray(signal, dtype=np.float64)
        total += values.sum(axis=1)
        squares += (values ** 2).sum(axis=1)
        samples += values.shape[1]
    mean = total / samples
    return mean, np.sqrt(squares / samples - mean ** 2)


def jepa_input_250(signal: np.ndarray) -> np.ndarray:
    """
    ECG-JEPA input of a 250 Hz waveform: its eight leads and 2,500 samples, unchanged.

    Parameters
    ----------
    signal : np.ndarray
        Array of shape ``(12, 2500)`` in mV, canonical lead order.

    Returns
    -------
    np.ndarray
        Float32 ``[8, 2500]`` waveform with leads I, II, V1-V6.

    Raises
    ------
    ValueError
        If the input shape is wrong.
    """
    if signal.shape != (12, JEPA_SAMPLES):
        raise ValueError(f"Expected 12 leads x {JEPA_SAMPLES} samples, got {signal.shape}")
    return np.ascontiguousarray(signal[list(JEPA_LEADS)], dtype=np.float32)


def xecg_input_250(signal: np.ndarray) -> np.ndarray:
    """
    Convert a 250 Hz waveform to the xECG input by Fourier resampling to 100 Hz, time-major, from float64.

    This is the resampler of ``xecg.preprocess_xecg``, applied from 2,500 instead of 5,000 samples.

    Parameters
    ----------
    signal : np.ndarray
        Array of shape ``(12, 2500)`` in mV, canonical lead order.

    Returns
    -------
    np.ndarray
        Contiguous float32 array of shape ``[1000, 12]``.

    Raises
    ------
    ValueError
        If the input shape is wrong or the result is nonfinite.
    """
    if signal.shape != (12, 2500):
        raise ValueError(f"Expected 12 leads x 2,500 samples, got {signal.shape}")
    output = resample(np.asarray(signal, dtype=np.float64).T, MODEL_SAMPLES, axis=0).astype(np.float32)
    if not np.isfinite(output).all():
        raise ValueError("Invalid xECG resampling output")
    return np.ascontiguousarray(output)


def cosine_summary(first: np.ndarray, second: np.ndarray) -> dict[str, float]:
    """
    Mean and minimum row-wise cosine similarity of two feature matrices.

    Parameters
    ----------
    first, second : np.ndarray
        Arrays of the same shape ``(records, width)``.

    Returns
    -------
    dict[str, float]
        ``mean_cosine`` and ``min_cosine``.

    Raises
    ------
    ValueError
        If the shapes differ.
    """
    if first.shape != second.shape:
        raise ValueError(f"Feature shapes differ: {first.shape}, {second.shape}")
    first, second = np.asarray(first, dtype=np.float64), np.asarray(second, dtype=np.float64)
    cosines = (first * second).sum(axis=1) / (np.linalg.norm(first, axis=1) * np.linalg.norm(second, axis=1))
    return {"mean_cosine": float(cosines.mean()), "min_cosine": float(cosines.min())}


def measured(frame: pd.DataFrame, flag: str) -> np.ndarray:
    """
    Rows whose echo value underlying a component flag is present.

    Parameters
    ----------
    frame : pd.DataFrame
        Rows with the component value columns.
    flag : str
        Component flag, a key of ``COMPONENTS``.

    Returns
    -------
    np.ndarray
        Boolean mask.
    """
    return frame[COMPONENTS[flag]].notna().to_numpy()
