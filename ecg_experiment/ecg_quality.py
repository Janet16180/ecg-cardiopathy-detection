"""Record-level waveform quality policy for canonical ten-second, twelve-lead ECGs.

The rules come from the dataset EDA (``notebooks/0*-jr-*.ipynb``), measured on
every raw record of PTB-XL, MIMIC-IV-ECG, Georgia, CPSC 2018, CPSC-Extra and
Chapman/Shaoxing. Exclusion rules describe recordings in which part of the
signal is missing or is not an ECG; each affects well under 1% of any source.
Review flags describe unusual but possibly genuine recordings and never exclude.

The policy only uses the waveform, never a diagnosis, so it can be applied to
training candidates without looking at outcomes. It is meant for training
cohorts; held-out evaluation partitions should keep their difficult cases.
"""

from __future__ import annotations

import numpy as np
from scipy.signal import welch

from .waveforms import LEADS, SAMPLE_RATE

SIGNAL_SHAPE = (len(LEADS), 5000)
RAIL_MV = 32.6
MAX_AMPLITUDE_MV = 20.0
REVIEW_AMPLITUDE_MV = 10.0
FLAT_SECONDS = 1.0
NEAR_FLAT_STD_MV = 0.02
NOISE_BAND_HZ = (40.0, 150.0)
NOISE_FRACTION = 0.5
BASELINE_BAND_HZ = (0.0, 0.7)
BASELINE_FRACTION = 0.8
LIMB_RESIDUAL_MV = 0.05

EXCLUSION_REASONS = (
    "nonfinite", "constant_lead", "flat_segment", "adc_rail", "extreme_amplitude",
    "near_flat_record", "noise_dominated",
)
REVIEW_FLAGS = ("amplitude_over_10mv", "limb_identity_violated", "strong_baseline_wander")


def longest_constant_runs(signal: np.ndarray) -> np.ndarray:
    """
    Length of the longest run of identical consecutive samples in each lead.

    Parameters
    ----------
    signal : np.ndarray
        Array of shape ``(leads, samples)``.

    Returns
    -------
    np.ndarray
        One run length in samples per lead.
    """
    runs = []
    for lead in signal:
        changes = np.flatnonzero(np.diff(lead) != 0)
        boundaries = np.concatenate(([-1], changes, [len(lead) - 1]))
        runs.append(int(np.diff(boundaries).max()))
    return np.array(runs)


def band_fractions(signal: np.ndarray, fs: int, band: tuple[float, float]) -> np.ndarray:
    """
    Fraction of each lead's power inside a frequency band.

    Parameters
    ----------
    signal : np.ndarray
        Array of shape ``(leads, samples)``.
    fs : int
        Sampling rate in Hz.
    band : tuple[float, float]
        Lower (inclusive) and upper (exclusive) edge in Hz.

    Returns
    -------
    np.ndarray
        One fraction per lead; NaN for a lead without power.
    """
    frequencies, power = welch(signal, fs=fs, nperseg=min(2048, signal.shape[1]), axis=1)
    inside = (frequencies >= band[0]) & (frequencies < band[1])
    with np.errstate(invalid="ignore", divide="ignore"):
        return power[:, inside].sum(axis=1) / power.sum(axis=1)


def limb_residual(signal: np.ndarray) -> float:
    """
    Largest violation of Einthoven's law, ``II = I + III``, in mV.

    Parameters
    ----------
    signal : np.ndarray
        Canonical array of shape ``(12, samples)`` in ``LEADS`` order.

    Returns
    -------
    float
        Maximum absolute residual over the recording.
    """
    lead_i, lead_ii, lead_iii = signal[0], signal[1], signal[2]
    return float(np.abs(lead_ii - lead_i - lead_iii).max())


def assess(signal: np.ndarray, fs: int = SAMPLE_RATE) -> tuple[list[str], list[str]]:
    """
    Apply the quality policy to one canonical ECG.

    Parameters
    ----------
    signal : np.ndarray
        Float array of shape ``(12, 5000)`` in ``LEADS`` order and physical mV.
    fs : int
        Sampling rate in Hz.

    Returns
    -------
    tuple[list[str], list[str]]
        Exclusion reasons (empty when the record passes) and review flags, in
        the order of ``EXCLUSION_REASONS`` and ``REVIEW_FLAGS``.

    Raises
    ------
    ValueError
        If the array does not have the canonical shape.
    """
    if signal.shape != SIGNAL_SHAPE:
        raise ValueError(f"Expected {SIGNAL_SHAPE}, got {signal.shape}")
    if not np.isfinite(signal).all():
        return ["nonfinite"], []

    signal = signal.astype(np.float64)
    peak = np.abs(signal).max()
    checks = {
        "constant_lead": bool(np.any(np.ptp(signal, axis=1) == 0)),
        "flat_segment": bool(np.any(longest_constant_runs(signal) >= FLAT_SECONDS * fs)),
        "adc_rail": bool(peak >= RAIL_MV),
        "extreme_amplitude": bool(MAX_AMPLITUDE_MV < peak < RAIL_MV),
        "near_flat_record": bool(np.median(signal.std(axis=1)) < NEAR_FLAT_STD_MV),
        "noise_dominated": bool(np.nanmedian(band_fractions(signal, fs, NOISE_BAND_HZ)) > NOISE_FRACTION),
    }
    flags = {
        "amplitude_over_10mv": bool(peak > REVIEW_AMPLITUDE_MV),
        "limb_identity_violated": limb_residual(signal) > LIMB_RESIDUAL_MV,
        "strong_baseline_wander": bool(
            np.nanmedian(band_fractions(signal, fs, BASELINE_BAND_HZ)) > BASELINE_FRACTION
        ),
    }
    reasons = [name for name, failed in checks.items() if failed]
    review = [name for name, raised in flags.items() if raised]
    return reasons, review
