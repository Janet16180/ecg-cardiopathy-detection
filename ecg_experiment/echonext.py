"""EchoNext waveform quality, labels and CPC input scaling.

EchoNext waveforms were median filtered, clipped and standardized by the dataset authors, so they have no
physical unit. Only the rules of the project quality policy that do not depend on the unit apply: nonfinite
samples, constant leads, flat segments and noise-dominated recordings. The flat-segment rule also catches
the 629 records whose leads hold only 2.5-5 s of signal and a constant filler elsewhere, found in the EDA
(``notebooks/08-jr-echonext.ipynb``).
"""

from __future__ import annotations

import numpy as np

from .ecg_quality import FLAT_SECONDS, NOISE_BAND_HZ, NOISE_FRACTION, band_fractions, longest_constant_runs

SAMPLING_RATE = 250
EXCLUSION_REASONS = ("nonfinite", "constant_lead", "flat_segment", "noise_dominated")


def assess(signal: np.ndarray) -> list[str]:
    """
    Apply the unit-free rules of the quality policy to one waveform.

    Parameters
    ----------
    signal : np.ndarray
        Array of shape ``(12, 2500)`` at 250 Hz.

    Returns
    -------
    list[str]
        Exclusion reasons in the order of ``EXCLUSION_REASONS``; empty when the waveform passes.

    Raises
    ------
    ValueError
        If the array does not have shape ``(12, 2500)``.
    """
    if signal.shape != (12, 2500):
        raise ValueError(f"Expected (12, 2500), got {signal.shape}")
    if not np.isfinite(signal).all():
        return ["nonfinite"]
    signal = signal.astype(np.float64)
    noise = np.nanmedian(band_fractions(signal, SAMPLING_RATE, NOISE_BAND_HZ))
    checks = {
        "constant_lead": bool(np.any(np.ptp(signal, axis=1) == 0)),
        "flat_segment": bool(np.any(longest_constant_runs(signal) >= FLAT_SECONDS * SAMPLING_RATE)),
        "noise_dominated": bool(noise > NOISE_FRACTION),
    }
    return [name for name, failed in checks.items() if failed]


def to_cpc_scale(signals: np.ndarray, lead_mean: np.ndarray, lead_std: np.ndarray,
                 target_mean: np.ndarray, target_std: np.ndarray) -> np.ndarray:
    """
    Map EchoNext waveforms onto the per-lead scale of the historical CPC inputs.

    Each lead is standardized with EchoNext training statistics and rescaled to the historical PTB-XL mean
    and standard deviation, so that the historical CPC normalization gives per-lead z-scores.

    Parameters
    ----------
    signals : np.ndarray
        Array of shape ``(records, 12, samples)``.
    lead_mean, lead_std : np.ndarray
        EchoNext training mean and standard deviation per lead, shape ``(12,)``.
    target_mean, target_std : np.ndarray
        Historical CPC normalization per lead, shape ``(12,)``.

    Returns
    -------
    np.ndarray
        Float32 array with the shape of ``signals``.
    """
    scale = (target_std / lead_std)[:, None]
    shift = (target_mean - lead_mean * target_std / lead_std)[:, None]
    return (signals * scale + shift).astype(np.float32)
