"""Versioned whole-record resampling for new manifests.

Frozen experiments keep ``cpc_input_audit.historical_resample`` and ``code15_cpc.to_cpc``,
which resample each 5 s half separately.
"""

from __future__ import annotations

import numpy as np
from scipy.signal import resample_poly


def resample_full(signal: np.ndarray, up: int, down: int) -> np.ndarray:
    """
    Resample a whole record in one polyphase pass along its sample axis.

    Resample the full source record before cropping or splitting it, so that no
    filter edge falls inside the kept samples.

    Parameters
    ----------
    signal : np.ndarray
        Finite lead-major waveform of shape ``(leads, samples)``.
    up : int
        Upsampling factor.
    down : int
        Downsampling factor.

    Returns
    -------
    np.ndarray
        Float32 waveform of shape ``(leads, ceil(samples * up / down))``.

    Raises
    ------
    ValueError
        If the input is not a finite two-dimensional array or a factor is not positive.
    """
    if signal.ndim != 2 or not np.isfinite(signal).all():
        raise ValueError("Expected a finite (leads, samples) waveform")
    if up < 1 or down < 1:
        raise ValueError("Resampling factors must be positive")
    return resample_poly(signal, up, down, axis=1).astype(np.float32, copy=False)
