"""Preserve fixed P including recording-edge spans in Experiment 059."""

import numpy as np

from ecg_experiment.morphology_boundaries import adaptive_boundaries, fixed_boundaries


def joint_hybrid(signal: np.ndarray, fs: int = 250) -> tuple[np.ndarray, dict]:
    """
    Combine adaptive QRS/T support while copying every fixed P span verbatim.

    Parameters
    ----------
    signal : np.ndarray
        Two-channel physical waveform, channels by samples.
    fs : int
        Native sampling frequency.

    Returns
    -------
    tuple[np.ndarray, dict]
        Common raw anchors and fixed/hybrid joint support, including edge P windows.
    """
    peaks, adaptive = adaptive_boundaries(signal, fs)
    fixed = fixed_boundaries(peaks, leads=signal.shape[0], fs=fs)[:, 0]
    valid = adaptive[:, :, :, 0] >= 0
    onset = np.where(valid, adaptive[:, :, :, 0], np.iinfo(np.int64).max).min(axis=1)
    offset = np.where(valid, adaptive[:, :, :, 1], -1).max(axis=1)
    joint = np.stack([onset, offset], axis=-1)
    joint[~valid.any(axis=1)] = -1
    joint[:, 0] = fixed[:, 0]
    if not np.array_equal(joint[:, 0], fixed[:, 0]):
        raise ValueError("Hybrid P differs from fixed P")
    return peaks, {"fixed": fixed, "hybrid": joint}
