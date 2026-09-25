"""Focused scientific regression checks for the CPC input audit."""

import numpy as np
import pytest
from scipy.signal import resample_poly

from ecg_experiment.cpc_input_audit import evenly_spaced_indices, historical_resample


def test_nonzero_constant_lead_gets_edge_variation() -> None:
    """A constant raw lead may look variable after the historical FIR edges."""
    signal = np.ones((12, 5000), dtype=np.float32)
    result = historical_resample(signal)
    assert np.ptp(signal[0]) == 0
    assert np.ptp(result[0]) > 0


def test_halves_resampled_independently() -> None:
    """A pulse near the split may not bleed into the other five-second half."""
    signal = np.zeros((12, 5000), dtype=np.float32)
    signal[:, 2499] = 1
    result = historical_resample(signal)
    assert np.array_equal(result[:, 1250:], np.zeros((12, 1250), dtype=np.float32))
    assert np.any(result[:, :1250] != 0)
    assert not np.array_equal(result, resample_poly(signal, 1, 2, axis=1).astype(np.float32))


@pytest.mark.parametrize("signal", [np.zeros((11, 5000), dtype=np.float32),
                                         np.zeros((12, 5000), dtype=np.float64),
                                         np.full((12, 5000), np.nan, dtype=np.float32)])
def test_rejects_wrong_input(signal: np.ndarray) -> None:
    """Resampling refuses malformed or nonfinite canonical records."""
    with pytest.raises(ValueError, match="Expected finite float32"):
        historical_resample(signal)


def test_evenly_spaced_indices() -> None:
    """Sampling includes the endpoints and refuses impossible populations."""
    positions = evenly_spaced_indices(100, 32)
    assert positions[0] == 0
    assert positions[-1] == 99
    assert len(set(positions)) == 32
    with pytest.raises(ValueError, match="Population too small"):
        evenly_spaced_indices(31)
