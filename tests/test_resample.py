import numpy as np
import pytest

from ecg_experiment.cpc_input_audit import historical_resample
from ecg_experiment.resample import resample_full

JOIN = 1250
MARGIN = 16


def smooth_record(rate: int, seconds: float = 10.0) -> np.ndarray:
    times = np.arange(int(rate * seconds)) / rate
    gains = np.linspace(0.5, 1.5, 12)[:, None]
    return 0.5 + gains * np.sin(2 * np.pi * 1.2 * times) + 0.2 * np.cos(2 * np.pi * 7 * times)


def test_output_shape_and_dtype() -> None:
    assert resample_full(smooth_record(500).astype(np.float32), 1, 2).shape == (12, 2500)
    code = resample_full(smooth_record(400).astype(np.float32), 5, 8)
    assert code.shape == (12, 2500)
    assert code.dtype == np.float32


def test_no_discontinuity_at_five_seconds() -> None:
    full = resample_full(smooth_record(500).astype(np.float32), 1, 2)
    historical = historical_resample(smooth_record(500).astype(np.float32))
    expected = smooth_record(250)
    near_join = slice(JOIN - MARGIN, JOIN + MARGIN)

    assert np.abs(full[:, near_join] - expected[:, near_join]).max() < 1e-3
    assert np.abs(historical[:, near_join] - expected[:, near_join]).max() > 0.05
    steps = np.abs(np.diff(full, axis=1))
    assert steps[:, JOIN - 1].max() <= steps[:, MARGIN:-MARGIN].max()


def test_matches_historical_away_from_join() -> None:
    signal = smooth_record(500).astype(np.float32)
    full = resample_full(signal, 1, 2)
    historical = historical_resample(signal)
    away = np.r_[0:JOIN - MARGIN, JOIN + MARGIN:2500]

    np.testing.assert_allclose(full[:, away], historical[:, away], atol=1e-6)


def test_rejects_nonfinite_input() -> None:
    signal = smooth_record(500).astype(np.float32)
    signal[3, 100] = np.nan
    with pytest.raises(ValueError, match="finite"):
        resample_full(signal, 1, 2)
