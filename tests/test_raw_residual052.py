"""Check exact-support localization and control invariants without patient data."""

import numpy as np
import pytest

from ecg_experiment.lead_wave_maps import UnitMap
from ecg_experiment.raw_residual052 import (
    baseline_field,
    baseline_support,
    fixed_window_map,
    location_metrics,
    perturb,
    residual_field,
    support_mask,
)


def _regular():
    peaks = np.arange(300, 4300, 500)
    signal = np.zeros((12, 4500))
    shape = np.exp(-(np.arange(-60, 61) ** 2) / 100)
    for peak in peaks:
        signal[:, peak - 60 : peak + 61] = shape
    return signal, peaks


def test_single_lead_known_st_change_localizes_exact_center_and_lead():
    signal, peaks = _regular()
    edited, supports = perturb(signal, peaks, int(peaks[3]), [8], "ST")
    mask = support_mask(signal.shape[1], peaks)
    found = location_metrics(fixed_window_map(residual_field(edited, peaks), mask), supports, [8])
    assert found["joint_hit"] == 1
    assert found["overlap_fraction"] == pytest.approx(0.10 / 0.14)


def test_qrs_transplant_is_bounded_and_localizes_changed_leads():
    signal, peaks = _regular()
    edited, supports = perturb(signal, peaks, int(peaks[3]), [1, 6, 9], "QRS")
    mask = support_mask(signal.shape[1], peaks)
    left, right = (round(x * 500) for x in supports[0])
    assert np.array_equal(edited[:, :left], signal[:, :left])
    assert np.array_equal(edited[:, right:], signal[:, right:])
    assert np.array_equal(edited[0], signal[0])
    found = location_metrics(fixed_window_map(residual_field(edited, peaks), mask), supports, [1, 6, 9])
    assert found["joint_hit"] == 1


def test_leave_one_out_zero_on_identical_beats_and_constant_offset():
    signal, peaks = _regular()
    original = residual_field(signal, peaks)
    shifted = residual_field(signal + 3.5, peaks)
    assert np.max(original) == 0
    assert np.allclose(shifted, original, atol=1e-20)
    assert np.isfinite(residual_field(np.zeros_like(signal), peaks)).all()


def test_midpoint_ownership_and_common_support_excludes_gaps():
    signal, peaks = _regular()
    mask = support_mask(signal.shape[1], peaks)
    units = fixed_window_map(np.ones_like(signal), mask)
    for low, high in zip(
        units.starts[: len(units.starts) // 12], units.ends[: len(units.ends) // 12], strict=True
    ):
        assert mask[round(low * 500) : round(high * 500)].all()
    close = np.array([300, 510, 1000])
    close_mask = support_mask(1300, close)
    assert close_mask[405:510].all()


def test_plateau_ties_are_expected_not_earliest_argmax():
    units = UnitMap(np.ones(3), np.array([0, 0, 1]), np.array([0.0, 0.1, 0.2]), np.array([0.14, 0.24, 0.34]))
    found = location_metrics(units, [(0.15, 0.30)], [1])
    assert found["hit"] == pytest.approx(2 / 3)
    assert found["joint_hit"] == pytest.approx(1 / 3)
    assert found["tie_count"] == 3


def test_baseline_broadcast_preserves_lead_and_clips_support():
    units = UnitMap(np.array([7.0, 9.0]), np.array([2, 2]), np.array([-0.1, 0.1]), np.array([0.2, 0.3]))
    field = baseline_field(units, 200)
    assert np.all(field[2, :50] == 7)
    assert np.all(field[2, 50:150] == 9)
    assert not field[0].any()
    assert np.array_equal(baseline_support(units, 200), np.arange(200) < 150)


def test_lead_permutation_equivariance_and_insufficient_beats_fail():
    signal, peaks = _regular()
    edited, _ = perturb(signal, peaks, int(peaks[3]), [8], "ST")
    permutation = np.roll(np.arange(12), 2)
    assert np.array_equal(
        residual_field(edited[permutation], peaks), residual_field(edited, peaks)[permutation]
    )
    with pytest.raises(ValueError, match="three"):
        residual_field(signal, peaks[:2])
