"""Scientific invariants of the prospective aligned phase residual map."""

import numpy as np

from ecg_experiment.lead_wave_maps import UnitMap
from ecg_experiment.raw_residual052 import fixed_window_map, perturb
from ecg_experiment.raw_residual054 import (
    aligned_field,
    calibrated_map,
    phase_indices,
    record_phase_maxima,
    shared_mask,
)


def _regular():
    peaks = np.arange(300, 4300, 500)
    signal = np.zeros((12, 4500))
    shape = np.exp(-(np.arange(-60, 61) ** 2) / 100)
    for peak in peaks:
        signal[:, peak - 60 : peak + 61] = shape
    return signal, peaks


def test_shared_qrs_shift_reduces_jitter_without_moving_output_timeline():
    signal, peaks = _regular()
    target = int(peaks[3])
    signal[:, target - 60 : target + 61] = 0
    shape = np.exp(-(np.arange(-60, 61) ** 2) / 100)
    signal[:, target - 57 : target + 64] = shape
    field, audit = aligned_field(signal, peaks)
    assert audit["shifts"][3] == 3
    assert np.max(field) < 1e-12
    assert np.array_equal(audit["peaks"], peaks)


def test_beat_amplitude_change_is_preserved_and_lead_permutation_equivariant():
    signal, peaks = _regular()
    target = int(peaks[3])
    signal[8, target - 30 : target + 40] *= 2
    field, audit = aligned_field(signal, peaks)
    assert field[8, target - 10 : target + 10].max() > 0.9
    assert field[0].max() < 1e-15
    assert np.all(audit["shifts"] == 0)
    permutation = np.roll(np.arange(12), 3)
    permuted, _ = aligned_field(signal[permutation], peaks)
    assert np.allclose(permuted, field[permutation])


def test_st_support_constant_offset_and_translation_invariance():
    signal, peaks = _regular()
    edited, _ = perturb(signal, peaks, int(peaks[3]), [8], "ST")
    field, _ = aligned_field(edited, peaks)
    shifted, _ = aligned_field(edited + 2.7, peaks)
    assert np.allclose(field, shifted)
    translated = np.pad(edited, ((0, 0), (100, 0)))
    translated_field, _ = aligned_field(translated, peaks + 100)
    assert np.allclose(translated_field[:, 100:], field)
    assert field[8, peaks[3] + 50 : peaks[3] + 100].max() > 0


def test_extended_edge_templates_are_finite_and_cdf_keeps_conservative_equal_tails():
    signal, peaks = _regular()
    edge_peaks = np.concatenate([[125], peaks])
    field, audit = aligned_field(signal, edge_peaks)
    assert np.isfinite(field).all()
    assert np.isfinite(audit["predictions"]).all()
    units = UnitMap(np.array([0.0, 1.0, 2.0, 3.0]), np.array([0, 0, 0, 0]), np.zeros(4), np.ones(4) * 0.14)
    references = np.broadcast_to(np.array([0.0, 1.0, 2.0])[:, None, None], (3, 12, 4))
    calibrated = calibrated_map(units, np.zeros(4, dtype=int), references)
    assert np.allclose(calibrated.scores, -np.log(np.array([4, 3, 2, 1]) / 4))


def test_every_common_candidate_has_phase_and_all_record_channels():
    signal, peaks = _regular()
    field, _ = aligned_field(signal, peaks)
    units = fixed_window_map(field, shared_mask(signal, peaks))
    phases = phase_indices(units, peaks, signal.shape[1])
    assert set(phases) == {0, 1, 2, 3}
    assert record_phase_maxima(units, phases).shape == (12, 4)


def test_calibrated_lead_permutation_requires_permuted_reference_channels():
    scores = np.array([0.2, 0.4, 0.8, 1.6])
    units = UnitMap(scores, np.arange(4), np.zeros(4), np.ones(4) * 0.14)
    references = np.broadcast_to(np.arange(1, 13)[None, :, None], (3, 12, 4)).copy() / 10
    permutation = np.roll(np.arange(12), 1)
    permuted_leads = np.array([np.flatnonzero(permutation == lead)[0] for lead in units.leads])
    permuted = UnitMap(scores, permuted_leads, units.starts, units.ends)
    phase = np.zeros(4, dtype=int)
    original = calibrated_map(units, phase, references)
    shifted = calibrated_map(permuted, phase, references[:, permutation, :])
    assert np.array_equal(original.scores, shifted.scores)
