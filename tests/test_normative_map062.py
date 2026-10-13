"""Invariants of the Experiment 062 normative phase-aligned map."""

import numpy as np

from ecg_experiment import normative_map062 as nm


def _wave(t: np.ndarray, centre: float, width: float, height: float) -> np.ndarray:
    return height * np.exp(-((t - centre) ** 2) / (2 * width**2))


def _synthetic(odd: int | None = None, t_height: float = 0.3) -> np.ndarray:
    t = np.arange(nm.SAMPLES) / nm.FS
    signal = np.zeros((nm.LEADS, nm.SAMPLES))
    gains = np.linspace(0.6, 1.4, nm.LEADS)
    for index, r in enumerate(np.arange(0.4, 9.8, 0.8)):
        beat = _wave(t, r - 0.16, 0.02, 0.1) + _wave(t, r, 0.012, 1.2) + _wave(t, r + 0.03, 0.01, -0.3)
        beat += _wave(t, r + 0.3, 0.05, -t_height if index == odd else t_height)
        signal += gains[:, None] * beat[None]
    return signal


def test_grid_segments_and_round_trip():
    assert nm.BINS == 380
    assert nm.SEGMENT_OF_BIN[0] == 0
    assert nm.SEGMENT_OF_BIN[nm.PRE] == 1
    assert nm.SEGMENT_OF_BIN[-1] == len(nm.SEGMENTS) - 1
    knots = np.array([[1000.0, 1150.0, 1200.0, 1260.0, 1350.0, 1400.0]])
    values = np.arange(nm.BINS, dtype=float)[None, None].repeat(nm.LEADS, axis=1)
    timeline = nm.to_timeline(values, knots)
    assert timeline[0, 1150] == nm.PRE
    assert timeline[0, 1200] == nm.GRID_KNOTS[2]
    assert timeline[0, 1350] == nm.GRID_KNOTS[4]
    assert timeline[0, 999] == timeline.min()


def test_features_find_regular_beats():
    features = nm.record_features(_synthetic())
    assert features is not None
    assert len(features.knots) >= 10
    assert np.all(np.diff(features.knots, axis=1) > 0)
    assert abs(features.intervals[0] - 75.0) < 1.0


def test_constant_offset_does_not_change_median_beat():
    signal = _synthetic()
    shifted = signal + 0.5
    first, second = nm.record_features(signal), nm.record_features(shifted)
    np.testing.assert_allclose(first.median, second.median, atol=1e-6)


def test_amplitude_change_is_kept_in_mv():
    small, large = nm.record_features(_synthetic(t_height=0.2)), nm.record_features(_synthetic(t_height=0.4))
    t_bins = nm.SEGMENTS.index("ST") <= nm.SEGMENT_OF_BIN
    difference = np.abs(large.median[:, t_bins] - small.median[:, t_bins]).max(axis=1)
    gains = np.linspace(0.6, 1.4, nm.LEADS)
    np.testing.assert_allclose(difference, 0.2 * gains, rtol=0.1)


def test_odd_beat_has_largest_focal_deviation():
    features = nm.record_features(_synthetic(odd=5))
    centre = features.knots[:, 1]
    odd = int(np.argmin(np.abs(centre - (0.4 + 5 * 0.8) * nm.FS)))
    assert not features.dominant[odd]
    assert int(np.argmax(np.abs(features.focal).max(axis=(1, 2)))) == odd


def test_tail_score_is_monotone_and_continuous():
    values = np.random.default_rng(0).exponential(size=100_000)
    table = nm.tail_table(values)
    assert table[-1, 0] <= 4.0
    grid = np.linspace(0, 30, 3001)
    scores = nm.tail_score(grid, table)
    assert np.all(np.diff(scores) >= 0)
    edge = table[-1, 1]
    np.testing.assert_allclose(
        nm.tail_score(np.array([edge - 1e-9, edge + 1e-9]), table), table[-1, 0], atol=1e-6
    )
    np.testing.assert_allclose(nm.tail_score(np.array([np.log(10.0)]), table), 1.0, atol=0.05)


def test_subspace_coordinates_are_bounded():
    rng = np.random.default_rng(1)
    medians = rng.normal(size=(200, 40)) @ rng.normal(size=(40, nm.LEADS * nm.BINS)) * 0.01
    covariates = nm.design(rng.uniform(50, 100, 200), rng.uniform(20, 80, 200), rng.integers(0, 2, 200))
    model = nm.fit_population(medians, covariates, rank=6)
    extreme = (model.basis[:, 0] * 100 * np.sqrt(model.eigenvalues[0])) * model.scale
    coordinates = nm.subspace_coefficients(model, extreme / model.scale)
    assert np.all(np.abs(coordinates) <= nm.BOX * np.sqrt(model.eigenvalues) + 1e-9)


def test_patient_folds_keep_patients_together():
    patients = np.array([1, 1, 2, 3, 3, 3, 4, 5, 6, 7])
    folds = nm.patient_folds(patients, folds=3, seed=0)
    for patient in np.unique(patients):
        assert len(set(folds[patients == patient])) == 1


def test_lexicographic_ranks_qrs_first_and_keeps_anomaly_order():
    knots = np.array([[1000.0, 1150.0, 1200.0, 1260.0, 1350.0, 1400.0]])
    anomaly = np.zeros((nm.LEADS, nm.SAMPLES), dtype=np.float32)
    anomaly[0, 1300] = 50.0
    anomaly[1, 1170] = 2.0
    combined = nm.qrs_lexicographic(anomaly, knots)
    assert combined[2, 1170] > combined[0, 1300]
    assert combined[1, 1170] > combined[2, 1170]
    assert combined.max() < 2.0
