"""Tests of the Experiment 041 section maps on synthetic data."""

from __future__ import annotations

import numpy as np

from ecg_experiment.fragment_localization import (
    SECTIONS,
    bootstrap_mean,
    fit_section_reference,
    gated,
    plot_sections,
    pointing,
    premature_windows,
    r_peaks,
    readout_contributions,
    red_threshold,
    section_kmeans,
    section_mahalanobis,
    section_overlap,
)
from ecg_experiment.full_development import fit_logistic


def synthetic_tokens(records: int, width: int = 80, seed: int = 0) -> np.ndarray:
    """Normal-like tokens with a position-dependent mean."""
    rng = np.random.default_rng(seed)
    positions = np.linspace(-1, 1, SECTIONS)[None, :, None]
    return rng.normal(size=(records, SECTIONS, width)) + positions


def test_reference_flags_an_altered_section() -> None:
    reference = fit_section_reference(synthetic_tokens(60), seed=1)
    tokens = synthetic_tokens(5, seed=2)
    tokens[:, 17] += 6.0
    for scores in (section_mahalanobis(reference, tokens), section_kmeans(reference, tokens)):
        assert scores.shape == (5, SECTIONS)
        assert (scores.argmax(axis=1) == 17).all()


def test_kmeans_distance_matches_brute_force() -> None:
    reference = fit_section_reference(synthetic_tokens(40), seed=1)
    tokens = synthetic_tokens(3, seed=3)
    centred = tokens - reference.position_means
    scaler, pca, _ = reference.centred
    coordinates = pca.transform(scaler.transform(centred.reshape(-1, tokens.shape[2]))) / reference.whitening
    brute = ((coordinates[:, None] - reference.centroids[None]) ** 2).sum(axis=2).min(axis=1)
    np.testing.assert_allclose(section_kmeans(reference, tokens).ravel(), brute, rtol=1e-9, atol=1e-9)


def test_contributions_average_to_the_logit() -> None:
    tokens = synthetic_tokens(80)
    y = np.arange(80) % 2
    tokens[y == 1, 5] += 1.0
    head = fit_logistic(tokens.mean(axis=1), y)
    scaler, model = head
    logits = model.decision_function(scaler.transform(tokens.mean(axis=1)))
    np.testing.assert_allclose(readout_contributions(head, tokens).mean(axis=1), logits, atol=1e-10)


def test_gated_keeps_only_positive_contributions() -> None:
    unsupervised = np.array([[1.0, 2.0, 3.0]])
    contributions = np.array([[-1.0, 0.5, 0.0]])
    np.testing.assert_array_equal(gated(unsupervised, contributions), [[0.0, 2.0, 0.0]])


def test_red_threshold_leaves_the_budget() -> None:
    maps = np.random.default_rng(0).normal(size=(1000, SECTIONS))
    threshold = red_threshold(maps, 0.05)
    assert abs((maps > threshold).any(axis=1).mean() - 0.05) < 0.002


def test_premature_beat_windows_and_sections() -> None:
    fs = 500
    peaks = np.array([0, 500, 1000, 1300, 2000, 2500]) + 100
    windows = premature_windows(peaks, fs)
    assert len(windows) == 1
    np.testing.assert_allclose(windows[0], (2.8 - 0.1, 2.8 + 0.4))
    overlap = section_overlap(windows)
    np.testing.assert_array_equal(np.flatnonzero(overlap), [10, 11, 12])
    assert premature_windows(peaks[:3], fs) == []


def test_r_peaks_on_synthetic_spikes() -> None:
    fs = 500
    signal = np.zeros((12, 5000))
    beats = np.arange(250, 5000, 400)
    for beat in beats:
        signal[:, beat - 5:beat + 5] = np.hanning(10)
    found = r_peaks(signal, fs)
    assert len(found) == len(beats)
    assert np.abs(found - beats).max() <= 5


def test_pointing_hit_and_chance() -> None:
    maps = np.zeros((2, SECTIONS))
    maps[0, 3] = maps[1, 30] = 1.0
    overlap = np.zeros((2, SECTIONS), dtype=bool)
    overlap[:, 2:6] = True
    hit, chance = pointing(maps, overlap)
    np.testing.assert_array_equal(hit, [1.0, 0.0])
    np.testing.assert_allclose(chance, [0.1, 0.1])


def test_bootstrap_mean_interval_contains_the_mean() -> None:
    values = np.arange(20, dtype=float)
    patients = np.repeat(np.arange(10), 2)
    result = bootstrap_mean(patients, values, draws=200, seed=0)
    assert result["ci_low"] <= result["value"] <= result["ci_high"]
    assert result["value"] == values.mean()


def test_plot_sections_draws_one_column_per_map() -> None:
    red = {"a": np.zeros(SECTIONS, dtype=bool), "b": np.ones(SECTIONS, dtype=bool)}
    figure = plot_sections(np.zeros((12, 5000)), 500, red, "title")
    assert len(figure.axes) == 24
