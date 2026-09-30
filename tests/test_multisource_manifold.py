"""Checks of the Experiment 026b fit-set helpers."""

import numpy as np

from ecg_experiment.ecg_quality import FLAT_SECONDS
from ecg_experiment.multisource_manifold import equal_family_subsample, window_reasons

NAMES = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]


def record(samples: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    time = np.arange(samples) / 500
    return 0.5 * np.sin(2 * np.pi * 1.2 * time)[:, None] + rng.normal(0, 0.05, (samples, 12))


def test_equal_family_subsample_takes_the_smallest_count_from_each_family():
    families = np.array(["b"] * 7 + ["a"] * 3 + ["c"] * 5)
    chosen = equal_family_subsample(families, seed=1)
    names, counts = np.unique(families[chosen], return_counts=True)
    assert list(names) == ["a", "b", "c"]
    assert list(counts) == [3, 3, 3]
    assert len(np.unique(chosen)) == len(chosen)
    assert np.all(np.diff(chosen) > 0)


def test_equal_family_subsample_depends_only_on_labels_and_seed():
    families = np.array(["x", "y"] * 20 + ["x"] * 5)
    first = equal_family_subsample(families, seed=30030)
    assert np.array_equal(first, equal_family_subsample(families.copy(), seed=30030))
    assert not np.array_equal(first, equal_family_subsample(families, seed=30031))


def test_window_reasons_reads_only_the_saved_window():
    signal = record(7000, seed=0)
    signal[:1000, 3] = 0.0
    assert window_reasons(signal, NAMES, start=1000) == []
    assert "flat_segment" in window_reasons(signal, NAMES, start=0)
    assert int(FLAT_SECONDS * 500) <= 1000


def test_window_reasons_flags_a_constant_lead():
    signal = record(5000, seed=2)
    signal[:, 7] = 0.0
    assert "constant_lead" in window_reasons(signal, NAMES, start=0)
