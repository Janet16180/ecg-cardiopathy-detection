"""Reference reconstruction and identity-denominator checks."""

from types import SimpleNamespace

import numpy as np
import pytest

from ecg_experiment.condition_keys import region_vectors, spans
from ecg_experiment.reference_ecg_audit import angle, selected


def test_saved_selection_reconstruction_matches_candidate_pooling():
    rng = np.random.default_rng(42)
    raw = rng.normal(size=(6, 40, 8)).astype(np.float16).astype(float)
    scaler = SimpleNamespace(mean_=rng.normal(size=8), scale_=rng.uniform(0.3, 2, size=8))
    choices = rng.integers(0, 109, size=(6, 4))
    expected = region_vectors(raw, scaler, spans())[np.arange(6)[:, None], choices]
    np.testing.assert_allclose(selected(raw, choices, spans(), scaler), expected, atol=1e-14)


def test_angle_respects_direction_and_not_vector_length():
    assert angle(np.array([1.0, 0.0]), np.array([2.0, 0.0])) == 0
    assert angle(np.array([1.0, 0.0]), np.array([0.0, 2.0])) == pytest.approx(90)
    assert angle(np.array([1.0, 0.0]), np.array([-2.0, 0.0])) == pytest.approx(180)
