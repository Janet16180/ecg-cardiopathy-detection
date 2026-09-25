"""Scientific invariants for the equal-width compact-CPC readout."""

from __future__ import annotations

import numpy as np

from ecg_experiment.cpc_local_readout import arm_matrix, bootstrap_draws


def test_arms_replace_only_context_maxima() -> None:
    """Both arms retain context means and expose exactly 512 coordinates."""
    features = np.arange(2 * 2 * 512, dtype=np.float32).reshape(2, 2, 512)
    a = arm_matrix(features, "A")
    b = arm_matrix(features, "B")
    assert a.shape == b.shape == (2, 512)
    np.testing.assert_array_equal(a[:, :256], b[:, :256])
    np.testing.assert_array_equal(b[:, 256:], features[:, 1, 256:])


def test_bootstrap_retains_patient_ecgs_together() -> None:
    """A sampled patient contributes every ECG on each occurrence."""
    patients = ["p1", "p1", "p2", "p3", "p3", "p3"]
    for draw in bootstrap_draws(patients, 20):
        counts = np.bincount(draw, minlength=len(patients))
        assert counts[0] == counts[1]
        assert counts[3] == counts[4] == counts[5]
