"""Scientific checks for independent empirical localization calibration."""

import numpy as np
import pytest

from ecg_experiment.calibrated_localization import calibrated_scores, equal_width_pieces, region_difference


def test_conservative_tail_ties_and_extremes() -> None:
    """Tied distances retain all equals in the right tail and extremes remain finite."""
    reference = np.array([1.0, 2.0, 2.0, 4.0])[:, None, None]
    observed = np.array([0.0, 2.0, 5.0])[:, None, None]
    np.testing.assert_allclose(calibrated_scores(reference, observed).ravel(), -np.log([1.0, 4 / 5, 1 / 5]))


def test_invariant_channel_rescaling() -> None:
    """Units with different positive scales have the same empirical rank probabilities."""
    reference = np.arange(6.0)[:, None, None] * np.array([1.0, 100.0])[None, None, :]
    observed = np.array([1.5, 3.5])[:, None, None] * np.array([1.0, 100.0])[None, None, :]
    found = calibrated_scores(reference, observed)
    np.testing.assert_array_equal(found[:, :, 0], found[:, :, 1])


def test_equal_width_preserves_constant_and_endpoint_signals() -> None:
    """Coordinate-count control preserves constant morphology and block endpoints."""
    pieces = {"P": np.linspace(-1, 1, 48)[None, None, :], "T": np.ones((1, 1, 63))}
    found = equal_width_pieces(pieces)
    assert found["P"].shape == (1, 1, 35)
    np.testing.assert_allclose(found["P"][..., [0, -1]], [[[-1, 1]]])
    np.testing.assert_allclose(found["T"], 1)


def test_patient_overlap_rejected() -> None:
    """A patient cannot enter both infarct contrast groups."""
    with pytest.raises(ValueError, match="disjoint"):
        region_difference(np.array([1]), np.array([0.0]), np.array([1]), np.array([0.0]), draws=2)


def test_ties_average_unique_top_leads() -> None:
    """Multiple tied blocks on one lead do not inflate its regional vote."""
    from ecg_experiment.calibrated_localization import tied_lead_membership

    assert tied_lead_membership(np.array([5.0, 5.0, 5.0, 1.0]), np.array([0, 0, 1, 2]), np.array([0])) == 0.5


def test_presorted_calibration_equivalent() -> None:
    """Precomputed sorted channel references retain exactly the empirical recipe."""
    reference = np.array([4.0, 2.0, 1.0, 2.0])[:, None, None]
    observed = np.array([0.0, 2.0, 5.0])[:, None, None]
    np.testing.assert_array_equal(
        calibrated_scores(reference, observed),
        calibrated_scores(np.sort(reference, axis=0), observed, presorted=True),
    )


def test_premature_ties_average_overlap() -> None:
    """Saturated units cannot win premature localization by array order."""
    from ecg_experiment.calibrated_localization import tied_premature_hit

    found = tied_premature_hit(
        np.array([5.0, 5.0, 1.0]), np.array([0.0, 1.0, 2.0]), np.array([0.5, 1.5, 2.5]), [(0.0, 0.5)]
    )
    assert found == (0.5, 1 / 3)
