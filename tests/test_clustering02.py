"""Tests of fitting-only transformations, stable selection and consensus invariance."""

import hashlib

import numpy as np
import pandas as pd
import pytest

from ecg_experiment.clustering01 import FEATURES
from ecg_experiment.clustering02 import (
    anchor_patient,
    balanced,
    coassociation,
    consensus_candidates,
    consensus_partition,
    eligibility,
    fit_representation,
    permitted_metadata,
    rank_key,
    shape_features,
    transform_representation,
)


@pytest.fixture
def measurements():
    rng = np.random.default_rng(81)
    features = pd.DataFrame(rng.normal(size=(100, 53)), columns=FEATURES)
    nuisance = pd.DataFrame(
        {
            "device": ["a"] * 50 + ["b"] * 50,
            "site": "one",
            "hr_bpm": rng.uniform(50, 90, 100),
            "signal_std": rng.uniform(0.1, 0.3, 100),
            "baseline_fraction": 0.05,
            "beat_consistency": 0.9,
            "constant_leads": 0,
            "quality": 0,
        }
    )
    return features, nuisance


def test_internal_anchor_rule_is_independently_reproducible():
    for patient in [1, 10, 400, 9000]:
        expected = (
            int(hashlib.sha256(f"cluster02_anchor:20261018:{patient}".encode()).hexdigest(), 16) % 5 == 0
        )
        assert anchor_patient(patient) == expected


@pytest.mark.parametrize("view", ["raw", "residual", "shape"])
def test_anchor_transformation_never_adapts_fitted_statistics(measurements, view):
    features, nuisance = measurements
    state = fit_representation(features.iloc[:80], nuisance.iloc[:80], view)
    before_clip = state["clip"].copy()
    before_pca = state["pca"].components_.copy()
    anchor = features.iloc[80:].copy()
    anchor.iloc[0] = 1e6
    anchor_nuisance = nuisance.iloc[80:].copy()
    anchor_nuisance["device"] = "unseen"
    transformed = transform_representation(state, anchor, anchor_nuisance)
    assert np.isfinite(transformed).all()
    assert transformed.shape[0] == 20
    assert transformed.shape[1] <= 16
    np.testing.assert_array_equal(before_clip, state["clip"])
    np.testing.assert_array_equal(before_pca, state["pca"].components_)
    individual = transform_representation(state, anchor.iloc[:1], anchor_nuisance.iloc[:1])
    np.testing.assert_allclose(individual[0], transformed[0], atol=1e-12)


def test_shape_normalization_preserves_duration_and_removes_common_amplitude_scale(measurements):
    features, nuisance = measurements
    original = shape_features(features, nuisance, "shape")
    scaled, changed = features.copy(), nuisance.copy()
    columns = [name for name in features if name.startswith(("r_amp_", "s_amp_", "st_", "t_amp_"))]
    scaled[columns] *= 7
    changed["signal_std"] *= 7
    normalized = shape_features(scaled, changed, "shape")
    np.testing.assert_allclose(original, normalized)
    np.testing.assert_array_equal(original.qrs_ms, features.qrs_ms)


def test_cluster_balance_rejects_collapse_and_tiny_components():
    assert balanced(np.repeat([0, 1], 50), 2)
    assert not balanced(np.zeros(100, dtype=int), 2)
    assert not balanced(np.array([0] * 96 + [1] * 4), 2)
    assert not balanced(np.repeat([0, 1], 50), 4)


def test_eligibility_requires_worst_refit_and_all_convergences():
    row = {
        "converged": [True] * 9,
        "reference_fit_balanced": True,
        "reference_anchor_balanced": True,
        "refits_anchor_balanced": [True] * 8,
        "silhouette": 0.2,
        "stability": {"median": 0.95, "minimum": 0.59},
    }
    assert not all(eligibility(row).values())
    row["stability"]["minimum"] = 0.7
    assert all(eligibility(row).values())
    row["converged"][3] = False
    assert not all(eligibility(row).values())


def test_selection_prefers_stability_over_silhouette():
    first = {
        "eligible": True,
        "stability": {"median": 0.9, "minimum": 0.7},
        "silhouette": 0.1,
        "clusters": 2,
        "covariance": "diag",
    }
    second = {**first, "stability": {"median": 0.85, "minimum": 0.7}, "silhouette": 0.5}
    assert rank_key(first) > rank_key(second)
    assert rank_key({**second, "eligible": True}) > rank_key({**first, "eligible": False})


def test_consensus_is_invariant_to_arbitrary_cluster_numbers():
    first = np.repeat([0, 1], 10)
    second = np.repeat([12, 7], 10)
    np.testing.assert_array_equal(coassociation([first]), coassociation([first, second]))
    result = consensus_partition([first, second], 2)
    np.testing.assert_array_equal(result[:, None] == result[None, :], first[:, None] == first[None, :])


def test_consensus_never_uses_ineligible_component_views():
    selected = {f"{view}:gmm": {"eligible": view != "shape"} for view in ("raw", "residual", "shape")}
    assert consensus_candidates(selected, {}, np.empty((0, 2))) == ([], {})


def test_metadata_does_not_interpret_unselected_diagnoses(tmp_path):
    path = tmp_path / "metadata.csv"
    header = (
        "ecg_id,device,site,baseline_drift,static_noise,burst_noise,"
        "electrodes_problems,extra_beats,pacemaker,scp_codes\n"
    )
    path.write_text(header + "1,a,b,,,,,,,not a dictionary\n2,c,d,,,,,,,{'NORM': 100}\n")
    safe = permitted_metadata(path, pd.Series([2]), diagnoses=True)
    assert safe.scp_codes.iloc[0] == {"NORM": 100}
    hidden = permitted_metadata(path, pd.Series([1]), diagnoses=False)
    assert "scp_codes" not in hidden
