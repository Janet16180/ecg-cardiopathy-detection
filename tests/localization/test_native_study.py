"""Behavioral checks for native reference controls, proposals and paired endpoints."""

from dataclasses import replace

import numpy as np
import pytest

from ecg_experiment.localization.native_study import (
    NATIVE_METHODS,
    banks_for_draw,
    descriptors,
    native_predictions,
    native_summary,
    region_matches,
    validate_profile,
)
from ecg_experiment.native_reference_ntt05 import (
    ReferenceInstance,
    SignalRegion,
    lead_patch_tokens,
    reference_bank,
)
from ecg_experiment.same_wave_ntt04 import FINDINGS, METHODS, SEEDS


def tokens():
    """Return a lead/time grid with distinguishable first and last half semantics."""
    values = np.ones((8, 50, 3))
    values[:, :25] = [1, 0, 0]
    values[:, 25:] = [0, 1, 0]
    return lead_patch_tokens(
        values,
        ("I", "II", "V1", "V2", "V3", "V4", "V5", "V6"),
        250,
        50,
        2500,
        "synthetic",
        "full context",
        "record-a",
    )


def test_control_vectors_use_actual_time_anchors():
    """Temporal controls must not accidentally alias the QRS descriptor."""
    t = tokens()
    vectors = descriptors(t, [SignalRegion(1, 1.3, ("V1", "V6"))], [SignalRegion(7, 7.3, ("V1", "V6"))])
    np.testing.assert_allclose(vectors["native_qrs"], [1, 0, 0])
    np.testing.assert_allclose(vectors["native_late"], [0, 1, 0])
    np.testing.assert_allclose(vectors["native_global_pair"], [2**-0.5, 2**-0.5, 0])


def test_regions_preserve_original_coordinates_and_separate_decay():
    """Disjoint repeated beats stay separate and all grid windows retain score provenance."""
    t = tokens()
    bank = reference_bank(
        [
            ReferenceInstance("support1", "a", np.array([1, 0, 0]), "synthetic", "synthetic"),
            ReferenceInstance("support2", "b", np.array([0, 1, 0]), "synthetic", "synthetic"),
        ],
        ("a", "b"),
    )
    qrs = [SignalRegion(1, 1.3, ("V1", "V6")), SignalRegion(2, 2.3, ("V1", "V6"))]
    matches = region_matches(t, qrs, bank, "target")
    beats = [r for r in matches if r["proposal"] == "automatic_qrs"]
    assert len(beats) == 2
    assert [r["start_seconds"] for r in beats] == [1, 2]
    assert any(r["stop_seconds"] == 10 for r in matches)
    assert all(
        r["context_scope"] == "full context" and r["record_frame_local"] == "record-a" for r in matches
    )
    assert all(r["adjusted_similarity"] <= r["raw_similarity"] + 1e-12 for r in matches)
    with pytest.raises(ValueError, match="reference bank"):
        region_matches(t, qrs, bank, "support1")
    with pytest.raises(ValueError, match="embedding"):
        region_matches(replace(t, embedding_space="other"), qrs, bank, "target")


def fixture_selection():
    """Build three independent finding references and targets across all fixed draws."""
    features, targets, groups, previous = {}, [], [], []
    for i, f in enumerate(FINDINGS):
        vector = np.eye(3)[i]
        for ecg in (i + 1, i + 101):
            features[ecg] = {
                m: vector.copy()
                for m in ("native_qrs", "native_late", "native_global_pair", "native_global_eight")
            }
        targets.append({"ecg_id": i + 1, "patient_id": i + 1, "finding": f})
        for seed in SEEDS:
            groups.append(
                {"seed": seed, "finding": f, "supports": [{"ecg_id": i + 101, "patient_id": i + 101}]}
            )
    for seed in SEEDS:
        for i, t in enumerate(targets):
            previous.append({**t, "seed": seed, "scores": {m: np.eye(3)[i].tolist() for m in METHODS}})
    return features, {"targets": targets, "support_draws": groups}, previous


def test_fixed_supports_preserve_labels_and_wrong_controls():
    """Cyclic label control changes choices without changing the native feature bank."""
    f, s, p = fixture_selection()
    predictions, banks = native_predictions(f, s, p, "synthetic")
    assert len(predictions) == 9
    assert len(banks) == 3
    assert all(
        r["correct"]["native_qrs"] == 1 and r["correct"]["native_shuffled_labels"] == 0 for r in predictions
    )
    s["targets"][0]["patient_id"] = 101
    with pytest.raises(ValueError, match="leaked"):
        native_predictions(f, s, p, "synthetic")


def test_primary_gate_cannot_be_rescued_by_a_secondary_model():
    """The declared QRS comparator alone governs advancement and all patients stay grouped."""
    f, s, p = fixture_selection()
    predictions, _ = native_predictions(f, s, p, "synthetic")
    for row in predictions:
        row["correct"]["native_qrs"] = 0
    result = native_summary(predictions, draws=100)
    assert result["primary"]["difference"] == -1
    assert result["primary"]["ci_high"] == -1
    assert result["balanced_accuracy"]["native_global_pair"] == 1
    assert not result["advance_gate_passed"]
    assert result["primary"]["baseline_maximum_recomputed_each_draw"]
    assert result["primary"]["skipped_draws"] > 0
    assert all(m in result["balanced_accuracy"] for m in NATIVE_METHODS)


def test_patient_reference_is_balanced_within_finding():
    """Duplicating a beat within one patient must not give it extra patient weight."""
    f, s, _ = fixture_selection()
    first = banks_for_draw(f, s["support_draws"], SEEDS[0], "synthetic")["native_qrs"]
    duplicate = [*s["support_draws"], s["support_draws"][0]]
    second = banks_for_draw(f, duplicate, SEEDS[0], "synthetic")["native_qrs"]
    np.testing.assert_allclose(first.prototypes, second.prototypes)


def test_profile_admission_binds_runtime_and_actual_measurements():
    """Changed libraries or a forged summary flag cannot admit an incompatible profile."""
    gates = dict.fromkeys(
        (
            "finite_dense_shape",
            "official_mean_equivalence",
            "deterministic_repeat",
            "forward_budget",
            "rss_budget",
            "profile_budget",
            "all_parameters_frozen",
        ),
        True,
    )
    p = {
        "stage": "synthetic_pretrained_cpu_profile",
        "source_sha256": {"x": "a"},
        "runtime": {"torch": "2.6"},
        "checkpoint_sha256": "sha",
        "profile_gate_passed": True,
        "gates": gates,
        "dense_shape": [400, 768],
        "parameters": 85374720,
        "mean_absolute_max_error": 0,
        "repeat_absolute_max_error": 0,
        "forward_seconds": [1, 1],
        "official_pooled_seconds": 1,
        "seconds": 10,
        "peak_rss_bytes": 1000,
    }
    validate_profile(p, {"x": "a"}, {"torch": "2.6"}, "sha")
    with pytest.raises(ValueError, match="runtime"):
        validate_profile(p, {"x": "a"}, {"torch": "other"}, "sha")
    with pytest.raises(ValueError, match="measurements"):
        validate_profile({**p, "forward_seconds": [21, 1]}, {"x": "a"}, {"torch": "2.6"}, "sha")
    with pytest.raises(ValueError, match="gate"):
        validate_profile({**p, "gates": {**gates, "rss_budget": False}}, {"x": "a"}, {"torch": "2.6"}, "sha")
