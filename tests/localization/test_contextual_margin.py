"""Scientific invariants for exact contextual signed contributions and interventions."""

import hashlib
from types import SimpleNamespace

import numpy as np
import pytest

from ecg_experiment.localization import contribution_inputs, contribution_study
from ecg_experiment.localization.contextual_margin import (
    bank_consistency,
    decomposition,
    intervention_summary,
    mass_summary,
    occupancy,
    rank_correlation,
    removed_margin,
)
from ecg_experiment.localization.contribution_study import (
    coordinate_artifact,
    evaluate_target,
    patient_summaries,
)
from ecg_experiment.native_reference_ntt05 import NativeTokens, SignalRegion, lead_patch_tokens
from ecg_experiment.same_wave_ntt04 import SEEDS


def fixture():
    tokens = lead_patch_tokens(
        np.array([[[2, 0], [0, 1], [-1, 2]]], dtype=np.float32),
        ("V1",),
        100,
        100,
        300,
        "synthetic",
        "full context",
        "fictional-record",
    )
    regions = [SignalRegion(0, 1.5, ("V1",)), SignalRegion(1, 2.5, ("V1",))]
    return tokens, regions


def test_overlap_and_two_normalizations_reconstruct_hand_computed_margin():
    tokens, regions = fixture()
    result = decomposition(tokens, regions, np.eye(2))
    weights = np.array([[2 / 3, 1 / 3, 0], [0, 2 / 3, 1 / 3]])
    means = np.array([[4 / 3, 1 / 3], [-1 / 3, 4 / 3]])
    beats = means / np.linalg.norm(means, axis=1)[:, None]
    vector = beats.sum(axis=0) / np.linalg.norm(beats.sum(axis=0))
    np.testing.assert_allclose(result["weights"], weights, atol=1e-15)
    np.testing.assert_allclose(result["beat_vectors"], beats, atol=1e-15)
    np.testing.assert_allclose(result["patient_vector"], vector, atol=1e-15)
    assert result["winner"] == 1
    assert result["runner"] == 0
    expected = vector[1] - vector[0]
    assert result["margin"] == pytest.approx(expected, abs=1e-15)
    assert result["beat_contributions"].sum() == pytest.approx(expected, abs=1e-15)
    assert result["token_contributions"].sum() == pytest.approx(expected, abs=1e-15)
    np.testing.assert_allclose(result["beat_token_contributions"].sum(axis=1), result["beat_contributions"])
    assert result["token_contributions"].shape == (3,)
    assert np.count_nonzero(result["weights"][:, 1]) == 2


def test_fractional_occupancy_keeps_active_tokens_even_when_contributions_are_zero():
    tokens, regions = fixture()
    result = decomposition(tokens, regions, np.array([[1, 0], [1, 0]]))
    assert result["winner"] == 0
    assert result["runner"] == 1
    assert result["active_tokens"].sum() == 3
    assert np.all(result["token_contributions"] == 0)
    mass = mass_summary(result["token_contributions"][result["active_tokens"]])
    assert mass["count"] == 3
    assert mass["cancellation"] is None
    assert mass["effective_count"] is None
    assert set(mass["top_absolute_shares"].values()) == {0.0}


def test_signed_mass_uses_absolute_mass_when_margin_nearly_cancels():
    mass = mass_summary(np.array([2, -1.99]))
    assert mass["absolute_mass"] == pytest.approx(3.99)
    assert mass["cancellation"] == pytest.approx(1 - 0.01 / 3.99)
    assert 0 <= mass["effective_fraction"] <= 1
    assert mass["top_absolute_shares"]["0.1"] == pytest.approx(2 / 3.99)


@pytest.mark.parametrize("prototypes", [np.array([[2, 0], [0, 1]]), np.ones((2, 3)), np.ones((1, 2))])
def test_nonunit_or_mismatched_prototypes_are_rejected(prototypes):
    tokens, regions = fixture()
    with pytest.raises(ValueError, match="unit prototypes"):
        decomposition(tokens, regions, prototypes)


def test_mixed_lead_tokens_cannot_be_falsely_sliced():
    tokens = NativeTokens(
        np.ones((1, 2)), np.array([[0, 1]]), (("V1", "V6"),), 1, "synthetic", "full context", "fictional"
    )
    with pytest.raises(ValueError, match="mixed-lead"):
        occupancy(tokens, [SignalRegion(0, 1, ("V1",))])


def test_representation_deletion_renormalizes_instead_of_subtracting_contribution():
    beats = np.eye(2)
    delta = np.array([1, -1])
    assert removed_margin(beats, delta, np.array([0])) == pytest.approx(-1)
    assert removed_margin(beats, delta, np.array([1])) == pytest.approx(1)
    assert removed_margin(np.array([[1, 0], [1, 0], [-1, 0]]), delta, np.array([0])) is None


@pytest.mark.parametrize("removed", [np.array([0, 0]), np.array([2]), np.array([-1]), np.array([0.0])])
def test_invalid_removed_indices_are_rejected(removed):
    with pytest.raises(ValueError, match="removed indices"):
        removed_margin(np.eye(2), np.array([1, -1]), removed)


def test_constant_or_undefined_rank_comparisons_remain_null():
    assert rank_correlation(np.ones(3), np.arange(3)) is None
    assert rank_correlation(np.arange(3), np.array([np.nan, np.nan, 1])) is None
    assert rank_correlation(np.array([3, 1, 2]), np.array([1, 3, 2])) == pytest.approx(-1)


def test_bank_pairs_require_identical_ordered_contrast():
    rows = [
        {"winner": 0, "runner": 1, "beat_contributions": np.array([1, 2, -1])},
        {"winner": 0, "runner": 1, "beat_contributions": np.array([3, 1, -1])},
        {"winner": 1, "runner": 0, "beat_contributions": np.array([1, 2, -1])},
    ]
    result = bank_consistency(rows)
    assert len(result["pairs"]) == 3
    assert result["same_contrast"] == 1
    assert result["pairs"][0]["top_quarter_jaccard"] == 0
    assert "top_quarter_jaccard" not in result["pairs"][1]


def test_empty_positive_top_sets_are_undefined_not_perfect_agreement():
    row = {"winner": 0, "runner": 1, "beat_contributions": np.zeros(3)}
    assert bank_consistency([row, row])["pairs"][0]["top_quarter_jaccard"] is None


def test_random_subsets_use_frozen_decimal_fraction_seed():
    result = {
        "beat_vectors": np.array([[1, 0], [0, 1], [-0.6, 0.8]]),
        "delta": np.array([1, -1]),
        "margin": 0.1,
        "beat_contributions": np.array([0.3, -0.1, -0.1]),
    }
    actual = intervention_summary(result, 17, 20261018)
    digest = hashlib.sha256(b"ntt08:20261020:17:20261018:0.10").hexdigest()
    rng = np.random.default_rng(int(digest, 16))
    expected = [rng.choice(3, 1, replace=False).tolist() for _ in range(100)]
    assert actual["fractions"]["0.1"]["random_subsets"] == expected
    assert actual == intervention_summary(result, 17, 20261018)


def test_population_summary_averages_patients_before_bank_observations():
    rows = []
    for patient, values in ((1, [1.0, 3.0, None]), (2, [9.0, 9.0, 9.0])):
        rows.extend(
            {
                "patient_id": patient,
                "metrics": {"example": value},
                "intervention": {"single_undefined_drops": 0, "fractions": {}},
            }
            for value in values
        )
    consistency = [{"same_contrast": 0, "pairs": []} for _ in range(2)]
    result = patient_summaries(rows, consistency)["patient_first_metrics"]["example"]
    assert result["mean"] == 5.5
    assert result["count"] == 2
    assert result["undefined_bank_evaluations"] == 1


def test_original_runner_tie_is_bound_to_saved_score_order(monkeypatch):
    monkeypatch.setattr(contribution_study, "cached_tokens", lambda *args: None)
    monkeypatch.setattr(contribution_study, "decomposition", lambda *args: {"winner": 0, "runner": 2})
    data = {
        "tokens": {17: None},
        "coordinates": {"17": {"qrs": []}},
        "banks": {SEEDS[0]: SimpleNamespace(prototypes=np.eye(3))},
        "predictions": [{"ecg_id": 17, "seed": SEEDS[0], "scores": {"native_qrs": [1, 0.5, 0.5]}}],
    }
    with pytest.raises(ValueError, match="winner/runner contrast"):
        evaluate_target(data, {"ecg_id": 17}, 0, {})


def test_changed_identity_metadata_is_rejected_before_read(monkeypatch, tmp_path):
    import json

    directory = tmp_path / "native"
    directory.mkdir()
    (directory / "selection_local.json").write_text("{}")
    manifest = tmp_path / "outputs/ntt04_same_wave/reservation_manifest_local.json"
    manifest.parent.mkdir(parents=True)
    manifest.write_text(json.dumps({"records": []}))
    monkeypatch.setattr(contribution_inputs, "ROOT", tmp_path)
    monkeypatch.setattr(contribution_inputs, "to_stored", lambda path: str(path.relative_to(tmp_path)))
    monkeypatch.setattr(contribution_inputs, "sha256_file", lambda path: "changed")
    monkeypatch.setattr(
        contribution_inputs.pd, "read_csv", lambda *args, **kwargs: pytest.fail("read before hash gate")
    )
    with pytest.raises(ValueError, match="identity metadata changed"):
        contribution_inputs.admitted_selection(directory, {})


def test_private_coordinate_artifact_binds_every_unique_lead_time_anchor():
    qrs = [{"start": 1, "stop": 1.3, "leads": ["V1", "V6"]}]
    data = {
        "selection": {"targets": [{"ecg_id": 17}]},
        "coordinates": {"17": {"qrs": qrs}},
        "tokens": {17: np.ones((400, 768), dtype=np.float32)},
    }
    row = coordinate_artifact(data)[0]
    assert row["target_index"] == 0
    assert row["qrs_regions"] == qrs
    assert len(row["token_intervals_seconds"]) == len(row["token_leads"]) == 400
    assert row["token_intervals_seconds"][49] == [9.8, 10]
    assert row["token_leads"][100] == ["V1"]
    assert row["token_leads"][399] == ["V6"]
    assert row["bank_order"] == list(SEEDS)
