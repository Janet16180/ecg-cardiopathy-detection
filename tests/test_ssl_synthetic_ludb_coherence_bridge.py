"""Artificial checks of frozen activity geometry, control seeds and independent cohort uncertainty."""

import numpy as np
import pytest

from ecg_experiment import ssl_synthetic_generated_quality as old
from ecg_experiment import ssl_synthetic_ludb_coherence_bridge as bridge


def test_generated_descriptor_reproduces_frozen_source_exactly():
    raw = np.random.default_rng(17).normal(size=(8, 5000)).astype(np.float32)
    row, masks = bridge.prepare(raw, 3, "generated")
    values, parameters = old.standardize(raw)
    assert row["standardization"] == parameters
    assert row["coherence"] == old.coherence(values, 3)
    np.testing.assert_array_equal(masks, old.activity_masks(values))
    assert len(row["native_pair_jaccards"]) == 28
    assert np.mean(row["native_pair_jaccards"]) == old.mean_jaccard(masks)


def test_real_control_seed_changes_only_phase_reference():
    raw = np.random.default_rng(18).normal(size=(8, 5000)).astype(np.float32)
    generated, first = bridge.prepare(raw, 0, "generated")
    real, second = bridge.prepare(raw, 0, "ludb")
    np.testing.assert_array_equal(first, second)
    assert generated["coherence"]["native_jaccard"] == real["coherence"]["native_jaccard"]
    assert real["coherence"]["shifts"] == np.random.default_rng(2026106300).integers(1, 5000, 8).tolist()
    assert generated["coherence"]["shifts"] != real["coherence"]["shifts"]
    assert real["record_id"] == 1


def test_flat_lead_fails_entire_representation_without_exclusion():
    raw = np.random.default_rng(19).normal(size=(8, 5000)).astype(np.float32)
    raw[2] = 0
    with pytest.raises(ValueError, match="small robust range"):
        bridge.prepare(raw, 0, "ludb")


def test_empty_union_is_retained_as_zero():
    masks = np.zeros((8, 5000), dtype=bool)
    assert bridge.pair_jaccards(masks) == [0.] * 28


def cohort(count, name, offset):
    rows = []
    for index in range(count):
        difference = offset + index / (count * 4)
        rows.append({"index": index, "cohort": name,
                     "coherence": {"native_jaccard": difference + .1,
                                   "shifted_jaccard": .1, "difference": difference},
                     "activity_occupancy": [.1] * 8, "empty_activity_masks": 0})
    return rows


def test_independent_bootstrap_draws_real_records_first_then_generator_seeds():
    generated, real = cohort(128, "generated", .1), cohort(200, "ludb", .3)
    statistics, draws = bridge.summarize(generated, real)
    rng = np.random.default_rng(2026106401)
    real_indices = rng.integers(0, 200, 200)
    generated_indices = rng.integers(0, 128, 128)
    first = np.mean([real[index]["coherence"]["difference"] for index in real_indices])
    second = np.mean([generated[index]["coherence"]["difference"] for index in generated_indices])
    np.testing.assert_array_equal(draws[0], [first, second, first - second])
    np.testing.assert_allclose(draws[:, 2], draws[:, 0] - draws[:, 1], atol=0, rtol=0)
    assert draws.shape == (2000, 3)
    assert statistics["primary_descriptive"]["mean"] > 0
    assert "scientific_passed" not in statistics
    assert not statistics["clinical_quality_claim"]


def test_missing_record_is_not_silently_removed():
    with pytest.raises(ValueError, match="complete sorted"):
        bridge.summarize(cohort(127, "generated", .1), cohort(200, "ludb", .3))


def test_distributions_include_every_record_and_fixed_edges():
    result = bridge.distributions(cohort(128, "generated", .1))
    assert result["records"] == 128
    for variable, statistics in result["variables"].items():
        assert sum(statistics["histogram_counts"]) == 128
        assert len(statistics["histogram_edges"]) == 21
        assert statistics["histogram_edges"][0] == (-1 if variable == "difference" else 0)
        assert statistics["histogram_edges"][-1] == 1


def test_failed_qualification_blocks_before_numerical_rows(tmp_path, monkeypatch):
    from ecg_experiment import ssl_synthetic_ludb_coherence_run as runner

    monkeypatch.setattr(runner, "from_stored", lambda name: tmp_path)
    monkeypatch.setattr(runner, "sha256_file", lambda path: "fixed")
    failed = {"technical_passed_before_independent_audit": True,
              "scientific_passed_before_independent_audit": False}
    monkeypatch.setattr(runner, "read_json", lambda path: failed)
    admission = {"qualification": {"directory": "irrelevant", "summary_sha256": "fixed",
                                    "ledger_sha256": "fixed", "audit_summary_sha256": "fixed",
                                    "audit_ledger_sha256": "fixed"}}
    with pytest.raises(ValueError, match="qualification PASS"):
        runner.qualified_predecessor(admission, tmp_path)
    assert not list(tmp_path.iterdir())
