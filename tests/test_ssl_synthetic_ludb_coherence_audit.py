import json

import numpy as np
import pytest

from ecg_experiment import ssl_synthetic_generated_quality as frozen_quality
from ecg_experiment import ssl_synthetic_ludb_coherence_audit as audit
from ecg_experiment import ssl_synthetic_ludb_coherence_audit_run as runner


def synthetic_signal():
    return np.tile(np.sin(np.arange(5000) * 0.018).astype(np.float32), (8, 1))


def synthetic_rows(cohort, count, offset):
    return [
        {
            "cohort": cohort,
            "index": index,
            "coherence": {"native_jaccard": 0.5 + offset, "shifted_jaccard": 0.2, "difference": 0.3 + offset},
            "activity_occupancy": [0.1] * 8,
            "empty_activity_masks": 0,
        }
        for index in range(count)
    ]


def test_robust_transform_preserves_power_of_two_scale():
    raw = synthetic_signal()
    first, metadata = audit.standardize(raw)
    second, _ = audit.standardize(raw * 8)
    np.testing.assert_array_equal(first, second)
    assert len(metadata["median"]) == len(metadata["q95_minus_q05"]) == 8


@pytest.mark.parametrize(
    "raw",
    [
        np.zeros((8, 5000), dtype=np.float32),
        np.ones((8, 5000), dtype=np.float64),
        np.zeros((12, 5000), dtype=np.float32),
    ],
)
def test_invalid_or_flat_representation_fails_entire_case(raw):
    with pytest.raises(ValueError, match="geometry|floor"):
        audit.standardize(raw)


def test_pair_order_and_empty_union_convention():
    masks = np.zeros((8, 5000), dtype=bool)
    assert audit.pair_jaccards(masks) == [0.0] * 28
    masks[:2, :10] = True
    assert audit.pair_jaccards(masks) == [1.0, *[0.0] * 27]


def test_matching_leads_and_fixed_cohort_seed():
    row, masks = audit.prepare(synthetic_signal(), 2, "ludb")
    assert row["control_seed"] == 2026106302
    assert row["record_id"] == 3
    assert row["coherence"]["native_jaccard"] == 1.0
    assert masks.shape == (8, 5000)
    assert masks.dtype == bool
    assert all(0 < value <= 0.1 for value in row["activity_occupancy"])
    np.testing.assert_array_equal(
        row["coherence"]["shifts"], np.random.default_rng(2026106302).integers(1, 5000, 8)
    )


def test_real_first_independent_bootstrap_order_and_no_pass_flag():
    generated = synthetic_rows("generated", 128, 0.1)
    real = synthetic_rows("ludb", 200, 0.0)
    for row in real:
        row["coherence"]["difference"] = row["index"] / 200
    for row in generated:
        row["coherence"]["difference"] = row["index"] / 128
    summary, draws = audit.summarize(generated, real)
    rng = np.random.default_rng(2026106401)
    first = np.asarray([row["coherence"]["difference"] for row in real])[rng.integers(0, 200, 200)].mean()
    second = np.asarray([row["coherence"]["difference"] for row in generated])[
        rng.integers(0, 128, 128)
    ].mean()
    np.testing.assert_array_equal(draws[0], [first, second, first - second])
    assert summary["primary_descriptive"]["seed"] == 2026106401
    assert "scientific_passed" not in summary
    assert summary["clinical_quality_claim"] is False


def test_missing_seed_rejected_without_dropping_case():
    with pytest.raises(ValueError, match="complete"):
        audit.summarize(synthetic_rows("generated", 127, 0.1), synthetic_rows("ludb", 200, 0.0))


def test_independent_original_descriptor_closes_frozen_arithmetic_exactly():
    raw = np.random.default_rng(2026107002).normal(size=(8, 5000)).astype(np.float32)
    standardized, parameters = frozen_quality.standardize(raw)
    row, masks = audit.prepare(raw, 3, "generated")
    assert row["standardization"] == parameters
    assert row["coherence"] == frozen_quality.coherence(standardized, 3)
    np.testing.assert_array_equal(masks, frozen_quality.activity_masks(standardized))


def cost_fixture(tmp_path):
    relative = "outputs/ssl_synthetic_localization/ludb_qualification_v1"
    prior_path = tmp_path / relative / "independent_audit_v1/prospective_ledger.json"
    prior_path.parent.mkdir(parents=True)
    prior = {
        "current_machine_stages": 24,
        "current_machine_cumulative_cpu_seconds": 500.0,
        "current_machine_cumulative_gpu_wall_seconds": 448.0,
    }
    prior_path.write_text(json.dumps(prior))
    digest = runner.file_sha256(prior_path)
    admission = {
        "qualification": {"directory": relative, "audit_ledger_sha256": digest},
        "prior_ledger": {
            "path": relative + "/independent_audit_v1/prospective_ledger.json",
            "sha256": digest,
            **prior,
        },
    }
    study = tmp_path / "study"
    study.mkdir()
    main = {"cpu_seconds": 1.0, "wall_seconds": 2.0, "peak_rss_bytes": 1000000}
    (study / "summary.json").write_text(json.dumps(main))
    ledger = {
        "receipt_sha256": runner.file_sha256(study / "summary.json"),
        "prior_ledger_sha256": digest,
        "current_machine_stages": 25,
        "cpu_seconds": 2.0,
        "wall_seconds": 3.0,
        "gpu_wall_seconds": 0.0,
        "current_machine_cumulative_cpu_seconds": 502.0,
        "current_machine_cumulative_gpu_wall_seconds": 448.0,
        "resource_gates": {"cpu": True},
        "output_bytes_before_terminal_ledger": (study / "summary.json").stat().st_size,
    }
    (study / "prospective_ledger.json").write_text(json.dumps(ledger))
    return study, main, admission, ledger


def test_immediate_qualified_audit_cost_is_not_reset(tmp_path):
    study, main, admission, _ = cost_fixture(tmp_path)
    costs = runner.accounting(tmp_path, study, main, admission)
    assert costs["cpu_through_main"] == 502.0
    assert costs["stages_through_main"] == 25


def test_duplicate_bridge_cpu_charge_is_rejected(tmp_path):
    study, main, admission, ledger = cost_fixture(tmp_path)
    ledger["current_machine_cumulative_cpu_seconds"] = 504.0
    (study / "prospective_ledger.json").write_text(json.dumps(ledger))
    with pytest.raises(ValueError, match="exactly-once"):
        runner.accounting(tmp_path, study, main, admission)


def test_last_real_record_never_indexes_smaller_generated_bank(tmp_path, monkeypatch):
    physical = np.arange(12 * 5000, dtype=np.float64).reshape(12, 5000)
    metadata = {"fs": 500}
    seen = []

    def read_waveform(path):
        seen.append(path)
        return physical, metadata

    monkeypatch.setattr(runner, "waveform", read_waveform)
    values, actual_metadata = runner.input_signal(np.empty((128, 8, 5000)), tmp_path, 199, "ludb")
    np.testing.assert_array_equal(values, physical[[0, 1, 6, 7, 8, 9, 10, 11]].astype(np.float32))
    assert values.dtype == np.float32
    assert actual_metadata == metadata
    assert seen == [tmp_path / "200"]
