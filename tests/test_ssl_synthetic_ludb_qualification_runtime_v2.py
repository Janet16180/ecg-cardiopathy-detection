"""Artificial preflight checks prevent another missing-roster or silent scientific-change attempt."""

import copy

import pytest

from ecg_experiment import ssl_synthetic_ludb_run_v2 as runtime


def test_complete_roster_includes_the_previously_omitted_saved_audit():
    sources = dict.fromkeys(runtime.REQUIRED, "fixed")
    runtime.validate_roster(sources)
    del sources["ecg_experiment/ssl_synthetic_pretrained_saved_audit.py"]
    with pytest.raises(ValueError, match="ssl_synthetic_pretrained_saved_audit.py"):
        runtime.validate_roster(sources)
    assert len(runtime.BASE_REQUIRED) == 27
    assert len(runtime.REQUIRED) == 32


def test_runtime_allows_only_study_and_output_overrides():
    base = {"study": "old", "output_directory": "old", "proxy": {"smoothing_samples": 20},
            "endpoint": {"bootstrap_seed": 2026106101, "minimum_primary_gain": .05}}
    successor = {**base, "study": "ssl_synthetic_ludb_qualification_v2",
                 "output_directory": "outputs/ssl_synthetic_localization/ludb_qualification_v2"}
    runtime.validate_semantics(base, successor)
    altered = copy.deepcopy(successor)
    altered["proxy"]["smoothing_samples"] = 21
    with pytest.raises(ValueError, match="scientific semantics"):
        runtime.validate_semantics(base, altered)
    altered = copy.deepcopy(successor)
    altered["endpoint"]["minimum_primary_gain"] = .04
    with pytest.raises(ValueError, match="scientific semantics"):
        runtime.validate_semantics(base, altered)


def test_missing_helper_preflight_stops_before_hashing_or_creating_outputs(tmp_path, monkeypatch):
    import json

    base = {"study": "old", "output_directory": "old", "endpoint": {"minimum_primary_gain": .05}}
    successor = {**base, "study": "ssl_synthetic_ludb_qualification_v2",
                 "output_directory": "outputs/ssl_synthetic_localization/ludb_qualification_v2"}
    sources = dict.fromkeys(runtime.REQUIRED, "fixed")
    del sources["ecg_experiment/ssl_synthetic_pretrained_saved_audit.py"]
    metadata = {runtime.original.CONFIG: base, runtime.CONFIG: successor,
                "configs/artificial_admission.json": {"source_sha256": sources}}
    for name, values in metadata.items():
        target = tmp_path / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(values))
    monkeypatch.setattr(runtime, "from_stored", lambda name: tmp_path / name)

    def forbidden_hash(path):
        raise AssertionError("Incomplete source rosters must stop before any hash or data operation")

    monkeypatch.setattr(runtime, "sha256_file", forbidden_hash)
    with pytest.raises(ValueError, match="ssl_synthetic_pretrained_saved_audit.py"):
        runtime.preflight("configs/artificial_admission.json")
    assert not (tmp_path / "outputs").exists()


def reconciliation_inputs():
    record = {"retained_failed_main_receipt": {"sha256": "failure"},
              "verified_acquisition_ledger": {"sha256": "acquisition"}, "current_machine_stages": 23,
              "current_machine_cumulative_cpu_seconds": 11.,
              "current_machine_cumulative_gpu_wall_seconds": 20., "failed_stage_cpu_seconds_counted_once": 1.,
              "failed_stage_wall_seconds": 2., "scientific_scores_computed": False, "decoded_LUDB_records": 0}
    acquisition = {"current_machine_stages": 22, "current_machine_cumulative_cpu_seconds": 10.,
                   "current_machine_cumulative_gpu_wall_seconds": 20., "resource_gates": {"cpu": True}}
    failed = {"receipt": "failure.json", "receipt_sha256": "failure", "cpu_seconds": 1.,
              "wall_seconds": 2., "gpu_wall_seconds": 0, "resource_gates": {"cpu": True},
              "prior_accounting_verified": False, "prior_ledger_sha256": None,
              "current_machine_cumulative_cpu_seconds": None,
              "current_machine_cumulative_gpu_wall_seconds": None}
    failure = {"error_type": "ValueError",
               "error": "Source admission omits required scientific or runtime sources",
               "scientific_result_valid": False}
    sources = dict.fromkeys(runtime.BASE_REQUIRED, "fixed")
    del sources["ecg_experiment/ssl_synthetic_pretrained_saved_audit.py"]
    old_admission = {"source_sha256": sources, "prior_ledger": {"sha256": "acquisition"}}
    return record, acquisition, failed, failure, old_admission


def test_reconciliation_rejects_double_counted_failure_and_changed_nulls():
    record, acquisition, failed, failure, old_admission = reconciliation_inputs()
    runtime.validate_reconciliation(record, acquisition, failed, failure, old_admission)
    changed = {**record, "current_machine_cumulative_cpu_seconds": 12.}
    with pytest.raises(ValueError, match="exactly once"):
        runtime.validate_reconciliation(changed, acquisition, failed, failure, old_admission)
    changed = {**failed, "current_machine_cumulative_cpu_seconds": 11.}
    with pytest.raises(ValueError, match="null/unverified"):
        runtime.validate_reconciliation(record, acquisition, changed, failure, old_admission)
    changed = {**failed, "receipt_sha256": "tampered"}
    with pytest.raises(ValueError, match="receipt"):
        runtime.validate_reconciliation(record, acquisition, changed, failure, old_admission)
