"""Binding, catalog and retained-failure accounting tests for the independent audit."""

import hashlib
import json
import time
from pathlib import Path

import pytest

from ecg_experiment import ssl_synthetic_mixed_background_audit_run as audit


def write_json(path: Path, value: dict) -> str:
    """Write an artificial local object and return its exact byte identity."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value) + "\n")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def complete_admission() -> dict:
    """Declare all mandatory independent files for source-union failure tests."""
    names = audit.SHARED_SOURCES | {
        audit.RUNNER,
        audit.CHECKER,
        audit.PROTOCOL,
        audit.TESTS,
        audit.CORE,
        audit.INPUTS,
        audit.CORE_TESTS,
    }
    return {"source_sha256": dict.fromkeys(names, "a" * 64)}


def test_unbound_independent_helper_and_frozen_override_are_rejected() -> None:
    """A runner hash cannot stand in for its independent raw decoder or silently replace main math."""
    declaration = complete_admission()
    original = {"source_sha256": {audit.CORE: "a" * 64}}
    assert audit.source_union(original, declaration)[audit.CORE] == "a" * 64
    del declaration["source_sha256"]["ecg_experiment/ssl_synthetic_randomized_saved_audit.py"]
    with pytest.raises(ValueError, match="imported independent helpers"):
        audit.source_union(original, declaration)
    declaration = complete_admission()
    declaration["source_sha256"][audit.CORE] = "b" * 64
    with pytest.raises(ValueError, match="override"):
        audit.source_union(original, declaration)


def test_main_catalog_excludes_readonly_raw_links_and_audit_snapshot_subtree(tmp_path: Path) -> None:
    """Sanitized .dat links stay separate from the main's immutable ordinary-file catalog."""
    study = tmp_path / "study"
    study.mkdir()
    (study / "example.txt").write_text("bound ordinary artifact")
    (study / "summary.json").write_text("{}")
    (study / "prospective_ledger.json").write_text("{}")
    raw = tmp_path / "raw.dat"
    raw.write_bytes(b"read-only example")
    (study / "sanitized_headers").mkdir()
    (study / "sanitized_headers/source.dat").symlink_to(raw)
    snapshots = study / "independent_audit_v1/source_snapshot"
    snapshots.mkdir(parents=True)
    (snapshots / "new.py").write_text("# independent source\n")
    main = {"source_sha256": {}, "output_sha256": {"example.txt": audit.file_sha256(study / "example.txt")}}
    assert audit.identities(tmp_path, tmp_path, study, main, {"input_sha256": {}}, lambda: None) == 1
    (study / "unmanifested.txt").write_text("unexpected output")
    with pytest.raises(ValueError, match="membership"):
        audit.identities(tmp_path, tmp_path, study, main, {"input_sha256": {}}, lambda: None)


def test_final_hash_failure_retains_failed_receipt_and_own_actual_cost(tmp_path: Path, monkeypatch) -> None:
    """A late timeout leaves failure evidence and charges the verified immediate main once."""
    real_hash = audit.file_sha256

    def interrupted_hash(path: Path, *args) -> str:
        if path.name == "summary.json":
            raise TimeoutError("artificial terminal hash timeout")
        return real_hash(path, *args)

    monkeypatch.setattr(audit, "file_sha256", interrupted_hash)
    costs = {
        "prior_ledger_sha256": "a" * 64,
        "stages_through_main": 38,
        "cpu_through_main": 1100.0,
        "gpu_through_main": 600.0,
    }
    clock = {"cpu": time.process_time(), "wall": time.perf_counter()}
    gates = audit.terminal(tmp_path, {"all_passed": True, "mismatch_count": 0}, clock, costs, {})
    ledger = json.loads((tmp_path / "prospective_ledger.json").read_text())
    failure = json.loads((tmp_path / "finalization_failure.json").read_text())
    assert (tmp_path / "summary.json").exists()
    assert not failure["all_passed"]
    assert failure["error_type"] == "TimeoutError"
    assert not gates["output"]
    assert not ledger["technical_passed"]
    assert ledger["current_machine_stages"] == 39
    assert ledger["prior_accounting_verified"]
    assert ledger["current_machine_cumulative_cpu_seconds"] == 1100.0 + ledger["cpu_seconds"]
    assert ledger["current_machine_cumulative_gpu_wall_seconds"] == 600.0
    assert ledger["sha256"] == real_hash(tmp_path / "finalization_failure.json")


def test_actual_cost_lineage_rejects_double_counting(tmp_path: Path, monkeypatch) -> None:
    """A Boolean PASS cannot hide duplicated main cost in a cumulative ledger."""
    prior_dir = tmp_path / "outputs/ssl_synthetic_localization/background_editor_transfer_figures_v1"
    prior_summary = write_json(prior_dir / "summary.json", {"status": "PASS"})
    prior = {
        "current_machine_stages": 37,
        "technical_passed": True,
        "resource_gates": {"cpu": True},
        "current_machine_cumulative_cpu_seconds": 1076.7763161960002,
        "current_machine_cumulative_gpu_wall_seconds": 524.1993697541766,
    }
    prior_sha = write_json(prior_dir / "prospective_ledger.json", prior)
    monkeypatch.setattr(audit, "PRIOR37_SHA", prior_sha)
    monkeypatch.setattr(audit, "PRIOR37_RECEIPT", prior_summary)
    study = tmp_path / "study"
    main = {"status": "PASS", "technical_passed_before_independent_audit": True}
    summary_sha = write_json(study / "summary.json", main)
    size = (study / "summary.json").stat().st_size
    ledger = {
        "prior_ledger_sha256": prior_sha,
        "prior_accounting_verified": True,
        "current_machine_stages": 38,
        "technical_passed": True,
        "resource_gates": {"cpu": True},
        "receipt_sha256": summary_sha,
        "cpu_seconds": 10.0,
        "wall_seconds": 12.0,
        "gpu_wall_seconds": 3.0,
        "peak_rss_bytes": 1000,
        "output_bytes_before_terminal_ledger": size,
        "current_machine_cumulative_cpu_seconds": prior["current_machine_cumulative_cpu_seconds"] + 10.0,
        "current_machine_cumulative_gpu_wall_seconds": prior["current_machine_cumulative_gpu_wall_seconds"]
        + 3.0,
    }
    write_json(study / "prospective_ledger.json", ledger)
    assert audit.accounting(tmp_path, study, main)["stages_through_main"] == 38
    ledger["current_machine_cumulative_cpu_seconds"] += 10.0
    write_json(study / "prospective_ledger.json", ledger)
    with pytest.raises(ValueError, match="additive"):
        audit.accounting(tmp_path, study, main)
