"""Artificial contracts for a resource-only retry with preserved failed history."""

import copy
import json
from pathlib import Path

import pytest

from ecg_experiment import ROOT
from ecg_experiment import ssl_synthetic_edb_acquisition_v2 as recovery
from ecg_experiment.files import sha256_file


def configurations() -> tuple[dict, dict]:
    """Read committed design files only, without accessing any real acquisition output."""
    return (json.loads((ROOT / recovery.frozen.CONFIG).read_text()),
            json.loads((ROOT / recovery.RECOVERY).read_text()))


def test_resource_overlay_changes_only_rss_and_keeps_original_immutable() -> None:
    """A resource retry cannot silently change the request or annotation contract."""
    base, amendment = configurations()
    before = copy.deepcopy(base)
    result = recovery.operating_config(base, amendment)
    assert base == before
    result["resources"]["acquisition"]["rss_bytes"] = 1024**3
    assert result == base
    bad = copy.deepcopy(amendment)
    bad["resources"]["wall_seconds"] += 1
    with pytest.raises(ValueError, match="other than RSS"):
        recovery.operating_config(base, bad)


def artificial_history(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[dict, dict]:
    """Create a synthetic pre-HTTP failed ledger with the actual declared cost constants."""
    base, amendment = configurations()
    original = tmp_path / amendment["original_admission"]
    original.parent.mkdir(parents=True)
    original.write_text("{}\n")
    amendment["original_admission_sha256"] = sha256_file(original)
    previous = {
        "current_machine_stages": 40,
        "current_machine_cumulative_cpu_seconds": base["accounting"]["initial_cpu_seconds"],
        "current_machine_cumulative_gpu_wall_seconds": base["accounting"]["initial_gpu_wall_seconds"],
    }
    monkeypatch.setattr(recovery.frozen, "verify_sources", lambda *args: None)
    monkeypatch.setattr(recovery.frozen, "verify_prior", lambda *args: previous)
    (tmp_path / recovery.frozen.CONFIG).write_text(json.dumps(base))
    directory = tmp_path / amendment["failed_directory"]
    directory.mkdir(parents=True)
    summary = {"status": "failed", "error_type": "MemoryError",
               "error": "Acquisition RSS ceiling exceeded"}
    summary.update(dict.fromkeys(("requests", "received_payload_bytes", "new_waveform_decodes",
                                 "new_annotation_decodes", "gpu_operations", "new_model_scores",
                                 "new_training_updates"), 0))
    (directory / "summary.json").write_text(json.dumps(summary))
    amendment["failed_summary_sha256"] = sha256_file(directory / "summary.json")
    ledger = {
        "sha256": amendment["failed_summary_sha256"], "prior_accounting_verified": True,
        "technical_passed": False,
        "resource_gates": {"cpu": True, "wall": True, "rss": False, "output": True, "network": True},
        "prior_ledger_sha256": base["accounting"]["initial_ledger_sha256"],
        "cpu_seconds": amendment["failed_cpu_seconds"],
        "wall_seconds": amendment["failed_wall_seconds"],
        "peak_rss_bytes": amendment["failed_peak_rss_bytes"], "gpu_wall_seconds": 0,
        "current_machine_stages": amendment["failed_stage"],
        "current_machine_cumulative_cpu_seconds": amendment["prior_cpu_seconds"],
        "current_machine_cumulative_gpu_wall_seconds": amendment["prior_gpu_wall_seconds"],
    }
    (directory / "prospective_ledger.json").write_text(json.dumps(ledger))
    amendment["failed_ledger_sha256"] = sha256_file(directory / "prospective_ledger.json")
    return amendment, ledger


def test_failed_cost_is_a_valid_once_only_prior_without_invented_snapshots(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A failed resource gate retains its cost; creating a fake snapshot is rejected."""
    amendment, ledger = artificial_history(tmp_path, monkeypatch)
    assert recovery.latest_prior(tmp_path, amendment) == ledger
    directory = tmp_path / amendment["failed_directory"]
    (directory / "source_snapshot").mkdir()
    with pytest.raises(ValueError, match="membership"):
        recovery.latest_prior(tmp_path, amendment)


def test_failed_cost_cannot_be_reset_or_counted_twice(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Even hash-consistent metadata cannot alter the declared exact cumulative cost."""
    amendment, ledger = artificial_history(tmp_path, monkeypatch)
    directory = tmp_path / amendment["failed_directory"]
    ledger["current_machine_cumulative_cpu_seconds"] += amendment["failed_cpu_seconds"]
    (directory / "prospective_ledger.json").write_text(json.dumps(ledger))
    amendment["failed_ledger_sha256"] = sha256_file(directory / "prospective_ledger.json")
    with pytest.raises(ValueError, match="exactly once"):
        recovery.latest_prior(tmp_path, amendment)


def test_missing_recovery_sources_rejected_before_any_http(tmp_path: Path) -> None:
    """A separate protocol cannot run with a silently omitted successor executable."""
    admission = tmp_path / "admission.json"
    admission.write_text(json.dumps({"source_sha256": {}}))
    with pytest.raises(ValueError, match="Incomplete"):
        recovery.preflight(tmp_path, admission)
