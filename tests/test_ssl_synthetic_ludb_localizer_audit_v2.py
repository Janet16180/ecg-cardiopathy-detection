"""Artificial terminal-failure accounting tests for the pre-execution runtime successor."""

import json
import time

from ecg_experiment import ssl_synthetic_ludb_localizer_audit_v2 as audit


def costs() -> dict:
    return {
        "prior_ledger_sha256": "actual31",
        "stages_through_main": 31,
        "cpu_through_main": 480.0,
        "gpu_through_main": 454.0,
    }


def test_terminal_success_adds_actual_cost_once(tmp_path) -> None:
    clock = {"cpu": time.process_time(), "wall": time.perf_counter()}
    gates = audit._terminal(tmp_path, clock, costs(), {"all_passed": True}, {})
    ledger = json.loads((tmp_path / "prospective_ledger.json").read_text())
    assert all(gates.values())
    assert ledger["current_machine_stages"] == 32
    assert ledger["current_machine_cumulative_cpu_seconds"] == 480.0 + ledger["cpu_seconds"]
    assert ledger["current_machine_cumulative_gpu_wall_seconds"] == 454.0
    assert ledger["technical_passed"] is True


def test_hash_failure_retains_separate_receipt_and_actual_ledger(tmp_path, monkeypatch) -> None:
    from ecg_experiment import synthetic_audit_io

    original = synthetic_audit_io.file_sha256

    def fail_primary(path, *args):
        if path.name == "summary.json":
            raise TimeoutError("artificial receipt hash timeout")
        return original(path, *args)

    monkeypatch.setattr(synthetic_audit_io, "file_sha256", fail_primary)
    clock = {"cpu": time.process_time(), "wall": time.perf_counter()}
    gates = audit._terminal(tmp_path, clock, costs(), {"all_passed": True}, {})
    summary = json.loads((tmp_path / "summary.json").read_text())
    failure = json.loads((tmp_path / "finalization_failure.json").read_text())
    ledger = json.loads((tmp_path / "prospective_ledger.json").read_text())
    assert summary["all_passed"] is True
    assert failure["all_passed"] is False
    assert ledger["receipt"] == "finalization_failure.json"
    assert ledger["output_bytes_before_terminal_ledger"] is None
    assert ledger["technical_passed"] is False
    assert gates["output"] is False
    assert ledger["current_machine_cumulative_cpu_seconds"] == 480.0 + ledger["cpu_seconds"]
    assert ledger["current_machine_cumulative_gpu_wall_seconds"] == 454.0
