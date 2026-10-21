"""Artificial conditional-bridge integrity checks; no released records are opened."""

import json

import numpy as np
import pytest

from ecg_experiment import ssl_synthetic_ludb_coherence_run_v2 as runtime
from ecg_experiment.files import sha256_file


def fixture_predecessor(tmp_path, monkeypatch):
    """Make200 artificial role rows and auditable metadata with exact endpoint closure."""
    rows = []
    for identifier in range(1, 201):
        row = {"record_id": identifier,
               "role": "parser_qualification" if identifier % 5 == 0 else "measurement",
               "leads": [{"span_samples": 3000, "N_samples": [100, 200], "N_mean_midrank": .8,
                           "mean_shift_N_midrank": .4, "gain": .4, "constant_score": False}] * 12}
        if identifier % 5:
            row.update(N_mean_midrank=.8, mean_shift_N_midrank=.4, shift_N_mean_midrank=[.4] * 6, gain=.4)
        rows.append(row)
    parent, output = tmp_path / "qualification", tmp_path / "bridge"
    parent.mkdir()
    output.mkdir()
    (parent / "independent_audit_v1").mkdir()
    (parent / "independent_audit_v2").mkdir()
    (parent / "record_rows.jsonl").write_text("\n".join(json.dumps(row) for row in rows) + "\n")
    statistics = runtime.peak.aggregate(rows)
    np.save(parent / "bootstrap_gains.npy", np.asarray(statistics.pop("bootstrap_gains")))
    assets = {f"{identifier}.{extension}": "artificial" for identifier in range(1, 201)
              for extension in ["dat", "hea"]}
    runtime.shared.write_new(parent / "release_file_identities.json", assets)
    main = {"technical_passed_before_independent_audit": True,
            "scientific_passed_before_independent_audit": True, "statistics": statistics,
            "output_sha256": {name: sha256_file(parent / name)
                              for name in ["record_rows.jsonl", "bootstrap_gains.npy",
                                           "release_file_identities.json"]}}
    runtime.shared.write_new(parent / "summary.json", main)
    runtime.shared.write_new(parent / "independent_audit_v2/summary.json",
                             {"all_passed": True, "mismatch_count": 0, "scientific_passed": True})
    runtime.shared.write_new(parent / "independent_audit_v1/summary.json",
                             {"all_passed": False, "error_type": "KeyError",
                              "error": "'current_machine_stages'"})
    failed = {"cpu_seconds": 1, "gpu_wall_seconds": 0,
              "sha256": sha256_file(parent / "independent_audit_v1/summary.json"),
              "resource_gates": dict.fromkeys(["cpu", "wall", "rss", "output"], True),
              **dict.fromkeys(["prior_ledger_sha256", "current_machine_stages",
                              "current_machine_cumulative_cpu_seconds",
                              "current_machine_cumulative_gpu_wall_seconds"])}
    runtime.shared.write_new(parent / "independent_audit_v1/prospective_ledger.json", failed)
    monkeypatch.setattr(runtime, "from_stored", lambda name: tmp_path / name)
    admission, config = rebind_chain(parent)
    admission["asset_sha256"] = assets
    return admission, parent, output, config


def rebind_chain(parent):
    """Bind the artificial main26, retained failed27, setup27 and successor28 chain."""
    main_ledger = {"current_machine_stages": 26, "cpu_seconds": 1, "gpu_wall_seconds": 0,
                   "current_machine_cumulative_cpu_seconds": 100,
                   "current_machine_cumulative_gpu_wall_seconds": 10,
                   "resource_gates": dict.fromkeys(["cpu", "wall", "rss", "output"], True),
                   "receipt_sha256": sha256_file(parent / "summary.json")}
    ledger_path = parent / "prospective_ledger.json"
    ledger_path.write_text(json.dumps(main_ledger))
    record = {"current_machine_stages": 27, "current_machine_cumulative_cpu_seconds": 101,
              "current_machine_cumulative_gpu_wall_seconds": 10, "failed_stage_cpu_seconds_counted_once": 1,
              "creation_is_setup": True, "new_cost_stage_created": False,
              "original_failed_audit_nulls_preserved": True}
    for key, name in [("verified_main_accounting_ledger", "prospective_ledger.json"),
                      ("retained_main_summary", "summary.json"),
                      ("retained_failed_audit_ledger", "independent_audit_v1/prospective_ledger.json"),
                      ("retained_failed_audit_receipt", "independent_audit_v1/summary.json")]:
        record[key] = {"path": "qualification/" + name, "sha256": sha256_file(parent / name)}
    reconciliation = parent / "reconciled_prior.json"
    reconciliation.write_text(json.dumps(record))
    binding = {"path": "qualification/reconciled_prior.json", "sha256": sha256_file(reconciliation)}
    audit_ledger = {**main_ledger, "current_machine_stages": 28,
                    "current_machine_cumulative_cpu_seconds": 102,
                    "sha256": sha256_file(parent / "independent_audit_v2/summary.json"),
                    "prior_ledger_sha256": binding["sha256"]}
    (parent / "independent_audit_v2/prospective_ledger.json").write_text(json.dumps(audit_ledger))
    spec = {"directory": "qualification", "failed_audit_reconciliation": binding}
    for name, key in [("summary.json", "summary_sha256"), ("prospective_ledger.json", "ledger_sha256"),
                      ("independent_audit_v2/summary.json", "audit_summary_sha256"),
                      ("independent_audit_v2/prospective_ledger.json", "audit_ledger_sha256")]:
        spec[key] = sha256_file(parent / name)
    config = {"conditional_admission": {"qualification_audit_directory": "independent_audit_v2",
               "qualification_main_stage": 26, "qualification_audit_stage": 28,
               "accounting_reconciliation": {**binding, **{key: record[key] for key in [
                   "current_machine_stages", "current_machine_cumulative_cpu_seconds",
                   "current_machine_cumulative_gpu_wall_seconds"]}}}}
    return {"qualification": spec}, config


def test_marker_prerequisite_reproduces_all_rows_controls_and_draws(tmp_path, monkeypatch):
    admission, _, output, config = fixture_predecessor(tmp_path, monkeypatch)
    closure = runtime.landmark_closure(admission, output, config)
    assert closure["all_exact"]
    assert closure["bootstrap_draws"] == 2000
    np.testing.assert_allclose(closure["statistics"]["shift_N_mean_midrank"], [.4] * 6, atol=1e-14)
    assert "AP" not in closure["statistics"]


def test_false_audit_gate_blocks_before_signal_reads(tmp_path, monkeypatch):
    admission, parent, output, config = fixture_predecessor(tmp_path, monkeypatch)
    path = parent / "independent_audit_v2/summary.json"
    path.write_text(json.dumps({"all_passed": False, "mismatch_count": 1, "scientific_passed": True}))
    admission["qualification"]["audit_summary_sha256"] = sha256_file(path)
    with pytest.raises(ValueError, match="full audited"):
        runtime.landmark_closure(admission, output, config)
    assert not list(output.iterdir())


def test_changed_saved_bootstrap_cannot_be_admitted_as_exact_closure(tmp_path, monkeypatch):
    admission, parent, output, config = fixture_predecessor(tmp_path, monkeypatch)
    draws = np.load(parent / "bootstrap_gains.npy")
    draws[0] += .001
    np.save(parent / "bootstrap_gains.npy", draws)
    main_path = parent / "summary.json"
    main = json.loads(main_path.read_text())
    main["output_sha256"]["bootstrap_gains.npy"] = sha256_file(parent / "bootstrap_gains.npy")
    main_path.write_text(json.dumps(main))
    admission["qualification"]["summary_sha256"] = sha256_file(main_path)
    assets = admission["asset_sha256"]
    admission, config = rebind_chain(parent)
    admission["asset_sha256"] = assets
    with pytest.raises(ValueError, match="2000draw closure"):
        runtime.landmark_closure(admission, output, config)
    assert not list(output.iterdir())


def test_bridge_rejects_different_asset_admission_before_signal_reads(tmp_path, monkeypatch):
    admission, _, output, config = fixture_predecessor(tmp_path, monkeypatch)
    admission["asset_sha256"]["1.dat"] = "different_release"
    with pytest.raises(ValueError, match="differ from the qualified"):
        runtime.landmark_closure(admission, output, config)
    assert not list(output.iterdir())


def test_double_counted_failed_audit_cpu_is_rejected():
    record = {"current_machine_stages": 27, "failed_stage_cpu_seconds_counted_once": 1,
              "creation_is_setup": True, "new_cost_stage_created": False,
              "original_failed_audit_nulls_preserved": True,
              "current_machine_cumulative_cpu_seconds": 102,
              "current_machine_cumulative_gpu_wall_seconds": 10}
    previous = {"current_machine_stages": 26, "current_machine_cumulative_cpu_seconds": 100,
                "current_machine_cumulative_gpu_wall_seconds": 10}
    failed = {"cpu_seconds": 1, "gpu_wall_seconds": 0,
              "resource_gates": dict.fromkeys(["cpu", "wall", "rss", "output"], True)}
    with pytest.raises(ValueError, match="counted more than once"):
        runtime.validate_reconciled_costs(record, previous, failed)
