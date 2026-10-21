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
    runtime.shared.write_new(parent / "independent_audit_v1/summary.json",
                             {"all_passed": True, "mismatch_count": 0, "scientific_passed": True})
    for name, stage in [("prospective_ledger.json", 26),
                        ("independent_audit_v1/prospective_ledger.json", 27)]:
        ledger = {"current_machine_stages": stage, "cpu_seconds": 1, "gpu_wall_seconds": 0,
                  "current_machine_cumulative_cpu_seconds": 100 + stage - 26,
                  "current_machine_cumulative_gpu_wall_seconds": 10,
                  "resource_gates": dict.fromkeys(["cpu", "wall", "rss", "output"], True)}
        if stage == 26:
            ledger["receipt_sha256"] = sha256_file(parent / "summary.json")
        else:
            ledger["sha256"] = sha256_file(parent / "independent_audit_v1/summary.json")
            ledger["prior_ledger_sha256"] = sha256_file(parent / "prospective_ledger.json")
        runtime.shared.write_new(parent / name, ledger)
    spec = {"directory": "qualification"}
    for name, key in [("summary.json", "summary_sha256"), ("prospective_ledger.json", "ledger_sha256"),
                      ("independent_audit_v1/summary.json", "audit_summary_sha256"),
                      ("independent_audit_v1/prospective_ledger.json", "audit_ledger_sha256")]:
        spec[key] = sha256_file(parent / name)
    monkeypatch.setattr(runtime, "from_stored", lambda name: tmp_path / name)
    return {"qualification": spec, "asset_sha256": assets}, parent, output


def test_marker_prerequisite_reproduces_all_rows_controls_and_draws(tmp_path, monkeypatch):
    admission, _, output = fixture_predecessor(tmp_path, monkeypatch)
    closure = runtime.landmark_closure(admission, output)
    assert closure["all_exact"]
    assert closure["bootstrap_draws"] == 2000
    np.testing.assert_allclose(closure["statistics"]["shift_N_mean_midrank"], [.4] * 6, atol=1e-14)
    assert "AP" not in closure["statistics"]


def test_false_audit_gate_blocks_before_signal_reads(tmp_path, monkeypatch):
    admission, parent, output = fixture_predecessor(tmp_path, monkeypatch)
    path = parent / "independent_audit_v1/summary.json"
    path.write_text(json.dumps({"all_passed": False, "mismatch_count": 1, "scientific_passed": True}))
    admission["qualification"]["audit_summary_sha256"] = sha256_file(path)
    with pytest.raises(ValueError, match="full audited"):
        runtime.landmark_closure(admission, output)
    assert not list(output.iterdir())


def test_changed_saved_bootstrap_cannot_be_admitted_as_exact_closure(tmp_path, monkeypatch):
    admission, parent, output = fixture_predecessor(tmp_path, monkeypatch)
    draws = np.load(parent / "bootstrap_gains.npy")
    draws[0] += .001
    np.save(parent / "bootstrap_gains.npy", draws)
    main_path = parent / "summary.json"
    main = json.loads(main_path.read_text())
    main["output_sha256"]["bootstrap_gains.npy"] = sha256_file(parent / "bootstrap_gains.npy")
    main_path.write_text(json.dumps(main))
    admission["qualification"]["summary_sha256"] = sha256_file(main_path)
    ledger_path = parent / "prospective_ledger.json"
    ledger = json.loads(ledger_path.read_text())
    ledger["receipt_sha256"] = admission["qualification"]["summary_sha256"]
    ledger_path.write_text(json.dumps(ledger))
    admission["qualification"]["ledger_sha256"] = sha256_file(ledger_path)
    audit_path = parent / "independent_audit_v1/prospective_ledger.json"
    audit = json.loads(audit_path.read_text())
    audit["prior_ledger_sha256"] = admission["qualification"]["ledger_sha256"]
    audit_path.write_text(json.dumps(audit))
    admission["qualification"]["audit_ledger_sha256"] = sha256_file(audit_path)
    with pytest.raises(ValueError, match="2000draw closure"):
        runtime.landmark_closure(admission, output)
    assert not list(output.iterdir())


def test_bridge_rejects_different_asset_admission_before_signal_reads(tmp_path, monkeypatch):
    admission, _, output = fixture_predecessor(tmp_path, monkeypatch)
    admission["asset_sha256"]["1.dat"] = "different_release"
    with pytest.raises(ValueError, match="differ from the qualified"):
        runtime.landmark_closure(admission, output)
    assert not list(output.iterdir())
