import hashlib
import json

import pytest

from ecg_experiment.ssl_synthetic_ludb_coherence_audit_v2 import accounting, prerequisite


def test_old_dense_audit_directory_cannot_satisfy_actual_landmark_gate(tmp_path):
    admission = {"qualification": {}}
    config = {
        "conditional_admission": {
            "qualification_audit_directory": "independent_audit_v1",
            "qualification_audit_stage": 27,
            "qualification_main_stage": 26,
        }
    }
    with pytest.raises(ValueError, match="actual landmark"):
        prerequisite(tmp_path, tmp_path, admission, config)


def test_bridge29_appends_actual28_once_and_excludes_new_audit_subtree(tmp_path):
    old = tmp_path / "prior"
    old.mkdir()
    previous = {
        "current_machine_stages": 28,
        "current_machine_cumulative_cpu_seconds": 454.04911085099997,
        "current_machine_cumulative_gpu_wall_seconds": 447.98139213304967,
    }
    (old / "prospective_ledger.json").write_text(json.dumps(previous))
    previous_sha = hashlib.sha256((old / "prospective_ledger.json").read_bytes()).hexdigest()
    study = tmp_path / "bridge"
    study.mkdir()
    main = {"cpu_seconds": 0.5, "wall_seconds": 0.75}
    text = json.dumps(main)
    (study / "summary.json").write_text(text)
    audit = study / "independent_audit_v1"
    audit.mkdir()
    (audit / "own_source.py").write_bytes(b"x" * 10000)
    ledger = {
        "receipt": "summary.json",
        "receipt_sha256": hashlib.sha256(text.encode()).hexdigest(),
        "current_machine_stages": 29,
        "prior_ledger_sha256": previous_sha,
        "current_machine_cumulative_cpu_seconds": previous["current_machine_cumulative_cpu_seconds"] + 1.25,
        "current_machine_cumulative_gpu_wall_seconds": previous[
            "current_machine_cumulative_gpu_wall_seconds"
        ],
        "cpu_seconds": 1.25,
        "wall_seconds": 2.0,
        "gpu_wall_seconds": 0,
        "peak_rss_bytes": 1000000,
        "resource_gates": {"cpu": True, "wall": True, "rss": True, "output": True},
        "output_bytes_before_terminal_ledger": len(text),
    }
    file = study / "prospective_ledger.json"
    file.write_text(json.dumps(ledger))
    admission = {"prior_ledger": {"path": "prior/prospective_ledger.json", "sha256": previous_sha}}
    assert accounting(tmp_path, study, main, admission)["stages_through_main"] == 29
    ledger["current_machine_cumulative_cpu_seconds"] += 1.25
    file.write_text(json.dumps(ledger))
    with pytest.raises(ValueError, match="once"):
        accounting(tmp_path, study, main, admission)
