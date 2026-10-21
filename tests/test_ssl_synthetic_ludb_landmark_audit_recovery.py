import hashlib
import json

import pytest

from ecg_experiment.ssl_synthetic_ludb_landmark_audit_recovery import main_files, next_stage


@pytest.mark.parametrize("receipt_path", ["summary.json", "stage/summary.json"])
def test_legacy_receipt_paths_and_summary_only_resource_gates(tmp_path, receipt_path):
    stage = tmp_path / "stage"
    stage.mkdir()
    receipt = json.dumps({"resource_gates": {"cpu": True, "wall": True, "rss": True}})
    (stage / "summary.json").write_text(receipt)
    previous = {
        "current_machine_stages": 18,
        "current_machine_cumulative_cpu_seconds": 10.0,
        "current_machine_cumulative_gpu_wall_seconds": 20.0,
    }
    ledger = {
        "current_machine_stages": 19,
        "prior_ledger_sha256": "prior",
        "receipt": receipt_path,
        "sha256": hashlib.sha256(receipt.encode()).hexdigest(),
        "cpu_seconds": 0.25,
        "gpu_wall_seconds": 0,
        "current_machine_cumulative_cpu_seconds": 10.25,
        "current_machine_cumulative_gpu_wall_seconds": 20.0,
    }
    path = stage / "prospective_ledger.json"
    path.write_text(json.dumps(ledger))
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    assert next_stage(tmp_path, path, digest, previous, "prior")["current_machine_stages"] == 19
    assert "resource_gates" not in json.loads(path.read_text())
    ledger["current_machine_cumulative_cpu_seconds"] += 0.25
    path.write_text(json.dumps(ledger))
    with pytest.raises(ValueError, match="once"):
        next_stage(tmp_path, path, hashlib.sha256(path.read_bytes()).hexdigest(), previous, "prior")


def test_catalog_filters_both_attempts_without_hiding_unrelated_main_artifacts(tmp_path):
    (tmp_path / "summary.json").write_text("{}")
    (tmp_path / "unexpected.txt").write_text("retained")
    for name in ("independent_audit_v1", "independent_audit_v2"):
        (tmp_path / name).mkdir()
        (tmp_path / name / "summary.json").write_text("{}")
    assert set(main_files(tmp_path)) == {"summary.json", "unexpected.txt"}
