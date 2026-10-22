"""Actual functional batching and immutable predecessor recovery integrity checks."""

import ast
import hashlib
import inspect
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from ecg_experiment import ssl_synthetic_mixed_background_audit_run_v2 as audit


def write_json(path: Path, value: dict) -> str:
    """Persist artificial local metadata with its exact identity."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value) + "\n")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_actual_functional_convolutions_receive_singleton_batch_and_return_one_record() -> None:
    """Exercise the real frozen helper rather than a mocked caller signature."""
    shapes = {
        "0.weight": (32, 12, 25),
        "0.bias": (32,),
        "2.weight": (32, 32, 25),
        "2.bias": (32,),
        "4.weight": (12, 32, 1),
        "4.bias": (12,),
    }
    state = {key: torch.zeros(shape, dtype=torch.float32) for key, shape in shapes.items()}
    record = np.zeros((12, 5000), dtype=np.float32)
    with pytest.raises(ValueError, match="batched actual"):
        audit.frozen.functional_replay(state, record)
    actual = audit.record_replay(state, record)
    assert actual.shape == record.shape
    assert actual.dtype == np.float32
    np.testing.assert_array_equal(actual, np.full(record.shape, 0.5, dtype=np.float32))
    np.testing.assert_array_equal(actual, audit.frozen.functional_replay(state, record[None])[0])
    with pytest.raises(ValueError, match="single declared"):
        audit.record_replay(state, record[None])


def test_numerical_dispatch_is_identical_except_single_record_adapter() -> None:
    """Freeze every endpoint, reduction and replay identity while correcting only call geometry."""
    original = ast.parse(inspect.getsource(audit.frozen.numerical))
    successor = ast.parse(inspect.getsource(audit.numerical))
    adapters = [
        node
        for node in ast.walk(successor)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "record_replay"
    ]
    assert len(adapters) == 1
    adapters[0].func.id = "functional_replay"
    assert ast.dump(original, include_attributes=False) == ast.dump(successor, include_attributes=False)


def test_both_retained_audit_subtrees_are_separate_from_immutable_main_catalog(tmp_path: Path) -> None:
    """Neither preserved failure snapshots nor new audit snapshots become fabricated main outputs."""
    study = tmp_path / "study"
    study.mkdir()
    (study / "bound.txt").write_text("immutable main artifact")
    (study / "summary.json").write_text("{}")
    (study / "prospective_ledger.json").write_text("{}")
    for name in audit.AUDIT_DIRS:
        snapshot = study / name / "source_snapshot"
        snapshot.mkdir(parents=True)
        (snapshot / "test.py").write_text("# separate audit source\n")
    main = {"source_sha256": {}, "output_sha256": {"bound.txt": audit.file_sha256(study / "bound.txt")}}
    external = tmp_path / "admitted.txt"
    external.write_text("admitted input")
    admission = {"input_sha256": {"admitted.txt": audit.file_sha256(external)}}
    assert audit.identities(tmp_path, tmp_path, study, main, admission, lambda: None) == 2
    (study / "extra.txt").write_text("unexpected main artifact")
    with pytest.raises(ValueError, match="membership"):
        audit.identities(tmp_path, tmp_path, study, main, admission, lambda: None)


def failed_fixture(root: Path, monkeypatch) -> tuple[Path, Path, dict, dict]:
    """Construct an artificial direct main38/failure39 chain with complete old-source snapshots."""
    source, study = root / "source", root / "study"
    source.mkdir()
    study.mkdir()
    main = {"status": "PASS"}
    main_sha = write_json(study / "summary.json", main)
    main_ledger = write_json(study / "prospective_ledger.json", {"current_machine_stages": 38})
    required = audit.frozen.SHARED_SOURCES | {
        audit.frozen.RUNNER,
        audit.frozen.CHECKER,
        audit.frozen.PROTOCOL,
        audit.frozen.TESTS,
        audit.frozen.CORE,
        audit.frozen.INPUTS,
        audit.frozen.CORE_TESTS,
    }
    sources = {}
    for name in required:
        path = source / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("# artificial immutable source: " + name)
        sources[name] = audit.file_sha256(path)
    main_admission_sha = write_json(source / audit.OLD_MAIN, {"source_sha256": {}})
    failed_admission_sha = write_json(source / audit.OLD_AUDIT, {"source_sha256": sources})
    failed = study / "independent_audit_v1"
    failed_sha = write_json(
        failed / "summary.json",
        {
            "all_passed": False,
            "mismatch_count": 1,
            "error_type": "ValueError",
            "error": "Require float32 batched actual native-rate model inputs",
        },
    )
    for name in [*sources, audit.OLD_MAIN, audit.OLD_AUDIT]:
        path = failed / "source_snapshot" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes((source / name).read_bytes())
    retained = sum(p.stat().st_size for p in failed.rglob("*") if p.is_file())
    costs = {
        "prior_ledger_sha256": main_ledger,
        "stages_through_main": 38,
        "cpu_through_main": 1172.911747755,
        "gpu_through_main": 564.4522484582849,
    }
    monkeypatch.setattr(audit, "main_accounting", lambda *args: costs)
    ledger = {
        "sha256": failed_sha,
        "technical_passed": False,
        "prior_accounting_verified": True,
        "resource_gates": {"cpu": True},
        "prior_ledger_sha256": main_ledger,
        "cpu_seconds": 35.699429857,
        "gpu_wall_seconds": 0.0,
        "current_machine_stages": 39,
        "current_machine_cumulative_cpu_seconds": 1208.6111776120001,
        "current_machine_cumulative_gpu_wall_seconds": 564.4522484582849,
        "output_bytes_before_terminal_ledger": retained,
    }
    failed_ledger_sha = write_json(failed / "prospective_ledger.json", ledger)
    recovery = {
        "main_summary_sha256": main_sha,
        "main_ledger_sha256": main_ledger,
        "main_admission": {"path": audit.OLD_MAIN, "sha256": main_admission_sha},
        "failed_audit_admission": {"path": audit.OLD_AUDIT, "sha256": failed_admission_sha},
        "failed_directory": "independent_audit_v1",
        "failed_summary_sha256": failed_sha,
        "failed_ledger_sha256": failed_ledger_sha,
        "failed_cpu_seconds": 35.699429857,
        "prior_stage": 39,
        "prior_cpu_seconds": 1208.6111776120001,
        "prior_gpu_wall_seconds": 564.4522484582849,
    }
    write_json(source / audit.RECOVERY, recovery)
    return source, study, main, recovery


def test_direct_failed_cost_is_counted_once_and_failed_sources_are_verified(
    tmp_path: Path, monkeypatch
) -> None:
    """Current ledger hashes alone cannot hide lost source evidence or duplicated failed CPU cost."""
    source, study, main, recovery = failed_fixture(tmp_path, monkeypatch)
    costs = audit.latest_accounting(tmp_path, source, study, main)
    assert costs["stages_through_main"] == 39
    assert costs["cpu_through_main"] == 1208.6111776120001
    snapshot = study / "independent_audit_v1/source_snapshot" / audit.frozen.CORE
    original = snapshot.read_bytes()
    snapshot.write_bytes(b"changed failure source")
    with pytest.raises(ValueError, match="failed39 source snapshot"):
        audit.latest_accounting(tmp_path, source, study, main)
    snapshot.write_bytes(original)
    ledger_file = study / "independent_audit_v1/prospective_ledger.json"
    ledger = json.loads(ledger_file.read_text())
    ledger["current_machine_cumulative_cpu_seconds"] += ledger["cpu_seconds"]
    recovery["failed_ledger_sha256"] = write_json(ledger_file, ledger)
    write_json(source / audit.RECOVERY, recovery)
    with pytest.raises(ValueError, match="additive lineage"):
        audit.latest_accounting(tmp_path, source, study, main)
