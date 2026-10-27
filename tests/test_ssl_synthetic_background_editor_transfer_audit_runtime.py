"""Typed-container, final reporting, full-bootstrap and retained-cost contract checks."""

import hashlib
import json
import time

import numpy as np
import pytest

from ecg_experiment import ssl_synthetic_background_editor_transfer_audit as core
from ecg_experiment import ssl_synthetic_background_editor_transfer_audit_run as audit
from ecg_experiment.synthetic_audit_io import read_json


def test_json_container_roles_and_numpy_headers_are_explicit(tmp_path) -> None:
    path = tmp_path / "metadata.json"
    path.write_text(json.dumps([{"record_id": 1, "null": None}]))
    assert audit.json_array(path) == [{"record_id": 1, "null": None}]
    with pytest.raises(ValueError, match="object"):
        read_json(path)
    path.write_text(json.dumps({"record_id": 1}))
    assert read_json(path) == {"record_id": 1}
    with pytest.raises(ValueError, match="array"):
        audit.json_array(path)
    np.save(tmp_path / "input.npy", np.zeros((2, 4), dtype=np.float32), allow_pickle=False)
    assert audit.array(tmp_path / "input.npy", (2, 4), np.float32).shape == (2, 4)
    with pytest.raises(ValueError, match="shape/dtype"):
        audit.array(tmp_path / "input.npy", (2, 4), np.float64)


def fake_rows() -> list[dict]:
    rows = []
    for source_index, source in enumerate(core.SOURCES):
        for arm_index, arm in enumerate(core.ARMS):
            for fit in core.FITS:
                for slot in range(64):
                    positive = dict.fromkeys(
                        core.METRICS,
                        0.1 + source_index * 0.01 + arm_index * 0.02 + fit * 0.005 + slot * 0.001,
                    )
                    control = dict.fromkeys(("mean", "q95", "maximum", "top600_mean", "maximum_gt_0p5"), 0.2)
                    rows.append(
                        {
                            "source": source,
                            "arm": arm,
                            "fit_id": fit,
                            "slot": slot,
                            "positives": dict.fromkeys(core.POSITIVES, positive),
                            "controls": {"sham": control, "nuisance": control},
                        }
                    )
    return rows


def test_new_reporting_preserves_every_frozen_bootstrap_value_and_all96_fit_mode_rows() -> None:
    rows = fake_rows()
    config = {"bootstrap": {"seed": 2026106601, "draws": 2000}}
    before, original = core.summarize(rows, config)
    after, actual = audit.aggregate_rows(rows, config)
    assert actual.shape == (2000, 352)
    assert len(after["per_fit_mode"]) == 96
    first = after["per_fit_mode"][0]
    assert first["source"] == "generated"
    assert first["fit_id"] == 1
    assert first["variant"] == "hann_focal"
    assert first["points"]["edited_ap"] == pytest.approx(0.1365, abs=1e-15)
    assert all(name.split("__")[2] in {"sham", "nuisance"} for name in after["endpoint_order"][-80:])
    for column, name in enumerate(after["endpoint_order"]):
        np.testing.assert_array_equal(actual[:, column], original[:, before["endpoint_order"].index(name)])
        assert after["endpoints"][name] == before["endpoints"][name]
    assert after["primary"] == before["primary"]
    assert after["scientific_gates"] == before["scientific_gates"]


def test_main35_cost_addition_includes_nonzero_gpu_and_excludes_current_audit(tmp_path, monkeypatch) -> None:
    previous = {
        "current_machine_cumulative_cpu_seconds": 759.933143906,
        "current_machine_cumulative_gpu_wall_seconds": 473.0495132012293,
    }
    monkeypatch.setattr(audit, "prior34", lambda root: previous)
    main = {"status": "PASS", "technical_passed_before_independent_audit": True, "output_sha256": {}}
    summary = tmp_path / "summary.json"
    summary.write_text(json.dumps(main))
    own = tmp_path / "independent_audit_v1"
    own.mkdir()
    (own / "source.py").write_bytes(b"independent source")
    ledger = {
        "receipt_sha256": hashlib.sha256(summary.read_bytes()).hexdigest(),
        "prior_ledger_sha256": audit.PRIOR34_SHA,
        "current_machine_stages": 35,
        "prior_accounting_verified": True,
        "cpu_seconds": 12.0,
        "wall_seconds": 20.0,
        "peak_rss_bytes": 1000,
        "gpu_wall_seconds": 3.0,
        "current_machine_cumulative_cpu_seconds": previous["current_machine_cumulative_cpu_seconds"] + 12.0,
        "current_machine_cumulative_gpu_wall_seconds": previous["current_machine_cumulative_gpu_wall_seconds"]
        + 3.0,
        "status": "PASS",
        "technical_passed": True,
        "resource_gates": {"cpu": True},
        "output_bytes_before_terminal_ledger": summary.stat().st_size,
    }
    path = tmp_path / "prospective_ledger.json"
    path.write_text(json.dumps(ledger))
    admission = {"prior_ledger": {"sha256": audit.PRIOR34_SHA}}
    costs = audit.accounting(tmp_path, tmp_path, main, admission)
    assert costs["stages_through_main"] == 35
    assert costs["gpu_through_main"] == previous["current_machine_cumulative_gpu_wall_seconds"] + 3.0
    ledger["current_machine_cumulative_cpu_seconds"] += 12.0
    path.write_text(json.dumps(ledger))
    with pytest.raises(ValueError, match="once-only"):
        audit.accounting(tmp_path, tmp_path, main, admission)


def test_finalization_failure_keeps_original_receipt_and_actual_once_only_cost(tmp_path, monkeypatch) -> None:
    original = audit.file_sha256

    def interrupted(path, *args):
        if path.name == "summary.json":
            raise TimeoutError("artificial primary receipt hash timeout")
        return original(path, *args)

    monkeypatch.setattr(audit, "file_sha256", interrupted)
    costs = {
        "prior_ledger_sha256": "actual35",
        "stages_through_main": 35,
        "cpu_through_main": 900.0,
        "gpu_through_main": 500.0,
    }
    clock = {"cpu": time.process_time(), "wall": time.perf_counter()}
    gates = audit.terminal(tmp_path, {"all_passed": True}, clock, costs, {})
    ledger = read_json(tmp_path / "prospective_ledger.json")
    assert read_json(tmp_path / "summary.json")["all_passed"] is True
    assert read_json(tmp_path / "finalization_failure.json")["all_passed"] is False
    assert ledger["current_machine_stages"] == 36
    assert ledger["current_machine_cumulative_cpu_seconds"] == 900.0 + ledger["cpu_seconds"]
    assert ledger["current_machine_cumulative_gpu_wall_seconds"] == 500.0
    assert ledger["receipt"] == "finalization_failure.json"
    assert ledger["output_bytes_before_terminal_ledger"] is None
    assert ledger["technical_passed"] is False
    assert gates["output"] is False


def test_separate_admission_must_bind_independent_core_and_cannot_override_main() -> None:
    main = {"source_sha256": dict.fromkeys(audit.SHARED_SOURCES, "old")}
    required = {audit.RUNNER, audit.CHECKER, audit.PROTOCOL, audit.TESTS, audit.AUDIT_CORE, audit.CORE_TESTS}
    separate = {"source_sha256": dict.fromkeys(required, "new")}
    assert set(audit.source_union(main, separate)) == audit.SHARED_SOURCES | required
    del separate["source_sha256"][audit.AUDIT_CORE]
    with pytest.raises(ValueError, match="explicit independent core"):
        audit.source_union(main, separate)
    separate["source_sha256"][audit.AUDIT_CORE] = "new"
    separate["source_sha256"]["ecg_experiment/intervals.py"] = "changed"
    with pytest.raises(ValueError, match="cannot override"):
        audit.source_union(main, separate)
