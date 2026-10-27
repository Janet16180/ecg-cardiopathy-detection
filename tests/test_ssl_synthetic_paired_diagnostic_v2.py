"""Artificial exact-cost and forward-boundary checks without scientific map reductions."""

import ast
import copy
from pathlib import Path

import pytest

from ecg_experiment import ssl_synthetic_paired_diagnostic_v2 as adapter


def test_exact_anchor_cost_is_read_without_rounded_comparison(monkeypatch: pytest.MonkeyPatch) -> None:
    failure = {
        "progress": {"completed": False, "control_rows": 0, "positive_rows": 0},
        "cpu_seconds": 1.3869068379999998,
        "wall_seconds": 1.387159193000116,
        "waveform_payload_reads": 0,
        "checkpoint_payload_reads": 0,
        "model_inferences": 0,
        "optimizer_updates": 0,
        "gpu_used": False,
    }
    exact_parent = {
        "cumulative_cpu_seconds": 545.1449704129989,
        "stage_cpu_seconds_including_both_audits": 111.012039341,
    }
    monkeypatch.setattr(adapter, "sha256_file", lambda path: "same")
    monkeypatch.setattr(
        adapter.frozen, "_json", lambda path: failure if path.name == "failure.json" else exact_parent
    )
    config = {
        "retained_failure_receipt": "outputs/failure.json",
        "retained_failure_sha256": "same",
        "parent_metrics_audit_sha256": "same",
    }
    stopped, parent = adapter.retained_attempt(config)
    assert parent["cumulative_cpu_seconds"] == 545.1449704129989
    assert parent["cumulative_cpu_seconds"] != 545.144970413
    assert stopped["cpu_seconds"] == failure["cpu_seconds"]
    wrong = copy.deepcopy(failure)
    wrong["progress"]["positive_rows"] = 1
    monkeypatch.setattr(adapter.frozen, "_json", lambda path: wrong)
    with pytest.raises(ValueError, match="precedes"):
        adapter.retained_attempt(config)


def test_shifted_clocks_charge_stopped_attempt_once(monkeypatch: pytest.MonkeyPatch) -> None:
    config = {
        "cpu_seconds_run_ceiling": 120,
        "failure_reporting_reserve_seconds": 2,
        "global_cpu_ceiling": 14400,
        "parent_stage_cpu_ceiling": 1800,
        "new_output_bytes_ceiling": 16 * 1024**2,
        "cache_bytes_ceiling": 4 * 1024**3,
    }
    context = {"config": config, "prior_cpu": 545.1449704129989, "prior_stage_cpu": 111.012039341}
    monkeypatch.setattr(adapter.time, "perf_counter", lambda: 200.0)
    monkeypatch.setattr(adapter.time, "process_time", lambda: 200.0)
    monkeypatch.setattr(adapter.frozen, "_stream_bytes", lambda *args: 0)
    adapter.guard_forward(context, 100.0, 100.0)
    with pytest.raises(TimeoutError, match="main-attempt"):
        adapter.guard_forward(context, 81.0, 100.0)
    with pytest.raises(TimeoutError, match="main-attempt"):
        adapter.guard_forward(context, 100.0, 81.0)


def test_storage_includes_diagnostic_family(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    for name in ["diagnostic_v1", "diagnostic_v2", "diagnostic_v2_audit"]:
        (tmp_path / name).mkdir()
    monkeypatch.setattr(adapter.frozen, "STUDY", tmp_path)
    monkeypatch.setattr(adapter.frozen, "_stream_bytes", lambda path=None: 6 * 1024**2 if path else 0)
    config = {
        "cpu_seconds_run_ceiling": 120,
        "failure_reporting_reserve_seconds": 2,
        "global_cpu_ceiling": 14400,
        "parent_stage_cpu_ceiling": 1800,
        "new_output_bytes_ceiling": 16 * 1024**2,
        "cache_bytes_ceiling": 4 * 1024**3,
    }
    with pytest.raises(ValueError, match="storage"):
        adapter.guard_forward(
            {"config": config, "prior_cpu": 0.0, "prior_stage_cpu": 0.0},
            adapter.time.perf_counter(),
            adapter.time.process_time(),
        )


def test_adapter_never_assigns_frozen_globals() -> None:
    tree = ast.parse(Path(adapter.__file__).read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute) and isinstance(node.ctx, ast.Store):
            assert not (isinstance(node.value, ast.Name) and node.value.id == "frozen")
