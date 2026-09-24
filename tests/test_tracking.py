"""Checks for safe, repeatable import of historical aggregate results."""

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from ecg_experiment.tracking import (
    discover_historical_runs,
    import_historical_runs,
    record_verified_result,
)


def _write(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


class MemoryClient:
    def __init__(self):
        self.runs = []
        self.artifacts = []

    def get_experiment_by_name(self, name):
        if not hasattr(self, "experiment_id"):
            return None
        return SimpleNamespace(experiment_id=self.experiment_id)

    def create_experiment(self, name):
        self.experiment_id = "1"
        return "1"

    def search_runs(self, ids, filter_string, max_results):
        key = filter_string.split("'")[1]
        return [run for run in self.runs if run.data.tags["ecg.source_key"] == key]

    def create_run(self, experiment_id, tags):
        run = SimpleNamespace(info=SimpleNamespace(run_id=str(len(self.runs) + 1), status="RUNNING"),
                              data=SimpleNamespace(tags=tags, params={}, metrics={}))
        self.runs.append(run)
        return run

    def log_param(self, run_id, key, value):
        self.runs[int(run_id) - 1].data.params[key] = value

    def log_metric(self, run_id, key, value):
        self.runs[int(run_id) - 1].data.metrics[key] = value

    def set_tag(self, run_id, key, value):
        self.runs[int(run_id) - 1].data.tags[key] = value

    def log_artifact(self, run_id, path, artifact_path):
        self.artifacts.append((Path(path).name, Path(path).read_text()))

    def set_terminated(self, run_id, status):
        self.runs[int(run_id) - 1].status = status
        self.runs[int(run_id) - 1].info.status = status


def test_historical_import_separates_test_and_development_and_is_idempotent(tmp_path):
    metrics = tmp_path / "outputs/experiment001/model_seed42/metrics.json"
    _write(metrics, {"model": "model", "label_seed": 42,
                     "test": {"auroc": 0.91, "n": 100, "bad": float("nan")},
                     "test_ci95_patient_bootstrap": {"auroc": [0.8, 0.95]}})
    fusion = tmp_path / "outputs/experiment014_jepa_cpc_fusion/full.json"
    _write(fusion, {"completed": True, "calibration_test_opened": False,
                    "development": {"selected_alpha": 1.0,
                                    "grid": {"1.0": {"auroc": 0.89}}}})
    pilot = tmp_path / "outputs/experiment017_morphology_templates_v2"
    _write(pilot / "completion.json", {"status": "development_pilot_complete"})
    _write(pilot / "none_fraction1_seed42/completion.json",
           {"best_development_auroc": 0.92, "best_epoch": 2})
    found = discover_historical_runs(tmp_path)
    assert len(found) == 3
    assert {run.stage for run in found} == {"development_screen", "held_out_test"}
    client = MemoryClient()
    assert import_historical_runs(tmp_path, "unused", client=client) == (3, 0)
    assert import_historical_runs(tmp_path, "unused", client=client) == (0, 3)
    assert not client.artifacts
    test_run = next(run for run in client.runs if run.data.tags["ecg.evaluation_stage"] == "held_out_test")
    assert test_run.data.metrics["test.auroc"] == 0.91
    assert "test.bad" not in test_run.data.metrics
    assert "mlflow.source.git.commit" not in test_run.data.tags
    assert all(run.status == "FINISHED" for run in client.runs)


def test_changed_source_is_rejected_and_receipt_hash_is_checked(tmp_path):
    path = tmp_path / "outputs/experiment007_model/arm/metrics.json"
    _write(path, {"test": {"auroc": 0.8}})
    receipt = path.with_name("complete.json")
    _write(receipt, {"sha256": {"metrics.json": hashlib.sha256(path.read_bytes()).hexdigest()}})
    client = MemoryClient()
    assert import_historical_runs(tmp_path, "unused", client=client) == (1, 0)
    _write(path, {"test": {"auroc": 0.9}})
    with pytest.raises(ValueError, match="Receipt hash mismatch"):
        import_historical_runs(tmp_path, "unused", client=client)
    receipt.unlink()
    with pytest.raises(ValueError, match="Source changed"):
        import_historical_runs(tmp_path, "unused", client=client)


def test_incomplete_import_requires_review_and_artifact_is_sanitized(tmp_path):
    path = tmp_path / "outputs/experiment001/arm/metrics.json"
    _write(path, {"test": {"auroc": 0.8, "patient_123": 999},
                  "patient_rows": [{"name": "private-value"}]})
    client = MemoryClient()
    assert import_historical_runs(tmp_path, "unused", client=client,
                                  log_aggregate_artifacts=True) == (1, 0)
    assert client.artifacts[0][0] == "aggregate_metrics.json"
    assert "private-value" not in client.artifacts[0][1]
    assert "patient_123" not in client.artifacts[0][1]
    client.runs[0].info.status = "FAILED"
    with pytest.raises(RuntimeError, match="incomplete, failed, or duplicated"):
        import_historical_runs(tmp_path, "unused", client=client)


def test_verified_result_requires_frozen_hashes_and_finite_metrics(tmp_path):
    receipt = tmp_path / "outputs/experiment018/receipt.json"
    _write(receipt, {"status": "development_complete"})
    arguments = {"root": tmp_path, "receipt": receipt, "tracking_uri": "unused",
                 "name": "018-seed42", "stage": "development_screen",
                 "metrics": {"auroc": 0.91}, "params": {"seed": 42, "objective": "bce"},
                 "manifest_sha256": "a" * 64, "code_sha256": "b" * 64,
                 "data_sha256": "c" * 64}
    client = MemoryClient()
    assert record_verified_result(**arguments, client=client) is True
    assert record_verified_result(**arguments, client=client) is False
    run = client.runs[0]
    assert run.data.metrics == {"development_screen.auroc": 0.91}
    assert run.data.tags["ecg.manifest_sha256"] == "a" * 64
    assert "mlflow.source.git.commit" not in run.data.tags
    with pytest.raises(ValueError, match="finite"):
        record_verified_result(**(arguments | {"metrics": {"auroc": float("nan")}}),
                               client=client)
    with pytest.raises(ValueError, match="Invalid code SHA"):
        record_verified_result(**(arguments | {"code_sha256": "bad"}), client=client)
    with pytest.raises(ValueError, match="Registry metric conflict"):
        record_verified_result(**(arguments | {"metrics": {"auroc": 0.90}}), client=client)
    with pytest.raises(ValueError, match="Registry tag conflict"):
        record_verified_result(**(arguments | {"data_sha256": "d" * 64}), client=client)


def test_verified_failed_run_records_status_without_metrics(tmp_path):
    receipt = tmp_path / "outputs/experiment018/failure.json"
    _write(receipt, {"error": "caught externally"})
    client = MemoryClient()
    assert record_verified_result(
        tmp_path, receipt, "unused", name="018-failed", stage="execution_failed",
        metrics={}, params={"seed": 42}, manifest_sha256="a" * 64,
        code_sha256="b" * 64, data_sha256="c" * 64,
        status="FAILED", client=client,
    ) is True
    assert client.runs[0].info.status == "FAILED"


def test_config_enrichment_is_allowlisted_and_detects_changes(tmp_path):
    run_dir = tmp_path / "outputs/experiment004_cpc/arm"
    _write(run_dir / "metrics.json", {"model": "cpc", "test": {"auroc": 0.9}})
    client = MemoryClient()
    assert import_historical_runs(tmp_path, "unused", client=client) == (1, 0)
    _write(run_dir / "config.json", {
        "fingerprint": "a" * 64,
        "model_parameters": 1000,
        "labeled_training_records": 15360,
        "epochs_budget": 40,
        "optimizer": "AdamW",
        "patient_identifiers": ["private-value"],
        "cache_dir": "/secret/path",
    })
    assert import_historical_runs(tmp_path, "unused", client=client) == (0, 1)
    params = client.runs[0].data.params
    tags = client.runs[0].data.tags
    assert params["model_parameters"] == "1000"
    assert params["labeled_training_records"] == "15360"
    assert "patient_identifiers" not in params
    assert "cache_dir" not in params
    assert tags["ecg.config_sha256"] == hashlib.sha256((run_dir / "config.json").read_bytes()).hexdigest()
    _write(run_dir / "config.json", {"model_parameters": 1001})
    with pytest.raises(ValueError, match="Registry tag conflict"):
        import_historical_runs(tmp_path, "unused", client=client)


def test_pilot_arm_hash_is_checked_against_parent_receipt(tmp_path):
    pilot = tmp_path / "outputs/experiment015_pilot"
    arm = pilot / "arm/completion.json"
    _write(arm, {"best_development_auroc": 0.7})
    _write(pilot / "completion.json", {"status": "development_pilot_complete",
                                       "arm_completions_sha256": {"arm": "0" * 64}})
    with pytest.raises(ValueError, match="Parent receipt hash mismatch"):
        discover_historical_runs(tmp_path)
    _write(pilot / "completion.json", {"status": "development_pilot_complete",
                                       "arm_completions_sha256": {
                                           "arm": hashlib.sha256(arm.read_bytes()).hexdigest()}})
    (run,) = discover_historical_runs(tmp_path)
    assert run.metrics == {"development.best_auroc": 0.7}
    assert run.params == {"arm": "arm"}


def test_config_values_fall_back_to_fingerprint_and_record_counts(tmp_path):
    run_dir = tmp_path / "outputs/experiment004_cpc/arm"
    _write(run_dir / "metrics.json", {"test": {"auroc": 0.9},
                                      "test_ci95_patient_bootstrap": {"auroc": [0.8, None],
                                                                      "brier": [0.1]}})
    _write(run_dir / "config.json", {"fingerprint": {"seed": 7, "runner_source_sha256": "f" * 64,
                                                     "lr": float("inf")},
                                     "records": {"train": 12}, "batch_size": True})
    (run,) = discover_historical_runs(tmp_path)
    assert run.params == {"model": "arm", "seed": "7", "labeled_training_records": "12"}
    assert run.tags["ecg.runner_source_sha256"] == "f" * 64
    assert run.metrics == {"test.auroc": 0.9, "test.ci95.auroc.lower": 0.8}


def test_verified_result_rejects_identifier_names_and_mismatched_status(tmp_path):
    receipt = tmp_path / "outputs/experiment018/receipt.json"
    _write(receipt, {})
    arguments = {"name": "018", "stage": "development_screen", "metrics": {}, "params": {},
                 "manifest_sha256": "a" * 64, "code_sha256": "b" * 64, "data_sha256": "c" * 64,
                 "client": MemoryClient()}
    with pytest.raises(ValueError, match="Invalid aggregate metric name"):
        record_verified_result(tmp_path, receipt, "unused", **(arguments | {"metrics": {"patient_id": 1}}))
    with pytest.raises(ValueError, match="Invalid parameter name"):
        record_verified_result(tmp_path, receipt, "unused", **(arguments | {"params": {"record_id": 1}}))
    with pytest.raises(ValueError, match="execution_failed"):
        record_verified_result(tmp_path, receipt, "unused", **(arguments | {"status": "FAILED"}))
    with pytest.raises(ValueError, match="outputs directory"):
        record_verified_result(tmp_path, tmp_path / "receipt.json", "unused", **arguments)
