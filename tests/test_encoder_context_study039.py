"""Integrity, immutable outputs and compute-budget checks for Experiment 039."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from ecg_experiment import encoder_context_study039 as study
from ecg_experiment.files import sha256_file, write_json_atomic


def cohort(root: Path, tier: int = 25) -> None:
    """Write tiny byte-identical cohort metadata and its published hashes."""
    for version in (3, 4):
        directory = root / f"data/processed/clean_{tier}k_v{version}"
        directory.mkdir(parents=True)
        (directory / "train_manifest.csv").write_text("record_id\na\n")
    directory = root / f"data/processed/clean_{tier}k_v4"
    metadata = directory / "metadata.json"
    metadata.write_text('{"complete":true}')
    write_json_atomic(root / "outputs/data_quality/clean_cohorts_v4/receipt.json", {
        "tiers": {f"{tier}k": {"manifest_sha256": sha256_file(directory / "train_manifest.csv"),
                               "metadata_sha256": sha256_file(metadata)}}})


def audited(root: Path, predictions: dict[str, np.ndarray], scores: dict | None = None) -> None:
    """Save an independently hashed synthetic result and prediction archive."""
    root.mkdir(parents=True, exist_ok=True)
    np.savez(root / "development_predictions.npz", **predictions)
    write_json_atomic(root / "result.json", {
        "status": "complete_development_only", "run_50k": False, "scores": scores,
        "predictions_sha256": sha256_file(root / "development_predictions.npz")})
    write_json_atomic(root / "audit.json", {
        "status": "passed_development_only", "result_sha256": sha256_file(root / "result.json")})


def stub_execution(monkeypatch: pytest.MonkeyPatch) -> None:
    """Leave stage dispatch and accounting live while stubbing expensive inputs."""
    monkeypatch.setattr(study, "protocol_commit", lambda root: "commit039")
    monkeypatch.setattr(study, "verify_v4", lambda root, tier: {})
    monkeypatch.setattr(study.base, "configure_runtime", lambda: None)


def test_configured_restores_hooks_and_matches_orders_after_failure(tmp_path: Path) -> None:
    names = ("SOURCE_FILES", "OUTPUT_NAME", "SEED", "ORDER_SEED", "identity", "create_model",
             "protocol_commit", "prior_integrity", "_pace_guard", "_elapsed_guard")
    before = {name: getattr(study.base, name) for name in names}
    with study.configured(tmp_path, "patch", 39043):
        assert study.base.SEED == 39043
        assert study.base.ORDER_SEED == 40043
        assert study.base.INTERVAL_SEED == 39045
        assert study.PROTOCOL in study.base.SOURCE_FILES
        assert set(before["SOURCE_FILES"]) <= set(study.base.SOURCE_FILES)
        assert study.base.OUTPUT_NAME.endswith("patch/seed39043")
        order = study.base.order(25)
    def fail_context() -> None:
        with study.configured(tmp_path, "patch", 39043):
            raise RuntimeError("forced")

    with pytest.raises(RuntimeError, match="forced"):
        fail_context()
    assert all(getattr(study.base, name) is value for name, value in before.items())
    with study.configured(tmp_path, "cnn", 39043):
        assert np.array_equal(study.base.order(25), order)


def test_v4_manifest_drift_rejected_before_legacy_identity(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    cohort(tmp_path)
    (tmp_path / "data/processed/clean_25k_v3/train_manifest.csv").write_text("changed")
    monkeypatch.setattr(study.base, "identity", lambda *args: pytest.fail("waveforms opened"))
    with study.configured(tmp_path, "cnn", 39042), pytest.raises(ValueError, match="v4 and v3"):
        study.base.identity(tmp_path, 25, {})


def test_published_metadata_drift_rejected(tmp_path: Path) -> None:
    cohort(tmp_path)
    (tmp_path / "data/processed/clean_25k_v4/metadata.json").write_text("changed")
    with pytest.raises(ValueError, match="Published v4"):
        study.verify_v4(tmp_path, 25)


def test_failed_attempt_charged_to_cell_and_day(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    stub_execution(monkeypatch)

    def fail(*args: object) -> None:
        raise RuntimeError("failed stage")

    monkeypatch.setattr(study.base, "prepare", fail)
    with pytest.raises(RuntimeError, match="failed stage"):
        study.execute("prepare", tmp_path, 25, "cnn", 39042)
    ledger = study.day_ledger(tmp_path)
    attempt = ledger["attempts"][0]
    assert attempt["status"] == "failed"
    assert attempt["elapsed_seconds"] > 0
    with study.configured(tmp_path, "cnn", 39042):
        assert study.used_seconds(tmp_path, 25) == ledger["total_seconds"]
        cache = tmp_path / "data/processed/clean_25k_v3_cpc/complete.json"
        write_json_atomic(cache, {"elapsed_seconds": 8000})
        assert study.used_seconds(tmp_path, 25) == ledger["total_seconds"]


def test_day_ceiling_includes_failures_and_stops_before_prepare(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    stub_execution(monkeypatch)
    write_json_atomic(tmp_path / "outputs" / study.NAME / "day_ledger.json", {
        "attempts": [{"status": "failed", "elapsed_seconds": 28800.0}], "total_seconds": 28800.0})
    monkeypatch.setattr(study.base, "prepare", lambda *args: pytest.fail("work admitted"))
    with pytest.raises(RuntimeError, match="ceiling exhausted"):
        study.execute("prepare", tmp_path, 25, "cnn", 39042)
    assert len(study.day_ledger(tmp_path)["attempts"]) == 2


def test_active_work_enforces_global_ceiling(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(study.time, "monotonic", lambda: 0.0)
    with study.configured(tmp_path, "cnn", 39042):
        monkeypatch.setattr(study.time, "monotonic", lambda: 28801.0)
        with pytest.raises(RuntimeError, match="executable day ceiling"):
            study.base._elapsed_guard(0.0, 28801.0)


def test_projection_reserves_all_future_cells_and_correction(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(study, "historical_cpu_seconds", lambda root, tier: 30.0)
    profile = {"projected_training_seconds": 100, "projected_checkpoint_seconds": 10,
               "projected_feature_seconds": 20, "cpu_report_reserve_seconds": 30,
               "profile_wall_seconds": 5, "preflight_seconds": 2}
    assert study.remaining_projection(tmp_path, profile) == 18 * 167
    study.admit_schedule(tmp_path, profile)
    profile["projected_training_seconds"] = 1400
    with pytest.raises(RuntimeError, match="Full remaining"):
        study.admit_schedule(tmp_path, profile)


def test_50k_requires_audited_25k_without_score_trigger(tmp_path: Path) -> None:
    directory = tmp_path / "outputs" / study.NAME / "cnn/seed39042/25k"
    audited(directory, {"targets": np.asarray([0, 1])})
    with study.configured(tmp_path, "cnn", 39042):
        study.base.require_50k_trigger(tmp_path)
        (directory / "development_predictions.npz").write_bytes(b"changed")
        with pytest.raises(ValueError, match="unchanged audited"):
            study.base.require_50k_trigger(tmp_path)


def test_existing_readout_not_overwritten(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    stub_execution(monkeypatch)
    directory = tmp_path / "outputs" / study.NAME / "cnn/seed39042/25k"
    write_json_atomic(directory / "result.json", {"sentinel": "original"})
    original = (directory / "result.json").read_bytes()
    monkeypatch.setattr(study.base, "readout", lambda *args: pytest.fail("score overwritten"))
    with pytest.raises(ValueError, match="already exists"):
        study.execute("readout", tmp_path, 25, "cnn", 39042)
    assert (directory / "result.json").read_bytes() == original


def test_run_checks_and_skips_unchanged_audited_cell(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    stub_execution(monkeypatch)
    directory = tmp_path / "outputs" / study.NAME / "cnn/seed39042/25k"
    audited(directory, {"targets": np.asarray([0, 1])})
    original = {path.name: path.read_bytes() for path in directory.iterdir()}
    monkeypatch.setattr(study.base, "ensure_manifest", lambda *args: {})
    monkeypatch.setattr(study, "execute", lambda *args: pytest.fail("completed stage rerun"))
    assert study.run(tmp_path, 25, "cnn", 39042)["status"] == "complete_development_only"
    assert {path.name: path.read_bytes() for path in directory.iterdir()} == original


def test_predecessor_exact_replay_and_metric_tampering(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    dev = [SimpleNamespace(record_id=f"r{i}", patient_id=f"p{i}", target=i % 2) for i in range(4)]
    monkeypatch.setattr(study.base, "pool", lambda root: None)
    monkeypatch.setattr(study.base, "training_examples", lambda pool: ([], []))
    monkeypatch.setattr(study.base, "development_only_examples", lambda pool, train: dev)
    predictions = {"record_ids": np.asarray([row.record_id for row in dev]),
                   "patient_ids": np.asarray([row.patient_id for row in dev]),
                   "targets": np.asarray([row.target for row in dev])}
    for budget in ("limited", "full"):
        for arm in study.base.ARMS:
            predictions[f"{budget}_{arm}"] = np.asarray([0.1, 0.8, 0.2, 0.9])
    scores = {budget: {arm: study.base.metrics(predictions["targets"], predictions[f"{budget}_{arm}"])
                       for arm in study.base.ARMS} for budget in ("limited", "full")}
    for arms in scores.values():
        for metrics in arms.values():
            metrics.update({"classifier_iterations": 5, "training_labels": 10})
    for tier in study.TIERS:
        audited(tmp_path / f"outputs/experiment038_cpc_xlstm_v2/{tier}k", predictions, scores)
    replay = study.prior_integrity(tmp_path, lambda root: {"status": "passed019"})
    assert replay["experiment019"]["status"] == "passed019"
    assert set(replay["experiment038"]) == {"25", "50"}
    scores["limited"]["gru"]["auroc"] = 0.5
    audited(tmp_path / "outputs/experiment038_cpc_xlstm_v2/50k", predictions, scores)
    with pytest.raises(ValueError, match="exact metric replay"):
        study.prior_integrity(tmp_path, lambda root: {})


def test_observed_slow_pace_charged_even_with_existing_profile(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(study, "ENCODERS", ("cnn",))
    monkeypatch.setattr(study, "SEEDS", (39042,))
    monkeypatch.setattr(study, "TIERS", (25,))
    monkeypatch.setattr(study, "historical_cpu_seconds", lambda root, tier: 0.0)
    monkeypatch.setattr(study.time, "monotonic", lambda: 0.0)
    monkeypatch.setattr(study.base, "_pace_guard", lambda *args: None)
    monkeypatch.setattr(study.base, "_elapsed_guard", lambda *args: None)
    write_json_atomic(tmp_path / "outputs" / study.NAME / "day_ledger.json", {
        "attempts": [{"elapsed_seconds": 24200.0}], "total_seconds": 24200.0})
    profile = {"projected_training_seconds": 100.0, "projected_checkpoint_seconds": 0.0,
               "projected_feature_seconds": 0.0, "profile_wall_seconds": 0.0,
               "preflight_seconds": 0.0, "arms": {"gru": {"seconds_per_update": 1.0}}}
    write_json_atomic(tmp_path / "outputs" / study.NAME / "cnn/seed39042/25k/profile.json", profile)
    study.admit_schedule(tmp_path, profile)
    with study.configured(tmp_path, "cnn", 39042), pytest.raises(RuntimeError, match="Full remaining"):
        study.base._pace_guard(profile, 0.0, 0.0, "gru", study.base.UPDATES - 1, 100.0)
