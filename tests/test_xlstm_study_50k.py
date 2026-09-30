"""Conditional 50k predecessor binding and resource-gate tests."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from ecg_experiment import xlstm_study as base
from ecg_experiment import xlstm_study_50k as successor
from ecg_experiment import xlstm_study_v2 as version_two
from ecg_experiment.files import sha256_file, sha256_json


def test_successor_hooks_preserve_v2_output_and_restore_bindings() -> None:
    original = (base.create_model, base.SOURCE_FILES, base.OUTPUT_NAME,
                base.protocol_commit, base.identity)
    with successor.configured():
        assert base.OUTPUT_NAME == version_two.OUTPUT_NAME
        assert set(successor.NEW_SOURCES) <= set(base.SOURCE_FILES)
        assert base.protocol_commit is successor.protocol_commit
        assert base.identity is successor.identity
    assert (base.create_model, base.SOURCE_FILES, base.OUTPUT_NAME,
            base.protocol_commit, base.identity) == original


def _predecessor(tmp_path: Path) -> Path:
    directory = tmp_path / "outputs/experiment038_cpc_xlstm_v2/25k"
    directory.mkdir(parents=True)
    manifest = {"scientific_input": "frozen"}
    (directory / "manifest.json").write_text(json.dumps(manifest))
    (directory / "development_predictions.npz").write_bytes(b"original predictions")
    result = {"identity_sha256": sha256_json(manifest), "status": "complete_development_only"}
    (directory / "result.json").write_text(json.dumps(result))
    (directory / "audit.json").write_text(json.dumps({
        "status": "passed_development_only",
        "result_sha256": sha256_file(directory / "result.json"),
    }))
    return directory


def test_50k_identity_binds_all_four_audited_25k_artifacts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    predecessor = _predecessor(tmp_path)
    monkeypatch.setattr(base, "require_50k_trigger", lambda root: None)
    monkeypatch.setattr(version_two, "protocol_commit", lambda root: "committed-v2")
    monkeypatch.setattr(version_two, "identity", lambda root, tier, receipt: {"tier": tier})
    monkeypatch.setattr(successor, "to_stored", lambda path: path.relative_to(tmp_path).as_posix())
    with successor.configured():
        first = successor.identity(tmp_path, 50, {})
        assert len(first["audited_25k_sha256"]) == 4
        assert set(first["audited_25k_sha256"].values()) == {
            sha256_file(predecessor / name) for name in successor.PREDECESSOR_FILES}
        (predecessor / "manifest.json").write_text(json.dumps({"scientific_input": "changed"}))
        with pytest.raises(ValueError, match="input manifest"):
            successor.identity(tmp_path, 50, {})
    with pytest.raises(ValueError, match="restricted"):
        successor.identity(tmp_path, 25, {})


def test_50k_readout_rejects_failed_latest_train_and_exhausted_budget(tmp_path: Path) -> None:
    directory = tmp_path / "outputs/experiment038_cpc_xlstm_v2/50k"
    directory.mkdir(parents=True)
    ledger = directory / "stage_walltime.json"
    cache = tmp_path / "data/processed/clean_50k_v3_cpc/complete.json"
    cache.parent.mkdir(parents=True)
    cache.write_text(json.dumps({"elapsed_seconds": 100.0}))
    attempts = [{"stage": "train", "status": "complete", "elapsed_seconds": 50.0},
                {"stage": "train", "status": "failed", "elapsed_seconds": 5.0}]
    ledger.write_text(json.dumps({"attempts": attempts}))
    with successor.configured():
        with pytest.raises(ValueError, match="successful latest"):
            successor.require_successful_train(tmp_path)
        attempts[-1]["status"] = "complete"
        ledger.write_text(json.dumps({"attempts": attempts}))
        successor.require_successful_train(tmp_path)
        attempts.append({"stage": "profile", "status": "complete", "elapsed_seconds": 7050.0})
        ledger.write_text(json.dumps({"attempts": attempts}))
        with pytest.raises(RuntimeError, match="exhausted"):
            successor.require_successful_train(tmp_path)
