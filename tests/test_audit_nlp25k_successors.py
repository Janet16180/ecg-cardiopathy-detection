"""Small synthetic receipts for the versioned 25k CPU artifact audit."""

from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import pytest
import torch
from scipy.special import expit
from sklearn.metrics import average_precision_score, roc_auc_score

from ecg_experiment.files import sha256_file
from scripts.validation import audit_nlp25k_successors as audit


def write_json(path: Path, value: dict[str, object]) -> None:
    """Write one fixture receipt beneath its parent directory."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def test_pending_successors_open_no_data(tmp_path: Path) -> None:
    """An absent profile is pending, never reported as an audited result."""
    for experiment in audit.ARMS:
        assert audit.audit(experiment, tmp_path) == {
            "experiment": experiment, "status": "pending_profile"}


def test_seal_requires_pinned_mode_digest_and_exact_paths(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The bounded seal must cover exactly the two expected signal paths."""
    seal_path = tmp_path / audit.SEAL
    write_json(seal_path, {"files": {
        "training_signals": {"path": str((tmp_path / audit.CACHE / "signals.npy").resolve())},
        "ptb_signals": {"path": str((tmp_path / audit.PTB / "signals.npy").resolve())},
    }})
    verified = {"status": "passed", "verification_mode": "cached_stat_and_bounded_blocks",
                "full_sha256_recomputed": False,
                "creation_verification": "external_completed_full_sha256_profile",
                "seal_sha256": "a" * 64}
    monkeypatch.setattr(audit, "validate_seal", lambda path: verified)
    identity = {"cache_seal_sha256": sha256_file(seal_path),
                "cache_seal_content_sha256": "a" * 64,
                "cache_verification_mode": "cached_stat_and_bounded_blocks",
                "cache_seal_creation_verification": verified["creation_verification"]}
    assert audit.check_seal(tmp_path, identity, "011") == verified
    verified["full_sha256_recomputed"] = True
    with pytest.raises(ValueError, match="verification mode"):
        audit.check_seal(tmp_path, identity, "011")
    verified["full_sha256_recomputed"] = False
    identity["cache_seal_content_sha256"] = "b" * 64
    with pytest.raises(ValueError, match="content SHA-256"):
        audit.check_seal(tmp_path, identity, "011")


def completed_arm(tmp_path: Path) -> tuple[Path, dict[str, object], torch.nn.Module]:
    """Create one tiny moved encoder and matching final training receipts."""
    output = tmp_path / audit.OUTPUTS["011"]
    arm_dir = output / "gru"
    arm_dir.mkdir(parents=True)
    profile = output / "profile.json"
    profile.write_text("{}")
    identity: dict[str, object] = {"model_seed": 9001}
    initial = torch.nn.Module()
    initial.add_module("encoder", torch.nn.Linear(2, 2))
    trained = torch.nn.Module()
    trained.add_module("encoder", torch.nn.Linear(2, 2))
    trained.load_state_dict(initial.state_dict())
    optimizer = torch.optim.AdamW(trained.parameters())
    trained.encoder(torch.ones(1, 2)).sum().backward()
    optimizer.step()
    for state in optimizer.state.values():
        state["step"].fill_(audit.UPDATES)
    checkpoint = arm_dir / "latest.pt"
    torch.save({"identity": identity, "arm": "gru", "completed_updates": audit.UPDATES,
                "loss_sum": float(audit.UPDATES), "model": trained.state_dict(),
                "optimizer": optimizer.state_dict(),
                "rng": {"python": random.getstate(), "numpy": np.random.get_state(),
                        "torch": torch.get_rng_state(), "cuda": [torch.get_rng_state()]}},
               checkpoint)
    write_json(arm_dir / "complete.json", {
        "identity": identity, "arm": "gru", "completed_updates": audit.UPDATES,
        "record_exposures": audit.EXPOSURES, "mean_cpc_loss": 1.0,
        "checkpoint_sha256": sha256_file(checkpoint),
        "profile_sha256": sha256_file(profile),
    })
    return output, identity, initial


def test_checkpoint_movement_and_tamper_before_deserialization(tmp_path: Path) -> None:
    """A completed arm proves encoder motion and hashes bytes before torch.load."""
    output, identity, initial = completed_arm(tmp_path)
    profile_hash = sha256_file(output / "profile.json")
    result = audit.check_arm(output, "gru", identity, profile_hash, initial)
    assert result["changed_encoder_tensors"] > 0
    assert result["optimizer_states"] > 0
    with (output / "gru/latest.pt").open("ab") as handle:
        handle.write(b"tamper")
    with pytest.raises(ValueError, match="completion receipt identity or SHA-256"):
        audit.check_arm(output, "gru", identity, profile_hash, initial)


def test_012_separate_readout_binds_completed_training_checkpoints(tmp_path: Path) -> None:
    """The 012 v3 readout identity pins all three separate training arms."""
    output = tmp_path / audit.OUTPUTS["012"]
    readout = tmp_path / audit.READOUT_012
    training_identity = {"cache_session_seal_sha256": "s",
                         "input_sha256": {"ptb_signals": "p"}}
    profile = output / "profile.json"
    write_json(profile, {"identity": training_identity, "gate_passed": True})
    paths = dict(audit.READOUT_INPUTS_012)
    paths["training_profile"] = str(profile.relative_to(tmp_path))
    for arm in audit.ARMS["012"]:
        paths[f"{arm}_completion"] = str((output / arm / "complete.json").relative_to(tmp_path))
        paths[f"{arm}_checkpoint"] = str((output / arm / "latest.pt").relative_to(tmp_path))
    hashes = {}
    for key, relative in paths.items():
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        if key != "training_profile":
            path.write_bytes(key.encode())
        hashes[key] = sha256_file(path)
    hashes["pool_signals"] = "p"
    readout_identity = {
        "sha256": hashes, "cache_session_seal_sha256": "s", "feature_width": 512,
        "classifier": "StandardScaler plus L2 logistic C=0.01 lbfgs max_iter=5000 tol=1e-8 seed=42",
        "bootstrap_seed": 12045, "bootstrap_draws": 2000, "full_labels": 15_359,
        "limited_labels": 1_518, "development_records": 1_306,
        "development_patients": 1_173,
    }
    write_json(readout / "result.json", {
        "status": "complete_development_only", "calibration_test_evaluated": False,
        "identity": readout_identity})
    wrong = dict.fromkeys(audit.ARMS["012"], "wrong")
    with pytest.raises(ValueError, match="checkpoint identity differs"):
        audit.check_readout("012", tmp_path, output, training_identity,
                            sha256_file(profile), wrong)


def test_013_readout_replays_heads_and_rejects_wrong_development_ids(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Hash-valid synthetic results still need exact IDs and head replay."""
    output = tmp_path / audit.OUTPUTS["013"]
    output.mkdir(parents=True)
    identity = {"selected_indices_sha256": audit.SELECTED_SHA256}
    profile_path = output / "profile.json"
    write_json(profile_path, {"identity": identity, "gate_passed": True})
    expected = [(f"ptbxl:{index}", f"ptbxl:patient:{index}", index % 2)
                for index in range(1_306)]
    monkeypatch.setattr(audit, "development_rows", lambda root: (expected, 2))
    targets = np.asarray([row[2] for row in expected])
    columns = {"record_ids": np.asarray([row[0] for row in expected]),
               "patient_ids": np.asarray([row[1] for row in expected]),
               "targets": targets}
    features = np.zeros((1_308, 512), dtype=np.float32)
    features[2:, 0] = np.linspace(-1, 1, 1_306)
    feature_hashes = {}
    head_hashes = {}
    scores: dict[str, dict[str, dict[str, float | int]]] = {}
    for arm in audit.ARMS["013"]:
        feature_path = output / f"{arm}_features.npy"
        np.save(feature_path, features)
        feature_hashes[arm] = sha256_file(feature_path)
    for budget, count in (("full", 15_359), ("limited", 1_518)):
        scores[budget] = {}
        for arm in audit.ARMS["013"]:
            name = f"{budget}_{arm}"
            head_path = output / f"{name}_head.npz"
            coef = np.zeros(512)
            coef[0] = 1
            np.savez(head_path, mean=np.zeros(512), scale=np.ones(512), coef=coef,
                     intercept=np.asarray(0.0), n_iter=np.asarray(3))
            head_hashes[name] = sha256_file(head_path)
            probabilities = expit(features[2:, 0].astype(np.float64))
            columns[name] = probabilities
            scores[budget][arm] = {
                "training_labels": count,
                "auroc": roc_auc_score(targets, probabilities),
                "average_precision": average_precision_score(targets, probabilities),
            }
    prediction_path = output / "development_predictions.npz"
    np.savez_compressed(prediction_path, **columns)
    checkpoint_hashes = dict.fromkeys(audit.ARMS["013"], "a" * 64)
    result = {"status": "complete_development_only", "identity": identity,
              "calibration_test_evaluated": False,
              "checkpoint_sha256": checkpoint_hashes,
              "profile_sha256": sha256_file(profile_path),
              "training_labels_full": 15_359, "training_labels_limited": 1_518,
              "development_records": 1_306, "development_patients": 1_173,
              "feature_sha256": feature_hashes, "head_sha256": head_hashes,
              "predictions_sha256": sha256_file(prediction_path), "scores": scores}
    write_json(output / "result.json", result)
    assert audit.check_readout("013", tmp_path, output, identity,
                               sha256_file(profile_path), checkpoint_hashes)["status"] == "passed"
    columns["record_ids"][0] = "ptbxl:wrong"
    np.savez_compressed(prediction_path, **columns)
    result["predictions_sha256"] = sha256_file(prediction_path)
    write_json(output / "result.json", result)
    with pytest.raises(ValueError, match="Development record, patient or target IDs"):
        audit.check_readout("013", tmp_path, output, identity,
                            sha256_file(profile_path), checkpoint_hashes)
