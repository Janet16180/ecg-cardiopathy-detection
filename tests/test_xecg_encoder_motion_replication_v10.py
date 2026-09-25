"""Targeted identity, cost and patient-paired checks for xECG V10."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from ecg_experiment.files import sha256_file
from ecg_experiment.xecg_encoder_motion_replication_v10 import run_arm
from ecg_experiment.xecg_encoder_motion_v10_analysis import (
    H_SECONDS,
    P_SECONDS,
    cost_projection,
    paired_seed_bootstrap,
    run_arm_compatibility,
)
from ecg_experiment.xecg_rescue_training import restore_checkpoint
from scripts.experiments.run_xecg_encoder_motion_replication016_v10 import _verify_map


def test_v9_scientific_loop_identity_and_fresh_seed_guard():
    """Retain all V9 update semantics, changing only run identity and cost guards."""
    receipt = run_arm_compatibility()
    assert receipt["scientific_training_loop_ast_identical_after_identity_and_gate_guards"]
    assert "cost_guard()" in receipt["reviewed_unified_diff"]
    with pytest.raises(ValueError, match="fresh seed 47"):
        run_arm({}, "M", 46, "cpu", "production")
    with pytest.raises(ValueError, match="production output"):
        run_arm({}, "F", 47, "cpu", "bridge")


def test_inherited_gate_charges_profiles_and_rejects_overrun():
    """The bounded bridge cannot replace the measured full-path V9 profile."""
    base = cost_projection(0, [])
    assert base["projected_total_seconds"] == pytest.approx(H_SECONDS + 1.25 * (2 * P_SECONDS + 300))
    assert base["passed"]
    assert not cost_projection(1385.176, [])["passed"]
    one = cost_projection(500, [900])
    assert one["projected_total_seconds"] == pytest.approx(H_SECONDS + 500 + 900 + 1.25 * (P_SECONDS + 300))
    with pytest.raises(ValueError, match="Malformed"):
        cost_projection(-1, [])
    with pytest.raises(ValueError, match="Malformed"):
        cost_projection(0, [1, 2, 3])


def test_patient_bootstrap_uses_same_draws_across_fixed_seeds():
    """A sampled patient's two ECGs remain together with multiplicity."""
    labels = np.array([0, 1, 0, 1, 0, 1])
    patients = np.array(["a", "a", "b", "b", "c", "c"])
    seed46 = {
        "M_joint": np.array([0.1, 0.8, 0.3, 0.6, 0.2, 0.9]),
        "F_joint": np.array([0.1, 0.9, 0.2, 0.8, 0.4, 0.7]),
        "M_refit": np.array([0.1, 0.9, 0.3, 0.8, 0.2, 0.8]),
        "F_refit": np.array([0.1, 0.8, 0.3, 0.7, 0.2, 0.9]),
    }
    seed47 = {name: value[::-1].copy() for name, value in seed46.items()}
    result = paired_seed_bootstrap(labels, patients, {"46": seed46, "47": seed47}, draws=1)
    sampled_patients = np.random.default_rng(16020).integers(3, size=3)
    sampled = np.concatenate([np.array([2 * index, 2 * index + 1]) for index in sampled_patients])
    from sklearn.metrics import roc_auc_score

    expected = roc_auc_score(labels[sampled], seed46["F_joint"][sampled]) - roc_auc_score(
        labels[sampled], seed46["M_joint"][sampled]
    )
    assert result["valid"] == 1
    assert result["single_class_invalid"] == 0
    assert result["contrasts"]["46"]["D"]["interval_95"] == pytest.approx([expected, expected])
    assert not result["interval_inference_resolved"]
    assert result["contrasts"]["mean"]["D"]["observed"] == pytest.approx(
        (result["contrasts"]["46"]["D"]["observed"] + result["contrasts"]["47"]["D"]["observed"]) / 2
    )
    with pytest.raises(ValueError, match="fixed optimization seeds"):
        paired_seed_bootstrap(labels, patients, {"46": seed46}, draws=1)
    with pytest.raises(ValueError, match="Incomplete"):
        paired_seed_bootstrap(labels, patients, {"46": seed46, "47": {"M_joint": seed47["M_joint"]}}, draws=1)


def test_changed_manifest_missing_source_and_old_checkpoint_rejected(tmp_path: Path):
    """Never run a changed executable or resume a seed-46 checkpoint as seed 47."""
    (tmp_path / "queue.json").write_text("{}")
    (tmp_path / "sources.json").write_text(json.dumps({"missing_v10_source.py": "0" * 64}))
    with pytest.raises(ValueError, match="manifest/source map changed"):
        _verify_map(tmp_path, "wrong", sha256_file(tmp_path / "sources.json"))
    with pytest.raises(FileNotFoundError):
        _verify_map(tmp_path, sha256_file(tmp_path / "queue.json"), sha256_file(tmp_path / "sources.json"))
    checkpoint = tmp_path / "old.pt"
    torch.save({"version": 1, "fingerprint": {"seed": 46}}, checkpoint)
    with pytest.raises(ValueError, match="differs from frozen inputs"):
        restore_checkpoint(checkpoint, {"seed": 47}, None, None, None, None, None)
