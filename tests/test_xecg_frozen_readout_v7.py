"""Identity, fixed fitting and readout-gate tests for the v7 xECG audit."""

from __future__ import annotations

import numpy as np
import pytest
import torch
from torch import nn

from ecg_experiment import xecg_readout_v7 as readout
from scripts.experiments.run_xecg_frozen_readout016_v7 import assert_immutable, cost_projection


def test_feature_join_preserves_requested_order_and_rejects_missing_or_duplicate():
    all_ids = np.array([17, 2, 9, 33])
    assert readout.selected_indices(all_ids, [9, 17, 33]).tolist() == [2, 0, 3]
    with pytest.raises(ValueError, match="Duplicate ECG identifier"):
        readout.selected_indices(all_ids, [9, 9])
    with pytest.raises(ValueError, match="absent"):
        readout.selected_indices(all_ids, [42])


def test_fixed_probe_fits_scaler_on_training_only(monkeypatch):
    monkeypatch.setattr(readout, "TRAIN_RECORDS", 20)
    monkeypatch.setattr(readout, "DEVELOPMENT_RECORDS", 4)
    monkeypatch.setattr(readout, "FEATURE_WIDTH", 3)
    rng = np.random.default_rng(12)
    train = rng.normal(size=(20, 3)).astype(np.float32)
    labels = np.array([0, 1] * 10)
    development = np.full((4, 3), 1000, dtype=np.float32)
    result = readout.fixed_probe(train, labels, development)
    np.testing.assert_allclose(result["mean"], train.mean(axis=0), atol=1e-6)
    assert result["iterations"] < 3000
    assert result["development_logits"].shape == (4,)
    np.testing.assert_allclose(
        result["development_logits"],
        readout.affine_logits(development, result["weight"], result["bias"]),
        atol=1e-9,
    )


def test_raw_head_dimension_contract_and_joint_logit_reproduction():
    features = np.ones((3, 1024), dtype=np.float32)
    weight = np.arange(1024, dtype=np.float32) / 1024
    logits = readout.affine_logits(features, weight, 0.2)
    assert logits.shape == (3,)
    np.testing.assert_allclose(logits, np.full(3, weight.sum() + 0.2), atol=1e-5)
    with pytest.raises(ValueError, match="dimensions differ"):
        readout.affine_logits(features[:, :10], weight, 0.0)


def test_encoder_state_immutability_guard():
    model = nn.Linear(3, 1)
    model.eval().requires_grad_(False)
    before = {name: tensor.detach().clone() for name, tensor in model.state_dict().items()}
    assert_immutable(model, before)
    with torch.no_grad():
        model.weight.add_(0.01)
    with pytest.raises(RuntimeError, match="changed"):
        assert_immutable(model, before)


def test_cost_and_prespecified_off_decisions():
    gate = cost_projection(220, 180, 20, [], [])
    assert gate["T_seconds"] == 190
    assert gate["P_seconds"] == 60
    assert gate["projected_total_seconds"] == pytest.approx(220 + 1.25 * (2 * 190 + 3 * 60 + 300))
    probe = {"auroc": 0.962, "mean_fold_sensitivity": 0.95}
    metrics = {
        "released_refit": probe,
        "off_joint": {"auroc": 0.948},
        "off_refit": {"auroc": 0.961, "mean_fold_sensitivity": 0.947},
        "residual_refit": {"auroc": 0.95},
        "legacy_refit": {"auroc": 0.94},
    }
    bootstrap = {"contrasts": {"off_refit_minus_off_joint": {"interval_95": [-0.001, 0.02]}}}
    decision = readout.decisions(metrics, bootstrap)
    assert decision["recoverable_readout_point_screen"]
    assert not decision["recoverable_interval_lower_above_zero"]
    assert decision["near_complete_practical_recovery"]
    assert not decision["potential_useful_adaptation"]
    metrics["off_refit"]["auroc"] = 0.958
    decision = readout.decisions(metrics, bootstrap)
    assert decision["all_adapted_refits_below_probe_by_more_than_0_002"]
