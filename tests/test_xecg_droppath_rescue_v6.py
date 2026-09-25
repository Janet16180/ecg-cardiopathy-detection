"""Frozen seed, schedule and decision checks for the one-epoch 016 successor."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn

from ecg_experiment.xecg_droppath_rescue import PairedDropPath
from ecg_experiment.xecg_rescue_v6 import (
    SCHEDULER_HORIZON_EPOCHS,
    SEED,
    UPDATES,
    build_model_seed43,
    decisions,
    projected_cost,
    warmup_multipliers,
)


class SmallClassifier(nn.Module):
    def __init__(self, backbone: nn.Module) -> None:
        super().__init__()
        self.backbone = backbone
        self.head = nn.Linear(1024, 1)


def test_seed43_streams_with_identical_seed42_initial_tensors(monkeypatch, tmp_path: Path):
    """Changing optimization seed must not change initial released/probe tensors."""
    from ecg_experiment import xecg_rescue_v6 as module

    monkeypatch.setattr(module, "load_xecg", lambda *args, **kwargs: nn.Linear(2, 2))
    monkeypatch.setattr(module, "install_droppath", lambda backbone, arm, seed: PairedDropPath(arm, seed))
    monkeypatch.setattr(module, "XECGBinaryClassifier", SmallClassifier)
    monkeypatch.setattr(
        module,
        "layerwise_parameter_groups",
        lambda model: [{"params": list(model.parameters()), "lr": 1e-3}],
    )
    probe = tmp_path / "probe.npz"
    np.savez(probe, raw_weight=np.zeros(1024), raw_bias=np.array(0.2))
    starts = []
    for arm in ("legacy", "residual", "off"):
        model, mask, optimizer, scheduler, permutation = build_model_seed43(
            tmp_path, probe, arm, "cpu", 15359
        )
        starts.append({name: tensor.clone() for name, tensor in model.state_dict().items()})
        expected = torch.Generator().manual_seed(SEED).get_state()
        assert torch.equal(mask.get_rng_state(), expected)
        assert torch.equal(permutation.get_state(), expected)
        assert scheduler.get_last_lr() == [0.0]
        assert scheduler.lr_lambdas[0](239) == pytest.approx(239 / UPDATES)
        actual_global = torch.rand(3)
        torch.manual_seed(SEED)
        assert torch.equal(actual_global, torch.rand(3))
        del optimizer
    for state in starts[1:]:
        assert all(torch.equal(starts[0][name], state[name]) for name in state)


def test_warmup_prefix_and_inherited_cost_gate():
    factors = warmup_multipliers()
    assert SCHEDULER_HORIZON_EPOCHS == 2
    assert len(factors) == UPDATES == 240
    assert factors[0] == 0.0
    assert factors[-1] == 239 / 240
    assert projected_cost(0, [])["projected_total_seconds"] == pytest.approx(5065.450323)
    assert projected_cost(2134.0, [])["passed"]
    assert not projected_cost(2135.0, [])["passed"]
    assert not projected_cost(100, [3000])["passed"]
    with pytest.raises(ValueError, match="Invalid 016 v6 cost input"):
        projected_cost(0, [1, 2, 3, 4])


def test_frozen_decision_gates_and_sampling_caveat():
    probe = {"auroc": 0.96194, "mean_fold_sensitivity": 0.95}
    metrics = {
        "legacy": {"auroc": 0.95, "mean_fold_sensitivity": 0.94},
        "residual": {"auroc": 0.959, "mean_fold_sensitivity": 0.949},
        "off": {"auroc": 0.962, "mean_fold_sensitivity": 0.951},
    }
    bootstrap = {"contrasts": {"residual_minus_legacy": {"interval_95": [-0.001, 0.02]}}}
    result = decisions(metrics, probe, bootstrap, {"legacy": 7.03, "residual": 4.10, "off": 0})
    assert result["mechanistic_support_point_screen"]
    assert not result["mechanistic_interval_lower_above_zero"]
    assert result["partial_if_residual_below_probe"]
    assert result["practical_rescue_arms"] == ["off"]
    assert result["preferred_if_any"] == "off"
    metrics["residual"]["auroc"] = 0.964
    result = decisions(metrics, probe, bootstrap, {"legacy": 7.03, "residual": 4.10, "off": 0})
    assert result["practical_rescue_arms"] == ["residual", "off"]
    assert result["preferred_if_any"] == "residual"
