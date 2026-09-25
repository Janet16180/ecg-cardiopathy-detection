"""Focused checks for the compute-matched zero-encoder-step intervention."""

from __future__ import annotations

from copy import deepcopy

import numpy as np
import pytest
import torch
from torch import nn

from ecg_experiment.xecg_encoder_motion_v9 import encoder_equal, encoder_state, train_update
from ecg_experiment.xecg_rescue_training import compare_replay_snapshots, make_scheduler


class ToyBackbone(nn.Module):
    """Small trainable encoder with xECG's intentionally unused mask token."""

    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(4, 3)
        self.mask_token = nn.Parameter(torch.ones(3))

    def forward(self, signal):
        return torch.tanh(self.linear(signal))


class ToyClassifier(nn.Module):
    """Expose the same backbone/head names as the released classifier."""

    def __init__(self):
        super().__init__()
        self.backbone = ToyBackbone()
        self.head = nn.Linear(3, 1)

    def forward(self, signal):
        return self.head(self.backbone(signal)).squeeze(-1)


def toy_data():
    """Return deterministic nondegenerate waveforms and binary annotations."""
    signals = torch.arange(32, dtype=torch.float32).reshape(8, 4) / 17
    labels = torch.tensor([0, 1, 0, 1, 1, 0, 1, 0], dtype=torch.float32)
    return list(zip(signals, labels, strict=True))


def configured(model, frozen):
    """Use equal optimizer groups and zero only the frozen encoder group."""
    groups = [
        {
            "name": "encoder",
            "params": list(model.backbone.parameters()),
            "lr": 0.0 if frozen else 0.03,
            "weight_decay": 0.1,
        },
        {"name": "binary_head", "params": list(model.head.parameters()), "lr": 0.1, "weight_decay": 0.1},
    ]
    optimizer = torch.optim.AdamW(groups, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.1)
    scheduler = make_scheduler(optimizer, 240)
    return optimizer, scheduler


def test_full_gradient_zero_encoder_lr_matches_explicit_reference():
    """Frozen control computes encoder gradients and clips them globally."""
    torch.manual_seed(11)
    initial = ToyClassifier()
    moving, frozen, reference = (deepcopy(initial) for _ in range(3))
    models = {"M": moving, "F": frozen}
    states = {arm: configured(model, arm == "F") for arm, model in models.items()}
    ref_optimizer, ref_scheduler = configured(reference, True)
    data = toy_data()
    group = torch.arange(len(data))
    ids = np.arange(100, 108, dtype=np.int64)
    baseline = encoder_state(frozen)

    for update in range(2):
        rows = {
            arm: train_update(model, *states[arm], data, group, ids, "cpu", update)
            for arm, model in models.items()
        }
        ref_optimizer.zero_grad(set_to_none=True)
        signals = torch.stack([row[0] for row in data])
        labels = torch.stack([row[1] for row in data])
        nn.functional.binary_cross_entropy_with_logits(reference(signals), labels).backward()
        encoder_norm = torch.linalg.vector_norm(
            torch.stack(
                [
                    torch.linalg.vector_norm(parameter.grad)
                    for name, parameter in reference.backbone.named_parameters()
                    if name != "mask_token"
                ]
            )
        )
        global_norm = nn.utils.clip_grad_norm_(reference.parameters(), 3.0)
        ref_optimizer.step()
        ref_scheduler.step()

        assert rows["F"]["preclip_encoder_norm"] == pytest.approx(float(encoder_norm), rel=1e-6)
        assert rows["F"]["preclip_global_norm"] == pytest.approx(float(global_norm), rel=1e-6)
        assert rows["F"]["preclip_global_norm"] > rows["F"]["preclip_head_norm"]
        assert rows["F"]["unused_gradient_parameters"] == ["backbone.mask_token"]
        assert encoder_equal(frozen, baseline)
        for actual, expected in zip(frozen.parameters(), reference.parameters(), strict=True):
            torch.testing.assert_close(actual, expected, rtol=0, atol=1e-7)
        assert rows["M"]["batch_ids_sha256"] == rows["F"]["batch_ids_sha256"]
        assert rows["M"]["preclip_global_norm"] == pytest.approx(rows["F"]["preclip_global_norm"], rel=1e-6)
        assert rows["M"]["head_update_norm"] == pytest.approx(rows["F"]["head_update_norm"], rel=1e-6)
        assert states["F"][1].base_lrs[0] == 0
        assert states["F"][0].param_groups[0]["lr"] == 0

    assert not encoder_equal(moving, baseline)
    assert rows["M"]["head_update_norm"] > 0
    assert rows["F"]["head_update_norm"] > 0


def test_replay_tolerance_keeps_exact_batch_and_rng():
    """Accept tiny FP32 drift but reject a changed batch or material update."""
    first = {
        "model": {"weight": torch.tensor([1.0])},
        "optimizer": {},
        "batch_ids_sha256": "same",
        "global_rng": torch.tensor([5], dtype=torch.uint8),
    }
    second = deepcopy(first)
    second["model"]["weight"] += 1e-7
    assert compare_replay_snapshots(first, second)["exact_fields"] == ["batch_ids_sha256", "global_rng"]
    second["model"]["weight"] += 1e-4
    with pytest.raises(RuntimeError, match="numeric tolerance"):
        compare_replay_snapshots(first, second)
    second = deepcopy(first)
    second["batch_ids_sha256"] = "changed"
    with pytest.raises(RuntimeError, match="exact state"):
        compare_replay_snapshots(first, second)
