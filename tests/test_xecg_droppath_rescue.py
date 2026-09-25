"""Mechanism and paired-RNG checks for Experiment 016 rescue."""

import weakref

import pytest
import torch
from torch import nn

from ecg_experiment.xecg_droppath_rescue import PairedDropPath
from ecg_experiment.xecg_rescue_training import (
    compare_replay_snapshots,
    layerwise_parameter_groups,
    restore_checkpoint,
    save_checkpoint,
    sequential_replay,
)
from scripts.experiments.run_xecg_droppath_rescue016 import environment_metadata
from scripts.experiments.run_xecg_probe_finetune016 import (
    layerwise_parameter_groups as historical_groups,
)


class AffineResidual(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(2.0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.weight * x


def test_complete_block_formula_and_paired_masks():
    x = torch.arange(1.0, 129.0).view(128, 1, 1)
    block = AffineResidual()
    legacy, residual = PairedDropPath("legacy", 16016), PairedDropPath("residual", 16016)
    legacy.train()
    residual.train()
    a, b = legacy(x, block, 0.5), residual(x, block, 0.5)
    mask = (a != x).float()
    torch.testing.assert_close(a, mask * (block(x) / 0.5) + (1 - mask) * x)
    torch.testing.assert_close(b, x + mask * (block(x) - x) / 0.5)
    assert torch.equal(legacy.get_rng_state(), residual.get_rng_state())
    assert legacy.draws == residual.draws == len(x)
    # Algebraic conditional means over the two possible masks.
    torch.testing.assert_close(0.5 * block(x) / 0.5 + 0.5 * x, block(x) + 0.5 * x)
    torch.testing.assert_close(0.5 * (x + (block(x) - x) / 0.5) + 0.5 * x, block(x))


def test_identity_eval_off_and_gradients():
    x = torch.ones(128, 1, 1, requires_grad=True)
    identity = nn.Identity()
    residual = PairedDropPath("residual", 42).train()
    torch.testing.assert_close(residual(x, identity, 0.5), x)
    block = AffineResidual()
    output = residual(x, block, 0.5)
    output.sum().backward()
    assert x.grad is not None
    assert torch.isfinite(x.grad).all()
    assert block.weight.grad is not None
    assert block.weight.grad.item() > 0
    residual.eval()
    torch.testing.assert_close(residual(x, block, 0.5), block(x))
    residual.train()
    torch.testing.assert_close(residual(x, block, 0), block(x))
    off = PairedDropPath("off", 42).train()
    torch.testing.assert_close(off(x, block, 0.5), block(x))


def test_optimizer_groups_match_historical_016():
    model = nn.Module()
    model.backbone = nn.Module()
    model.backbone.patch_embedding = nn.Linear(2, 2)
    model.backbone.core = nn.Module()
    model.backbone.core.model = nn.Module()
    model.backbone.core.model.blocks = nn.ModuleList([nn.Linear(2, 2) for _ in range(9)])
    model.backbone.other = nn.Linear(2, 2)
    model.head = nn.Linear(2, 1)
    actual = layerwise_parameter_groups(model)
    expected = historical_groups(model)
    for left, right in zip(actual, expected, strict=True):
        assert left["name"] == right["name"]
        assert left["lr"] == right["lr"]
        assert left["weight_decay"] == right["weight_decay"]
        assert [id(p) for p in left["params"]] == [id(p) for p in right["params"]]


def test_checkpoint_restores_exact_update_state(tmp_path):
    environment = environment_metadata()
    assert all(type(value) is str for value in environment.values())
    fingerprint = {"environment": environment}
    model = nn.Linear(2, 1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
    mask = PairedDropPath("residual", 42)
    permutation = torch.Generator().manual_seed(42)
    loss = model(torch.ones(2, 2)).sum()
    loss.backward()
    optimizer.step()
    scheduler.step()
    order = torch.randperm(8, generator=permutation)
    path = tmp_path / "state.pt"
    save_checkpoint(path, fingerprint, model, mask, optimizer, scheduler, permutation, 0, 4, order, 1)
    restored_model = nn.Linear(2, 1)
    restored_optimizer = torch.optim.AdamW(restored_model.parameters(), lr=1e-3)
    restored_scheduler = torch.optim.lr_scheduler.LambdaLR(restored_optimizer, lambda _: 1.0)
    restored_mask = PairedDropPath("residual", 1)
    restored_permutation = torch.Generator().manual_seed(1)
    saved = restore_checkpoint(
        path,
        fingerprint,
        restored_model,
        restored_mask,
        restored_optimizer,
        restored_scheduler,
        restored_permutation,
    )
    assert saved["next_index"] == 4
    assert saved["updates"] == 1
    for left, right in zip(model.parameters(), restored_model.parameters(), strict=True):
        torch.testing.assert_close(left, right)
    torch.testing.assert_close(mask.get_rng_state(), restored_mask.get_rng_state())
    torch.testing.assert_close(permutation.get_state(), restored_permutation.get_state())


def test_replay_updates_sequentially_and_detects_difference():
    live: list[weakref.ReferenceType] = []
    calls = 0

    def run_once() -> dict:
        nonlocal calls
        assert not any(reference() is not None for reference in live)
        calls += 1
        torch.manual_seed(42)
        model = nn.Linear(2, 1)
        live.append(weakref.ref(model))
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        model(torch.ones(2, 2)).sum().backward()
        optimizer.step()
        result = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
        del model, optimizer
        return result

    sequential_replay(run_once, "cpu")
    assert calls == 2

    def different_once() -> dict:
        result = run_once()
        if calls == 4:
            result["bias"] += 1
        return result

    with pytest.raises(RuntimeError, match="Replay exact state differs"):
        sequential_replay(different_once, "cpu")


def test_replay_accepts_tiny_float_drift_but_rejects_material_changes():
    first = {
        "model": {"weight": torch.tensor([0.0], dtype=torch.float64)},
        "optimizer": {
            "state": {0: {"exp_avg": torch.tensor([0.0], dtype=torch.float64), "step": torch.tensor(1.0)}}
        },
        "scheduler": {"step": 1},
        "batch_order_sha256": "same",
        "mask_rng": torch.tensor([1], dtype=torch.uint8),
    }
    second = {
        "model": {"weight": torch.tensor([5e-9], dtype=torch.float64)},
        "optimizer": {
            "state": {0: {"exp_avg": torch.tensor([1e-10], dtype=torch.float64), "step": torch.tensor(1.0)}}
        },
        "scheduler": {"step": 1},
        "batch_order_sha256": "same",
        "mask_rng": torch.tensor([1], dtype=torch.uint8),
    }
    result = compare_replay_snapshots(first, second)
    assert result["tensors_with_numeric_difference"] == 2
    assert result["max_abs_difference"] == 5e-9
    second["model"]["weight"] = torch.tensor([1e-5], dtype=torch.float64)
    with pytest.raises(RuntimeError, match="numeric tolerance exceeded"):
        compare_replay_snapshots(first, second)
    second["model"]["weight"] = first["model"]["weight"]
    second["optimizer"]["state"][0]["step"] = torch.tensor(2.0)
    with pytest.raises(RuntimeError, match="update-step tensor differs"):
        compare_replay_snapshots(first, second)
