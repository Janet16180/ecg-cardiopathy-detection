"""Epoch-boundary fine-tuning checkpoints preserve optimizer and RNG state."""

import random

import numpy as np
import pytest
import torch
from torch import nn

from ecg_experiment.reproducibility import (
    cpu_state,
    flat_rng_state,
    restore_flat_rng_state,
    restore_rng_lists,
    rng_state_lists,
)
from ecg_experiment.training import optimizer_state_bytes, peak_gpu_bytes, require_cuda
from scripts.experiments.finetune_pretrained import (
    cache_shape,
    load_resume_checkpoint,
    run_name,
    save_resume_checkpoint,
)


def new_training():
    model = nn.Sequential(nn.Linear(3, 8), nn.ReLU(), nn.Dropout(0.2), nn.Linear(8, 1))
    return model, torch.optim.AdamW(model.parameters(), lr=0.01)


def train_epoch(model, optimizer):
    model.train()
    inputs = torch.arange(24, dtype=torch.float32).reshape(8, 3) / 10
    for index in torch.randperm(8).split(4):
        scale = random.random() + float(np.random.random())
        prediction = model(inputs[index] * scale)
        loss = prediction.square().mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def test_interrupted_training_matches_uninterrupted_training(tmp_path):
    path = tmp_path / "resume.pt"
    fingerprint = {"manifest_sha256": {"labeled_train.csv": "test"}, "epochs": 4}
    seed_all(7)
    uninterrupted, uninterrupted_optimizer = new_training()
    for _ in range(4):
        train_epoch(uninterrupted, uninterrupted_optimizer)
    expected = cpu_state(uninterrupted)
    expected_draws = (random.random(), float(np.random.random()), torch.rand(1))

    seed_all(7)
    interrupted, interrupted_optimizer = new_training()
    for _ in range(2):
        train_epoch(interrupted, interrupted_optimizer)
    save_resume_checkpoint(path, fingerprint, interrupted, interrupted_optimizer,
                           cpu_state(interrupted), 2, 0.7, [{"epoch": 1}, {"epoch": 2}], 12.0)
    assert [item.name for item in tmp_path.iterdir()] == ["resume.pt"]

    seed_all(999)
    resumed, resumed_optimizer = new_training()
    with pytest.raises(ValueError, match="configuration differs"):
        load_resume_checkpoint(path, {"epochs": 5}, resumed, resumed_optimizer)
    checkpoint = load_resume_checkpoint(path, fingerprint, resumed, resumed_optimizer)
    assert checkpoint["elapsed_seconds"] == 12.0
    for _ in range(2):
        train_epoch(resumed, resumed_optimizer)
    for name, parameter in expected.items():
        torch.testing.assert_close(resumed.state_dict()[name], parameter, rtol=0, atol=0)
    assert random.random() == expected_draws[0]
    assert float(np.random.random()) == expected_draws[1]
    torch.testing.assert_close(torch.rand(1), expected_draws[2], rtol=0, atol=0)


def test_resume_rejects_best_epoch_outside_history(tmp_path):
    model, optimizer = new_training()
    path = tmp_path / "resume.pt"
    save_resume_checkpoint(path, {}, model, optimizer, cpu_state(model), 2, 0.7, [{"epoch": 1}], 1.0)
    with pytest.raises(ValueError, match="Malformed"):
        load_resume_checkpoint(path, {}, model, optimizer)


def test_flat_rng_state_uses_legacy_keys_and_roundtrips():
    seed_all(11)
    state = flat_rng_state()
    assert list(state) == ["torch_rng", "cuda_rng", "python_rng", "numpy_rng"]
    assert isinstance(state["numpy_rng"][1], list)
    expected = (random.random(), float(np.random.random()), torch.rand(1))
    seed_all(12)
    restore_flat_rng_state(state)
    assert (random.random(), float(np.random.random())) == expected[:2]
    torch.testing.assert_close(torch.rand(1), expected[2], rtol=0, atol=0)


def test_nested_rng_state_roundtrips_through_torch_save(tmp_path):
    seed_all(13)
    torch.save(rng_state_lists(), tmp_path / "rng.pt")
    expected = (random.random(), float(np.random.random()), torch.rand(1))
    seed_all(14)
    restore_rng_lists(torch.load(tmp_path / "rng.pt", weights_only=True))
    assert (random.random(), float(np.random.random())) == expected[:2]
    torch.testing.assert_close(torch.rand(1), expected[2], rtol=0, atol=0)


def test_device_helpers_on_cpu(monkeypatch):
    model, optimizer = new_training()
    assert optimizer_state_bytes(optimizer) == 0
    model(torch.ones(2, 3)).sum().backward()
    optimizer.step()
    parameters = sum(param.numel() for param in model.parameters())
    # AdamW keeps float32 exp_avg and exp_avg_sq per parameter plus a scalar step.
    assert optimizer_state_bytes(optimizer) == 8 * parameters + 4 * len(list(model.parameters()))
    assert peak_gpu_bytes("cpu") is None
    require_cuda("cpu")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="CUDA requested but unavailable"):
        require_cuda("cuda")


def test_cache_shape_and_run_name(tmp_path):
    assert cache_shape("hubert-small", 3) == (3, 2, 6000)
    assert cache_shape("ecg-fm", 3) == (3, 2, 12, 2500)

    class Args:
        model = "ecg-fm"
        adapted_backbone = None

    assert run_name(Args, None) == "ecg-fm_finetuned"
    Args.adapted_backbone = tmp_path / "adapted.pt"
    assert run_name(Args, {"external_ssl": None}) == "ecg-fm_adapted_finetuned"
    assert run_name(Args, {"external_ssl": {"records": 1}}) == "ecg-fm_pooled_adapted_finetuned"
