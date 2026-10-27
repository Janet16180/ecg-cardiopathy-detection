"""Check matched initialization, joint masks and actual optimization behavior on artificial data."""

import copy

import numpy as np
import pytest
import torch

from ecg_experiment.ssl_synthetic_model import (
    DenseWaveformModel,
    localization_update,
    masked_waveforms,
    matched_schedule,
    ssl_update,
    state_digest,
)


def test_joint_masks_preserve_exact_coverage_and_replay():
    signal = torch.ones(3, 12, 2500)
    first = masked_waveforms(signal, torch.Generator().manual_seed(12))
    second = masked_waveforms(signal, torch.Generator().manual_seed(12))
    assert torch.equal(first[0], second[0])
    assert torch.equal(first[1], second[1])
    assert torch.equal(first[1][:, 0], first[1][:, 11])
    assert torch.all(first[1].sum(dim=-1) == 650)
    assert torch.all(first[0][first[1]] == 0)
    assert torch.all(first[0][~first[1]] == 1)


def test_mask_cannot_remove_all_context_or_leave_partial_blocks():
    with pytest.raises(ValueError, match="visible samples"):
        masked_waveforms(torch.ones(1, 12, 2499), torch.Generator())
    with pytest.raises(ValueError, match="visible samples"):
        masked_waveforms(torch.ones(1, 12, 2500), torch.Generator(), blocks=50)


def test_model_outputs_native_grid_and_all_encoder_parameters_receive_gradient():
    torch.manual_seed(11)
    model = DenseWaveformModel()
    signal = torch.randn(2, 12, 2500)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
    original = state_digest(model.encoder.state_dict())
    loss = ssl_update(model, optimizer, signal, torch.Generator().manual_seed(12))
    assert np.isfinite(loss)
    assert model(signal).shape == signal.shape
    assert state_digest(model.encoder.state_dict()) != original
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.encoder.parameters())


def test_frozen_arm_changes_head_but_keeps_every_encoder_byte():
    torch.manual_seed(11)
    model = DenseWaveformModel()
    for parameter in model.encoder.parameters():
        parameter.requires_grad_(False)
    encoder = state_digest(model.encoder.state_dict())
    head = state_digest(model.head.state_dict())
    optimizer = torch.optim.AdamW(model.head.parameters(), lr=0.001)
    signal = torch.randn(2, 12, 2500)
    support = torch.rand_like(signal) > 0.9
    loss = localization_update(model, optimizer, signal, support, 10.0, frozen_encoder=True)
    assert np.isfinite(loss)
    assert not model.encoder.training
    assert state_digest(model.encoder.state_dict()) == encoder
    assert state_digest(model.head.state_dict()) != head


def test_frozen_flag_cannot_silently_update_encoder():
    model = DenseWaveformModel()
    optimizer = torch.optim.AdamW(model.parameters())
    signal = torch.ones(1, 12, 2500)
    with pytest.raises(ValueError, match="non-trainable encoder"):
        localization_update(model, optimizer, signal, signal.bool(), 1.0, frozen_encoder=True)


def test_cpu_optimizer_and_masking_replay_are_exact():
    torch.manual_seed(11)
    initial = DenseWaveformModel()
    signal = torch.randn(2, 12, 2500)
    outputs = []
    for _ in range(2):
        model = copy.deepcopy(initial)
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
        masks = torch.Generator().manual_seed(15)
        losses = [ssl_update(model, optimizer, signal, masks) for _ in range(3)]
        outputs.append((losses, state_digest(model.state_dict())))
    assert outputs[0] == outputs[1]


def test_matched_schedule_has_frozen_class_composition_and_cycles_every_pool():
    kinds = np.array(["positive"] * 375 + ["sham"] * 63 + ["nuisance"] * 62)
    first = matched_schedule(kinds, 600, 15)
    assert np.array_equal(first, matched_schedule(kinds, 600, 15))
    assert first.shape == (600, 16)
    assert np.all(kinds[first[:, :12]] == "positive")
    assert np.all(kinds[first[:, 12:14]] == "sham")
    assert np.all(kinds[first[:, 14:]] == "nuisance")
    assert len(np.unique(first)) == 500
