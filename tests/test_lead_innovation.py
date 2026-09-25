"""Checks that hidden raw samples stay hidden and only the student is optimized."""

import pytest
import torch

from ecg_experiment.lead_innovation import (
    LeadMultiscaleEncoder,
    LeadSSL,
    content_targets,
    lead_patch_mask,
)
from ecg_experiment.models import Classifier


def test_static_lead_time_pattern_has_zero_targets():
    torch.manual_seed(9)
    constant_position_pattern = torch.randn(1, 8, 25, 96).repeat(4, 1, 1, 1)
    ordinary, innovation, centered, residual = content_targets(constant_position_pattern)
    assert torch.equal(ordinary, torch.zeros_like(ordinary))
    assert torch.equal(innovation, torch.zeros_like(innovation))
    assert torch.equal(centered, torch.zeros_like(centered))
    assert torch.equal(residual, torch.zeros_like(residual))


def test_mask_visibility_and_gradient_flow():
    torch.set_num_threads(1)
    torch.manual_seed(7)
    signal = torch.randn(3, 8, 1000)
    mask = lead_patch_mask(3, signal.device)
    assert mask.shape == (3, 8, 25)
    assert torch.all(mask.sum(dim=(1, 2)) == 35)

    encoder = LeadMultiscaleEncoder().eval()
    changed = signal.clone()
    changed[mask.repeat_interleave(40, dim=-1)] = 1000.0
    with torch.no_grad():
        assert torch.allclose(encoder.tokens(signal, mask), encoder.tokens(changed, mask))
        assert encoder(signal).shape == (3, 192)

    model = LeadSSL(0.5)
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=1e-3)
    before = model.encoder.morphology.weight.detach().clone()
    loss, _ = model(signal)
    assert torch.isfinite(loss).item()
    loss.backward()
    assert all(p.grad is None for p in model.teacher.parameters())
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
    optimizer.step()
    assert not torch.equal(before, model.encoder.morphology.weight)
    assert torch.isfinite(model.encoder.morphology.weight).all()


def test_rejects_unsupported_weight_single_record_and_bad_shapes():
    with pytest.raises(ValueError, match="0 or 0.5"):
        LeadSSL(0.25)
    with pytest.raises(ValueError, match="at least two records"):
        LeadSSL(0.0)(torch.randn(1, 8, 1000))
    encoder = LeadMultiscaleEncoder()
    with pytest.raises(ValueError, match="Expected"):
        encoder.tokens(torch.randn(2, 12, 1000))
    with pytest.raises(ValueError, match="mask shape"):
        encoder.tokens(torch.randn(2, 8, 1000), torch.zeros(2, 8, 24, dtype=torch.bool))


def test_teacher_update_interpolates_toward_student():
    model = LeadSSL(0.0)
    with torch.no_grad():
        for parameter in model.encoder.parameters():
            parameter.add_(1.0)
    teacher_before = [p.clone() for p in model.teacher.parameters()]
    model.update_teacher(0.75)
    for old, new, online in zip(teacher_before, model.teacher.parameters(), model.encoder.parameters(),
                                strict=True):
        torch.testing.assert_close(new, old + 0.25 * (online - old))


def test_classifier_returns_one_logit_per_record():
    classifier = Classifier(LeadMultiscaleEncoder()).eval()
    with torch.no_grad():
        assert classifier(torch.randn(3, 8, 1000)).shape == (3,)
