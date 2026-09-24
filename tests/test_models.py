"""Shapes, determinism and masking invariants of the compact SSL encoders."""

import pytest
import torch

from ecg_experiment.models import CNN, JEPA, Classifier, MaskedAutoencoder, PatchTransformer, random_mask


@pytest.mark.parametrize(("factory", "width"), [(CNN, 256), (PatchTransformer, 192)])
def test_encoders_pool_to_feature_dim_and_classifier_gives_one_logit(factory, width):
    torch.manual_seed(0)
    encoder = factory().eval()
    signal = torch.randn(2, 12, 1000)
    with torch.no_grad():
        assert encoder(signal).shape == (2, width)
        assert Classifier(encoder).eval()(signal).shape == (2,)


def test_same_seed_gives_identical_initialization_and_loss():
    signal = torch.randn(2, 12, 1000, generator=torch.Generator().manual_seed(1))
    losses = []
    for _ in range(2):
        torch.manual_seed(3)
        model = MaskedAutoencoder()
        loss, details = model(signal)
        losses.append(loss)
        assert details["reconstruction"] == float(loss)
    assert torch.equal(losses[0], losses[1])


def test_random_mask_hides_exactly_half_in_two_patch_blocks():
    torch.manual_seed(0)
    mask = random_mask(4, "cpu")
    assert mask.shape == (4, 40)
    assert torch.all(mask.sum(dim=1) == 20)
    assert torch.equal(mask[:, 0::2], mask[:, 1::2])


def test_patch_mask_replaces_masked_tokens_before_encoding():
    torch.manual_seed(0)
    encoder = PatchTransformer().eval()
    signal = torch.randn(1, 12, 1000)
    changed = signal.clone()
    changed[:, :, :25] = 100.0
    mask = torch.zeros(1, 40, dtype=torch.bool)
    mask[:, 0] = True
    with torch.no_grad():
        torch.testing.assert_close(encoder.tokens(signal, mask), encoder.tokens(changed, mask),
                                   atol=0, rtol=0)


def test_jepa_optimizes_only_the_student_and_ema_moves_teacher():
    torch.manual_seed(0)
    model = JEPA()
    loss, details = model(torch.randn(3, 12, 1000))
    loss.backward()
    assert torch.isfinite(loss)
    assert set(details) == {"latent_prediction", "variance_penalty", "feature_std"}
    assert all(p.grad is None for p in model.teacher.parameters())
    with torch.no_grad():
        for parameter in model.encoder.parameters():
            parameter.add_(1.0)
    before = [p.clone() for p in model.teacher.parameters()]
    model.update_teacher(0.5)
    for old, new, online in zip(before, model.teacher.parameters(), model.encoder.parameters(), strict=True):
        torch.testing.assert_close(new, 0.5 * old + 0.5 * online)
