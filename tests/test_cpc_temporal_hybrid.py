"""Mechanism and invariance checks for Experiment 012's temporal hybrid."""

import numpy as np
import torch
from torch.nn import functional as F  # noqa: N812

from ecg_experiment.cpc_scaling_readout import Example
from ecg_experiment.cpc_temporal_hybrid import (
    TemporalHybridBlock,
    TemporalHybridEncoder,
    TemporalHybridPretrainer,
    causal_fft_conv,
    fold_to_local,
    matched_initial_models,
)
from scripts.experiments.run_cpc_temporal_hybrid012_readout import paired_difference


def test_fft_filter_matches_direct_causal_convolution_and_gradients() -> None:
    """Check linear padding, causal tap order, and both input/filter gradients."""
    torch.manual_seed(12)
    signal = torch.randn(2, 3, 11, dtype=torch.float64, requires_grad=True)
    taps = torch.randn(3, 7, dtype=torch.float64, requires_grad=True)
    actual = causal_fft_conv(signal, taps)
    direct = F.conv1d(F.pad(signal, (6, 0)), taps.flip(-1).unsqueeze(1), groups=3)
    torch.testing.assert_close(actual, direct, atol=1e-12, rtol=1e-12)
    actual_grads = torch.autograd.grad(actual.square().sum(), (signal, taps), retain_graph=True)
    direct_grads = torch.autograd.grad(direct.square().sum(), (signal, taps))
    for found, expected in zip(actual_grads, direct_grads, strict=True):
        torch.testing.assert_close(found, expected, atol=1e-11, rtol=1e-11)


def test_fft_filter_has_no_circular_wrap() -> None:
    """An impulse at the end must not affect the beginning of a half."""
    signal = torch.zeros(1, 1, 9)
    signal[0, 0, -1] = 1
    taps = torch.ones(1, 9)
    output = causal_fft_conv(signal, taps)
    torch.testing.assert_close(output[..., :-1], torch.zeros_like(output[..., :-1]),
                               atol=1e-6, rtol=0)
    torch.testing.assert_close(output[..., -1], torch.ones_like(output[..., -1]))


def test_local_fold_uses_every_parameter_with_local_support() -> None:
    """Local arm retains a gradient path through all long-filter coefficients."""
    taps = torch.arange(1, 18, dtype=torch.float64).reshape(1, -1).requires_grad_()
    folded = fold_to_local(taps)
    assert folded.shape == (1, 5)
    folded.square().sum().backward()
    assert taps.grad is not None
    assert torch.all(taps.grad != 0)


def test_blocks_are_causal_and_parameter_matched() -> None:
    """Changing future tokens leaves earlier outputs identical in both arms."""
    torch.manual_seed(14)
    for kind in ("short", "medium", "long"):
        mixed = TemporalHybridBlock(kind, local_control=False, width=8, token_count=12)
        local = TemporalHybridBlock(kind, local_control=True, width=8, token_count=12)
        local.load_state_dict(mixed.state_dict())
        assert sum(p.numel() for p in mixed.parameters()) == sum(p.numel() for p in local.parameters())
        tokens = torch.randn(2, 12, 8)
        changed = tokens.clone()
        changed[:, 7:] += 3
        for model in (mixed, local):
            torch.testing.assert_close(model(tokens)[:, :7], model(changed)[:, :7],
                                       atol=1e-5, rtol=1e-5)


def test_halves_are_independent_and_pooling_is_512_wide() -> None:
    """No information from the first ECG half enters the second-half context."""
    torch.manual_seed(15)
    encoder = TemporalHybridEncoder(local_control=False).eval()
    signal = torch.randn(1, 12, 2500)
    changed = signal.clone()
    changed[..., :1250] += 2
    with torch.no_grad():
        tokens, contexts = encoder(signal)
        _, altered = encoder(changed)
    assert tokens.shape == contexts.shape == (1, 2, 79, 256)
    assert encoder.pooled(contexts).shape == (1, 512)
    torch.testing.assert_close(contexts[:, 1], altered[:, 1], atol=0, rtol=0)


def test_three_arms_share_initial_stem_and_heads() -> None:
    """Only the temporal-support policy differs between hybrid and local arms."""
    torch.manual_seed(17)
    before = torch.random.get_rng_state()
    models = matched_initial_models(12012)
    torch.testing.assert_close(torch.random.get_rng_state(), before, atol=0, rtol=0)
    gru, mixed, local = (models[name] for name in ("gru", "mixed", "local"))
    for component in ("convs",):
        reference = getattr(gru.encoder, component).state_dict()
        for model in (mixed, local):
            for name, tensor in reference.items():
                torch.testing.assert_close(tensor, getattr(model.encoder, component).state_dict()[name],
                                           atol=0, rtol=0)
    for name, tensor in gru.heads.state_dict().items():
        torch.testing.assert_close(tensor, mixed.heads.state_dict()[name], atol=0, rtol=0)
        torch.testing.assert_close(tensor, local.heads.state_dict()[name], atol=0, rtol=0)
    for name, tensor in mixed.encoder.context.state_dict().items():
        torch.testing.assert_close(tensor, local.encoder.context.state_dict()[name], atol=0, rtol=0)


def test_patient_bootstrap_is_paired_and_reproducible(monkeypatch) -> None:
    """The prespecified development contrast resamples patients, not arms."""
    monkeypatch.setattr("scripts.experiments.run_cpc_temporal_hybrid012_readout.BOOTSTRAP_DRAWS", 40)
    patients = [f"p{i}" for i in range(20) for _ in range(2)]
    labels = np.asarray([i % 2 for i in range(20) for _ in range(2)])
    dev = [Example(f"r{i}", patient, int(labels[i]), {})
           for i, patient in enumerate(patients)]
    mixed = labels * 0.8 + (1 - labels) * 0.2
    local = labels * 0.6 + (1 - labels) * 0.4
    first = paired_difference(dev, mixed, local)
    assert first == paired_difference(dev, mixed, local)
    assert first["valid_draws"] + first["invalid_draws"] == 40
    assert first["difference"] == 0


def test_integrated_cpc_loss_has_finite_gradients() -> None:
    """The hybrid participates in the ordinary causal CPC objective."""
    torch.manual_seed(18)
    model = TemporalHybridPretrainer(local_control=False)
    loss, components = model(torch.randn(2, 12, 2500))
    assert torch.isfinite(loss)
    assert components["cmsc"] == 0.0
    loss.backward()
    assert all(parameter.grad is not None and torch.isfinite(parameter.grad).all()
               for parameter in model.parameters())
