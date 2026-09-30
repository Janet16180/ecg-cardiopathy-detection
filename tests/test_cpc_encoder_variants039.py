"""Scientific invariants for matched causal CPC waveform encoder variants."""

from __future__ import annotations

import pytest
import torch

from ecg_experiment.cpc import HALF_SAMPLES, HORIZONS, SIGNAL_SAMPLES, TOKEN_COUNT, WIDTH
from ecg_experiment.cpc_encoder_variants039 import (
    CONTEXTS,
    ENCODERS,
    TOKEN_STRIDE,
    architecture_spec,
    create_model,
)
from ecg_experiment.cpc_xlstm_native_gru import NativeGRU
from ecg_experiment.cpc_xlstm_native_gru import create_model as native_create_model


@pytest.fixture(autouse=True)
def single_cpu_thread() -> None:
    torch.set_num_threads(1)


@pytest.mark.parametrize("encoder", ENCODERS)
@pytest.mark.parametrize("context", CONTEXTS)
def test_sample_prefix_causality_and_independent_halves(encoder: str, context: str) -> None:
    model = create_model(encoder, context, 39042).eval()
    signal = torch.randn(2, 12, SIGNAL_SAMPLES)
    with torch.inference_mode():
        tokens, contexts = model.encoder(signal)
        assert tokens.shape == contexts.shape == (2, 2, TOKEN_COUNT, WIDTH)
        assert model.encoder.pooled(contexts).shape == (2, 2 * WIDTH)
        for half in (0, 1):
            changed_half = signal.clone()
            start = half * HALF_SAMPLES
            changed_half[:, :, start : start + HALF_SAMPLES] += 10 * torch.randn(2, 12, HALF_SAMPLES)
            half_tokens, half_contexts = model.encoder(changed_half)
            torch.testing.assert_close(half_tokens[:, 1 - half], tokens[:, 1 - half], rtol=0, atol=0)
            torch.testing.assert_close(half_contexts[:, 1 - half], contexts[:, 1 - half], rtol=0, atol=0)
        for cutoff in (0, 16, 517, 1248):
            changed_future = signal.clone()
            for half in (0, 1):
                start, end = half * HALF_SAMPLES + cutoff + 1, (half + 1) * HALF_SAMPLES
                changed_future[:, :, start:end] += 10 * torch.randn_like(signal[:, :, start:end])
            future_tokens, future_contexts = model.encoder(changed_future)
            valid = torch.arange(TOKEN_COUNT) * TOKEN_STRIDE <= cutoff
            torch.testing.assert_close(future_tokens[:, :, valid], tokens[:, :, valid], rtol=0, atol=0)
            torch.testing.assert_close(future_contexts[:, :, valid], contexts[:, :, valid], rtol=0, atol=0)


@pytest.mark.parametrize("context", CONTEXTS)
def test_context_and_heads_match_across_encoder_arms(context: str) -> None:
    baseline = native_create_model(context, 39042)
    expected = {
        name: tensor
        for name, tensor in baseline.state_dict().items()
        if name.startswith(("encoder.context.", "heads."))
    }
    for encoder in ENCODERS:
        model = create_model(encoder, context, 39042)
        for name, tensor in expected.items():
            assert torch.equal(tensor, model.state_dict()[name]), name
        if context == "gru":
            assert isinstance(model.encoder.context, NativeGRU)


@pytest.mark.parametrize("encoder", ENCODERS)
def test_frontend_and_heads_match_across_context_arms(encoder: str) -> None:
    gru = create_model(encoder, "gru", 39042)
    xlstm = create_model(encoder, "xlstm", 39042)
    expected = {
        name: tensor for name, tensor in gru.state_dict().items() if not name.startswith("encoder.context.")
    }
    for name, tensor in expected.items():
        assert torch.equal(tensor, xlstm.state_dict()[name]), name


@pytest.mark.parametrize("context", CONTEXTS)
def test_cnn_exactly_matches_native_frozen_factory(context: str) -> None:
    actual = create_model("cnn", context, 39042).eval()
    expected = native_create_model(context, 39042).eval()
    assert actual.state_dict().keys() == expected.state_dict().keys()
    for name, tensor in expected.state_dict().items():
        assert torch.equal(tensor, actual.state_dict()[name]), name
    signal = torch.randn(2, 12, SIGNAL_SAMPLES)
    with torch.inference_mode():
        actual_outputs = actual.encoder(signal)
        expected_outputs = expected.encoder(signal)
    for observed, reference in zip(actual_outputs, expected_outputs, strict=True):
        torch.testing.assert_close(observed, reference, rtol=0, atol=0)


@pytest.mark.parametrize("encoder", ENCODERS)
@pytest.mark.parametrize("context", CONTEXTS)
def test_initialization_is_deterministic_and_does_not_advance_rng(encoder: str, context: str) -> None:
    before = torch.get_rng_state().clone()
    first = create_model(encoder, context, 39042)
    second = create_model(encoder, context, 39042)
    assert torch.equal(before, torch.get_rng_state())
    for name, tensor in first.state_dict().items():
        assert torch.equal(tensor, second.state_dict()[name]), name
    different_seed = create_model(encoder, context, 39043)
    assert any(
        not torch.equal(tensor, different_seed.state_dict()[name])
        for name, tensor in first.state_dict().items()
    )


@pytest.mark.parametrize("encoder", ENCODERS)
@pytest.mark.parametrize("context", CONTEXTS)
def test_finite_cpc_loss_and_backward(encoder: str, context: str) -> None:
    model = create_model(encoder, context, 39042)
    loss, components = model(torch.randn(2, 12, SIGNAL_SAMPLES))
    assert torch.isfinite(loss)
    assert components["cmsc"] == 0
    assert components["cpc"] == float(loss.detach())
    loss.backward()
    for name, parameter in model.named_parameters():
        assert parameter.grad is not None, name
        assert torch.isfinite(parameter.grad).all(), name


@pytest.mark.parametrize(("encoder", "support"), [("cnn", 33), ("multiscale", 57), ("patch", 33)])
def test_frontend_actual_receptive_field_and_last_sample_alignment(encoder: str, support: int) -> None:
    model = create_model(encoder, "gru", 39042).eval()
    signal = torch.randn(1, 12, HALF_SAMPLES, requires_grad=True)
    tokens = model.encoder.convs(signal)
    assert tokens.shape == (1, WIDTH, TOKEN_COUNT)
    for index in (0, 31, TOKEN_COUNT - 1):
        (gradient,) = torch.autograd.grad(tokens[0, 0, index], signal, retain_graph=True)
        active = gradient[0].abs().sum(dim=0).nonzero().flatten()
        end = index * TOKEN_STRIDE
        assert int(active.min()) == max(0, end - support + 1)
        assert int(active.max()) == end
        assert len(active) == min(support, end + 1)


@pytest.mark.parametrize("encoder", ENCODERS)
@pytest.mark.parametrize("context", CONTEXTS)
def test_receipt_spec_matches_measured_parameters(encoder: str, context: str) -> None:
    spec = architecture_spec(encoder, context)
    model = create_model(encoder, context, 39042)
    assert spec["total_parameters"] == sum(parameter.numel() for parameter in model.parameters())
    assert spec["total_parameters"] < 3_000_000
    assert spec["token_end_samples"] == list(range(0, 1250, 16))
    assert spec["token_spacing_ms"] == 64
    assert (
        spec["frontend_parameters"] + spec["context_parameters"] + spec["prediction_head_parameters"]
        == (spec["total_parameters"])
    )


@pytest.mark.parametrize(("encoder", "context"), [("unknown", "gru"), ("cnn", "unknown")])
def test_unknown_architectures_raise(encoder: str, context: str) -> None:
    with pytest.raises(ValueError, match="Unknown CPC"):
        create_model(encoder, context, 39042)


@pytest.mark.parametrize("encoder", ENCODERS)
def test_future_targets_have_strictly_disjoint_raw_support(encoder: str) -> None:
    spec = architecture_spec(encoder, "gru")
    for horizon in HORIZONS:
        for query in range(TOKEN_COUNT - horizon):
            query_end = query * TOKEN_STRIDE
            target_end = (query + horizon) * TOKEN_STRIDE
            target_start = max(0, target_end - spec["receptive_field_samples"] + 1)
            assert target_start > query_end
