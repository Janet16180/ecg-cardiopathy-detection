"""Scientific and integration checks for the causal CPC mLSTM context."""

from __future__ import annotations

import io
import math

import pytest
import torch

from ecg_experiment.cpc import HALF_SAMPLES, SIGNAL_SAMPLES, TOKEN_COUNT, WIDTH
from ecg_experiment.cpc_xlstm import create_model, mlstm_parallel


def recurrent_reference(query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
                        input_gate: torch.Tensor, forget_gate: torch.Tensor) -> torch.Tensor:
    """Evaluate the unscaled matrix-memory recurrence independently in float64."""
    batch, heads, length, width = query.shape
    memory = query.new_zeros(batch, heads, width, width)
    normalizer = query.new_zeros(batch, heads, width)
    outputs = []
    for step in range(length):
        key_step = key[:, :, step] / math.sqrt(width)
        value_step = value[:, :, step]
        input_step = input_gate[:, :, step].exp()
        forget_step = forget_gate[:, :, step].sigmoid()
        memory = (forget_step[..., None, None] * memory
                  + input_step[..., None, None] * key_step.unsqueeze(-1) * value_step.unsqueeze(-2))
        normalizer = forget_step[..., None] * normalizer + input_step[..., None] * key_step
        query_step = query[:, :, step]
        numerator = (query_step.unsqueeze(-1) * memory).sum(dim=-2)
        denominator = (query_step * normalizer).sum(dim=-1).abs().clamp_min(1)
        outputs.append(numerator / denominator.unsqueeze(-1))
    return torch.stack(outputs, dim=-2)


def test_parallel_matches_independent_recurrence_and_gradients() -> None:
    """The parallel kernel obeys recurrent memory equations and derivatives."""
    torch.manual_seed(10)
    inputs = [torch.randn(2, 3, 11, 8, dtype=torch.float64, requires_grad=True) for _ in range(3)]
    inputs += [torch.randn(2, 3, 11, dtype=torch.float64, requires_grad=True) for _ in range(2)]
    reference_inputs = [tensor.detach().clone().requires_grad_() for tensor in inputs]
    parallel = mlstm_parallel(*inputs, eps=0)
    recurrent = recurrent_reference(*reference_inputs)
    torch.testing.assert_close(parallel, recurrent, rtol=1e-10, atol=1e-10)
    weights = torch.randn_like(parallel)
    (parallel * weights).sum().backward()
    (recurrent * weights).sum().backward()
    for actual, expected in zip(inputs, reference_inputs, strict=True):
        torch.testing.assert_close(actual.grad, expected.grad, rtol=1e-9, atol=1e-9)


@pytest.mark.parametrize("arm", ["gru", "xlstm"])
def test_context_shapes_prefix_causality_and_independent_halves(arm: str) -> None:
    """No context may use a future token or the other five-second half."""
    model = create_model(arm, 42).eval()
    signal = torch.randn(2, 12, SIGNAL_SAMPLES)
    changed_half = signal.clone()
    changed_half[:, :, HALF_SAMPLES:] = torch.randn_like(changed_half[:, :, HALF_SAMPLES:])
    with torch.no_grad():
        tokens, contexts = model.encoder(signal)
        changed_tokens, changed_contexts = model.encoder(changed_half)
        assert tokens.shape == contexts.shape == (2, 2, TOKEN_COUNT, WIDTH)
        torch.testing.assert_close(tokens[:, 0], changed_tokens[:, 0], rtol=0, atol=0)
        torch.testing.assert_close(contexts[:, 0], changed_contexts[:, 0], rtol=0, atol=0)
        context_input = torch.randn(2, TOKEN_COUNT, WIDTH)
        changed = context_input.clone()
        changed[:, 29:] += 7 * torch.randn_like(changed[:, 29:])
        original_context, _ = model.encoder.context(context_input)
        changed_context, _ = model.encoder.context(changed)
        torch.testing.assert_close(original_context[:, :29], changed_context[:, :29], rtol=0, atol=0)
        assert model.encoder.pooled(contexts).shape == (2, 2 * WIDTH)


def test_common_initialization_and_context_parameter_counts() -> None:
    """The experiment changes only context tensors, with measured capacity."""
    initial_rng = torch.get_rng_state().clone()
    gru = create_model("gru", 38042)
    xlstm = create_model("xlstm", 38042)
    assert torch.equal(initial_rng, torch.get_rng_state())
    common = {name: tensor for name, tensor in gru.state_dict().items()
              if not name.startswith("encoder.context.")}
    for name, tensor in common.items():
        assert torch.equal(tensor, xlstm.state_dict()[name]), name
    assert sum(parameter.numel() for parameter in gru.encoder.context.parameters()) == 789504
    assert sum(parameter.numel() for parameter in xlstm.encoder.context.parameters()) == 831248
    repeated = create_model("xlstm", 38042)
    for name, tensor in xlstm.state_dict().items():
        assert torch.equal(tensor, repeated.state_dict()[name]), name


def test_cpc_loss_and_all_gradients_are_finite() -> None:
    """The inherited CPC objective trains every parameter in the replacement."""
    model = create_model("xlstm", 38042)
    loss, components = model(torch.randn(2, 12, SIGNAL_SAMPLES))
    assert torch.isfinite(loss)
    assert math.isfinite(components["cpc"])
    assert components["cmsc"] == 0
    loss.backward()
    for name, parameter in model.named_parameters():
        assert parameter.grad is not None, name
        assert torch.isfinite(parameter.grad).all(), name


def test_checkpoint_resume_reproduces_next_optimizer_step_exactly() -> None:
    """Restoring model, optimizer and RNG reproduces dropout and next update."""
    torch.manual_seed(38)
    model = create_model("xlstm", 38042)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
    signal = torch.randn(2, 12, SIGNAL_SAMPLES)
    optimizer.zero_grad(set_to_none=True)
    model(signal)[0].backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
    optimizer.step()
    checkpoint = io.BytesIO()
    torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                "rng": torch.get_rng_state()}, checkpoint)
    optimizer.zero_grad(set_to_none=True)
    expected_loss = model(signal)[0]
    expected_loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
    optimizer.step()
    checkpoint.seek(0)
    saved = torch.load(checkpoint, weights_only=True)
    resumed = create_model("xlstm", 0)
    resumed.load_state_dict(saved["model"])
    resumed_optimizer = torch.optim.AdamW(resumed.parameters(), lr=1e-4, weight_decay=0.01)
    resumed_optimizer.load_state_dict(saved["optimizer"])
    torch.set_rng_state(saved["rng"])
    resumed_optimizer.zero_grad(set_to_none=True)
    resumed_loss = resumed(signal)[0]
    resumed_loss.backward()
    torch.nn.utils.clip_grad_norm_(resumed.parameters(), 1)
    resumed_optimizer.step()
    assert torch.equal(expected_loss, resumed_loss)
    for name, tensor in model.state_dict().items():
        assert torch.equal(tensor, resumed.state_dict()[name]), name


def test_unknown_arm_is_rejected() -> None:
    """A misspelled arm cannot silently become a different experiment."""
    with pytest.raises(ValueError, match="Unknown CPC arm"):
        create_model("lstm", 42)


def test_extreme_finite_gates_have_finite_outputs_and_gradients() -> None:
    """Large exponential-gate logits stay finite and cannot leak future tokens."""
    torch.manual_seed(38)
    inputs = [torch.randn(2, 4, 17, 8, requires_grad=True) for _ in range(3)]
    gate_values = torch.linspace(-80, 80, 17).expand(2, 4, -1)
    inputs += [gate_values.clone().requires_grad_(), gate_values.flip(-1).clone().requires_grad_()]
    output = mlstm_parallel(*inputs)
    assert torch.isfinite(output).all()
    changed = [tensor.detach().clone() for tensor in inputs]
    for tensor in changed:
        tensor[:, :, 9:] = tensor[:, :, 9:].flip(2)
    torch.testing.assert_close(output[:, :, :9], mlstm_parallel(*changed)[:, :, :9], rtol=0, atol=0)
    output.square().mean().backward()
    for tensor in inputs:
        assert tensor.grad is not None
        assert torch.isfinite(tensor.grad).all()
