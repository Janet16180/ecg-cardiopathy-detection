"""Correctness checks for the Experiment 011 KDA/CKDA reference recurrence."""

import torch

from ecg_experiment.cpc_delta_memory import (
    DeltaCPCEncoder,
    DeltaMemoryBlock,
    matched_initial_models,
    recurrent_delta_memory,
)


def _matrix_reference(queries, keys, values, decay, write_rate):
    """Apply the paper's explicit transition matrix instead of the delta form."""
    batch, length, heads, key_width = keys.shape
    state = values.new_zeros(batch, heads, key_width, values.shape[-1])
    identity = torch.eye(key_width, dtype=values.dtype)
    outputs = []
    for time in range(length):
        key = keys[:, time]
        rate = write_rate[:, time]
        transition = identity - rate[..., None] * key[..., :, None] * key[..., None, :]
        decayed = decay[:, time, :, :, None] * state
        written = rate[..., None] * key[..., :, None] * values[:, time, :, None, :]
        state = transition @ decayed + written
        outputs.append(torch.einsum("bhk,bhkv->bhv", queries[:, time], state))
    return torch.stack(outputs, dim=1)


def test_delta_scan_matches_matrix_transition_and_gradients() -> None:
    torch.manual_seed(11)
    query = torch.nn.functional.normalize(torch.randn(2, 5, 2, 3, dtype=torch.float64), dim=-1)
    key = torch.nn.functional.normalize(torch.randn(2, 5, 2, 3, dtype=torch.float64), dim=-1)
    value = torch.randn(2, 5, 2, 4, dtype=torch.float64)
    decay = torch.empty(2, 5, 2, 3, dtype=torch.float64).uniform_(-0.9, 0.9)
    rate = torch.empty(2, 5, 2, 1, dtype=torch.float64).uniform_(0.1, 1.9)
    inputs = [tensor.requires_grad_() for tensor in (query, key, value, decay, rate)]

    actual = recurrent_delta_memory(*inputs)
    expected = _matrix_reference(*inputs)
    torch.testing.assert_close(actual, expected, atol=1e-12, rtol=1e-12)

    actual_grad = torch.autograd.grad(actual.square().sum(), inputs, retain_graph=True)
    expected_grad = torch.autograd.grad(expected.square().sum(), inputs)
    for found, reference in zip(actual_grad, expected_grad, strict=True):
        torch.testing.assert_close(found, reference, atol=1e-11, rtol=1e-11)


def test_kda_and_ckda_are_parameter_matched_and_causal() -> None:
    torch.manual_seed(22)
    kda = DeltaMemoryBlock(complex_ranges=False, width=8, heads=2)
    ckda = DeltaMemoryBlock(complex_ranges=True, width=8, heads=2)
    ckda.load_state_dict(kda.state_dict())
    assert sum(p.numel() for p in kda.parameters()) == sum(p.numel() for p in ckda.parameters())

    input_tokens = torch.randn(2, 6, 8)
    changed_future = input_tokens.clone()
    changed_future[:, 4:] += 10
    for model in (kda, ckda):
        original = model(input_tokens)
        changed = model(changed_future)
        torch.testing.assert_close(original[:, :4], changed[:, :4], atol=0, rtol=0)
        torch.testing.assert_close(original[:1], model(input_tokens[:1]), atol=1e-6, rtol=1e-6)


def test_encoder_resets_state_between_ecg_halves() -> None:
    torch.manual_seed(33)
    encoder = DeltaCPCEncoder(complex_ranges=True).eval()
    signal = torch.randn(1, 12, 2500)
    changed_first_half = signal.clone()
    changed_first_half[:, :, :1250] += 2

    with torch.no_grad():
        tokens, contexts = encoder(signal)
        _, changed_contexts = encoder(changed_first_half)

    assert tokens.shape == contexts.shape == (1, 2, 79, 256)
    torch.testing.assert_close(contexts[:, 1], changed_contexts[:, 1], atol=0, rtol=0)


def test_three_arms_share_stem_heads_and_delta_initialization() -> None:
    torch.manual_seed(77)
    before = torch.random.get_rng_state()
    models = matched_initial_models(9001)
    torch.testing.assert_close(torch.random.get_rng_state(), before, atol=0, rtol=0)

    gru = models["gru"]
    kda = models["kda"]
    ckda = models["ckda"]
    for name, tensor in gru.encoder.convs.state_dict().items():
        torch.testing.assert_close(tensor, kda.encoder.convs.state_dict()[name], atol=0, rtol=0)
        torch.testing.assert_close(tensor, ckda.encoder.convs.state_dict()[name], atol=0, rtol=0)
    for name, tensor in gru.heads.state_dict().items():
        torch.testing.assert_close(tensor, kda.heads.state_dict()[name], atol=0, rtol=0)
        torch.testing.assert_close(tensor, ckda.heads.state_dict()[name], atol=0, rtol=0)
    for name, tensor in kda.encoder.context.state_dict().items():
        torch.testing.assert_close(tensor, ckda.encoder.context.state_dict()[name], atol=0, rtol=0)
