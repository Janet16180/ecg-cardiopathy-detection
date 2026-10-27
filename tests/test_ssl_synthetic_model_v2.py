"""Check interpolation operator and gradient parity independently of training results."""

import pytest
import torch
from torch.nn import functional

from ecg_experiment.ssl_synthetic_model import DenseWaveformModel as OriginalModel
from ecg_experiment.ssl_synthetic_model import state_digest
from ecg_experiment.ssl_synthetic_model_v2 import DenseWaveformModel, deterministic_linear_upsample


@pytest.mark.parametrize("length", [1, 2, 3, 17, 500])
def test_float64_linear_operator_and_gradients_match_reference(length):
    generator = torch.Generator().manual_seed(18)
    signal = torch.randn(2, 3, length, generator=generator, dtype=torch.float64, requires_grad=True)
    triangle = torch.tensor([0.2, 0.4, 0.6, 0.8, 1, 0.8, 0.6, 0.4, 0.2], dtype=torch.float64)
    kernel = triangle[None, None].repeat(3, 1, 1)
    actual = deterministic_linear_upsample(signal, kernel)
    expected = functional.interpolate(signal, scale_factor=5, mode="linear", align_corners=False)
    torch.testing.assert_close(actual, expected, atol=1e-12, rtol=1e-12)
    gradient = torch.randn(actual.shape, generator=generator, dtype=torch.float64)
    actual_grad = torch.autograd.grad(actual, signal, gradient, retain_graph=True)[0]
    expected_grad = torch.autograd.grad(expected, signal, gradient)[0]
    torch.testing.assert_close(actual_grad, expected_grad, atol=1e-12, rtol=1e-12)


@pytest.mark.parametrize("kind", ["constant", "ramp", "first_impulse", "last_impulse"])
def test_endpoints_and_no_amplitude_gain(kind):
    signal = torch.zeros(1, 1, 17, dtype=torch.float64)
    if kind == "constant":
        signal.fill_(1)
    elif kind == "ramp":
        signal[0, 0] = torch.arange(17)
    elif kind == "first_impulse":
        signal[0, 0, 0] = 1
    else:
        signal[0, 0, -1] = 1
    kernel = torch.tensor([[[0.2, 0.4, 0.6, 0.8, 1, 0.8, 0.6, 0.4, 0.2]]], dtype=torch.float64)
    actual = deterministic_linear_upsample(signal, kernel)
    expected = functional.interpolate(signal, scale_factor=5, mode="linear", align_corners=False)
    torch.testing.assert_close(actual, expected, atol=1e-12, rtol=1e-12)


def test_native_float32_forward_gradients_and_initialization_match():
    torch.manual_seed(18)
    original = OriginalModel()
    torch.manual_seed(18)
    successor = DenseWaveformModel()
    assert state_digest(original.state_dict()) == state_digest(successor.state_dict())
    signal = torch.randn(2, 12, 2500)
    first = original(signal)
    second = successor(signal)
    torch.testing.assert_close(first, second, atol=1e-4, rtol=1e-4)
    first.square().mean().backward()
    second.square().mean().backward()
    for first_parameter, second_parameter in zip(original.parameters(), successor.parameters(), strict=True):
        torch.testing.assert_close(first_parameter.grad, second_parameter.grad, atol=1e-4, rtol=1e-4)


def test_noninteger_native_scale_is_rejected():
    model = DenseWaveformModel()
    with pytest.raises(ValueError, match="divisible"):
        model(torch.zeros(1, 12, 2499))


def test_double_model_preserves_full_interpolation_precision():
    torch.manual_seed(18)
    original = OriginalModel().double()
    torch.manual_seed(18)
    successor = DenseWaveformModel().double()
    signal = torch.randn(1, 12, 2500, dtype=torch.float64)
    torch.testing.assert_close(original(signal), successor(signal), atol=1e-12, rtol=1e-12)
