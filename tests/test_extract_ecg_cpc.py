"""Numerical and strict-loading checks for released S4 inference."""
import torch
import pytest

from scripts.features.extract_ecg_cpc import load_exact, torch_cauchy_conj


def test_cauchy_reduction_matches_official_real_component_algebra():
    torch.manual_seed(7)
    v = torch.randn(2, 2, 3, 4, dtype=torch.complex128)
    w = torch.randn(3, 4, dtype=torch.complex128) - 3
    z = torch.randn(9, dtype=torch.complex128) + 3
    actual = torch_cauchy_conj(v, z, w)
    # Official KeOps expression: 2*(z*Re(v)-Sum([vr,vi]*[wr,wi]))
    # / ((z-w)*(z-conj(w))), summed over conjugate-pair representatives.
    reference = torch.zeros((2, 2, 3, 9), dtype=torch.complex128)
    for n in range(4):
        numerator = z * v[..., n, None].real - (
            v[..., n, None].real * w[:, n, None].real +
            v[..., n, None].imag * w[:, n, None].imag)
        reference += 2 * numerator / ((z - w[:, n, None]) * (z - w[:, n, None].conj()))
    torch.testing.assert_close(actual, reference, atol=1e-14, rtol=1e-14)


def test_checkpoint_loading_rejects_missing_and_mismatched_tensors():
    model = torch.nn.Linear(2, 3)
    with pytest.raises(RuntimeError):
        load_exact(model, {"model.weight": torch.ones(3, 2)}, "model.")
    with pytest.raises(RuntimeError):
        load_exact(model, {"model.weight": torch.ones(4, 2), "model.bias": torch.zeros(3)}, "model.")
    assert load_exact(model, {"model.weight": torch.ones(3, 2), "model.bias": torch.zeros(3)}, "model.") == 2
    torch.testing.assert_close(model.weight, torch.ones(3, 2))


def test_strict_assignment_supports_expanded_official_s4_buffers():
    model = torch.nn.Module()
    model.register_buffer("expanded", torch.zeros(1, 4).expand(3, 4))
    target = torch.randn(3, 4)
    load_exact(model, {"model.expanded": target}, "model.")
    torch.testing.assert_close(model.expanded, target, atol=0, rtol=0)
