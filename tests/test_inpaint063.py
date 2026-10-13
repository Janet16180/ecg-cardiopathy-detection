"""Tests for the Experiment 063 masked reconstruction helpers."""

from __future__ import annotations

import numpy as np
import torch

from ecg_experiment.inpaint063 import (
    FAMILIES,
    InpaintNet,
    beta_nll,
    electrode_noise,
    finish_map,
    inference_masks,
    record_errors,
    train_mask,
    upsample,
)


def test_electrode_noise_keeps_limb_relations() -> None:
    noise = electrode_noise(4, 500, np.random.default_rng(0))
    np.testing.assert_allclose(noise[:, 2], noise[:, 1] - noise[:, 0], atol=1e-6)
    np.testing.assert_allclose(noise[:, 3] + noise[:, 4] + noise[:, 5], 0, atol=1e-6)


def test_inference_mask_coverage() -> None:
    peaks = np.arange(100, 2500, 200)
    masks, families = inference_masks(peaks, 2500)
    assert len(masks) == 140
    block = masks[families == 0].sum(axis=0)
    whole = masks[families == 2].sum(axis=0)
    assert (block == 2).all()
    assert (whole == 1).all()
    phase = masks[families == 1]
    assert phase.any(axis=0)[:, 100:2400].all()
    for mask in masks:
        assert (mask[:6] == mask[0]).all()


def test_train_masks_hide_limb_leads_together() -> None:
    rng = np.random.default_rng(1)
    peaks = np.arange(50, 2048, 210)
    for _ in range(200):
        mask = train_mask(peaks, 2048, rng)
        assert mask.any()
        assert (mask[:6] == mask[0]).all()


def test_upsample_alignment() -> None:
    values = np.random.default_rng(2).normal(size=(12, 2500))
    up = upsample(values)
    assert up.shape == (12, 5000)
    np.testing.assert_allclose(up[:, ::2], values, rtol=1e-6, atol=1e-6)


def test_hidden_samples_do_not_leak() -> None:
    torch.manual_seed(0)
    model = InpaintNet(widths=(16, 16, 16, 16, 16), layers=1, heads=2).eval()
    signal = torch.randn(1, 12, 256)
    mask = torch.zeros(1, 12, 256, dtype=torch.bool)
    mask[:, 6, 100:150] = True
    changed = signal.clone()
    changed[mask] = 100.0
    with torch.no_grad():
        mean_a, logvar_a = model(signal, mask)
        mean_b, logvar_b = model(changed, mask)
    torch.testing.assert_close(mean_a, mean_b)
    torch.testing.assert_close(logvar_a, logvar_b)
    assert logvar_a.min() > -9
    assert logvar_a.max() < 3


def test_loss_counts_only_hidden_samples() -> None:
    target = torch.zeros(1, 12, 8)
    mean = torch.zeros(1, 12, 8)
    mean[0, 0, 0] = 5.0
    logvar = torch.zeros(1, 12, 8)
    mask = torch.zeros(1, 12, 8, dtype=torch.bool)
    mask[0, 1] = True
    assert float(beta_nll(target, mean, logvar, mask)) == 0.0


def test_finish_map_fills_uncovered() -> None:
    covered = np.ones((12, 2500))
    partial = np.full((12, 2500), np.nan)
    partial[:, :100] = 3.0
    out = finish_map([covered, partial])
    assert out.shape == (12, 5000)
    assert np.isfinite(out).all()
    assert out[:, 1000:].max() == 1.0


def test_record_errors_shapes() -> None:
    model = InpaintNet(widths=(16, 16, 16, 16, 16), layers=1, heads=2).eval()
    signal = np.random.default_rng(3).normal(size=(12, 2500)).astype(np.float32)
    errors = record_errors(model, signal, np.arange(100, 2500, 200), "cpu")
    for family in FAMILIES:
        assert errors[f"z_{family}"].shape == (12, 2500)
        assert np.isfinite(errors[f"z_{family}"]).all() or family == "phase"
