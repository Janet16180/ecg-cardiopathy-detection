"""Check the 016 probe transfer and matched-cost decision without a GPU."""

from argparse import Namespace

import numpy as np
import pytest
import torch
from torch import nn

from ecg_experiment.files import write_npz_atomic
from scripts.experiments.run_xecg_probe_finetune016 import (
    _cost_gate,
    _load_probe_head,
    affine_from_standardized,
)


def test_folding_training_scaler_preserves_probe_logits(tmp_path):
    rng = np.random.default_rng(42)
    features = rng.normal(size=(17, 8)).astype(np.float32)
    mean = rng.normal(size=8)
    scale = rng.uniform(0.5, 3, size=8)
    coefficient = rng.normal(size=8)
    intercept = -0.7
    raw_weight, raw_bias = affine_from_standardized(coefficient, intercept, mean, scale)
    expected = ((features.astype(np.float64) - mean) / scale) @ coefficient + intercept
    np.testing.assert_allclose(features @ raw_weight + raw_bias, expected, rtol=1e-12, atol=1e-12)

    write_npz_atomic(tmp_path / "probe.npz", raw_weight=raw_weight, raw_bias=np.array(raw_bias),
                     mean=mean, scale=scale, coefficient=coefficient,
                     intercept=np.array(intercept))
    model = nn.Module()
    model.head = nn.Linear(8, 1)
    assert _load_probe_head(model, tmp_path / "probe.npz", features) < 1e-5
    with torch.no_grad():
        actual = model.head(torch.from_numpy(features)).squeeze(-1).numpy()
    np.testing.assert_allclose(actual, expected, rtol=2e-5, atol=2e-4)


def test_cost_gate_counts_both_profile_passes_and_all_four_training_passes(tmp_path):
    args = Namespace(output_dir=tmp_path)
    profile = {"passes": [{"seconds": 500}, {"seconds": 600}]}
    result = _cost_gate(args, profile, {"seconds": 400}, {"seconds": 100})
    assert result["projected_total_seconds"] == 4300
    with pytest.raises(RuntimeError, match="two-hour gate"):
        _cost_gate(args, profile, {"seconds": 4000}, {"seconds": 100})
