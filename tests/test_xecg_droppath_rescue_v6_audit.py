"""Fixed-set gradient audit checks for the versioned 016 v6 report stage."""

from __future__ import annotations

import math

import torch
from torch import nn
from torch.utils.data import TensorDataset

from scripts.experiments.audit_xecg_droppath_rescue016_v6 import fixed_set_gradient


class TinyClassifier(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(0.0))

    def forward(self, signal: torch.Tensor) -> torch.Tensor:
        return self.weight * signal.squeeze(-1)


def test_fixed_training_subset_eval_gradient_and_hypothetical_clip():
    model = TinyClassifier()
    data = TensorDataset(torch.full((128, 1), 10.0), torch.zeros(128))
    initial = model.weight.detach().clone()
    diagnostic = fixed_set_gradient(model, data, "cpu")
    assert diagnostic["record_exposures"] == 128
    assert diagnostic["mode"] == "evaluation"
    assert math.isclose(diagnostic["gradient_global_l2_before_hypothetical_clip"], 5.0, rel_tol=1e-6)
    assert diagnostic["would_clip_at_3"]
    assert torch.equal(model.weight.detach(), initial)
    assert model.weight.grad is None
    assert diagnostic["actual_training_batch_clip"] == "not measured by this fixed-set audit"
