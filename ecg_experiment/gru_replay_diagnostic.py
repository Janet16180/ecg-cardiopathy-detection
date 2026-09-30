"""Synthetic CUDA diagnosis of recurrent dropout checkpoint recovery."""

from __future__ import annotations

import copy
import time
from typing import Any

import torch
from torch import nn

from ecg_experiment.cpc_xlstm_native_gru import NativeGRU
from ecg_experiment.gpu import gpu_lock
from ecg_experiment.reproducibility import (
    capture_rng_state,
    cpu_state,
    restore_rng_state,
    seed_everything,
)
from ecg_experiment.xlstm_study import configure_runtime, tree_equal


def replay_case(native: bool, dropout: float) -> dict[str, Any]:
    """Measure next-update equality on synthetic CUDA GRU tokens.

    Parameters
    ----------
    native : bool
        Disable cuDNN only inside the recurrent forward when true.
    dropout : float
        Inter-layer dropout probability used by this diagnostic control.

    Returns
    -------
    dict[str, Any]
        Loss, model, optimizer and RNG replay comparisons.
    """
    seed_everything(38042)
    constructor = NativeGRU if native else nn.GRU
    model = constructor(256, 256, num_layers=2, batch_first=True, dropout=dropout).cuda()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
    tokens = torch.randn(4, 79, 256, device="cuda")

    def update() -> float:
        """Apply one synthetic squared-context AdamW update."""
        optimizer.zero_grad(set_to_none=True)
        context, _ = model(tokens)
        loss = context.square().mean()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        return float(loss.detach())

    for _ in range(3):
        update()
    saved = (cpu_state(model), copy.deepcopy(optimizer.state_dict()), capture_rng_state())
    expected_loss = update()
    expected = (cpu_state(model), copy.deepcopy(optimizer.state_dict()), capture_rng_state())
    model.load_state_dict(saved[0])
    optimizer.load_state_dict(saved[1])
    restore_rng_state(saved[2])
    observed_loss = update()
    observed = (cpu_state(model), optimizer.state_dict(), capture_rng_state())
    comparisons = {name: tree_equal(a, b) for name, a, b in
                   zip(("model_equal", "optimizer_equal", "rng_equal"),
                       expected, observed, strict=True)}
    return {"native": native, "dropout": dropout,
            "expected_loss": expected_loss, "observed_loss": observed_loss,
            "loss_equal": expected_loss == observed_loss,
            "maximum_parameter_difference": max(
                float((expected[0][name] - value).abs().max())
                for name, value in observed[0].items()), **comparisons}


def diagnose() -> dict[str, Any]:
    """Run locked synthetic controls without ECG or development data.

    Returns
    -------
    dict[str, Any]
        Backend/dropout controls, elapsed time and diagnostic decision.
    """
    started = time.monotonic()
    configure_runtime()
    with gpu_lock("cuda", blocking=False):
        cases = [replay_case(False, 0.1), replay_case(False, 0.0), replay_case(True, 0.1)]
    equal = [all(case[key] for key in
                 ("loss_equal", "model_equal", "optimizer_equal", "rng_equal")) for case in cases]
    return {"cases": cases, "supports_cudnn_dropout_state_cause": equal == [False, True, True],
            "elapsed_seconds": time.monotonic() - started,
            "torch_version": torch.__version__, "cuda_version": torch.version.cuda,
            "cudnn_version": torch.backends.cudnn.version(),
            "gpu_name": torch.cuda.get_device_name(0)}
