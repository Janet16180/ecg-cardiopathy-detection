"""Experiment 010 CUDA profile with a numerical resume comparison.

The frozen runner's bitwise CUDA comparison rejected a 1.49e-8 difference
after save/restore. This entry point retains its profile and training recipe,
while checking the resumed model and AdamW states with a tight FP32 tolerance.
"""

from __future__ import annotations

import copy
import json
from typing import Any

import torch
from torch import nn

from ecg_experiment.reproducibility import capture_rng_state, cpu_state, restore_rng_state
from scripts.experiments import run_cpc_crosslead as crosslead

ATOL = 1e-7
RTOL = 1e-5


def _compare_tensor(left: torch.Tensor, right: Any, path: str, maxima: dict[str, float]) -> None:
    if not isinstance(right, torch.Tensor):
        raise AssertionError(f"Resume tensor missing at {path}")
    if not left.is_floating_point():
        torch.testing.assert_close(left.cpu(), right.cpu(), atol=0, rtol=0, msg=path)
        return
    top = path.split(".", 1)[0]
    maxima[top] = max(maxima.get(top, 0.0), float((left.cpu() - right.cpu()).abs().max()))
    torch.testing.assert_close(left.cpu(), right.cpu(), atol=ATOL, rtol=RTOL, msg=path)


def compare_state(left: Any, right: Any, path: str, maxima: dict[str, float]) -> None:
    """
    Recursively compare two saved states, allowing a tight tolerance on float tensors.

    Parameters
    ----------
    left : Any
        Expected state.
    right : Any
        Resumed state.
    path : str
        Location of this value, used in error messages.
    maxima : dict[str, float]
        Largest absolute float difference per top-level state, updated in place.

    Raises
    ------
    AssertionError
        If structure, integer tensors or scalars differ, or floats exceed the tolerance.
    """
    if isinstance(left, dict):
        if not isinstance(right, dict) or left.keys() != right.keys():
            raise AssertionError(f"Resume state keys differ at {path}")
        for key in left:
            compare_state(left[key], right[key], f"{path}.{key}", maxima)
    elif isinstance(left, list | tuple):
        if not isinstance(right, type(left)) or len(left) != len(right):
            raise AssertionError(f"Resume state sequence differs at {path}")
        for index, (item_left, item_right) in enumerate(zip(left, right, strict=True)):
            compare_state(item_left, item_right, f"{path}[{index}]", maxima)
    elif isinstance(left, torch.Tensor):
        _compare_tensor(left, right, path, maxima)
    elif left != right:
        raise AssertionError(f"Resume scalar differs at {path}: {left!r} != {right!r}")


def profile_roundtrip(model: nn.Module, optimizer: torch.optim.Optimizer, signal: torch.Tensor, variant: str,
                      generator: torch.Generator) -> bool:
    """
    Check that a restored model, AdamW state and RNG reproduce the next update within tolerance.

    Parameters
    ----------
    model : nn.Module
        Profiled model.
    optimizer : torch.optim.Optimizer
        Its optimizer, with populated moments.
    signal : torch.Tensor
        Batch used for the compared update.
    variant : str
        Cross-lead arm.
    generator : torch.Generator
        Loader generator saved with the random states.

    Returns
    -------
    bool
        True when both updates agree; the random states are restored afterwards.
    """
    snapshot = {"model": cpu_state(model),
                "optimizer": copy.deepcopy(optimizer.state_dict()),
                "rng": capture_rng_state(generator)}
    duplicate = crosslead.CrossLeadPretrainer(variant).to(signal.device)
    duplicate.load_state_dict(snapshot["model"])
    copy_optimizer = torch.optim.AdamW(duplicate.parameters(), lr=crosslead.SSL_LR,
                                       weight_decay=crosslead.WEIGHT_DECAY)
    copy_optimizer.load_state_dict(snapshot["optimizer"])
    restore_rng_state(snapshot["rng"], generator)
    crosslead.crosslead_step(model, optimizer, signal)
    expected_model = cpu_state(model)
    expected_optimizer = copy.deepcopy(optimizer.state_dict())
    restore_rng_state(snapshot["rng"], generator)
    crosslead.crosslead_step(duplicate, copy_optimizer, signal)
    maxima = {}
    compare_state(expected_model, cpu_state(duplicate), "model", maxima)
    compare_state(expected_optimizer, copy_optimizer.state_dict(), "optimizer", maxima)
    restore_rng_state(snapshot["rng"], generator)
    print(json.dumps({"stage": "profile_resume_check", "variant": variant,
                      "atol": ATOL, "rtol": RTOL, "max_abs_delta": maxima}), flush=True)
    return True


if __name__ == "__main__":
    crosslead.main(roundtrip=profile_roundtrip)
