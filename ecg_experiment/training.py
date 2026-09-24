"""Optimizer steps, learning-rate schedule, and profile timing shared by training runners."""

from __future__ import annotations

import math
import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import Any

import torch
from torch import nn

WARMUP_EPOCHS = 2
COSINE_FLOOR = 0.1
GRADIENT_CLIP_NORM = 1.0


def warmup_cosine_lr(base_lr: float, epoch: int, epochs: int) -> float:
    """
    Learning rate for one epoch of linear warmup followed by a floored cosine decay.

    Parameters
    ----------
    base_lr : float
        Peak learning rate.
    epoch : int
        Zero-based epoch index.
    epochs : int
        Total number of epochs in the schedule.

    Returns
    -------
    float
        Learning rate for ``epoch``.
    """
    # The expression order matches the frozen runners so rates are bitwise identical.
    warmup = min(1.0, (epoch + 1) / WARMUP_EPOCHS)
    decay = COSINE_FLOOR + (1 - COSINE_FLOOR) * (1 + math.cos(math.pi * epoch / epochs)) / 2
    return base_lr * warmup * decay


def checked_step(loss: torch.Tensor, model: nn.Module, optimizer: torch.optim.Optimizer, what: str,
                 *, clip: bool = True) -> torch.Tensor | None:
    """
    Backpropagate a finite loss, clip gradients, and take one optimizer step.

    Parameters
    ----------
    loss : torch.Tensor
        Scalar loss from the forward pass.
    model : nn.Module
        Model whose parameters are clipped.
    optimizer : torch.optim.Optimizer
        Optimizer to step; its gradients are cleared before backpropagation.
    what : str
        Name used in error messages, such as ``"SSL"``.
    clip : bool
        Clip the global gradient norm to ``GRADIENT_CLIP_NORM`` and check it.

    Returns
    -------
    torch.Tensor | None
        Gradient norm before clipping, or ``None`` when ``clip`` is false.

    Raises
    ------
    RuntimeError
        If the loss or the gradient norm is not finite.
    """
    optimizer.zero_grad(set_to_none=True)
    if not torch.isfinite(loss):
        raise RuntimeError(f"Nonfinite {what} loss")
    loss.backward()
    norm = None
    if clip:
        norm = nn.utils.clip_grad_norm_(model.parameters(), GRADIENT_CLIP_NORM)
        if not torch.isfinite(norm):
            raise RuntimeError(f"Nonfinite {what} gradients")
    optimizer.step()
    return norm


@dataclass
class ProfileTiming:
    """
    Wall time of the measured updates in a profile run.

    Attributes
    ----------
    updates : int
        Updates taken, including warmup.
    measured_records : int
        Records seen by the measured updates.
    seconds : float
        Wall time of the measured updates.
    last : Any
        Value returned by the final update.
    """

    updates: int
    measured_records: int
    seconds: float
    last: Any


def time_updates(batches: Iterable[Any], step: Callable[[Any], tuple[Any, int]], warmup: int,
                 measured: int, device: str, too_small: str) -> ProfileTiming:
    """
    Time optimizer updates after unmeasured warmup updates.

    Parameters
    ----------
    batches : Iterable[Any]
        Training batches.
    step : Callable[[Any], tuple[Any, int]]
        Performs one update on a batch and returns a result and its record count.
    warmup : int
        Leading updates excluded from the measurement.
    measured : int
        Updates to measure after warmup.
    device : str
        Torch device; CUDA is synchronized around the measurement and its
        peak-memory counter is reset when measurement starts.
    too_small : str
        Error message when the batches end before measurement starts.

    Returns
    -------
    ProfileTiming
        Update counts, measured wall time, and the last step result.

    Raises
    ------
    RuntimeError
        If no measured update ran.
    """
    started = None
    measured_records = 0
    update = 0
    last = None
    for update, batch in enumerate(batches, 1):
        if update == warmup + 1:
            if device == "cuda":
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()
            started = time.monotonic()
        last, records = step(batch)
        if update > warmup:
            measured_records += records
        if update >= warmup + measured:
            break
    if started is None:
        raise RuntimeError(too_small)
    if device == "cuda":
        torch.cuda.synchronize()
    return ProfileTiming(update, measured_records, time.monotonic() - started, last)


def peak_gpu_gb(device: str) -> float | None:
    """
    Peak allocated CUDA memory in gigabytes.

    Parameters
    ----------
    device : str
        Torch device.

    Returns
    -------
    float | None
        Peak memory since the last reset, or ``None`` off CUDA.
    """
    return torch.cuda.max_memory_allocated() / 1e9 if device == "cuda" else None


def parameter_count(module: nn.Module) -> int:
    """
    Count every parameter element in a module.

    Parameters
    ----------
    module : nn.Module
        Module to count.

    Returns
    -------
    int
        Total number of parameter elements.
    """
    return sum(parameter.numel() for parameter in module.parameters())
