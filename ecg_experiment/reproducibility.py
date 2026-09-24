"""Seeding, random-state checkpoints, and CPU copies of model weights."""

from __future__ import annotations

import random
from typing import Any

import numpy as np
import torch


def seed_everything(seed: int) -> None:
    """
    Seed the Python, NumPy, and torch global generators.

    ``torch.manual_seed`` also seeds every CUDA device.

    Parameters
    ----------
    seed : int
        Seed shared by all generators.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def capture_rng_state(generator: torch.Generator | None = None) -> dict[str, Any]:
    """
    Capture global random states, and optionally a data-loader generator.

    Parameters
    ----------
    generator : torch.Generator | None
        Loader generator stored under ``"loader"`` when given.

    Returns
    -------
    dict[str, Any]
        States keyed by ``python``, ``numpy``, ``torch``, ``cuda`` and
        optionally ``loader``.
    """
    state = {"python": random.getstate(), "numpy": np.random.get_state(),
             "torch": torch.get_rng_state(),
             "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}
    if generator is not None:
        state["loader"] = generator.get_state()
    return state


def restore_rng_state(state: dict[str, Any], generator: torch.Generator | None = None) -> None:
    """
    Restore states captured by ``capture_rng_state``.

    Parameters
    ----------
    state : dict[str, Any]
        Output of ``capture_rng_state``.
    generator : torch.Generator | None
        Loader generator restored from ``state["loader"]`` when given.
    """
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])
    if generator is not None:
        generator.set_state(state["loader"])


def cpu_state(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    """
    Copy a model's state dict to detached CPU tensors.

    Parameters
    ----------
    model : torch.nn.Module
        Model to copy.

    Returns
    -------
    dict[str, torch.Tensor]
        Independent CPU copies of every parameter and buffer.
    """
    return {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
