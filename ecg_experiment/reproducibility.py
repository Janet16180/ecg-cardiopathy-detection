"""Seeding, random-state checkpoints, and CPU copies of model weights."""

from __future__ import annotations

import random
from typing import Any

import numpy as np
import torch

RNG_NAMES = ("torch", "cuda", "python", "numpy")


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


def rng_state_lists() -> dict[str, Any]:
    """
    Capture global random states with the NumPy key stored as a plain list.

    The list form survives ``torch.load(weights_only=True)``, unlike a NumPy
    array.

    Returns
    -------
    dict[str, Any]
        States keyed by ``python``, ``numpy``, ``torch`` and ``cuda``.
    """
    numpy_state = np.random.get_state()
    return {"python": random.getstate(), "numpy": (numpy_state[0], numpy_state[1].tolist(), *numpy_state[2:]),
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}


def restore_rng_lists(state: dict[str, Any]) -> None:
    """
    Restore states captured by ``rng_state_lists``.

    Parameters
    ----------
    state : dict[str, Any]
        Output of ``rng_state_lists``.

    Raises
    ------
    ValueError
        If CUDA is available but the saved state covers another device count.
    """
    random.setstate(state["python"])
    numpy_state = state["numpy"]
    np.random.set_state((numpy_state[0], np.asarray(numpy_state[1], dtype=np.uint32), *numpy_state[2:]))
    torch.set_rng_state(state["torch"])
    if torch.cuda.is_available():
        if len(state["cuda"]) != torch.cuda.device_count():
            raise ValueError("CUDA device count differs from resume checkpoint")
        torch.cuda.set_rng_state_all(state["cuda"])


def flat_rng_state() -> dict[str, Any]:
    """
    Capture global random states under the flat ``<name>_rng`` resume keys.

    Returns
    -------
    dict[str, Any]
        States keyed by ``torch_rng``, ``cuda_rng``, ``python_rng`` and
        ``numpy_rng``.
    """
    state = rng_state_lists()
    return {f"{name}_rng": state[name] for name in RNG_NAMES}


def restore_flat_rng_state(saved: dict[str, Any]) -> None:
    """
    Restore states stored under the flat ``<name>_rng`` resume keys.

    Parameters
    ----------
    saved : dict[str, Any]
        Checkpoint holding the keys written by ``flat_rng_state``.

    Raises
    ------
    ValueError
        If CUDA is available but the saved state covers another device count.
    """
    restore_rng_lists({name: saved[f"{name}_rng"] for name in RNG_NAMES})


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
