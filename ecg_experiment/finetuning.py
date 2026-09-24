"""Device checks, resume random states, and completion receipts for fine-tuning runners."""

from __future__ import annotations

import json
import random
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .files import sha256_file

RNG_NAMES = ("torch", "cuda", "python", "numpy")


def require_cuda(device: str) -> None:
    """
    Fail early when CUDA is requested on a machine without it.

    Parameters
    ----------
    device : str
        Torch device name.

    Raises
    ------
    RuntimeError
        If ``device`` is ``"cuda"`` and CUDA is unavailable.
    """
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")


def peak_cuda_memory(device: str) -> int | None:
    """
    Return the peak allocated CUDA memory, or None on CPU.

    Parameters
    ----------
    device : str
        Torch device name.

    Returns
    -------
    int | None
        Peak allocated bytes since the last reset on CUDA; None otherwise.
    """
    return torch.cuda.max_memory_allocated() if device == "cuda" else None


def optimizer_state_bytes(optimizer: torch.optim.Optimizer) -> int:
    """
    Count the bytes held by an optimizer's tensor state.

    Parameters
    ----------
    optimizer : torch.optim.Optimizer
        Optimizer after at least one step.

    Returns
    -------
    int
        Total bytes of every tensor in the per-parameter state.
    """
    return sum(value.numel() * value.element_size()
               for state in optimizer.state.values() for value in state.values()
               if isinstance(value, torch.Tensor))


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


def artifact_hashes(directory: Path, names: Iterable[str]) -> dict[str, str]:
    """
    Hash the named artifacts of a run directory.

    Parameters
    ----------
    directory : Path
        Run directory.
    names : Iterable[str]
        Artifact file names inside ``directory``.

    Returns
    -------
    dict[str, str]
        SHA-256 digest keyed by file name.
    """
    return {name: sha256_file(directory / name) for name in names}


def verified_completion(directory: Path, fingerprint: dict[str, Any],
                        names: Iterable[str]) -> dict[str, Any] | None:
    """
    Read a run's ``complete.json`` and verify its inputs and artifacts.

    Parameters
    ----------
    directory : Path
        Run directory.
    fingerprint : dict[str, Any]
        Inputs the completed run must have used.
    names : Iterable[str]
        Artifacts whose digests the receipt records.

    Returns
    -------
    dict[str, Any] | None
        The receipt, or None when the run has not completed.

    Raises
    ------
    ValueError
        If the completed run used other inputs or an artifact changed.
    """
    marker = directory / "complete.json"
    if not marker.is_file():
        return None
    receipt = json.loads(marker.read_text())
    if receipt.get("fingerprint") != fingerprint:
        raise ValueError(f"Completed run differs from requested inputs: {directory}")
    if receipt.get("sha256") != artifact_hashes(directory, names):
        raise ValueError(f"Completed artifact checksum mismatch: {directory}")
    return receipt
