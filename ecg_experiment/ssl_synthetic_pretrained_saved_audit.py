"""Independent arithmetic checks for saved public-generator technical outputs."""

from __future__ import annotations

import hashlib
from typing import Any

import numpy as np
import torch

SEEDS = (2026102901, 2026102902, 2026102903, 2026102904)


def require(condition: bool, label: str) -> None:
    """Raise for a failed audit condition.

    Parameters
    ----------
    condition : bool
        Truth value to require.
    label : str
        Failure description.
    """
    if not condition:
        raise ValueError(label)


def noise_for_seed(seed: int) -> np.ndarray:
    """Regenerate a frozen CPU torch uniform input.

    Parameters
    ----------
    seed : int
        Frozen input seed.

    Returns
    -------
    np.ndarray
        Native float32 noise with shape (1, 8, 5000).
    """
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    return torch.rand((1, 8, 5000), generator=generator, dtype=torch.float32).mul_(2).sub_(1).numpy()


def verify_noise(noise: np.ndarray) -> None:
    """Check every retained input byte against its fixed seed.

    Parameters
    ----------
    noise : np.ndarray
        Four saved generator inputs.
    """
    require(noise.dtype == np.float32 and noise.shape == (4, 1, 8, 5000), "noise geometry")
    for index, seed in enumerate(SEEDS):
        require(noise[index].tobytes() == noise_for_seed(seed).tobytes(), f"seed {seed} noise bytes")


def verify_case(raw: np.ndarray, export: np.ndarray, canonical: np.ndarray) -> dict[str, bool]:
    """Check output conversion and lead equations without main helpers.

    Parameters
    ----------
    raw, export, canonical : np.ndarray
        Saved native, author-export and twelve-lead numerical arrays.

    Returns
    -------
    dict[str, bool]
        Independently established technical checks.
    """
    require(raw.dtype == np.float32 and raw.shape == (8, 5000), "native geometry")
    require(bool(np.isfinite(raw).all()), "raw finite")
    require(export.dtype == np.int32 and export.shape == (5000, 8), "export geometry")
    scaled = np.multiply(raw, np.float32(6000))
    require(bool((np.abs(scaled) < np.iinfo(np.int32).max).all()), "export int32 range")
    converted = np.trunc(scaled).astype(np.int32).transpose()
    require(np.array_equal(export, converted), "author export arithmetic")
    require(canonical.dtype == np.float32 and canonical.shape == (12, 5000), "canonical geometry")
    require(bool(np.isfinite(canonical).all()), "canonical finite")
    independent = export.transpose().astype(np.float32)
    expected = np.empty_like(canonical)
    expected[[0, 1, 6, 7, 8, 9, 10, 11]] = independent
    expected[2] = independent[1] - independent[0]
    expected[3] = -np.float32(0.5) * (independent[0] + independent[1])
    expected[4] = independent[0] - np.float32(0.5) * independent[1]
    expected[5] = independent[1] - np.float32(0.5) * independent[0]
    require(bool(np.allclose(canonical, expected, rtol=0, atol=1e-5)), "canonical algebra")
    return {
        "native_geometry": True,
        "finite": True,
        "author_export_exact": True,
        "canonical_algebra_atol_1e_minus5": True,
        "every_raw_lead_varies": bool(np.all(np.max(raw, axis=1) > np.min(raw, axis=1))),
    }


def checkpoint_digest(state: Any) -> tuple[str, int]:
    """Bind a plain checkpoint tensor dictionary independently of model code.

    Parameters
    ----------
    state : Any
        Object returned by strict weights-only CPU loading.

    Returns
    -------
    tuple[str, int]
        Named-state byte digest and total parameter elements.
    """
    require(isinstance(state, dict) and bool(state), "plain nonempty state dictionary")
    require(all(isinstance(key, str) for key in state), "state names")
    digest = hashlib.sha256()
    elements = 0
    for name in sorted(state):
        tensor = state[name]
        require(type(tensor) is torch.Tensor, "plain state tensor")
        require(tensor.dtype == torch.float32 and tensor.device.type == "cpu", "state dtype/device")
        require(bool(torch.isfinite(tensor).all()), "finite checkpoint")
        digest.update(name.encode())
        digest.update(str((tensor.dtype, tuple(tensor.shape))).encode())
        digest.update(tensor.contiguous().numpy().tobytes())
        elements += tensor.numel()
    return digest.hexdigest(), elements
