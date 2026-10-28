"""Checksum-verified canonical views shared by mechanical recovery stages."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .files import sha256_file
from .public_sources import load_view

VIEW_REJECTIONS = {"duration_contract", "nonfinite_or_shape", "constant_lead",
                   "lead_or_rate_contract", "units_contract"}


def verify_pair(raw: Path, stem: str, expected: dict[str, str], stage: str) -> None:
    """Require safe release-relative paths and exact official hashes for one raw pair."""
    relative = Path(stem)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError("Unsafe raw recovery path")
    for suffix, digest in expected.items():
        if sha256_file(raw / (stem + suffix)) != digest:
            raise ValueError(f"Official checksum mismatch {stage}: {stem}{suffix}")


def read_verified_view(
    raw: Path, stem: str, policy: str, expected: dict[str, str],
) -> tuple[np.ndarray | None, int, int, str, list[str]]:
    """
    Read a frozen canonical view with official raw checksums before and after decoding.

    Parameters
    ----------
    raw : Path
        Local release directory.
    stem : str
        Release-relative record path without extension.
    policy : str
        Frozen strict_10s or ssl_center_crop transform.
    expected : dict[str, str]
        Official SHA-256 digests keyed by raw-file suffix.

    Returns
    -------
    tuple[np.ndarray or None, int, int, str, list[str]]
        Canonical signal, start sample, source samples, view flags and explicit contract rejections.
        No model or quality-policy assessment is performed here.
    """
    verify_pair(raw, stem, expected, "before decoding")
    try:
        signal, start, samples, flags = load_view(raw, stem, policy)
    except ValueError as error:
        if str(error) not in VIEW_REJECTIONS:
            raise
        result = (None, 0, 0, "", [str(error)])
    else:
        result = (signal, start, samples, flags, [])
    verify_pair(raw, stem, expected, "after decoding")
    return result
