"""Small independent-audit I/O and scalar comparison utilities."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np

from ecg_experiment.ssl_synthetic_pretrained_saved_audit import require


def file_sha256(path: Path, guard: Callable[[], None] = lambda: None) -> str:
    """Hash an authorized file in bounded chunks.

    Parameters
    ----------
    path : Path
        File admitted by the calling audit.
    guard : callable
        Deadline and byte-budget check.

    Returns
    -------
    str
        SHA-256 of exact file bytes.
    """
    result = hashlib.sha256()
    with path.open("rb") as stream:
        while True:
            guard()
            block = stream.read(1024**2)
            if not block:
                break
            result.update(block)
    return result.hexdigest()


def relative_file(root: Path, name: str) -> Path:
    """Confine an admitted binding to its declared lexical root.

    Parameters
    ----------
    root : Path
        Authorized artifact directory.
    name : str
        Repository-relative POSIX filename.

    Returns
    -------
    Path
        Confined file path without resolving symlinks.
    """
    value = Path(name)
    require(not value.is_absolute() and ".." not in value.parts, "relative audit binding required")
    target = root / value
    require(not target.is_symlink(), "bound file symlink prohibited")
    return target


def read_json(path: Path) -> dict[str, Any]:
    """Read admitted JSON object metadata.

    Parameters
    ----------
    path : Path
        Authorized metadata file.

    Returns
    -------
    dict[str, Any]
        Decoded object.
    """
    value = json.loads(path.read_text())
    require(isinstance(value, dict), "audit metadata must be an object")
    return value


def compare_values(expected: Any, actual: Any, label: str, tolerance: float = 1e-12) -> int:
    """Compare complete nested metadata and count checked scalar values.

    Parameters
    ----------
    expected, actual : Any
        Independently reconstructed and saved values.
    label : str
        Error location.
    tolerance : float
        Prospective absolute numerical tolerance.

    Returns
    -------
    int
        Number of checked scalar values.
    """
    if isinstance(expected, dict):
        require(isinstance(actual, dict) and set(expected) == set(actual), f"{label}: fields")
        return sum(
            compare_values(value, actual[key], f"{label}/{key}", tolerance) for key, value in expected.items()
        )
    if isinstance(expected, list):
        require(isinstance(actual, list) and len(expected) == len(actual), f"{label}: length")
        return sum(
            compare_values(left, right, label, tolerance)
            for left, right in zip(expected, actual, strict=True)
        )
    if isinstance(expected, float):
        require(bool(np.isfinite(actual)) and abs(expected - actual) <= tolerance, f"{label}: numeric")
    else:
        require(expected == actual, f"{label}: value")
    return 1
