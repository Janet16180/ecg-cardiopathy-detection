"""
Machine-independent project paths for receipts, manifests and metadata.

New code stores a path inside the repository as ``to_stored(path)``, a POSIX path relative to
``ROOT``, and opens it with ``from_stored(value)``. The stored text, and therefore any SHA-256
of a file that contains it, is then the same on every machine and in every clone location.

Receipts written before this module store absolute paths such as
``/home/JanetRivera/ecg-cardiopathy-detection/data/raw/...``. ``from_stored`` maps those to
the same place under the current ``ROOT``. Frozen runners keep their own path handling.
"""

from __future__ import annotations

import os
from pathlib import Path, PurePosixPath

from ecg_experiment import ROOT

LEGACY_REPOSITORY_NAME = "ecg-cardiopathy-detection"


def to_stored(path: Path | str) -> str:
    """
    Repository-relative POSIX form of a path, for writing into a receipt.

    Parameters
    ----------
    path : Path or str
        File or directory inside the repository, absolute or relative to the working directory.

    Returns
    -------
    str
        Path relative to ``ROOT`` with forward slashes, such as ``data/raw/ptb-xl/1.0.3``.

    Raises
    ------
    ValueError
        If ``path`` is outside the repository.
    """
    # abspath rather than resolve: a symlinked data/raw (for example to shared storage) must
    # keep its repository-relative name.
    return Path(os.path.abspath(path)).relative_to(ROOT).as_posix()


def from_stored(value: str) -> Path:
    """
    Absolute path on this machine for a path read from a receipt.

    Parameters
    ----------
    value : str
        A ``to_stored`` value, or a legacy absolute path. A legacy path under a directory
        named ``ecg-cardiopathy-detection`` is moved under ``ROOT``; any other absolute
        path is returned unchanged.

    Returns
    -------
    Path
        Absolute path to open on this machine.
    """
    stored = PurePosixPath(value)
    if stored.is_absolute() and LEGACY_REPOSITORY_NAME not in stored.parts:
        return Path(value)

    parts = stored.parts
    if stored.is_absolute():
        parts = parts[parts.index(LEGACY_REPOSITORY_NAME) + 1 :]
    return ROOT.joinpath(*parts)
