"""Hashing and atomic file writes shared by data preparation and experiments."""

from __future__ import annotations

import csv
import hashlib
import json
import os
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

SHA256_HEX_LENGTH = 64


def sha256_file(path: str | Path) -> str:
    """
    Compute the SHA-256 digest of a file.

    Parameters
    ----------
    path : str | Path
        File to hash.

    Returns
    -------
    str
        Hexadecimal digest.
    """
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def sha256_json(value: Any) -> str:
    """
    Compute the SHA-256 digest of a canonical JSON encoding.

    Keys are sorted and separators are compact, so equal values always give
    the same digest.

    Parameters
    ----------
    value : Any
        JSON-serializable value.

    Returns
    -------
    str
        Hexadecimal digest.
    """
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _temporary_path(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    return path.with_name(path.name + ".tmp")


def write_text_atomic(path: str | Path, text: str) -> None:
    """
    Write text through a temporary file and rename it into place.

    Parameters
    ----------
    path : str | Path
        Destination file; parent directories are created.
    text : str
        Content to write.
    """
    path = Path(path)
    temporary = _temporary_path(path)
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def write_json_atomic(path: str | Path, value: Any, *, sort_keys: bool = False,
                      allow_nan: bool = False) -> None:
    """
    Write indented JSON atomically, followed by a newline.

    Parameters
    ----------
    path : str | Path
        Destination file; parent directories are created.
    value : Any
        JSON-serializable value.
    sort_keys : bool
        Sort object keys in the output.
    allow_nan : bool
        Permit NaN and infinity; by default they raise ``ValueError``.
    """
    text = json.dumps(value, indent=2, sort_keys=sort_keys, allow_nan=allow_nan) + "\n"
    write_text_atomic(path, text)


def write_torch_atomic(path: str | Path, value: Any) -> None:
    """
    Save a torch object through a temporary file and rename it into place.

    The bytes equal those of a plain ``torch.save(value, path)``.

    Parameters
    ----------
    path : str | Path
        Destination file; parent directories are created.
    value : Any
        Object accepted by ``torch.save``.
    """
    # Imported here so download entry points that hash files do not load torch.
    import torch

    path = Path(path)
    # torch names the archive after the file stem, so the temporary keeps the
    # destination name inside a hidden staging directory.
    staging = path.parent / f".{path.name}.tmp"
    staging.mkdir(parents=True, exist_ok=True)
    temporary = staging / path.name
    torch.save(value, temporary)
    os.replace(temporary, path)
    staging.rmdir()


def write_npz_atomic(path: str | Path, **arrays: Any) -> None:
    """
    Save arrays with ``numpy.savez`` through a temporary file and rename it into place.

    Parameters
    ----------
    path : str | Path
        Destination file, used exactly; no ``.npz`` suffix is added.
    **arrays : Any
        Arrays stored under their keyword names, in order.
    """
    # Imported here for the same reason as torch in write_torch_atomic.
    import numpy as np

    path = Path(path)
    temporary = _temporary_path(path)
    with temporary.open("wb") as handle:
        np.savez(handle, **arrays)
    os.replace(temporary, path)


def read_json(path: str | Path) -> Any:
    """
    Read a UTF-8 JSON file.

    Parameters
    ----------
    path : str | Path
        JSON file.

    Returns
    -------
    Any
        Parsed value.
    """
    return json.loads(Path(path).read_text(encoding="utf-8"))


def read_csv(path: str | Path, required: Iterable[str] = ()) -> list[dict[str, str]]:
    """
    Read a CSV file with a header row.

    Parameters
    ----------
    path : str | Path
        CSV file.
    required : Iterable[str]
        Columns the header must include.

    Returns
    -------
    list[dict[str, str]]
        One dictionary per row, keyed by column name.

    Raises
    ------
    ValueError
        If a required column is missing.
    """
    with Path(path).open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        missing = set(required) - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Missing columns in {path}: {sorted(missing)}")
        return list(reader)


def write_csv_atomic(path: str | Path, rows: Iterable[Mapping[str, Any]],
                     fieldnames: Iterable[str], *, ignore_extra: bool = False) -> None:
    """
    Write rows as CSV through a temporary file and rename it into place.

    Parameters
    ----------
    path : str | Path
        Destination file; parent directories are created.
    rows : Iterable[Mapping[str, Any]]
        Rows to write.
    fieldnames : Iterable[str]
        Column order.
    ignore_extra : bool
        Drop row keys outside ``fieldnames``; by default they raise ``ValueError``.
    """
    path = Path(path)
    temporary = _temporary_path(path)
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames),
                                extrasaction="ignore" if ignore_extra else "raise")
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)
