"""Official-checksum and header-annotation checks for public WFDB record files."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from pathlib import Path

from .files import sha256_file


def header_label_codes(header: Path) -> str:
    """
    Return the ``# Dx:`` annotation of a WFDB header.

    Parameters
    ----------
    header : Path
        WFDB ``.hea`` file.

    Returns
    -------
    str
        The comma-separated codes as written, or an empty string when the
        header has no ``# Dx:`` line.
    """
    lines = header.read_text(encoding="utf-8").splitlines()
    return next((line.split(":", 1)[1].strip() for line in lines if line.startswith("# Dx:")), "")


def first_unverified_file(raw_dir: Path, stem: str, suffixes: Iterable[str],
                          checksums: Mapping[str, str]) -> str | None:
    """
    Find the first record file that is missing or differs from its official hash.

    Parameters
    ----------
    raw_dir : Path
        Release root that the checksum names are relative to.
    stem : str
        Record path relative to ``raw_dir``, without extension.
    suffixes : Iterable[str]
        File extensions of the record, such as ``(".hea", ".mat")``.
    checksums : Mapping[str, str]
        Official SHA-256 digests keyed by relative file name.

    Returns
    -------
    str | None
        Relative name of the first file without a matching official checksum,
        or ``None`` when every file verifies.
    """
    unverified = None
    for suffix in suffixes:
        name = stem + suffix
        path = raw_dir / name
        if name not in checksums or not path.is_file() or sha256_file(path) != checksums[name]:
            unverified = name
            break
    return unverified
