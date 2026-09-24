"""Checksum manifests and retrying downloads for public ECG releases."""

from __future__ import annotations

import os
import re
import time
import urllib.error
import urllib.request
from collections.abc import Callable
from pathlib import Path


def parse_checksums(contents: str) -> dict[str, str]:
    """
    Parse a ``SHA256SUMS.txt`` manifest.

    Parameters
    ----------
    contents : str
        Manifest text with one ``<digest> <relative path>`` entry per line.

    Returns
    -------
    dict[str, str]
        Lowercase hexadecimal digest keyed by relative path.

    Raises
    ------
    ValueError
        If a line is malformed or a path has two different digests.
    """
    checksums = {}
    for line in contents.splitlines():
        if not line.strip():
            continue
        match = re.fullmatch(r"([0-9a-fA-F]{64})[ \t]+\*?(?:\./)?(.+)", line)
        if not match:
            raise ValueError(f"Invalid SHA256SUMS line: {line[:100]!r}")
        digest, name = match.groups()
        if name in checksums and checksums[name] != digest.lower():
            raise ValueError(f"Conflicting checksum for {name}")
        checksums[name] = digest.lower()
    return checksums


def fetch_file(url: str, destination: Path, timeout: float, retries: int,
               verify: Callable[[Path], bool] | None = None) -> None:
    """
    Download a file through a partial file, retrying with exponential backoff.

    Parameters
    ----------
    url : str
        Source URL.
    destination : Path
        Final path; parent directories are created.
    timeout : float
        Socket timeout in seconds for each attempt.
    retries : int
        Additional attempts after the first failure.
    verify : Callable[[Path], bool] | None
        Integrity check applied to the partial file before it is renamed.

    Raises
    ------
    OSError, urllib.error.URLError, ValueError
        The last error once every attempt has failed.
    """
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_name(destination.name + ".part")
    for attempt in range(retries + 1):
        try:
            with urllib.request.urlopen(url, timeout=timeout) as response, partial.open("wb") as handle:
                while chunk := response.read(1024 * 1024):
                    handle.write(chunk)
            if verify is not None and not verify(partial):
                raise ValueError(f"Integrity check failed for {url}")
            os.replace(partial, destination)
            return
        except (OSError, urllib.error.URLError, ValueError):
            partial.unlink(missing_ok=True)
            if attempt == retries:
                raise
            time.sleep(min(2 ** attempt, 30))
