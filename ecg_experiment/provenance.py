"""Source-revision lookups recorded in experiment and feature metadata."""

from __future__ import annotations

import subprocess
from pathlib import Path


def git_head(path: Path) -> str | None:
    """
    Return the checked-out commit of a Git working tree.

    Parameters
    ----------
    path : Path
        Any directory inside the working tree.

    Returns
    -------
    str | None
        Full commit hash, or ``None`` when Git is unavailable or ``path`` is
        not inside a repository.
    """
    try:
        return subprocess.check_output(["git", "-C", str(path), "rev-parse", "HEAD"],
                                       stderr=subprocess.DEVNULL, text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        return None
