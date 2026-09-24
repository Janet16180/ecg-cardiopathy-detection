"""Publish a directory of outputs all at once, or not at all."""

from __future__ import annotations

import os
import shutil
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path


def staging_path(output_dir: Path) -> Path:
    """
    Return a unique hidden staging directory next to ``output_dir``.

    Parameters
    ----------
    output_dir : Path
        Final published directory.

    Returns
    -------
    Path
        ``<parent>/.<name>.staging-<random hex>``.
    """
    return output_dir.parent / f".{output_dir.name}.staging-{uuid.uuid4().hex}"


@contextmanager
def published_directory(output_dir: Path, stage: Path | None = None) -> Iterator[Path]:
    """
    Yield an empty staging directory and rename it to ``output_dir`` on success.

    If the body raises, including on ``KeyboardInterrupt``, the staging
    directory is removed and the exception propagates. ``output_dir`` is never
    created partially.

    Parameters
    ----------
    output_dir : Path
        Final directory; it must not exist when the body finishes.
    stage : Path | None
        Staging directory to create; defaults to :func:`staging_path`.

    Yields
    ------
    Path
        The newly created staging directory.

    Raises
    ------
    FileExistsError
        If ``stage`` already exists, or ``output_dir`` appeared while staging.
    """
    stage = staging_path(output_dir) if stage is None else stage
    stage.parent.mkdir(parents=True, exist_ok=True)
    stage.mkdir()
    try:
        yield stage
        if output_dir.exists():
            raise FileExistsError(f"Output appeared while staging: {output_dir}")
        os.rename(stage, output_dir)
    except BaseException:
        shutil.rmtree(stage, ignore_errors=True)
        raise
