"""The single lock that serializes project jobs on the shared GPU."""

from __future__ import annotations

import fcntl
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

GPU_LOCK_PATH = Path("/tmp/ecg_project_gpu.lock")


@contextmanager
def gpu_lock(device: str, *, blocking: bool = True, path: Path = GPU_LOCK_PATH) -> Iterator[None]:
    """
    Hold the shared GPU lock while a CUDA job runs.

    CPU jobs do not take the lock.

    Parameters
    ----------
    device : str
        Torch device name; only ``"cuda"`` acquires the lock.
    blocking : bool
        Wait for the lock instead of failing when another job holds it.
    path : Path
        Lock file shared by all project runners.

    Yields
    ------
    None
        Control while the lock is held.

    Raises
    ------
    RuntimeError
        If ``blocking`` is false and another job holds the lock.
    """
    if device != "cuda":
        yield
        return
    with path.open("a+") as handle:
        flags = fcntl.LOCK_EX if blocking else fcntl.LOCK_EX | fcntl.LOCK_NB
        if blocking:
            print(f"Waiting for GPU lock {path}", flush=True)
        try:
            fcntl.flock(handle, flags)
        except BlockingIOError as exc:
            raise RuntimeError(f"GPU is reserved by another project run: {path}") from exc
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)
