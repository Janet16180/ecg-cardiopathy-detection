"""Locate the existing PyTorch CUDA runtime for the frozen xLSTM import path."""

from __future__ import annotations

import os
import sysconfig
from pathlib import Path


def configure_cuda_runtime() -> str:
    """Set CUDA_HOME from installed runtime headers when the user has not configured it."""
    current = os.environ.get("CUDA_HOME")
    path = Path(current) if current else Path(sysconfig.get_path("purelib")) / "nvidia/cuda_runtime"
    if not (path / "include/cuda_runtime.h").is_file():
        raise RuntimeError("xLSTM needs existing CUDA runtime headers; check the pinned uv environment")
    os.environ.setdefault("CUDA_HOME", str(path))
    return str(path)
