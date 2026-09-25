"""Cheap executable and CUDA preflight for the frozen xECG v11 attempt."""

from __future__ import annotations

import hashlib
import json
import os
import re
import stat
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import torch

from ecg_experiment.files import write_json_atomic
from ecg_experiment.gpu import gpu_lock

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/experiment016_encoder_motion_replication_v11"
STAGES = ("preflight",)
SOURCE_FILES = (
    "docs/experiment-016-encoder-motion-replication-v11.md",
    "ecg_experiment/xecg_encoder_motion_v11.py",
    "ecg_experiment/xecg_encoder_motion_v11_verification.py",
    "scripts/experiments/run_xecg_encoder_motion_replication016_v11.py",
    "tests/test_xecg_encoder_motion_v11.py",
)
SHA256 = re.compile(r"[0-9a-f]{64}\Z")


def safe_leaf(name: str) -> Path:
    """Resolve a repository leaf without path escape or symlink substitution."""
    path = Path(name)
    if path.is_absolute() or ".." in path.parts or path == Path("."):
        raise ValueError(f"Unsafe source path: {name}")
    current = ROOT
    for part in path.parts:
        current = current / part
        if current.is_symlink():
            raise ValueError(f"Symlink substitution: {name}")
    if not current.is_file() or not stat.S_ISREG(current.stat().st_mode):
        raise ValueError(f"Missing or nonregular source: {name}")
    return current


def stable_sha256(path: Path) -> tuple[str, int]:
    """Hash complete bytes while rejecting concurrent file replacement or edits."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        before = os.fstat(handle.fileno())
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
        after_fd = os.fstat(handle.fileno())
    after = path.stat()
    def identity(row: Any) -> tuple[int, ...]:
        """Retain the file attributes that make a byte read stable."""
        return (row.st_dev, row.st_ino, row.st_size, row.st_mtime_ns, row.st_ctime_ns)
    if identity(before) != identity(after_fd) or identity(before) != identity(after):
        raise ValueError(f"Concurrent modification: {path}")
    return digest.hexdigest(), before.st_size


def validate_manifest(path: Path, expected_sha256: str, stage: str) -> dict[str, Any]:
    """Bind the exact command to a versioned source map before large reads."""
    if stage not in STAGES or not SHA256.fullmatch(expected_sha256):
        raise ValueError("Invalid v11 stage or manifest digest")
    if not path.absolute().is_relative_to(OUT) or path.name != "manifest.json":
        raise ValueError("V11 requires its own output-root manifest.json")
    actual, _ = stable_sha256(path)
    if actual != expected_sha256:
        raise ValueError("V11 manifest bytes differ from CLI digest")
    manifest = json.loads(path.read_text())
    if set(manifest) != {"version", "seed", "output_dir", "sources", "sources_sha256", "stages"}:
        raise ValueError("Malformed v11 manifest schema")
    if (
        manifest["version"] != 11
        or manifest["seed"] != 47
        or manifest["output_dir"] != str(OUT.relative_to(ROOT))
        or manifest["stages"] != list(STAGES)
    ):
        raise ValueError("Wrong v11 run identity or production seed")
    source_path = safe_leaf(manifest["sources"])
    source_digest, _ = stable_sha256(source_path)
    if source_digest != manifest["sources_sha256"]:
        raise ValueError("V11 source-map bytes changed")
    mapped = json.loads(source_path.read_text())
    if set(mapped) != set(SOURCE_FILES):
        raise ValueError("V11 source-map coverage differs")
    for name, expected in mapped.items():
        if not SHA256.fullmatch(expected) or stable_sha256(safe_leaf(name))[0] != expected:
            raise ValueError(f"V11 source changed: {name}")
    return {"manifest_sha256": actual, "sources_sha256": source_digest, "source_hashes": mapped}


def installed_cuda_api(cuda: Any) -> None:
    """Require precisely the callable CUDA API used in the device receipt."""
    for name in ("device_count", "is_available", "get_device_name"):
        if not callable(getattr(cuda, name, None)):
            raise RuntimeError(f"Installed CUDA API lacks callable {name}")


def installed_device_api(torch_module: Any) -> None:
    """Require every installed API used by the live device receipt."""
    installed_cuda_api(torch_module.cuda)
    required = (
        (torch_module.backends.cudnn, "version"),
        (torch_module, "are_deterministic_algorithms_enabled"),
        (torch_module, "get_float32_matmul_precision"),
    )
    for owner, name in required:
        if not callable(getattr(owner, name, None)):
            raise RuntimeError(f"Installed torch API lacks callable {name}")
    if not isinstance(torch_module.version.cuda, str):
        raise RuntimeError("Installed torch lacks a CUDA version string")


def device_receipt(torch_module: Any, run_query: Any) -> dict[str, Any]:
    """Query the real one-device V100 using installed torch and nvidia-smi."""
    cuda = torch_module.cuda
    installed_device_api(torch_module)
    try:
        count = cuda.device_count()
        available = cuda.is_available()
        name = cuda.get_device_name(0) if count == 1 and available else None
    except Exception as exc:
        raise RuntimeError("Malformed CUDA device query") from exc
    if not isinstance(count, int) or isinstance(count, bool) or count != 1 or available is not True:
        raise RuntimeError("V11 requires exactly one available CUDA device")
    if not isinstance(name, str) or "V100" not in name:
        raise RuntimeError("V11 requires the verified V100 device class")
    query = run_query(
        ["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    lines = query.splitlines()
    if len(lines) != 1:
        raise RuntimeError("Malformed nvidia-smi query")
    fields = [part.strip() for part in lines[0].split(",")]
    if len(fields) != 3 or "V100" not in fields[0] or not re.fullmatch(r"\d+(?:\.\d+)+", fields[1]):
        raise RuntimeError("Malformed nvidia-smi device identity")
    memory = re.fullmatch(r"(\d+) MiB", fields[2])
    if memory is None or int(memory.group(1)) != 16384:
        raise RuntimeError("V11 requires a 16 GB V100")
    if str(torch_module.__version__) != "2.6.0+cu124" or torch_module.version.cuda != "12.4":
        raise RuntimeError("Torch/CUDA differs from frozen v9 environment")
    return {
        "torch_device": name,
        "torch": str(torch_module.__version__),
        "torch_cuda": torch_module.version.cuda,
        "nvidia_smi": query,
        "driver_version": fields[1],
        "historical_driver_version": "not recorded in frozen v9 receipts",
        "cudnn": torch_module.backends.cudnn.version(),
        "deterministic_algorithms": torch_module.are_deterministic_algorithms_enabled(),
        "float32_matmul_precision": torch_module.get_float32_matmul_precision(),
    }


def preflight(manifest: Path, digest: str, stage: str, device: str) -> dict[str, Any]:
    """Perform cheap identity and API checks before a historical input read."""
    started = time.monotonic()
    if device not in ("cpu", "cuda"):
        raise ValueError("Unknown v11 device")
    identity = validate_manifest(manifest, digest, stage)
    installed_device_api(torch)
    receipt: dict[str, Any] = {
        "stage": stage,
        "device": device,
        "identity": identity,
        "command": sys.argv,
        "torch": str(torch.__version__),
        "torch_cuda": torch.version.cuda,
        "device_api": [
            "cuda.device_count",
            "cuda.is_available",
            "cuda.get_device_name",
            "backends.cudnn.version",
            "are_deterministic_algorithms_enabled",
            "get_float32_matmul_precision",
        ],
    }
    if device == "cuda":
        with gpu_lock("cuda", blocking=False):
            receipt["device_receipt"] = device_receipt(torch, subprocess.run)
    receipt["seconds"] = time.monotonic() - started
    receipt["status"] = "passed_before_large_input_reads"
    write_json_atomic(manifest.parent / "preflight.json", receipt)
    return receipt
