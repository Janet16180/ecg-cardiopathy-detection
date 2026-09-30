"""Build and stage a locally sealed, byte-exact selected ECG cache."""

from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .cache_session_seal import validate_seal
from .files import sha256_file, sha256_json

VERSION = 1
ROW_SHAPE = (12, 2_500)
DTYPE = np.dtype("<f4")
ROW_BYTES = math.prod(ROW_SHAPE) * DTYPE.itemsize


def _stat(path: Path) -> dict[str, int]:
    """Return the file identity fields used by the parent seal."""
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"Expected a regular, non-symlink file: {path}")
    item = path.stat()
    return {"size": item.st_size, "dev": item.st_dev, "inode": item.st_ino,
            "mtime_ns": item.st_mtime_ns, "ctime_ns": item.st_ctime_ns}


def _header(path: Path) -> tuple[tuple[int, ...], np.dtype[Any], bool, int]:
    """Read the NPY contract and payload offset without loading waveforms."""
    with path.open("rb") as handle:
        version = np.lib.format.read_magic(handle)
        if version == (1, 0):
            shape, fortran, dtype = np.lib.format.read_array_header_1_0(handle)
        elif version in ((2, 0), (3, 0)):
            shape, fortran, dtype = np.lib.format.read_array_header_2_0(handle)
        else:
            raise ValueError(f"Unsupported NPY version: {version}")
        return shape, dtype, fortran, handle.tell()


def _check_indices(indices: np.ndarray, rows: int, expected_sha256: str) -> np.ndarray:
    """Verify the original selection order and return ascending cache rows."""
    if indices.ndim != 1 or indices.dtype.kind not in "iu" or not len(indices):
        raise ValueError("Selected indices must be a nonempty integer vector")
    selected = np.sort(indices.astype(np.int64))
    if selected[0] < 0 or selected[-1] >= rows or np.any(np.diff(selected) == 0):
        raise ValueError("Selected indices are out of range or duplicated")
    if sha256_json(indices.tolist()) != expected_sha256:
        raise ValueError("Selected-index SHA-256 mismatch")
    return selected


def _verify_parent(
    seal_path: Path, seal_sha256: str, source_path: Path,
    source_receipt_sha256: str, source_sha256: str,
) -> tuple[dict[str, Any], dict[str, int]]:
    """Check the pinned seal, receipt, source path, and parent stat."""
    if sha256_file(seal_path) != seal_sha256:
        raise ValueError("Parent seal file SHA-256 changed")
    checked = validate_seal(seal_path)
    seal = json.loads(seal_path.read_text())
    entry = seal["files"]["training_signals"]
    if (Path(entry["path"]) != source_path.resolve()
            or entry["full_sha256"] != source_sha256
            or entry["source_receipt"]["sha256"] != source_receipt_sha256
            or checked["full_sha256_from_creation"]["training_signals"] != source_sha256):
        raise ValueError("Parent seal does not pin the requested training cache")
    stat = _stat(source_path)
    if stat != entry["stat"] or sha256_file(seal_path) != seal_sha256:
        raise ValueError("Parent cache stat or seal changed")
    return entry, stat


def _copy_chunks(
    source_path: Path, signal_path: Path, selected: np.ndarray,
    source_offset: int, chunk_rows: int, parent_stat: dict[str, int],
) -> tuple[list[dict[str, Any]], int]:
    """Copy bounded raw row chunks and compare source with readback hashes."""
    chunks = []
    with source_path.open("rb", buffering=0) as source, signal_path.open("w+b", buffering=0) as target:
        np.lib.format.write_array_header_1_0(
            target, {"descr": DTYPE.str, "fortran_order": False,
                     "shape": (len(selected), *ROW_SHAPE)},
        )
        compact_offset = target.tell()
        for first in range(0, len(selected), chunk_rows):
            batch = selected[first:first + chunk_rows]
            raw = bytearray()
            for row in batch:
                payload = os.pread(source.fileno(), ROW_BYTES, source_offset + int(row) * ROW_BYTES)
                if len(payload) != ROW_BYTES:
                    raise ValueError("Short parent waveform read")
                raw.extend(payload)
            digest = hashlib.sha256(raw).hexdigest()
            target.write(raw)
            target.flush()
            position = compact_offset + first * ROW_BYTES
            target.seek(position)
            written = target.read(len(raw))
            if hashlib.sha256(written).hexdigest() != digest:
                raise ValueError("Compact chunk differs from source bytes")
            target.seek(0, os.SEEK_END)
            chunks.append({"first_output_row": first, "row_count": len(batch),
                           "source_first_row": int(batch[0]),
                           "source_last_row": int(batch[-1]),
                           "source_payload_sha256": digest,
                           "compact_payload_sha256": hashlib.sha256(written).hexdigest()})
            if _stat(source_path) != parent_stat:
                raise ValueError("Parent cache changed during copy")
        target.flush()
        os.fsync(target.fileno())
    return chunks, compact_offset


def build_compact_cache(
    source_path: Path, output_dir: Path, seal_path: Path, indices: np.ndarray,
    *, seal_sha256: str, source_receipt_sha256: str, source_sha256: str,
    selected_sha256: str, source_manifest_sha256: str, builder_path: Path,
    entrypoint_path: Path | None = None, chunk_rows: int = 128,
) -> dict[str, Any]:
    """Publish a new NPY and receipt after byte-level copy verification.

    ``indices`` retains Experiment 019's shuffled selection order. Output
    rows are ascending original cache indices. The existing destination is
    never overwritten; a failed build removes only its private staging dir.
    """
    source_path, output_dir, seal_path = map(Path, (source_path, output_dir, seal_path))
    if chunk_rows < 1 or chunk_rows * ROW_BYTES > 64 * 1024 * 1024:
        raise ValueError("Chunk must fit the 64 MiB copy bound")
    if output_dir.exists():
        raise FileExistsError(output_dir)
    entry, parent_stat = _verify_parent(
        seal_path, seal_sha256, source_path, source_receipt_sha256, source_sha256,
    )
    shape, dtype, fortran, offset = _header(source_path)
    if (len(shape) != 3 or shape[1:] != ROW_SHAPE or dtype != DTYPE or fortran
            or entry["npy"]["shape"] != list(shape)
            or parent_stat["size"] != offset + shape[0] * ROW_BYTES):
        raise ValueError("Parent NPY payload contract changed")
    selected = _check_indices(indices, shape[0], selected_sha256)
    builder_sha256 = sha256_file(builder_path)
    entrypoint_sha256 = sha256_file(entrypoint_path) if entrypoint_path else None
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}.", dir=output_dir.parent))
    try:
        signal_path = stage / "signals.npy"
        chunks, compact_offset = _copy_chunks(
            source_path, signal_path, selected, offset, chunk_rows, parent_stat,
        )
        compact_shape, compact_dtype, compact_fortran, compact_offset_check = _header(signal_path)
        if (compact_shape != (len(selected), *ROW_SHAPE) or compact_dtype != DTYPE
                or compact_fortran or compact_offset_check != compact_offset):
            raise ValueError("Compact NPY header changed")
        compact_sha256 = sha256_file(signal_path)
        _verify_parent(seal_path, seal_sha256, source_path,
                       source_receipt_sha256, source_sha256)
        if (_stat(source_path) != parent_stat or sha256_file(builder_path) != builder_sha256
                or (entrypoint_path and sha256_file(entrypoint_path) != entrypoint_sha256)):
            raise ValueError("Parent cache or builder source changed during copy")
        receipt = {
            "version": VERSION, "selected_indices": indices.astype(np.int64).tolist(),
            "selected_indices_sha256": selected_sha256,
            "sorted_source_rows": selected.tolist(),
            "source_manifest_sha256": source_manifest_sha256,
            "parent": {"seal_path": str(seal_path.resolve()), "seal_file_sha256": seal_sha256,
                       "seal_content_sha256": json.loads(seal_path.read_text())["seal_sha256"],
                       "source_path": str(source_path.resolve()), "source_stat": parent_stat,
                       "source_receipt_sha256": source_receipt_sha256,
                       "source_signals_sha256": source_sha256},
            "builder": {"path": str(Path(builder_path).resolve()), "sha256": builder_sha256,
                        "entrypoint_path": str(Path(entrypoint_path).resolve()) if entrypoint_path else None,
                        "entrypoint_sha256": entrypoint_sha256},
            "signals": {"path": "signals.npy", "shape": list(compact_shape),
                        "dtype": compact_dtype.str, "fortran_order": False,
                        "payload_offset": compact_offset, "row_bytes": ROW_BYTES,
                        "sha256": compact_sha256},
            "chunks": chunks,
        }
        receipt_path = stage / "receipt.json"
        with receipt_path.open("w", encoding="utf-8") as handle:
            json.dump(receipt, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        if output_dir.exists():
            raise FileExistsError(output_dir)
        os.rename(stage, output_dir)
        return receipt
    finally:
        if stage.exists():
            shutil.rmtree(stage)


def stage_compact_selected(
    compact_dir: Path, order: np.ndarray, mean: np.ndarray, std: np.ndarray,
    *, expected_receipt_sha256: str, exposures: int = 115_359,
    updates: int = 902, batch_size: int = 128, device: str = "cuda",
    chunk_rows: int = 128,
) -> tuple[torch.Tensor, np.ndarray]:
    """Stage contiguous compact rows with the historical float32 operations."""
    compact_dir = Path(compact_dir)
    receipt_path = compact_dir / "receipt.json"
    if sha256_file(receipt_path) != expected_receipt_sha256:
        raise ValueError("Compact receipt SHA-256 mismatch")
    receipt = json.loads(receipt_path.read_text())
    selected = np.asarray(receipt["sorted_source_rows"], dtype=np.int64)
    if (receipt["version"] != VERSION or len(selected) == 0
            or np.any(np.diff(selected) <= 0) or len(order) != exposures
            or math.ceil(exposures / batch_size) != updates or chunk_rows < 1
            or order.ndim != 1):
        raise ValueError("Invalid compact staging contract")
    positions = np.full(int(selected[-1]) + 1, -1, dtype=np.int32)
    positions[selected] = np.arange(len(selected), dtype=np.int32)
    if np.any(order < 0) or np.any(order >= len(positions)):
        raise ValueError("Exposure order is outside parent cache")
    staged_order = positions[order]
    if np.any(staged_order < 0) or len(np.unique(order)) != len(selected):
        raise ValueError("Exposure order does not cover the selected subset")
    path = compact_dir / "signals.npy"
    if sha256_file(path) != receipt["signals"]["sha256"]:
        raise ValueError("Compact signal SHA-256 mismatch")
    signals = np.load(path, mmap_mode="r", allow_pickle=False)
    if (signals.shape != tuple(receipt["signals"]["shape"])
            or signals.shape != (len(selected), *ROW_SHAPE) or signals.dtype != DTYPE):
        raise ValueError("Compact signal contract mismatch")
    mean = np.asarray(mean, dtype=np.float32)
    std = np.asarray(std, dtype=np.float32)
    if (mean.shape != (12, 1) or std.shape != (12, 1)
            or not np.isfinite(mean).all() or not np.isfinite(std).all()
            or np.any(std <= 0)):
        raise ValueError("Invalid train-only normalizer")
    staged = torch.empty(signals.shape, dtype=torch.float32, device=device)
    for first in range(0, len(selected), chunk_rows):
        block = np.array(signals[first:first + chunk_rows], dtype=np.float32, copy=True)
        block -= mean
        block /= std
        if not np.isfinite(block).all():
            raise ValueError("Nonfinite normalized waveform")
        staged[first:first + len(block)] = torch.from_numpy(block).to(device)
    return staged, staged_order
