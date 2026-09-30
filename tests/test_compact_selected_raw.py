"""Small byte-copy and staging checks for the local selected ECG cache."""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pytest

from ecg_experiment.cache_session_seal import CacheFileSpec, create_seal
from ecg_experiment.compact_selected_raw import build_compact_cache, stage_compact_selected
from ecg_experiment.files import sha256_file, sha256_json


def _source(tmp_path: Path) -> tuple[Path, Path, Path, np.ndarray]:
    values = np.arange(5 * 12 * 2_500, dtype=np.float32).reshape(5, 12, 2_500) / 7
    source = tmp_path / "parent.npy"
    np.save(source, values)
    receipt = tmp_path / "parent.json"
    receipt.write_text(json.dumps({"signals_sha256": sha256_file(source)}))
    seal = tmp_path / "seal.json"
    create_seal(seal, [CacheFileSpec("training_signals", source, receipt,
                                     sha256_file(receipt), "signals_sha256")])
    return source, receipt, seal, values


def _build(tmp_path: Path, source: Path, receipt: Path, seal: Path, indices: np.ndarray) -> Path:
    output = tmp_path / "compact"
    build_compact_cache(
        source, output, seal, indices, seal_sha256=sha256_file(seal),
        source_receipt_sha256=sha256_file(receipt), source_sha256=sha256_file(source),
        selected_sha256=sha256_json(indices.tolist()), source_manifest_sha256="a" * 64,
        builder_path=Path(__file__), chunk_rows=1,
    )
    return output


def test_exact_copy_and_float32_staging(tmp_path: Path) -> None:
    source, receipt, seal, values = _source(tmp_path)
    indices = np.array([4, 1, 3], dtype=np.int64)
    output = _build(tmp_path, source, receipt, seal, indices)
    raw = np.load(output / "signals.npy", allow_pickle=False)
    assert raw.tobytes() == values[[1, 3, 4]].tobytes()
    complete = json.loads((output / "receipt.json").read_text())
    assert complete["selected_indices"] == [4, 1, 3]
    assert complete["sorted_source_rows"] == [1, 3, 4]
    assert all(chunk["source_payload_sha256"] == chunk["compact_payload_sha256"]
               for chunk in complete["chunks"])
    mean = np.arange(12, dtype=np.float32)[:, None]
    std = np.full((12, 1), 3, dtype=np.float32)
    order = np.array([4, 1, 3, 4], dtype=np.int64)
    staged, staged_order = stage_compact_selected(
        output, order, mean, std, expected_receipt_sha256=sha256_file(output / "receipt.json"),
        exposures=4, updates=2, batch_size=2, device="cpu", chunk_rows=2,
    )
    expected = values[[1, 3, 4]].copy()
    expected -= mean
    expected /= std
    assert staged.numpy().tobytes() == expected.tobytes()
    assert staged_order.tolist() == [2, 0, 1, 2]


def test_reject_changed_parent_stat_without_publishing(tmp_path: Path) -> None:
    source, receipt, seal, _ = _source(tmp_path)
    item = source.stat()
    os.utime(source, ns=(item.st_atime_ns, item.st_mtime_ns + 1_000_000))
    with pytest.raises(ValueError, match="stat"):
        _build(tmp_path, source, receipt, seal, np.array([0, 2], dtype=np.int64))
    assert not (tmp_path / "compact").exists()


def test_reject_wrong_selection_hash_and_tampered_compact(tmp_path: Path) -> None:
    source, receipt, seal, _ = _source(tmp_path)
    indices = np.array([1, 4], dtype=np.int64)
    with pytest.raises(ValueError, match="Selected-index"):
        build_compact_cache(
            source, tmp_path / "compact", seal, indices, seal_sha256=sha256_file(seal),
            source_receipt_sha256=sha256_file(receipt), source_sha256=sha256_file(source),
            selected_sha256="0" * 64, source_manifest_sha256="a" * 64,
            builder_path=Path(__file__), chunk_rows=1,
        )
    output = _build(tmp_path, source, receipt, seal, indices)
    with (output / "signals.npy").open("r+b") as handle:
        handle.seek(-1, os.SEEK_END)
        handle.write(b"x")
    with pytest.raises(ValueError, match="signal SHA-256"):
        stage_compact_selected(
            output, np.array([1, 4]), np.zeros((12, 1)), np.ones((12, 1)),
            expected_receipt_sha256=sha256_file(output / "receipt.json"),
            exposures=2, updates=1, batch_size=2, device="cpu",
        )
