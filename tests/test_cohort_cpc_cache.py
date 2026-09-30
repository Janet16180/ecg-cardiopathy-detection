"""Synthetic contracts for the clean-cohort v3 CPC cache."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pytest

from ecg_experiment import cohort_cpc_cache as cache
from ecg_experiment.files import sha256_file
from ecg_experiment.public_sources import signal_sha256


def _write_json(path: Path, value: object) -> None:
    """Write one test metadata file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value) + "\n")


def _write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    """Write one test manifest or exclusion table."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _waveform(offset: float) -> np.ndarray:
    """Return a distinct finite canonical 500 Hz waveform."""
    base = np.linspace(-0.5, 0.5, 5000, dtype=np.float32)
    return np.ascontiguousarray(np.stack([base + offset + lead / 100 for lead in range(12)]))


def _source_shard(root: Path, relative: str, signal: np.ndarray, chapman: bool) -> None:
    """Publish one canonical shard with the two source metadata formats."""
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, signal[None])
    digest = sha256_file(path)
    metadata = {"complete": True, "dtype": "float32", "sampling_rate_hz": 500, "units": "mV",
                "lead_order": list(cache.LEADS)}
    if chapman:
        metadata["canonical_shape"] = [12, 5000]
        metadata["shards"] = [{"file": path.name, "records": 1, "sha256": digest}]
    else:
        metadata["shape_per_record"] = [12, 5000]
        metadata["shards"] = {path.name: {"shape": [1, 12, 5000], "sha256": digest}}
    _write_json(path.parent / "metadata.json", metadata)


def _cohort(root: Path, monkeypatch: pytest.MonkeyPatch,
            heldout: bool = False) -> tuple[list[np.ndarray], Path]:
    """Create three v3 rows, source receipts and metadata-only exclusions."""
    monkeypatch.setattr(cache, "_tier_count", lambda tier: 3 if tier == 25 else 6)
    signals = [_waveform(offset) for offset in (0.0, 1.0, 2.0)]
    paths = ["data/processed/training_union_500hz_v1/shard_00000.npy",
             "data/raw/challenge-2021/1.0.3/training/ningbo/g1/N1",
             "data/processed/challenge_ecg_views/chapman_strict_10s_v1/shard_00000.npy"]
    _source_shard(root, paths[0], signals[0], chapman=False)
    _source_shard(root, paths[2], signals[2], chapman=True)
    sources = ["ptbxl", "ningbo", "chapman_shaoxing"]
    records = ["ptbxl:1", "ningbo:N1", "chapman_shaoxing:C1"]
    rows = []
    for index, (source, record, path, signal) in enumerate(
        zip(sources, records, paths, signals, strict=True)
    ):
        rows.append({"record_id": record, "patient_id": "ptbxl:1" if index == 0 else "",
                     "source": source, "split": "train",
                     "backend": "challenge_wfdb" if index == 1 else "shard",
                     "path": path, "index": "" if index == 1 else "0", "window_start": "0",
                     "signal_sha256": signal_sha256(signal), "quality_status": "passed", "block": "curated",
                     "order": str(index)})
    if heldout:
        rows[1]["record_id"] = "ningbo:N2"
    cohort_dir = root / "data/processed/clean_25k_v3"
    manifest = cohort_dir / "train_manifest.csv"
    _write_csv(manifest, rows)
    split = root / cache.CHALLENGE_SPLITS
    _write_csv(split, [
        {"source": "ningbo", "record": "N1", "split": "train", "duplicate_group": "g1",
         "signal_sha256": signal_sha256(signals[1]),
         "window_sha256": signal_sha256(signals[1])},
        {"source": "chapman_shaoxing", "record": "C1", "split": "train", "duplicate_group": "g2",
         "signal_sha256": signal_sha256(signals[2]), "window_sha256": signal_sha256(signals[2])},
        {"source": "ningbo", "record": "N2", "split": "test", "duplicate_group": "g3",
         "signal_sha256": signal_sha256(_waveform(3.0)),
         "window_sha256": signal_sha256(_waveform(3.0))},
    ])
    heldout_path = root / cache.HELDOUT_REFERENCES
    _write_csv(heldout_path, [{"record_id": "ptbxl:99", "patient_id": "ptbxl:99"}])
    reference_path = root / cache.PTBXL_REFERENCE
    _write_json(reference_path, {"hashes": [signal_sha256(signals[0]), signal_sha256(_waveform(4.0))]})
    _write_json(root / cache.NORMALIZATION, {"mean": [0.25] * 12, "std": [2.0] * 12})
    _write_json(cohort_dir / "metadata.json", {
        "schema_version": 3, "complete": True, "tier": "25k", "size": 3, "records": 3,
        "block_counts": {"curated": 3}, "quality_status": {"passed": 3},
        "source_counts": dict.fromkeys(sources, 1),
        "table_sha256": {"train_manifest.csv": sha256_file(manifest)},
        "input_sha256": {cache.CHALLENGE_SPLITS.as_posix(): sha256_file(split),
                         cache.HELDOUT_REFERENCES.as_posix(): sha256_file(heldout_path),
                         cache.PTBXL_REFERENCE.as_posix(): sha256_file(reference_path)},
    })
    _write_json(root / cache.COHORT_RECEIPT,
                {"tiers": {"25k": {"metadata_sha256": sha256_file(cohort_dir / "metadata.json")}}})
    monkeypatch.setattr(cache, "read_record", lambda *_: signals[1])
    return signals, manifest


def test_build_verify_and_serve_in_manifest_order(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Scatter grouped source reads into original order and normalize only when serving."""
    signals, _ = _cohort(tmp_path, monkeypatch)
    directory = cache.build_cohort_cpc_cache(25, tmp_path)
    receipt = cache.verify_cohort_cpc_cache(25, tmp_path)
    assert receipt["count"] == 3
    assert receipt["signals_path"] == "data/processed/clean_25k_v3_cpc/signals.npy"
    assert receipt["signals_sha256"] == sha256_file(directory / "signals.npy")
    assert receipt["elapsed_seconds"] > 0
    raw = np.load(directory / "signals.npy")
    for index, signal in enumerate(signals):
        np.testing.assert_array_equal(raw[index], cache.historical_resample(signal))
    dataset = cache.CohortCPCDataset(25, tmp_path)
    assert len(dataset) == 3
    served, target = dataset[1]
    assert target == -1
    np.testing.assert_array_equal(served.numpy(), (raw[1] - 0.25) / 2)


def test_heldout_identity_is_rejected_before_waveform_read(tmp_path: Path,
                                                           monkeypatch: pytest.MonkeyPatch) -> None:
    """Reject an evaluation record using only split metadata."""
    _cohort(tmp_path, monkeypatch, heldout=True)
    monkeypatch.setattr(cache, "read_record", lambda *_: pytest.fail("Evaluation waveform was read"))
    with pytest.raises(ValueError, match="Held-out record"):
        cache.build_cohort_cpc_cache(25, tmp_path)


def test_stale_signal_and_source_shard_are_rejected(tmp_path: Path,
                                                    monkeypatch: pytest.MonkeyPatch) -> None:
    """Reject tampered cache bytes and source shard bytes."""
    _cohort(tmp_path, monkeypatch)
    directory = cache.build_cohort_cpc_cache(25, tmp_path)
    signals = np.load(directory / "signals.npy", mmap_mode="r+")
    signals[0, 0, 0] += 1
    signals.flush()
    with pytest.raises(ValueError, match="stale or incomplete"):
        cache.verify_cohort_cpc_cache(25, tmp_path)
    source = tmp_path / "data/processed/training_union_500hz_v1/shard_00000.npy"
    source_signal = np.load(source, mmap_mode="r+")
    source_signal[0, 0, 0] += 1
    source_signal.flush()
    with pytest.raises(ValueError, match="Input hash mismatch"):
        cache.build_cohort_cpc_cache(25, tmp_path)
