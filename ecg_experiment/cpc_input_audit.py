"""Sample-only comparison of canonical 500 Hz and historical CPC 250 Hz inputs."""

from __future__ import annotations

import json
import sqlite3
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import scipy
from scipy.signal import resample_poly

from .files import read_csv, sha256_file
from .public_sources import signal_sha256
from .training_dataset import TrainingECGDataset
from .waveforms import LEADS, read_record

SOURCE_SHAPE = (12, 5000)
CPC_SHAPE = (12, 2500)
SAMPLE_PER_SOURCE = 32


def historical_resample(signal: np.ndarray) -> np.ndarray:
    """Reproduce the historical CPC transform on separate five-second halves.

    Parameters
    ----------
    signal : np.ndarray
        Finite float32 waveform in canonical lead order, physical mV, 500 Hz.

    Returns
    -------
    np.ndarray
        Float32 ``[12, 2500]`` waveform in physical mV.
    """
    if signal.shape != SOURCE_SHAPE or signal.dtype != np.float32 or not np.isfinite(signal).all():
        raise ValueError("Expected finite float32 [12,5000] input")
    halves = (resample_poly(signal[:, start:start + 2500], 1, 2, axis=1)
              for start in (0, 2500))
    result = np.concatenate(tuple(halves), axis=1).astype(np.float32, copy=False)
    if result.shape != CPC_SHAPE or not np.isfinite(result).all():
        raise ValueError("Unexpected CPC output")
    return result


def evenly_spaced_indices(length: int, count: int = SAMPLE_PER_SOURCE) -> list[int]:
    """Choose distinct deterministic positions including both ends of a population."""
    if length < count or count < 2:
        raise ValueError("Population too small for evenly spaced sample")
    indices = np.linspace(0, length - 1, count, dtype=np.int64).tolist()
    if len(set(indices)) != count:
        raise ValueError("Sample positions repeated")
    return indices


def _checked_small_file(path: Path, expected: str) -> str:
    """Verify a metadata or manifest file against its frozen digest."""
    actual = sha256_file(path)
    if actual != expected:
        raise ValueError(f"Frozen input changed: {path}")
    return actual


def _identity(record_id: str, source: str) -> str:
    """Map namespaced union identity to the historical cache identity."""
    if source == "ptbxl" and record_id.startswith("ptbxl:"):
        return record_id.removeprefix("ptbxl:")
    if source == "mimic" and record_id.startswith("mimic:"):
        return record_id
    raise ValueError(f"Unexpected historical identity: {record_id}")


def _raw_exclusion_paths(root: Path, metadata: dict[str, Any],
                         exclusions: list[dict[str, str]]
                         ) -> tuple[dict[str, tuple[Path, str]], dict[str, str]]:
    """Find excluded records in the frozen PTB and MIMIC train manifests."""
    expected_inputs = metadata["input_sha256"]
    locations: dict[str, tuple[Path, str]] = {}
    pinned: dict[str, str] = {}
    ptb_manifest = root / "data/processed/ptbxl/seed42_fraction1/all_train_ssl.csv"
    mimic_manifest = root / "data/processed/mimic_ssl_40k_cpc/ssl_manifest.csv"
    for path in (ptb_manifest, mimic_manifest):
        pinned[str(path.relative_to(root))] = _checked_small_file(path, expected_inputs[str(path)])
    needed = {row["record_id"] for row in exclusions}
    ptb_raw = root / "data/raw/ptb-xl/1.0.3"
    for row in read_csv(ptb_manifest):
        identity = "ptbxl:" + row["ecg_id"]
        if identity in needed:
            locations[identity] = (ptb_raw, row["filename_hr"])
    for row in read_csv(mimic_manifest):
        if row["ecg_id"] in needed:
            locations[row["ecg_id"]] = (Path(row["raw_dir"]), row["filename_hr"])
    if set(locations) != needed:
        raise ValueError("Excluded record absent from frozen raw manifests")
    return locations, pinned


def audit_inputs(root: Path, dataset_dir: Path, cache_dir: Path) -> dict[str, Any]:  # noqa: C901
    """Audit 64 retained records and every new historical-pool exclusion.

    Only selected waveform cache rows are read. The historical 6.8 GB signal
    array is memory-mapped and its complete-file SHA-256 is inherited from its
    completion receipt, not recomputed by this sampled audit.
    """
    root = root.resolve()
    dataset_dir = dataset_dir.resolve()
    cache_dir = cache_dir.resolve()
    metadata_path = dataset_dir / "metadata.json"
    cache_metadata_path = cache_dir / "complete.json"
    metadata = json.loads(metadata_path.read_text())
    cache_metadata = json.loads(cache_metadata_path.read_text())
    release_receipt_path = root / "outputs/data_quality/training_union_v1/receipt.json"
    release_receipt = json.loads(release_receipt_path.read_text())
    if sha256_file(metadata_path) != release_receipt["metadata_sha256"]:
        raise ValueError("Canonical metadata differs from published release receipt")
    if sha256_file(cache_metadata_path) != metadata["input_sha256"][str(cache_metadata_path)]:
        raise ValueError("Historical CPC completion receipt differs from canonical build input")
    if (not metadata.get("complete") or metadata.get("sampling_rate_hz") != 500
            or metadata.get("shape_per_record") != list(SOURCE_SHAPE)
            or metadata.get("dtype") != "float32" or metadata.get("units") != "mV"
            or metadata.get("lead_order") != list(LEADS)
            or cache_metadata.get("sample_rate_hz") != 250
            or cache_metadata.get("shape") != [60641, *CPC_SHAPE]
            or cache_metadata.get("dtype") != "float32"):
        raise ValueError("Unexpected canonical or historical metadata")
    if scipy.__version__ != cache_metadata["scipy_version"]:
        raise ValueError("SciPy version differs from historical CPC cache")
    input_hashes = {"canonical_release_receipt.json": sha256_file(release_receipt_path),
                    "canonical_metadata.json": sha256_file(metadata_path),
                    "historical_complete.json": sha256_file(cache_metadata_path)}
    for name in ("rows.csv", "ecg_ids.npy"):
        expected = cache_metadata["rows_sha256" if name == "rows.csv" else "ecg_ids_sha256"]
        input_hashes[f"historical_{name}"] = _checked_small_file(cache_dir / name, expected)
    dataset = TrainingECGDataset(dataset_dir, purpose="ssl", max_open_shards=4)
    exclusions = read_csv(dataset_dir / "exclusions.csv")
    if len(exclusions) != 66 or Counter(r["source"] for r in exclusions) != {"ptbxl": 1, "mimic": 65}:
        raise ValueError("Unexpected new exclusion set")
    rows = read_csv(cache_dir / "rows.csv")
    ids = np.load(cache_dir / "ecg_ids.npy", allow_pickle=False)
    signals = np.load(cache_dir / "signals.npy", mmap_mode="r", allow_pickle=False)
    if (signals.shape != tuple(cache_metadata["shape"]) or signals.dtype != np.float32
            or len(rows) != len(ids) or len(rows) != len(signals)
            or any(row["ecg_id"] != str(ids[i]) for i, row in enumerate(rows))):
        raise ValueError("Historical cache shape or row alignment changed")
    historical = {row["ecg_id"]: i for i, row in enumerate(rows)}
    if len(historical) != len(rows):
        raise ValueError("Duplicate historical cache ID")

    selected = []
    for source in ("ptbxl", "mimic"):
        positions = [i for i, row in enumerate(dataset.rows) if row["source"] == source]
        selected.extend(positions[i] for i in evenly_spaced_indices(len(positions)))
    retained = []
    max_abs = 0.0
    exact = 0
    for position in selected:
        item = dataset[position]  # hashes the containing shard and this signal
        record_id, source = item["record_id"], item["source"]
        old_id = _identity(record_id, source)
        old_position = historical[old_id]
        old_row = rows[old_position]
        if (old_row["source"] != source or old_row["split"] != "train"
                or ("ptbxl:" + old_row["patient_id"] if source == "ptbxl"
                    else old_row["patient_id"]) != item["patient_id"]):
            raise ValueError(f"Historical row identity mismatch: {record_id}")
        reduced = historical_resample(item["signal"])
        cached = np.asarray(signals[old_position])
        difference = float(np.max(np.abs(reduced.astype(np.float64) - cached.astype(np.float64))))
        equal = bool(np.array_equal(reduced, cached))
        exact += equal
        max_abs = max(max_abs, difference)
        retained.append({"record_id": record_id, "source": source,
                         "canonical_signal_sha256": dataset.rows[position]["signal_sha256"],
                         "historical_row": old_position, "bitwise_equal": equal,
                         "max_abs_mV": difference})

    locations, manifest_hashes = _raw_exclusion_paths(root, metadata, exclusions)
    input_hashes.update(manifest_hashes)
    db_path = root / "data/processed/mimic_ssl_40k_cpc/audit.sqlite3"
    _checked_small_file(db_path, metadata["input_sha256"][str(db_path)])
    input_hashes["mimic_audit.sqlite3"] = metadata["input_sha256"][str(db_path)]
    with sqlite3.connect(db_path.as_uri() + "?mode=ro&immutable=1", uri=True) as db:
        mimic_expected = dict(db.execute("SELECT name, signal_sha256 FROM outcomes WHERE status='accepted'"))
    excluded = []
    ringing_records = 0
    for row in exclusions:
        record_id, source = row["record_id"], row["source"]
        old_position = historical[_identity(record_id, source)]
        old_row = rows[old_position]
        if old_row["source"] != source or old_row["split"] != "train":
            raise ValueError(f"Historical exclusion identity mismatch: {record_id}")
        raw_dir, relative = locations[record_id]
        original = read_record(raw_dir, relative)
        if signal_sha256(original) != row["signal_sha256"]:
            raise ValueError(f"Excluded raw waveform hash mismatch: {record_id}")
        if source == "mimic" and mimic_expected[relative] != row["signal_sha256"]:
            raise ValueError(f"MIMIC audited waveform hash mismatch: {record_id}")
        constant = [i for i, name in enumerate(LEADS) if name in row["detail"].split(";")]
        if (row["reason"] != "full_constant_lead" or not constant
                or [LEADS[i] for i in np.flatnonzero(np.ptp(original, axis=1) == 0)]
                != row["detail"].split(";")):
            raise ValueError(f"Exclusion reason differs from raw waveform: {record_id}")
        reduced = historical_resample(original)
        cached = np.asarray(signals[old_position])
        equal = bool(np.array_equal(reduced, cached))
        difference = float(np.max(np.abs(reduced.astype(np.float64) - cached.astype(np.float64))))
        changed = [LEADS[i] for i in constant if np.ptp(reduced[i]) > 0]
        ringing_records += bool(changed)
        excluded.append({"record_id": record_id, "source": source,
                         "constant_leads_500hz": row["detail"].split(";"),
                         "nonconstant_after_resampling": changed,
                         "historical_row": old_position, "bitwise_equal": equal,
                         "max_abs_mV": difference})
    return {"scope": ("64 retained sampled (32/source) plus all 66 new exclusions; "
                      "no full cache waveform hash"),
            "input_sha256": input_hashes,
            "historical_signals_receipt_sha256": cache_metadata["signals_sha256"],
            "historical_cache_shape": cache_metadata["shape"],
            "historical_scipy_version": cache_metadata["scipy_version"],
            "current_scipy_version": scipy.__version__,
            "selection": "np.linspace endpoints, 32 positions per source in canonical train_manifest order",
            "retained": retained, "exclusions": excluded,
            "summary": {"retained_count": len(retained), "retained_bitwise_equal": exact,
                        "retained_max_abs_mV": max_abs, "excluded_count": len(excluded),
                        "excluded_bitwise_equal": sum(r["bitwise_equal"] for r in excluded),
                        "excluded_with_resampling_edge_variation": ringing_records}}
