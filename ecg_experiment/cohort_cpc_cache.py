"""Verified 250 Hz CPC caches for the curated clean-cohort v3 tiers."""

from __future__ import annotations

import csv
import json
import re
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from ecg_experiment import ROOT

from .cpc_input_audit import historical_resample
from .files import sha256_file
from .paths import to_stored
from .public_sources import signal_sha256
from .waveforms import LEADS, read_record

SOURCE_SHAPE = (12, 5000)
CPC_SHAPE = (12, 2500)
CHALLENGE_ROOT = Path("data/raw/challenge-2021/1.0.3")
NORMALIZATION = Path("outputs/experiment004_cpc_40k/normalization.json")
CHALLENGE_SPLITS = Path("data/processed/challenge_splits_v1/rows.csv")
HELDOUT_REFERENCES = Path("data/processed/training_union_500hz_v1/heldout_references.csv")
PTBXL_REFERENCE = Path("outputs/data_quality/ptbxl_reference_hashes_v2.json")
COHORT_RECEIPT = Path("outputs/data_quality/clean_cohorts_v3/receipt.json")
REQUIRED_COLUMNS = {"record_id", "patient_id", "source", "split", "backend", "path", "index",
                    "window_start", "signal_sha256", "quality_status", "block", "order"}
HEX_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
CHALLENGE_SOURCES = {"ningbo", "chapman_shaoxing", "georgia", "cpsc_2018", "cpsc_2018_extra"}


def _tier_count(tier: int) -> int:
    """Return the exact curated tier size."""
    if tier not in (25, 50):
        raise ValueError("Only the curated 25k and 50k v3 cohorts are supported")
    return tier * 1000


def _stored(path: Path, root: Path) -> str:
    """Store repository-relative paths, including paths below a synthetic test root."""
    if root == ROOT:
        return to_stored(path)
    return path.relative_to(root).as_posix()


def _input_path(root: Path, relative: str) -> Path:
    """Accept only paths within the selected repository root."""
    path = Path(relative)
    if path.is_absolute() or ".." in path.parts or not path.parts:
        raise ValueError(f"Unsafe cohort path: {relative}")
    return root / path


def _checked_hash(root: Path, relative: str, expected: str) -> None:
    """Verify a frozen metadata file or source shard by SHA-256."""
    if not HEX_SHA256.fullmatch(expected) or sha256_file(_input_path(root, relative)) != expected:
        raise ValueError(f"Input hash mismatch: {relative}")


def _exclusion_sets(root: Path, metadata: dict[str, Any]
                    ) -> tuple[set[str], set[str], set[str], set[str], set[str]]:
    """Read held-out identities and hashes from metadata, never their ECGs or labels."""
    for relative in (CHALLENGE_SPLITS, HELDOUT_REFERENCES, PTBXL_REFERENCE):
        _checked_hash(root, relative.as_posix(), metadata["input_sha256"][relative.as_posix()])
    known_challenge: set[str] = set()
    held_ids: set[str] = set()
    held_patients: set[str] = set()
    held_hashes: set[str] = set()
    challenge_columns = ["source", "record", "split", "duplicate_group", "signal_sha256", "window_sha256"]
    challenge = pd.read_csv(root / CHALLENGE_SPLITS, dtype=str, keep_default_na=False,
                            usecols=challenge_columns)
    if (challenge.groupby("duplicate_group")["split"].nunique() > 1).any():
        raise ValueError("Challenge duplicate group spans train and evaluation")
    identities = challenge["source"] + ":" + challenge["record"]
    known_challenge.update(identities)
    held = challenge["split"] != "train"
    held_ids.update(identities[held])
    held_hashes.update(challenge.loc[held, "signal_sha256"])
    held_hashes.update(challenge.loc[held, "window_sha256"])
    held_hashes.discard("")
    references = pd.read_csv(root / HELDOUT_REFERENCES, dtype=str, keep_default_na=False,
                             usecols=["record_id", "patient_id"])
    held_ids.update(references["record_id"])
    held_patients.update(references["patient_id"])
    held_patients.discard("")
    ptb_hashes = set(json.loads((root / PTBXL_REFERENCE).read_text())["hashes"])
    return known_challenge, held_ids, held_patients, held_hashes, ptb_hashes


def _validated_manifest(tier: int, root: Path) -> tuple[list[dict[str, str]], dict[str, str]]:  # noqa: C901
    """Verify the v3 manifest and its metadata-only evaluation exclusions."""
    count = _tier_count(tier)
    directory = root / f"data/processed/clean_{tier}k_v3"
    metadata_path = directory / "metadata.json"
    metadata = json.loads(metadata_path.read_text())
    receipt_path = root / COHORT_RECEIPT
    published = json.loads(receipt_path.read_text())
    metadata_hash = sha256_file(metadata_path)
    if published["tiers"][f"{tier}k"]["metadata_sha256"] != metadata_hash:
        raise ValueError("Cohort metadata differs from published v3 receipt")
    if (metadata.get("schema_version") != 3 or metadata.get("complete") is not True
            or metadata.get("tier") != f"{tier}k" or metadata.get("size") != count
            or metadata.get("records") != count or metadata.get("block_counts") != {"curated": count}
            or metadata.get("quality_status") != {"passed": count}):
        raise ValueError("Unexpected cohort v3 metadata")
    for name, expected in metadata["table_sha256"].items():
        _checked_hash(root, _stored(directory / name, root), expected)
    with (directory / "train_manifest.csv").open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        if not set(reader.fieldnames or []) >= REQUIRED_COLUMNS:
            raise ValueError("Incomplete cohort manifest")
        rows = list(reader)
    if len(rows) != count or dict(Counter(row["source"] for row in rows)) != metadata["source_counts"]:
        raise ValueError("Cohort count or source composition differs from metadata")
    known, held_ids, held_patients, held_hashes, ptb_hashes = _exclusion_sets(root, metadata)
    seen_ids: set[str] = set()
    seen_hashes: set[str] = set()
    for position, row in enumerate(rows):
        identity = row["record_id"]
        digest = row["signal_sha256"]
        source = row["source"]
        if (row["split"] != "train" or row["block"] != "curated"
                or row["quality_status"] != "passed" or row["order"] != str(position)
                or row["window_start"] != "0" or not HEX_SHA256.fullmatch(digest)
                or identity in seen_ids or digest in seen_hashes):
            raise ValueError(f"Invalid cohort row: {identity}")
        if identity in held_ids or row["patient_id"] in held_patients or digest in held_hashes:
            raise ValueError(f"Held-out record in cohort: {identity}")
        if source != "ptbxl" and digest in ptb_hashes:
            raise ValueError(f"PTB-XL waveform overlap: {identity}")
        if source in CHALLENGE_SOURCES and identity not in known:
            raise ValueError(f"Challenge record absent from split: {identity}")
        if source not in CHALLENGE_SOURCES | {"ptbxl"}:
            raise ValueError(f"Non-curated source: {identity}")
        if row["backend"] == "challenge_wfdb":
            if source != "ningbo" or row["index"] or not Path(row["path"]).is_relative_to(CHALLENGE_ROOT):
                raise ValueError(f"Invalid Ningbo pointer: {identity}")
        elif row["backend"] == "shard":
            if source == "ningbo" or not row["index"].isdigit():
                raise ValueError(f"Invalid shard pointer: {identity}")
        else:
            raise ValueError(f"Unsupported backend: {row['backend']}")
        _input_path(root, row["path"])
        seen_ids.add(identity)
        seen_hashes.add(digest)
    hashes = {"metadata": metadata_hash, "published_receipt": sha256_file(receipt_path),
              "manifest": metadata["table_sha256"]["train_manifest.csv"],
              "challenge_splits": metadata["input_sha256"][CHALLENGE_SPLITS.as_posix()],
              "heldout_references": metadata["input_sha256"][HELDOUT_REFERENCES.as_posix()],
              "ptbxl_reference": metadata["input_sha256"][PTBXL_REFERENCE.as_posix()]}
    return rows, hashes


def _source_shards(rows: list[dict[str, str]], root: Path
                   ) -> tuple[dict[str, dict[str, Any]], dict[str, str]]:
    """Verify each referenced source shard against its published metadata."""
    names = {row["path"] for row in rows if row["backend"] == "shard"}
    shard_info: dict[str, dict[str, Any]] = {}
    metadata_hashes: dict[str, str] = {}
    source_metadata: dict[str, dict[str, Any]] = {}
    for relative in sorted(names):
        path = _input_path(root, relative)
        metadata_path = path.parent / "metadata.json"
        metadata_relative = _stored(metadata_path, root)
        if metadata_relative not in source_metadata:
            metadata_hashes[metadata_relative] = sha256_file(metadata_path)
            source_metadata[metadata_relative] = json.loads(metadata_path.read_text())
        source = source_metadata[metadata_relative]
        if (source.get("complete") is not True
                or source.get("shape_per_record", source.get("canonical_shape")) != list(SOURCE_SHAPE)
                or source.get("dtype") != "float32" or source.get("sampling_rate_hz") != 500
                or source.get("units") != "mV" or source.get("lead_order") != list(LEADS)):
            raise ValueError(f"Invalid source shard metadata: {metadata_relative}")
        published = source.get("shards", {})
        if isinstance(published, list):
            info = next((item for item in published if item["file"] == path.name), None)
            if info is not None:
                info = {"sha256": info["sha256"], "shape": [info["records"], *SOURCE_SHAPE]}
        else:
            info = published.get(path.name)
        if not info or info.get("shape", [])[1:] != list(SOURCE_SHAPE):
            raise ValueError(f"Shard absent from metadata: {relative}")
        _checked_hash(root, relative, info["sha256"])
        shard_info[relative] = info
    for row in rows:
        if row["backend"] == "shard" and int(row["index"]) >= shard_info[row["path"]]["shape"][0]:
            raise ValueError(f"Shard index out of bounds: {row['record_id']}")
    return shard_info, metadata_hashes


def _normalization(root: Path) -> tuple[np.ndarray, np.ndarray, str]:
    """Load the frozen Experiment 004 per-lead training mean and std."""
    path = root / NORMALIZATION
    values = json.loads(path.read_text())
    mean = np.asarray(values["mean"], dtype=np.float32)
    std = np.asarray(values["std"], dtype=np.float32)
    if (mean.shape != (12,) or std.shape != (12,) or not np.isfinite(mean).all()
            or not np.isfinite(std).all() or np.any(std <= 0)):
        raise ValueError("Invalid CPC normalization")
    return mean, std, sha256_file(path)


def _cache_dir(tier: int, root: Path) -> Path:
    """Return one cache directory after checking the supported tier."""
    _tier_count(tier)
    return root / f"data/processed/clean_{tier}k_v3_cpc"


def _code_hashes() -> dict[str, str]:
    """Pin the waveform reader, hash and historical transform implementations."""
    names = ("ecg_experiment/cpc_input_audit.py", "ecg_experiment/public_sources.py",
             "ecg_experiment/waveforms.py", "ecg_experiment/cohort_cpc_cache.py")
    return {name: sha256_file(ROOT / name) for name in names}


def verify_cohort_cpc_cache(tier: int, root: Path = ROOT) -> dict[str, Any]:
    """Verify a completed cache, its full signal file and source receipts.

    Parameters
    ----------
    tier : int
        Curated v3 tier in thousands: 25 or 50.
    root : Path
        Repository root, or a temporary synthetic root for tests.

    Returns
    -------
    dict[str, Any]
        Verified completion receipt.
    """
    root = Path(root).absolute()
    rows, manifest_hashes = _validated_manifest(tier, root)
    shards, metadata_hashes = _source_shards(rows, root)
    _, _, normalization_hash = _normalization(root)
    directory = _cache_dir(tier, root)
    receipt = json.loads((directory / "complete.json").read_text())
    signal_path = directory / "signals.npy"
    if (receipt.get("complete") is not True or receipt.get("tier") != f"{tier}k"
            or receipt.get("count") != len(rows) or receipt.get("shape") != [len(rows), *CPC_SHAPE]
            or receipt.get("dtype") != "float32" or receipt.get("sample_rate_hz") != 250
            or receipt.get("manifest_sha256") != manifest_hashes
            or receipt.get("source_metadata_sha256") != metadata_hashes
            or receipt.get("source_shard_sha256") != {name: info["sha256"] for name, info in shards.items()}
            or receipt.get("normalization_sha256") != normalization_hash
            or receipt.get("code_sha256") != _code_hashes()
            or receipt.get("signals_path") != _stored(signal_path, root)
            or not isinstance(receipt.get("elapsed_seconds"), (int, float))
            or receipt["elapsed_seconds"] <= 0
            or sha256_file(signal_path) != receipt.get("signals_sha256")):
        raise ValueError("CPC cache is stale or incomplete")
    signal = np.load(signal_path, mmap_mode="r", allow_pickle=False)
    if signal.shape != (len(rows), *CPC_SHAPE) or signal.dtype != np.float32:
        raise ValueError("CPC cache array contract mismatch")
    return receipt


def _write_shard_group(output: np.ndarray, root: Path, relative: str,
                       selected: list[tuple[int, dict[str, str]]], info: dict[str, Any]) -> None:
    """Read one verified source shard sequentially and scatter rows into cohort order."""
    shard = np.load(_input_path(root, relative), mmap_mode="r", allow_pickle=False)
    if list(shard.shape) != info["shape"] or shard.dtype != np.float32:
        raise ValueError(f"Source shard array contract mismatch: {relative}")
    for position, row in sorted(selected, key=lambda item: int(item[1]["index"])):
        signal = np.asarray(shard[int(row["index"])])
        if signal.shape != SOURCE_SHAPE or signal_sha256(signal) != row["signal_sha256"]:
            raise ValueError(f"Source row hash mismatch: {row['record_id']}")
        output[position] = historical_resample(signal)


def _write_challenge_group(output: np.ndarray, root: Path,
                           selected: list[tuple[int, dict[str, str]]]) -> None:
    """Read and hash Ningbo train ECGs before applying the historical CPC transform."""
    for position, row in selected:
        relative_record = str(Path(row["path"]).relative_to(CHALLENGE_ROOT))
        signal = read_record(root / CHALLENGE_ROOT, relative_record)
        if signal.shape != SOURCE_SHAPE or signal_sha256(signal) != row["signal_sha256"]:
            raise ValueError(f"Challenge row hash mismatch: {row['record_id']}")
        output[position] = historical_resample(signal)


def build_cohort_cpc_cache(tier: int, root: Path = ROOT) -> Path:
    """Build a verified raw-mV 250 Hz cache in original v3 manifest order.

    Parameters
    ----------
    tier : int
        Curated v3 tier in thousands: 25 or 50.
    root : Path
        Repository root, or a temporary synthetic root for tests.

    Returns
    -------
    Path
        Completed cache directory.
    """
    started = time.perf_counter()
    root = Path(root).absolute()
    rows, manifest_hashes = _validated_manifest(tier, root)
    shards, metadata_hashes = _source_shards(rows, root)
    _, _, normalization_hash = _normalization(root)
    directory = _cache_dir(tier, root)
    if directory.exists():
        verify_cohort_cpc_cache(tier, root)
        return directory
    stage = directory.with_name(directory.name + ".partial")
    if stage.exists():
        raise ValueError(f"Partial CPC cache exists: {stage}")
    stage.mkdir(parents=True)
    output = np.lib.format.open_memmap(stage / "signals.npy", mode="w+", dtype=np.float32,
                                       shape=(len(rows), *CPC_SHAPE))
    grouped: dict[str, list[tuple[int, dict[str, str]]]] = defaultdict(list)
    for position, row in enumerate(rows):
        grouped[row["path"]].append((position, row))
    for relative, selected in grouped.items():
        if selected[0][1]["backend"] == "shard":
            _write_shard_group(output, root, relative, selected, shards[relative])
        else:
            _write_challenge_group(output, root, selected)
    output.flush()
    del output
    receipt = {"complete": True, "tier": f"{tier}k", "count": len(rows),
               "shape": [len(rows), *CPC_SHAPE], "dtype": "float32", "sample_rate_hz": 250,
               "signals_path": _stored(directory / "signals.npy", root),
               "signals_sha256": sha256_file(stage / "signals.npy"),
               "manifest_sha256": manifest_hashes, "source_metadata_sha256": metadata_hashes,
               "source_shard_sha256": {name: info["sha256"] for name, info in shards.items()},
               "normalization_path": _stored(root / NORMALIZATION, root),
               "normalization_sha256": normalization_hash, "code_sha256": _code_hashes(),
               "elapsed_seconds": time.perf_counter() - started}
    (stage / "complete.json").write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    stage.rename(directory)
    return directory


class CohortCPCDataset:
    """Normalized CPC signals with a masked SSL target for PyTorch training."""

    def __init__(self, tier: int, root: Path = ROOT, load_into_ram: bool = True) -> None:
        """Verify a cache and optionally copy it into RAM for shuffled batches.

        Parameters
        ----------
        tier : int
            Curated v3 tier in thousands.
        root : Path
            Repository root.
        load_into_ram : bool
            Copy the array into RAM when true.
        """
        self.root = Path(root).absolute()
        self.receipt = verify_cohort_cpc_cache(tier, self.root)
        self.mean, self.std, _ = _normalization(self.root)
        signals = np.load(_cache_dir(tier, self.root) / "signals.npy", mmap_mode="r", allow_pickle=False)
        self.signals = np.array(signals, copy=True) if load_into_ram else signals

    def __len__(self) -> int:
        """Return the exact number of ECGs."""
        return len(self.signals)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        """Return one normalized [12,2500] waveform and target -1."""
        signal = (np.asarray(self.signals[index], dtype=np.float32) - self.mean[:, None]) / self.std[:, None]
        return torch.from_numpy(np.ascontiguousarray(signal)), -1
