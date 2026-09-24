"""Versioned canonical training data; labels are opt-in and SSL never exposes them."""

from __future__ import annotations

import hashlib
import json
from collections import OrderedDict
from pathlib import Path
from typing import Any

import numpy as np

from .files import read_csv, sha256_file

SCHEMA_VERSION = 1
SIGNAL_SHAPE = (12, 5000)
SAMPLING_RATE_HZ = 500
PURPOSES = ("ssl", "supervised")
LABEL_BUDGETS = ("1", "0.1")
IDENTIFIED_SOURCES = ("ptbxl", "mimic")


def signal_hash(signal: np.ndarray) -> str:
    """
    Identify a waveform by its little-endian float32 sample bytes.

    Parameters
    ----------
    signal : np.ndarray
        Waveform array.

    Returns
    -------
    str
        Hexadecimal SHA-256 digest.
    """
    return hashlib.sha256(np.ascontiguousarray(signal, dtype="<f4").tobytes()).hexdigest()


def checked_path(directory: Path, relative: str) -> Path:
    """
    Resolve a dataset-relative path that must stay inside ``directory``.

    Parameters
    ----------
    directory : Path
        Dataset directory.
    relative : str
        Path recorded in the dataset metadata.

    Returns
    -------
    Path
        Resolved absolute path.

    Raises
    ------
    ValueError
        If ``relative`` is absolute or escapes ``directory``.
    """
    path = (directory / relative).resolve()
    if Path(relative).is_absolute() or not path.is_relative_to(directory.resolve()):
        raise ValueError("Dataset path escapes its directory")
    return path


def validate_signal(signal: np.ndarray) -> None:
    """
    Check the canonical waveform contract.

    Parameters
    ----------
    signal : np.ndarray
        Waveform to check.

    Raises
    ------
    ValueError
        If the signal is not a finite float32 ``[12, 5000]`` array, or a lead
        is constant.
    """
    if signal.dtype != np.float32 or signal.shape != SIGNAL_SHAPE or not np.isfinite(signal).all():
        raise ValueError("Expected finite float32 [12,5000] waveform")
    if np.any(np.ptp(signal, axis=1) == 0):
        raise ValueError("Full constant lead")


def _validate_metadata(metadata: dict[str, Any]) -> None:
    complete = (metadata.get("complete") and metadata.get("schema_version") == SCHEMA_VERSION
                and metadata.get("shape_per_record") == list(SIGNAL_SHAPE)
                and metadata.get("sampling_rate_hz") == SAMPLING_RATE_HZ
                and metadata.get("units") == "mV")
    if not complete:
        raise ValueError("Invalid or incomplete canonical dataset")


def _validate_training_rows(rows: list[dict[str, str]], record_count: int) -> None:
    if len(rows) != record_count:
        raise ValueError("Record count mismatch")
    if any(row["split"] != "train" for row in rows):
        raise ValueError("SSL split or label leakage")
    if len({r["record_id"] for r in rows}) != len(rows):
        raise ValueError("Duplicate training record identity")


def _select_labels(directory: Path, rows: list[dict[str, str]],
                   label_budget: str) -> tuple[list[dict[str, str]], dict[str, int]]:
    """Return the labeled PTB training rows of one budget and their targets."""
    labels = read_csv(directory / f"labels_fraction{label_budget}.csv")
    targets = {r["record_id"]: int(r["target"]) for r in labels}
    if len(targets) != len(labels) or not set(targets.values()) <= {0, 1}:
        raise ValueError("Invalid endpoint labels")
    selected = [r for r in rows if r["record_id"] in targets]
    if len(selected) != len(labels) or any(r["source"] != "ptbxl" for r in selected):
        raise ValueError("Labels must belong to training PTB rows")
    patients = {r["record_id"]: r["patient_id"] for r in selected}
    if any(patients[r["record_id"]] != r["patient_id"] for r in labels):
        raise ValueError("Label patient identity mismatch")
    return selected, targets


class TrainingECGDataset:
    """
    PyTorch-compatible map dataset returning a signal and masked/available target.

    ``purpose='ssl'`` returns every train row with target=-1 and
    target_available=False. ``purpose='supervised'`` selects only PTB labels
    from budget '1' or '0.1'. Arrays are copied out of read-only mmap storage;
    no preprocessing is fitted.

    Parameters
    ----------
    directory : str | Path
        Published dataset holding ``metadata.json``, the manifests and shards.
    purpose : str
        ``"ssl"`` or ``"supervised"``.
    label_budget : str
        Label fraction for supervised use, ``"1"`` or ``"0.1"``.
    max_open_shards : int
        Memory-mapped shards kept open at once.

    Raises
    ------
    ValueError
        If the metadata, tables, rows or labels break the dataset contract.
    """

    def __init__(self, directory: str | Path, purpose: str = "ssl", label_budget: str = "1",
                 max_open_shards: int = 4) -> None:
        self.directory = Path(directory).resolve()
        self.metadata = json.loads((self.directory / "metadata.json").read_text())
        _validate_metadata(self.metadata)
        if purpose not in PURPOSES or label_budget not in LABEL_BUDGETS:
            raise ValueError("Unsupported purpose or label budget")
        if max_open_shards < 1:
            raise ValueError("max_open_shards must be positive")
        for name, expected in self.metadata["table_sha256"].items():
            if sha256_file(checked_path(self.directory, name)) != expected:
                raise ValueError(f"Dataset table hash mismatch: {name}")
        self.rows = read_csv(self.directory / "train_manifest.csv")
        _validate_training_rows(self.rows, self.metadata["record_count"])
        self.targets = {}
        if purpose == "supervised":
            self.rows, self.targets = _select_labels(self.directory, self.rows, label_budget)
        self.max_open_shards = max_open_shards
        self._arrays = OrderedDict()
        self._verified_shards = set()

    def __len__(self) -> int:
        """Return the number of served records."""
        return len(self.rows)

    def __getstate__(self) -> dict[str, Any]:
        """Drop open memory maps when pickled for loader workers."""
        state = self.__dict__.copy()
        state["_arrays"] = OrderedDict()
        return state

    def _shard(self, name: str) -> np.ndarray:
        """Open a shard once, verifying its hash on first use, with LRU eviction."""
        if name not in self._arrays:
            path = checked_path(self.directory, name)
            expected = self.metadata["shards"][name]
            if name not in self._verified_shards:
                if sha256_file(path) != expected["sha256"]:
                    raise ValueError(f"Shard hash mismatch: {name}")
                self._verified_shards.add(name)
            array = np.load(path, mmap_mode="r", allow_pickle=False)
            if list(array.shape) != expected["shape"] or array.dtype != np.float32:
                raise ValueError("Shard contract mismatch")
            self._arrays[name] = array
            while len(self._arrays) > self.max_open_shards:
                self._arrays.popitem(last=False)
        self._arrays.move_to_end(name)
        return self._arrays[name]

    def __getitem__(self, index: int) -> dict[str, Any]:
        """
        Return one verified record.

        Parameters
        ----------
        index : int
            Position in ``rows``.

        Returns
        -------
        dict[str, Any]
            ``signal``, ``target`` (-1 when unavailable), ``target_available``,
            ``record_id``, ``source`` and ``patient_id``.

        Raises
        ------
        ValueError
            If the shard or waveform fails its hash or contract check.
        """
        row = self.rows[index]
        signal = np.array(self._shard(row["shard"])[int(row["shard_index"])], copy=True)
        validate_signal(signal)
        if signal_hash(signal) != row["signal_sha256"]:
            raise ValueError("Waveform hash mismatch")
        available = row["record_id"] in self.targets
        return {"signal": signal, "target": self.targets.get(row["record_id"], -1),
                "target_available": available, "record_id": row["record_id"],
                "source": row["source"], "patient_id": row["patient_id"]}


def _identity_semantics_valid(row: dict[str, str]) -> bool:
    """Only PTB-XL and MIMIC rows carry patient identity; only PTB-XL has proxy labels."""
    known = row["source"] in IDENTIFIED_SOURCES
    if row["patient_identity_known"] != str(known).lower():
        return False
    if not known and row["patient_id"]:
        return False
    expected_scope = "ptbxl_proxy_available_separately" if row["source"] == "ptbxl" else "ssl_only"
    return row["label_scope"] == expected_scope


def verify_dataset(directory: str | Path) -> dict[str, Any]:
    """
    Independently reread every published array and both supervised selections.

    Parameters
    ----------
    directory : str | Path
        Published dataset directory.

    Returns
    -------
    dict[str, Any]
        Verified record and shard counts, supervised sizes per budget and
        confirmation that SSL targets are masked.

    Raises
    ------
    ValueError
        If any identity, pointer, label mask or shard row is inconsistent.
    """
    dataset = TrainingECGDataset(directory, max_open_shards=1)
    records, signals, pointers = set(), set(), set()
    for index, row in enumerate(dataset.rows):
        item = dataset[index]
        pointer = (row["shard"], row["shard_index"])
        if row["record_id"] in records or row["signal_sha256"] in signals or pointer in pointers:
            raise ValueError("Duplicate training identity or pointer")
        if item["target_available"] or item["target"] != -1 or row["split"] != "train":
            raise ValueError("SSL split or label leakage")
        if not _identity_semantics_valid(row):
            raise ValueError("Unsupported identity or label semantics")
        records.add(row["record_id"])
        signals.add(row["signal_sha256"])
        pointers.add(pointer)
    expected = sum(info["shape"][0] for info in dataset.metadata["shards"].values())
    if len(pointers) != expected:
        raise ValueError("Unreferenced shard rows")
    budgets = {budget: len(TrainingECGDataset(directory, "supervised", budget)) for budget in LABEL_BUDGETS}
    return {"verified_records": len(dataset), "verified_shards": len(dataset.metadata["shards"]),
            "supervised_records": budgets, "ssl_targets_masked": True}
