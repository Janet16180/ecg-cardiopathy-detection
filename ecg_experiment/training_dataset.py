"""Versioned canonical training data; labels are opt-in and SSL never exposes them."""

from __future__ import annotations

import json
from collections import OrderedDict
from pathlib import Path
from typing import Any

import numpy as np

from .files import read_csv, sha256_file
from .public_sources import signal_sha256
from .training_contracts import (
    identity_semantics_valid,
    select_labels,
    validate_metadata,
    validate_signal,
    validate_training_rows,
)

PURPOSES = ("ssl", "supervised")
LABEL_BUDGETS = ("1", "0.1")


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


def _validate_table_hashes(directory: Path, expected_hashes: dict[str, str]) -> None:
    """Verify every manifest and label table before using its contents."""
    for name, expected in expected_hashes.items():
        path = checked_path(directory, name)
        if sha256_file(path) != expected:
            raise ValueError(f"Dataset table hash mismatch: {name}")


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

    def __init__(
        self,
        directory: str | Path,
        purpose: str = "ssl",
        label_budget: str = "1",
        max_open_shards: int = 4,
    ) -> None:
        self.directory = Path(directory).resolve()
        self.metadata = json.loads((self.directory / "metadata.json").read_text())
        validate_metadata(self.metadata)

        if purpose not in PURPOSES or label_budget not in LABEL_BUDGETS:
            raise ValueError("Unsupported purpose or label budget")
        if max_open_shards < 1:
            raise ValueError("max_open_shards must be positive")

        _validate_table_hashes(self.directory, self.metadata["table_sha256"])
        self.rows = read_csv(self.directory / "train_manifest.csv")
        validate_training_rows(self.rows, self.metadata["record_count"])

        self.targets = {}
        if purpose == "supervised":
            labels = read_csv(self.directory / f"labels_fraction{label_budget}.csv")
            self.rows, self.targets = select_labels(self.rows, labels)

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

    def _open_shard(self, name: str) -> np.ndarray:
        """Verify a shard on first use and open its array without loading it into RAM."""
        path = checked_path(self.directory, name)
        expected = self.metadata["shards"][name]
        if name not in self._verified_shards:
            if sha256_file(path) != expected["sha256"]:
                raise ValueError(f"Shard hash mismatch: {name}")
            self._verified_shards.add(name)

        array = np.load(path, mmap_mode="r", allow_pickle=False)
        if list(array.shape) != expected["shape"] or array.dtype != np.float32:
            raise ValueError("Shard contract mismatch")
        return array

    def _shard(self, name: str) -> np.ndarray:
        """Reuse open arrays, evicting the least recently used shard when full."""
        if name not in self._arrays:
            self._arrays[name] = self._open_shard(name)
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
        shard = self._shard(row["shard"])
        signal = np.array(shard[int(row["shard_index"])], copy=True)
        validate_signal(signal)
        if signal_sha256(signal) != row["signal_sha256"]:
            raise ValueError("Waveform hash mismatch")

        available = row["record_id"] in self.targets
        return {
            "signal": signal,
            "target": self.targets.get(row["record_id"], -1),
            "target_available": available,
            "record_id": row["record_id"],
            "source": row["source"],
            "patient_id": row["patient_id"],
        }


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
    record_ids = set()
    signal_hashes = set()
    shard_positions = set()

    for index, row in enumerate(dataset.rows):
        item = dataset[index]
        pointer = (row["shard"], row["shard_index"])
        if (
            row["record_id"] in record_ids
            or row["signal_sha256"] in signal_hashes
            or pointer in shard_positions
        ):
            raise ValueError("Duplicate training identity or pointer")
        if item["target_available"] or item["target"] != -1 or row["split"] != "train":
            raise ValueError("SSL split or label leakage")
        if not identity_semantics_valid(row):
            raise ValueError("Unsupported identity or label semantics")
        record_ids.add(row["record_id"])
        signal_hashes.add(row["signal_sha256"])
        shard_positions.add(pointer)

    expected_records = sum(info["shape"][0] for info in dataset.metadata["shards"].values())
    if len(shard_positions) != expected_records:
        raise ValueError("Unreferenced shard rows")

    budgets = {
        budget: len(TrainingECGDataset(directory, "supervised", budget))
        for budget in LABEL_BUDGETS
    }
    return {
        "verified_records": len(dataset),
        "verified_shards": len(dataset.metadata["shards"]),
        "supervised_records": budgets,
        "ssl_targets_masked": True,
    }
