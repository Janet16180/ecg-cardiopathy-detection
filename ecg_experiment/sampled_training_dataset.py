"""Read a sampled ECG cohort without copying its underlying waveforms."""

from __future__ import annotations

import json
from collections import OrderedDict
from pathlib import Path
from typing import Any

import numpy as np

from .files import read_csv, sha256_file
from .public_sources import signal_sha256
from .training_contracts import select_labels, validate_signal
from .waveforms import read_record

BACKENDS = ("shard", "mimic_wfdb")


def _checked_rows(directory: Path, metadata: dict[str, Any]) -> list[dict[str, str]]:
    """Verify the published tables and basic row identities."""
    for name, expected in metadata["table_sha256"].items():
        if sha256_file(directory / name) != expected:
            raise ValueError(f"Sampled cohort table changed: {name}")
    rows = read_csv(directory / "train_manifest.csv")
    if len(rows) != metadata["record_count"]:
        raise ValueError("Sampled cohort count mismatch")
    if len({row["record_id"] for row in rows}) != len(rows):
        raise ValueError("Duplicate sampled record ID")
    if any(row["split"] != "train" or row["backend"] not in BACKENDS for row in rows):
        raise ValueError("Invalid sampled row")
    return rows


class SampledTrainingECGDataset:
    """Serve a fixed sampled cohort for SSL or PTB-only supervised training.

    The dataset directory contains only manifests. Shards stay in the verified
    source datasets; additional MIMIC records stay in their original WFDB files.
    SSL always masks targets, including PTB rows with available labels.

    Parameters
    ----------
    directory : str | Path
        Published sampled-cohort directory.
    purpose : str
        ``"ssl"`` or ``"supervised"``.
    label_budget : str
        ``"1"`` for all eligible PTB labels or ``"0.1"`` for the fixed subset.
    max_open_shards : int
        Maximum live memory-mapped source shards per loader process.
    """

    def __init__(
        self,
        directory: str | Path,
        purpose: str = "ssl",
        label_budget: str = "1",
        max_open_shards: int = 4,
    ) -> None:
        if purpose not in ("ssl", "supervised") or label_budget not in ("1", "0.1"):
            raise ValueError("Unsupported purpose or label budget")
        if max_open_shards < 1:
            raise ValueError("max_open_shards must be positive")

        self.directory = Path(directory).resolve()
        self.root = Path(__file__).resolve().parents[1]
        self.metadata = json.loads((self.directory / "metadata.json").read_text())
        if not self.metadata.get("complete") or self.metadata.get("schema_version") != 1:
            raise ValueError("Incomplete or unsupported sampled cohort")
        if self.metadata.get("shape_per_record") != [12, 5000]:
            raise ValueError("Unexpected waveform shape")

        self.rows = _checked_rows(self.directory, self.metadata)

        self.targets: dict[str, int] = {}
        if purpose == "supervised":
            labels = read_csv(self.directory / f"labels_fraction{label_budget}.csv")
            self.rows, self.targets = select_labels(self.rows, labels)
        self.max_open_shards = max_open_shards
        self._arrays: OrderedDict[str, np.ndarray] = OrderedDict()
        self._verified_shards: set[str] = set()

    def __len__(self) -> int:
        """Return the number of rows served for the requested purpose."""
        return len(self.rows)

    def __getstate__(self) -> dict[str, Any]:
        """Drop open memory maps when copied into a data-loader worker."""
        state = self.__dict__.copy()
        state["_arrays"] = OrderedDict()
        state["_verified_shards"] = set()
        return state

    def _safe_path(self, relative: str) -> Path:
        """Resolve a source pointer inside the repository tree."""
        path = (self.root / relative).resolve()
        if Path(relative).is_absolute() or not path.is_relative_to(self.root):
            raise ValueError("Source pointer escapes repository")
        return path

    def _shard(self, relative: str) -> np.ndarray:
        """Verify and memory-map a canonical source shard on first access."""
        if relative not in self._arrays:
            path = self._safe_path(relative)
            expected = self.metadata["source_shards"][relative]
            if relative not in self._verified_shards:
                if sha256_file(path) != expected["sha256"]:
                    raise ValueError(f"Source shard changed: {relative}")
                self._verified_shards.add(relative)
            array = np.load(path, mmap_mode="r", allow_pickle=False)
            if list(array.shape) != expected["shape"] or array.dtype != np.float32:
                raise ValueError("Source shard contract mismatch")
            self._arrays[relative] = array
            if len(self._arrays) > self.max_open_shards:
                self._arrays.popitem(last=False)
        self._arrays.move_to_end(relative)
        return self._arrays[relative]

    def __getitem__(self, index: int) -> dict[str, Any]:
        """Return a copied, hash-checked ECG and an opt-in target."""
        row = self.rows[index]
        if row["backend"] == "shard":
            array = self._shard(row["path"])
            signal = np.array(array[int(row["index"])], copy=True)
        else:
            root = self._safe_path("data/raw/mimic-iv-ecg/1.0")
            path = self._safe_path(row["path"])
            signal = read_record(root, str(path.relative_to(root)))
        # The accepted MIMIC audit requires finite 12-lead signals but does
        # not exclude a constant lead. Preserve that published source policy.
        validate_signal(signal, require_variable_leads=row["source"] != "mimic")
        if signal_sha256(signal) != row["signal_sha256"]:
            raise ValueError(f"Waveform changed: {row['record_id']}")

        available = row["record_id"] in self.targets
        return {
            "signal": signal,
            "target": self.targets.get(row["record_id"], -1),
            "target_available": available,
            "record_id": row["record_id"],
            "source": row["source"],
            "patient_id": row["patient_id"],
        }
