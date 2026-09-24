"""Decoding of raw twelve-lead WFDB records and labeled PTB-XL split manifests."""

# read_record cannot gain a docstring or inline noqa: its exact text is frozen.
# ruff: noqa: D103

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from .files import sha256_file

LEADS = ("I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6")
SPLITS = ("labeled_train", "validation", "test")
SAMPLE_RATE = 500


def manifest_rows(manifest_dir: Path, limit: int | None) -> tuple[list[dict[str, str]], dict[str, str]]:
    """
    Read the labeled PTB-XL split manifests in a fixed split order.

    Parameters
    ----------
    manifest_dir : Path
        Directory holding ``labeled_train.csv``, ``validation.csv`` and
        ``test.csv``.
    limit : int | None
        Keep only the first ``limit`` records, for smoke tests.

    Returns
    -------
    tuple[list[dict[str, str]], dict[str, str]]
        Rows with ``ecg_id``, ``filename_hr`` and ``split``, and the SHA-256
        of each manifest file keyed by file name.

    Raises
    ------
    ValueError
        If a manifest lacks required columns, an ECG ID repeats across
        manifests, or no records remain.
    """
    rows: list[dict[str, str]] = []
    hashes = {}
    seen = set()
    for split in SPLITS:
        path = manifest_dir / f"{split}.csv"
        hashes[path.name] = sha256_file(path)
        with path.open(newline="", encoding="utf-8") as stream:
            reader = csv.DictReader(stream)
            if not {"ecg_id", "filename_hr"}.issubset(reader.fieldnames or []):
                raise ValueError(f"{path} needs ecg_id and filename_hr columns")
            for row in reader:
                ecg_id = row["ecg_id"].strip()
                if ecg_id in seen:
                    raise ValueError(f"Duplicate ecg_id across manifests: {ecg_id}")
                seen.add(ecg_id)
                rows.append({"ecg_id": ecg_id, "filename_hr": row["filename_hr"].strip(), "split": split})
    if limit is not None:
        rows = rows[:limit]
    if not rows:
        raise ValueError("No records selected")
    return rows, hashes


# read_record is hashed by inspect.getsource() into cached leakage-reference
# identities (scripts/data/prepare_public_ecg.py). Keep its text byte-identical.
def read_record(raw_dir: Path, relative_name: str) -> np.ndarray:
    import wfdb

    path = (raw_dir / relative_name).resolve()
    if not path.is_relative_to(raw_dir.resolve()):
        raise ValueError(f"Waveform path escapes raw directory: {relative_name}")
    record = wfdb.rdrecord(str(path))
    if record.fs != SAMPLE_RATE:
        raise ValueError(f"Expected 500 Hz, got {record.fs} at {path}")
    names = tuple(record.sig_name)
    upper_names = tuple(name.upper() for name in names)
    upper_leads = tuple(name.upper() for name in LEADS)
    if set(upper_names) != set(upper_leads) or len(names) != 12:
        raise ValueError(f"Unexpected lead names at {path}: {names}")
    signal = np.asarray(record.p_signal, dtype=np.float32)
    if signal.shape[0] != 5000:
        raise ValueError(f"Expected a 10-second ECG at {path}; got {signal.shape}")
    signal = signal[:, [upper_names.index(name) for name in upper_leads]].T
    if not np.isfinite(signal).all():
        raise ValueError(f"Nonfinite waveform at {path}")
    return signal
