"""Checksum-verified access to the CODE-15% release archives and exam metadata."""

from __future__ import annotations

import csv
import hashlib
import json
import shutil
import tempfile
import zipfile
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import h5py
import numpy as np

TRACE_SAMPLES = 4096
LEAD_COUNT = 12
TRACE_SHAPE = (TRACE_SAMPLES, LEAD_COUNT)
DIAGNOSIS_FIELDS = ("1dAVb", "RBBB", "LBBB", "SB", "ST", "AF")
METADATA_FIELDS = {"exam_id", "patient_id", "trace_file", "age", "is_male",
                   "nn_predicted_age", "normal_ecg", *DIAGNOSIS_FIELDS}
EDGE_ASYMMETRY_SAMPLES = 2
NEAR_FLAT_STD = 0.01
COPY_CHUNK_BYTES = 4 * 1024 * 1024


def md5_file(path: Path) -> str:
    """
    Compute the MD5 digest that Zenodo publishes for each release file.

    Parameters
    ----------
    path : Path
        File to hash.

    Returns
    -------
    str
        Hexadecimal digest.
    """
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "md5").hexdigest()


def verified_files(receipt_path: Path) -> dict[str, dict[str, Any]]:
    """
    Read the acquisition receipt's verified files keyed by name.

    Parameters
    ----------
    receipt_path : Path
        ``code_15pct.json`` acquisition receipt.

    Returns
    -------
    dict[str, dict[str, Any]]
        Receipt entries with ``size`` and ``checksum`` keyed by file name.
    """
    receipt = json.loads(receipt_path.read_text())
    return {item["name"]: item for item in receipt["verified_files"]}


def verify_file(path: Path, expected: dict[str, Any]) -> None:
    """
    Check a release file against its official size and MD5 checksum.

    Parameters
    ----------
    path : Path
        Local release file.
    expected : dict[str, Any]
        Receipt entry with ``size`` and ``checksum`` (``"md5:<hex>"``).

    Raises
    ------
    ValueError
        If the file is missing, has the wrong size, or its checksum differs.
    """
    if not path.is_file() or path.stat().st_size != expected["size"]:
        raise ValueError(f"Missing or wrong-size verified input: {path}")
    algorithm, value = expected["checksum"].split(":", 1)
    if algorithm != "md5" or md5_file(path) != value:
        raise ValueError(f"Official checksum mismatch: {path}")


def metadata_rows(path: Path) -> dict[int, dict[str, str]]:
    """
    Read ``exams.csv`` keyed by integer exam ID.

    Parameters
    ----------
    path : Path
        Official ``exams.csv``.

    Returns
    -------
    dict[int, dict[str, str]]
        Raw CSV rows keyed by exam ID.

    Raises
    ------
    ValueError
        If required columns are missing or an exam ID repeats.
    """
    rows = {}
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if not METADATA_FIELDS.issubset(reader.fieldnames or []):
            raise ValueError("exams.csv lacks required columns")
        for row in reader:
            exam_id = int(row["exam_id"])
            if exam_id in rows:
                raise ValueError(f"Duplicate exam_id in exams.csv: {exam_id}")
            rows[exam_id] = row
    return rows


def inspect_trace(trace: np.ndarray) -> tuple[str | None, int, int, int, str]:
    """
    Screen one native trace without changing it.

    Parameters
    ----------
    trace : np.ndarray
        Native ``[4096, 12]`` trace.

    Returns
    -------
    tuple[str | None, int, int, int, str]
        Exclusion reason (``None`` when accepted), exact-zero samples at the
        left and right edges, active span, and semicolon-separated review flags.
    """
    if trace.shape != TRACE_SHAPE:
        return "shape_contract", 0, 0, 0, ""
    if not np.isfinite(trace).all():
        return "nonfinite_signal", 0, 0, 0, ""
    active = np.flatnonzero(np.any(trace != 0, axis=1))
    if not len(active):
        return "all_zero_signal", TRACE_SAMPLES, TRACE_SAMPLES, 0, ""
    if np.any(np.ptp(trace, axis=0) == 0):
        return "constant_lead", 0, 0, 0, ""
    left, right = int(active[0]), int(TRACE_SAMPLES - 1 - active[-1])
    flags = []
    if abs(left - right) > EDGE_ASYMMETRY_SAMPLES:
        flags.append("asymmetric_zero_edges_review")
    if np.any(np.std(trace, axis=0) < NEAR_FLAT_STD):
        flags.append("near_flat_native_units_review")
    return None, left, right, int(active[-1] - active[0] + 1), ";".join(flags)


@contextmanager
def extracted_hdf5(archive_path: Path, member_name: str) -> Iterator[h5py.File]:
    """
    Open the single HDF5 member of a release ZIP through a temporary copy.

    The archive itself is only read; the extracted copy is deleted on exit.

    Parameters
    ----------
    archive_path : Path
        Release ZIP such as ``exams_part0.zip``.
    member_name : str
        The one member the archive must contain.

    Yields
    ------
    h5py.File
        The extracted file, open read-only.

    Raises
    ------
    ValueError
        If the archive holds anything other than exactly ``member_name``.
    """
    with zipfile.ZipFile(archive_path) as archive:
        members = archive.infolist()
        if len(members) != 1 or members[0].filename != member_name:
            raise ValueError(f"Unexpected ZIP contents: {archive_path.name}")
        with tempfile.TemporaryDirectory(prefix="code15_") as directory:
            temporary = Path(directory) / member_name
            with archive.open(members[0]) as source, temporary.open("wb") as target:
                shutil.copyfileobj(source, target, length=COPY_CHUNK_BYTES)
            with h5py.File(temporary, "r") as handle:
                yield handle
