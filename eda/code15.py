"""Load CODE-15% metadata and analyze all 18 waveform parts from the export archive.

The 18 original ZIPs were replaced by one verified ``tar.xz`` holding the HDF5
files (see ``outputs/data_export/README.md``). Fully extracted it needs about
70 GB, so ``build_all_parts`` streams it, extracting one part at a time, caching
its statistics and deleting the extracted file before the next.
"""

import re
import subprocess
import tarfile
from collections.abc import Iterator
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

from eda.ptbxl import OUTPUT_DIR, ROOT
from eda.signals import compute_features, record_summary

RAW_DIR = ROOT / "data/raw/code-15pct/zenodo-4916206"
PREPARED_DIR = ROOT / "data/processed/code15_quality"
EXPORT_ARCHIVE = ROOT / "outputs/data_export/code_15pct_waveforms_2026-09-25.tar.xz"
PART0_NATIVE = PREPARED_DIR / "part0_native/exams_part0_native.hdf5"
CACHE_DIR = OUTPUT_DIR / "features" / "code15"
EXTRACT_DIR = OUTPUT_DIR / "code15_extract"
SAMPLING_RATE = 400
LABELS = ["1dAVb", "RBBB", "LBBB", "SB", "ST", "AF"]
PART_PATTERN = re.compile(r"exams_part(\d+)\.hdf5$")


def load_exams() -> pd.DataFrame:
    """
    Load the official exam table.

    Returns
    -------
    pd.DataFrame
        One row per exam indexed by ``exam_id``.
    """
    return pd.read_csv(RAW_DIR / "exams.csv").set_index("exam_id")


def zero_padding(tracing: np.ndarray) -> tuple[int, int]:
    """
    Count all-lead zero rows at the start and end of one tracing.

    Parameters
    ----------
    tracing : np.ndarray
        Array of shape ``(4096, 12)``.

    Returns
    -------
    tuple[int, int]
        Leading and trailing zero rows. An all-zero tracing returns its full
        length for both.
    """
    nonzero = np.flatnonzero(np.any(tracing != 0, axis=1))
    if len(nonzero) == 0:
        return len(tracing), len(tracing)
    return int(nonzero[0]), int(len(tracing) - 1 - nonzero[-1])


def part_padding(path: Path, part: int) -> pd.DataFrame:
    """
    Zero padding of every tracing in one HDF5 part.

    Parameters
    ----------
    path : Path
        Extracted ``exams_partN.hdf5``.
    part : int
        Part number.

    Returns
    -------
    pd.DataFrame
        One row per stored tracing with ``exam_id``, ``part``,
        ``storage_index``, ``left``, ``right`` and ``active_samples``.
    """
    rows = []
    with h5py.File(path, "r") as handle:
        ids, tracings = handle["exam_id"][:], handle["tracings"]
        length = tracings.shape[1]
        for start in range(0, len(ids), 1000):
            for offset, tracing in enumerate(tracings[start:start + 1000]):
                left, right = zero_padding(tracing)
                rows.append({"exam_id": int(ids[start + offset]), "part": part,
                             "storage_index": start + offset, "left": left, "right": right})
    table = pd.DataFrame(rows)
    table["active_samples"] = (length - table["left"] - table["right"]).clip(lower=0)
    return table


def read_active(item: str) -> tuple[np.ndarray, int]:
    """
    Read one tracing with its zero padding removed.

    Parameters
    ----------
    item : str
        ``"hdf5 path|storage index|left|right"``.

    Returns
    -------
    tuple[np.ndarray, int]
        Active part of the tracing and the 400 Hz sampling rate.
    """
    path, index, left, right = item.split("|")
    with h5py.File(path, "r") as handle:
        tracing = handle["tracings"][int(index)]
    return tracing[int(left):len(tracing) - int(right)], SAMPLING_RATE


def summarize_part(path: Path, part: int, min_active: int = 2000) -> None:
    """
    Cache padding and signal features of one extracted part.

    Parameters
    ----------
    path : Path
        Extracted ``exams_partN.hdf5``.
    part : int
        Part number.
    min_active : int
        Tracings with fewer non-padding samples (5 s at 400 Hz) get no signal features.
    """
    padding = part_padding(path, part)
    usable = padding[padding["active_samples"] >= min_active]
    items = [(f"{row.part}:{row.storage_index}", f"{path}|{row.storage_index}|{row.left}|{row.right}")
             for row in usable.itertuples()]
    compute_features(items, read_active, CACHE_DIR / f"part{part}_features.parquet")
    padding.to_parquet(CACHE_DIR / f"part{part}_padding.parquet")


def extracted_parts() -> Iterator[tuple[int, Path]]:
    """
    Stream the export archive and yield each HDF5 part as a temporary file.

    The archive is decompressed by the ``xz`` command-line tool, which is much
    faster than Python's ``lzma``. Each file is deleted once the caller moves on.

    Yields
    ------
    tuple[int, Path]
        Part number and path of the extracted file.
    """
    EXTRACT_DIR.mkdir(parents=True, exist_ok=True)
    process = subprocess.Popen(["xz", "-dc", "-T0", str(EXPORT_ARCHIVE)], stdout=subprocess.PIPE)
    with tarfile.open(fileobj=process.stdout, mode="r|") as archive:
        for member in archive:
            match = PART_PATTERN.search(member.name)
            if not match:
                continue
            target = EXTRACT_DIR / Path(member.name).name
            with archive.extractfile(member) as source, target.open("wb") as destination:
                while chunk := source.read(64 * 1024 * 1024):
                    destination.write(chunk)
            yield int(match.group(1)), target
            target.unlink()
    process.wait()


def build_all_parts() -> None:
    """Stream the archive once and cache every part that is not cached yet."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    for part, path in extracted_parts():
        if (CACHE_DIR / f"part{part}_padding.parquet").exists():
            continue
        summarize_part(path, part)
        print(f"CODE-15 part {part} cached", flush=True)


def padding_table() -> pd.DataFrame:
    """
    Zero padding of all stored tracings from the cached parts.

    Returns
    -------
    pd.DataFrame
        One row per stored tracing, including any rows whose exam ID is not
        in ``exams.csv``.

    Raises
    ------
    FileNotFoundError
        If ``build_all_parts`` has not been run.
    """
    paths = sorted(CACHE_DIR.glob("part*_padding.parquet"))
    if not paths:
        raise FileNotFoundError("No CODE-15 caches; run eda.code15.build_all_parts() first")
    return pd.concat([pd.read_parquet(path) for path in paths], ignore_index=True)


def code15_summary() -> pd.DataFrame:
    """
    Per-tracing signal summary of all cached parts, with padding columns.

    Returns
    -------
    pd.DataFrame
        ``record_summary`` columns plus padding, indexed by ``"part:storage_index"``.
    """
    features = pd.concat([pd.read_parquet(path) for path in sorted(CACHE_DIR.glob("part*_features.parquet"))])
    summary = record_summary(features)
    padding = padding_table()
    padding.index = padding["part"].astype(str) + ":" + padding["storage_index"].astype(str)
    return summary.join(padding)


def lead_features() -> pd.DataFrame:
    """
    Per-lead features of all cached parts.

    Returns
    -------
    pd.DataFrame
        One row per tracing and lead, with ``exam_id`` added.
    """
    features = pd.concat([pd.read_parquet(path) for path in sorted(CACHE_DIR.glob("part*_features.parquet"))])
    padding = padding_table()
    ids = dict(zip(padding["part"].astype(str) + ":" + padding["storage_index"].astype(str),
                   padding["exam_id"], strict=True))
    features["exam_id"] = features["record_id"].map(ids)
    return features


def read_native_tracing(exam_id: int) -> np.ndarray:
    """
    Read one part-0 tracing from the project's native copy, for plotting.

    Parameters
    ----------
    exam_id : int
        Exam identifier present in part 0.

    Returns
    -------
    np.ndarray
        Array of shape ``(4096, 12)`` in the release's lead order (DI, DII,
        DIII, AVR, AVL, AVF, V1-V6), which is the standard order.
    """
    with h5py.File(PART0_NATIVE, "r") as handle:
        index = int(np.flatnonzero(handle["exam_id"][:] == exam_id)[0])
        return handle["tracings"][index]
