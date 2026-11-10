"""Local waveform cache of the MLC-01 records for MLC-02 Stage 1.

Every included manifest record is stored as its first 10 s, twelve leads in the order I, II, III, aVR, aVL,
aVF, V1-V6, 500 Hz, mV, float32, in one ``[records, 12, 5000]`` array per role and source
(``docs/mlc02-condition-models-protocol-2026-11-10.md``, Stage 1). Records are read with the MLC-01 reader and
checked against the manifest's ``sha256_10s`` (int32 microvolts, ``mlc01_dataset.microvolts``) both before and
after the float32 cast, and again from the written files. Only identity columns of the manifest are read;
no label column is opened. A sample WFDB reads as missing stays NaN in the cache.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import wfdb

from ecg_experiment import ROOT, mlc01_dataset
from ecg_experiment.eda.signals import LEADS, canonical_order
from ecg_experiment.mlc01_dataset import array_sha256, challenge_key, microvolts, require_open
from ecg_experiment.paths import from_stored
from ecg_experiment.public_sources import signal_sha256

MANIFEST = ROOT / "outputs/mlc01_condition_dataset_20261110_v3/manifest.csv"
OTHER_MANIFESTS = (
    ROOT / "outputs/mlc01_condition_dataset_20261110/manifest.csv",
    ROOT / "outputs/mlc01_condition_dataset_20261110_v2/manifest.csv",
)
CACHE = Path("/var/tmp/janetrivera/mlc02/waveforms")
IDENTITY_COLUMNS = [
    "record_key",
    "source",
    "family",
    "record",
    "role",
    "raw_path",
    "included",
    "fs",
    "n_samples",
    "sha256_10s",
]
COMPARED_COLUMNS = ["record_key", "source", "record", "role", "raw_path", "sha256_10s"]
SOURCES = ("ptbxl", "ningbo", "chapman_shaoxing", "georgia", "cpsc_2018", "cpsc_2018_extra", "sph")
INCLUDED_RECORDS = 99167
SAMPLES = 5000
RATE = 500
MISSING_DIGITAL = -32768
RAIL_MV = -32.768
GUARD: dict[str, frozenset[str]] = {"closed": frozenset(), "allowed": frozenset()}


def read_rows(path: Path) -> pd.DataFrame:
    """
    Identity columns of every manifest row, in manifest order.

    Parameters
    ----------
    path : Path
        MLC-01 ``manifest.csv``.

    Returns
    -------
    pd.DataFrame
        ``IDENTITY_COLUMNS`` (``included`` as bool) plus ``manifest_row`` (0-based row of the file).

    Raises
    ------
    ValueError
        If ``included`` holds a value other than True or False.
    """
    frame = pd.read_csv(
        path,
        usecols=IDENTITY_COLUMNS,
        dtype={"record": str, "raw_path": str, "included": str},
        keep_default_na=False,
    )
    if not set(frame["included"]) <= {"True", "False"}:
        raise ValueError(f"Unexpected included values in {path}")
    frame = frame[IDENTITY_COLUMNS].assign(included=frame["included"] == "True")
    return frame.assign(manifest_row=np.arange(len(frame)))


def read_identity(path: Path = MANIFEST, expected: int | None = INCLUDED_RECORDS) -> pd.DataFrame:
    """
    Identity columns of the included manifest records, in manifest order.

    Parameters
    ----------
    path : Path
        MLC-01 ``manifest.csv``.
    expected : int | None
        Required number of included records; None skips the check.

    Returns
    -------
    pd.DataFrame
        ``IDENTITY_COLUMNS`` plus ``manifest_row`` for included records only.

    Raises
    ------
    ValueError
        If a record key repeats, a source or role is unknown, or the count differs from the MLC-01 build.
    """
    frame = read_rows(path)
    included = frame[frame["included"]].reset_index(drop=True)
    if included["record_key"].duplicated().any():
        raise ValueError("Duplicate record key among included records")
    if not set(included["source"]) <= set(SOURCES) or not set(included["role"]) <= set(mlc01_dataset.ROLES):
        raise ValueError("Unknown source or role in the manifest")
    if (included["fs"].astype(int) != RATE).any() or (included["n_samples"].astype(int) < SAMPLES).any():
        raise ValueError("An included record is not at least 10 s at 500 Hz")
    if expected is not None and len(included) != expected:
        raise ValueError(f"Expected {expected} included records, got {len(included)}")
    return included


def compare_manifests(used: pd.DataFrame, other: pd.DataFrame) -> dict[str, object]:
    """
    Compare two manifests row by row on identity, role, path and waveform hash, and on inclusion.

    Parameters
    ----------
    used, other : pd.DataFrame
        Outputs of ``read_rows``.

    Returns
    -------
    dict[str, object]
        ``identity_equal`` (``COMPARED_COLUMNS`` equal on every row), ``included_used`` and
        ``included_other`` (counts), and the records included in only one of them, by source and role.
    """
    equal = len(used) == len(other) and used[COMPARED_COLUMNS].reset_index(drop=True).equals(
        other[COMPARED_COLUMNS].reset_index(drop=True)
    )
    record = {
        "identity_equal": bool(equal),
        "included_used": int(used["included"].sum()),
        "included_other": int(other["included"].sum()),
    }
    if equal:
        for name, mask in (
            ("only_used", used["included"].to_numpy() & ~other["included"].to_numpy()),
            ("only_other", ~used["included"].to_numpy() & other["included"].to_numpy()),
        ):
            counts = used[mask].groupby(["source", "role"]).size()
            record[name] = {f"{source}/{role}": int(size) for (source, role), size in counts.items()}
    return record


def block_name(role: str, source: str) -> str:
    """
    File stem of one role and source array.

    Parameters
    ----------
    role, source : str
        Manifest role and source.

    Returns
    -------
    str
        ``role__source``.
    """
    return f"{role}__{source}"


def cache_index(identity: pd.DataFrame) -> pd.DataFrame:
    """
    Position of every record in its role and source array, in manifest order.

    Parameters
    ----------
    identity : pd.DataFrame
        Output of ``read_identity``.

    Returns
    -------
    pd.DataFrame
        ``position`` (global order, equal to the manifest order of included records), ``manifest_row``,
        ``record_key``, ``source``, ``role``, ``block`` and ``row`` (row inside the block array),
        ``raw_path`` and ``sha256_10s``.
    """
    index = identity.sort_values("manifest_row").reset_index(drop=True)
    block = [block_name(role, source) for role, source in zip(index["role"], index["source"], strict=True)]
    index = index.assign(block=block)
    index["row"] = index.groupby("block").cumcount()
    index.insert(0, "position", np.arange(len(index)))
    return index[
        ["position", "manifest_row", "record_key", "source", "role", "block", "row", "raw_path", "sha256_10s"]
    ]


def block_sizes(index: pd.DataFrame) -> dict[str, int]:
    """
    Count the records of every block array.

    Parameters
    ----------
    index : pd.DataFrame
        Output of ``cache_index``.

    Returns
    -------
    dict[str, int]
        Block name to record count, in first-appearance order.
    """
    return {name: int(size) for name, size in index.groupby("block", sort=False).size().items()}


def closed_keys() -> frozenset[str]:
    """
    Paths of the closed Challenge groups, from the identity columns of the frozen split table.

    Returns
    -------
    frozenset[str]
        ``challenge_key`` of every test-group record.
    """
    return mlc01_dataset.closed_challenge_keys(mlc01_dataset.read_split_table())


def init_worker(closed: frozenset[str], allowed: frozenset[str]) -> None:
    """
    Give a reader process the closed Challenge paths and the manifest's included paths.

    Parameters
    ----------
    closed : frozenset[str]
        Output of ``closed_keys``.
    allowed : frozenset[str]
        ``raw_path`` of every included manifest record.
    """
    GUARD["closed"] = closed
    GUARD["allowed"] = allowed


def guard_path(raw_path: str, source: str) -> Path:
    """
    Absolute path of a manifest record after the closed-data checks.

    Parameters
    ----------
    raw_path : str
        Manifest ``raw_path`` (repository-relative).
    source : str
        Manifest source.

    Returns
    -------
    Path
        Path to open.

    Raises
    ------
    PermissionError
        If the guard is not set, the path is not an included manifest path, or it is a closed Challenge
        record.
    """
    if not GUARD["closed"] or not GUARD["allowed"]:
        raise PermissionError("The closed-data guard is not set")
    if raw_path not in GUARD["allowed"]:
        raise PermissionError(f"Path outside the included manifest records: {raw_path}")
    path = from_stored(raw_path)
    if source in mlc01_dataset.RELEASES:
        release = mlc01_dataset.RELEASES[source]
        require_open(challenge_key(source, path.relative_to(release).as_posix()), GUARD["closed"])
    return path


def read_window(path: Path, source: str) -> np.ndarray:
    """
    First 10 s of one record in mV and canonical lead order, read as MLC-01 read it.

    Parameters
    ----------
    path : Path
        WFDB record stem or SPH HDF5 file.
    source : str
        Manifest source.

    Returns
    -------
    np.ndarray
        Float64 ``(5000, 12)`` array; samples WFDB reads as missing are NaN.

    Raises
    ------
    ValueError
        If the record is not 500 Hz or shorter than 10 s.
    """
    job = {"format": "h5" if source == "sph" else "wfdb", "stem": str(path)}
    signal, fs, names = mlc01_dataset.read_signal(job)
    window = canonical_order(signal, names)[:SAMPLES]
    if fs != RATE or window.shape != (SAMPLES, len(LEADS)):
        raise ValueError(f"Not 10 s of twelve leads at 500 Hz: {path}")
    return window


def check_missing_rail(path: Path, window: np.ndarray) -> int:
    """
    Require that every missing sample is the format-16 value -32768 at gain 1000 and baseline 0.

    Parameters
    ----------
    path : Path
        WFDB record stem.
    window : np.ndarray
        Output of ``read_window``.

    Returns
    -------
    int
        Number of missing samples in the window.

    Raises
    ------
    ValueError
        If a missing sample has another stored value or its lead another scaling.
    """
    missing = np.isnan(window)
    if not missing.any():
        return 0
    record = wfdb.rdrecord(str(path), physical=False, sampto=SAMPLES)
    upper = [name.upper() for name in record.sig_name]
    order = [upper.index(lead.upper()) for lead in LEADS]
    digital = record.d_signal[:, order]
    gain = np.asarray(record.adc_gain, dtype=float)[order]
    baseline = np.asarray(record.baseline, dtype=float)[order]
    leads = missing.any(axis=0)
    if (
        (digital[missing] != MISSING_DIGITAL).any()
        or (gain[leads] != 1000).any()
        or (baseline[leads] != 0).any()
    ):
        raise ValueError(f"Missing samples that are not the negative rail: {path}")
    return int(missing.sum())


def cache_record(raw_path: str, source: str, expected: str) -> tuple[np.ndarray, dict[str, object]]:
    """
    Read one record, cast it to float32 and check both forms against the manifest hash.

    Parameters
    ----------
    raw_path : str
        Manifest ``raw_path``.
    source : str
        Manifest source.
    expected : str
        Manifest ``sha256_10s``.

    Returns
    -------
    tuple[np.ndarray, dict[str, object]]
        Float32 ``[12, 5000]`` window and its checks: ``sha256_read_ok`` (float64 read),
        ``sha256_float32_ok`` (after the cast), ``missing_samples_10s``, ``flat_leads_10s`` and
        ``sha256_f32`` (``public_sources.signal_sha256`` of the float32 window, the convention of earlier
        feature caches).
    """
    path = guard_path(raw_path, source)
    window = read_window(path, source)
    missing = check_missing_rail(path, window) if source != "sph" else int(np.isnan(window).sum())
    stored = np.ascontiguousarray(window.T, dtype=np.float32)
    finite = np.isfinite(window)
    high = np.where(finite, window, -np.inf).max(axis=0)
    low = np.where(finite, window, np.inf).min(axis=0)
    checks = {
        "sha256_read_ok": array_sha256(microvolts(window)) == expected,
        "sha256_float32_ok": array_sha256(microvolts(stored.T)) == expected,
        "missing_samples_10s": missing,
        "flat_leads_10s": int((~(high > low)).sum()),
        "sha256_f32": signal_sha256(stored),
    }
    return stored, checks


def open_block(directory: Path, name: str, records: int, mode: str) -> np.memmap:
    """
    Open one block array of the cache.

    Parameters
    ----------
    directory : Path
        Cache directory.
    name : str
        Block name.
    records : int
        Expected number of records.
    mode : str
        ``"r"``, ``"r+"`` or ``"w+"`` (creates the file).

    Returns
    -------
    np.memmap
        Float32 ``[records, 12, 5000]`` array.

    Raises
    ------
    ValueError
        If an existing file has another shape or dtype.
    """
    path = directory / f"{name}.npy"
    if mode == "w+":
        return np.lib.format.open_memmap(
            path, mode="w+", dtype=np.float32, shape=(records, len(LEADS), SAMPLES)
        )
    array = np.load(path, mmap_mode=mode)
    if array.shape != (records, len(LEADS), SAMPLES) or array.dtype != np.float32:
        raise ValueError(f"Unexpected cache block {path}: {array.shape} {array.dtype}")
    return array


def write_chunk(task: dict[str, object]) -> list[dict[str, object]]:
    """
    Read and store the records of one chunk of a block.

    Parameters
    ----------
    task : dict[str, object]
        ``directory``, ``block``, ``records`` (block size) and ``jobs`` (dicts with ``position``, ``row``,
        ``raw_path``, ``source`` and ``sha256_10s``).

    Returns
    -------
    list[dict[str, object]]
        ``position`` and the checks of ``cache_record`` per job.
    """
    array = open_block(Path(task["directory"]), task["block"], task["records"], "r+")
    results = []
    for job in task["jobs"]:
        stored, checks = cache_record(job["raw_path"], job["source"], job["sha256_10s"])
        array[job["row"]] = stored
        results.append({"position": job["position"], **checks})
    array.flush()
    del array
    return results


def chunk_tasks(index: pd.DataFrame, directory: Path, size: int) -> list[dict[str, object]]:
    """
    Split every block into chunks of consecutive rows.

    Parameters
    ----------
    index : pd.DataFrame
        Output of ``cache_index`` (or a subset with consistent rows).
    directory : Path
        Cache directory.
    size : int
        Records per chunk.

    Returns
    -------
    list[dict[str, object]]
        Tasks for ``write_chunk``, each with a ``name`` unique within the cache.
    """
    sizes = block_sizes(index)
    tasks = []
    columns = ["position", "row", "raw_path", "source", "sha256_10s"]
    for name, rows in index.groupby("block", sort=False):
        jobs = rows[columns].to_dict("records")
        for start in range(0, len(jobs), size):
            part = jobs[start : start + size]
            tasks.append(
                {
                    "directory": str(directory),
                    "block": name,
                    "records": sizes[name],
                    "jobs": part,
                    "name": f"{name}__{part[0]['row']:06d}",
                }
            )
    return tasks


def verify_chunk(task: dict[str, object]) -> list[dict[str, object]]:
    """
    Recompute the manifest hash of every record of a chunk from the written cache file.

    Parameters
    ----------
    task : dict[str, object]
        A task of ``chunk_tasks``.

    Returns
    -------
    list[dict[str, object]]
        ``position``, ``sha256_cache_ok`` and ``sha256_f32`` per record.
    """
    array = open_block(Path(task["directory"]), task["block"], task["records"], "r")
    results = []
    for job in task["jobs"]:
        window = np.asarray(array[job["row"]])
        results.append(
            {
                "position": job["position"],
                "sha256_cache_ok": array_sha256(microvolts(window.T)) == job["sha256_10s"],
                "sha256_f32": signal_sha256(window),
            }
        )
    del array
    return results


def load_cache(directory: Path = CACHE) -> tuple[pd.DataFrame, dict[str, np.memmap]]:
    """
    Open the cache index and every block array read-only.

    Parameters
    ----------
    directory : Path
        Cache directory holding ``index.csv`` and the block files.

    Returns
    -------
    tuple[pd.DataFrame, dict[str, np.memmap]]
        The index and the block arrays by name.
    """
    index = pd.read_csv(directory / "index.csv", dtype={"record_key": str, "raw_path": str})
    blocks = {name: open_block(directory, name, size, "r") for name, size in block_sizes(index).items()}
    return index, blocks


def windows(index: pd.DataFrame, blocks: dict[str, np.memmap], positions: np.ndarray) -> np.ndarray:
    """
    Read the cached windows of selected records.

    Parameters
    ----------
    index : pd.DataFrame
        Output of ``load_cache``; ``position`` equals the row number.
    blocks : dict[str, np.memmap]
        Output of ``load_cache``.
    positions : np.ndarray
        Global positions to read.

    Returns
    -------
    np.ndarray
        Float32 ``[len(positions), 12, 5000]`` copy.
    """
    rows = index.iloc[np.asarray(positions)]
    return np.stack(
        [np.asarray(blocks[block][row]) for block, row in zip(rows["block"], rows["row"], strict=True)]
    )


def index_sha256(index: pd.DataFrame) -> str:
    """
    SHA-256 of the identity part of the index, independent of file formatting.

    Parameters
    ----------
    index : pd.DataFrame
        Output of ``cache_index`` or ``load_cache``.

    Returns
    -------
    str
        Hex digest over position, record key, block, row and manifest hash.
    """
    columns = ["position", "record_key", "block", "row", "sha256_10s"]
    text = index[columns].to_csv(index=False, lineterminator="\n")
    return hashlib.sha256(text.encode()).hexdigest()
