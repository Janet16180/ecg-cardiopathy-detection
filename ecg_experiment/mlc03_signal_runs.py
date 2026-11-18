"""
Runs of the MLC-03 Part A measures over the MLC-02 waveform cache and the raw files.

Development reads only the roles ``train`` and ``validation``: ``check_roles`` raises on any other role
before an index row, a cache block or a raw file is opened. The test roles (PTB-XL fold 10 and both SPH
halves) are opened only by the final frozen run (``allow_test=True``), which the run script refuses unless
the working tree is clean. Only identity columns of the manifest are read here, never a label.

Every cached window is checked against its manifest ``sha256_10s`` before it is measured. Work is split into
chunks of consecutive rows of one cache block; each finished chunk is written as its own parquet file, so an
interrupted run resumes where it stopped.
"""

from __future__ import annotations

import hashlib
import os
import subprocess
import time
from multiprocessing import Pool
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ecg_experiment import ROOT
from ecg_experiment.eda.signals import canonical_order
from ecg_experiment.files import sha256_file, write_json_atomic
from ecg_experiment.mlc01_dataset import array_sha256, microvolts, read_signal
from ecg_experiment.mlc02_waveforms import CACHE, closed_keys, guard_path, init_worker
from ecg_experiment.mlc03_signal_quality import SAMPLES, measure_full_record, measure_window
from ecg_experiment.paths import to_stored

MANIFEST = ROOT / "outputs/mlc01_condition_dataset_20261110_v4/manifest.csv"
INDEX = CACHE / "index_all.csv"
OUTPUT = ROOT / "outputs/mlc03_data_quality_20261111/signal"
WORK = Path("/var/tmp/janetrivera/mlc03/signal")
DEVELOPMENT_ROLES = ("train", "validation")
TEST_ROLES = ("test_same_source", "test_external", "sph_calibration")
IDENTITY = ["record_key", "source", "role", "raw_path", "included", "duplicate_status", "n_samples",
            "duration_s", "exclusion_reason", "sha256_10s"]
CHUNK = 256
EDAC_COUNTER = Path("/sys/devices/system/edac/mc/mc0/ue_count")
THREAD_VARIABLES = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_CORETYPE",
                    "PYTHONPATH")


def check_roles(roles: list[str] | tuple[str, ...], allow_test: bool = False) -> tuple[str, ...]:
    """
    Refuse a test role unless the final frozen run asks for it.

    Parameters
    ----------
    roles : list[str] | tuple[str, ...]
        Requested roles.
    allow_test : bool
        True only for the final run over the test roles.

    Returns
    -------
    tuple[str, ...]
        The roles, unchanged.

    Raises
    ------
    PermissionError
        If a test role is requested without ``allow_test``.
    ValueError
        If a role is unknown.
    """
    roles = tuple(roles)
    unknown = set(roles) - set(DEVELOPMENT_ROLES) - set(TEST_ROLES)
    if unknown:
        raise ValueError(f"unknown roles {sorted(unknown)}")
    if not allow_test and set(roles) & set(TEST_ROLES):
        closed = sorted(set(roles) & set(TEST_ROLES))
        raise PermissionError(f"test roles stay closed during development: {closed}")
    return roles


def read_identity(roles: tuple[str, ...], allow_test: bool = False,
                  manifest: Path = MANIFEST) -> pd.DataFrame:
    """
    Identity columns of the manifest records of some roles that have a cached window.

    Parameters
    ----------
    roles : tuple[str, ...]
        Roles to keep.
    allow_test : bool
        Passed to ``check_roles``.
    manifest : Path
        MLC-01 v4 ``manifest.csv``.

    Returns
    -------
    pd.DataFrame
        ``IDENTITY`` columns (``included`` as bool) of records without an exclusion reason, in manifest order.
    """
    roles = check_roles(roles, allow_test)
    frame = pd.read_csv(manifest, usecols=IDENTITY, dtype={"raw_path": str, "included": str,
                                                           "exclusion_reason": str},
                        keep_default_na=False, low_memory=False)
    frame = frame[frame["role"].isin(roles) & (frame["exclusion_reason"] == "")].reset_index(drop=True)
    return frame.assign(included=frame["included"] == "True", n_samples=frame["n_samples"].astype(int),
                        duration_s=frame["duration_s"].astype(float))


def read_index(roles: tuple[str, ...], allow_test: bool = False, index: Path = INDEX) -> pd.DataFrame:
    """
    Cache index rows of some roles.

    Parameters
    ----------
    roles : tuple[str, ...]
        Roles to keep.
    allow_test : bool
        Passed to ``check_roles``.
    index : Path
        ``index_all.csv`` of the MLC-02 cache.

    Returns
    -------
    pd.DataFrame
        ``position``, ``record_key``, ``source``, ``role``, ``block``, ``row`` and ``sha256_10s``.

    Raises
    ------
    PermissionError
        If a selected block name belongs to another role.
    """
    roles = check_roles(roles, allow_test)
    columns = ["position", "record_key", "source", "role", "block", "row", "sha256_10s"]
    frame = pd.read_csv(index, usecols=columns, dtype={"record_key": str})
    frame = frame[frame["role"].isin(roles)].reset_index(drop=True)
    if not frame["block"].str.split("__").str[0].isin(roles).all():
        raise PermissionError("a selected block belongs to another role")
    return frame


def chunk_tasks(index: pd.DataFrame, size: int = CHUNK, allow_test: bool = False) -> list[dict[str, Any]]:
    """
    Split the index into chunks of consecutive rows of one block.

    Parameters
    ----------
    index : pd.DataFrame
        Output of ``read_index``.
    size : int
        Records per chunk.
    allow_test : bool
        Carried into each task for the worker's role check.

    Returns
    -------
    list[dict[str, Any]]
        Tasks with ``name``, ``block``, ``rows``, ``positions``, ``keys``, ``hashes`` and ``allow_test``.
    """
    tasks = []
    for block, rows in index.groupby("block", sort=False):
        for start in range(0, len(rows), size):
            part = rows.iloc[start : start + size]
            tasks.append({"name": f"{block}__{int(part['row'].iloc[0]):06d}", "block": block,
                          "rows": part["row"].to_numpy(), "positions": part["position"].to_numpy(),
                          "keys": part["record_key"].tolist(), "hashes": part["sha256_10s"].tolist(),
                          "allow_test": allow_test})
    return tasks


def open_block(block: str, allow_test: bool, cache: Path = CACHE) -> np.ndarray:
    """
    Memory-map one cache block after checking its role.

    Parameters
    ----------
    block : str
        Block name ``<role>__<source>[__supplement]``.
    allow_test : bool
        Passed to ``check_roles``.
    cache : Path
        Cache directory.

    Returns
    -------
    np.ndarray
        Read-only float32 ``[records, 12, 5000]``.
    """
    check_roles((block.split("__")[0],), allow_test)
    array = np.load(cache / f"{block}.npy", mmap_mode="r")
    if array.dtype != np.float32 or array.shape[1:] != (12, SAMPLES):
        raise ValueError(f"unexpected cache block {block}: {array.shape} {array.dtype}")
    return array


def read_checked(array: np.ndarray, row: int, expected: str, key: str) -> np.ndarray:
    """
    Read one cached window and check it against its manifest hash.

    Parameters
    ----------
    array : np.ndarray
        Output of ``open_block``.
    row : int
        Row in the block.
    expected : str
        Manifest ``sha256_10s``.
    key : str
        Record key, for the error message.

    Returns
    -------
    np.ndarray
        Float64 ``(12, 5000)``.

    Raises
    ------
    ValueError
        If the window does not reproduce the hash.
    """
    window = np.asarray(array[row])
    if array_sha256(microvolts(window.T)) != expected:
        raise ValueError(f"cached window of {key} does not reproduce its manifest hash")
    return window.astype(np.float64)


def measure_chunk(task: dict[str, Any]) -> pd.DataFrame:
    """
    Measure every record of one chunk.

    Parameters
    ----------
    task : dict[str, Any]
        One output of ``chunk_tasks``.

    Returns
    -------
    pd.DataFrame
        ``position``, ``record_key`` and every column of ``measure_window``.
    """
    array = open_block(task["block"], task["allow_test"])
    rows = []
    for row, position, key, expected in zip(task["rows"], task["positions"], task["keys"], task["hashes"],
                                            strict=True):
        window = read_checked(array, int(row), expected, key)
        rows.append({"position": int(position), "record_key": key, **measure_window(window)})
    del array
    return pd.DataFrame(rows)


def run_chunk(task: dict[str, Any]) -> tuple[str, int, float]:
    """
    Measure one chunk and write it, unless its file already exists.

    Parameters
    ----------
    task : dict[str, Any]
        One output of ``chunk_tasks`` plus ``directory``.

    Returns
    -------
    tuple[str, int, float]
        Chunk name, records written (0 when skipped) and seconds.
    """
    path = Path(task["directory"]) / f"{task['name']}.parquet"
    if path.exists():
        return task["name"], 0, 0.0
    start = time.perf_counter()
    frame = measure_chunk(task)
    temporary = path.with_suffix(".tmp")
    frame.to_parquet(temporary, index=False)
    os.replace(temporary, path)
    return task["name"], len(frame), time.perf_counter() - start


def run_measures(directory: Path, roles: tuple[str, ...], workers: int, limit: int | None = None,
                 allow_test: bool = False, log: Path | None = None) -> dict[str, Any]:
    """
    Measure every cached window of some roles in parallel, resuming from finished chunks.

    Parameters
    ----------
    directory : Path
        Chunk directory.
    roles : tuple[str, ...]
        Roles to measure.
    workers : int
        Processes.
    limit : int | None
        Keep only the first ``limit`` records of each block (smoke test).
    allow_test : bool
        Passed to ``check_roles``.
    log : Path | None
        Progress log, appended.

    Returns
    -------
    dict[str, Any]
        ``records``, ``chunks``, ``written``, ``wall_seconds``.
    """
    index = read_index(roles, allow_test)
    if limit is not None:
        index = index.groupby("block", sort=False).head(limit).reset_index(drop=True)
    tasks = [{**task, "directory": str(directory)} for task in chunk_tasks(index, allow_test=allow_test)]
    directory.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    written = 0
    with Pool(workers) as pool:
        for done, (name, count, seconds) in enumerate(pool.imap_unordered(run_chunk, tasks), start=1):
            written += count
            if log is not None:
                with open(log, "a") as handle:
                    stamp = time.strftime("%H:%M:%S")
                    handle.write(f"{stamp} {done}/{len(tasks)} {name} {count} {seconds:.1f}s\n")
    return {"records": len(index), "chunks": len(tasks), "written": written,
            "wall_seconds": time.perf_counter() - start}


def assemble(directory: Path, index: pd.DataFrame) -> pd.DataFrame:
    """
    Concatenate the chunk files and check that they cover the index exactly.

    Parameters
    ----------
    directory : Path
        Chunk directory.
    index : pd.DataFrame
        Output of ``read_index`` for the same roles.

    Returns
    -------
    pd.DataFrame
        One row per index row, in position order.

    Raises
    ------
    ValueError
        If a position is missing, repeated or foreign, or its key differs from the index.
    """
    frame = pd.concat([pd.read_parquet(path) for path in sorted(directory.glob("*.parquet"))],
                      ignore_index=True)
    frame = frame.sort_values("position").reset_index(drop=True)
    expected = index.sort_values("position").reset_index(drop=True)
    if not (len(frame) == len(expected) and np.array_equal(frame["position"], expected["position"])
            and np.array_equal(frame["record_key"], expected["record_key"])):
        raise ValueError("chunk files do not cover the index exactly")
    return frame


def full_record_job(task: dict[str, Any]) -> dict[str, Any]:
    """
    Read one raw record through the closed-data guard and measure its whole length.

    Parameters
    ----------
    task : dict[str, Any]
        ``record_key``, ``raw_path``, ``source`` and ``role``.

    Returns
    -------
    dict[str, Any]
        ``record_key`` and the output of ``measure_full_record``.
    """
    check_roles((task["role"],), task.get("allow_test", False))
    path = guard_path(task["raw_path"], task["source"])
    job = {"format": "h5" if task["source"] == "sph" else "wfdb", "stem": str(path)}
    signal, rate, names = read_signal(job)
    if rate != 500:
        raise ValueError(f"not 500 Hz: {task['record_key']}")
    return {"record_key": task["record_key"], **measure_full_record(canonical_order(signal, names).T)}


def run_full_records(identity: pd.DataFrame, workers: int, allow_test: bool = False) -> pd.DataFrame:
    """
    Measure the whole raw recording of every record longer than 10 s.

    Parameters
    ----------
    identity : pd.DataFrame
        Output of ``read_identity``.
    workers : int
        Processes.
    allow_test : bool
        Passed to ``check_roles``.

    Returns
    -------
    pd.DataFrame
        One row per long record.
    """
    long = identity[identity["n_samples"] > SAMPLES]
    tasks = [{**row, "allow_test": allow_test}
             for row in long[["record_key", "raw_path", "source", "role"]].to_dict("records")]
    allowed = frozenset(identity["raw_path"])
    with Pool(workers, initializer=init_worker, initargs=(closed_keys(), allowed)) as pool:
        rows = pool.map(full_record_job, tasks, chunksize=16)
    return pd.DataFrame(rows, columns=None if rows else ["record_key"])


def git_state(path: Path = ROOT) -> dict[str, Any]:
    """
    Commit and modified tracked files of the working tree.

    Parameters
    ----------
    path : Path
        Any directory inside the working tree.

    Returns
    -------
    dict[str, Any]
        ``commit`` and ``modified_tracked`` (paths from ``git status --porcelain``, untracked excluded).
    """
    commit = subprocess.check_output(["git", "-C", str(path), "rev-parse", "HEAD"], text=True).strip()
    status = subprocess.check_output(
        ["git", "-C", str(path), "status", "--porcelain", "--untracked-files=no"], text=True)
    return {"commit": commit, "modified_tracked": [line[3:] for line in status.splitlines()]}


def environment() -> dict[str, Any]:
    """
    Thread settings, package versions and the memory-error counter.

    Returns
    -------
    dict[str, Any]
        ``variables``, ``packages`` and ``edac_uncorrected`` (None when the counter is unreadable).
    """
    import scipy
    import wfdb

    counter = EDAC_COUNTER.read_text().strip() if EDAC_COUNTER.exists() else None
    return {"variables": {name: os.environ.get(name) for name in THREAD_VARIABLES},
            "packages": {"numpy": np.__version__, "pandas": pd.__version__, "scipy": scipy.__version__,
                         "wfdb": wfdb.__version__},
            "edac_uncorrected": counter}


def input_hashes() -> dict[str, str]:
    """
    SHA-256 of the manifest, the cache index and the cache receipts.

    Returns
    -------
    dict[str, str]
        Repository-relative or absolute path to digest.
    """
    files = [MANIFEST, INDEX, CACHE / "receipt.json", CACHE / "receipt_supplement.json"]
    return {(to_stored(path) if path.is_relative_to(ROOT) else str(path)): sha256_file(path)
            for path in files}


def code_hashes(names: list[str]) -> dict[str, str]:
    """
    SHA-256 of code files, by repository-relative path.

    Parameters
    ----------
    names : list[str]
        Repository-relative paths.

    Returns
    -------
    dict[str, str]
        Path to digest.
    """
    return {name: sha256_file(ROOT / name) for name in names}


def frame_sha256(frame: pd.DataFrame) -> str:
    """
    SHA-256 of a table's CSV text, independent of the file format it is stored in.

    Parameters
    ----------
    frame : pd.DataFrame
        Any table.

    Returns
    -------
    str
        Hex digest.
    """
    return hashlib.sha256(frame.to_csv(index=False, lineterminator="\n").encode()).hexdigest()


def write_receipt(path: Path, content: dict[str, Any]) -> None:
    """
    Write a receipt with the git state and environment added.

    Parameters
    ----------
    path : Path
        Destination JSON.
    content : dict[str, Any]
        Step-specific content.
    """
    write_json_atomic(path, {**content, "git": git_state(), "environment": environment(),
                             "written": time.strftime("%Y-%m-%d %H:%M:%S %Z")})
