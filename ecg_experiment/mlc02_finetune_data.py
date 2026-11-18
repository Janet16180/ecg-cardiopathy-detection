"""
Waveform inputs of the MLC-02 fine-tuned candidate F, behind a role guard.

F fine-tunes the released xECG encoder on role ``train`` and is scored on role ``validation`` only. The test
roles (PTB-XL fold 10, the SPH evaluation half and the SPH calibration half) stay closed until the selection
record is committed: every loader here calls ``mlc02_feature_data.check_open_roles`` before it opens any file,
and ``open_blocks`` refuses to memory-map a cache block of any other role.

Records are the final build's included records (``mlc02_data.load_manifest``), read from the MLC-02 waveform
cache (``index_all.csv``; float32 ``[12, 5000]``, 500 Hz, mV, first 10 s, some records in ``__supplement``
blocks). Each record is matched to the manifest by key, role and source, the index's ``sha256_10s`` must equal
the manifest's, and ``prepare_inputs`` recomputes that hash from every cached window.

The encoder input is Stage 1's (``mlc02_features.encoder_inputs("xecg", ...)``): missing samples set to the
rail value, the window widened to float64 and FFT-resampled to 100 Hz, time-major float32 ``[1000, 12]``.
``prepare_inputs`` stores it once per role on local disk, with a JSON record of hashes, so that training reads
it from memory.
"""

from __future__ import annotations

import time
from multiprocessing import Pool
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ecg_experiment.external_encoders import xecg_input
from ecg_experiment.files import read_json, sha256_file, write_json_atomic
from ecg_experiment.mlc01_dataset import array_sha256, microvolts
from ecg_experiment.mlc02_data import MANIFEST, load_manifest, subset
from ecg_experiment.mlc02_feature_data import (
    LOCAL,
    OPEN_ROLES,
    aligned_rows,
    check_open_roles,
    feature_positions,
)
from ecg_experiment.mlc02_features import model_input
from ecg_experiment.mlc02_waveforms import CACHE, open_block
from ecg_experiment.xecg import MODEL_SAMPLES

INDEX = "index_all.csv"
WORK = Path("/var/tmp/janetrivera/mlc02/finetune")
INPUTS = WORK / "inputs"
CACHE_RECEIPTS = ("receipt.json", "receipt_supplement.json")
INPUT_SHAPE = (MODEL_SAMPLES, 12)
CHUNK = 256
WORKER: dict[str, Any] = {}


def block_role(block: str) -> str:
    """
    Role of a cache block from its name.

    Parameters
    ----------
    block : str
        Block name ``<role>__<source>`` or ``<role>__<source>__supplement``.

    Returns
    -------
    str
        The role.
    """
    return block.split("__", 1)[0]


def cache_rows(
    roles: tuple[str, ...], cache: Path = CACHE, manifest_path: Path = MANIFEST
) -> tuple[dict[str, np.ndarray], pd.DataFrame, dict[str, int]]:
    """
    Manifest arrays and cache positions of the included records of open roles.

    The role guard runs before any file is opened. Only index rows of the requested roles are kept.

    Parameters
    ----------
    roles : tuple[str, ...]
        Roles to read; ``train`` and ``validation`` only.
    cache : Path
        Waveform cache directory holding ``index_all.csv`` and the block files.
    manifest_path : Path
        MLC-01 ``manifest.csv``.

    Returns
    -------
    tuple[dict[str, np.ndarray], pd.DataFrame, dict[str, int]]
        ``load_manifest`` arrays of those records in manifest order; their index rows in the same order
        (``record_key``, ``role``, ``source``, ``block``, ``row``, ``sha256_10s``); and the record count of
        every block of those roles.

    Raises
    ------
    PermissionError
        If a closed role is requested.
    ValueError
        If a record has no cache row, or its role, source or first-10-s hash differs from the manifest.
    """
    roles = check_open_roles(roles)
    data = load_manifest(manifest_path)
    data = subset(data, np.isin(data["role"], roles))
    hashes = pd.read_csv(manifest_path, usecols=["record_key", "sha256_10s", "included"],
                         dtype={"record_key": str, "sha256_10s": str}, low_memory=False)
    hashes = hashes[hashes["included"].astype(bool)].set_index("record_key")
    columns = ["record_key", "role", "source", "block", "row", "sha256_10s"]
    index = pd.read_csv(Path(cache) / INDEX, usecols=columns, dtype={"record_key": str, "sha256_10s": str})
    index = index[index["role"].isin(roles)]
    sizes = {name: int(size) for name, size in index.groupby("block").size().items()}
    if index["record_key"].duplicated().any():
        raise ValueError("the cache index repeats a record key")
    keyed = index.set_index("record_key")
    missing = set(data["record_key"]) - set(keyed.index)
    if missing:
        raise ValueError(f"{len(missing)} records have no cache row")
    rows = keyed.loc[data["record_key"]].reset_index()
    expected = hashes.loc[data["record_key"], "sha256_10s"].to_numpy()
    for name, values in (("role", data["role"]), ("source", data["source"]), ("sha256_10s", expected)):
        if not np.array_equal(rows[name].to_numpy(dtype=str), np.asarray(values, dtype=str)):
            raise ValueError(f"cache index {name} is not aligned with the manifest")
    return data, rows[columns], sizes


def open_blocks(blocks: list[str], sizes: dict[str, int], cache: Path = CACHE) -> dict[str, np.memmap]:
    """
    Memory-map cache blocks of open roles, read-only.

    Parameters
    ----------
    blocks : list[str]
        Block names.
    sizes : dict[str, int]
        Record count of every block (from ``cache_rows``).
    cache : Path
        Waveform cache directory.

    Returns
    -------
    dict[str, np.memmap]
        Float32 ``[records, 12, 5000]`` arrays by name.

    Raises
    ------
    PermissionError
        If a block belongs to a closed role.
    """
    closed = sorted({name for name in blocks if block_role(name) not in OPEN_ROLES})
    if closed:
        raise PermissionError(f"cache blocks {closed} belong to roles that stay closed")
    return {name: open_block(Path(cache), name, sizes[name], "r") for name in dict.fromkeys(blocks)}


def window_input(window: np.ndarray) -> np.ndarray:
    """
    Compute the xECG input of one cached window, as Stage 1 computed it.

    Parameters
    ----------
    window : np.ndarray
        Float32 ``[12, 5000]`` window in mV; NaN where WFDB read a missing sample.

    Returns
    -------
    np.ndarray
        Float32 ``[1000, 12]``.
    """
    return xecg_input(model_input(window))


def window_inputs(windows: np.ndarray) -> np.ndarray:
    """
    Compute the xECG inputs of several waveforms.

    Parameters
    ----------
    windows : np.ndarray
        Float32 ``[records, 12, 5000]`` waveforms at 500 Hz in mV.

    Returns
    -------
    np.ndarray
        Float32 ``[records, 1000, 12]``.

    Raises
    ------
    ValueError
        If the array is not float32 ``[records, 12, 5000]``.
    """
    windows = np.asarray(windows)
    if windows.ndim != 3 or windows.shape[1:] != (12, 5000) or windows.dtype != np.float32:
        raise ValueError(f"expected float32 [records, 12, 5000] waveforms, got {windows.dtype} "
                         f"{windows.shape}")
    return np.stack([window_input(window) for window in windows]).astype(np.float32, copy=False)


def read_windows(
    roles: tuple[str, ...], record_keys: np.ndarray, cache: Path = CACHE, manifest_path: Path = MANIFEST
) -> np.ndarray:
    """
    Read the cached windows of selected records of open roles and check their first-10-s hash.

    Parameters
    ----------
    roles : tuple[str, ...]
        Roles the records belong to; ``train`` and ``validation`` only.
    record_keys : np.ndarray
        Keys to read, in the order wanted.
    cache : Path
        Waveform cache directory.
    manifest_path : Path
        MLC-01 ``manifest.csv``.

    Returns
    -------
    np.ndarray
        Float32 ``[records, 12, 5000]``.

    Raises
    ------
    PermissionError
        If a closed role is requested.
    ValueError
        If a key is not an included record of those roles or a window's hash differs from the manifest.
    """
    data, rows, sizes = cache_rows(roles, cache, manifest_path)
    position = pd.Series(np.arange(len(rows)), index=rows["record_key"].to_numpy())
    missing = set(record_keys) - set(position.index)
    if missing:
        raise ValueError(f"{len(missing)} keys are not included records of roles {list(roles)}")
    chosen = rows.iloc[position.loc[np.asarray(record_keys, dtype=str)].to_numpy()]
    blocks = open_blocks(chosen["block"].tolist(), sizes, cache)
    windows = np.stack(
        [np.asarray(blocks[block][row]) for block, row in zip(chosen["block"], chosen["row"], strict=True)]
    )
    found = [array_sha256(microvolts(window.T)) for window in windows]
    if found != chosen["sha256_10s"].tolist():
        raise ValueError("a cached window differs from its manifest hash")
    return windows


def init_worker(cache: str, blocks: list[str], sizes: dict[str, int]) -> None:
    """
    Open the cache blocks once per preparation process.

    Parameters
    ----------
    cache : str
        Waveform cache directory.
    blocks : list[str]
        Blocks to open.
    sizes : dict[str, int]
        Record count of every block.
    """
    WORKER["blocks"] = open_blocks(blocks, sizes, Path(cache))


def prepare_chunk(task: dict[str, Any]) -> dict[str, Any]:
    """
    Read, hash-check and convert one chunk of records in a worker process.

    Parameters
    ----------
    task : dict[str, Any]
        ``start`` (first output row), ``blocks``, ``rows`` and ``sha256_10s`` of the chunk's records.

    Returns
    -------
    dict[str, Any]
        ``start``, ``inputs`` (float32 ``[records, 1000, 12]``) and ``hash_ok`` (one flag per record).
    """
    blocks = WORKER["blocks"]
    pairs = zip(task["blocks"], task["rows"], strict=True)
    windows = [np.asarray(blocks[block][row]) for block, row in pairs]
    hash_ok = [
        array_sha256(microvolts(window.T)) == expected
        for window, expected in zip(windows, task["sha256_10s"], strict=True)
    ]
    inputs = np.stack([window_input(window) for window in windows])
    return {"start": task["start"], "inputs": inputs, "hash_ok": hash_ok}


def input_paths(role: str, directory: Path = INPUTS) -> dict[str, Path]:
    """
    Files of one role's prepared inputs.

    Parameters
    ----------
    role : str
        ``train`` or ``validation``.
    directory : Path
        Input directory.

    Returns
    -------
    dict[str, Path]
        ``inputs`` (``.npy``), ``records`` (``.csv``) and ``record`` (``.json``).
    """
    stem = Path(directory) / f"{role}__xecg_inputs"
    return {"inputs": stem.with_suffix(".npy"), "records": stem.with_suffix(".csv"),
            "record": stem.with_suffix(".json")}


def block_hashes(blocks: list[str], cache: Path = CACHE) -> dict[str, str]:
    """
    SHA-256 of cache block files, checked against the cache receipts.

    Parameters
    ----------
    blocks : list[str]
        Block names.
    cache : Path
        Waveform cache directory with ``receipt.json`` and ``receipt_supplement.json``.

    Returns
    -------
    dict[str, str]
        Digest per block.

    Raises
    ------
    ValueError
        If a block is not in a receipt or its digest differs.
    """
    expected = {}
    for name in CACHE_RECEIPTS:
        listed = read_json(Path(cache) / name)["blocks"]
        expected.update({block: value["sha256"] for block, value in listed.items()})
    found = {block: sha256_file(Path(cache) / f"{block}.npy") for block in sorted(set(blocks))}
    differing = [block for block, digest in found.items() if expected.get(block) != digest]
    if differing:
        raise ValueError(f"cache blocks differ from their receipts: {differing}")
    return found


def prepare_inputs(
    role: str, workers: int, cache: Path = CACHE, directory: Path = INPUTS, manifest_path: Path = MANIFEST
) -> dict[str, Any]:
    """
    Compute and store the xECG inputs of one open role, checking every window's first-10-s hash.

    Parameters
    ----------
    role : str
        ``train`` or ``validation``.
    workers : int
        Worker processes (at most 8 by the brief).
    cache : Path
        Waveform cache directory.
    directory : Path
        Output directory on local disk.
    manifest_path : Path
        MLC-01 ``manifest.csv``.

    Returns
    -------
    dict[str, Any]
        The JSON record written beside the arrays: records, hashes, block hashes and timing.

    Raises
    ------
    PermissionError
        If ``role`` is closed.
    ValueError
        If a window's hash differs from the manifest.
    """
    (role,) = check_open_roles((role,))
    started = time.perf_counter()
    data, rows, sizes = cache_rows((role,), cache, manifest_path)
    blocks = sorted(set(rows["block"]))
    blocks_sha256 = block_hashes(blocks, cache)
    paths = input_paths(role, directory)
    paths["inputs"].parent.mkdir(parents=True, exist_ok=True)
    temporary = paths["inputs"].with_suffix(".tmp.npy")
    output = np.lib.format.open_memmap(temporary, mode="w+", dtype=np.float32,
                                       shape=(len(rows), *INPUT_SHAPE))
    tasks = [
        {"start": start, "blocks": rows["block"].iloc[start:start + CHUNK].tolist(),
         "rows": rows["row"].iloc[start:start + CHUNK].tolist(),
         "sha256_10s": rows["sha256_10s"].iloc[start:start + CHUNK].tolist()}
        for start in range(0, len(rows), CHUNK)
    ]
    hash_ok = np.zeros(len(rows), dtype=bool)
    with Pool(workers, initializer=init_worker, initargs=(str(cache), blocks, sizes)) as pool:
        for result in pool.imap_unordered(prepare_chunk, tasks):
            stop = result["start"] + len(result["inputs"])
            output[result["start"]:stop] = result["inputs"]
            hash_ok[result["start"]:stop] = result["hash_ok"]
    output.flush()
    del output
    if not hash_ok.all():
        raise ValueError(f"{int((~hash_ok).sum())} cached windows differ from their manifest hash")
    temporary.replace(paths["inputs"])
    rows.to_csv(paths["records"], index=False)
    record = {
        "role": role,
        "records": len(rows),
        "shape": [len(rows), *INPUT_SHAPE],
        "dtype": "float32",
        "pipeline": "mlc02_features.model_input then external_encoders.xecg_input (float64 FFT to 100 Hz)",
        "windows_hash_checked": int(hash_ok.sum()),
        "cache": str(cache),
        "cache_index_sha256": sha256_file(Path(cache) / INDEX),
        "cache_blocks_sha256": blocks_sha256,
        "manifest_sha256": sha256_file(manifest_path),
        "inputs_sha256": sha256_file(paths["inputs"]),
        "records_sha256": sha256_file(paths["records"]),
        "workers": workers,
        "seconds": time.perf_counter() - started,
        "sources": {str(name): int(size) for name, size in pd.Series(data["source"]).value_counts().items()},
    }
    write_json_atomic(paths["record"], record)
    return record


def load_inputs(
    role: str, directory: Path = INPUTS, manifest_path: Path = MANIFEST, check_hash: bool = True
) -> tuple[dict[str, np.ndarray], np.ndarray, dict[str, Any]]:
    """
    Manifest arrays and in-memory xECG inputs of one open role.

    Parameters
    ----------
    role : str
        ``train`` or ``validation``.
    directory : Path
        Directory of ``prepare_inputs``.
    manifest_path : Path
        MLC-01 ``manifest.csv``.
    check_hash : bool
        Recompute the SHA-256 of the input file and compare it with its record.

    Returns
    -------
    tuple[dict[str, np.ndarray], np.ndarray, dict[str, Any]]
        The role's ``load_manifest`` arrays in manifest order, float32 ``[records, 1000, 12]`` inputs in the
        same order, and the preparation record.

    Raises
    ------
    PermissionError
        If ``role`` is closed.
    ValueError
        If the files differ from their record or the records are not the manifest's, in order.
    """
    (role,) = check_open_roles((role,))
    paths = input_paths(role, directory)
    record = read_json(paths["record"])
    if record["role"] != role or record["manifest_sha256"] != sha256_file(manifest_path):
        raise ValueError(f"{paths['record']} was prepared for another role or manifest")
    if sha256_file(paths["records"]) != record["records_sha256"]:
        raise ValueError(f"{paths['records']} differs from its record")
    if check_hash and sha256_file(paths["inputs"]) != record["inputs_sha256"]:
        raise ValueError(f"{paths['inputs']} differs from its record")
    data = load_manifest(manifest_path)
    data = subset(data, data["role"] == role)
    keys = pd.read_csv(paths["records"], usecols=["record_key"], dtype=str)["record_key"].to_numpy()
    if not np.array_equal(keys, data["record_key"]):
        raise ValueError(f"{paths['records']} is not the manifest's {role} records in order")
    inputs = np.load(paths["inputs"])
    if inputs.shape != (len(keys), *INPUT_SHAPE) or inputs.dtype != np.float32:
        raise ValueError(f"{paths['inputs']} has shape {inputs.shape} and dtype {inputs.dtype}")
    return data, inputs, record


def cached_xecg_features(data: dict[str, np.ndarray], directory: Path = LOCAL) -> np.ndarray:
    """
    Stage 1 pooled xECG features of manifest records of open roles.

    Parameters
    ----------
    data : dict[str, np.ndarray]
        ``load_manifest`` arrays of the records.
    directory : Path
        Stage 1 feature directory with ``rows.csv`` and ``xecg.npy``.

    Returns
    -------
    np.ndarray
        Float32 ``[records, 1024]`` in the order of ``data``.

    Raises
    ------
    PermissionError
        If a record belongs to a closed role.
    """
    check_open_roles(tuple(sorted(set(data["role"]))))
    rows = aligned_rows(Path(directory) / "rows.csv")
    index = feature_positions(data["record_key"], data["role"], rows)
    return np.asarray(np.load(Path(directory) / "xecg.npy", mmap_mode="r")[index])
