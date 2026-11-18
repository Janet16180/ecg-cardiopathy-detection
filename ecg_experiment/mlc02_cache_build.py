"""Build and verify the MLC-02 waveform cache with resumable chunks and a receipt."""

from __future__ import annotations

import importlib.metadata
import json
import os
import resource
import subprocess
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import pandas as pd

from ecg_experiment import ROOT
from ecg_experiment.files import sha256_file, write_json_atomic
from ecg_experiment.mlc02_waveforms import (
    MANIFEST,
    OTHER_MANIFESTS,
    block_sizes,
    cache_index,
    chunk_tasks,
    closed_keys,
    compare_manifests,
    index_sha256,
    init_worker,
    open_block,
    read_identity,
    read_rows,
    verify_chunk,
    write_chunk,
)
from ecg_experiment.paths import to_stored

CHUNK = 256
CODE = (
    "ecg_experiment/mlc02_waveforms.py",
    "ecg_experiment/mlc02_cache_build.py",
    "ecg_experiment/mlc01_dataset.py",
    "ecg_experiment/eda/signals.py",
    "ecg_experiment/public_sources.py",
    "ecg_experiment/paths.py",
    "scripts/experiments/run_mlc02_inputs.py",
)
PACKAGES = ("numpy", "pandas", "wfdb", "h5py")
ENVIRONMENT = (
    "OPENBLAS_CORETYPE",
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "PYTHONPATH",
)


def log(path: Path, message: str) -> None:
    """
    Print a message and append it to a log file.

    Parameters
    ----------
    path : Path
        Log file.
    message : str
        Text to record.
    """
    line = f"{time.strftime('%Y-%m-%d %H:%M:%S')} {message}"
    print(line, flush=True)
    with path.open("a") as handle:
        handle.write(line + "\n")


def git_state() -> dict[str, object]:
    """
    Commit of the checkout and whether tracked files differ from it.

    Returns
    -------
    dict[str, object]
        ``commit`` and ``modified_tracked`` (paths from ``git status``).
    """
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    status = subprocess.check_output(
        ["git", "status", "--porcelain", "--untracked-files=no"], cwd=ROOT, text=True
    ).splitlines()
    return {"commit": commit, "modified_tracked": [line[3:] for line in status]}


def code_sha256(paths: tuple[str, ...]) -> dict[str, str]:
    """
    SHA-256 of repository files.

    Parameters
    ----------
    paths : tuple[str, ...]
        Repository-relative paths.

    Returns
    -------
    dict[str, str]
        Digest per path.
    """
    return {path: sha256_file(ROOT / path) for path in paths}


def other_manifests() -> dict[str, dict[str, object]]:
    """
    Compare the manifest used with the other MLC-01 builds that exist.

    Returns
    -------
    dict[str, dict[str, object]]
        Per other manifest: ``present`` and, when present, ``sha256`` and ``compare_manifests``.

    Raises
    ------
    ValueError
        If another build differs in record identity, role, path or waveform hash.
    """
    used = read_rows(MANIFEST)
    record = {}
    for path in OTHER_MANIFESTS:
        record[to_stored(path)] = {"present": path.exists()}
        if path.exists():
            comparison = compare_manifests(used, read_rows(path))
            if not comparison["identity_equal"]:
                raise ValueError(f"{path} differs in identity, role, path or hash")
            record[to_stored(path)] |= {"sha256": sha256_file(path), **comparison}
    return record


def manifest_identity(limit: int | None) -> tuple[pd.DataFrame, dict[str, object]]:
    """
    Included records of the final MLC-01 build to cache, and its comparison with the other builds.

    Parameters
    ----------
    limit : int | None
        First records per source only, for a profile.

    Returns
    -------
    tuple[pd.DataFrame, dict[str, object]]
        Identity rows and a record of the manifests used.

    Raises
    ------
    ValueError
        If another build differs in identity, role, path or hash.
    """
    identity = read_identity(MANIFEST)
    record = {
        "manifest": to_stored(MANIFEST),
        "manifest_sha256": sha256_file(MANIFEST),
        "other_manifests": other_manifests(),
    }
    if limit is not None:
        identity = identity.groupby("source", sort=False).head(limit)
    return identity, record


def build(directory: Path, workers: int, limit: int | None) -> dict[str, object]:
    """
    Write every block of the cache, resuming after the last finished chunk.

    Parameters
    ----------
    directory : Path
        Cache directory.
    workers : int
        Reader processes.
    limit : int | None
        First records per source only, for a profile.

    Returns
    -------
    dict[str, object]
        Build summary written to ``build.json``.

    Raises
    ------
    ValueError
        If an existing cache was planned from other records, or a record does not reproduce its hash.
    """
    started = time.perf_counter()
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "chunks").mkdir(exist_ok=True)
    log_path = directory / "build.log"
    identity, manifests = manifest_identity(limit)
    index = cache_index(identity)
    plan = {
        "index_sha256": index_sha256(index),
        "records": len(index),
        "limit_per_source": limit,
        "chunk": CHUNK,
        **manifests,
    }
    plan_path = directory / "plan.json"
    if plan_path.exists():
        previous = json.loads(plan_path.read_text())
        if previous["index_sha256"] != plan["index_sha256"] or previous["chunk"] != CHUNK:
            raise ValueError("An existing cache was planned from other records")
    else:
        index.to_csv(directory / "index_plan.csv", index=False)
        write_json_atomic(plan_path, plan)
    sizes = block_sizes(index)
    for name, size in sizes.items():
        if not (directory / f"{name}.npy").exists():
            array = open_block(directory, name, size, "w+")
            del array
    tasks = chunk_tasks(index, directory, CHUNK)
    pending = [task for task in tasks if not (directory / "chunks" / f"{task['name']}.json").exists()]
    log(log_path, f"{len(index)} records, {len(tasks)} chunks, {len(pending)} pending, {workers} workers")
    allowed = frozenset(index["raw_path"])
    with Pool(workers, initializer=init_worker, initargs=(closed_keys(), allowed)) as pool:
        for count, (task, results) in enumerate(zip(pending, pool.imap(write_chunk, pending), strict=True)):
            write_json_atomic(directory / "chunks" / f"{task['name']}.json", results)
            if count % 20 == 0 or count == len(pending) - 1:
                log(
                    log_path,
                    f"chunk {count + 1}/{len(pending)} {task['name']} {time.perf_counter() - started:.1f} s",
                )
    checks = pd.DataFrame(
        [
            row
            for task in tasks
            for row in json.loads((directory / "chunks" / f"{task['name']}.json").read_text())
        ]
    )
    checks = checks.sort_values("position").reset_index(drop=True)
    if not checks["position"].equals(index["position"]):
        raise ValueError("Chunk results do not cover the index")
    if not checks["sha256_read_ok"].all():
        raise ValueError(f"{(~checks['sha256_read_ok']).sum()} records do not reproduce the manifest hash")
    summary = {
        "records": len(index),
        "wall_seconds": time.perf_counter() - started,
        "workers": workers,
        "sha256_read_ok": int(checks["sha256_read_ok"].sum()),
        "sha256_float32_ok": int(checks["sha256_float32_ok"].sum()),
        "records_with_missing_samples": int((checks["missing_samples_10s"] > 0).sum()),
        "missing_samples": int(checks["missing_samples_10s"].sum()),
        "records_with_flat_leads": int((checks["flat_leads_10s"] > 0).sum()),
        "parent_max_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "children_max_rss_kib": resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss,
    }
    index = index.merge(checks, on="position", validate="one_to_one")
    index.to_csv(directory / "index_build.csv", index=False)
    write_json_atomic(directory / "build.json", summary)
    log(log_path, f"build finished: {json.dumps(summary)}")
    return summary


def file_sha256(path: str) -> str:
    """
    SHA-256 of one file, for a process pool.

    Parameters
    ----------
    path : str
        File path.

    Returns
    -------
    str
        Hex digest.
    """
    return sha256_file(path)


def verify(directory: Path, workers: int, command: list[str]) -> dict[str, object]:
    """
    Recompute every record's manifest hash from the written files and write the index and receipt.

    Parameters
    ----------
    directory : Path
        Cache directory after ``build``.
    workers : int
        Reader processes.
    command : list[str]
        Command line, for the receipt.

    Returns
    -------
    dict[str, object]
        The receipt written to ``receipt.json``.

    Raises
    ------
    ValueError
        If any cached record does not reproduce its manifest hash.
    """
    started = time.perf_counter()
    index = pd.read_csv(directory / "index_build.csv", dtype={"record_key": str, "raw_path": str})
    tasks = chunk_tasks(index, directory, CHUNK * 4)
    with Pool(workers) as pool:
        checks = pd.DataFrame([row for part in pool.imap(verify_chunk, tasks) for row in part])
        names = list(block_sizes(index))
        digests = dict(
            zip(names, pool.map(file_sha256, [str(directory / f"{name}.npy") for name in names]), strict=True)
        )
    checks = (
        checks.sort_values("position").reset_index(drop=True).rename(columns={"sha256_f32": "reread_f32"})
    )
    index = index.merge(checks, on="position", validate="one_to_one")
    if not index["sha256_cache_ok"].all() or not (index["reread_f32"] == index["sha256_f32"]).all():
        raise ValueError(f"{(~index['sha256_cache_ok']).sum()} cached records do not reproduce their hash")
    index = index.drop(columns=["reread_f32"])
    index.to_csv(directory / "index.csv", index=False)
    build_summary = json.loads((directory / "build.json").read_text())
    plan = json.loads((directory / "plan.json").read_text())
    counts = index.groupby(["role", "source"]).size()
    receipt = {
        "stage": "mlc02_waveform_cache",
        "command": command,
        "plan": plan,
        "build": build_summary,
        "verify_wall_seconds": time.perf_counter() - started,
        "records": len(index),
        "sha256_cache_ok": int(index["sha256_cache_ok"].sum()),
        "counts": {f"{role}/{source}": int(size) for (role, source), size in counts.items()},
        "blocks": {
            name: {"records": size, "sha256": digests[name]} for name, size in block_sizes(index).items()
        },
        "index_sha256": sha256_file(directory / "index.csv"),
        "index_identity_sha256": index_sha256(index),
        "layout": "float32 [records, 12, 5000], leads I, II, III, aVR, aVL, aVF, V1-V6, 500 Hz, mV, "
        "first 10 s; "
        "missing WFDB samples are NaN",
        "hash_rule": "sha256 of int32 round(mV * 1000), (5000, 12) C order, NaN as -99999 mV (MLC-01)",
        "git": git_state(),
        "code_sha256": code_sha256(CODE),
        "python": sys.version,
        "packages": {name: importlib.metadata.version(name) for name in PACKAGES},
        "environment": {name: os.environ.get(name, "") for name in ENVIRONMENT},
    }
    write_json_atomic(directory / "receipt.json", receipt)
    return receipt
