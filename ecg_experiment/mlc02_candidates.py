"""Every MLC-01 candidate record for MLC-02 Stage 1, not only one build's included set.

A candidate is a manifest record with no exclusion reason: readable, twelve named leads, 500 Hz and at least
10 s, whether the duplicate rule kept it or not. Candidate keys, roles, paths and first-10-s hashes are the
same in every MLC-01 build; the builds differ only in which duplicate copies they include. The waveform cache
was first built for the included records of v3 (``index.csv``); the remaining candidates are added as
``<role>__<source>__supplement`` blocks after them, and ``index_all.csv`` lists every candidate. Only
identity, exclusion and duplicate-status columns of the manifests are read; no label column is opened.
"""

from __future__ import annotations

import json
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

from ecg_experiment import ROOT, mlc01_dataset
from ecg_experiment.files import sha256_file, write_json_atomic
from ecg_experiment.mlc02_cache_build import CHUNK, code_sha256, git_state, log
from ecg_experiment.mlc02_waveforms import (
    IDENTITY_COLUMNS,
    RATE,
    SAMPLES,
    SOURCES,
    block_sizes,
    chunk_tasks,
    closed_keys,
    index_sha256,
    init_worker,
    open_block,
    verify_chunk,
    write_chunk,
)
from ecg_experiment.paths import to_stored

BUILDS = {
    "v1": ROOT / "outputs/mlc01_condition_dataset_20261110/manifest.csv",
    "v2": ROOT / "outputs/mlc01_condition_dataset_20261110_v2/manifest.csv",
    "v3": ROOT / "outputs/mlc01_condition_dataset_20261110_v3/manifest.csv",
    "v4": ROOT / "outputs/mlc01_condition_dataset_20261110_v4/manifest.csv",
}
FINAL = "v4"
CANDIDATE_COLUMNS = IDENTITY_COLUMNS + ["exclusion_reason", "duplicate_status"]
COMPARED = ["manifest_row", "record_key", "source", "role", "raw_path", "sha256_10s"]
CANDIDATE_RECORDS = 100158
SUPPLEMENT = "__supplement"
CODE = (
    "ecg_experiment/mlc02_candidates.py",
    "ecg_experiment/mlc02_waveforms.py",
    "ecg_experiment/mlc02_cache_build.py",
    "ecg_experiment/mlc01_dataset.py",
    "ecg_experiment/eda/signals.py",
    "ecg_experiment/public_sources.py",
    "scripts/experiments/run_mlc02_inputs.py",
)


def read_candidates(path: Path, expected: int | None = None) -> pd.DataFrame:
    """
    Identity, inclusion and duplicate status of every candidate record of a manifest, in manifest order.

    Parameters
    ----------
    path : Path
        MLC-01 ``manifest.csv``.
    expected : int | None
        Required number of candidates; None skips the check.

    Returns
    -------
    pd.DataFrame
        ``IDENTITY_COLUMNS``, ``duplicate_status`` and ``manifest_row`` of rows without an exclusion reason;
        ``included`` as bool.

    Raises
    ------
    ValueError
        If an excluded record is included, a key repeats, a source or role is unknown, a candidate is not 10 s
        at 500 Hz, or the count differs.
    """
    frame = pd.read_csv(
        path,
        usecols=CANDIDATE_COLUMNS,
        keep_default_na=False,
        dtype={"record": str, "raw_path": str, "included": str, "exclusion_reason": str},
    )
    if not set(frame["included"]) <= {"True", "False"}:
        raise ValueError(f"Unexpected included values in {path}")
    frame = frame[CANDIDATE_COLUMNS].assign(
        included=frame["included"] == "True", manifest_row=np.arange(len(frame))
    )
    if (frame["included"] & (frame["exclusion_reason"] != "")).any():
        raise ValueError("An excluded record is marked included")
    candidates = (
        frame[frame["exclusion_reason"] == ""].drop(columns="exclusion_reason").reset_index(drop=True)
    )
    if candidates["record_key"].duplicated().any():
        raise ValueError("Duplicate record key among candidates")
    if not set(candidates["source"]) <= set(SOURCES) or not set(candidates["role"]) <= set(
        mlc01_dataset.ROLES
    ):
        raise ValueError("Unknown source or role among candidates")
    if (candidates["fs"].astype(int) != RATE).any() or (candidates["n_samples"].astype(int) < SAMPLES).any():
        raise ValueError("A candidate is not at least 10 s at 500 Hz")
    if expected is not None and len(candidates) != expected:
        raise ValueError(f"Expected {expected} candidates, got {len(candidates)}")
    return candidates


def supplement_index(base: pd.DataFrame, candidates: pd.DataFrame) -> pd.DataFrame:
    """
    Cache rows of the candidates missing from the first cache, numbered after it.

    Parameters
    ----------
    base : pd.DataFrame
        The first cache's ``index.csv``.
    candidates : pd.DataFrame
        Output of ``read_candidates``.

    Returns
    -------
    pd.DataFrame
        ``cache_index`` columns; ``position`` continues after the base, ``block`` is
        ``<role>__<source>__supplement`` and ``row`` counts within it, in manifest order.

    Raises
    ------
    ValueError
        If a base record is not a candidate with the same identity.
    """
    joined = base.merge(
        candidates, on="record_key", how="left", suffixes=("", "_manifest"), validate="one_to_one"
    )
    for column in COMPARED[:1] + COMPARED[2:]:
        if not (joined[column].astype(str) == joined[f"{column}_manifest"].astype(str)).all():
            raise ValueError(f"Cache records differ from the candidates in {column}")
    extra = candidates[~candidates["record_key"].isin(base["record_key"])].sort_values("manifest_row")
    extra = extra.reset_index(drop=True)
    block = [
        f"{role}__{source}{SUPPLEMENT}" for role, source in zip(extra["role"], extra["source"], strict=True)
    ]
    extra = extra.assign(block=block, position=len(base) + np.arange(len(extra)))
    extra["row"] = extra.groupby("block").cumcount()
    return extra[
        ["position", "manifest_row", "record_key", "source", "role", "block", "row", "raw_path", "sha256_10s"]
    ]


def build_supplement(directory: Path, workers: int, command: list[str]) -> dict[str, object]:
    """
    Cache the remaining candidates, verify them from the written files and write ``index_all.csv``.

    Parameters
    ----------
    directory : Path
        The waveform cache directory holding ``index.csv``.
    workers : int
        Reader processes.
    command : list[str]
        Command line, for the receipt.

    Returns
    -------
    dict[str, object]
        The receipt written to ``receipt_supplement.json``.

    Raises
    ------
    ValueError
        If a record does not reproduce its manifest hash, or an existing supplement was planned differently.
    """
    started = time.perf_counter()
    log_path = directory / "build.log"
    base = pd.read_csv(directory / "index.csv", dtype={"record_key": str, "raw_path": str})
    candidates = read_candidates(BUILDS[FINAL], CANDIDATE_RECORDS)
    extra = supplement_index(base, candidates)
    plan = {
        "index_sha256": index_sha256(extra),
        "records": len(extra),
        "chunk": CHUNK,
        "candidates_manifest": to_stored(BUILDS[FINAL]),
        "candidates_manifest_sha256": sha256_file(BUILDS[FINAL]),
    }
    plan_path = directory / "plan_supplement.json"
    if plan_path.exists() and json.loads(plan_path.read_text())["index_sha256"] != plan["index_sha256"]:
        raise ValueError("An existing supplement was planned from other records")
    write_json_atomic(plan_path, plan)
    for name, size in block_sizes(extra).items():
        if not (directory / f"{name}.npy").exists():
            array = open_block(directory, name, size, "w+")
            del array
    markers = directory / "chunks_supplement"
    markers.mkdir(exist_ok=True)
    tasks = chunk_tasks(extra, directory, CHUNK)
    pending = [task for task in tasks if not (markers / f"{task['name']}.json").exists()]
    log(log_path, f"supplement: {len(extra)} records, {len(tasks)} chunks, {len(pending)} pending")
    with Pool(
        workers, initializer=init_worker, initargs=(closed_keys(), frozenset(extra["raw_path"]))
    ) as pool:
        for task, results in zip(pending, pool.imap(write_chunk, pending), strict=True):
            write_json_atomic(markers / f"{task['name']}.json", results)
        built = pd.DataFrame(
            [row for task in tasks for row in json.loads((markers / f"{task['name']}.json").read_text())]
        )
        reread = pd.DataFrame([row for part in pool.imap(verify_chunk, tasks) for row in part])
    built = built.sort_values("position").reset_index(drop=True)
    reread = (
        reread.sort_values("position").reset_index(drop=True).rename(columns={"sha256_f32": "reread_f32"})
    )
    extra = extra.merge(built, on="position", validate="one_to_one").merge(
        reread, on="position", validate="one_to_one"
    )
    ok = extra["sha256_read_ok"] & extra["sha256_float32_ok"] & extra["sha256_cache_ok"]
    if not ok.all() or not (extra["reread_f32"] == extra["sha256_f32"]).all():
        raise ValueError(f"{(~ok).sum()} supplement records do not reproduce their manifest hash")
    extra = extra.drop(columns="reread_f32")
    combined = pd.concat([base, extra[base.columns]], ignore_index=True)
    if not combined["position"].equals(pd.Series(np.arange(len(combined)))):
        raise ValueError("Combined index positions are not consecutive")
    combined.to_csv(directory / "index_all.csv", index=False)
    receipt = {
        "stage": "mlc02_waveform_cache_supplement",
        "command": command,
        "plan": plan,
        "records": len(extra),
        "base_records": len(base),
        "all_records": len(combined),
        "sha256_read_ok": int(extra["sha256_read_ok"].sum()),
        "sha256_float32_ok": int(extra["sha256_float32_ok"].sum()),
        "sha256_cache_ok": int(extra["sha256_cache_ok"].sum()),
        "records_with_missing_samples": int((extra["missing_samples_10s"] > 0).sum()),
        "missing_samples": int(extra["missing_samples_10s"].sum()),
        "records_with_flat_leads": int((extra["flat_leads_10s"] > 0).sum()),
        "counts": {
            f"{role}/{source}": int(size)
            for (role, source), size in extra.groupby(["role", "source"]).size().items()
        },
        "blocks": {
            name: {"records": size, "sha256": sha256_file(directory / f"{name}.npy")}
            for name, size in block_sizes(extra).items()
        },
        "index_all_sha256": sha256_file(directory / "index_all.csv"),
        "index_all_identity_sha256": index_sha256(combined),
        "wall_seconds": time.perf_counter() - started,
        "workers": workers,
        "git": git_state(),
        "code_sha256": code_sha256(CODE),
    }
    write_json_atomic(directory / "receipt_supplement.json", receipt)
    log(log_path, f"supplement finished: {len(extra)} records, {receipt['wall_seconds']:.1f} s")
    return receipt


def coverage(rows: pd.DataFrame) -> dict[str, dict[str, object]]:
    """
    Compare every MLC-01 build with the published rows: identity, receipt and coverage of its included set.

    Parameters
    ----------
    rows : pd.DataFrame
        Published rows with ``manifest_row``, ``record_key``, ``source``, ``role`` and ``sha256_10s``.

    Returns
    -------
    dict[str, dict[str, object]]
        Per build: manifest and receipt SHA-256, the receipt's git state, ``identity_equal``, ``included`` and
        ``included_covered`` (records of its included set present in ``rows``).
    """
    record = {}
    ordered = rows.sort_values("manifest_row").reset_index(drop=True)
    for name, path in BUILDS.items():
        if not path.exists():
            record[name] = {"present": False}
            continue
        candidates = read_candidates(path, CANDIDATE_RECORDS)
        columns = [column for column in COMPARED if column != "raw_path"]
        receipt = path.parent / "receipt.json"
        record[name] = {
            "present": True,
            "manifest": to_stored(path),
            "manifest_sha256": sha256_file(path),
            "receipt_sha256": sha256_file(receipt) if receipt.exists() else None,
            "receipt_git": json.loads(receipt.read_text())["git"] if receipt.exists() else None,
            "identity_equal": bool(candidates[columns].astype(str).equals(ordered[columns].astype(str))),
            "included": int(candidates["included"].sum()),
            "included_covered": int(
                candidates.loc[candidates["included"], "record_key"].isin(rows["record_key"]).sum()
            ),
        }
    return record
