"""
Records, features, the reference classifier and the test-role guard of KEY-05.

Labels are the v5 states (1.0, 0.0, NaN) of the 14 conditions. ``load_records`` reads the v5 manifest
with the rows of every other role skipped by the CSV reader, so labels of a role not asked for are never
parsed. The test roles (PTB-XL fold 10 and the SPH evaluation half) open only with ``allow_test=True`` and
only once the selection record is committed unchanged in Git (``selection_committed``). The SPH calibration
half never opens. Features are the MLC-02 pooled xECG embeddings, read by feature row after the rows are
matched to the v5 manifest by key, role, source and first-10-s hash.
"""

from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ecg_experiment import ROOT
from ecg_experiment.files import read_json, sha256_file
from ecg_experiment.mlc02_data import CONDITIONS, PATIENT_SOURCES, check_states
from ecg_experiment.mlc02_feature_models import load_candidate, predict_loaded

V5 = ROOT / "outputs/mlc01_condition_dataset_20261111_v5"
MANIFEST = V5 / "manifest.csv"
FEATURES = ROOT / "outputs/mlc02_condition_models_20261110/features"
P0_DIRECTORY = ROOT / "outputs/mlc03_data_quality_20261111/effect/candidates"
P0_VALIDATION = P0_DIRECTORY / "P0" / "validation__seeds-42.npz"
P0_STORED_METRIC = 0.9697697420282113
OUTPUT = ROOT / "outputs/key05_condition_keys_20261111"
SELECTION = ROOT / "docs/key05-selection-record-2026-11-11.md"
OPEN_ROLES = ("train", "validation")
TEST_ROLES = ("test_same_source", "test_external")
NEVER_ROLES = ("sph_calibration",)
STRUCTURE = ("record_key", "source", "family", "role", "origin", "group_id", "group_kind", "included", "sex")
FAMILIES = ("ptbxl", "chapman_ningbo", "georgia", "cpsc")


def selection_committed(path: Path | None = None) -> dict[str, Any]:
    """
    Read the committed selection record; refuse if it is missing, untracked or modified.

    Parameters
    ----------
    path : Path | None
        Selection record, a Markdown file with one fenced ``json`` block; None reads ``SELECTION``.

    Returns
    -------
    dict[str, Any]
        The JSON block, with ``commit`` (the last commit that touched the file).

    Raises
    ------
    PermissionError
        If the file is not tracked, differs from ``HEAD``, or holds no selection.
    """
    path = SELECTION if path is None else Path(path)
    relative = path.relative_to(ROOT).as_posix()
    tracked = subprocess.run(["git", "ls-files", "--error-unmatch", relative], cwd=ROOT, capture_output=True)
    if tracked.returncode != 0:
        raise PermissionError(f"{relative} is not committed; the test roles stay closed")
    changed = subprocess.run(["git", "diff", "--quiet", "HEAD", "--", relative], cwd=ROOT)
    if changed.returncode != 0:
        raise PermissionError(f"{relative} differs from HEAD; the test roles stay closed")
    match = re.search(r"```json\n(.*?)\n```", Path(path).read_text(), flags=re.DOTALL)
    if not match:
        raise PermissionError(f"{relative} holds no selection block")
    record = json.loads(match.group(1))
    if not {"selected_single", "selected_k7"} <= set(record):
        raise PermissionError(f"{relative} does not name both selections")
    commit = subprocess.check_output(["git", "log", "-1", "--format=%H", "--", relative], cwd=ROOT, text=True)
    return {**record, "commit": commit.strip()}


def check_roles(roles: tuple[str, ...], allow_test: bool = False) -> tuple[str, ...]:
    """
    Refuse roles this experiment may not read now.

    Parameters
    ----------
    roles : tuple[str, ...]
        Requested roles.
    allow_test : bool
        Whether the test roles are requested on purpose (the one test read).

    Returns
    -------
    tuple[str, ...]
        The roles.

    Raises
    ------
    PermissionError
        If the SPH calibration half is requested, or a test role without ``allow_test`` or before the
        selection record is committed.
    ValueError
        If a role is unknown or none is given.
    """
    roles = tuple(roles)
    if set(roles) & set(NEVER_ROLES):
        raise PermissionError("the SPH calibration half is not used by KEY-05")
    if set(roles) & set(TEST_ROLES):
        if not allow_test:
            raise PermissionError("test roles are filtered out unless the test read asks for them")
        selection_committed()
    unknown = set(roles) - set(OPEN_ROLES) - set(TEST_ROLES)
    if unknown or not roles:
        raise ValueError(f"unknown roles {sorted(unknown)}")
    return roles


def manifest_lines(roles: tuple[str, ...], path: Path = MANIFEST) -> set[int]:
    """
    File line numbers of the included v5 rows of some roles; only key, role and inclusion are read.

    Parameters
    ----------
    roles : tuple[str, ...]
        Roles already passed through ``check_roles``.
    path : Path
        v5 manifest.

    Returns
    -------
    set[int]
        One-based line numbers (line 0 is the header).
    """
    frame = pd.read_csv(path, usecols=["record_key", "role", "included"], dtype=str)
    keep = (frame["included"] == "True") & frame["role"].isin(roles)
    return set((np.flatnonzero(keep.to_numpy()) + 1).tolist())


def load_records(
    roles: tuple[str, ...] = OPEN_ROLES, allow_test: bool = False, path: Path = MANIFEST
) -> dict[str, np.ndarray]:
    """
    Identity and v5 labels of the included records of some roles, in manifest order.

    Parameters
    ----------
    roles : tuple[str, ...]
        Roles to read.
    allow_test : bool
        Passed to ``check_roles``.
    path : Path
        v5 manifest.

    Returns
    -------
    dict[str, np.ndarray]
        ``record_key``, ``source``, ``family``, ``role``, ``origin``, ``group`` (patient for PTB-XL and SPH,
        record otherwise), ``sex``, ``manifest_row`` and ``labels`` (``[records, 14]``).

    Raises
    ------
    ValueError
        If a row of another role was read, a label is not 1, 0 or NaN, or a bootstrap group does not match
        its source.
    """
    roles = check_roles(roles, allow_test)
    lines = manifest_lines(roles, path)
    columns = [f"b_{name}" for name in CONDITIONS]
    frame = pd.read_csv(
        path,
        usecols=[*STRUCTURE, *columns],
        skiprows=lambda i: i > 0 and i not in lines,
        dtype=dict.fromkeys(STRUCTURE, str),
        low_memory=False,
    )
    if not frame["role"].isin(roles).all() or not (frame["included"] == "True").all():
        raise ValueError("a row outside the requested roles was read")
    labels = frame[columns].to_numpy(dtype=np.float64)
    check_states(labels, "v5 space B")
    patient = frame["source"].isin(PATIENT_SOURCES)
    if (
        not (frame.loc[patient, "group_kind"] == "patient").all()
        or not (frame.loc[~patient, "group_id"] == frame.loc[~patient, "record_key"]).all()
    ):
        raise ValueError("bootstrap groups do not match their sources")
    text = {
        name: frame[name].to_numpy(dtype=str) for name in ("record_key", "source", "family", "role", "origin")
    }
    return {
        **text,
        "group": frame["group_id"].to_numpy(dtype=str),
        "sex": frame["sex"].to_numpy(dtype=str),
        "manifest_row": np.array(sorted(lines)) - 1,
        "labels": labels,
    }


def subset(data: dict[str, np.ndarray], mask: np.ndarray) -> dict[str, np.ndarray]:
    """
    Select records of ``data`` by a mask, in order.

    Parameters
    ----------
    data : dict[str, np.ndarray]
        Output of ``load_records``.
    mask : np.ndarray
        Boolean mask.

    Returns
    -------
    dict[str, np.ndarray]
        The same entries restricted to the mask.
    """
    return {name: values[mask] for name, values in data.items()}


def feature_rows(
    data: dict[str, np.ndarray], directory: Path = FEATURES, path: Path = MANIFEST
) -> np.ndarray:
    """
    Feature-array rows of the records, after matching ``rows.csv`` to the v5 manifest.

    Parameters
    ----------
    data : dict[str, np.ndarray]
        Output of ``load_records``.
    directory : Path
        Feature directory with ``rows.csv``.
    path : Path
        v5 manifest.

    Returns
    -------
    np.ndarray
        One feature row per record.

    Raises
    ------
    ValueError
        If a row's key, role, source or first-10-s hash differs from its manifest row, or a record has no row.
    """
    columns = ["record_key", "role", "source", "sha256_10s"]
    rows = pd.read_csv(
        Path(directory) / "rows.csv", usecols=["manifest_row", *columns], dtype=dict.fromkeys(columns, str)
    )
    manifest = pd.read_csv(path, usecols=columns, dtype=str)
    matched = manifest.iloc[rows["manifest_row"].to_numpy()].reset_index(drop=True)
    for name in columns:
        if not np.array_equal(matched[name].to_numpy(), rows[name].to_numpy()):
            raise ValueError(f"rows.csv {name} is not aligned with the v5 manifest")
    position = pd.Series(rows.index.to_numpy(), index=rows["manifest_row"].to_numpy())
    missing = set(data["manifest_row"]) - set(position.index)
    if missing:
        raise ValueError(f"{len(missing)} records have no feature row")
    index = position.loc[data["manifest_row"]].to_numpy()
    if not np.array_equal(rows["record_key"].to_numpy()[index], data["record_key"]):
        raise ValueError("feature rows hold other records")
    return index


def load_features(data: dict[str, np.ndarray], directory: Path = FEATURES) -> np.ndarray:
    """
    Raw pooled xECG embeddings of the records.

    Parameters
    ----------
    data : dict[str, np.ndarray]
        Output of ``load_records``.
    directory : Path
        Feature directory.

    Returns
    -------
    np.ndarray
        Float64 ``[records, 1024]``.
    """
    index = feature_rows(data, directory)
    array = np.load(Path(directory) / "xecg.npy", mmap_mode="r")
    order = np.argsort(index)
    out = np.empty((len(index), array.shape[1]))
    out[order] = np.asarray(array[index[order]], dtype=np.float64)
    return out


def feature_receipt(directory: Path = FEATURES) -> dict[str, Any]:
    """
    SHA-256 of the xECG array and ``rows.csv``, checked against the MLC-02 Stage 1 receipt.

    Parameters
    ----------
    directory : Path
        Feature directory holding ``receipt.json``.

    Returns
    -------
    dict[str, Any]
        ``xecg.npy``, ``rows.csv`` and ``equal`` (both match the receipt).

    Raises
    ------
    ValueError
        If a file differs from the receipt.
    """
    expected = read_json(Path(directory) / "receipt.json")["files_sha256"]
    found = {name: sha256_file(Path(directory) / name) for name in ("xecg.npy", "rows.csv")}
    if any(found[name] != expected[name] for name in found):
        raise ValueError("feature files differ from the MLC-02 Stage 1 receipt")
    return {**found, "equal": True}


def p0_model(directory: Path = P0_DIRECTORY) -> dict[str, Any]:
    """
    Load the v5-trained P0 of MLC-03 Part D, its artefacts checked against ``model.json``.

    Parameters
    ----------
    directory : Path
        Directory holding ``P0/model.json``.

    Returns
    -------
    dict[str, Any]
        ``load_candidate`` output.
    """
    return load_candidate("P0", directory)


def p0_probability(model: dict[str, Any], raw: np.ndarray) -> np.ndarray:
    """
    P0 probabilities of raw xECG rows.

    Parameters
    ----------
    model : dict[str, Any]
        Output of ``p0_model``.
    raw : np.ndarray
        ``[records, 1024]`` raw embeddings.

    Returns
    -------
    np.ndarray
        ``[records, 14]``.
    """
    return predict_loaded(model, {"xecg": raw})


def support_counts(data: dict[str, np.ndarray]) -> pd.DataFrame:
    """
    Positives, negatives and undefined labels per role, source and condition.

    Parameters
    ----------
    data : dict[str, np.ndarray]
        Output of ``load_records``.

    Returns
    -------
    pd.DataFrame
        ``role``, ``unit`` (source), ``label`` (``b_<condition>``), ``positive``, ``negative``, ``undefined``.
    """
    rows = []
    for role in np.unique(data["role"]):
        for source in np.unique(data["source"][data["role"] == role]):
            labels = data["labels"][(data["role"] == role) & (data["source"] == source)]
            for column, name in enumerate(CONDITIONS):
                rows.append(
                    {
                        "role": str(role),
                        "unit": str(source),
                        "label": f"b_{name}",
                        "positive": int((labels[:, column] == 1).sum()),
                        "negative": int((labels[:, column] == 0).sum()),
                        "undefined": int(np.isnan(labels[:, column]).sum()),
                    }
                )
    return pd.DataFrame(rows)


def check_support(data: dict[str, np.ndarray], directory: Path = V5) -> dict[str, Any]:
    """
    Compare label counts of the loaded roles with the v5 build's ``support_v4_v5.csv``.

    The support file's SHA-256 is first checked against the v5 receipt.

    Parameters
    ----------
    data : dict[str, np.ndarray]
        Output of ``load_records``.
    directory : Path
        v5 directory.

    Returns
    -------
    dict[str, Any]
        ``cells`` compared, ``differing`` cells, ``equal`` and the support file's ``sha256``.

    Raises
    ------
    ValueError
        If the support file differs from the receipt.
    """
    digest = sha256_file(Path(directory) / "support_v4_v5.csv")
    if digest != read_json(Path(directory) / "receipt.json")["outputs_sha256"]["support_v4_v5.csv"]:
        raise ValueError("support_v4_v5.csv differs from the v5 receipt")
    stored = pd.read_csv(Path(directory) / "support_v4_v5.csv")
    stored = stored[(stored["space"] == "B") & (stored["level"] == "source")]
    stored = stored[stored["role"].isin(np.unique(data["role"]))]
    found = support_counts(data)
    joined = stored.merge(found, on=["role", "unit", "label"], how="outer", indicator=True)
    differing = joined[
        (joined["_merge"] != "both")
        | (joined["positive"] != joined["positive_v5"])
        | (joined["negative"] != joined["negative_v5"])
        | (joined["undefined"] != joined["undefined_v5"])
    ]
    return {
        "cells": int(len(joined)),
        "differing": int(len(differing)),
        "equal": bool(differing.empty),
        "sha256": digest,
    }
