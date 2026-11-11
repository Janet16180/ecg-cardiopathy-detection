"""
Train and validation features of the MLC-02 feature-based candidates, behind a role guard.

Stage 2 fits on role ``train`` and scores on role ``validation`` only. The test roles (PTB-XL fold 10, the
SPH evaluation half and the SPH calibration half) stay closed until the selection record is committed, so
``load_split`` refuses them before it opens any file. Features come from the Stage 1 arrays, one row per
MLC-01 candidate in manifest order (``rows.csv``); a model uses the rows included in the final build
(``included_v4``). ``aligned_rows`` checks ``rows.csv`` against the manifest by record key, first-10-s hash,
role, source and inclusion.

Missing measurements (flat leads, no beats) are imputed with the training median plus a missing flag, as the
protocol states for H-All.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ecg_experiment import ROOT
from ecg_experiment.files import read_json, sha256_file
from ecg_experiment.mlc02_data import MANIFEST, load_manifest, subset

PUBLISHED = ROOT / "outputs/mlc02_condition_models_20261110/features"
LOCAL = Path("/var/tmp/janetrivera/mlc02/features/final")
FEATURE_FILES = {"xecg": "xecg.npy", "jepa": "jepa.npy", "cpc": "cpc.npy", "measurements": "measurements.npy"}
FEATURE_WIDTHS = {"xecg": 1024, "jepa": 768, "cpc": 512, "measurements": 130}
OPEN_ROLES = ("train", "validation")
CLOSED_ROLES = ("test_same_source", "test_external", "sph_calibration")
FAMILIES = ("ptbxl", "chapman_ningbo", "georgia", "cpsc")


def check_open_roles(roles: tuple[str, ...] | list[str]) -> tuple[str, ...]:
    """
    Refuse every role that Stage 2 may not read.

    Parameters
    ----------
    roles : tuple[str, ...] | list[str]
        Requested roles.

    Returns
    -------
    tuple[str, ...]
        The roles, when all are open.

    Raises
    ------
    PermissionError
        If a test role (``CLOSED_ROLES``) is requested.
    ValueError
        If no role or an unknown role is requested.
    """
    roles = tuple(roles)
    closed = sorted(set(roles) & set(CLOSED_ROLES))
    if closed:
        raise PermissionError(f"roles {closed} stay closed until the MLC-02 selection record is committed")
    unknown = sorted(set(roles) - set(OPEN_ROLES))
    if unknown or not roles:
        raise ValueError(f"Stage 2 reads roles {OPEN_ROLES} only, got {list(roles)}")
    return roles


def feature_hashes(directory: Path = LOCAL, published: Path = PUBLISHED) -> dict[str, object]:
    """
    SHA-256 of the feature files read, checked against the Stage 1 receipt.

    Parameters
    ----------
    directory : Path
        Directory the features are read from.
    published : Path
        Published Stage 1 directory holding ``receipt.json``.

    Returns
    -------
    dict[str, object]
        ``directory``, ``files_sha256`` and ``receipt_sha256``.

    Raises
    ------
    ValueError
        If a file differs from the published receipt.
    """
    expected = read_json(published / "receipt.json")["files_sha256"]
    names = [*FEATURE_FILES.values(), "rows.csv"]
    found = {name: sha256_file(Path(directory) / name) for name in names}
    differing = [name for name in names if found[name] != expected[name]]
    if differing:
        raise ValueError(f"feature files differ from the Stage 1 receipt: {differing}")
    return {
        "directory": str(directory),
        "files_sha256": found,
        "receipt_sha256": sha256_file(published / "receipt.json"),
    }


def aligned_rows(rows_path: Path, manifest_path: Path = MANIFEST) -> pd.DataFrame:
    """
    Read ``rows.csv`` and check it against the manifest.

    Parameters
    ----------
    rows_path : Path
        Stage 1 ``rows.csv``.
    manifest_path : Path
        MLC-01 ``manifest.csv``.

    Returns
    -------
    pd.DataFrame
        ``record_key``, ``role``, ``source``, ``sha256_10s`` and ``included_v4``, indexed by feature row.

    Raises
    ------
    ValueError
        If a row's key, first-10-s hash, role, source or inclusion differs from its manifest row.
    """
    columns = ["record_key", "role", "source", "sha256_10s"]
    rows = pd.read_csv(
        rows_path, usecols=["manifest_row", *columns, "included_v4"], dtype=dict.fromkeys(columns, str)
    )
    manifest = pd.read_csv(
        manifest_path, usecols=[*columns, "included"], dtype=dict.fromkeys(columns, str), low_memory=False
    )
    matched = manifest.iloc[rows["manifest_row"].to_numpy()].reset_index(drop=True)
    for name in columns:
        if not np.array_equal(matched[name].to_numpy(), rows[name].to_numpy()):
            raise ValueError(f"rows.csv {name} is not aligned with the manifest")
    if not np.array_equal(
        matched["included"].astype(bool).to_numpy(), rows["included_v4"].astype(bool).to_numpy()
    ):
        raise ValueError("rows.csv included_v4 differs from the manifest's inclusion")
    if rows["record_key"].duplicated().any():
        raise ValueError("rows.csv repeats a record key")
    return rows[[*columns, "included_v4"]]


def feature_positions(record_keys: np.ndarray, roles: np.ndarray, rows: pd.DataFrame) -> np.ndarray:
    """
    Feature-array rows of manifest records.

    Parameters
    ----------
    record_keys : np.ndarray
        Keys of included manifest records.
    roles : np.ndarray
        Their roles.
    rows : pd.DataFrame
        Output of ``aligned_rows``.

    Returns
    -------
    np.ndarray
        One feature row per record.

    Raises
    ------
    ValueError
        If a record is missing, not included in ``rows.csv``, or has another role there.
    """
    included = rows[rows["included_v4"].astype(bool)]
    position = pd.Series(included.index.to_numpy(), index=included["record_key"].to_numpy())
    missing = set(record_keys) - set(position.index)
    if missing:
        raise ValueError(f"{len(missing)} records have no included feature row")
    index = position.loc[record_keys].to_numpy()
    if not np.array_equal(rows["role"].to_numpy()[index], roles):
        raise ValueError("feature rows hold other roles than the manifest records")
    return index


def load_split(
    roles: tuple[str, ...], names: tuple[str, ...], directory: Path = LOCAL, manifest_path: Path = MANIFEST
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """
    Manifest arrays and float64 features of the included records of open roles.

    The role guard runs before any file is opened. Feature arrays are memory-mapped and only the requested
    rows are read.

    Parameters
    ----------
    roles : tuple[str, ...]
        Roles to load; ``train`` and ``validation`` only.
    names : tuple[str, ...]
        Feature sets, keys of ``FEATURE_FILES``.
    directory : Path
        Feature directory holding the arrays and ``rows.csv``.
    manifest_path : Path
        MLC-01 ``manifest.csv``.

    Returns
    -------
    tuple[dict[str, np.ndarray], dict[str, np.ndarray]]
        The ``load_manifest`` arrays of those records in manifest order, with ``feature_row``; and one
        float64 ``[records, width]`` matrix per feature set.

    Raises
    ------
    PermissionError
        If a closed role is requested.
    ValueError
        If a feature set is unknown, rows are misaligned, or an array has another shape.
    """
    roles = check_open_roles(roles)
    unknown = sorted(set(names) - set(FEATURE_FILES))
    if unknown:
        raise ValueError(f"unknown feature sets {unknown}")
    data = load_manifest(manifest_path)
    data = subset(data, np.isin(data["role"], roles))
    rows = aligned_rows(Path(directory) / "rows.csv", manifest_path)
    index = feature_positions(data["record_key"], data["role"], rows)
    features = {}
    for name in names:
        array = np.load(Path(directory) / FEATURE_FILES[name], mmap_mode="r")
        if array.shape != (len(rows), FEATURE_WIDTHS[name]):
            raise ValueError(f"{name} has shape {array.shape}")
        features[name] = np.asarray(array[index], dtype=np.float64)
    return {**data, "feature_row": index}, features


def select(
    data: dict[str, np.ndarray], features: dict[str, np.ndarray], mask: np.ndarray
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """
    Select records of a split by a mask, with their features.

    Parameters
    ----------
    data : dict[str, np.ndarray]
        Manifest arrays of ``load_split``.
    features : dict[str, np.ndarray]
        Feature matrices of the same records.
    mask : np.ndarray
        Boolean mask over the records.

    Returns
    -------
    tuple[dict[str, np.ndarray], dict[str, np.ndarray]]
        Both restricted to the mask, in order.
    """
    return subset(data, mask), {name: values[mask] for name, values in features.items()}


def fit_imputation(measurements: np.ndarray) -> np.ndarray:
    """
    Training medians of the measurements, ignoring missing values.

    Parameters
    ----------
    measurements : np.ndarray
        ``[training records, 130]`` measurements with NaN where unavailable.

    Returns
    -------
    np.ndarray
        One median per column.

    Raises
    ------
    ValueError
        If a column has no value or a value is infinite.
    """
    measurements = np.asarray(measurements, dtype=np.float64)
    if np.isinf(measurements).any():
        raise ValueError("measurements must be finite or NaN")
    if np.isnan(measurements).all(axis=0).any():
        raise ValueError("a measurement has no training value")
    return np.nanmedian(measurements, axis=0)


def impute(measurements: np.ndarray, medians: np.ndarray) -> np.ndarray:
    """
    Measurements with missing values set to the training median, followed by one missing flag per column.

    Parameters
    ----------
    measurements : np.ndarray
        ``[records, k]`` measurements with NaN where unavailable.
    medians : np.ndarray
        Output of ``fit_imputation``.

    Returns
    -------
    np.ndarray
        ``[records, 2 k]``: imputed values, then flags (1.0 where the value was missing).
    """
    measurements = np.asarray(measurements, dtype=np.float64)
    missing = np.isnan(measurements)
    return np.hstack([np.where(missing, medians[None, :], measurements), missing.astype(np.float64)])
