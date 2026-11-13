"""
Manifest arrays and prediction files of MLC-02.

The manifest is the final MLC-01 build. Only structural and label-state columns are read; no waveform is
opened. Label states are 1.0 (positive), 0.0 (negative) and NaN (undefined), in the protocol's order of the
14 space-B conditions (``CONDITIONS``) or the four space-A groups (``GROUPS_A``). ``load_manifest`` keeps the
included records only.

A prediction file is an ``.npz`` archive written by ``save_predictions`` and read by ``load_predictions``:

- ``format``: ``PREDICTION_FORMAT``;
- ``record_key``: the keys of one role's records, in manifest order;
- ``condition``: the label space's names, in protocol order;
- ``probability``: float64 ``[records, conditions]``, every value finite and in [0, 1];
- ``model_id``: the candidate id of the MLC-02 protocol table (for example ``P0`` or ``H-All``);
- ``seeds``: the training seeds whose mean probability the file holds; one seed for a single run.

``prediction_path`` names a file ``<directory>/<model_id>/<role>__seeds-<s1>-<s2>-<s3>.npz``.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

from ecg_experiment import ROOT, mlc_labels
from ecg_experiment.files import write_npz_atomic

MANIFEST = ROOT / "outputs/mlc01_condition_dataset_20261110_v4/manifest.csv"
CONDITIONS = ("sinus_tachycardia", "sinus_bradycardia", "sinus_arrhythmia", "af_or_flutter",
              "premature_ventricular_complex", "first_degree_av_block", "complete_rbbb", "incomplete_rbbb",
              "lafb", "left_axis_deviation", "lvh", "st_depression", "t_wave_abnormality", "mi_any")
GROUPS_A = ("MI", "STTC", "CD", "HYP")
ROLES = ("train", "validation", "test_same_source", "test_external", "sph_calibration")
CHALLENGE_SOURCES = ("ningbo", "chapman_shaoxing", "georgia", "cpsc_2018", "cpsc_2018_extra")
VALIDATION_SOURCES = ("ptbxl", *CHALLENGE_SOURCES)
ROLE_SOURCES = {"train": frozenset(VALIDATION_SOURCES), "validation": frozenset(VALIDATION_SOURCES),
                "test_same_source": frozenset({"ptbxl"}), "test_external": frozenset({"sph"}),
                "sph_calibration": frozenset({"sph"})}
ROLE_ORIGINS = {"validation": frozenset({"fold_9", "calibration"}),
                "test_same_source": frozenset({"fold_10"}),
                "test_external": frozenset({"half_evaluation"}),
                "sph_calibration": frozenset({"half_calibration"})}
PATIENT_SOURCES = frozenset({"ptbxl", "sph"})
STRUCTURAL = ("record_key", "source", "family", "role", "origin", "group_id", "group_kind", "included", "sex",
              "n_flat_leads", "n_missing_leads")
PREDICTION_FORMAT = "mlc02-predictions-v1"
SAFE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.+-]*$")


def label_columns() -> dict[str, list[str]]:
    """
    Manifest columns of each label matrix.

    Returns
    -------
    dict[str, list[str]]
        ``labels`` (space B), ``labels_a`` (space A) and ``lik50`` (space-B likelihood flags).

    Raises
    ------
    ValueError
        If the frozen label code does not list the 14 conditions and four groups in protocol order.
    """
    if tuple(mlc_labels.SPACE_B) != CONDITIONS or len(CONDITIONS) != 14:
        raise ValueError("mlc_labels.SPACE_B is not the protocol's 14 conditions in order")
    if tuple(mlc_labels.SPACE_A) != GROUPS_A:
        raise ValueError("mlc_labels.SPACE_A is not the protocol's four groups in order")
    return {"labels": [f"b_{name}" for name in CONDITIONS], "labels_a": [f"a_{name}" for name in GROUPS_A],
            "lik50": [f"lik50_{name}" for name in CONDITIONS]}


def check_roles(frame: pd.DataFrame) -> None:
    """
    Check roles, sources and origins of the included records.

    Parameters
    ----------
    frame : pd.DataFrame
        Included manifest rows.

    Raises
    ------
    ValueError
        If a key repeats, a role is unknown, a role holds other sources than the protocol's, or a test or
        validation record has another origin than its role's.
    """
    if frame["record_key"].duplicated().any():
        raise ValueError("record keys repeat in the manifest")
    unknown = set(frame["role"]) - set(ROLES)
    if unknown:
        raise ValueError(f"unknown roles {sorted(unknown)}")
    for role, sources in frame.groupby("role")["source"]:
        if set(sources) != ROLE_SOURCES[role]:
            raise ValueError(f"role {role} holds sources {sorted(set(sources))}")
    for role, origins in ROLE_ORIGINS.items():
        found = set(frame.loc[frame["role"] == role, "origin"])
        if not found <= origins:
            raise ValueError(f"role {role} holds origins {sorted(found - origins)}")


def check_groups(frame: pd.DataFrame) -> None:
    """
    Check bootstrap groups: patients for PTB-XL and SPH, records for the Challenge sources.

    Parameters
    ----------
    frame : pd.DataFrame
        Included manifest rows.

    Raises
    ------
    ValueError
        If a group kind does not match its source, a record group is not the record key, or a patient
        appears in two roles.
    """
    patient = frame["source"].isin(PATIENT_SOURCES)
    if not (frame.loc[patient, "group_kind"] == "patient").all():
        raise ValueError("PTB-XL and SPH records must be grouped by patient")
    records = frame.loc[~patient]
    if not ((records["group_kind"] == "record") & (records["group_id"] == records["record_key"])).all():
        raise ValueError("Challenge records must be their own group")
    if (frame.loc[patient].groupby("group_id")["role"].nunique() > 1).any():
        raise ValueError("a patient appears in two roles")


def check_states(states: np.ndarray, name: str) -> None:
    """
    Check that a label matrix holds only 1.0, 0.0 and NaN.

    Parameters
    ----------
    states : np.ndarray
        Label states.
    name : str
        Matrix name for the error message.

    Raises
    ------
    ValueError
        If any other value occurs.
    """
    if not np.isin(states[~np.isnan(states)], (0.0, 1.0)).all():
        raise ValueError(f"{name} holds values other than 1, 0 and NaN")


def load_manifest(path: Path = MANIFEST) -> dict[str, np.ndarray]:
    """
    Arrays of the included manifest records.

    Parameters
    ----------
    path : Path
        MLC-01 ``manifest.csv``.

    Returns
    -------
    dict[str, np.ndarray]
        One entry per record, in manifest order: ``record_key``, ``source``, ``family``, ``role``, ``origin``,
        ``group`` (bootstrap group: patient for PTB-XL and SPH, record otherwise), ``sex`` (``F``, ``M`` or
        ``unknown``), ``lead_flag`` (a flat or missing lead in the first 10 s), ``labels`` ([n, 14] space B),
        ``labels_lik50`` (space B with ``lik50_`` positives undefined) and ``labels_a`` ([n, 4] space A).

    Raises
    ------
    ValueError
        If any check of ``check_roles``, ``check_groups`` or ``check_states`` fails, or a likelihood flag
        marks anything but a PTB-XL positive.
    """
    columns = label_columns()
    usecols = [*STRUCTURAL, *columns["labels"], *columns["labels_a"], *columns["lik50"]]
    frame = pd.read_csv(path, usecols=usecols,
                        dtype={"record_key": str, "group_id": str, "sex": str}, low_memory=False)
    frame = frame[frame["included"].astype(bool)].reset_index(drop=True)
    check_roles(frame)
    check_groups(frame)
    labels = frame[columns["labels"]].to_numpy(dtype=np.float64)
    labels_a = frame[columns["labels_a"]].to_numpy(dtype=np.float64)
    flags = frame[columns["lik50"]].to_numpy(dtype=bool)
    check_states(labels, "space B")
    check_states(labels_a, "space A")
    if (flags & ((labels != 1.0) | (frame["source"].to_numpy() != "ptbxl")[:, None])).any():
        raise ValueError("a likelihood flag marks a record that is not a PTB-XL positive")
    text = {name: frame[name].to_numpy(dtype=str)
            for name in ("record_key", "source", "family", "role", "origin")}
    return {**text, "group": frame["group_id"].to_numpy(dtype=str),
            "sex": frame["sex"].where(frame["sex"].isin(["F", "M"]), "unknown").to_numpy(dtype=str),
            "lead_flag": ((frame["n_flat_leads"] > 0) | (frame["n_missing_leads"] > 0)).to_numpy(),
            "labels": labels, "labels_lik50": np.where(flags, np.nan, labels), "labels_a": labels_a}


def subset(data: dict[str, np.ndarray], mask: np.ndarray) -> dict[str, np.ndarray]:
    """
    Select records of ``data`` by a mask, keeping their order.

    Parameters
    ----------
    data : dict[str, np.ndarray]
        Output of ``load_manifest`` or of this function.
    mask : np.ndarray
        Boolean mask over the records.

    Returns
    -------
    dict[str, np.ndarray]
        The same entries, restricted to the mask.
    """
    return {name: values[mask] for name, values in data.items()}


def role_data(data: dict[str, np.ndarray], role: str) -> dict[str, np.ndarray]:
    """
    Select the records of one role.

    Parameters
    ----------
    data : dict[str, np.ndarray]
        Output of ``load_manifest``.
    role : str
        One of ``ROLES``.

    Returns
    -------
    dict[str, np.ndarray]
        The role's records in manifest order.

    Raises
    ------
    ValueError
        If the role is unknown or empty.
    """
    if role not in ROLES:
        raise ValueError(f"unknown role {role}")
    mask = data["role"] == role
    if not mask.any():
        raise ValueError(f"role {role} has no record")
    return subset(data, mask)


def check_probabilities(probability: np.ndarray, records: int, conditions: int) -> np.ndarray:
    """
    Check the shape and range of a probability matrix.

    Parameters
    ----------
    probability : np.ndarray
        Candidate probabilities.
    records, conditions : int
        Expected shape.

    Returns
    -------
    np.ndarray
        The matrix as float64.

    Raises
    ------
    ValueError
        If the shape differs or a value is not finite or outside [0, 1].
    """
    probability = np.asarray(probability, dtype=np.float64)
    if probability.shape != (records, conditions):
        raise ValueError(f"probabilities have shape {probability.shape}, expected {(records, conditions)}")
    if not np.isfinite(probability).all() or (probability < 0).any() or (probability > 1).any():
        raise ValueError("probabilities must be finite and in [0, 1]")
    return probability


def check_identity(model_id: str, seeds: tuple[int, ...]) -> np.ndarray:
    """
    Check a model id and its seeds.

    Parameters
    ----------
    model_id : str
        Candidate id; letters, digits and ``_.+-`` only.
    seeds : tuple[int, ...]
        Nonempty, distinct training seeds.

    Returns
    -------
    np.ndarray
        The seeds as int64.

    Raises
    ------
    ValueError
        If the id is not safe for a file name or the seeds are empty or repeat.
    """
    if not SAFE_ID.match(model_id):
        raise ValueError(f"model id {model_id!r} must use letters, digits and _.+- only")
    seeds_array = np.asarray(seeds, dtype=np.int64)
    if seeds_array.ndim != 1 or not len(seeds_array) or len(set(seeds_array.tolist())) != len(seeds_array):
        raise ValueError("seeds must be a nonempty sequence of distinct integers")
    return seeds_array


def prediction_path(directory: Path, model_id: str, role: str, seeds: tuple[int, ...]) -> Path:
    """
    File name of one prediction file.

    Parameters
    ----------
    directory : Path
        Prediction directory of the experiment.
    model_id : str
        Candidate id.
    role : str
        One of ``ROLES``.
    seeds : tuple[int, ...]
        Training seeds.

    Returns
    -------
    Path
        ``<directory>/<model_id>/<role>__seeds-<s1>-...npz``.
    """
    if role not in ROLES:
        raise ValueError(f"unknown role {role}")
    joined = "-".join(str(seed) for seed in check_identity(model_id, seeds))
    return Path(directory) / model_id / f"{role}__seeds-{joined}.npz"


def save_predictions(path: Path, record_keys: np.ndarray, probability: np.ndarray, model_id: str,
                     seeds: tuple[int, ...], conditions: tuple[str, ...] = CONDITIONS) -> None:
    """
    Write one prediction file.

    Parameters
    ----------
    path : Path
        Destination ``.npz``.
    record_keys : np.ndarray
        Record keys, in the manifest order of the role.
    probability : np.ndarray
        ``[records, conditions]`` probabilities.
    model_id : str
        Candidate id.
    seeds : tuple[int, ...]
        Training seeds whose mean probability this is.
    conditions : tuple[str, ...]
        Label space, ``CONDITIONS`` or ``GROUPS_A``.

    Raises
    ------
    ValueError
        If keys repeat, or any check of ``check_probabilities`` or ``check_identity`` fails.
    """
    keys = np.asarray(record_keys, dtype=str)
    if len(np.unique(keys)) != len(keys):
        raise ValueError("record keys repeat")
    probability = check_probabilities(probability, len(keys), len(conditions))
    write_npz_atomic(path, format=np.array(PREDICTION_FORMAT), record_key=keys,
                     condition=np.asarray(conditions, dtype=str), probability=probability,
                     model_id=np.array(model_id), seeds=check_identity(model_id, seeds))


def load_predictions(path: Path, record_keys: np.ndarray,
                     conditions: tuple[str, ...] = CONDITIONS) -> dict[str, object]:
    """
    Read one prediction file and check that it is aligned with the manifest.

    Parameters
    ----------
    path : Path
        File written by ``save_predictions``.
    record_keys : np.ndarray
        The manifest keys the file must hold, in the same order.
    conditions : tuple[str, ...]
        Expected label space.

    Returns
    -------
    dict[str, object]
        ``probability`` (float64 array), ``model_id`` (str) and ``seeds`` (tuple of int).

    Raises
    ------
    ValueError
        If the format, conditions or record keys (count or order) differ, or the probabilities fail
        ``check_probabilities``.
    """
    with np.load(path, allow_pickle=False) as archive:
        content = {name: archive[name] for name in archive.files}
    if str(content["format"]) != PREDICTION_FORMAT:
        raise ValueError(f"{path} is not an {PREDICTION_FORMAT} file")
    if tuple(content["condition"].tolist()) != tuple(conditions):
        raise ValueError(f"{path} conditions differ from the expected label space")
    keys = np.asarray(record_keys, dtype=str)
    if content["record_key"].shape != keys.shape or not np.array_equal(content["record_key"], keys):
        raise ValueError(f"{path} record keys are not aligned with the manifest")
    model_id = str(content["model_id"])
    seeds = check_identity(model_id, tuple(content["seeds"].tolist()))
    return {"probability": check_probabilities(content["probability"], len(keys), len(conditions)),
            "model_id": model_id, "seeds": tuple(int(seed) for seed in seeds)}
