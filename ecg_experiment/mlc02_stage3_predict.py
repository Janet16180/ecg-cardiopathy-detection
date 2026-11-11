"""
Stage 3 predictions of MLC-02: every candidate of the selection record on one role, once.

``run_test_predict`` scores PTB-XL fold 10, the SPH evaluation half and the SPH calibration half. It refuses
to start unless all of these hold:

- the ``--i-am-the-final-run`` flag is given;
- no test prediction file exists anywhere under the experiment's outputs (it runs once);
- the selection record exists, is not provisional, and its Markdown document is committed, unmodified, and
  equal to the text generated from it (``mlc02_stage3.check_record_committed``);
- the operating points exist and were fitted for that record;
- the code is committed and every validation file still has the SHA-256 the record holds.

All predictions are computed in memory first and written only when every candidate and role has succeeded,
so a run that fails leaves no test file and can be repeated; once files exist, the step is closed.

``run_dry_run`` takes the same code path (``role_predictions``) on the validation role, writes to a scratch
directory, and compares every candidate with its saved validation predictions.

Inputs per candidate kind:

- feature candidates read their Stage 1 rows (``mlc02_feature_data.LOCAL``, checked against the Stage 1
  receipt and the manifest) and call ``mlc02_feature_models.predict``;
- S-R and F read the cached first-10-s waveforms of the role's blocks only, check each window against its
  manifest hash, and call ``mlc02_scratch_network.predict`` or ``mlc02_finetune.predict`` (imported only when
  F is in the record);
- E averages its members, in protocol-table order.
"""

from __future__ import annotations

import importlib
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ecg_experiment.files import read_json, sha256_file, write_json_atomic
from ecg_experiment.gpu import GPU_LOCK_PATH, gpu_lock
from ecg_experiment.mlc01_dataset import array_sha256, microvolts
from ecg_experiment.mlc02_data import MANIFEST, load_manifest, prediction_path, role_data, save_predictions
from ecg_experiment.mlc02_feature_data import (
    FEATURE_FILES,
    FEATURE_WIDTHS,
    LOCAL,
    aligned_rows,
    feature_hashes,
)
from ecg_experiment.mlc02_feature_data import feature_positions as feature_rows
from ecg_experiment.mlc02_stage3 import (
    BASE_CANDIDATES,
    CANDIDATES_DIR,
    ENSEMBLE,
    OUTPUT,
    PREDICTIONS,
    RECORD_DOC,
    STAGE3,
    TEST_RECEIPT_NAME,
    TEST_ROLES,
    check_record_committed,
    code_state,
    ensemble_probability,
    json_ready,
    read_operating_points,
    read_record,
    receipt,
    refuse_after_tests,
    stored,
    validation_data,
    validation_probabilities,
)
from ecg_experiment.mlc02_waveforms import CACHE, SAMPLES

FEATURE_CANDIDATES = ("P0", "L-J", "L-C", "G-M", "H-X", "H-J", "H-C", "H-All")
WAVEFORM_CANDIDATES = ("S-R", "F")
DRY_RUN_TOLERANCE = 1e-6
DRY_RUN_SEED = 20261110
CHUNK = 2048
GPU_NEEDED_MIB = 3072
LEADS = 12


def candidate_inputs(candidate: str) -> tuple[str, ...]:
    """
    Feature sets a feature candidate reads.

    Parameters
    ----------
    candidate : str
        One of ``FEATURE_CANDIDATES``.

    Returns
    -------
    tuple[str, ...]
        Keys of ``mlc02_feature_data.FEATURE_FILES``.
    """
    from ecg_experiment.mlc02_feature_models import INPUTS

    return INPUTS[candidate]


def feature_inputs(
    record_keys: np.ndarray,
    roles: np.ndarray,
    names: tuple[str, ...],
    directory: Path = LOCAL,
    manifest: Path = MANIFEST,
) -> dict[str, np.ndarray]:
    """
    Float64 Stage 1 features of the given records, rows checked against the manifest.

    The arrays hold one row per MLC-01 candidate record; only the requested rows are read from the memory map.

    Parameters
    ----------
    record_keys : np.ndarray
        Keys of included records, in the order wanted.
    roles : np.ndarray
        Their manifest roles (checked against ``rows.csv``).
    names : tuple[str, ...]
        Feature sets.
    directory : Path
        Feature directory holding the arrays and ``rows.csv``.
    manifest : Path
        MLC-01 ``manifest.csv``.

    Returns
    -------
    dict[str, np.ndarray]
        One ``[records, width]`` matrix per feature set.

    Raises
    ------
    ValueError
        If rows are misaligned or an array has another shape.
    """
    rows = aligned_rows(Path(directory) / "rows.csv", manifest)
    index = feature_rows(np.asarray(record_keys, dtype=str), np.asarray(roles, dtype=str), rows)
    features = {}
    for name in names:
        array = np.load(Path(directory) / FEATURE_FILES[name], mmap_mode="r")
        if array.shape != (len(rows), FEATURE_WIDTHS[name]):
            raise ValueError(f"{name} has shape {array.shape}")
        features[name] = np.asarray(array[index], dtype=np.float64)
        del array
    return features


def waveform_rows(record_keys: np.ndarray, role: str) -> pd.DataFrame:
    """
    Cache block, row and manifest hash of records of one role, in the order given.

    Parameters
    ----------
    record_keys : np.ndarray
        Keys of included records of ``role``.
    role : str
        Their role.

    Returns
    -------
    pd.DataFrame
        ``record_key``, ``role``, ``block``, ``row`` and ``sha256_10s``.

    Raises
    ------
    ValueError
        If a key is not an included record of the role.
    """
    from ecg_experiment.mlc02_scratch_network import read_records

    records = read_records()
    records = records[records["role"] == role].set_index("record_key")
    missing = set(np.asarray(record_keys, dtype=str)) - set(records.index)
    if missing:
        raise ValueError(f"{len(missing)} keys are not included records of role {role}")
    return records.loc[np.asarray(record_keys, dtype=str)].reset_index()[
        ["record_key", "role", "block", "row", "sha256_10s"]
    ]


def open_role_blocks(rows: pd.DataFrame, role: str, cache: Path = CACHE) -> dict[str, np.ndarray]:
    """
    Memory-map the cache blocks of one role, read-only, refusing any block or record of another role.

    The role check runs before any block file is opened.

    Parameters
    ----------
    rows : pd.DataFrame
        Output of ``waveform_rows``.
    role : str
        The only role allowed.
    cache : Path
        Waveform cache directory.

    Returns
    -------
    dict[str, np.ndarray]
        Float32 ``[records, 12, 5000]`` arrays by block name.

    Raises
    ------
    PermissionError
        If a record or block belongs to another role.
    ValueError
        If a block has another dtype or shape.
    """
    from ecg_experiment.mlc02_scratch_network import block_role

    blocks = list(dict.fromkeys(rows["block"]))
    if set(rows["role"]) != {role} or any(block_role(block) != role for block in blocks):
        raise PermissionError(
            f"waveforms of role {role} only; got roles {sorted(set(rows['role']))} and blocks {blocks}"
        )
    arrays = {}
    for block in blocks:
        array = np.load(Path(cache) / f"{block}.npy", mmap_mode="r")
        if array.dtype != np.float32 or array.shape[1:] != (LEADS, SAMPLES):
            raise ValueError(f"unexpected cache block {block}: {array.shape} {array.dtype}")
        arrays[block] = array
    return arrays


def checked_windows(rows: pd.DataFrame, blocks: dict[str, np.ndarray]) -> np.ndarray:
    """
    Read cached windows and check each against its manifest first-10-s hash.

    Parameters
    ----------
    rows : pd.DataFrame
        Rows of ``waveform_rows``.
    blocks : dict[str, np.ndarray]
        Output of ``open_role_blocks``.

    Returns
    -------
    np.ndarray
        Float32 ``[records, 12, 5000]``, NaN where WFDB read a missing sample.

    Raises
    ------
    ValueError
        If a window does not reproduce its hash.
    """
    windows = np.stack(
        [np.asarray(blocks[block][row]) for block, row in zip(rows["block"], rows["row"], strict=True)]
    )
    for window, expected, key in zip(windows, rows["sha256_10s"], rows["record_key"], strict=True):
        if array_sha256(microvolts(window.T)) != expected:
            raise ValueError(f"cached window of {key} does not reproduce its manifest hash")
    return windows


def waveform_predictor(candidate: str, seeds: list[int], directory: Path, device: str) -> Any:
    """
    Build the function from cached windows to probabilities of S-R or F, after checking the saved weights.

    Parameters
    ----------
    candidate : str
        ``S-R`` or ``F``.
    seeds : list[int]
        Seeds of the candidate in the selection record.
    directory : Path
        Stage 2 candidate directory holding ``<id>/``.
    device : str
        Torch device.

    Returns
    -------
    Any
        Callable taking float32 ``[n, 12, 5000]`` windows and returning ``[n, 14]`` probabilities.

    Raises
    ------
    ValueError
        If the candidate's seeds or weight files differ from what was recorded.
    """
    folder = Path(directory) / candidate
    if candidate == "S-R":
        from ecg_experiment import mlc02_scratch_network

        settings = read_json(folder / "settings.json")
        if [int(run["seed"]) for run in settings["seeds"]] != list(seeds):
            raise ValueError("S-R seeds differ from the selection record")
        for run in settings["seeds"]:
            if sha256_file(folder / run["checkpoint"]) != run["sha256"]:
                raise ValueError(f"S-R checkpoint {run['checkpoint']} differs from settings.json")
        return lambda windows: mlc02_scratch_network.predict(windows, folder, device)
    finetune = importlib.import_module("ecg_experiment.mlc02_finetune")
    if list(finetune.checked_index(folder)["seeds"]) != list(seeds):
        raise ValueError("F seeds differ from the selection record")
    return lambda windows: finetune.predict(windows, folder, device=device)


def feature_predictions(
    names: list[str],
    records: dict[str, np.ndarray],
    seeds: dict[str, list[int]],
    candidates_dir: Path,
    feature_dir: Path,
    manifest: Path,
) -> dict[str, np.ndarray]:
    """
    Probabilities of the feature candidates for one role's records.

    Parameters
    ----------
    names : list[str]
        Feature candidates to score.
    records : dict[str, np.ndarray]
        Manifest arrays of the records.
    seeds : dict[str, list[int]]
        Candidate seeds in the selection record (checked against ``model.json``).
    candidates_dir : Path
        Stage 2 candidate directory.
    feature_dir : Path
        Stage 1 feature directory.
    manifest : Path
        MLC-01 ``manifest.csv``.

    Returns
    -------
    dict[str, np.ndarray]
        Candidate id to ``[records, 14]``.
    """
    from ecg_experiment.mlc02_feature_models import load_candidate, predict_loaded

    needed = tuple(dict.fromkeys(name for candidate in names for name in candidate_inputs(candidate)))
    features = feature_inputs(records["record_key"], records["role"], needed, feature_dir, manifest)
    result = {}
    for candidate in names:
        model = load_candidate(candidate, candidates_dir)
        if list(model["seeds"]) != list(seeds[candidate]):
            raise ValueError(f"{candidate} seeds differ from the selection record")
        result[candidate] = predict_loaded(
            model, {name: features[name] for name in candidate_inputs(candidate)}
        )
    return result


def role_predictions(
    record: dict[str, Any],
    role: str,
    records: dict[str, np.ndarray],
    device: str,
    candidates_dir: Path = CANDIDATES_DIR,
    feature_dir: Path = LOCAL,
    cache: Path = CACHE,
    manifest: Path = MANIFEST,
    waveform_subset: np.ndarray | None = None,
    lock: Path = GPU_LOCK_PATH,
    chunk: int = CHUNK,
) -> dict[str, dict[str, Any]]:
    """
    Probabilities of every candidate of the selection record on the records of one role.

    Parameters
    ----------
    record : dict[str, Any]
        Selection record.
    role : str
        Role of every record.
    records : dict[str, np.ndarray]
        Manifest arrays of the role's records, in manifest order.
    device : str
        Torch device of S-R and F.
    candidates_dir : Path
        Stage 2 candidate directory.
    feature_dir : Path
        Stage 1 feature directory.
    cache : Path
        Waveform cache directory.
    manifest : Path
        MLC-01 ``manifest.csv``.
    waveform_subset : np.ndarray | None
        Sorted record positions for S-R and F (dry runs on the CPU); None scores every record.
    lock : Path
        GPU lock file, taken when ``device`` is ``cuda``.
    chunk : int
        Records per waveform read.

    Returns
    -------
    dict[str, dict[str, Any]]
        Candidate id to ``rows`` (positions in ``records``), ``probability`` and ``seconds``.

    Raises
    ------
    ValueError
        If a record of another role is given.
    """
    if set(records["role"]) != {role}:
        raise ValueError(f"records must all belong to role {role}")
    seeds = {row["id"]: row["seeds"] for row in record["candidates"]}
    every = np.arange(len(records["record_key"]))
    result: dict[str, dict[str, Any]] = {}
    features = [name for name in FEATURE_CANDIDATES if name in seeds]
    if features:
        started = time.time()
        probabilities = feature_predictions(features, records, seeds, candidates_dir, feature_dir, manifest)
        for name in features:
            result[name] = {
                "rows": every,
                "probability": probabilities[name],
                "seconds": (time.time() - started) / len(features),
            }
    waveforms = [name for name in WAVEFORM_CANDIDATES if name in seeds]
    if waveforms:
        subset = every if waveform_subset is None else np.asarray(waveform_subset)
        rows = waveform_rows(records["record_key"][subset], role)
        with gpu_lock(device, path=lock):
            if device == "cuda":
                from ecg_experiment.mlc02_scratch_runs import require_free_gpu

                require_free_gpu(GPU_NEEDED_MIB)
            blocks = open_role_blocks(rows, role, cache)
            for name in waveforms:
                started = time.time()
                predictor = waveform_predictor(name, seeds[name], candidates_dir, device)
                parts = [
                    predictor(checked_windows(rows.iloc[low : low + chunk], blocks))
                    for low in range(0, len(rows), chunk)
                ]
                result[name] = {
                    "rows": subset,
                    "probability": np.concatenate(parts),
                    "seconds": time.time() - started,
                }
            del blocks
    if ENSEMBLE in seeds:
        members = record["ensemble"]["members"]
        common = every
        for name in members:
            common = np.intersect1d(common, result[name]["rows"])
        aligned = {
            name: result[name]["probability"][np.searchsorted(result[name]["rows"], common)]
            for name in members
        }
        result[ENSEMBLE] = {
            "rows": common,
            "probability": ensemble_probability(aligned, members),
            "seconds": 0.0,
        }
    unknown = set(seeds) - set(result)
    if unknown:
        raise ValueError(f"no predictor for {sorted(unknown)}")
    return {name: result[name] for name in (*BASE_CANDIDATES, ENSEMBLE) if name in result}


def run_test_predict(
    final_run: bool,
    device: str,
    output: Path = STAGE3,
    doc: Path = RECORD_DOC,
    candidates_dir: Path = CANDIDATES_DIR,
    feature_dir: Path = LOCAL,
    cache: Path = CACHE,
    manifest: Path = MANIFEST,
    lock: Path = GPU_LOCK_PATH,
    roots: tuple[Path, ...] | None = None,
    command: list[str] | None = None,
) -> dict[str, Any]:
    """
    Score every candidate once on PTB-XL fold 10, the SPH evaluation half and the SPH calibration half.

    Parameters
    ----------
    final_run : bool
        The ``--i-am-the-final-run`` flag.
    device : str
        Torch device of S-R and F.
    output : Path
        Stage 3 directory with the record and operating points; receives ``predictions/<id>/<role>__...npz``
        and ``test_predictions_receipt.json``.
    doc : Path
        Committed Markdown document of the record.
    candidates_dir : Path
        Stage 2 candidate directory.
    feature_dir : Path
        Stage 1 feature directory.
    cache : Path
        Waveform cache directory.
    manifest : Path
        MLC-01 ``manifest.csv``.
    lock : Path
        GPU lock file.
    roots : tuple[Path, ...] | None
        Directories searched for test predictions; None is ``(OUTPUT, output, candidates_dir)``.
    command : list[str] | None
        Command line for the receipt.

    Returns
    -------
    dict[str, Any]
        The receipt.

    Raises
    ------
    PermissionError
        If any guard of the module docstring fails.
    """
    started = time.time()
    output = Path(output)
    roots = (OUTPUT, output, Path(candidates_dir)) if roots is None else roots
    if not final_run:
        raise PermissionError("test-predict scores the test roles once; pass --i-am-the-final-run")
    refuse_after_tests(roots, "test-predict")
    committed = check_record_committed(doc, output)
    record, record_sha = read_record(output)
    if record["provisional"]:
        raise PermissionError(f"the selection record is provisional (missing {record['missing_candidates']})")
    _, operating_sha = read_operating_points(output, record_sha)
    code = code_state(require_committed=True)
    validation = validation_data(manifest)
    validation_probabilities(record, validation)
    features = (
        feature_hashes(feature_dir)
        if any(row["id"] in FEATURE_CANDIDATES for row in record["candidates"])
        else None
    )
    data = load_manifest(manifest)
    computed, records = {}, {}
    for role in TEST_ROLES:
        records[role] = role_data(data, role)
        computed[role] = role_predictions(
            record, role, records[role], device, candidates_dir, feature_dir, cache, manifest, None, lock
        )
    refuse_after_tests(roots, "test-predict")
    seeds = {row["id"]: tuple(row["seeds"]) for row in record["candidates"]}
    files: dict[str, dict[str, Any]] = {}
    for role in TEST_ROLES:
        for name, entry in computed[role].items():
            path = prediction_path(output / PREDICTIONS, name, role, seeds[name])
            save_predictions(path, records[role]["record_key"], entry["probability"], name, seeds[name])
            files.setdefault(name, {})[role] = {
                "path": stored(path),
                "sha256": sha256_file(path),
                "records": len(entry["rows"]),
                "seconds": entry["seconds"],
            }
    result = json_ready(
        receipt(
            "test-predict",
            command or [],
            started,
            code,
            {
                "selection_record_sha256": record_sha,
                "operating_points_sha256": operating_sha,
                "record_doc": committed,
                "device": device,
                "roles": list(TEST_ROLES),
                "inputs": {
                    "manifest": stored(manifest),
                    "manifest_sha256": sha256_file(manifest),
                    "features": features,
                    "cache": str(cache),
                    "cache_index_sha256": sha256_file(Path(cache) / "index_all.csv")
                    if any(name in WAVEFORM_CANDIDATES for name in seeds)
                    else None,
                },
                "files": files,
                "test_roles_scored": True,
            },
        )
    )
    write_json_atomic(output / TEST_RECEIPT_NAME, result)
    return result


def run_dry_run(
    scratch: Path,
    device: str,
    role: str = "validation",
    output: Path = STAGE3,
    candidates_dir: Path = CANDIDATES_DIR,
    feature_dir: Path = LOCAL,
    cache: Path = CACHE,
    manifest: Path = MANIFEST,
    waveform_records: int | None = None,
    lock: Path = GPU_LOCK_PATH,
    tolerance: float = DRY_RUN_TOLERANCE,
    command: list[str] | None = None,
) -> dict[str, Any]:
    """
    Run ``role_predictions`` on the validation role and compare with the saved validation predictions.

    Parameters
    ----------
    scratch : Path
        Scratch directory for the predictions and the report.
    device : str
        Torch device of S-R and F.
    role : str
        Must be ``validation``.
    output : Path
        Stage 3 directory holding the selection record to follow (a provisional one is accepted).
    candidates_dir : Path
        Stage 2 candidate directory.
    feature_dir : Path
        Stage 1 feature directory.
    cache : Path
        Waveform cache directory.
    manifest : Path
        MLC-01 ``manifest.csv``.
    waveform_records : int | None
        Score S-R and F on this many random validation records (seed ``DRY_RUN_SEED``); None scores all.
    lock : Path
        GPU lock file.
    tolerance : float
        Largest allowed absolute difference per probability.
    command : list[str] | None
        Command line for the report.

    Returns
    -------
    dict[str, Any]
        The report, also written to ``<scratch>/dry_run_test_predict.json``.

    Raises
    ------
    PermissionError
        If ``role`` is not ``validation``.
    AssertionError
        After writing the report, if a candidate differs by more than ``tolerance``.
    """
    if role != "validation":
        raise PermissionError("the dry run scores the validation role only")
    started = time.time()
    scratch = Path(scratch)
    record, record_sha = read_record(output)
    data = validation_data(manifest)
    subset = None
    if waveform_records is not None:
        rng = np.random.default_rng(DRY_RUN_SEED)
        subset = np.sort(rng.choice(len(data["record_key"]), waveform_records, replace=False))
    computed = role_predictions(
        record, role, data, device, candidates_dir, feature_dir, cache, manifest, subset, lock
    )
    saved = validation_probabilities(record, data)
    candidates = {}
    for row in record["candidates"]:
        name = row["id"]
        entry = computed[name]
        path = prediction_path(scratch / PREDICTIONS, name, role, tuple(row["seeds"]))
        save_predictions(
            path, data["record_key"][entry["rows"]], entry["probability"], name, tuple(row["seeds"])
        )
        difference = float(np.max(np.abs(entry["probability"] - saved[name][entry["rows"]])))
        candidates[name] = {
            "records": len(entry["rows"]),
            "max_abs_difference": difference,
            "passed": difference <= tolerance,
            "seconds": entry["seconds"],
            "scratch_file": stored(path),
        }
    code = code_state(require_committed=False)
    report = json_ready(
        receipt(
            "test-predict dry run",
            command or [],
            started,
            code,
            {
                "role": role,
                "device": device,
                "tolerance": tolerance,
                "waveform_records": waveform_records,
                "selection_record_sha256": record_sha,
                "selection_record_provisional": record["provisional"],
                "missing_candidates": record["missing_candidates"],
                "candidates": candidates,
                "test_roles_opened": False,
            },
        )
    )
    write_json_atomic(scratch / "dry_run_test_predict.json", report)
    failed = [name for name, entry in candidates.items() if not entry["passed"]]
    if failed:
        raise AssertionError(f"dry run differs beyond {tolerance} for {failed}")
    return report
