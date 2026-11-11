"""
Records, labels, cleaning rules and features of one Challenge role for MLC-04, the one final test.

``docs/mlc04-final-test-protocol-2026-11-11.md`` is the contract. One code path serves two roles:

- ``validation``: the calibration groups of the Challenge split (the dry run);
- ``challenge_test``: the test groups, opened only after the freeze record is committed.

Every loader starts with ``check_role``, which refuses ``challenge_test`` unless
``docs/mlc04-freeze-record-2026-11-11.md`` is tracked by Git, the tracked tree is clean, and the code commit
the record names is an ancestor of ``HEAD``. Records are enumerated from the identity columns of the frozen
split table only (``mlc01_dataset.read_split_table``), and every reader process refuses a Challenge path
outside the role: its closed set is every split row of the other groups.

The path reuses the frozen code: the MLC-01 scan, labels, exclusions and duplicate rule (``mlc01_dataset``,
``mlc01_build``) with the v4 minimum-positive table; the MLC-02 waveform cache and extractors
(``mlc02_waveforms``, ``mlc02_features``); the MLC-03 Part A measures and flags, method B, the whole-record
windows and rules C1-C12 (``mlc03_signal_quality``, ``mlc03_signal_flags``, ``mlc03_measurements``,
``mlc03_dataset_v5``). In the role manifests the test groups take MLC-01's role name for a same-source test,
``test_same_source``, with origin ``test``.
"""

from __future__ import annotations

import json
import re
import subprocess
import time
from multiprocessing import Pool
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ecg_experiment import ROOT, mlc01_build, mlc01_dataset, mlc_labels
from ecg_experiment import mlc03_dataset_v5 as mv
from ecg_experiment.eda import ptbxl as ptbxl_eda
from ecg_experiment.files import read_json, write_json_atomic
from ecg_experiment.measurement_extractor042 import FEATURE_NAMES
from ecg_experiment.mlc01_dataset import RELEASES, array_sha256, challenge_key, microvolts
from ecg_experiment.mlc02_cache_build import git_state
from ecg_experiment.mlc02_waveforms import (
    block_sizes,
    cache_index,
    chunk_tasks,
    init_worker,
    open_block,
    verify_chunk,
    write_chunk,
)
from ecg_experiment.mlc03_measurements import measure_record
from ecg_experiment.mlc03_signal_flags import compute_flags
from ecg_experiment.mlc03_signal_quality import measure_window
from ecg_experiment.mlc03_signal_runs import read_checked

ROLES = ("validation", "challenge_test")
SPLITS = {"validation": "calibration", "challenge_test": "test"}
MANIFEST_ROLES = {"validation": "validation", "challenge_test": "test_same_source"}
OUTPUT = ROOT / "outputs/mlc04_final_test_20261111"
WORK = Path("/var/tmp/janetrivera/mlc04")
FREEZE_RECORD = ROOT / "docs/mlc04-freeze-record-2026-11-11.md"
MINIMUM_TABLE = mv.V4 / "minimum_positive_rule.csv"
HEADER_FIELDS = ["header_age", "header_sex", "header_dx"]
CACHE_CHUNK = 256
MEASURE_CHUNK = 64


def git_lines(*arguments: str) -> list[str]:
    """
    Run a Git command in the repository and return its output lines.

    Parameters
    ----------
    *arguments : str
        Arguments after ``git``.

    Returns
    -------
    list[str]
        Output lines.
    """
    return subprocess.check_output(["git", *arguments], cwd=ROOT, text=True).splitlines()


def freeze_committed(path: Path = FREEZE_RECORD) -> dict[str, Any]:
    """
    Read the committed freeze record; refuse if it is missing, untracked, modified or names no code commit.

    Parameters
    ----------
    path : Path
        Freeze record, a Markdown file with one fenced ``json`` block holding ``code_commit``.

    Returns
    -------
    dict[str, Any]
        The JSON block, with ``freeze_commit`` (the last commit that touched the file) and ``head``.

    Raises
    ------
    PermissionError
        If the file is absent or untracked, any tracked file differs from ``HEAD``, the block is missing, or
        its code commit is not an ancestor of ``HEAD``.
    """
    path = Path(path)
    if not path.exists():
        raise PermissionError(f"{path.name} does not exist; the Challenge test groups stay closed")
    relative = path.relative_to(ROOT).as_posix()
    tracked = subprocess.run(["git", "ls-files", "--error-unmatch", relative], cwd=ROOT, capture_output=True)
    if tracked.returncode != 0:
        raise PermissionError(f"{relative} is not committed; the Challenge test groups stay closed")
    modified = git_lines("status", "--porcelain", "--untracked-files=no")
    if modified:
        raise PermissionError(f"tracked files differ from HEAD: {modified}")
    match = re.search(r"```json\n(.*?)\n```", path.read_text(), flags=re.DOTALL)
    if not match:
        raise PermissionError(f"{relative} holds no JSON block")
    record = json.loads(match.group(1))
    commit = str(record.get("code_commit", ""))
    ancestor = subprocess.run(
        ["git", "merge-base", "--is-ancestor", commit, "HEAD"], cwd=ROOT, capture_output=True
    )
    if not commit or ancestor.returncode != 0:
        raise PermissionError(f"the code commit {commit!r} of {relative} is not an ancestor of HEAD")
    freeze = git_lines("log", "-1", "--format=%H", "--", relative)[0]
    return {**record, "freeze_commit": freeze, "head": git_lines("rev-parse", "HEAD")[0]}


def check_role(role: str) -> str:
    """
    Refuse an unknown role, and the test groups before the freeze record is committed.

    Parameters
    ----------
    role : str
        ``validation`` or ``challenge_test``.

    Returns
    -------
    str
        The role.

    Raises
    ------
    ValueError
        If the role is unknown.
    PermissionError
        If ``challenge_test`` is asked for without a committed freeze record (``freeze_committed``).
    """
    if role not in ROLES:
        raise ValueError(f"role must be one of {ROLES}, got {role!r}")
    if role == "challenge_test":
        freeze_committed()
    return role


def open_marker(directory: Path, rerun_after_crash: bool) -> dict[str, Any]:
    """
    Write the test-read marker before the first score; refuse a second read.

    Parameters
    ----------
    directory : Path
        Score directory of the test groups.
    rerun_after_crash : bool
        Continue a read that started and did not finish, once its partial outputs were deleted.

    Returns
    -------
    dict[str, Any]
        The marker.

    Raises
    ------
    FileExistsError
        If the read finished, or started without ``rerun_after_crash``, or partial outputs remain.
    FileNotFoundError
        If ``rerun_after_crash`` is given and no read started.
    """
    path = directory / "marker.json"
    attempt = {"started_unix": time.time(), "git": git_state()}
    if path.exists():
        marker = read_json(path)
        if marker.get("completed_unix") is not None:
            raise FileExistsError("the test read has been made; it cannot be repeated")
        if not rerun_after_crash:
            raise FileExistsError("a test read started; rerun only with --rerun-after-crash")
        if [item.name for item in directory.iterdir() if item.name != "marker.json"]:
            raise FileExistsError("delete the partial score outputs unread before rerunning")
        marker["attempts"].append(attempt)
    elif rerun_after_crash:
        raise FileNotFoundError("no test read has started")
    else:
        directory.mkdir(parents=True, exist_ok=True)
        marker = {
            "experiment": "MLC-04",
            "freeze": freeze_committed(),
            "attempts": [attempt],
            "completed_unix": None,
        }
    write_json_atomic(path, marker)
    return marker


def split_identity(rows: pd.DataFrame, split: str) -> tuple[pd.DataFrame, frozenset[str]]:
    """
    Select the records of one split group and the Challenge paths of every other group.

    Parameters
    ----------
    rows : pd.DataFrame
        Identity columns of the split table (``source``, ``family``, ``record``, ``path``, ``split``).
    split : str
        ``calibration`` or ``test``.

    Returns
    -------
    tuple[pd.DataFrame, frozenset[str]]
        The group's rows in table order (Ningbo JS13118 left out, as in MLC-01), with ``key`` from
        ``challenge_key``; and the ``challenge_key`` of every row outside the group.

    Raises
    ------
    ValueError
        If the split name is unknown or a kept row shares a path with a closed one.
    """
    if split not in set(SPLITS.values()):
        raise ValueError(f"unknown split {split!r}")
    keys = [challenge_key(source, path) for source, path in zip(rows["source"], rows["path"], strict=True)]
    rows = rows.assign(key=keys)
    record_keys = rows["source"] + ":" + rows["record"]
    inside = (rows["split"] == split).to_numpy()
    kept = rows[inside & ~record_keys.isin(mlc01_dataset.UPSTREAM_EXCLUDED).to_numpy()].reset_index(drop=True)
    closed = frozenset(rows.loc[~inside, "key"])
    if set(kept["key"]) & closed:
        raise ValueError("a record of the group shares a path with another group")
    return kept, closed


def role_rows(role: str) -> tuple[pd.DataFrame, frozenset[str]]:
    """
    Select a role's records from the frozen split table, and the paths its readers must refuse.

    Parameters
    ----------
    role : str
        ``validation`` or ``challenge_test``.

    Returns
    -------
    tuple[pd.DataFrame, frozenset[str]]
        Output of ``split_identity`` for the role's split group.
    """
    role = check_role(role)
    return split_identity(mlc01_dataset.read_split_table(), SPLITS[role])


def scan_jobs(rows: pd.DataFrame) -> list[dict[str, object]]:
    """
    MLC-01 scan jobs of a role's records, with their official release checksums.

    Parameters
    ----------
    rows : pd.DataFrame
        Output of ``split_identity``.

    Returns
    -------
    list[dict[str, object]]
        ``mlc01_dataset.wfdb_job`` per record, sources in ``RELEASES`` order as MLC-01 built them.
    """
    jobs = []
    for source, root in RELEASES.items():
        part = rows[rows["source"] == source]
        names = {f"{path}{suffix}" for path in part["path"] for suffix in (".hea", ".mat")}
        sums = mlc01_dataset.checksums(root / "SHA256SUMS.txt", names)
        jobs += [
            mlc01_dataset.wfdb_job(source, record, f"{source}:{record}", root, path, sums)
            for record, path in zip(part["record"], part["path"], strict=True)
        ]
    return jobs


def run_scan(role: str, workers: int) -> pd.DataFrame:
    """
    Verify, read and summarize every record of a role with the MLC-01 scan, behind the role guard.

    Parameters
    ----------
    role : str
        ``validation`` or ``challenge_test``.
    workers : int
        Reader processes.

    Returns
    -------
    pd.DataFrame
        ``mlc01_build.normalize_scan`` of the scan rows, in job order.

    Raises
    ------
    PermissionError
        If a job is a path of another group.
    """
    rows, closed = role_rows(role)
    jobs = scan_jobs(rows)
    if any(job["guard"] in closed for job in jobs):
        raise PermissionError("a path of another group reached the scan jobs")
    with Pool(workers, initializer=mlc01_dataset.init_worker, initargs=(closed,)) as pool:
        scanned = list(pool.imap(mlc01_dataset.scan, jobs, chunksize=8))
    return mlc01_build.normalize_scan(scanned)


def frozen_minimum(path: Path = MINIMUM_TABLE) -> pd.DataFrame:
    """
    Read the minimum-positive table of the final MLC-01 build (Amendment 1, ruling 6), as frozen in v4.

    Parameters
    ----------
    path : Path
        v4 ``minimum_positive_rule.csv``.

    Returns
    -------
    pd.DataFrame
        ``source``, ``label``, ``positives``, ``defined`` and boolean ``not_annotated``.
    """
    table = pd.read_csv(path)
    table["not_annotated"] = table["not_annotated"].astype(str) == "True"
    return table


def identity(role: str, rows: pd.DataFrame, scan: pd.DataFrame) -> pd.DataFrame:
    """
    Identity, demographics and raw codes of a role's scanned records, as ``mlc01_build.identity_frame``.

    Parameters
    ----------
    role : str
        ``validation`` or ``challenge_test``.
    rows : pd.DataFrame
        Output of ``split_identity``.
    scan : pd.DataFrame
        Output of ``run_scan``.

    Returns
    -------
    pd.DataFrame
        Indexed by record key, in split-table order; role and origin per ``MANIFEST_ROLES`` and ``SPLITS``.
    """
    frame = mlc01_dataset.challenge_records(rows.assign(split="calibration")).set_index("record_key")
    frame["role"] = MANIFEST_ROLES[role]
    frame["origin"] = SPLITS[role]
    header = scan.set_index("record_key").reindex(frame.index)
    frame["age"] = pd.to_numeric(header["header_age"], errors="coerce").astype(float)
    frame["age_over_89"] = False
    frame["sex"] = header["header_sex"].map({"Male": "M", "Female": "F"})
    frame["raw_codes"] = header["header_dx"]
    return frame.loc[frame.index.isin(scan["record_key"])]


def assemble(role: str, rows: pd.DataFrame, scan: pd.DataFrame) -> pd.DataFrame:
    """
    Build a role's MLC-01 manifest rows: identity, scan, labels, exclusions, duplicates and inclusion.

    The steps of ``mlc01_build.assemble``, except that the minimum-positive rule applies the frozen v4 table
    instead of recounting positives among the role's records, and duplicates are resolved inside the role
    (the split assigns every duplicate group to one group).

    Parameters
    ----------
    role : str
        ``validation`` or ``challenge_test``.
    rows : pd.DataFrame
        Output of ``split_identity``.
    scan : pd.DataFrame
        Output of ``run_scan``.

    Returns
    -------
    pd.DataFrame
        Indexed by record key, ready for ``mlc01_build.manifest_table``.
    """
    frame = identity(role, rows, scan)
    classes = ptbxl_eda.diagnostic_classes(ptbxl_eda.load_statements())
    labels = mlc01_dataset.label_frame(frame, classes, frozenset(classes))
    states = [name for name in labels.columns if name.startswith(("a_", "b_", "c_"))]
    labels[states] = mlc_labels.apply_minimum_positives(labels[states], frame["source"], frozen_minimum())
    scanned = scan.set_index("record_key").drop(columns=HEADER_FIELDS, errors="ignore")
    frame = frame.join(scanned).join(labels)
    frame["exclusion_reason"] = mlc01_dataset.exclusion_reason(frame.reset_index()).to_numpy()
    eligible = frame[frame["exclusion_reason"] == ""]
    frame = frame.join(mlc01_dataset.resolve_duplicates(eligible, mlc01_dataset.label_columns(labels)))
    frame["duplicate_status"] = frame["duplicate_status"].fillna("")
    frame["included"] = frame["duplicate_status"].isin(mlc01_dataset.KEPT)
    flags = mlc01_build.FLAG_COLUMNS
    frame["n_flat_leads"] = frame[flags[:12]].fillna(False).sum(axis=1)
    frame["n_missing_leads"] = frame[flags[12:]].fillna(False).sum(axis=1)
    return frame


def withhold_screening_labels(frame: pd.DataFrame, role: str) -> pd.DataFrame:
    """
    Empty the space-A groups and the ``normal`` column of the test groups before anything is written.

    The binary screening label is built from these columns (any MI, STTC, CD or HYP code against sinus rhythm
    alone); the protocol computes no binary label on the test groups. The MLC-01 label code still produces
    them in memory, because its duplicate rule compares every label space.

    Parameters
    ----------
    frame : pd.DataFrame
        Output of ``assemble``.
    role : str
        ``validation`` or ``challenge_test``.

    Returns
    -------
    pd.DataFrame
        The frame, with those columns undefined for ``challenge_test`` and unchanged for ``validation``.
    """
    if role != "challenge_test":
        return frame
    columns = [name for name in frame.columns if name.startswith("a_") or name == "normal"]
    return frame.assign(**dict.fromkeys(columns, np.nan))


def read_text(path: Path) -> pd.DataFrame:
    """
    Read a manifest as text, empty cells kept empty.

    Parameters
    ----------
    path : Path
        ``manifest.csv``.

    Returns
    -------
    pd.DataFrame
        Every column as ``str``.
    """
    return pd.read_csv(path, dtype=str, keep_default_na=False, low_memory=False)


def build_cache(role: str, manifest: pd.DataFrame, directory: Path, workers: int) -> pd.DataFrame:
    """
    Cache the first 10 s of every eligible record of a role with the MLC-02 cache code, behind the guard.

    Parameters
    ----------
    role : str
        ``validation`` or ``challenge_test``.
    manifest : pd.DataFrame
        The role's v4 manifest as text (``read_text``).
    directory : Path
        Cache directory; it must not hold an index yet.
    workers : int
        Reader processes.

    Returns
    -------
    pd.DataFrame
        The cache index (``mlc02_waveforms.cache_index`` columns) with the read and file checks.

    Raises
    ------
    FileExistsError
        If the directory already holds an index.
    ValueError
        If a record does not reproduce its manifest hash before or after the float32 cast, or from the file.
    """
    _, closed = role_rows(role)
    if (directory / "index.csv").exists():
        raise FileExistsError(f"{directory} already holds a cache")
    directory.mkdir(parents=True, exist_ok=True)
    eligible = manifest.assign(manifest_row=np.arange(len(manifest)))
    eligible = eligible[eligible["exclusion_reason"] == ""]
    index = cache_index(eligible[["record_key", "source", "role", "raw_path", "sha256_10s", "manifest_row"]])
    for name, size in block_sizes(index).items():
        array = open_block(directory, name, size, "w+")
        del array
    tasks = chunk_tasks(index, directory, CACHE_CHUNK)
    allowed = frozenset(index["raw_path"])
    with Pool(workers, initializer=init_worker, initargs=(closed, allowed)) as pool:
        written = pd.DataFrame([row for part in pool.imap(write_chunk, tasks) for row in part])
        reread = pd.DataFrame([row for part in pool.imap(verify_chunk, tasks) for row in part])
    checks = written.merge(reread.rename(columns={"sha256_f32": "reread_f32"}), on="position")
    index = index.merge(checks, on="position", validate="one_to_one")
    ok = index["sha256_read_ok"] & index["sha256_float32_ok"] & index["sha256_cache_ok"]
    if not ok.all() or not (index["reread_f32"] == index["sha256_f32"]).all():
        raise ValueError(f"{int((~ok).sum())} records do not reproduce their manifest hash")
    index.drop(columns=["reread_f32"]).to_csv(directory / "index.csv", index=False)
    return index


def cache_tasks(index: pd.DataFrame, directory: Path, size: int = MEASURE_CHUNK) -> list[dict[str, Any]]:
    """
    Chunks of consecutive rows of one block, for the measurement workers.

    Parameters
    ----------
    index : pd.DataFrame
        Cache index (or a subset of its rows).
    directory : Path
        Cache directory.
    size : int
        Records per chunk.

    Returns
    -------
    list[dict[str, Any]]
        ``directory``, ``block``, ``records`` (block size) and ``jobs`` (``position``, ``row``,
        ``record_key``, ``sha256_10s``).
    """
    sizes = block_sizes(pd.read_csv(directory / "index.csv", usecols=["block"]))
    tasks = []
    for name, rows in index.groupby("block", sort=False):
        jobs = rows[["position", "row", "record_key", "sha256_10s"]].to_dict("records")
        tasks += [
            {
                "directory": str(directory),
                "block": name,
                "records": sizes[name],
                "jobs": jobs[start : start + size],
            }
            for start in range(0, len(jobs), size)
        ]
    return tasks


def part_a_chunk(task: dict[str, Any]) -> list[dict[str, Any]]:
    """
    Part A measures of one chunk of cached windows, each checked against its hash first.

    Parameters
    ----------
    task : dict[str, Any]
        One output of ``cache_tasks``.

    Returns
    -------
    list[dict[str, Any]]
        ``position``, ``record_key`` and ``measure_window`` per record.
    """
    array = open_block(Path(task["directory"]), task["block"], task["records"], "r")
    rows = []
    for job in task["jobs"]:
        window = read_checked(array, int(job["row"]), job["sha256_10s"], job["record_key"])
        rows.append(
            {"position": int(job["position"]), "record_key": job["record_key"], **measure_window(window)}
        )
    return rows


def method_b_chunk(task: dict[str, Any]) -> list[dict[str, Any]]:
    """
    Measure one chunk of cached windows with method B, as ``mlc03_dataset_v5.measure_cached`` does.

    Parameters
    ----------
    task : dict[str, Any]
        One output of ``cache_tasks``.

    Returns
    -------
    list[dict[str, Any]]
        ``record_key`` and ``measure_record`` of the float32 window per record.

    Raises
    ------
    ValueError
        If a cached window does not reproduce its hash.
    """
    array = open_block(Path(task["directory"]), task["block"], task["records"], "r")
    rows = []
    for job in task["jobs"]:
        signal = np.asarray(array[int(job["row"])])
        if array_sha256(microvolts(signal.T)) != job["sha256_10s"]:
            raise ValueError(f"cached window of {job['record_key']} does not reproduce its manifest hash")
        rows.append({"record_key": job["record_key"], **measure_record(signal)})
    return rows


def run_part_a(role: str, directory: Path, manifest: pd.DataFrame, workers: int) -> tuple[pd.DataFrame, ...]:
    """
    Part A measures and flags of every cached record of a role.

    Parameters
    ----------
    role : str
        ``validation`` or ``challenge_test``.
    directory : Path
        The role's cache.
    manifest : pd.DataFrame
        The role's v4 manifest as text.
    workers : int
        Processes.

    Returns
    -------
    tuple[pd.DataFrame, ...]
        Measures (cache order) and flags (``compute_flags`` with ``source``, ``role`` and boolean
        ``included``).
    """
    check_role(role)
    index = pd.read_csv(directory / "index.csv", dtype={"record_key": str, "raw_path": str})
    with Pool(workers) as pool:
        rows = [row for part in pool.imap(part_a_chunk, cache_tasks(index, directory)) for row in part]
    measures = pd.DataFrame(rows).sort_values("position").reset_index(drop=True)
    if measures["record_key"].tolist() != index["record_key"].tolist():
        raise ValueError("Part A measures do not cover the cache")
    keys = manifest.set_index("record_key").loc[measures["record_key"]]
    flags = compute_flags(measures).assign(
        source=keys["source"].to_numpy(),
        role=keys["role"].to_numpy(),
        included=(keys["included"] == "True").to_numpy(),
    )
    return measures, flags


def run_method_b(role: str, directory: Path, keys: list[str], workers: int) -> pd.DataFrame:
    """
    Measure some cached records of a role with method B.

    Parameters
    ----------
    role : str
        ``validation`` or ``challenge_test``.
    directory : Path
        The role's cache.
    keys : list[str]
        Record keys, in output order.
    workers : int
        Processes.

    Returns
    -------
    pd.DataFrame
        One row per key in key order, ``status`` as text.
    """
    check_role(role)
    index = pd.read_csv(directory / "index.csv", dtype={"record_key": str, "raw_path": str})
    index = index.set_index("record_key").loc[keys].reset_index()
    with Pool(workers) as pool:
        found = {
            row["record_key"]: row
            for part in pool.imap(method_b_chunk, cache_tasks(index, directory))
            for row in part
        }
    frame = pd.DataFrame([found[key] for key in keys])
    frame["status"] = frame["status"].astype(str)
    return frame


def run_windows(role: str, records: pd.DataFrame, workers: int) -> pd.DataFrame:
    """
    Summarise the 10 s windows of every record longer than 10 s, read whole through the role guard.

    Parameters
    ----------
    role : str
        ``validation`` or ``challenge_test``.
    records : pd.DataFrame
        ``mlc03_dataset_v5.load_records`` of the role.
    workers : int
        Processes.

    Returns
    -------
    pd.DataFrame
        ``record_key`` and the ``window_summary`` columns, lists as JSON text, as ``windows_all`` wrote them.
    """
    _, closed = role_rows(role)
    long = records[records["duration_s"] > 10.0 + 1e-6]
    jobs = list(zip(long["record_key"], long["raw_path"], long["source"], strict=True))
    with Pool(workers, initializer=init_worker, initargs=(closed, frozenset(long["raw_path"]))) as pool:
        rows = list(pool.map(mv.record_windows, jobs, chunksize=8))
    frame = pd.DataFrame(rows, columns=None if rows else ["record_key"])
    for column in ("ectopy_by_window", "af_by_window", "hr_by_window"):
        if column in frame:
            frame[column] = frame[column].map(json.dumps)
    return frame


def method_a(keys: np.ndarray, values: np.ndarray, status: pd.Series) -> pd.DataFrame:
    """
    Derive method A of some records from their MLC-02 measurement rows, as Part B did.

    The derivation is ``mlc03_dataset_v5.measurements_a``'s, applied to arrays given here instead of the
    stored MLC-02 feature directory.

    Parameters
    ----------
    keys : np.ndarray
        Record keys.
    values : np.ndarray
        ``[records, 130]`` measurements in ``FEATURE_NAMES`` order, aligned with ``keys``.
    status : pd.Series
        Extractor status of each record, aligned with ``keys``.

    Returns
    -------
    pd.DataFrame
        The columns of ``measurements_a``.
    """
    frame = pd.DataFrame(
        np.asarray(values, dtype=np.float64), columns=[f"a_{name}" for name in FEATURE_NAMES]
    )
    frame.insert(0, "record_key", np.asarray(keys))
    frame["a_status"] = np.asarray(status)
    frame["hr_a"], frame["qrs_a"], frame["pr_a"] = frame["a_hr_bpm"], frame["a_qrs_dur_ms"], frame["a_pr_ms"]
    frame["axis_a"] = np.degrees(np.arctan2(frame["a_qrs_area_aVF"], frame["a_qrs_area_I"]))
    frame["rr_nrmssd_a"] = frame["a_rr_rmssd_ms"] / frame["a_rr_mean_ms"]
    frame["rr_range_a"] = frame["a_rr_range_rel"] * frame["a_rr_mean_ms"]
    return frame


def feature_positions(rows: pd.DataFrame, keys: np.ndarray) -> np.ndarray:
    """
    Rows of a role's feature arrays for some record keys.

    Parameters
    ----------
    rows : pd.DataFrame
        The role's feature ``rows.csv`` (one row per cached record, array order).
    keys : np.ndarray
        Record keys.

    Returns
    -------
    np.ndarray
        Array rows.

    Raises
    ------
    KeyError
        If a key has no feature row.
    """
    position = pd.Series(np.arange(len(rows)), index=rows["record_key"].to_numpy())
    return position.loc[np.asarray(keys)].to_numpy()


def apply_v5(
    manifest_v4: pd.DataFrame, table: pd.DataFrame, flags: pd.DataFrame, windows: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Apply rules C1-C12 to a role's v4 manifest with the v5 builder's code, and add the 18 Part A flags.

    Parameters
    ----------
    manifest_v4 : pd.DataFrame
        The role's v4 manifest as text.
    table : pd.DataFrame
        ``mlc03_dataset_v5.analysis`` of the role's included records.
    flags : pd.DataFrame
        Part A flags of every cached record of the role.
    windows : pd.DataFrame
        Window summaries of the role's records longer than 10 s.

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame]
        The v5 manifest rows (text) and the flow of ``mlc03_dataset_v5.apply_rules``.
    """
    aligned = mv.aligned_flags(table, flags)
    record_masks = mv.record_rule_masks(table, aligned)
    label_masks = mv.label_rule_masks(mv.part_b_masks(table, aligned, windows), table)
    v5, flow = mv.apply_rules(manifest_v4, record_masks.assign(record_key=table["record_key"]), label_masks)
    flag_table = mv.flag_columns(flags)
    if flag_table["record_key"].duplicated().any():
        raise ValueError("a record has two Part A flag rows")
    flag_table = flag_table.set_index("record_key").reindex(v5["record_key"])
    for name in flag_table.columns:
        v5[name] = flag_table[name].map({True: "True", False: "False"}).fillna("").to_numpy()
    return v5, flow


def cell_mismatches(first: pd.DataFrame, second: pd.DataFrame) -> dict[str, int]:
    """
    Count differing cells per column of two aligned tables; NaN equals NaN, numbers compare exactly.

    Parameters
    ----------
    first, second : pd.DataFrame
        Tables with the same rows in the same order.

    Returns
    -------
    dict[str, int]
        Columns that differ with their count of differing cells; ``-1`` for a column in one table only.

    Raises
    ------
    ValueError
        If the tables have a different number of rows.
    """
    if len(first) != len(second):
        raise ValueError(f"tables have {len(first)} and {len(second)} rows")
    found = {}
    for name in sorted(set(first.columns) | set(second.columns)):
        if name not in first.columns or name not in second.columns:
            found[name] = -1
            continue
        a, b = first[name].reset_index(drop=True), second[name].reset_index(drop=True)
        if pd.api.types.is_numeric_dtype(a) and pd.api.types.is_numeric_dtype(b):
            x, y = a.to_numpy(dtype=float), b.to_numpy(dtype=float)
            differ = ~((x == y) | (np.isnan(x) & np.isnan(y)))
        else:
            differ = ~((a.astype(str) == b.astype(str)) | (a.isna() & b.isna())).to_numpy()
        if differ.any():
            found[name] = int(differ.sum())
    return found
