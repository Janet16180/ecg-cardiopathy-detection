"""Records, waveform checks, roles and duplicate resolution of the MLC-01 condition dataset.

Sources are PTB-XL 1.0.3 at 500 Hz, the train and calibration groups of the recovered Challenge split table
and SPH (``docs/mlc01-condition-dataset-protocol-2026-11-10.md``). Challenge records are enumerated from the
split table's identity and group columns only (source, family, record, path, split). Test-group rows are
filtered out before any file is opened, and every worker refuses a test-group path.

A waveform is reduced to integer microvolts in the lead order I, II, III, aVR, aVL, aVF, V1-V6, as a
``(samples, 12)`` int32 array in C order, a missing sample taking the value -99999 mV before scaling. This is
the convention of the Challenge split's ``signal_sha256``. ``sha256_10s`` hashes the first 10 s and
``sha256_full`` the whole recording. Exact duplicates share ``sha256_10s``, because every model sees the
first 10 s.
"""

from __future__ import annotations

import ast
import hashlib
import resource
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import wfdb

from ecg_experiment import ROOT, mlc_labels, sph
from ecg_experiment.eda.signals import LEADS, canonical_order
from ecg_experiment.files import sha256_file
from ecg_experiment.measurement_screening042 import patient_role
from ecg_experiment.paths import to_stored

PTBXL_DIR = ROOT / "data/raw/ptb-xl/1.0.3"
RELEASES = {"ningbo": ROOT / "data/raw/challenge-2021/1.0.3",
            "chapman_shaoxing": ROOT / "data/raw/challenge-2021/1.0.3",
            "georgia": ROOT / "data/raw/challenge-2020/1.0.2",
            "cpsc_2018": ROOT / "data/raw/challenge-2020/1.0.2",
            "cpsc_2018_extra": ROOT / "data/raw/challenge-2020/1.0.2"}
SPLIT_TABLE = ROOT / "outputs/cohort25_recovery_20261028/challenge/full/rows.csv"
SPLIT_TABLE_SHA256 = "5ecd82154fe9ed574b9706d9b7efbe417a578197845cf2a6a48615ca54959cc8"
SPLIT_COLUMNS = ["source", "family", "record", "path", "split"]
SPLIT_NAMES = frozenset({"train", "calibration", "test"})
OPEN_SPLITS = frozenset({"train", "calibration"})
SPH_ROWS = ROOT / "outputs/pr74_fresh_baseline_exact_20261028/features/sph_rows.csv"
RATE = 500
SECONDS = 10
MISSING_MV = -99999.0
UPSTREAM_EXCLUDED = frozenset({"ningbo:JS13118"})
PTBXL_GROUP_RECORDS = {"folds_1_8": 17418, "fold_9": 2183, "fold_10": 2198}
SPH_RECORDS = 25770
SPH_PATIENTS = 24666
CHALLENGE_SPLIT_RECORDS = {("ningbo", "train"): 20941, ("ningbo", "calibration"): 6979,
                           ("chapman_shaoxing", "train"): 6147, ("chapman_shaoxing", "calibration"): 2050,
                           ("georgia", "train"): 6207, ("georgia", "calibration"): 2066,
                           ("cpsc_2018", "train"): 4121, ("cpsc_2018", "calibration"): 1375,
                           ("cpsc_2018_extra", "train"): 2074, ("cpsc_2018_extra", "calibration"): 692}
ROLE_RANK = {"test_same_source": 0, "test_external": 0, "validation": 1, "sph_calibration": 2, "train": 3}
ROLES = ("train", "validation", "test_same_source", "test_external", "sph_calibration")
EXCLUSIONS = ("upstream_corrupted", "unreadable", "wrong_leads", "wrong_rate", "short")
KEPT = frozenset({"unique", "kept"})
GUARD: dict[str, frozenset[str]] = {"closed": frozenset()}


def read_split_table(path: Path = SPLIT_TABLE) -> pd.DataFrame:
    """
    Identity and group columns of the recovered Challenge split table, after checking its SHA-256.

    Parameters
    ----------
    path : Path
        The split table ``rows.csv``.

    Returns
    -------
    pd.DataFrame
        ``source``, ``family``, ``record``, ``path`` and ``split`` as text; no label or hash column is read.

    Raises
    ------
    ValueError
        If the file differs from the frozen split or holds an unknown split name.
    """
    if sha256_file(path) != SPLIT_TABLE_SHA256:
        raise ValueError(f"Split table differs from the frozen SHA-256: {path}")
    rows = pd.read_csv(path, usecols=SPLIT_COLUMNS, dtype=str)[SPLIT_COLUMNS]
    if not set(rows["split"]) <= SPLIT_NAMES:
        raise ValueError("Unknown split name in the split table")
    return rows


def challenge_key(source: str, path: str) -> str:
    """
    Identity of one Challenge record path, as used by the open-path guard.

    Parameters
    ----------
    source : str
        Challenge source.
    path : str
        Release-relative path without extension, such as ``training/georgia/g1/E00001``.

    Returns
    -------
    str
        ``source:path``.
    """
    return f"{source}:{path}"


def open_challenge_rows(rows: pd.DataFrame) -> pd.DataFrame:
    """
    Keep the train and calibration groups of the split table.

    Parameters
    ----------
    rows : pd.DataFrame
        Output of ``read_split_table``.

    Returns
    -------
    pd.DataFrame
        The open rows, with ``key`` from ``challenge_key``.

    Raises
    ------
    ValueError
        If a kept row is not in an open split or shares a path with a closed row.
    """
    keys = [challenge_key(source, path) for source, path in zip(rows["source"], rows["path"], strict=True)]
    rows = rows.assign(key=keys)
    kept = rows[rows["split"].isin(OPEN_SPLITS)].reset_index(drop=True)
    if not set(kept["split"]) <= OPEN_SPLITS or set(kept["key"]) & closed_challenge_keys(rows):
        raise ValueError("A closed Challenge row survived the filter")
    return kept


def closed_challenge_keys(rows: pd.DataFrame) -> frozenset[str]:
    """
    Paths of the closed (test) Challenge groups, from the identity columns only.

    Parameters
    ----------
    rows : pd.DataFrame
        Output of ``read_split_table``.

    Returns
    -------
    frozenset[str]
        ``challenge_key`` of every row outside the open splits.
    """
    closed = rows[~rows["split"].isin(OPEN_SPLITS)]
    return frozenset(challenge_key(source, path) for source, path in zip(closed["source"], closed["path"],
                                                                          strict=True))


def require_open(key: str, closed: frozenset[str]) -> None:
    """
    Refuse to open a closed Challenge record.

    Parameters
    ----------
    key : str
        Output of ``challenge_key``.
    closed : frozenset[str]
        Output of ``closed_challenge_keys``.

    Raises
    ------
    PermissionError
        If ``key`` belongs to a closed group.
    """
    if key in closed:
        raise PermissionError(f"Closed Challenge record requested: {key}")


def init_worker(closed: frozenset[str]) -> None:
    """
    Give a scan worker the closed Challenge paths.

    Parameters
    ----------
    closed : frozenset[str]
        Output of ``closed_challenge_keys``.
    """
    GUARD["closed"] = closed


def checksums(manifest: Path, names: set[str]) -> dict[str, str]:
    """
    Official SHA-256 of selected files from a release ``SHA256SUMS.txt``.

    Parameters
    ----------
    manifest : Path
        The release checksum file, one ``digest path`` pair per line.
    names : set[str]
        Release-relative file names to look up.

    Returns
    -------
    dict[str, str]
        Digest per requested name present in the manifest.
    """
    found = {}
    for line in manifest.read_text().splitlines():
        digest, name = line.split(maxsplit=1)
        if name.strip() in names:
            found[name.strip()] = digest
    return found


def wfdb_job(source: str, record: str, key: str, root: Path, stem: str, sums: dict[str, str]
             ) -> dict[str, object]:
    """
    Scan job of one WFDB record.

    Parameters
    ----------
    source, record : str
        Source and record name.
    key : str
        Manifest key, ``source:record``.
    root : Path
        Release root.
    stem : str
        Release-relative path without extension.
    sums : dict[str, str]
        Output of ``checksums`` for the release.

    Returns
    -------
    dict[str, object]
        Job for ``scan``.
    """
    suffix = ".dat" if source == "ptbxl" else ".mat"
    files = {"header": (str(root / f"{stem}.hea"), sums[f"{stem}.hea"]),
             "signal": (str(root / f"{stem}{suffix}"), sums[f"{stem}{suffix}"])}
    guard = challenge_key(source, stem) if source in RELEASES else ""
    return {"key": key, "source": source, "record": record, "format": "wfdb", "stem": str(root / stem),
            "relative": stem, "guard": guard, "files": files}


def sph_job(record: str) -> dict[str, object]:
    """
    Scan job of one SPH record; the release lists no per-record checksum.

    Parameters
    ----------
    record : str
        SPH ``ECG_ID``.

    Returns
    -------
    dict[str, object]
        Job for ``scan``.
    """
    path = str(sph.RECORDS / f"{record}.h5")
    return {"key": f"sph:{record}", "source": "sph", "record": record, "format": "h5", "stem": path,
            "guard": "", "files": {"signal": (path, None)}}


def verify_files(files: dict[str, tuple[str, str | None]]) -> dict[str, object]:
    """
    Hash the raw files of one record and compare them with the official checksums.

    Parameters
    ----------
    files : dict[str, tuple[str, str | None]]
        Role (``header`` or ``signal``) to path and official digest (None when the release has none).

    Returns
    -------
    dict[str, object]
        ``<role>_file_sha256`` per file and ``checksum_verified`` (True when every file had an official
        digest).

    Raises
    ------
    ValueError
        If a file differs from its official digest.
    """
    row: dict[str, object] = {}
    for role, (path, expected) in files.items():
        digest = sha256_file(path)
        if expected is not None and digest != expected:
            raise ValueError(f"Raw file differs from its official checksum: {path}")
        row[f"{role}_file_sha256"] = digest
    row["checksum_verified"] = all(expected is not None for _, expected in files.values())
    return row


def header_fields(path: str) -> dict[str, str]:
    """
    Age, sex and diagnosis codes from the comment lines of a Challenge header.

    Parameters
    ----------
    path : str
        The ``.hea`` file.

    Returns
    -------
    dict[str, str]
        ``header_age``, ``header_sex`` and ``header_dx``; empty when a field is absent.
    """
    fields = {"Age": "", "Sex": "", "Dx": ""}
    for line in Path(path).read_text().splitlines():
        if line.startswith("#") and ":" in line:
            name, value = line[1:].split(":", 1)
            if name.strip() in fields:
                fields[name.strip()] = value.strip()
    return {"header_age": fields["Age"], "header_sex": fields["Sex"], "header_dx": fields["Dx"]}


def read_signal(job: dict[str, object]) -> tuple[np.ndarray, int, list[str]]:
    """
    Read one record in mV.

    Parameters
    ----------
    job : dict[str, object]
        Output of ``wfdb_job`` or ``sph_job``.

    Returns
    -------
    tuple[np.ndarray, int, list[str]]
        Float64 array ``(samples, leads)``, sampling rate and lead names. SPH files carry neither, so the
        dataset's documented 500 Hz and lead order are used.
    """
    if job["format"] == "h5":
        with h5py.File(job["stem"], "r") as handle:
            signal = handle["ecg"][:].astype(np.float64).T
        return signal, RATE, list(LEADS)
    record = wfdb.rdrecord(job["stem"])
    return np.asarray(record.p_signal, dtype=np.float64), int(record.fs), list(record.sig_name)


def microvolts(signal: np.ndarray) -> np.ndarray:
    """
    Signal in integer microvolts, missing samples set to -99999 mV first.

    Parameters
    ----------
    signal : np.ndarray
        Float array in mV.

    Returns
    -------
    np.ndarray
        Int32 array of the same shape.
    """
    return np.round(np.nan_to_num(signal.astype(np.float64), nan=MISSING_MV) * 1000).astype(np.int32)


def array_sha256(values: np.ndarray) -> str:
    """
    SHA-256 of an array's bytes in C order.

    Parameters
    ----------
    values : np.ndarray
        Any array.

    Returns
    -------
    str
        Hex digest.
    """
    return hashlib.sha256(np.ascontiguousarray(values).tobytes()).hexdigest()


def window_summary(window: np.ndarray, micro: np.ndarray) -> dict[str, object]:
    """
    Hashes and lead flags of the first 10 s.

    Parameters
    ----------
    window : np.ndarray
        Float array ``(samples, 12)`` in mV and canonical lead order.
    micro : np.ndarray
        ``microvolts(window)``.

    Returns
    -------
    dict[str, object]
        ``sha256_10s``; ``lead_sha256_10s`` (first 16 hex digits per lead, comma-separated);
        ``shape_sha256_10s`` (hash of the sign of successive differences, unchanged by gain and offset);
        ``flat_<lead>`` (constant, or no finite sample) and ``missing_<lead>`` (any missing sample).
    """
    finite = np.isfinite(window)
    high = np.where(finite, window, -np.inf).max(axis=0)
    low = np.where(finite, window, np.inf).min(axis=0)
    flat = ~(high > low)
    missing = ~finite.all(axis=0)
    shape = np.sign(np.diff(micro.astype(np.int64), axis=0)).astype(np.int8)
    row: dict[str, object] = {
        "sha256_10s": array_sha256(micro),
        "lead_sha256_10s": ",".join(array_sha256(micro[:, index])[:16] for index in range(len(LEADS))),
        "shape_sha256_10s": array_sha256(shape),
    }
    row |= {f"flat_{lead}": bool(flat[index]) for index, lead in enumerate(LEADS)}
    row |= {f"missing_{lead}": bool(missing[index]) for index, lead in enumerate(LEADS)}
    return row


def summarize(signal: np.ndarray, fs: int, names: list[str]) -> dict[str, object]:
    """
    Duration, lead identity, hashes and lead flags of one record.

    Parameters
    ----------
    signal : np.ndarray
        Float array ``(samples, leads)`` in mV.
    fs : int
        Sampling rate.
    names : list[str]
        Lead names.

    Returns
    -------
    dict[str, object]
        ``fs``, ``n_samples``, ``n_leads``, ``lead_names``, ``duration_s`` and ``leads_ok``; when the twelve
        leads are identified, ``sha256_full`` and ``missing_samples_full``, and for records of at least
        10 s the fields of ``window_summary``.
    """
    row: dict[str, object] = {"fs": fs, "n_samples": len(signal), "n_leads": signal.shape[1],
                              "lead_names": ",".join(names), "duration_s": len(signal) / fs}
    try:
        ordered = canonical_order(signal, names)
    except ValueError:
        return {**row, "leads_ok": False}
    micro = microvolts(ordered)
    row |= {"leads_ok": True, "sha256_full": array_sha256(micro),
            "missing_samples_full": int(np.isnan(ordered).sum())}
    samples = int(SECONDS * fs)
    if len(ordered) >= samples:
        row |= window_summary(ordered[:samples], micro[:samples])
    return row


def scan(job: dict[str, object]) -> dict[str, object]:
    """
    Verify, read and summarize one record.

    Parameters
    ----------
    job : dict[str, object]
        Output of ``wfdb_job`` or ``sph_job``.

    Returns
    -------
    dict[str, object]
        ``record_key``, file hashes, header fields (Challenge), ``readable``, ``read_error`` and the fields
        of ``summarize``.
    """
    if job["source"] in RELEASES:
        if not GUARD["closed"] or job["guard"] != challenge_key(job["source"], job["relative"]):
            raise PermissionError("The closed-path guard is not set for a Challenge job")
        require_open(job["guard"], GUARD["closed"])
    row: dict[str, object] = {"record_key": job["key"], **verify_files(job["files"])}
    if job["source"] in RELEASES:
        row |= header_fields(job["files"]["header"][0])
    try:
        signal, fs, names = read_signal(job)
    except (OSError, ValueError, IndexError, KeyError) as error:
        return {**row, "readable": False, "read_error": f"{type(error).__name__}: {error}"}
    row |= {"readable": True, "read_error": "", **summarize(signal, fs, names),
            "worker_max_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}
    return row


def ptbxl_records(table: pd.DataFrame) -> pd.DataFrame:
    """
    One row per PTB-XL record with its role, patient group and codes.

    Parameters
    ----------
    table : pd.DataFrame
        ``ptbxl_database.csv`` indexed by ``ecg_id``.

    Returns
    -------
    pd.DataFrame
        Manifest identity columns; folds 1-8 train, fold 9 validation, fold 10 same-source test.
    """
    fold = table["strat_fold"].astype(int)
    record = table.index.astype(str)
    patient = table["patient_id"].astype(int).astype(str)
    return pd.DataFrame({
        "record_key": "ptbxl:" + record, "source": "ptbxl", "family": "ptbxl", "record": record,
        "role": np.select([fold <= 8, fold == 9], ["train", "validation"], "test_same_source"),
        "origin": "fold_" + fold.astype(str), "group_id": "ptbxl:" + patient, "group_kind": "patient",
        "raw_path": [to_stored(PTBXL_DIR / name) for name in table["filename_hr"]],
        "age": table["age"].where(table["age"] < 300).astype(float), "age_over_89": table["age"] == 300,
        "sex": table["sex"].map({0: "M", 1: "F"}), "raw_codes": table["scp_codes"],
        "heart_axis": table["heart_axis"].fillna(""),
    }).reset_index(drop=True)


def challenge_records(rows: pd.DataFrame) -> pd.DataFrame:
    """
    One row per open Challenge record; age, sex and codes come later from its header.

    Parameters
    ----------
    rows : pd.DataFrame
        Output of ``open_challenge_rows``.

    Returns
    -------
    pd.DataFrame
        Manifest identity columns; train groups train, calibration groups validation. Each record is its
        own group because the Challenge headers carry no patient identifier.
    """
    key = rows["source"] + ":" + rows["record"]
    return pd.DataFrame({
        "record_key": key, "source": rows["source"], "family": rows["family"], "record": rows["record"],
        "role": rows["split"].map({"train": "train", "calibration": "validation"}), "origin": rows["split"],
        "group_id": key, "group_kind": "record",
        "raw_path": [to_stored(RELEASES[source] / path) for source, path in zip(rows["source"], rows["path"],
                                                                                 strict=True)],
        "heart_axis": "",
    })


def sph_records(meta: pd.DataFrame) -> pd.DataFrame:
    """
    One row per SPH record, with the patient-hash half as role.

    Parameters
    ----------
    meta : pd.DataFrame
        SPH ``metadata.csv`` read as text.

    Returns
    -------
    pd.DataFrame
        Manifest identity columns; the evaluation half is the external test, the other half the SPH
        calibration set.
    """
    half = meta["Patient_ID"].map(patient_role)
    return pd.DataFrame({
        "record_key": "sph:" + meta["ECG_ID"], "source": "sph", "family": "sph", "record": meta["ECG_ID"],
        "role": half.map({"evaluation": "test_external", "calibration": "sph_calibration"}),
        "origin": "half_" + half, "group_id": "sph:" + meta["Patient_ID"], "group_kind": "patient",
        "raw_path": [to_stored(sph.RECORDS / f"{record}.h5") for record in meta["ECG_ID"]],
        "age": pd.to_numeric(meta["Age"], errors="coerce"), "age_over_89": False,
        "sex": meta["Sex"].where(meta["Sex"].isin(["M", "F"])), "raw_codes": meta["AHA_Code"],
        "heart_axis": "", "metadata_samples": pd.to_numeric(meta["N"]),
    })


def parsed_codes(source: str, raw: str, heart_axis: str) -> object:
    """
    Parse a record's codes into the form ``mlc_labels`` expects.

    Parameters
    ----------
    source : str
        One of ``mlc_labels.SOURCES``.
    raw : str
        PTB-XL ``scp_codes`` text, Challenge ``Dx`` field or SPH ``AHA_Code``.
    heart_axis : str
        PTB-XL ``heart_axis``; empty elsewhere.

    Returns
    -------
    object
        Statement-likelihood dict, SNOMED code set or SPH statement tuple.
    """
    vocabulary = mlc_labels.VOCABULARY[source]
    if vocabulary == "scp":
        return mlc_labels.ptbxl_codes(ast.literal_eval(raw), heart_axis)
    if vocabulary == "aha":
        return mlc_labels.parse_aha(raw)
    return mlc_labels.parse_snomed(raw)


def label_frame(frame: pd.DataFrame, classes: dict[str, str], diagnostic: frozenset[str]) -> pd.DataFrame:
    """
    Every label column of every record.

    Parameters
    ----------
    frame : pd.DataFrame
        With ``source``, ``raw_codes`` and ``heart_axis``.
    classes : dict[str, str]
        PTB-XL diagnostic statement to superclass.
    diagnostic : frozenset[str]
        PTB-XL diagnostic statements.

    Returns
    -------
    pd.DataFrame
        ``mlc_labels.record_labels`` columns on the index of ``frame``. A record without any code has every
        state undefined.
    """
    rows = []
    for source, raw, axis in zip(frame["source"], frame["raw_codes"], frame["heart_axis"], strict=True):
        record = parsed_codes(source, raw, axis)
        labels = mlc_labels.record_labels(source, record, classes, diagnostic)
        if not record:
            labels = {name: False if name.startswith("lik50_") else np.nan for name in labels}
        rows.append(labels)
    return pd.DataFrame(rows, index=frame.index)


def exclusion_reason(frame: pd.DataFrame) -> pd.Series:
    """
    First inclusion rule each record fails, or an empty string.

    Parameters
    ----------
    frame : pd.DataFrame
        Manifest rows joined with the scan.

    Returns
    -------
    pd.Series
        One of ``EXCLUSIONS`` or ``""``.
    """
    rules = {
        "upstream_corrupted": frame["record_key"].isin(UPSTREAM_EXCLUDED),
        "unreadable": ~frame["readable"].fillna(False).astype(bool),
        "wrong_leads": ~frame["leads_ok"].fillna(False).astype(bool),
        "wrong_rate": frame["fs"] != RATE,
        "short": ~(frame["duration_s"] >= SECONDS),
    }
    reason = pd.Series("", index=frame.index)
    for name in reversed(EXCLUSIONS):
        reason[rules[name].to_numpy()] = name
    return reason


def survivor_status(labels: pd.DataFrame) -> pd.Series:
    """
    Resolve identical copies that share the most protected role of their group.

    Parameters
    ----------
    labels : pd.DataFrame
        Label states (1.0, 0.0, NaN) of the copies, indexed by record key.

    Returns
    -------
    pd.Series
        ``dropped_conflict`` for every copy when one copy is positive and another negative for any label;
        otherwise ``kept`` for the copy with the most defined labels (ties: lowest key) and ``dropped_copy``
        for the others.
    """
    values = labels.to_numpy(dtype=float)
    if ((values == 1).any(axis=0) & (values == 0).any(axis=0)).any():
        return pd.Series("dropped_conflict", index=labels.index)
    defined = (~np.isnan(values)).sum(axis=1)
    keep = min(range(len(labels)), key=lambda index: (-defined[index], labels.index[index]))
    status = pd.Series("dropped_copy", index=labels.index)
    status.iloc[keep] = "kept"
    return status


def group_status(members: pd.DataFrame, label_columns: list[str]) -> pd.Series:
    """
    Resolve one group of records with an identical first 10 s.

    Parameters
    ----------
    members : pd.DataFrame
        The group's records, indexed by record key, with ``role`` and the label columns.
    label_columns : list[str]
        Labels compared between copies.

    Returns
    -------
    pd.Series
        ``removed_cross_role`` for copies in a less protected role than the group's most protected one
        (``ROLE_RANK``: tests, then validation, then SPH calibration, then training); copies in the most
        protected role or roles are resolved by ``survivor_status`` within each role.
    """
    rank = members["role"].map(ROLE_RANK)
    status = pd.Series("removed_cross_role", index=members.index)
    for _, survivors in members[rank == rank.min()].groupby("role"):
        status[survivors.index] = survivor_status(survivors[label_columns])
    return status


def resolve_duplicates(frame: pd.DataFrame, label_columns: list[str]) -> pd.DataFrame:
    """
    Group eligible records by their first-10-s hash and resolve every group.

    Parameters
    ----------
    frame : pd.DataFrame
        Eligible records indexed by record key, with ``sha256_10s``, ``role`` and the label columns.
    label_columns : list[str]
        Labels compared between copies.

    Returns
    -------
    pd.DataFrame
        ``duplicate_group`` (lowest key of the group), ``duplicate_size`` and ``duplicate_status``
        (``unique``, ``kept``, ``dropped_copy``, ``dropped_conflict`` or ``removed_cross_role``).
    """
    keys = frame.index.to_series()
    size = frame.groupby("sha256_10s")["role"].transform("size")
    status = pd.Series("unique", index=frame.index)
    for _, members in frame[size > 1].groupby("sha256_10s"):
        status[members.index] = group_status(members, label_columns)
    return pd.DataFrame({"duplicate_group": keys.groupby(frame["sha256_10s"]).transform("min"),
                         "duplicate_size": size, "duplicate_status": status})


def label_columns(frame: pd.DataFrame) -> list[str]:
    """
    Label-state columns compared between duplicate copies: spaces A, B, C and the normal column.

    Parameters
    ----------
    frame : pd.DataFrame
        Output of ``label_frame`` or a frame containing it.

    Returns
    -------
    list[str]
        Column names in frame order.
    """
    return [name for name in frame.columns if name.startswith(("a_", "b_", "c_")) or name == "normal"]


def mapping_table(scp: dict[str, str], snomed: dict[str, str], aha: dict[str, str]) -> pd.DataFrame:
    """
    Describe every code of the frozen mapping from its official table.

    Parameters
    ----------
    scp, snomed, aha : dict[str, str]
        Code to description from ``scp_statements.csv``, the official SNOMED tables and SPH ``code.csv``.

    Returns
    -------
    pd.DataFrame
        ``mlc_labels.mapping_rows`` with a ``description`` column.

    Raises
    ------
    KeyError
        If a code is absent from its official table.
    """
    rows = pd.DataFrame(mlc_labels.mapping_rows())
    texts = []
    for vocabulary, code in zip(rows["vocabulary"], rows["code"], strict=True):
        if not code:
            texts.append("")
        elif vocabulary == "scp":
            texts.append(f"heart_axis = {code[5:]}" if code.startswith("axis:") else scp[code])
        elif vocabulary == "snomed":
            texts.append(snomed[code])
        else:
            base, *modifier = code.rstrip("=").split("+")
            suffix = " without modifier" if code.endswith("=") else ""
            texts.append(aha[base] + (f" with {aha[modifier[0]]}" if modifier else suffix))
    return rows.assign(description=texts)
