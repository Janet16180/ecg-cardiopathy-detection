"""Whole-patient selection and waveform audit for the MIMIC-IV-ECG SSL pool.

Only record_list.csv supplies selection and patient identity. No diagnoses,
machine measurements, ECG times, or cardiologist reports enter the manifest.
"""

from __future__ import annotations

import csv
import hashlib
import json
import re
import sqlite3
import time
from collections import Counter, defaultdict
from contextlib import closing
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import wfdb

from .files import sha256_file, write_csv_atomic, write_json_atomic
from .public_sources import signal_sha256
from .waveforms import read_record

BASE = "https://physionet.org/files/mimic-iv-ecg/1.0"
FIELDS = ("ecg_id", "patient_id", "raw_dir", "filename_hr", "source")
SHA_LINE = re.compile(r"([0-9a-fA-F]{64})[ \t]+\*?(?:\./)?(.+)")
PATH = re.compile(r"files/p(\d{4})/p(\d{8})/s(\d{8})/(\d{8})")


def safe_record_path(subject: str, study: str, name: str) -> bool:
    """
    Check that an official record path matches its subject and study.

    Parameters
    ----------
    subject : str
        MIMIC ``subject_id``.
    study : str
        MIMIC ``study_id``.
    name : str
        Relative waveform path from ``record_list.csv``, without extension.

    Returns
    -------
    bool
        Whether ``name`` has the expected ``files/pXXXX/pSUBJECT/sSTUDY/STUDY`` form.
    """
    match = PATH.fullmatch(name)
    return bool(match and match[2] == subject and match[1] == subject[:4]
                and match[3] == study and match[4] == study)


def read_patients(path: Path) -> dict[str, list[tuple[str, str]]]:
    """
    Group official MIMIC-IV-ECG records by patient.

    Only identity and path columns are used; ECG times are ignored.

    Parameters
    ----------
    path : Path
        Official ``record_list.csv``.

    Returns
    -------
    dict[str, list[tuple[str, str]]]
        ``(study_id, path)`` pairs keyed by ``subject_id``.

    Raises
    ------
    ValueError
        If columns differ from the official release, a path is unsafe or
        repeated, or no records are listed.
    """
    patients = defaultdict(list)
    seen = set()
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if set(reader.fieldnames or []) != {"subject_id", "study_id", "file_name", "ecg_time", "path"}:
            raise ValueError("Unexpected official record_list.csv columns")
        for row in reader:
            subject, study, name = row["subject_id"], row["study_id"], row["path"]
            if not safe_record_path(subject, study, name) or row["file_name"] != study:
                raise ValueError(f"Unsafe or inconsistent MIMIC waveform path: {name!r}")
            if name in seen:
                raise ValueError(f"Duplicate official waveform path: {name}")
            seen.add(name)
            patients[subject].append((study, name))
    if not patients:
        raise ValueError("Official record_list.csv contains no ECGs")
    return patients


def select_patients(patients: dict[str, list[tuple[str, str]]], seed: int,
                    max_records: int) -> tuple[list[tuple[str, str, str]], list[str]]:
    """
    Choose whole patients by seeded SHA-256 rank, stopping before cap overflow.

    Parameters
    ----------
    patients : dict[str, list[tuple[str, str]]]
        Output of ``read_patients``.
    seed : int
        Seed mixed into each subject's rank.
    max_records : int
        Maximum number of selected records.

    Returns
    -------
    tuple[list[tuple[str, str, str]], list[str]]
        Selected ``(subject_id, study_id, path)`` rows in selection order, and
        the chosen subjects.

    Raises
    ------
    ValueError
        If ``max_records`` is not positive or no whole patient fits.
    """
    if max_records < 1:
        raise ValueError("max_records must be positive")
    ranking = sorted(patients, key=lambda subject:
                     (hashlib.sha256(f"{seed}:{subject}".encode()).digest(), subject))
    selected, subjects = [], []
    for subject in ranking:
        records = sorted(patients[subject])
        if len(selected) + len(records) > max_records:
            break
        subjects.append(subject)
        selected.extend((subject, study, name) for study, name in records)
    if not selected:
        raise ValueError("No whole patient fits under max_records")
    return selected, subjects


def selection_hash(rows: list[tuple[str, str, str]], seed: int, cap: int,
                   list_hash: str) -> str:
    """
    Identify a patient selection and the inputs that produced it.

    Parameters
    ----------
    rows : list[tuple[str, str, str]]
        Selected ``(subject_id, study_id, path)`` rows in order.
    seed : int
        Selection seed.
    cap : int
        Selection record cap.
    list_hash : str
        SHA-256 of the official ``record_list.csv``.

    Returns
    -------
    str
        Hexadecimal SHA-256 digest.
    """
    digest = hashlib.sha256()
    digest.update(json.dumps({"seed": seed, "max_records": cap,
                              "record_list_sha256": list_hash}, sort_keys=True).encode() + b"\n")
    for row in rows:
        digest.update("\t".join(row).encode() + b"\n")
    return digest.hexdigest()


def required_checksums(manifest: Path, paths: set[str]) -> dict[str, str]:
    """
    Read the official checksums of the metadata files and selected records.

    Parameters
    ----------
    manifest : Path
        Official ``SHA256SUMS.txt``.
    paths : set[str]
        Selected record paths without extension; both ``.hea`` and ``.dat``
        are required.

    Returns
    -------
    dict[str, str]
        Lowercase digest keyed by relative file name.

    Raises
    ------
    ValueError
        If a line is malformed, a required file repeats, or any is missing.
    """
    wanted = {"record_list.csv", "LICENSE.txt"} | {name + extension for name in paths
                                                   for extension in (".hea", ".dat")}
    found = {}
    with manifest.open(encoding="utf-8") as handle:
        for line in handle:
            match = SHA_LINE.fullmatch(line.rstrip("\r\n"))
            if match is None:
                raise ValueError("Malformed official SHA256SUMS.txt line")
            digest, name = match.groups()
            if name in wanted:
                if name in found:
                    raise ValueError(f"Duplicate official checksum for {name}")
                found[name] = digest.lower()
    missing = wanted - found.keys()
    if missing:
        raise ValueError(f"Official SHA256SUMS.txt lacks {len(missing)} required files, e.g. {min(missing)}")
    return found


def ptbxl_hashes(ptbxl_dir: Path) -> tuple[set[str], int]:
    """
    Decode every PTB-XL record to build the exact-duplicate reference.

    Parameters
    ----------
    ptbxl_dir : Path
        PTB-XL release root holding ``ptbxl_database.csv``.

    Returns
    -------
    tuple[set[str], int]
        Signal identities from ``signal_sha256`` and the number of records read.

    Raises
    ------
    ValueError
        If metadata lacks ``filename_hr`` or lists no records.
    """
    hashes = set()
    with (ptbxl_dir / "ptbxl_database.csv").open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if "filename_hr" not in (reader.fieldnames or []):
            raise ValueError("PTB-XL metadata lacks filename_hr")
        count = 0
        for row in reader:
            hashes.add(signal_sha256(read_record(ptbxl_dir, row["filename_hr"])))
            count += 1
            if count % 5000 == 0:
                print(f"Audited {count:,} PTB-XL ECG identities", flush=True)
    if count == 0:
        raise ValueError("No PTB-XL records for leakage audit")
    return hashes, count


def check_waveform(raw_dir: Path, name: str) -> np.ndarray:
    """
    Decode one MIMIC record after checking its storage contract.

    Parameters
    ----------
    raw_dir : Path
        MIMIC-IV-ECG release root.
    name : str
        Relative record path without extension.

    Returns
    -------
    np.ndarray
        Canonical ``[12, 5000]`` physical-mV signal.

    Raises
    ------
    ValueError
        If the data file size or physical units are unexpected.
    """
    if (raw_dir / (name + ".dat")).stat().st_size != 12 * 5000 * 2:
        raise ValueError("Expected 120,000-byte 16-bit 12-lead waveform data")
    header = wfdb.rdheader(str(raw_dir / name))
    if len(header.units) != 12 or any(unit != "mV" for unit in header.units):
        raise ValueError(f"Expected 12 physical-mV leads, found {header.units}")
    return read_record(raw_dir, name)


def audit(rows: list[tuple[str, str, str]], raw_dir: Path, output_dir: Path,
          selection_digest: str, ptb_hashes: set[str]) -> tuple[list[dict[str, str]], Counter]:
    """
    Checkpoint exclusions and identities in SQLite; resume in selection order.

    Records are excluded when they break the input contract or exactly repeat
    a PTB-XL or earlier MIMIC signal.

    Parameters
    ----------
    rows : list[tuple[str, str, str]]
        Selected ``(subject_id, study_id, path)`` rows.
    raw_dir : Path
        MIMIC-IV-ECG release root.
    output_dir : Path
        Directory holding ``audit.sqlite3``.
    selection_digest : str
        Output of ``selection_hash``; an existing audit must match it.
    ptb_hashes : set[str]
        PTB-XL signal identities from ``ptbxl_hashes``.

    Returns
    -------
    tuple[list[dict[str, str]], Counter]
        Accepted manifest rows and exclusion counts by reason.

    Raises
    ------
    ValueError
        If the audit database belongs to another selection or its cached
        outcomes are inconsistent.
    """
    db = sqlite3.connect(output_dir / "audit.sqlite3")
    try:
        db.execute("CREATE TABLE IF NOT EXISTS state (key TEXT PRIMARY KEY, value TEXT NOT NULL)")
        db.execute("CREATE TABLE IF NOT EXISTS outcomes (name TEXT PRIMARY KEY, status TEXT NOT NULL, "
                   "signal_sha256 TEXT, reason TEXT, detail TEXT)")
        previous = db.execute("SELECT value FROM state WHERE key='selection_sha256'").fetchone()
        if previous and previous[0] != selection_digest:
            raise ValueError("Existing audit has different patient selection; use another output directory")
        db.execute("INSERT OR IGNORE INTO state VALUES ('selection_sha256', ?)", (selection_digest,))
        db.commit()
        seen = set()
        accepted = []
        reasons = Counter()
        started = time.monotonic()
        for index, (subject, study, name) in enumerate(rows, 1):
            status, digest, reason = _audit_outcome(db, raw_dir, name, ptb_hashes, seen)
            if status == "accepted":
                seen.add(digest)
                accepted.append({"ecg_id": "mimic:" + study,
                                 "patient_id": "mimic:" + subject,
                                 "raw_dir": str(raw_dir.resolve()),
                                 "filename_hr": name, "source": "mimic"})
            else:
                reasons[reason] += 1
            if index % 500 == 0 or index == len(rows):
                db.commit()
                if index % 5000 == 0 or time.monotonic() - started >= 30 or index == len(rows):
                    print(f"Audited {index:,}/{len(rows):,} MIMIC ECGs; "
                          f"accepted {len(accepted):,}; excluded {sum(reasons.values()):,}", flush=True)
                    started = time.monotonic()
        db.commit()
        return accepted, reasons
    finally:
        db.close()


def _new_outcome(raw_dir: Path, name: str, ptb_hashes: set[str],
                 seen: set[str]) -> tuple[str, str | None, str | None, str | None]:
    """Decode and classify one record not yet in the audit database."""
    try:
        digest = signal_sha256(check_waveform(raw_dir, name))
    except (ValueError, OSError, TypeError, IndexError) as error:
        return "excluded", None, "input_contract", str(error)
    status, reason = "accepted", None
    if digest in ptb_hashes:
        status, reason = "excluded", "exact_duplicate_ptbxl"
    elif digest in seen:
        status, reason = "excluded", "exact_duplicate_mimic"
    return status, digest, reason, None


def _audit_outcome(db: sqlite3.Connection, raw_dir: Path, name: str, ptb_hashes: set[str],
                   seen: set[str]) -> tuple[str, str | None, str | None]:
    """Return the cached or newly recorded ``(status, signal_sha256, reason)`` of one record."""
    outcome = db.execute("SELECT status, signal_sha256, reason, detail FROM outcomes WHERE name=?",
                         (name,)).fetchone()
    if outcome is None:
        outcome = _new_outcome(raw_dir, name, ptb_hashes, seen)
        db.execute("INSERT INTO outcomes VALUES (?, ?, ?, ?, ?)", (name, *outcome))
    status, digest, reason, _ = outcome
    if status == "accepted" and digest in ptb_hashes:
        raise ValueError(f"Previously accepted {name} now matches PTB-XL")
    if status == "accepted" and digest in seen:
        raise ValueError(f"Audit cache has duplicate accepted hash at {name}")
    return status, digest, reason


def lock_selection(output_dir: Path, rows: list[tuple[str, str, str]],
                   digest: str, seed: int, cap: int, list_hash: str) -> None:
    """
    Record the exact chosen subjects/studies before costly waveform fetches.

    Parameters
    ----------
    output_dir : Path
        Output directory; must be empty or already hold this selection.
    rows : list[tuple[str, str, str]]
        Selected ``(subject_id, study_id, path)`` rows.
    digest : str
        Output of ``selection_hash``.
    seed : int
        Selection seed.
    cap : int
        Selection record cap.
    list_hash : str
        SHA-256 of the official ``record_list.csv``.

    Raises
    ------
    ValueError
        If the directory holds another selection or unrelated files.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    config = {"selection_sha256": digest, "seed": seed, "max_records": cap,
              "record_list_sha256": list_hash, "selected_records": len(rows)}
    config_path = output_dir / "selection.json"
    if config_path.is_file():
        if json.loads(config_path.read_text(encoding="utf-8")) != config:
            raise ValueError("Existing output directory uses another patient selection")
        selected_path = output_dir / "selected_records.csv"
        if not selected_path.is_file():
            raise ValueError("Existing selection is missing selected_records.csv")
        with selected_path.open(newline="", encoding="utf-8") as handle:
            reader = csv.reader(handle)
            if next(reader, None) != ["subject_id", "study_id", "path"] or list(map(tuple, reader)) != rows:
                raise ValueError("Existing selected_records.csv differs from deterministic selection")
        return
    if any(output_dir.iterdir()):
        raise ValueError("Output directory contains files but no selection lock")
    header = ("subject_id", "study_id", "path")
    selected = (dict(zip(header, row, strict=True)) for row in rows)
    write_csv_atomic(output_dir / "selected_records.csv", selected, header)
    write_json_atomic(config_path, config)


@dataclass(frozen=True)
class SelectionProvenance:
    """
    Selection inputs and source counts recorded in ``metadata.json``.

    Attributes
    ----------
    seed : int
        Selection seed.
    max_records : int
        Selection record cap.
    selection_sha256 : str
        Output of ``selection_hash``.
    record_list_sha256 : str
        SHA-256 of ``record_list.csv``.
    checksums_sha256 : str
        SHA-256 of ``SHA256SUMS.txt``.
    license_sha256 : str
        SHA-256 of ``LICENSE.txt``.
    source_records : int
        Records in the official list.
    source_patients : int
        Patients in the official list.
    ptbxl_records_checked : int
        PTB-XL records checked for exact duplicates.
    """

    seed: int
    max_records: int
    selection_sha256: str
    record_list_sha256: str
    checksums_sha256: str
    license_sha256: str
    source_records: int
    source_patients: int
    ptbxl_records_checked: int


def write_outputs(output_dir: Path, raw_dir: Path, rows: list[tuple[str, str, str]],
                  subjects: list[str], accepted: list[dict[str, str]], reasons: Counter,
                  stats: dict[str, int], provenance: SelectionProvenance) -> None:
    """
    Write the SSL manifest, exclusion list and provenance metadata.

    Parameters
    ----------
    output_dir : Path
        Directory holding the audit database; outputs are written here.
    raw_dir : Path
        MIMIC-IV-ECG release root recorded in the metadata.
    rows : list[tuple[str, str, str]]
        Selected rows.
    subjects : list[str]
        Selected subjects.
    accepted : list[dict[str, str]]
        Accepted manifest rows from ``audit``.
    reasons : Counter
        Exclusion counts from ``audit``.
    stats : dict[str, int]
        Download statistics.
    provenance : SelectionProvenance
        Selection inputs and source counts.
    """
    destination = output_dir / "ssl_manifest.csv"
    write_csv_atomic(destination, accepted, FIELDS)
    exclusion_path = output_dir / "exclusions.csv"
    exclusion_fields = ("filename_hr", "reason", "detail")
    with closing(sqlite3.connect(output_dir / "audit.sqlite3")) as db:
        exclusions = db.execute(
            "SELECT name, reason, detail FROM outcomes WHERE status='excluded' ORDER BY name")
        excluded = (dict(zip(exclusion_fields, row, strict=True)) for row in exclusions)
        write_csv_atomic(exclusion_path, excluded, exclusion_fields)
    metadata = {
        "source_url": BASE, "license": "PhysioNet MIMIC-IV-ECG 1.0; see official LICENSE.txt",
        "selection": ("Seeded SHA256 rank of subject_id; take complete patients until next would "
                      "exceed max_records"),
        "seed": provenance.seed, "max_records": provenance.max_records,
        "source_records": provenance.source_records, "source_patients": provenance.source_patients,
        "selected_records": len(rows),
        "selected_patients": len(subjects), "accepted_records": len(accepted),
        "accepted_patients": len({r["patient_id"] for r in accepted}),
        "exclusion_counts": dict(sorted(reasons.items())),
        "selection_sha256": provenance.selection_sha256,
        "record_list_sha256": provenance.record_list_sha256,
        "official_checksums_sha256": provenance.checksums_sha256,
        "license_sha256": provenance.license_sha256,
        "selected_records_sha256": sha256_file(output_dir / "selected_records.csv"),
        "manifest_sha256": sha256_file(destination), "exclusions_sha256": sha256_file(exclusion_path),
        "ptbxl_records_checked": provenance.ptbxl_records_checked,
        "patient_identity": ("subject_id from official MIMIC-IV-ECG record_list.csv; all selected "
                             "records of each chosen subject kept before waveform audit"),
        "signal_identity": "SHA256 of canonical lead-ordered float32 physical-mV 12 x 5000 decoded samples",
        "contract": "12 named leads, physical mV, 500 Hz, 5000 samples, finite decoded signal",
        "labels": "No diagnostic labels, machine measurements, reports, or ECG times used",
        "raw_dir": str(raw_dir.resolve()),
        "download": stats,
    }
    write_json_atomic(output_dir / "metadata.json", metadata)
    print(json.dumps({"manifest": str(destination), "selected_records": len(rows),
                      "accepted_records": len(accepted), "excluded": dict(reasons)}, indent=2), flush=True)

