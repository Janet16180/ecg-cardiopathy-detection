"""Build the MLC-01 condition dataset: manifest, count tables and receipt. No model is fitted or scored.

The waveform scan is cached per source in chunks of ``CHUNK`` records, each bound to the hash of its job
list and of the scan code, so an interrupted build resumes without losing finished chunks.
"""

from __future__ import annotations

import importlib.metadata
import json
import os
import resource
import shutil
import subprocess
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import pandas as pd

from ecg_experiment import ROOT, challenge_labels, mlc01_dataset, mlc01_tables, mlc_labels, sph
from ecg_experiment.eda import ptbxl as ptbxl_eda
from ecg_experiment.eda.signals import LEADS
from ecg_experiment.files import sha256_file, sha256_json, write_json_atomic
from ecg_experiment.mlc01_dataset import (
    CHALLENGE_SPLIT_RECORDS,
    PTBXL_DIR,
    PTBXL_GROUP_RECORDS,
    RELEASES,
    SPH_PATIENTS,
    SPH_RECORDS,
    SPH_ROWS,
    SPLIT_TABLE,
)
from ecg_experiment.paths import to_stored

PROTOCOL = ROOT / "docs/mlc01-condition-dataset-protocol-2026-11-10.md"
CHUNK = 2000
CODE = ("ecg_experiment/mlc_labels.py", "ecg_experiment/mlc01_dataset.py", "ecg_experiment/mlc01_tables.py",
        "ecg_experiment/mlc01_build.py", "scripts/data/build_mlc01_dataset.py",
        "ecg_experiment/challenge_labels.py", "ecg_experiment/challenge_splits.py", "ecg_experiment/sph.py",
        "ecg_experiment/eda/signals.py", "ecg_experiment/eda/ptbxl.py",
        "ecg_experiment/measurement_screening042.py", "ecg_experiment/files.py", "ecg_experiment/paths.py")
INPUTS = (PTBXL_DIR / "ptbxl_database.csv", PTBXL_DIR / "scp_statements.csv", PTBXL_DIR / "SHA256SUMS.txt",
          ROOT / "data/raw/challenge-2021/1.0.3/SHA256SUMS.txt",
          ROOT / "data/raw/challenge-2020/1.0.2/SHA256SUMS.txt", SPLIT_TABLE,
          challenge_labels.MAPPING_DIR / "dx_mapping_scored.csv",
          challenge_labels.MAPPING_DIR / "dx_mapping_unscored.csv", sph.METADATA,
          sph.EXTRACTED / "17912507/code.csv", sph.EXTRACTED / "download_manifest.json", SPH_ROWS, PROTOCOL)
FLAG_COLUMNS = [f"{kind}_{lead}" for kind in ("flat", "missing") for lead in LEADS]
PACKAGES = ("numpy", "pandas", "wfdb", "h5py", "pyarrow", "scipy")


def log(path: Path, message: str) -> None:
    """
    Print a timestamped line and append it to the build log.

    Parameters
    ----------
    path : Path
        Log file.
    message : str
        Text to record.
    """
    line = f"{time.strftime('%Y-%m-%d %H:%M:%S')} {message}"
    print(line, flush=True)
    with path.open("a") as stream:
        stream.write(line + "\n")


def check(checks: dict[str, dict[str, object]], name: str, expected: object, observed: object) -> None:
    """
    Record one integrity check and stop the build when it fails.

    Parameters
    ----------
    checks : dict[str, dict[str, object]]
        Collected checks; updated in place.
    name : str
        Check name.
    expected, observed : object
        Values that must be equal.

    Raises
    ------
    ValueError
        If ``observed`` differs from ``expected``.
    """
    checks[name] = {"expected": expected, "observed": observed, "passed": expected == observed}
    if expected != observed:
        raise ValueError(f"Integrity check failed: {name}: expected {expected}, observed {observed}")


def load_inputs(checks: dict[str, dict[str, object]]) -> dict[str, object]:
    """
    Read and check every metadata input.

    Parameters
    ----------
    checks : dict[str, dict[str, object]]
        Collected integrity checks; updated in place.

    Returns
    -------
    dict[str, object]
        PTB-XL table, open Challenge rows, closed Challenge keys, SPH metadata, the official SNOMED table,
        PTB-XL statements, SPH code table and the SPH small-file MD5s.
    """
    rows = mlc01_dataset.read_split_table()
    check(checks, "challenge_split_table_sha256", mlc01_dataset.SPLIT_TABLE_SHA256, sha256_file(SPLIT_TABLE))
    open_rows = mlc01_dataset.open_challenge_rows(rows)
    closed = mlc01_dataset.closed_challenge_keys(rows)
    found = open_rows.groupby(["source", "split"]).size().to_dict()
    check(checks, "challenge_open_records", {f"{s}:{p}": n for (s, p), n in CHALLENGE_SPLIT_RECORDS.items()},
          {f"{s}:{p}": int(n) for (s, p), n in found.items()})
    ptbxl = pd.read_csv(PTBXL_DIR / "ptbxl_database.csv", index_col="ecg_id")
    groups = pd.cut(ptbxl["strat_fold"], [0, 8, 9, 10], labels=list(PTBXL_GROUP_RECORDS))
    check(checks, "ptbxl_fold_records", PTBXL_GROUP_RECORDS,
          {name: int(n) for name, n in groups.value_counts().sort_index().items()})
    meta = pd.read_csv(sph.METADATA, dtype=str)
    check(checks, "sph_records_and_patients", [SPH_RECORDS, SPH_PATIENTS],
          [len(meta), int(meta["Patient_ID"].nunique())])
    return {"ptbxl": ptbxl, "open_rows": open_rows, "closed": closed, "sph": meta,
            "official": challenge_labels.load_official(), "statements": ptbxl_eda.load_statements(),
            "aha": pd.read_csv(sph.EXTRACTED / "17912507/code.csv", dtype=str).set_index("Code"),
            "sph_md5": sph.verify_small_files(), "closed_records": len(closed)}


def check_sph_roles(meta: pd.DataFrame, checks: dict[str, dict[str, object]]) -> None:
    """
    Require the patient hash to reproduce the SPH roles of the earlier baseline table.

    Parameters
    ----------
    meta : pd.DataFrame
        SPH ``metadata.csv`` read as text.
    checks : dict[str, dict[str, object]]
        Collected integrity checks; updated in place.
    """
    earlier = pd.read_csv(SPH_ROWS, usecols=["ecg_id", "patient_id", "role"], dtype=str)
    half = meta.set_index("ECG_ID")["Patient_ID"].map(mlc01_dataset.patient_role)
    observed = earlier["ecg_id"].map(half)
    agree = int((observed == earlier["role"]).sum())
    check(checks, "sph_patient_hash_reproduces_earlier_roles", len(earlier), agree)


def build_jobs(inputs: dict[str, object], limit: int | None) -> dict[str, list[dict[str, object]]]:
    """
    Scan jobs per source, Challenge jobs only for open records.

    Parameters
    ----------
    inputs : dict[str, object]
        Output of ``load_inputs``.
    limit : int | None
        Keep only the first ``limit`` jobs per source, for a profile run.

    Returns
    -------
    dict[str, list[dict[str, object]]]
        Jobs keyed by source in ``mlc_labels.SOURCES`` order.
    """
    ptbxl = inputs["ptbxl"]
    names = {f"{stem}{suffix}" for stem in ptbxl["filename_hr"] for suffix in (".hea", ".dat")}
    sums = mlc01_dataset.checksums(PTBXL_DIR / "SHA256SUMS.txt", names)
    jobs = {"ptbxl": [mlc01_dataset.wfdb_job("ptbxl", str(ecg_id), f"ptbxl:{ecg_id}", PTBXL_DIR, stem, sums)
                      for ecg_id, stem in ptbxl["filename_hr"].items()]}
    rows = inputs["open_rows"]
    for source, root in RELEASES.items():
        part = rows[rows["source"] == source]
        names = {f"{path}{suffix}" for path in part["path"] for suffix in (".hea", ".mat")}
        sums = mlc01_dataset.checksums(root / "SHA256SUMS.txt", names)
        jobs[source] = [mlc01_dataset.wfdb_job(source, record, f"{source}:{record}", root, path, sums)
                        for record, path in zip(part["record"], part["path"], strict=True)
                        if f"{source}:{record}" not in mlc01_dataset.UPSTREAM_EXCLUDED]
    jobs["sph"] = [mlc01_dataset.sph_job(record) for record in inputs["sph"]["ECG_ID"]]
    closed = inputs["closed"]
    if any(job["guard"] in closed for source in RELEASES for job in jobs[source]):
        raise PermissionError("A closed Challenge record reached the scan jobs")
    return {source: items[:limit] if limit else items for source, items in jobs.items()}


def normalize_scan(rows: list[dict[str, object]]) -> pd.DataFrame:
    """
    Scan rows as a frame with stable column types.

    Parameters
    ----------
    rows : list[dict[str, object]]
        Outputs of ``mlc01_dataset.scan``.

    Returns
    -------
    pd.DataFrame
        One row per record; flag columns are nullable booleans.
    """
    frame = pd.DataFrame(rows)
    for name in [*FLAG_COLUMNS, "readable", "leads_ok", "checksum_verified"]:
        if name not in frame:
            frame[name] = pd.NA
        frame[name] = frame[name].astype("boolean")
    return frame


def scan_chunk(pool: Pool, jobs: list[dict[str, object]], stem: Path, code_sha256: str, log_path: Path
               ) -> pd.DataFrame:
    """
    Scan one chunk of jobs, or load it from the cache when it matches the jobs and the code.

    Parameters
    ----------
    pool : Pool
        Worker pool initialized with the closed Challenge paths.
    jobs : list[dict[str, object]]
        Scan jobs.
    stem : Path
        Cache path without suffix.
    code_sha256 : str
        SHA-256 of ``mlc01_dataset.py``.
    log_path : Path
        Build log.

    Returns
    -------
    pd.DataFrame
        Output of ``normalize_scan``.
    """
    binding = {"jobs_sha256": sha256_json(jobs), "code_sha256": code_sha256, "records": len(jobs)}
    sidecar, table = stem.with_suffix(".json"), stem.with_suffix(".parquet")
    if sidecar.exists() and table.exists() and json.loads(sidecar.read_text()) == binding:
        log(log_path, f"cached {stem.parent.name}/{stem.name}: {len(jobs)} records")
        return pd.read_parquet(table)
    started = time.monotonic()
    frame = normalize_scan(list(pool.imap(mlc01_dataset.scan, jobs, chunksize=8)))
    stem.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(table.with_suffix(".tmp"), index=False)
    table.with_suffix(".tmp").rename(table)
    write_json_atomic(sidecar, binding)
    log(log_path, f"scanned {stem.parent.name}/{stem.name}: {len(jobs)} records in "
                  f"{time.monotonic() - started:.1f} s")
    return frame


def run_scan(jobs: dict[str, list[dict[str, object]]], cache: Path, workers: int, closed: frozenset[str],
             log_path: Path) -> pd.DataFrame:
    """
    Scan every record with a process pool, chunk by chunk.

    Parameters
    ----------
    jobs : dict[str, list[dict[str, object]]]
        Output of ``build_jobs``.
    cache : Path
        Chunk cache directory.
    workers : int
        Worker processes.
    closed : frozenset[str]
        Closed Challenge paths, given to every worker.
    log_path : Path
        Build log.

    Returns
    -------
    pd.DataFrame
        Scan rows of every record, with a ``source`` column.
    """
    code_sha256 = sha256_file(ROOT / "ecg_experiment/mlc01_dataset.py")
    frames = []
    with Pool(workers, initializer=mlc01_dataset.init_worker, initargs=(closed,)) as pool:
        for source, items in jobs.items():
            for start in range(0, len(items), CHUNK):
                stem = cache / source / f"chunk_{start // CHUNK:03d}"
                frames.append(scan_chunk(pool, items[start:start + CHUNK], stem, code_sha256, log_path))
    return pd.concat(frames, ignore_index=True)


def identity_frame(inputs: dict[str, object], scan: pd.DataFrame) -> pd.DataFrame:
    """
    Identity, demographics and raw codes of every scanned record.

    Parameters
    ----------
    inputs : dict[str, object]
        Output of ``load_inputs``.
    scan : pd.DataFrame
        Output of ``run_scan``.

    Returns
    -------
    pd.DataFrame
        One row per scanned record, indexed by record key, in source order.
    """
    challenge = mlc01_dataset.challenge_records(inputs["open_rows"]).set_index("record_key")
    header = scan.set_index("record_key").reindex(challenge.index)
    challenge["age"] = pd.to_numeric(header["header_age"], errors="coerce")
    challenge["age_over_89"] = False
    challenge["sex"] = header["header_sex"].map({"Male": "M", "Female": "F"})
    challenge["raw_codes"] = header["header_dx"]
    frames = [mlc01_dataset.ptbxl_records(inputs["ptbxl"]).set_index("record_key"), challenge,
              mlc01_dataset.sph_records(inputs["sph"]).set_index("record_key")]
    frame = pd.concat(frames)
    return frame.loc[frame.index.isin(scan["record_key"])]


def assemble(inputs: dict[str, object], scan: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Manifest of every scanned record: identity, scan, labels, exclusions, duplicates and inclusion.

    The minimum-positive rule (Amendment 1, ruling 6) is applied to the label states of every opened record
    before duplicates are resolved.

    Parameters
    ----------
    inputs : dict[str, object]
        Output of ``load_inputs``.
    scan : pd.DataFrame
        Output of ``run_scan``.

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame]
        The manifest indexed by record key, and ``mlc_labels.minimum_positive_table`` of its labels.
    """
    frame = identity_frame(inputs, scan)
    classes = ptbxl_eda.diagnostic_classes(inputs["statements"])
    labels = mlc01_dataset.label_frame(frame, classes, frozenset(classes))
    states = [name for name in labels.columns if name.startswith(("a_", "b_", "c_"))]
    minimum = mlc_labels.minimum_positive_table(labels[states], frame["source"])
    labels[states] = mlc_labels.apply_minimum_positives(labels[states], frame["source"], minimum)
    scanned = scan.set_index("record_key").drop(columns=["header_age", "header_sex", "header_dx"],
                                                errors="ignore")
    frame = frame.join(scanned).join(labels)
    frame["exclusion_reason"] = mlc01_dataset.exclusion_reason(frame.reset_index()).to_numpy()
    eligible = frame[frame["exclusion_reason"] == ""]
    duplicates = mlc01_dataset.resolve_duplicates(eligible, mlc01_dataset.label_columns(labels))
    frame = frame.join(duplicates)
    frame["duplicate_status"] = frame["duplicate_status"].fillna("")
    frame["included"] = frame["duplicate_status"].isin(mlc01_dataset.KEPT)
    frame["n_flat_leads"] = frame[FLAG_COLUMNS[:12]].fillna(False).sum(axis=1)
    frame["n_missing_leads"] = frame[FLAG_COLUMNS[12:]].fillna(False).sum(axis=1)
    return frame, minimum


def status_table(minimum: pd.DataFrame) -> pd.DataFrame:
    """
    Annotation status of every label and source, with the minimum-positive rule.

    Parameters
    ----------
    minimum : pd.DataFrame
        Output of ``mlc_labels.minimum_positive_table``.

    Returns
    -------
    pd.DataFrame
        ``mlc_labels.status_rows`` where pairs that the rule newly makes undefined read ``not annotated
        (fewer than 5 positives)``.
    """
    status = pd.DataFrame(mlc_labels.status_rows())
    caught = minimum[minimum["not_annotated"] & (minimum["defined"] > 0)]
    keys = set(zip(caught["label"], caught["source"], strict=True))
    pairs = zip(status["label"], status["source"], strict=True)
    flagged = [(label, source) in keys for label, source in pairs]
    status.loc[flagged, "status"] = f"not annotated (fewer than {mlc_labels.MIN_POSITIVES} positives)"
    return status


def check_reference(frame: pd.DataFrame, reference: Path | None, checks: dict[str, dict[str, object]]
                    ) -> None:
    """
    Require the same records, roles, raw paths and waveform hashes as an earlier build.

    Parameters
    ----------
    frame : pd.DataFrame
        Output of ``assemble``.
    reference : Path | None
        Earlier build directory, or None to skip the check.
    checks : dict[str, dict[str, object]]
        Collected integrity checks; updated in place.
    """
    if reference is None:
        return
    columns = ["role", "raw_path", "sha256_10s", "sha256_full"]
    earlier = pd.read_csv(reference / "manifest.csv", index_col="record_key",
                          usecols=["record_key", *columns], dtype=str).fillna("")
    now = frame[columns].astype(str).where(frame[columns].notna(), "")
    same_keys = set(earlier.index) == set(now.index)
    mismatches = int((earlier.reindex(now.index) != now).any(axis=1).sum()) if same_keys else len(now)
    check(checks, "records_roles_paths_and_waveform_hashes_match_reference",
          {"reference": to_stored(reference), "records": len(earlier), "mismatches": 0},
          {"reference": to_stored(reference), "records": len(now), "mismatches": mismatches})


def code_coverage(frame: pd.DataFrame, inputs: dict[str, object]) -> dict[str, list[str]]:
    """
    Codes in use that are absent from their official table.

    Parameters
    ----------
    frame : pd.DataFrame
        Output of ``assemble``.
    inputs : dict[str, object]
        Output of ``load_inputs``.

    Returns
    -------
    dict[str, list[str]]
        Unknown codes per vocabulary.
    """
    found: dict[str, set[str]] = {"scp": set(), "snomed": set(), "aha": set()}
    for source, raw, axis in zip(frame["source"], frame["raw_codes"], frame["heart_axis"], strict=True):
        record = mlc01_dataset.parsed_codes(source, raw, axis)
        vocabulary = mlc_labels.VOCABULARY[source]
        if vocabulary == "aha":
            found["aha"] |= {code for base, modifiers in record for code in (base, *modifiers)}
        else:
            found[vocabulary] |= {code for code in record if not code.startswith("axis:")}
    known = {"scp": set(inputs["statements"].index), "snomed": set(inputs["official"].index),
             "aha": set(inputs["aha"].index)}
    return {name: sorted(found[name] - known[name]) for name in found}


def manifest_table(frame: pd.DataFrame) -> pd.DataFrame:
    """
    Order the manifest columns and store label states as nullable integers (1, 0, empty if undefined).

    Parameters
    ----------
    frame : pd.DataFrame
        Output of ``assemble``.

    Returns
    -------
    pd.DataFrame
        Columns in a fixed order; lead hashes and shape hashes stay in the scan table.
    """
    labels = mlc01_dataset.label_columns(frame)
    flags = [name for name in frame.columns if name.startswith("lik50_")]
    first = ["source", "family", "record", "role", "origin", "group_id", "group_kind", "raw_path",
             "included", "exclusion_reason", "duplicate_status", "duplicate_group", "duplicate_size", "age",
             "age_over_89", "sex", "fs", "n_samples", "duration_s", "n_leads", "lead_names", "readable",
             "read_error", "leads_ok", "checksum_verified", "header_file_sha256", "signal_file_sha256",
             "sha256_10s", "sha256_full", "n_flat_leads", "n_missing_leads", "missing_samples_full",
             *FLAG_COLUMNS, "raw_codes", "heart_axis"]
    table = frame[first + labels + flags].copy()
    for name in labels:
        table[name] = table[name].astype(float).astype("Int8")
    table["duplicate_size"] = table["duplicate_size"].astype("Int64")
    return table


def write_tables(frame: pd.DataFrame, stage: Path) -> dict[str, object]:
    """
    Write every count table.

    Parameters
    ----------
    frame : pd.DataFrame
        Output of ``assemble``.
    stage : Path
        Destination directory.

    Returns
    -------
    dict[str, object]
        Summary values for the receipt.
    """
    spaces = {"A": [f"a_{name}" for name in mlc_labels.SPACE_A],
              "B": [f"b_{key}" for key in mlc_labels.SPACE_B],
              "C": [f"c_{key}" for key in mlc_labels.SPACE_C]}
    for space, columns in spaces.items():
        counts = mlc01_tables.state_counts(frame, columns, space)
        counts.to_csv(stage / f"label_counts_{space}.csv", index=False)
        mlc01_tables.support_table(counts).to_csv(stage / f"support_{space}.csv", index=False)
    counts = mlc01_tables.state_counts(mlc01_tables.sensitivity_frame(frame), spaces["B"], "B_lik50")
    counts.to_csv(stage / "label_counts_B_lik50.csv", index=False)
    mlc01_tables.support_table(counts).to_csv(stage / "support_B_lik50.csv", index=False)
    state_counts = mlc01_tables.state_counts(frame, ["normal"], "normal")
    state_counts.to_csv(stage / "label_counts_normal.csv", index=False)
    for space, prefix in (("B", "b_"), ("C", "c_")):
        keys = mlc_labels.SPACE_B if space == "B" else mlc_labels.SPACE_C
        distribution, pairs = mlc01_tables.cooccurrence(frame, keys, prefix)
        distribution.to_csv(stage / f"cooccurrence_{space}_count.csv", index=False)
        pairs.to_csv(stage / f"cooccurrence_{space}_pairs.csv", index=False)
    flow = mlc01_tables.exclusion_flow(frame)
    flow.to_csv(stage / "exclusion_flow.csv", index=False)
    groups = mlc01_tables.group_counts(frame)
    groups.to_csv(stage / "group_counts.csv", index=False)
    mlc01_tables.lead_flag_counts(frame).to_csv(stage / "lead_flags.csv", index=False)
    duplicate_groups = mlc01_tables.duplicate_groups(frame)
    duplicate_groups.to_csv(stage / "duplicate_groups.csv", index=False)
    accounting = mlc01_tables.duplicate_accounting(duplicate_groups)
    accounting.to_csv(stage / "duplicate_accounting.csv", index=False)
    eligible = frame[frame["exclusion_reason"] == ""]
    shape = mlc01_tables.shape_near_duplicates(eligible)
    shape.to_csv(stage / "near_duplicates_shape.csv", index=False)
    leads = mlc01_tables.lead_near_duplicates(eligible)
    leads.to_csv(stage / "near_duplicates_leads.csv", index=False)
    final = flow[flow["role"] != "all"].pivot_table(index="role", columns="source", values="final",
                                                    aggfunc="sum")
    return {"final_records_by_role_and_source": final.fillna(0).astype(int).to_dict(orient="index"),
            "duplicate_groups": len(duplicate_groups),
            "duplicate_groups_with_different_full_hashes":
                int((duplicate_groups.get("distinct_full_hashes", pd.Series(dtype=int)) > 1).sum()),
            "near_duplicate_shape_groups": len(shape), "near_duplicate_lead_pairs": len(leads)}


def write_outputs(inputs: dict[str, object], frame: pd.DataFrame, scan: pd.DataFrame, stage: Path
                  ) -> dict[str, object]:
    """
    Write the manifest, the scan table, the mapping, the annotation status and every count table.

    Parameters
    ----------
    inputs : dict[str, object]
        Output of ``load_inputs``.
    frame : pd.DataFrame
        Output of ``assemble``.
    scan : pd.DataFrame
        Output of ``run_scan``.
    stage : Path
        Destination directory.

    Returns
    -------
    dict[str, object]
        Output of ``write_tables``.
    """
    manifest_table(frame).to_csv(stage / "manifest.csv")
    scan.to_parquet(stage / "waveform_scan.parquet", index=False)
    mapping = mlc01_dataset.mapping_table(inputs["statements"]["description"].to_dict(),
                                          inputs["official"]["name"].to_dict(),
                                          inputs["aha"]["Description"].to_dict())
    mapping.to_csv(stage / "mapping.csv", index=False)
    return write_tables(frame, stage)


def git_state() -> dict[str, object]:
    """
    Commit and modified tracked files of the working tree.

    Returns
    -------
    dict[str, object]
        ``commit`` and ``modified_tracked_files``.
    """
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True,
                            check=True)
    status = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], cwd=ROOT,
                            capture_output=True, text=True, check=True)
    return {"commit": commit.stdout.strip(), "modified_tracked_files": status.stdout.split("\n")[:-1]}


def record_set_sha256(frame: pd.DataFrame) -> dict[str, str]:
    """
    One SHA-256 per source over its sorted record keys and raw signal-file hashes.

    Parameters
    ----------
    frame : pd.DataFrame
        Output of ``assemble``.

    Returns
    -------
    dict[str, str]
        Digest per source.
    """
    found = {}
    for source, part in frame.groupby("source"):
        lines = sorted(f"{key} {digest}" for key, digest in part["signal_file_sha256"].items())
        found[source] = sha256_json(lines)
    return found


def receipt(inputs: dict[str, object], frame: pd.DataFrame, checks: dict[str, dict[str, object]],
            summary: dict[str, object], run: dict[str, object]) -> dict[str, object]:
    """
    Provenance of the build.

    Parameters
    ----------
    inputs : dict[str, object]
        Output of ``load_inputs``.
    frame : pd.DataFrame
        Output of ``assemble``.
    checks : dict[str, dict[str, object]]
        Integrity checks.
    summary : dict[str, object]
        Output of ``write_tables``.
    run : dict[str, object]
        Command, timing, workers and limit.

    Returns
    -------
    dict[str, object]
        The receipt.
    """
    verified = frame.groupby("source")["checksum_verified"].sum().astype(int).to_dict()
    return {
        "experiment": "MLC-01", "scope": "dataset build only; no model fitted and no score computed",
        "protocol": to_stored(PROTOCOL), "run": run, "git": git_state(),
        "python": sys.version, "packages": {name: importlib.metadata.version(name) for name in PACKAGES},
        "inputs_sha256": {to_stored(path): sha256_file(path) for path in INPUTS},
        "code_sha256": {name: sha256_file(ROOT / name) for name in CODE},
        "sph_small_files_md5": inputs["sph_md5"],
        "raw_files_verified_against_release_checksums": verified,
        "record_set_sha256": record_set_sha256(frame),
        "closed_data": {"challenge_test_records_in_split_table": inputs["closed_records"],
                        "challenge_test_records_opened": 0, "echonext": "not read"},
        "integrity": checks,
        "hash_convention": mlc01_dataset.__doc__.split("\n\n")[1].replace("\n", " "),
        "normal_definitions": {
            "ptbxl": "NORM listed and every statement is NORM or SR",
            "challenge": "426783006 sinus rhythm is the only code; CPSC-Extra undefined (no normal record)",
            "sph": "AHA code 1 alone"},
        "likelihood_sensitivity": "lik50_<condition>: a PTB-XL positive resting only on diagnostic "
                                  "statements with likelihood below 50; undefined in the sensitivity "
                                  "analysis",
        "summary": summary,
    }


def amendment_record(minimum: pd.DataFrame) -> dict[str, object]:
    """
    Receipt entry of Amendment 1: the code lists of rulings 2 and 3 and the pairs caught by ruling 6.

    Parameters
    ----------
    minimum : pd.DataFrame
        Output of ``mlc_labels.minimum_positive_table``.

    Returns
    -------
    dict[str, object]
        Threshold, newly undefined pairs (with their positives and defined records before the rule) and the
        number of flagged pairs that were already undefined throughout.
    """
    flagged = minimum[minimum["not_annotated"]]
    caught = flagged[flagged["defined"] > 0]
    return {"st_t_nonspecific_or_ischemic": {name: sorted(codes) for name, codes in
                                             mlc_labels.ST_T_NONSPECIFIC_OR_ISCHEMIC.items()},
            "ventricular_tachycardia_snomed": list(mlc_labels.VENTRICULAR_TACHYCARDIA_SNOMED),
            "minimum_positives": mlc_labels.MIN_POSITIVES,
            "minimum_positive_pairs_newly_undefined": caught[["source", "label", "positives", "defined"]]
            .to_dict(orient="records"),
            "minimum_positive_pairs_already_undefined": int((flagged["defined"] == 0).sum())}


def build(output: Path, cache: Path, workers: int, limit: int | None, command: list[str],
          reference: Path | None = None) -> dict[str, object]:
    """
    Run the whole build into a staging directory and rename it into place.

    Parameters
    ----------
    output : Path
        Final output directory; must not exist.
    cache : Path
        Scan chunk cache.
    workers : int
        Worker processes.
    limit : int | None
        Records per source for a profile run.
    command : list[str]
        The command line, recorded in the receipt.
    reference : Path | None
        Earlier build whose records, waveform hashes and roles must be reproduced exactly.

    Returns
    -------
    dict[str, object]
        The receipt.

    Raises
    ------
    FileExistsError
        If ``output`` exists.
    PermissionError
        If a closed Challenge record was scanned.
    """
    if output.exists():
        raise FileExistsError(f"Destination exists: {output}")
    stage = output.with_name(output.name + ".partial")
    if stage.exists():
        shutil.rmtree(stage)
    stage.mkdir(parents=True)
    log_path, started, checks = stage / "build.log", time.monotonic(), {}
    log(log_path, f"start {' '.join(command)}")
    inputs = load_inputs(checks)
    check_sph_roles(inputs["sph"], checks)
    jobs = build_jobs(inputs, limit)
    log(log_path, "jobs " + json.dumps({source: len(items) for source, items in jobs.items()}))
    scan = run_scan(jobs, cache, workers, inputs["closed"], log_path)
    scanned = {job["guard"] for source in RELEASES for job in jobs[source]}
    if scanned & inputs["closed"]:
        raise PermissionError("A closed Challenge record was scanned")
    log(log_path, "assembling")
    frame, minimum = assemble(inputs, scan)
    check_reference(frame, reference, checks)
    check(checks, "patients_in_several_roles", {"ptbxl": 0, "sph": 0},
          mlc01_tables.patients_in_several_roles(frame))
    check(checks, "challenge_scanned_paths_in_closed_groups", 0, len(scanned & inputs["closed"]))
    checks["codes_outside_official_tables"] = {"observed": code_coverage(frame, inputs)}
    sph_rows = frame[frame["source"] == "sph"]
    matched = int((sph_rows["metadata_samples"] == sph_rows["n_samples"]).sum())
    checks["sph_metadata_samples_match_file"] = {"observed": matched, "records": len(sph_rows)}
    summary = write_outputs(inputs, frame, scan, stage)
    status_table(minimum).to_csv(stage / "annotation_status.csv", index=False)
    minimum.to_csv(stage / "minimum_positive_rule.csv", index=False)
    usage = {who: resource.getrusage(kind).ru_maxrss for who, kind in
             (("parent", resource.RUSAGE_SELF), ("children", resource.RUSAGE_CHILDREN))}
    run = {"command": command, "workers": workers, "limit_per_source": limit,
           "omp_num_threads": os.environ.get("OMP_NUM_THREADS"),
           "seconds": round(time.monotonic() - started, 1), "max_rss_kib": usage, "scan_cache": str(cache)}
    log(log_path, "writing receipt")
    result = receipt(inputs, frame, checks, summary, run)
    result["amendment_1"] = amendment_record(minimum)
    result["outputs_sha256"] = {path.name: sha256_file(path) for path in sorted(stage.iterdir())
                                if path.name != "build.log"}
    write_json_atomic(stage / "receipt.json", result, sort_keys=True)
    log(log_path, f"done in {run['seconds']} s")
    stage.rename(output)
    return result
