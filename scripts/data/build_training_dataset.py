#!/usr/bin/env python3
"""Publish a new canonical 500 Hz train union without altering frozen cohorts."""

from __future__ import annotations

import argparse
import csv
import fcntl
import json
import os
import sqlite3
import sys
import time
from collections import Counter
from collections.abc import Iterable, Mapping
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
import wfdb

from ecg_experiment.downloads import parse_checksums
from ecg_experiment.evaluation import partition_validation
from ecg_experiment.files import read_csv, sha256_file
from ecg_experiment.pool_overlap import (
    MIMIC_POOL,
    PinInput,
    load_candidates,
    load_mimic_reference,
    load_ptb_reference,
    verify_candidate_views,
)
from ecg_experiment.provenance import git_head
from ecg_experiment.public_sources import signal_sha256
from ecg_experiment.staging import published_directory
from ecg_experiment.training_dataset import verify_dataset
from ecg_experiment.waveforms import LEADS, read_record

ROOT = Path(__file__).resolve().parents[2]
PTB_PROCESSED = ROOT / "data/processed/ptbxl/seed42_fraction1"
PTB_RAW = ROOT / "data/raw/ptb-xl/1.0.3"
CPC_POOL = ROOT / "data/processed/cpc_pool_40k"
CANDIDATE_DIR = ROOT / "outputs/data_quality/processed_eda"
OVERLAP_AUDIT = ROOT / "outputs/data_quality/astra_review/pool_overlap"
GEORGIA_QUARANTINE = ROOT / "outputs/data_quality/astra_review/frozen_georgia_constant_leads.json"
HISTORICAL_COUNTS = {"ptbxl": 17418, "mimic": 39457}
NOVEL_CHALLENGE_RECORDS = 18844
GEORGIA_OVERLAP_RECORDS = 945
SIGNAL_SHAPE = (12, 5000)
DECODE_BATCH = 128
PROGRESS_INTERVAL = 2048
MAX_WORKERS = 8
FIELDS = ("record_id", "source_record_id", "source", "patient_id", "patient_identity_known",
          "split", "label_scope", "shard", "shard_index", "signal_sha256",
          "origin_manifest", "origin_path", "origin_index", "view", "window_start",
          "source_samples", "qc_flags")
EXCLUSION_FIELDS = ("record_id", "source", "reason", "detail", "signal_sha256")
LABEL_FIELDS = ("record_id", "patient_id", "target")
REFERENCE_FIELDS = ("record_id", "patient_id", "split", "target", "raw_path")
SOURCE_FILES = ("ecg_experiment/training_dataset.py", "ecg_experiment/run.py",
                "scripts/extract_pretrained.py", "scripts/validation/audit_public_pool_overlap.py",
                "ecg_experiment/pool_overlap.py", "ecg_experiment/staging.py",
                "ecg_experiment/evaluation.py", "ecg_experiment/waveforms.py",
                "ecg_experiment/downloads.py", "ecg_experiment/public_sources.py")


def write_csv(path: Path, fields: Iterable[str], rows: Iterable[Mapping[str, Any]]) -> None:
    """
    Write the named columns of each row, ignoring any other keys.

    Parameters
    ----------
    path : Path
        Destination CSV file.
    fields : Iterable[str]
        Column order.
    rows : Iterable[Mapping[str, Any]]
        Rows that may carry extra decoding keys.
    """
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(fields), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def historical_rows(pin: PinInput, ptb_checksums: dict[str, str]) -> list[dict[str, Any]]:
    """
    List the frozen PTB-XL and MIMIC training records with their decoding sources.

    Parameters
    ----------
    pin : PinInput
        Records each input file read.
    ptb_checksums : dict[str, str]
        Official PTB-XL SHA256SUMS entries.

    Returns
    -------
    list[dict[str, Any]]
        Candidate rows in manifest order, PTB-XL first.

    Raises
    ------
    ValueError
        If a frozen manifest changed or the historical pool counts differ.
    """
    with sqlite3.connect((MIMIC_POOL / "audit.sqlite3").as_uri() + "?mode=ro&immutable=1", uri=True) as db:
        mimic_expected = dict(db.execute("SELECT name,signal_sha256 FROM outcomes WHERE status='accepted'"))
    pool = json.loads(pin(CPC_POOL / "complete.json").read_text())
    for name, digest in pool["ptb_manifest_sha256"].items():
        if sha256_file(pin(PTB_PROCESSED / name)) != digest:
            raise ValueError("Frozen PTB manifest mismatch")
    mimic_raw = Path(json.loads((MIMIC_POOL / "metadata.json").read_text())["raw_dir"])
    rows = []
    for source, manifest, raw_dir in (("ptbxl", PTB_PROCESSED / "all_train_ssl.csv", PTB_RAW),
                                      ("mimic", MIMIC_POOL / "ssl_manifest.csv", mimic_raw)):
        is_ptb = source == "ptbxl"
        for original in read_csv(pin(manifest)):
            source_id = original["ecg_id"]
            filename = original["filename_hr"]
            rows.append({"record_id": f"ptbxl:{source_id}" if is_ptb else source_id,
                         "source_record_id": source_id, "source": source,
                         "patient_id": (f"ptbxl:{original['patient_id']}" if is_ptb
                                        else original["patient_id"]),
                         "patient_identity_known": "true", "split": "train",
                         "label_scope": "ptbxl_proxy_available_separately" if is_ptb else "ssl_only",
                         "origin_manifest": str(manifest.relative_to(ROOT)),
                         "origin_path": str((raw_dir / filename).relative_to(ROOT)),
                         "origin_index": "", "view": "original_10s", "window_start": 0,
                         "source_samples": SIGNAL_SHAPE[1], "qc_flags": "", "raw_dir": raw_dir,
                         "filename_hr": filename,
                         "official_sha256": ({suffix: ptb_checksums[filename + suffix]
                                              for suffix in (".hea", ".dat")} if is_ptb else {}),
                         "expected_hash": mimic_expected[filename] if source == "mimic" else ""})
    if Counter(r["source"] for r in rows) != HISTORICAL_COUNTS:
        raise ValueError("Historical training pool changed")
    return rows


def curated_challenge_rows(pin: PinInput) -> list[dict[str, str]]:
    """
    Load the curated Challenge candidates behind the verified overlap gate.

    Parameters
    ----------
    pin : PinInput
        Records each input file read.

    Returns
    -------
    list[dict[str, str]]
        Curated candidate rows.

    Raises
    ------
    ValueError
        If the overlap audit changed, its gate differs, or a rail-affected
        record entered the candidate.
    """
    curated, receipt = load_candidates(CANDIDATE_DIR, pin)
    verify_candidate_views(curated, receipt, pin)
    audit = json.loads(pin(OVERLAP_AUDIT / "receipt.json").read_text())
    for name, digest in audit["input_sha256"].items():
        if sha256_file(pin(Path(name))) != digest:
            raise ValueError("Overlap audit input changed")
    for name, digest in audit["output_sha256"].items():
        if sha256_file(pin(OVERLAP_AUDIT / name)) != digest:
            raise ValueError("Overlap audit output changed")
    novel = read_csv(OVERLAP_AUDIT / "novel_challenge_ssl_manifest.csv")
    overlaps = read_csv(OVERLAP_AUDIT / "overlaps.csv")
    if (len(novel) != NOVEL_CHALLENGE_RECORDS or len(overlaps) != GEORGIA_OVERLAP_RECORDS or
            {r["ecg_id"] for r in curated} != {r["ecg_id"] for r in novel + overlaps} or
            any(r["exact_signal_reference_pools"] != "georgia_frozen_g1" for r in overlaps)):
        raise ValueError("Unexpected append-overlap gate")
    overlay = read_csv(pin(CANDIDATE_DIR / "training_exclusion_overlay.csv"))
    if {r["ecg_id"] for r in overlay} & {r["ecg_id"] for r in curated}:
        raise ValueError("Rail-affected candidate entered train")
    return curated


def challenge_row(original: dict[str, str]) -> dict[str, Any]:
    """
    Describe one curated Challenge candidate as an SSL-only training row.

    Parameters
    ----------
    original : dict[str, str]
        Curated candidate row pointing at a published view shard.

    Returns
    -------
    dict[str, Any]
        Candidate row with its expected signal hash.
    """
    return {"record_id": original["ecg_id"], "source_record_id": original["ecg_id"],
            "source": original["source"], "patient_id": "", "patient_identity_known": "false",
            "split": "train", "label_scope": "ssl_only",
            "origin_manifest": "outputs/data_quality/processed_eda/challenge_ssl_curated_manifest.csv",
            "origin_path": original["shard_path"], "origin_index": original["shard_index"],
            "view": original["view"], "window_start": original["window_start"],
            "source_samples": original["source_samples"], "qc_flags": original["qc_flags"],
            "expected_hash": original["signal_sha256"]}


def label_budgets(pin: PinInput) -> dict[str, list[dict[str, str]]]:
    """
    Read both frozen PTB-XL label budgets in namespaced form.

    Parameters
    ----------
    pin : PinInput
        Records each input file read.

    Returns
    -------
    dict[str, list[dict[str, str]]]
        Label rows keyed by budget ``"1"`` and ``"0.1"``.
    """
    labels = {}
    for budget in ("1", "0.1"):
        path = ROOT / f"data/processed/ptbxl/seed42_fraction{budget}/labeled_train.csv"
        labels[budget] = [{"record_id": "ptbxl:" + r["ecg_id"], "target": r["target"],
                           "patient_id": "ptbxl:" + r["patient_id"]} for r in read_csv(pin(path))]
    return labels


def heldout_references(rows: list[dict[str, Any]]) -> list[dict[str, str]]:
    """
    List the held-out PTB-XL references and check patient separation from train.

    Parameters
    ----------
    rows : list[dict[str, Any]]
        Training candidate rows.

    Returns
    -------
    list[dict[str, str]]
        Development, calibration and test references.

    Raises
    ------
    ValueError
        If a patient appears in more than one split.
    """
    development, calibration = partition_validation(read_csv(PTB_PROCESSED / "validation.csv"))
    seen_patients = {r["patient_id"] for r in rows if r["source"] == "ptbxl"}
    references = []
    for split, original_rows in (("development", development), ("calibration", calibration),
                                 ("test", read_csv(PTB_PROCESSED / "test.csv"))):
        patients = {"ptbxl:" + r["patient_id"] for r in original_rows}
        if seen_patients & patients:
            raise ValueError("PTB patient split leakage")
        seen_patients |= patients
        references.extend({"record_id": "ptbxl:" + r["ecg_id"], "patient_id": "ptbxl:" + r["patient_id"],
                           "split": split, "target": r["target"],
                           "raw_path": str((PTB_RAW / r["filename_hr"]).relative_to(ROOT))}
                          for r in original_rows)
    return references


def prepare_inputs(pin: PinInput) -> tuple[list[dict[str, Any]], dict[str, list[dict[str, str]]],
                                          list[dict[str, str]], set[str]]:
    """
    Verify every input and list the train candidates, labels and references.

    Parameters
    ----------
    pin : PinInput
        Records each input file read, in a fixed order.

    Returns
    -------
    tuple
        Candidate rows, label budgets, held-out references, and PTB-XL
        reference signal hashes.

    Raises
    ------
    ValueError
        If any input check fails or a Challenge record duplicates PTB/MIMIC.
    """
    ptb_checksums = parse_checksums(pin(PTB_RAW / "SHA256SUMS.txt").read_text())
    ptb_reference = load_ptb_reference(pin)
    mimic_hashes, _ = load_mimic_reference(pin)
    rows = historical_rows(pin, ptb_checksums)
    frozen_hashes = ptb_reference | mimic_hashes
    for original in curated_challenge_rows(pin):
        if original["signal_sha256"] in frozen_hashes:
            raise ValueError("Challenge duplicates frozen PTB/MIMIC")
        rows.append(challenge_row(original))
    labels = label_budgets(pin)
    return rows, labels, heldout_references(rows), ptb_reference


def _changed_ptb_file(row: dict[str, Any]) -> str | None:
    """Return the first suffix whose PTB-XL raw file differs from its official hash."""
    changed = None
    for suffix, expected in row["official_sha256"].items():
        if sha256_file(ROOT / (row["origin_path"] + suffix)) != expected:
            changed = suffix
            break
    return changed


def _decode_ptb(row: dict[str, Any]) -> np.ndarray:
    """Decode a PTB-XL record, verifying official bytes before and after reading."""
    changed = _changed_ptb_file(row)
    if changed is not None:
        raise ValueError(f"Official PTB checksum mismatch: {row['record_id']}{changed}")
    header = wfdb.rdheader(str(ROOT / row["origin_path"]))
    if header.units != ["mV"] * SIGNAL_SHAPE[0]:
        raise ValueError(f"Unverified amplitude units: {row['record_id']}")
    signal = read_record(row["raw_dir"], row["filename_hr"])
    if _changed_ptb_file(row) is not None:
        raise ValueError("PTB raw file changed while decoding")
    return signal


def decode(row: dict[str, Any]) -> tuple[np.ndarray, str, list[str]]:
    """
    Decode one candidate from its published shard or official raw record.

    Parameters
    ----------
    row : dict[str, Any]
        Candidate row from :func:`prepare_inputs`.

    Returns
    -------
    tuple[np.ndarray, str, list[str]]
        Float32 ``[12, 5000]`` mV signal, its hash, and constant lead names.

    Raises
    ------
    ValueError
        If raw bytes, units, identity or the waveform contract fail.
    """
    if row["origin_index"] != "":
        shard = np.load(ROOT / row["origin_path"], mmap_mode="r", allow_pickle=False)
        signal = np.array(shard[int(row["origin_index"])], copy=True)
    else:
        signal = _decode_ptb(row)
    digest = signal_sha256(signal)
    if row["expected_hash"] and digest != row["expected_hash"]:
        raise ValueError(f"Source waveform identity changed: {row['record_id']}")
    if signal.shape != SIGNAL_SHAPE or signal.dtype != np.float32 or not np.isfinite(signal).all():
        raise ValueError("Source waveform contract failed")
    constant = [LEADS[i] for i in np.flatnonzero(np.ptp(signal, axis=1) == 0)]
    return signal, digest, constant


def _write_shard(stage: Path, number: int, arrays: list[np.ndarray],
                 rows: list[dict[str, Any]]) -> tuple[str, dict[str, Any]]:
    """Save one shard, number its rows in place, and return its name and receipt."""
    name = f"shard_{number:05d}.npy"
    matrix = np.stack(arrays)
    np.save(stage / name, matrix, allow_pickle=False)
    for index, row in enumerate(rows):
        row.update(shard=name, shard_index=index)
    return name, {"sha256": sha256_file(stage / name), "shape": list(matrix.shape)}


def write_shards(stage: Path, rows: list[dict[str, Any]], ptb_reference: set[str], workers: int,
                 shard_size: int, started: float) -> tuple[list[dict[str, Any]], list[dict[str, Any]],
                                                           dict[str, dict[str, Any]]]:
    """
    Decode every candidate and write accepted signals to fixed-size shards.

    Parameters
    ----------
    stage : Path
        Staging directory.
    rows : list[dict[str, Any]]
        Candidate rows in output order.
    ptb_reference : set[str]
        Checksum-verified PTB-XL signal hashes.
    workers : int
        Decoding threads.
    shard_size : int
        Records per shard.
    started : float
        ``time.monotonic()`` at build start, for progress lines.

    Returns
    -------
    tuple
        Accepted rows, constant-lead exclusions, and shard receipts by name.

    Raises
    ------
    ValueError
        If a PTB-XL signal is unverified or two candidates are identical.
    """
    accepted, exclusions, arrays, shard_rows, shards, seen = [], [], [], [], {}, set()
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for start in range(0, len(rows), DECODE_BATCH):
            batch = rows[start:start + DECODE_BATCH]
            for row, (signal, digest, constants) in zip(batch, pool.map(decode, batch), strict=True):
                if row["source"] == "ptbxl" and digest not in ptb_reference:
                    raise ValueError("PTB waveform not in checksum-verified reference")
                if digest in seen:
                    raise ValueError("Unexpected exact duplicate within new train union")
                seen.add(digest)
                if constants:
                    exclusions.append({"record_id": row["record_id"], "source": row["source"],
                                       "reason": "full_constant_lead", "detail": ";".join(constants),
                                       "signal_sha256": digest})
                    continue
                row["signal_sha256"] = digest
                arrays.append(signal)
                shard_rows.append(row)
                if len(arrays) == shard_size:
                    name, shards[name] = _write_shard(stage, len(shards), arrays, shard_rows)
                    accepted.extend(shard_rows)
                    arrays, shard_rows = [], []
            if start % PROGRESS_INTERVAL == 0:
                print(f"Read {min(start + DECODE_BATCH, len(rows)):,}/{len(rows):,}; "
                      f"excluded {len(exclusions)}; {time.monotonic() - started:.1f}s", flush=True)
    if arrays:
        name, shards[name] = _write_shard(stage, len(shards), arrays, shard_rows)
        accepted.extend(shard_rows)
    return accepted, exclusions, shards


def write_label_tables(stage: Path, accepted: list[dict[str, Any]],
                       labels: dict[str, list[dict[str, str]]]) -> dict[str, dict[str, int]]:
    """
    Write each label budget restricted to retained records.

    Parameters
    ----------
    stage : Path
        Staging directory.
    accepted : list[dict[str, Any]]
        Accepted training rows.
    labels : dict[str, list[dict[str, str]]]
        Label rows keyed by budget.

    Returns
    -------
    dict[str, dict[str, int]]
        Input and retained label counts per budget.

    Raises
    ------
    ValueError
        If a label's patient differs from its retained record's patient.
    """
    kept = {r["record_id"]: r for r in accepted}
    label_counts = {}
    for budget, original in labels.items():
        selected = [r for r in original if r["record_id"] in kept]
        if any(kept[r["record_id"]]["patient_id"] != r["patient_id"] for r in selected):
            raise ValueError("PTB label patient mismatch")
        write_csv(stage / f"labels_fraction{budget}.csv", LABEL_FIELDS, selected)
        label_counts[budget] = {"input": len(original), "retained": len(selected)}
    return label_counts


def build_metadata(output: Path, stage: Path, rows: list[dict[str, Any]], accepted: list[dict[str, Any]],
                   exclusions: list[dict[str, Any]], references: list[dict[str, str]],
                   label_counts: dict[str, dict[str, int]], inputs: dict[str, str],
                   shards: dict[str, dict[str, Any]], started: float) -> dict[str, Any]:
    """
    Describe the staged dataset, its policies, inputs and code provenance.

    Parameters
    ----------
    output : Path
        Final dataset directory; its name is the dataset ID.
    stage : Path
        Staging directory whose CSV and JSON tables are hashed.
    rows, accepted, exclusions : list[dict[str, Any]]
        Candidates, accepted rows and new exclusions.
    references : list[dict[str, str]]
        Held-out references.
    label_counts : dict[str, dict[str, int]]
        Output of :func:`write_label_tables`.
    inputs : dict[str, str]
        SHA-256 of every pinned input keyed by resolved path.
    shards : dict[str, dict[str, Any]]
        Shard receipts by name.
    started : float
        ``time.monotonic()`` at build start.

    Returns
    -------
    dict[str, Any]
        Metadata written as ``metadata.json``.

    Raises
    ------
    RuntimeError
        If the Git revision cannot be read.
    """
    revision = git_head(ROOT)
    if revision is None:
        raise RuntimeError("Cannot record the Git revision of this build")
    return {
        "schema_version": 1, "complete": True, "dataset_id": output.name,
        "record_count": len(accepted), "candidate_count": len(rows),
        "source_counts": dict(Counter(r["source"] for r in accepted)),
        "new_exclusions_by_source": dict(Counter(r["source"] for r in exclusions)),
        "shape_per_record": list(SIGNAL_SHAPE), "dtype": "float32", "units": "mV",
        "sampling_rate_hz": 500, "lead_order": list(LEADS),
        "preprocessing": ("Original physical mV; no filtering, clipping, resampling, normalization "
                          "or fitted parameters"),
        "label_counts": label_counts,
        "heldout_reference_counts": dict(Counter(r["split"] for r in references)),
        "split_policy": ("PTB official frozen train only; existing seed9001 patient development/calibration "
                         "partition retained as references; no held-out arrays in train shards"),
        "label_policy": ("PTB diagnostic annotation proxy only, opt-in exact retained frozen label budgets; "
                         "all other sources SSL-only"),
        "patient_identity": ("PTB/MIMIC namespaced official patient IDs; Challenge unknown, patient_id "
                             "deliberately empty; source independence is not proven"),
        "composition": ("Historical PTB+MIMIC 56875 candidates plus curated Challenge 19789; Challenge "
                        "includes 945 clean Georgia pilot overlaps and 18844 novel records; 3 historical "
                        "Georgia constants never candidates"),
        "limitations": ["No clinical quality or performance validation",
                        ("Exact hashes do not detect transformed near-duplicates or prove cross-source "
                         "patient independence"),
                        "CODE native and incomplete Chapman/MIMIC200k are excluded",
                        ("New data-scaling cohort; not an equivalent replacement for frozen "
                         "experiment comparisons")],
        "input_sha256": inputs, "shards": shards,
        "table_sha256": {p.name: sha256_file(p) for p in stage.iterdir() if p.suffix in (".csv", ".json")},
        "git_revision": revision,
        "source_sha256": {str(p.relative_to(ROOT)): sha256_file(p) for p in
                          (Path(__file__), *(ROOT / name for name in SOURCE_FILES))},
        "command": [sys.executable, *sys.argv], "build_seconds": time.monotonic() - started,
    }


def _check_output(output: Path) -> None:
    """Refuse an existing output or one inside raw data."""
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite {output}")
    if output.is_relative_to((ROOT / "data/raw").resolve()):
        raise ValueError("Output must be outside raw data")


def _build_locked(output: Path, workers: int = 4, shard_size: int = 128) -> dict[str, Any]:
    """Build and publish the dataset while holding the builder lock."""
    output = output.resolve()
    _check_output(output)
    if not 1 <= workers <= MAX_WORKERS or shard_size < 1:
        raise ValueError("Invalid workers or shard size")
    inputs: dict[str, str] = {}

    def pin(path: Path) -> Path:
        inputs[str(path.resolve())] = sha256_file(path)
        return path

    rows, labels, references, ptb_reference = prepare_inputs(pin)
    print(f"Inputs verified: {len(rows):,} train candidates; {len(references):,} held-out references",
          flush=True)
    started = time.monotonic()
    with published_directory(output) as stage:
        accepted, exclusions, shards = write_shards(stage, rows, ptb_reference, workers, shard_size, started)
        write_csv(stage / "train_manifest.csv", FIELDS, accepted)
        write_csv(stage / "exclusions.csv", EXCLUSION_FIELDS, exclusions)
        # These historical records never enter the candidates, and are tracked separately.
        quarantine = json.loads(pin(GEORGIA_QUARANTINE).read_text())
        (stage / "historical_georgia_quarantine.json").write_text(json.dumps(quarantine, indent=2) + "\n")
        label_counts = write_label_tables(stage, accepted, labels)
        write_csv(stage / "heldout_references.csv", REFERENCE_FIELDS, references)
        for path, expected in inputs.items():
            if sha256_file(path) != expected:
                raise ValueError(f"Input changed while building: {path}")
        metadata = build_metadata(output, stage, rows, accepted, exclusions, references,
                                  label_counts, inputs, shards, started)
        (stage / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
        print("Independently verifying every new shard and both label budgets", flush=True)
        verification = verify_dataset(stage)
        verification.update(metadata_sha256=sha256_file(stage / "metadata.json"),
                            elapsed_seconds=time.monotonic() - started)
        (stage / "verification.json").write_text(json.dumps(verification, indent=2) + "\n")
    return verification


def build(output: Path, workers: int = 4, shard_size: int = 128) -> dict[str, Any]:
    """
    Build the canonical train union under an exclusive per-output lock.

    Parameters
    ----------
    output : Path
        New dataset directory, outside ``data/raw``.
    workers : int
        Decoding threads, 1 through 8.
    shard_size : int
        Records per shard.

    Returns
    -------
    dict[str, Any]
        Independent verification of the published dataset.

    Raises
    ------
    FileExistsError
        If the output exists.
    ValueError
        If arguments or inputs are invalid.
    RuntimeError
        If another builder holds the lock.
    """
    output = output.resolve()
    _check_output(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.with_name("." + output.name + ".lock").open("a+") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError("A builder already holds this dataset lock") from None
        lock.seek(0)
        lock.truncate()
        lock.write(str(os.getpid()) + "\n")
        lock.flush()
        return _build_locked(output, workers, shard_size)


def main() -> None:
    """Build or verify a dataset and print the verification result."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    result = verify_dataset(args.output_dir) if args.verify_only else build(args.output_dir, args.workers)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
