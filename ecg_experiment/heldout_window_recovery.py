"""Recover fixed canonical hash references mechanically; never compute model or quality scores."""

from __future__ import annotations

import json
import resource
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import pandas as pd

from . import ROOT
from .challenge_training_recovery import (
    RECOVERY,
    RELEASES,
    SPLIT_SHA256,
    original_split,
    source_checkpoint,
    stored,
)
from .challenge_training_recovery import (
    REFERENCE_ROOT as REFERENCE_ROOT,
)
from .downloads import parse_checksums
from .files import sha256_file, write_json_atomic
from .public_sources import signal_sha256
from .recovery_views import read_verified_view, verify_pair
from .waveforms import read_record

PTB_RAW = Path("data/raw/ptb-xl/1.0.3")
PTB_QUALITY = RECOVERY / "ptb/quality"
SPH_REFERENCE = Path("outputs/screen01_matched_transfer_20261027/qualify/sph_rows.csv")
SPH_RECEIPT = SPH_REFERENCE.with_name("completion.json")
SOURCE_FILES = (
    "ecg_experiment/heldout_window_recovery.py", "ecg_experiment/recovery_views.py",
    "ecg_experiment/challenge_training_recovery.py", "ecg_experiment/public_sources.py",
    "ecg_experiment/waveforms.py", "ecg_experiment/paths.py", "ecg_experiment/downloads.py",
    "scripts/data/recover_heldout_window_references.py",
)


def challenge_references(input_root: Path) -> tuple[list[dict], list[tuple], dict[str, str]]:
    """List every original held-out record using label-free split and header columns."""
    split, metadata, inputs = original_split(input_root)
    held = split[split["split"] != "train"]
    recovered, jobs = [], []
    for source, release in RELEASES.items():
        headers, header_inputs = source_checkpoint(
            input_root, source, metadata, ["source", "path", "signal_sha256", "window_sha256", "samples"],
        )
        inputs.update(header_inputs)
        selected = held[held["source"] == source].set_index("record")
        evidence = headers.reindex(selected.index)
        identity = ("source", "path", "signal_sha256", "window_sha256")
        if not all((evidence[name].fillna("") == selected[name].fillna("")).all() for name in identity):
            raise ValueError(f"Held-out header/split identity mismatch: {source}")
        checksum_path = Path("data/raw") / release / "SHA256SUMS.txt"
        checksums = parse_checksums((input_root / checksum_path).read_text())
        inputs[stored(checksum_path)] = sha256_file(input_root / checksum_path)
        for record, row in selected.iterrows():
            samples = int(evidence.loc[record, "samples"])
            item = {
                "record_id": source + ":" + record, "source": source, "role": row["split"],
                "patient_id": "", "patient_identity_known": False,
                "path": stored(Path("data/raw") / release / row["path"]),
                "full_signal_sha256": row["signal_sha256"], "canonical_sha256": "",
                "view_policy": "strict_10s" if samples == 5000 else "ssl_center_crop",
                "source_samples": samples, "window_start": max(0, (samples - 5000) // 2), "skip_reason": "",
            }
            if source == "ningbo":
                if not row["window_sha256"]:
                    raise ValueError("Missing frozen Ningbo window reference")
                item.update(canonical_sha256=row["window_sha256"], view_policy="frozen_ningbo_whole_window")
                recovered.append(item)
            else:
                expected = {suffix: checksums[row["path"] + suffix] for suffix in (".hea", ".mat")}
                jobs.append((str(input_root), release, row["path"], item, expected))
    return recovered, jobs, inputs


def hash_challenge_reference(item: tuple[str, str, str, dict, dict[str, str]]) -> dict:
    """Hash one fixed held-out view with official file checksums and explicit contract skips."""
    input_root, release, stem, row, expected = item
    if row["source_samples"] < 5000:
        row["skip_reason"] = "duration_contract"
    else:
        signal, start, samples, _, reasons = read_verified_view(
            Path(input_root) / "data/raw" / release, stem, row["view_policy"], expected,
        )
        row["skip_reason"] = ";".join(reasons)
        if signal is not None:
            row.update(canonical_sha256=signal_sha256(signal), window_start=start, source_samples=samples)
    row["worker_peak_rss_kib"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return row


def ptb_reference_jobs(input_root: Path) -> tuple[list[dict], list[tuple], dict[str, str]]:
    """Import completed fold-8 hashes and list folds 9/10 without reading any PTB labels."""
    receipt_path = PTB_QUALITY / "receipt.json"
    receipt = json.loads((input_root / receipt_path).read_text())
    quality_path = PTB_QUALITY / "record_quality.csv"
    if (receipt.get("scope") != "training_quality" or not receipt.get("complete") or
            sha256_file(input_root / quality_path) != receipt["quality_sha256"]):
        raise ValueError("Verified PTB training hash recovery is required")
    quality = pd.read_csv(input_root / quality_path, dtype=str, keep_default_na=False,
                          usecols=["record_id", "patient_id", "path", "strat_fold", "signal_sha256"])
    recovered = [{
        "record_id": row.record_id, "source": "ptbxl", "role": "fold8_calibration", "path": row.path,
        "patient_id": row.patient_id, "patient_identity_known": True,
        "canonical_sha256": row.signal_sha256, "full_signal_sha256": "", "view_policy": "strict_10s",
        "source_samples": 5000, "window_start": 0, "skip_reason": "",
    } for row in quality[quality["strat_fold"] == "8"].itertuples(index=False)]
    metadata_path = PTB_RAW / "ptbxl_database.csv"
    metadata = pd.read_csv(input_root / metadata_path, dtype=str, keep_default_na=False,
                           usecols=["ecg_id", "patient_id", "filename_hr", "strat_fold"])
    checksum_path = PTB_RAW / "SHA256SUMS.txt"
    for path in (metadata_path, checksum_path):
        if sha256_file(input_root / path) != receipt["input_sha256"][stored(path)]:
            raise ValueError("PTB reference source differs from the completed training recovery")
    expected_fold8 = set("ptbxl:" + metadata.loc[metadata["strat_fold"] == "8", "ecg_id"])
    if {row["record_id"] for row in recovered} != expected_fold8:
        raise ValueError("PTB fold-8 reference coverage differs from the official metadata")
    checksums = parse_checksums((input_root / checksum_path).read_text())
    jobs = []
    for row in metadata[metadata["strat_fold"].isin(["9", "10"])].itertuples(index=False):
        expected = {suffix: checksums[row.filename_hr + suffix] for suffix in (".hea", ".dat")}
        jobs.append((str(input_root), row.ecg_id, row.patient_id, row.filename_hr, row.strat_fold, expected))
    inputs = {stored(path): sha256_file(input_root / path)
              for path in (receipt_path, quality_path, metadata_path, checksum_path)}
    return recovered, jobs, inputs


def hash_ptb_reference(item: tuple[str, str, str, str, str, dict[str, str]]) -> dict:
    """Hash one official PTB held-out recording without computing its labels or quality score."""
    input_root, ecg_id, patient_id, stem, fold, expected = item
    raw = Path(input_root) / PTB_RAW
    verify_pair(raw, stem, expected, "before decoding")
    signal = read_record(raw, stem)
    verify_pair(raw, stem, expected, "after decoding")
    return {
        "record_id": "ptbxl:" + ecg_id, "source": "ptbxl", "role": "fold" + fold,
        "patient_id": "ptbxl:" + patient_id, "patient_identity_known": True,
        "path": stored(PTB_RAW / stem), "canonical_sha256": signal_sha256(signal),
        "full_signal_sha256": "", "view_policy": "strict_10s", "source_samples": 5000,
        "window_start": 0, "skip_reason": "",
        "worker_peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
    }


def sph_references(input_root: Path) -> tuple[list[dict], dict[str, str]]:
    """Import the existing SPH qualification hashes without reading outcomes or waveforms."""
    receipt = json.loads((input_root / SPH_RECEIPT).read_text())
    digest = sha256_file(input_root / SPH_REFERENCE)
    if receipt["status"] != "completed" or digest != receipt["artifacts"][stored(SPH_REFERENCE)]:
        raise ValueError("SPH reference differs from its completed qualification receipt")
    table = pd.read_csv(input_root / SPH_REFERENCE, dtype=str, keep_default_na=False,
                        usecols=["ecg_id", "patient_id", "role", "signal_sha256"])
    if len(table) != 17402 or set(table["role"]) != {"calibration", "evaluation"}:
        raise ValueError("Unexpected existing SPH reference scope")
    if (table["patient_id"] == "").any():
        raise ValueError("Missing SPH patient identity in the existing reference table")
    if not table["signal_sha256"].str.fullmatch("[0-9a-f]{64}").all():
        raise ValueError("Invalid SPH canonical reference hash")
    rows = [{
        "record_id": "sph:" + row.ecg_id.removeprefix("sph:"), "source": "sph", "role": row.role,
        "patient_id": "sph:" + row.patient_id.removeprefix("sph:"), "patient_identity_known": True,
        "path": "", "canonical_sha256": row.signal_sha256, "full_signal_sha256": "",
        "view_policy": "existing_screen01_canonical_hash", "source_samples": 5000, "window_start": 0,
        "skip_reason": "",
    } for row in table.itertuples(index=False)]
    return rows, {stored(SPH_REFERENCE): digest, stored(SPH_RECEIPT): sha256_file(input_root / SPH_RECEIPT)}


def validate_reference_table(table: pd.DataFrame, intended_counts: dict, complete: bool) -> dict:
    """Require explicit coverage, unique record identities, valid hashes and honest contract skips."""
    if table["record_id"].duplicated().any():
        raise ValueError("Repeated reference record identity")
    actual_counts = table["source"].value_counts().sort_index().to_dict()
    if complete and actual_counts != intended_counts:
        raise ValueError("Incomplete intended reference coverage")
    if not table["canonical_sha256"].str.fullmatch("[0-9a-f]{64}|").all():
        raise ValueError("Malformed canonical reference hashes")
    if ((table["canonical_sha256"] == "") & (table["skip_reason"] == "")).any():
        raise ValueError("Missing reference hash without explicit skip reason")
    return actual_counts


def recover(input_root: Path, output: Path, workers: int, limit: int | None) -> dict[str, object]:
    """
    Write fixed-window held-out references as mechanical data-cleaning provenance.

    Parameters
    ----------
    input_root : Path
        Checkout holding local raw data and completed reconstruction artifacts.
    output : Path
        New root-relative output directory.
    workers : int
        One to four CPU processes with external thread limits.
    limit : int or None
        Per-source Challenge and PTB job limit for a resource profile, or all records.

    Returns
    -------
    dict[str, object]
        Reference coverage, checksum provenance, explicit skips and resource receipt; no performance scores.
    """
    if not 1 <= workers <= 4 or (limit is not None and limit < 1):
        raise ValueError("Use one to four workers and a positive profile limit")
    if output.is_absolute() or ".." in output.parts:
        raise ValueError("Output must be root relative")
    destination = input_root / output
    if destination.exists():
        raise FileExistsError(destination)
    start = time.monotonic()
    challenge, challenge_jobs, inputs = challenge_references(input_root)
    ptb, ptb_jobs, ptb_inputs = ptb_reference_jobs(input_root)
    sph, sph_inputs = sph_references(input_root)
    inputs.update(ptb_inputs | sph_inputs)
    intended = pd.Series([row["source"] for row in challenge + ptb + sph] +
                         [job[3]["source"] for job in challenge_jobs] + ["ptbxl"] * len(ptb_jobs))
    intended_counts = intended.value_counts().sort_index().to_dict()
    if limit is not None:
        groups = pd.Series([job[3]["source"] for job in challenge_jobs])
        indices = groups.groupby(groups, sort=False).head(limit).index
        challenge_jobs = [challenge_jobs[index] for index in indices]
        ptb_jobs = ptb_jobs[:limit]
    computed = []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for index, row in enumerate(pool.map(hash_challenge_reference, challenge_jobs, chunksize=16), 1):
            computed.append(row)
            if index % 2048 == 0:
                print(f"Recovered {index}/{len(challenge_jobs)} fixed Challenge held-out hashes", flush=True)
        computed.extend(pool.map(hash_ptb_reference, ptb_jobs, chunksize=16))
    table = pd.DataFrame(challenge + ptb + sph + computed)
    worker_peak = int(table.pop("worker_peak_rss_kib").max())
    table = table.sort_values(["source", "record_id"], kind="stable").reset_index(drop=True)
    actual_counts = validate_reference_table(table, intended_counts, limit is None)
    for name, digest in inputs.items():
        if sha256_file(input_root / name) != digest:
            raise ValueError(f"Reference input changed during processing: {name}")
    destination.mkdir(parents=True)
    table_path = output / "reference_rows.csv"
    table.to_csv(input_root / table_path, index=False)
    receipt = {
        "complete": True, "scope": "mechanical_reference_recovery" if limit is None else "resource_profile",
        "original_split_sha256": SPLIT_SHA256, "records": len(table),
        "coverage_complete": limit is None and actual_counts == intended_counts,
        "intended_source_counts": intended_counts, "actual_source_counts": actual_counts,
        "source_role_counts": table.groupby(["source", "role"]).size().unstack(fill_value=0).to_dict(),
        "canonical_hashes": int((table["canonical_sha256"] != "").sum()),
        "skip_reasons": table.loc[table["skip_reason"] != "", "skip_reason"].value_counts().to_dict(),
        "table_path": stored(table_path), "table_sha256": sha256_file(input_root / table_path),
        "input_sha256": inputs, "source_sha256": {name: sha256_file(ROOT / name) for name in SOURCE_FILES},
        "view_policy": "All original Challenge held-out records regardless labels/duplicates; frozen Ningbo "
                       "whole-window hashes reused; other sources strict10s or centered10s when longer; "
                       "PTB folds8/9/10 and existing SPH calibration/evaluation hashes",
        "workers": workers, "wall_seconds": time.monotonic() - start,
        "parent_peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "worker_peak_rss_kib": worker_peak,
        "child_user_seconds": resource.getrusage(resource.RUSAGE_CHILDREN).ru_utime,
        "child_system_seconds": resource.getrusage(resource.RUSAGE_CHILDREN).ru_stime,
        "models_or_quality_scores_computed": False,
        "limitation": "Exact fixed canonical-window reference, not an exhaustive shifted-window search or "
                      "patient identity verification; unusable views remain listed with explicit reasons",
    }
    write_json_atomic(destination / "receipt.json", receipt, sort_keys=True)
    return receipt
