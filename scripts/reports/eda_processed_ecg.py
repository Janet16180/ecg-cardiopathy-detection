#!/usr/bin/env python3
"""Reproducible EDA of published Challenge views and CODE-15% native part 0.

Reads only completed processed outputs. It does not change ECG arrays or labels.
Challenge amplitude statistics are those recorded for *all* accepted views by the
materializer; a deterministic waveform sample independently checks their hashes.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from datetime import UTC, datetime
from pathlib import Path
from types import ModuleType
from typing import Any

import h5py
import numpy as np
import wfdb

from ecg_experiment.files import read_csv, sha256_file, write_csv_atomic, write_json_atomic
from ecg_experiment.public_sources import signal_sha256
from ecg_experiment.waveforms import LEADS

ROOT = Path(__file__).resolve().parents[2]
STRICT = ROOT / "data/processed/challenge_ecg_views/strict_10s"
CROP = ROOT / "data/processed/challenge_ecg_views/cpsc_ssl_center_crop"
CODE = ROOT / "data/processed/code15_quality/part0_native"
OUTPUT = ROOT / "outputs/data_quality/processed_eda"
RAW_CHALLENGE = Path("data/raw/challenge-2020/1.0.2")

STRICT_VIEW = "strict_10s"
CROP_VIEW = "cpsc_ssl_center_crop"
CHALLENGE_SHAPE = (12, 5000)
CHALLENGE_SAMPLES = 5000
CHALLENGE_RATE_HZ = 500
CODE_SHAPE = (4096, 12)
CODE_HDF5 = "exams_part0_native.hdf5"
CODE_TABLES = ("manifest.csv", "exclusions.csv", "demographics.csv", "source_labels.csv")
CODE_DIAGNOSES = ("1dAVb", "RBBB", "LBBB", "SB", "ST", "AF", "normal_ecg")
MAX_PLAUSIBLE_AGE = 120
YOUNG_ADULT_AGES = (18, 30)
TOP_COUNTS = 12
# Exact physical limits of a signed 16-bit ADC at 1000 counts/mV.
RAIL_POSITIVE_MV = 32.767
RAIL_NEGATIVE_MV = -32.768
RAIL_TOLERANCE_MV = 1e-4
HIGH_AMPLITUDE_MV = 10

QUANTILE_KEYS = ("min", "p05", "p25", "median", "p75", "p95", "p99", "max")
QUANTILE_LEVELS = (0, .05, .25, .5, .75, .95, .99, 1)

REVIEW_FIELDS = ("view", "ecg_id", "source", "qc_flags", "peak_positive_mv",
                 "peak_negative_mv", "samples_abs_over_10mv",
                 "samples_at_positive_32_767mv", "samples_at_negative_32_768mv",
                 "minimum_lead_std_mv")
OVERLAY_FIELDS = ("view", "ecg_id", "source", "signal_sha256", "decision", "reason",
                  "positive_32767_samples", "negative_32768_samples", "wfdb_header_sha256")
CURATED_FIELDS = ("ecg_id", "source", "view", "shard_path", "shard_index", "signal_sha256",
                  "source_samples", "window_start", "window_samples", "sampling_rate_hz",
                  "units", "lead_order", "qc_flags", "patient_id", "patient_identity_known",
                  "patient_independent_eval_eligible", "label_scope", "endpoint_supervised_eligible")
OMISSION_FIELDS = ("view", "ecg_id", "signal_sha256", "reason")
LIMITATIONS = [
    "Challenge record IDs are not verified patient identities; patient-separated evaluation is unavailable.",
    "Challenge source diagnosis codes are unmapped to the PTB-XL endpoint.",
    "CODE native units and original duration remain unresolved; exact-zero edges are descriptive only.",
    "CODE part 0 is one of 18 archives, not the complete release.",
    "Waveform sample hashes validate sampled storage rows; Challenge full-manifest QC summaries "
    "originate from materialization.",
]

FIGURE_SOURCES = ("georgia", "cpsc_2018", "cpsc_2018_extra")
FIGURE_SOURCE_NAMES = ("Georgia", "CPSC 2018", "CPSC Extra")
REVIEW_EXAMPLES_PER_FLAG = 2
FIGURE_DPI = 160
BLUE = "#2664a5"
GOLD = "#d3a649"
RED = "#be6b62"
ORANGE = "#b86b32"
TEAL = "#487b74"
SAGE = "#97ada0"
TRACE_RED = "#a24b43"


def quantiles(values: list[float]) -> dict[str, float | None]:
    """
    Summarize values at fixed quantiles.

    Parameters
    ----------
    values : list[float]
        Values to summarize.

    Returns
    -------
    dict[str, float | None]
        Minimum, selected percentiles and maximum; all ``None`` when empty.
    """
    if not values:
        return dict.fromkeys(QUANTILE_KEYS)
    array = np.asarray(values, dtype=np.float64)
    return {key: float(value) for key, value in
            zip(QUANTILE_KEYS, np.quantile(array, QUANTILE_LEVELS), strict=True)}


def bool_count(data: list[dict[str, str]], key: str) -> int:
    """
    Count rows whose CSV field is the string ``"true"``.

    Parameters
    ----------
    data : list[dict[str, str]]
        Manifest rows.
    key : str
        Boolean column.

    Returns
    -------
    int
        Number of true rows.
    """
    return sum(row[key] == "true" for row in data)


def flag_counts(data: list[dict[str, str]]) -> dict[str, int]:
    """
    Count each semicolon-separated QC flag across rows.

    Parameters
    ----------
    data : list[dict[str, str]]
        Rows with a ``qc_flags`` column.

    Returns
    -------
    dict[str, int]
        Flag counts sorted by flag name.
    """
    counts = Counter(flag for row in data for flag in row["qc_flags"].split(";") if flag)
    return dict(sorted(counts.items()))


def _sorted_counts(values: list[str]) -> dict[str, int]:
    """Count values and sort the counts by value."""
    return dict(sorted(Counter(values).items()))


def deterministic_indices(count: int, sample_size: int, seed: int) -> np.ndarray:
    """
    Draw a sorted sample of row indices without replacement.

    Parameters
    ----------
    count : int
        Number of rows.
    sample_size : int
        Requested sample size; capped at ``count``.
    seed : int
        Generator seed.

    Returns
    -------
    np.ndarray
        Sorted row indices.
    """
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(count, size=min(count, sample_size), replace=False))


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _load_shard(directory: Path, name: str, cache: dict[str, np.ndarray]) -> np.ndarray:
    """Memory-map a published waveform shard once per name."""
    if name not in cache:
        cache[name] = np.load(directory / name, mmap_mode="r", allow_pickle=False)
    return cache[name]


def _shard_records(metadata: dict[str, Any]) -> dict[str, int]:
    """Map each shard file in a view receipt to its record count."""
    return {part["file"]: part["records"] for part in metadata["shards"]}


def validate_challenge_manifest(directory: Path, metadata: dict[str, Any]) -> list[dict[str, str]]:
    """
    Read a published Challenge manifest and check it against its receipt.

    Parameters
    ----------
    directory : Path
        View directory.
    metadata : dict[str, Any]
        The view's ``metadata.json``.

    Returns
    -------
    list[dict[str, str]]
        Manifest rows.

    Raises
    ------
    ValueError
        If the view is unpublished, or rows, hashes or shards disagree with the receipt.
    """
    if not metadata.get("complete") or metadata.get("schema_version") != 1:
        raise ValueError(f"Unpublished or unsupported Challenge output: {directory}")
    manifest = directory / "manifest.csv"
    if sha256_file(manifest) != metadata["manifest_sha256"]:
        raise ValueError(f"Challenge manifest hash mismatch: {manifest}")
    data = read_csv(manifest)
    if len(data) != metadata["materialized_records"] or len(data) != metadata["accepted_records"]:
        raise ValueError(f"Challenge row-count mismatch: {directory}")
    if len({r["signal_sha256"] for r in data}) != len(data):
        raise ValueError(f"Duplicate canonical signal within view: {directory}")
    shards = _shard_records(metadata)
    if any(r["shard"] not in shards for r in data):
        raise ValueError(f"Unknown shard in manifest: {directory}")
    return data


def verify_challenge_sample(directory: Path, metadata: dict[str, Any], data: list[dict[str, str]],
                            sample_size: int, seed: int) -> int:
    """
    Recompute hashes for a deterministic sample of Challenge waveforms.

    Parameters
    ----------
    directory : Path
        View directory.
    metadata : dict[str, Any]
        The view's ``metadata.json``.
    data : list[dict[str, str]]
        Manifest rows.
    sample_size : int
        Number of waveforms to check.
    seed : int
        Sampling seed.

    Returns
    -------
    int
        Number of verified waveforms.

    Raises
    ------
    ValueError
        If a shard shape or a sampled waveform differs from the manifest.
    """
    records = _shard_records(metadata)
    shards: dict[str, np.ndarray] = {}
    sample = deterministic_indices(len(data), sample_size, seed)
    for index in sample:
        row = data[int(index)]
        shard = _load_shard(directory, row["shard"], shards)
        if shard.shape != (records[row["shard"]], *CHALLENGE_SHAPE):
            raise ValueError(f"Unexpected Challenge shard shape: {row['shard']}")
        signal = np.asarray(shard[int(row["shard_index"])])
        if signal_sha256(signal) != row["signal_sha256"] or not np.isfinite(signal).all():
            raise ValueError(f"Sampled Challenge waveform differs from manifest: {row['ecg_id']}")
    return len(sample)


def _source_summary(group: list[dict[str, str]]) -> dict[str, Any]:
    """Summarize the accepted rows of one Challenge source."""
    return {
        "accepted": len(group),
        "original_samples": _sorted_counts([r["source_samples"] for r in group]),
        "longer_than_10s": sum(int(r["source_samples"]) > CHALLENGE_SAMPLES for r in group),
        "qc_flags": flag_counts(group),
        "max_abs_mv": quantiles([float(r["max_abs_mv"]) for r in group]),
        "minimum_lead_std_mv": quantiles([float(r["min_lead_std_mv"]) for r in group]),
        "label_conflicts": bool_count(group, "duplicate_label_conflict"),
        "record_annotation_available": bool_count(group, "record_annotation_available"),
        "endpoint_supervised_eligible": bool_count(group, "endpoint_supervised_eligible"),
    }


def challenge_view(directory: Path, metadata: dict[str, Any], sample_size: int,
                   seed: int) -> tuple[dict[str, Any], list[dict[str, str]]]:
    """
    Validate and summarize one published Challenge view.

    Parameters
    ----------
    directory : Path
        View directory.
    metadata : dict[str, Any]
        The view's ``metadata.json``.
    sample_size : int
        Number of waveforms whose hashes are recomputed.
    seed : int
        Sampling seed.

    Returns
    -------
    tuple[dict[str, Any], list[dict[str, str]]]
        View summary and manifest rows.

    Raises
    ------
    ValueError
        If the manifest, sampled waveforms or recorded counts disagree with the receipt.
    """
    data = validate_challenge_manifest(directory, metadata)
    verified = verify_challenge_sample(directory, metadata, data, sample_size, seed)
    sources = {source: _source_summary([r for r in data if r["source"] == source])
               for source in sorted({r["source"] for r in data})}
    if sum(bool(r["qc_flags"]) for r in data) != metadata["counts"]["qc_flagged"]:
        raise ValueError(f"Challenge QC flag count mismatch: {directory}")
    if bool_count(data, "duplicate_label_conflict") != metadata["counts"]["label_conflict_retained"]:
        raise ValueError(f"Challenge conflict count mismatch: {directory}")
    original_codes = Counter(code.strip() for row in data
                             for code in row["original_label_codes"].split(",") if code.strip())
    summary = {
        "directory": str(directory.resolve().relative_to(ROOT)), "policy": metadata["policy"],
        "manifest_sha256": metadata["manifest_sha256"],
        "candidate_records": metadata["candidate_records"],
        "accepted_records": len(data), "excluded_records": metadata["excluded_records"],
        "source_counts": sources,
        "label_scope": _sorted_counts([r["label_scope"] for r in data]),
        "qc_flags": flag_counts(data),
        "label_conflicts": bool_count(data, "duplicate_label_conflict"),
        "record_annotation_available": bool_count(data, "record_annotation_available"),
        "endpoint_supervised_eligible": bool_count(data, "endpoint_supervised_eligible"),
        "patient_independent_eval_eligible": bool_count(data, "patient_independent_eval_eligible"),
        "patient_identity_known": bool_count(data, "patient_identity_known"),
        "overlap_by_record_flagged": bool_count(data, "overlap_by_record"),
        "overlap_by_signal_flagged": bool_count(data, "overlap_by_signal"),
        "original_snomed_code_top12": original_codes.most_common(TOP_COUNTS),
        "waveform_sample": {"seed": seed, "selected": verified, "hashes_verified": verified},
    }
    return summary, data


def review_flagged_waveforms(directory: Path, data: list[dict[str, str]]) -> list[dict[str, Any]]:
    """
    Inspect every accepted Challenge QC flag, without changing inclusion.

    Parameters
    ----------
    directory : Path
        View directory.
    data : list[dict[str, str]]
        Manifest rows.

    Returns
    -------
    list[dict[str, Any]]
        Peak, rail and flat-lead measurements for each flagged row.

    Raises
    ------
    ValueError
        If a flagged waveform's hash differs from the manifest.
    """
    inspected = []
    shards: dict[str, np.ndarray] = {}
    for row in data:
        if not row["qc_flags"]:
            continue
        signal = np.asarray(_load_shard(directory, row["shard"], shards)[int(row["shard_index"])])
        if signal_sha256(signal) != row["signal_sha256"]:
            raise ValueError(f"Flagged waveform hash mismatch: {row['ecg_id']}")
        inspected.append({
            "ecg_id": row["ecg_id"], "source": row["source"], "qc_flags": row["qc_flags"],
            "peak_positive_mv": float(signal.max()), "peak_negative_mv": float(signal.min()),
            "samples_abs_over_10mv": int((np.abs(signal) > HIGH_AMPLITUDE_MV).sum()),
            "samples_at_positive_32_767mv": int(
                np.isclose(signal, RAIL_POSITIVE_MV, atol=RAIL_TOLERANCE_MV).sum()),
            "samples_at_negative_32_768mv": int(
                np.isclose(signal, RAIL_NEGATIVE_MV, atol=RAIL_TOLERANCE_MV).sum()),
            "minimum_lead_std_mv": float(signal.std(axis=1).min()),
        })
    return inspected


def _has_rail_values(item: dict[str, Any]) -> bool:
    """Tell whether an inspected waveform reaches an exact ADC rail value."""
    return bool(item["samples_at_positive_32_767mv"] or item["samples_at_negative_32_768mv"])


def verified_rail_header(filename_hr: str, ecg_id: str) -> Path:
    """
    Check that a raw WFDB record uses signed 16-bit samples at 1000 counts/mV.

    Parameters
    ----------
    filename_hr : str
        Record stem relative to the raw Challenge release.
    ecg_id : str
        Record identifier, for error messages.

    Returns
    -------
    Path
        The verified ``.hea`` header.

    Raises
    ------
    ValueError
        If the path escapes the raw release or the header has another encoding.
    """
    raw_dir = ROOT / RAW_CHALLENGE
    stem = (raw_dir / filename_hr).resolve()
    if not stem.is_relative_to(raw_dir.resolve()):
        raise ValueError("Rail record path escaped Challenge source")
    header = wfdb.rdheader(str(stem))
    if (set(header.fmt) != {"16"} or set(header.adc_res) != {16}
            or set(header.adc_gain) != {1000.0} or set(header.baseline) != {0}
            or set(header.adc_zero) != {0} or set(header.units) != {"mV"}):
        raise ValueError(f"Rail-like value lacks signed-16/1000 mV encoding: {ecg_id}")
    return stem.with_suffix(".hea")


def rail_exclusion_overlay(strict_rows: list[dict[str, str]], crop_rows: list[dict[str, str]],
                           flagged_strict: list[dict[str, Any]],
                           flagged_crop: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """
    Recommend excluding exact signed-16 rail hits; leave other flags for review.

    Parameters
    ----------
    strict_rows, crop_rows : list[dict[str, str]]
        Manifest rows of the strict and centered views.
    flagged_strict, flagged_crop : list[dict[str, Any]]
        Inspected flagged waveforms of each view.

    Returns
    -------
    tuple[list[dict[str, Any]], dict[str, Any]]
        Overlay rows and a summary of the rule and its counts.
    """
    rows_by_view = {
        STRICT_VIEW: {r["ecg_id"]: r for r in strict_rows},
        CROP_VIEW: {r["ecg_id"]: r for r in crop_rows},
    }
    rail_hits = [(STRICT_VIEW, item) for item in flagged_strict if _has_rail_values(item)]
    rail_hits += [(CROP_VIEW, item) for item in flagged_crop if _has_rail_values(item)]
    overlay = []
    for view, item in rail_hits:
        source_row = rows_by_view[view][item["ecg_id"]]
        header_path = verified_rail_header(source_row["filename_hr"], item["ecg_id"])
        overlay.append({
            "view": view, "ecg_id": item["ecg_id"], "source": item["source"],
            "signal_sha256": source_row["signal_sha256"],
            "decision": "recommended_exclude_from_training",
            "reason": "exact_signed_16_adc_rail_hit",
            "positive_32767_samples": item["samples_at_positive_32_767mv"],
            "negative_32768_samples": item["samples_at_negative_32_768mv"],
            "wfdb_header_sha256": sha256_file(header_path),
        })
    summary = {
        "strict_view_rows": sum(r["view"] == STRICT_VIEW for r in overlay),
        "center_crop_view_rows": sum(r["view"] == CROP_VIEW for r in overlay),
        "unique_ecg_ids": len({r["ecg_id"] for r in overlay}),
        "positive_32767_samples_across_view_rows": sum(r["positive_32767_samples"] for r in overlay),
        "negative_32768_samples_across_view_rows": sum(r["negative_32768_samples"] for r in overlay),
        "rule": "exact physical value +32.767/-32.768 mV at a 16-bit ADC limit, "
                "verified from WFDB header gain=1000/mV, baseline=0",
        "scope": "recommended downstream training exclusion only; "
                 "published manifests and shards are unchanged",
        "other_qc_flags": "review only; no automatic exclusion from this overlay",
    }
    return overlay, summary


def _curated_row(view: str, directory: Path, row: dict[str, str],
                 shard_counts: dict[str, int]) -> dict[str, str]:
    """Build one SSL-only curated row that points into an existing shard."""
    ecg_id = row["ecg_id"]
    shard = (directory / row["shard"]).resolve()
    if not shard.is_file() or not shard.is_relative_to(ROOT):
        raise ValueError(f"Curated shard is missing or outside project: {shard}")
    if (row["shard"] not in shard_counts
            or not 0 <= int(row["shard_index"]) < shard_counts[row["shard"]]):
        raise ValueError(f"Curated shard index outside published receipt: {ecg_id}")
    return {
        "ecg_id": ecg_id, "source": row["source"], "view": view,
        "shard_path": str(shard.relative_to(ROOT)), "shard_index": row["shard_index"],
        "signal_sha256": row["signal_sha256"], "source_samples": row["source_samples"],
        "window_start": row["window_start"], "window_samples": row["window_samples"],
        "sampling_rate_hz": row["sampling_rate_hz"], "units": row["units"],
        "lead_order": row["lead_order"], "qc_flags": row["qc_flags"],
        "patient_id": row["patient_id"],
        "patient_identity_known": "false", "patient_independent_eval_eligible": "false",
        "label_scope": "ssl_only", "endpoint_supervised_eligible": "false",
    }


def _omission_reason(ecg_id: str, digest: str, rail_ids: set[str], seen_signals: set[str],
                    seen_ids: set[str]) -> str:
    """Return why a row is left out of the curated SSL index, or an empty string."""
    reason = ""
    if ecg_id in rail_ids:
        reason = "verified_signed_16_rail"
    elif digest in seen_signals:
        reason = "strict_first_exact_signal_overlap"
    elif ecg_id in seen_ids:
        reason = "strict_first_record_overlap"
    return reason


def _check_rail_overlay(rail_overlay: list[dict[str, Any]],
                       by_key: dict[tuple[str, str], dict[str, str]]) -> None:
    """Check that every overlay row matches exactly one published view row."""
    if len({(r["view"], r["ecg_id"]) for r in rail_overlay}) != len(rail_overlay):
        raise ValueError("Duplicate row in rail exclusion overlay")
    for excluded in rail_overlay:
        source = by_key.get((excluded["view"], excluded["ecg_id"]))
        if source is None or source["signal_sha256"] != excluded["signal_sha256"]:
            raise ValueError("Rail overlay row does not match a published view")


def _check_curated(selected: list[dict[str, str]], omitted: list[dict[str, str]], total: int,
                  rail_ids: set[str]) -> None:
    """Check uniqueness, accounting and label scope of the curated SSL index."""
    if (len({r["signal_sha256"] for r in selected}) != len(selected)
            or len({r["ecg_id"] for r in selected}) != len(selected)):
        raise ValueError("Curated SSL IDs or waveforms are not unique")
    if len(selected) + len(omitted) != total:
        raise ValueError("Curated SSL accounting mismatch")
    if any(r["ecg_id"] in rail_ids for r in selected):
        raise ValueError("Rail-hit ECG leaked into curated SSL candidate")
    if any(r["label_scope"] != "ssl_only" or r["endpoint_supervised_eligible"] != "false"
           for r in selected):
        raise ValueError("Curated SSL candidate has an endpoint label")


def curated_ssl_rows(strict_dir: Path, crop_dir: Path,
                     strict_rows: list[dict[str, str]], crop_rows: list[dict[str, str]],
                     rail_overlay: list[dict[str, Any]], shard_counts: dict[str, dict[str, int]],
                     ) -> tuple[list[dict[str, str]], list[dict[str, str]], dict[str, Any]]:
    """
    Build a strict-first SSL-only candidate index into existing immutable shards.

    Parameters
    ----------
    strict_dir, crop_dir : Path
        Strict and centered view directories.
    strict_rows, crop_rows : list[dict[str, str]]
        Their manifest rows.
    rail_overlay : list[dict[str, Any]]
        Rows recommended for exclusion.
    shard_counts : dict[str, dict[str, int]]
        Records per shard, keyed by view name.

    Returns
    -------
    tuple[list[dict[str, str]], list[dict[str, str]], dict[str, Any]]
        Selected rows, omitted rows and their counts.

    Raises
    ------
    ValueError
        If the overlay does not match the views or a curated invariant fails.
    """
    rail_ids = {r["ecg_id"] for r in rail_overlay}
    all_rows = [(STRICT_VIEW, strict_dir, row) for row in strict_rows]
    all_rows.extend((CROP_VIEW, crop_dir, row) for row in crop_rows)
    _check_rail_overlay(rail_overlay, {(view, row["ecg_id"]): row for view, _, row in all_rows})
    selected, omitted = [], []
    seen_signals, seen_ids = set(), set()
    for view, directory, row in all_rows:
        ecg_id, digest = row["ecg_id"], row["signal_sha256"]
        reason = _omission_reason(ecg_id, digest, rail_ids, seen_signals, seen_ids)
        if reason:
            omitted.append({"view": view, "ecg_id": ecg_id, "signal_sha256": digest, "reason": reason})
            continue
        selected.append(_curated_row(view, directory, row, shard_counts[view]))
        seen_signals.add(digest)
        seen_ids.add(ecg_id)
    _check_curated(selected, omitted, len(all_rows), rail_ids)
    counts = {
        "strict_input_rows": len(strict_rows), "centered_input_rows": len(crop_rows),
        "selected_total": len(selected),
        "selected_strict": sum(r["view"] == STRICT_VIEW for r in selected),
        "selected_centered": sum(r["view"] == CROP_VIEW for r in selected),
        "omitted_total": len(omitted),
        "omitted_by_reason": _sorted_counts([r["reason"] for r in omitted]),
        "rail_unique_ecg_ids": len(rail_ids),
        "duplicate_signal_count_after_curation": 0,
        "duplicate_ecg_id_count_after_curation": 0,
    }
    return selected, omitted, counts


def exclusions_by_source(metadata: dict[str, Any]) -> dict[str, dict[str, int]]:
    """
    Count audit exclusions by source and reason, after checking their hashes.

    Parameters
    ----------
    metadata : dict[str, Any]
        Challenge view ``metadata.json``, which names its prepared audit.

    Returns
    -------
    dict[str, dict[str, int]]
        Exclusion reasons and counts per source, both sorted.

    Raises
    ------
    ValueError
        If the audit files changed since materialization or the count differs.
    """
    prepared = Path(metadata["input_prepared_dir"])
    if sha256_file(prepared / "metadata.json") != metadata["input_metadata_sha256"]:
        raise ValueError("Challenge audit metadata changed since materialization")
    excluded = read_csv(prepared / "exclusions.csv")
    if sha256_file(prepared / "exclusions.csv") != metadata["input_exclusions_sha256"]:
        raise ValueError("Challenge exclusions changed since materialization")
    if len(excluded) != metadata["excluded_records"]:
        raise ValueError("Challenge exclusion count mismatch")
    grouped: dict[str, Counter] = defaultdict(Counter)
    for item in excluded:
        grouped[item["source"]][item["reason"]] += 1
    return {source: dict(sorted(counts.items())) for source, counts in sorted(grouped.items())}


def check_source_candidates(metadata: dict[str, Any], summary: dict[str, Any]) -> None:
    """
    Check that accepted plus excluded records equal each source's candidates.

    Parameters
    ----------
    metadata : dict[str, Any]
        Challenge view ``metadata.json``.
    summary : dict[str, Any]
        View summary including ``exclusions_by_source``.

    Raises
    ------
    ValueError
        If a source's records are not fully accounted for.
    """
    for source, provenance in metadata["official_provenance"].items():
        accepted = summary["source_counts"].get(source, {}).get("accepted", 0)
        excluded = sum(summary["exclusions_by_source"].get(source, {}).values())
        if accepted + excluded != provenance["candidate_records"]:
            raise ValueError(f"Challenge source candidate count mismatch: {source}")


def cross_view_overlap(strict: dict[str, Any], crop: dict[str, Any], strict_rows: list[dict[str, str]],
                       crop_rows: list[dict[str, str]]) -> dict[str, Any]:
    """
    Recompute record and signal overlap between the two Challenge views.

    Parameters
    ----------
    strict, crop : dict[str, Any]
        View summaries with their recorded overlap flag counts.
    strict_rows, crop_rows : list[dict[str, str]]
        Manifest rows of each view.

    Returns
    -------
    dict[str, Any]
        Overlap and union counts.

    Raises
    ------
    ValueError
        If either view's overlap flags differ from the recomputed overlap.
    """
    strict_ids = {r["ecg_id"] for r in strict_rows}
    crop_ids = {r["ecg_id"] for r in crop_rows}
    strict_hashes = {r["signal_sha256"] for r in strict_rows}
    crop_hashes = {r["signal_sha256"] for r in crop_rows}
    record_overlap = strict_ids & crop_ids
    signal_overlap = strict_hashes & crop_hashes
    if (len(record_overlap) != strict["overlap_by_record_flagged"]
            or len(signal_overlap) != strict["overlap_by_signal_flagged"]):
        raise ValueError("Strict manifest overlap flags differ from recomputed overlap")
    if (len(record_overlap) != crop["overlap_by_record_flagged"]
            or len(signal_overlap) != crop["overlap_by_signal_flagged"]):
        raise ValueError("Crop manifest overlap flags differ from recomputed overlap")
    return {
        "same_ecg_id": len(record_overlap), "same_canonical_signal": len(signal_overlap),
        "unique_ecg_ids_across_views": len(strict_ids | crop_ids),
        "unique_canonical_signals_across_views": len(strict_hashes | crop_hashes),
        "interpretation": "Two overlapping views, not independent cohorts; exact matching only.",
    }


def read_code_tables(directory: Path, metadata: dict[str, Any]) -> dict[str, list[dict[str, str]]]:
    """
    Read the CODE native tables and check them against their receipt.

    Parameters
    ----------
    directory : Path
        CODE native output directory.
    metadata : dict[str, Any]
        Its ``metadata.json``.

    Returns
    -------
    dict[str, list[dict[str, str]]]
        Rows keyed by ``manifest``, ``exclusions``, ``demographics`` and ``labels``.

    Raises
    ------
    ValueError
        If the output is limited, checksums or counts differ, or tables are misaligned.
    """
    if (metadata.get("limit_per_archive") is not None
            or metadata.get("canonical_500hz_10s_eligible") is not False):
        raise ValueError("CODE output is a limited or unexpectedly canonicalized view")
    for name in CODE_TABLES:
        if sha256_file(directory / name) != metadata["output_sha256"][name]:
            raise ValueError(f"CODE output checksum mismatch: {name}")
    manifest = read_csv(directory / "manifest.csv")
    exclusions = read_csv(directory / "exclusions.csv")
    demographics = read_csv(directory / "demographics.csv")
    labels = read_csv(directory / "source_labels.csv")
    counts = metadata["counts"]
    if len(manifest) != counts["accepted"] or len(exclusions) + len(manifest) != counts["audited"]:
        raise ValueError("CODE accepted and excluded rows do not match receipt")
    if len(demographics) != len(manifest) or len(labels) != len(manifest):
        raise ValueError("CODE linked tables do not match manifest length")
    if sum(bool(r["qc_flags"]) for r in manifest) != counts["review_flagged"]:
        raise ValueError("CODE review-flag count mismatch")
    identifiers = [r["exam_id"] for r in manifest]
    if (len(set(identifiers)) != len(identifiers)
            or identifiers != [r["exam_id"] for r in demographics]
            or identifiers != [r["exam_id"] for r in labels]):
        raise ValueError("CODE exam IDs are duplicated or linked tables are misaligned")
    return {"manifest": manifest, "exclusions": exclusions, "demographics": demographics, "labels": labels}


def verify_code_sample(directory: Path, manifest: list[dict[str, str]], sample_size: int,
                       seed: int) -> tuple[list[float], list[float]]:
    """
    Recompute hashes for a deterministic sample of CODE native traces.

    Parameters
    ----------
    directory : Path
        CODE native output directory.
    manifest : list[dict[str, str]]
        Accepted manifest rows, aligned with the HDF5 rows.
    sample_size : int
        Number of traces to check.
    seed : int
        Sampling seed.

    Returns
    -------
    tuple[list[float], list[float]]
        Maximum absolute value and minimum lead standard deviation of each sampled trace.

    Raises
    ------
    ValueError
        If the HDF5 shape, an exam ID or a sampled trace differs from the manifest.
    """
    sampled_max_abs, sampled_min_std = [], []
    with h5py.File(directory / CODE_HDF5, "r") as handle:
        signals, ids = handle["tracings"], handle["exam_id"]
        if len(signals) != len(manifest) or signals.shape[1:] != CODE_SHAPE or len(ids) != len(signals):
            raise ValueError("CODE native HDF5 shape/count mismatch")
        for index in deterministic_indices(len(manifest), sample_size, seed):
            i = int(index)
            trace = np.asarray(signals[i])
            if str(int(ids[i])) != manifest[i]["exam_id"]:
                raise ValueError("CODE native HDF5 exam ID mismatch")
            if signal_sha256(trace) != manifest[i]["signal_sha256"] or not np.isfinite(trace).all():
                raise ValueError(f"CODE sampled waveform hash mismatch: {manifest[i]['exam_id']}")
            sampled_max_abs.append(float(np.max(np.abs(trace))))
            sampled_min_std.append(float(np.std(trace, axis=0).min()))
    return sampled_max_abs, sampled_min_std


def code_view(directory: Path, sample_size: int, seed: int) -> tuple[dict[str, Any], dict[str, Any]]:
    """
    Validate and summarize the CODE-15% native part 0 output.

    Parameters
    ----------
    directory : Path
        CODE native output directory.
    sample_size : int
        Number of traces whose hashes are recomputed.
    seed : int
        Sampling seed.

    Returns
    -------
    tuple[dict[str, Any], dict[str, Any]]
        Summary and the ages and edge counts used by the figure.
    """
    metadata = _read_json(directory / "metadata.json")
    tables = read_code_tables(directory, metadata)
    manifest, exclusions = tables["manifest"], tables["exclusions"]
    demographics, labels = tables["demographics"], tables["labels"]
    age = [float(r["age"]) for r in demographics if r["age"] and 0 <= float(r["age"]) <= MAX_PLAUSIBLE_AGE]
    patient_counts = Counter(r["patient_id"] for r in manifest)
    edge_counts = Counter((int(r["edge_zero_left"]), int(r["edge_zero_right"])) for r in manifest)
    sampled_max_abs, sampled_min_std = verify_code_sample(directory, manifest, sample_size, seed)
    diagnosis = {field: sum(r[field] == "True" for r in labels) for field in CODE_DIAGNOSES}
    young_low, young_high = YOUNG_ADULT_AGES
    summary = {
        "directory": str(directory.resolve().relative_to(ROOT)),
        "manifest_sha256": metadata["output_sha256"]["manifest.csv"],
        "audited_records": metadata["counts"]["audited"],
        "accepted_records": len(manifest), "excluded_records": len(exclusions),
        "exclusions": _sorted_counts([r["reason"] for r in exclusions]),
        "distinct_patients": len(patient_counts),
        "patients_with_multiple_exams": sum(n > 1 for n in patient_counts.values()),
        "max_exams_per_patient": max(patient_counts.values()),
        "age_years": quantiles(age), "age_known": len(age),
        "age_18_to_30": sum(young_low <= value <= young_high for value in age),
        "sex_values": _sorted_counts([r["is_male"] for r in demographics]),
        "released_flag_positive_counts": diagnosis,
        "qc_flags": flag_counts(manifest),
        "edge_zero_top12": [[list(pair), count] for pair, count in edge_counts.most_common(TOP_COUNTS)],
        "edge_zero_unique_pairs": len(edge_counts),
        "edge_zero_symmetric": sum(count for (left, right), count in edge_counts.items() if left == right),
        "edge_zero_dominant_0_0": edge_counts[(0, 0)],
        "edge_zero_dominant_581_581": edge_counts[(581, 581)],
        "edge_zero_other": len(manifest) - edge_counts[(0, 0)] - edge_counts[(581, 581)],
        "native_waveform_sample": {"seed": seed, "hashes_verified": len(sampled_max_abs),
                                   "max_abs_native_units": quantiles(sampled_max_abs),
                                   "minimum_lead_std_native_units": quantiles(sampled_min_std)},
        "canonical_500hz_10s_eligible": False,
        "label_provenance": {"diagnosis": sorted({r["diagnosis_label_provenance"] for r in labels}),
                             "normal_ecg": sorted({r["normal_ecg_provenance"] for r in labels})},
    }
    figure_data = {"ages": age, "edge_counts": edge_counts}
    return summary, figure_data


def load_pyplot() -> ModuleType:
    """
    Import pyplot with the non-interactive Agg backend.

    Matplotlib is imported lazily so validation-only use does not load it.

    Returns
    -------
    ModuleType
        ``matplotlib.pyplot``.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def plot_audit_outcomes(ax: Any, strict: dict[str, Any]) -> None:
    """Stack accepted, duration-excluded and other-excluded strict records per source."""
    accepted = [strict["source_counts"][key]["accepted"] for key in FIGURE_SOURCES]
    exclusions = strict["exclusions_by_source"]
    duration = [exclusions.get(key, {}).get("duration_contract", 0) for key in FIGURE_SOURCES]
    other = [sum(exclusions.get(key, {}).values()) - value
             for key, value in zip(FIGURE_SOURCES, duration, strict=True)]
    ax.bar(FIGURE_SOURCE_NAMES, accepted, label="Accepted", color=BLUE)
    ax.bar(FIGURE_SOURCE_NAMES, duration, bottom=accepted, label="Duration excluded", color=GOLD)
    ax.bar(FIGURE_SOURCE_NAMES, other, bottom=np.asarray(accepted) + duration, label="Other excluded",
           color=RED)
    ax.set_title("Strict 10 s audit outcomes")
    ax.set_ylabel("Recordings")
    ax.legend(fontsize=8)
    ax.tick_params(axis="x", rotation=12)


def plot_source_amplitudes(ax: Any, strict: dict[str, Any], crop: dict[str, Any]) -> None:
    """Plot each view's median peak absolute amplitude per source."""
    for name, summary, color in (("Strict 10 s", strict, BLUE), ("CPSC centered", crop, ORANGE)):
        # Source medians are plotted rather than pooling groups with different sizes.
        present = [key for key in FIGURE_SOURCES if key in summary["source_counts"]]
        medians = [summary["source_counts"][key]["max_abs_mv"]["median"] for key in present]
        labels = [key.replace("_", " ") for key in present]
        ax.scatter(labels, medians, label=name, color=color, s=55)
    ax.set_title("Median peak absolute amplitude by source")
    ax.set_ylabel("mV; full accepted manifests")
    ax.tick_params(axis="x", rotation=12)
    ax.legend(fontsize=8)


def plot_code_ages(ax: Any, ages: list[float]) -> None:
    """Histogram accepted CODE ages in five-year bins."""
    ax.hist(ages, bins=np.arange(0, 111, 5), color=TEAL, edgecolor="white")
    ax.set_title("CODE part 0 accepted ages")
    ax.set_xlabel("Years")
    ax.set_ylabel("Exams")


def plot_code_edges(ax: Any, code: dict[str, Any]) -> None:
    """Bar-plot the dominant exact-zero edge-run patterns in CODE."""
    names = ["0 / 0", "581 / 581", "Other"]
    counts = [code["edge_zero_dominant_0_0"], code["edge_zero_dominant_581_581"], code["edge_zero_other"]]
    ax.bar(names, counts, color=[TEAL, SAGE, GOLD])
    ax.set_title("CODE exact-zero edge runs")
    ax.set_xlabel("Left / right samples; not proven duration")
    ax.set_ylabel("Exams")


def make_figure(path: Path, strict: dict[str, Any], crop: dict[str, Any], code: dict[str, Any],
                code_figure: dict[str, Any]) -> None:
    """
    Save the four-panel overview figure.

    Parameters
    ----------
    path : Path
        Destination PNG.
    strict, crop : dict[str, Any]
        Challenge view summaries.
    code : dict[str, Any]
        CODE summary.
    code_figure : dict[str, Any]
        CODE ages for the histogram.
    """
    plt = load_pyplot()
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    plot_audit_outcomes(axes[0, 0], strict)
    plot_source_amplitudes(axes[0, 1], strict, crop)
    plot_code_ages(axes[1, 0], code_figure["ages"])
    plot_code_edges(axes[1, 1], code)
    fig.suptitle("Processed ECG exploratory analysis; no clinical quality or endpoint labels inferred",
                 fontsize=12)
    fig.savefig(path, dpi=FIGURE_DPI)
    plt.close(fig)


def first_per_source(rows: list[dict[str, str]], count: int) -> list[dict[str, str]]:
    """
    Keep the first row of each source, in order, up to ``count`` rows.

    Parameters
    ----------
    rows : list[dict[str, str]]
        Candidate rows in priority order.
    count : int
        Maximum number of rows.

    Returns
    -------
    list[dict[str, str]]
        Rows from distinct sources.
    """
    selected: list[dict[str, str]] = []
    for row in rows:
        if len(selected) == count:
            break
        if row["source"] not in {r["source"] for r in selected}:
            selected.append(row)
    return selected


def review_examples(data: list[dict[str, str]]) -> list[tuple[dict[str, str], str]]:
    """
    Choose the highest-amplitude and near-flat review examples from distinct sources.

    Parameters
    ----------
    data : list[dict[str, str]]
        Strict manifest rows.

    Returns
    -------
    list[tuple[dict[str, str], str]]
        Rows paired with ``"peak"`` or ``"flat"``.

    Raises
    ------
    ValueError
        If two examples of each kind are unavailable.
    """
    high = sorted((r for r in data if "amplitude_over_10mV_review" in r["qc_flags"]),
                  key=lambda r: -float(r["max_abs_mv"]))
    flat = [r for r in data if "near_flat_lead_review" in r["qc_flags"]]
    selected = [(row, "peak") for row in first_per_source(high, REVIEW_EXAMPLES_PER_FLAG)]
    selected += [(row, "flat") for row in first_per_source(flat, REVIEW_EXAMPLES_PER_FLAG)]
    if len(selected) != 2 * REVIEW_EXAMPLES_PER_FLAG:
        raise ValueError("Expected two high-amplitude and two near-flat review examples")
    return selected


def make_review_figure(path: Path, directory: Path, data: list[dict[str, str]]) -> None:
    """
    Show representative flagged leads for human inspection, without adjudication.

    Parameters
    ----------
    path : Path
        Destination PNG.
    directory : Path
        Strict view directory.
    data : list[dict[str, str]]
        Strict manifest rows.
    """
    selected = review_examples(data)
    plt = load_pyplot()
    fig, axes = plt.subplots(2, 2, figsize=(13, 6), constrained_layout=True)
    time = np.arange(CHALLENGE_SAMPLES) / CHALLENGE_RATE_HZ
    shards: dict[str, np.ndarray] = {}
    for ax, (row, kind) in zip(axes.flat, selected, strict=True):
        signal = _load_shard(directory, row["shard"], shards)[int(row["shard_index"])]
        if kind == "peak":
            lead = int(np.unravel_index(np.argmax(np.abs(signal)), signal.shape)[0])
            ax.plot(time, signal[lead], lw=.8, color=TRACE_RED, label=LEADS[lead])
        else:
            lead = int(signal.std(axis=1).argmin())
            ax.plot(time, signal[lead], lw=.8, color=TRACE_RED, label=f"{LEADS[lead]} near-flat")
            ax.plot(time, signal[1], lw=.6, alpha=.6, color=BLUE, label="II context")
        ax.set_title(f"{row['ecg_id']} — {kind} review")
        ax.set_xlabel("Seconds")
        ax.set_ylabel("mV")
        ax.legend(fontsize=8, loc="upper right")
    fig.suptitle("Selected review flags; displayed traces are not clinical diagnoses")
    fig.savefig(path, dpi=FIGURE_DPI)
    plt.close(fig)


def write_outputs(output_dir: Path, result: dict[str, Any], flagged_strict: list[dict[str, Any]],
                  flagged_crop: list[dict[str, Any]], rail_overlay: list[dict[str, Any]],
                  curated: list[dict[str, str]], curated_omitted: list[dict[str, str]]) -> None:
    """
    Write the summary JSON and the review, overlay and curated CSV files.

    Parameters
    ----------
    output_dir : Path
        Output directory.
    result : dict[str, Any]
        Full EDA summary.
    flagged_strict, flagged_crop : list[dict[str, Any]]
        Inspected flagged waveforms of each view.
    rail_overlay : list[dict[str, Any]]
        Rail exclusion overlay rows.
    curated, curated_omitted : list[dict[str, str]]
        Curated SSL rows and omitted rows.
    """
    write_json_atomic(output_dir / "summary.json", result, sort_keys=True, allow_nan=True)
    review_rows = [{"view": STRICT_VIEW, **item} for item in flagged_strict]
    review_rows += [{"view": CROP_VIEW, **item} for item in flagged_crop]
    write_csv_atomic(output_dir / "challenge_review_flags.csv", review_rows, REVIEW_FIELDS)
    write_csv_atomic(output_dir / "training_exclusion_overlay.csv", rail_overlay, OVERLAY_FIELDS)
    write_csv_atomic(output_dir / "challenge_ssl_curated_manifest.csv", curated, CURATED_FIELDS)
    write_csv_atomic(output_dir / "challenge_ssl_curated_omissions.csv", curated_omitted, OMISSION_FIELDS)


def write_curated_receipt(output_dir: Path, strict: dict[str, Any], crop: dict[str, Any],
                          curated_counts: dict[str, Any]) -> None:
    """
    Write the receipt that binds the curated SSL index to its inputs.

    Parameters
    ----------
    output_dir : Path
        Output directory containing the written overlay and curated CSV files.
    strict, crop : dict[str, Any]
        Challenge view summaries.
    curated_counts : dict[str, Any]
        Curated selection counts.
    """
    receipt = {
        "status": "candidate_ssl_only_not_scheduled",
        "source_sha256": sha256_file(Path(__file__)),
        "strict_manifest_sha256": strict["manifest_sha256"],
        "centered_manifest_sha256": crop["manifest_sha256"],
        "rail_overlay_sha256": sha256_file(output_dir / "training_exclusion_overlay.csv"),
        "curated_manifest_sha256": sha256_file(output_dir / "challenge_ssl_curated_manifest.csv"),
        "curated_omissions_sha256": sha256_file(output_dir / "challenge_ssl_curated_omissions.csv"),
        "counts": curated_counts,
        "data_contract": "[12,5000] float32 physical mV, 500 Hz, original unfiltered materialized shards",
        "label_contract": "ssl_only; no endpoint labels and no patient-independent evaluation eligibility",
        "patient_identity": "Challenge patient IDs unavailable; patient_id is a record surrogate",
        "storage": "manifest references existing verified shard/index pairs; "
                   "arrays are not copied or modified",
    }
    write_json_atomic(output_dir / "challenge_ssl_curated_receipt.json", receipt,
                      sort_keys=True, allow_nan=True)


def run(strict_dir: Path, crop_dir: Path, code_dir: Path, output_dir: Path,
        sample_size: int = 128, seed: int = 42) -> dict[str, Any]:
    """
    Validate the published views, write the EDA outputs and return the summary.

    Parameters
    ----------
    strict_dir, crop_dir : Path
        Strict and centered Challenge view directories.
    code_dir : Path
        CODE native part 0 directory.
    output_dir : Path
        Destination; must not be inside ``data/raw``.
    sample_size : int
        Waveforms per view whose hashes are recomputed.
    seed : int
        Sampling seed; the centered and CODE views use ``seed + 1`` and ``seed + 2``.

    Returns
    -------
    dict[str, Any]
        The summary written to ``summary.json``.

    Raises
    ------
    ValueError
        If the arguments are invalid or any published output fails a check.
    """
    if sample_size < 1:
        raise ValueError("sample_size must be positive")
    if output_dir.resolve().is_relative_to((ROOT / "data/raw").resolve()):
        raise ValueError("EDA output cannot be inside data/raw")
    strict_metadata = _read_json(strict_dir / "metadata.json")
    crop_metadata = _read_json(crop_dir / "metadata.json")
    strict, strict_rows = challenge_view(strict_dir, strict_metadata, sample_size, seed)
    crop, crop_rows = challenge_view(crop_dir, crop_metadata, sample_size, seed + 1)
    if strict["policy"] != "strict_10s" or crop["policy"] != "ssl_center_crop":
        raise ValueError("Unexpected Challenge view policies")
    strict["exclusions_by_source"] = exclusions_by_source(strict_metadata)
    crop["exclusions_by_source"] = exclusions_by_source(crop_metadata)
    check_source_candidates(strict_metadata, strict)
    check_source_candidates(crop_metadata, crop)
    flagged_strict = review_flagged_waveforms(strict_dir, strict_rows)
    flagged_crop = review_flagged_waveforms(crop_dir, crop_rows)
    rail_overlay, rail_summary = rail_exclusion_overlay(strict_rows, crop_rows, flagged_strict, flagged_crop)
    shard_counts = {STRICT_VIEW: _shard_records(strict_metadata), CROP_VIEW: _shard_records(crop_metadata)}
    curated, curated_omitted, curated_counts = curated_ssl_rows(
        strict_dir, crop_dir, strict_rows, crop_rows, rail_overlay, shard_counts)
    strict["flagged_waveforms_inspected"] = len(flagged_strict)
    strict["rail_value_examples"] = [r for r in flagged_strict if _has_rail_values(r)]
    crop["flagged_waveforms_inspected"] = len(flagged_crop)
    cross_view = cross_view_overlap(strict, crop, strict_rows, crop_rows)
    code, code_figure = code_view(code_dir, sample_size, seed + 2)
    result = {
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "eda_source_sha256": sha256_file(Path(__file__)),
        "sampling": {"sample_size_per_view": sample_size, "seed": seed},
        "strict": strict, "center_crop": crop, "code_part0": code,
        "recommended_rail_exclusions": rail_summary,
        "curated_ssl_candidate": curated_counts,
        "challenge_cross_view": cross_view,
        "limitations": LIMITATIONS,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    write_outputs(output_dir, result, flagged_strict, flagged_crop, rail_overlay, curated, curated_omitted)
    write_curated_receipt(output_dir, strict, crop, curated_counts)
    make_figure(output_dir / "overview.png", strict, crop, code, code_figure)
    make_review_figure(output_dir / "review_examples.png", strict_dir, strict_rows)
    return result


def main() -> None:
    """Run the EDA from the command line and print where outputs were written."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--strict-dir", type=Path, default=STRICT)
    parser.add_argument("--crop-dir", type=Path, default=CROP)
    parser.add_argument("--code-dir", type=Path, default=CODE)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT)
    parser.add_argument("--sample-size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    result = run(args.strict_dir, args.crop_dir, args.code_dir, args.output_dir,
                 args.sample_size, args.seed)
    print(json.dumps({"summary": str(args.output_dir / "summary.json"),
                      "figure": str(args.output_dir / "overview.png"),
                      "review_examples": str(args.output_dir / "review_examples.png"),
                      "strict_records": result["strict"]["accepted_records"],
                      "center_crop_records": result["center_crop"]["accepted_records"],
                      "code_part0_records": result["code_part0"]["accepted_records"]}, indent=2))


if __name__ == "__main__":
    main()
