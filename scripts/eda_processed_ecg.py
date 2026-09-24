#!/usr/bin/env python3
"""Reproducible EDA of published Challenge views and CODE-15% native part 0.

Reads only completed processed outputs. It does not change ECG arrays or labels.
Challenge amplitude statistics are those recorded for *all* accepted views by the
materializer; a deterministic waveform sample independently checks their hashes.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import h5py
import numpy as np
import wfdb


ROOT = Path(__file__).resolve().parents[1]
STRICT = ROOT / "data/processed/challenge_ecg_views/strict_10s"
CROP = ROOT / "data/processed/challenge_ecg_views/cpsc_ssl_center_crop"
CODE = ROOT / "data/processed/code15_quality/part0_native"
OUTPUT = ROOT / "outputs/data_quality/processed_eda"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def quantiles(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {key: None for key in ("min", "p05", "p25", "median", "p75", "p95", "p99", "max")}
    array = np.asarray(values, dtype=np.float64)
    return {key: float(value) for key, value in zip(
        ("min", "p05", "p25", "median", "p75", "p95", "p99", "max"),
        np.quantile(array, (0, .05, .25, .5, .75, .95, .99, 1)))}


def bool_count(data: list[dict[str, str]], key: str) -> int:
    return sum(row[key] == "true" for row in data)


def flag_counts(data: list[dict[str, str]]) -> dict[str, int]:
    counts = Counter(flag for row in data for flag in row["qc_flags"].split(";") if flag)
    return dict(sorted(counts.items()))


def deterministic_indices(count: int, sample_size: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(count, size=min(count, sample_size), replace=False))


def challenge_view(directory: Path, sample_size: int, seed: int) -> tuple[dict, list[dict[str, str]]]:
    metadata = json.loads((directory / "metadata.json").read_text())
    if not metadata.get("complete") or metadata.get("schema_version") != 1:
        raise ValueError(f"Unpublished or unsupported Challenge output: {directory}")
    manifest = directory / "manifest.csv"
    if sha256(manifest) != metadata["manifest_sha256"]:
        raise ValueError(f"Challenge manifest hash mismatch: {manifest}")
    data = rows(manifest)
    if len(data) != metadata["materialized_records"] or len(data) != metadata["accepted_records"]:
        raise ValueError(f"Challenge row-count mismatch: {directory}")
    if len({r["signal_sha256"] for r in data}) != len(data):
        raise ValueError(f"Duplicate canonical signal within view: {directory}")
    expected_shards = {part["file"]: part for part in metadata["shards"]}
    if any(r["shard"] not in expected_shards for r in data):
        raise ValueError(f"Unknown shard in manifest: {directory}")
    checks = Counter()
    shard_cache = {}
    for index in deterministic_indices(len(data), sample_size, seed):
        row = data[int(index)]
        name = row["shard"]
        if name not in shard_cache:
            shard_cache[name] = np.load(directory / name, mmap_mode="r", allow_pickle=False)
            info = expected_shards[name]
            if shard_cache[name].shape != (info["records"], 12, 5000):
                raise ValueError(f"Unexpected Challenge shard shape: {name}")
        signal = np.asarray(shard_cache[name][int(row["shard_index"])])
        digest = hashlib.sha256(np.ascontiguousarray(signal, dtype="<f4").tobytes()).hexdigest()
        if digest != row["signal_sha256"] or not np.isfinite(signal).all():
            raise ValueError(f"Sampled Challenge waveform differs from manifest: {row['ecg_id']}")
        checks["waveform_hashes_verified"] += 1
    sources = {}
    for source in sorted({r["source"] for r in data}):
        group = [r for r in data if r["source"] == source]
        sources[source] = {
            "accepted": len(group),
            "original_samples": dict(sorted(Counter(r["source_samples"] for r in group).items())),
            "longer_than_10s": sum(int(r["source_samples"]) > 5000 for r in group),
            "qc_flags": flag_counts(group),
            "max_abs_mv": quantiles([float(r["max_abs_mv"]) for r in group]),
            "minimum_lead_std_mv": quantiles([float(r["min_lead_std_mv"]) for r in group]),
            "label_conflicts": bool_count(group, "duplicate_label_conflict"),
            "record_annotation_available": bool_count(group, "record_annotation_available"),
            "endpoint_supervised_eligible": bool_count(group, "endpoint_supervised_eligible"),
        }
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
        "label_scope": dict(sorted(Counter(r["label_scope"] for r in data).items())),
        "qc_flags": flag_counts(data),
        "label_conflicts": bool_count(data, "duplicate_label_conflict"),
        "record_annotation_available": bool_count(data, "record_annotation_available"),
        "endpoint_supervised_eligible": bool_count(data, "endpoint_supervised_eligible"),
        "patient_independent_eval_eligible": bool_count(data, "patient_independent_eval_eligible"),
        "patient_identity_known": bool_count(data, "patient_identity_known"),
        "overlap_by_record_flagged": bool_count(data, "overlap_by_record"),
        "overlap_by_signal_flagged": bool_count(data, "overlap_by_signal"),
        "original_snomed_code_top12": original_codes.most_common(12),
        "waveform_sample": {"seed": seed, "selected": int(checks["waveform_hashes_verified"]),
                            "hashes_verified": int(checks["waveform_hashes_verified"])},
    }
    return summary, data


def review_flagged_waveforms(directory: Path, data: list[dict[str, str]]) -> list[dict]:
    """Inspect every accepted Challenge QC flag, without changing inclusion."""
    inspected = []
    shard_cache = {}
    for row in data:
        if not row["qc_flags"]:
            continue
        name = row["shard"]
        if name not in shard_cache:
            shard_cache[name] = np.load(directory / name, mmap_mode="r", allow_pickle=False)
        signal = np.asarray(shard_cache[name][int(row["shard_index"])])
        digest = hashlib.sha256(np.ascontiguousarray(signal, dtype="<f4").tobytes()).hexdigest()
        if digest != row["signal_sha256"]:
            raise ValueError(f"Flagged waveform hash mismatch: {row['ecg_id']}")
        inspected.append({
            "ecg_id": row["ecg_id"], "source": row["source"], "qc_flags": row["qc_flags"],
            "peak_positive_mv": float(signal.max()), "peak_negative_mv": float(signal.min()),
            "samples_abs_over_10mv": int((np.abs(signal) > 10).sum()),
            "samples_at_positive_32_767mv": int(np.isclose(signal, 32.767, atol=1e-4).sum()),
            "samples_at_negative_32_768mv": int(np.isclose(signal, -32.768, atol=1e-4).sum()),
            "minimum_lead_std_mv": float(signal.std(axis=1).min()),
        })
    return inspected


def rail_exclusion_overlay(strict_rows: list[dict[str, str]], crop_rows: list[dict[str, str]],
                           flagged_strict: list[dict], flagged_crop: list[dict]) -> tuple[list[dict], dict]:
    """Recommend excluding exact signed-16 rail hits; leave other flags for review."""
    rows_by_view = {
        "strict_10s": {r["ecg_id"]: r for r in strict_rows},
        "cpsc_ssl_center_crop": {r["ecg_id"]: r for r in crop_rows},
    }
    overlay = []
    for view, flagged in (("strict_10s", flagged_strict),
                          ("cpsc_ssl_center_crop", flagged_crop)):
        for item in flagged:
            positive = item["samples_at_positive_32_767mv"]
            negative = item["samples_at_negative_32_768mv"]
            if not positive and not negative:
                continue
            source_row = rows_by_view[view][item["ecg_id"]]
            raw_dir = ROOT / "data/raw/challenge-2020/1.0.2"
            stem = (raw_dir / source_row["filename_hr"]).resolve()
            if not stem.is_relative_to(raw_dir.resolve()):
                raise ValueError("Rail record path escaped Challenge source")
            header_path = stem.with_suffix(".hea")
            header = wfdb.rdheader(str(stem))
            if (set(header.fmt) != {"16"} or set(header.adc_res) != {16}
                    or set(header.adc_gain) != {1000.0} or set(header.baseline) != {0}
                    or set(header.adc_zero) != {0} or set(header.units) != {"mV"}):
                raise ValueError(f"Rail-like value lacks signed-16/1000 mV encoding: {item['ecg_id']}")
            overlay.append({
                "view": view, "ecg_id": item["ecg_id"], "source": item["source"],
                "signal_sha256": source_row["signal_sha256"],
                "decision": "recommended_exclude_from_training",
                "reason": "exact_signed_16_adc_rail_hit",
                "positive_32767_samples": positive, "negative_32768_samples": negative,
                "wfdb_header_sha256": sha256(header_path),
            })
    summary = {
        "strict_view_rows": sum(r["view"] == "strict_10s" for r in overlay),
        "center_crop_view_rows": sum(r["view"] == "cpsc_ssl_center_crop" for r in overlay),
        "unique_ecg_ids": len({r["ecg_id"] for r in overlay}),
        "positive_32767_samples_across_view_rows": sum(r["positive_32767_samples"] for r in overlay),
        "negative_32768_samples_across_view_rows": sum(r["negative_32768_samples"] for r in overlay),
        "rule": "exact physical value +32.767/-32.768 mV at a 16-bit ADC limit, verified from WFDB header gain=1000/mV, baseline=0",
        "scope": "recommended downstream training exclusion only; published manifests and shards are unchanged",
        "other_qc_flags": "review only; no automatic exclusion from this overlay",
    }
    return overlay, summary


def curated_ssl_rows(strict_dir: Path, crop_dir: Path,
                     strict_rows: list[dict[str, str]], crop_rows: list[dict[str, str]],
                     rail_overlay: list[dict]) -> tuple[list[dict], list[dict], dict]:
    """Strict-first SSL-only candidate index into existing immutable shards."""
    overlay_keys = {(r["view"], r["ecg_id"]) for r in rail_overlay}
    if len(overlay_keys) != len(rail_overlay):
        raise ValueError("Duplicate row in rail exclusion overlay")
    rail_ids = {r["ecg_id"] for r in rail_overlay}
    shard_counts = {
        "strict_10s": {part["file"]: part["records"] for part in
                       json.loads((strict_dir / "metadata.json").read_text())["shards"]},
        "cpsc_ssl_center_crop": {part["file"]: part["records"] for part in
                                 json.loads((crop_dir / "metadata.json").read_text())["shards"]},
    }
    all_rows = [("strict_10s", strict_dir, row) for row in strict_rows]
    all_rows.extend(("cpsc_ssl_center_crop", crop_dir, row) for row in crop_rows)
    by_key = {(view, row["ecg_id"]): row for view, _, row in all_rows}
    for excluded in rail_overlay:
        source = by_key.get((excluded["view"], excluded["ecg_id"]))
        if source is None or source["signal_sha256"] != excluded["signal_sha256"]:
            raise ValueError("Rail overlay row does not match a published view")
    selected, omitted = [], []
    seen_signals, seen_ids = set(), set()
    for view, directory, row in all_rows:
        ecg_id, digest = row["ecg_id"], row["signal_sha256"]
        if ecg_id in rail_ids:
            reason = "verified_signed_16_rail"
        elif digest in seen_signals:
            reason = "strict_first_exact_signal_overlap"
        elif ecg_id in seen_ids:
            reason = "strict_first_record_overlap"
        else:
            reason = ""
        if reason:
            omitted.append({"view": view, "ecg_id": ecg_id, "signal_sha256": digest,
                            "reason": reason})
            continue
        shard = (directory / row["shard"]).resolve()
        if not shard.is_file() or not shard.is_relative_to(ROOT):
            raise ValueError(f"Curated shard is missing or outside project: {shard}")
        if (row["shard"] not in shard_counts[view] or
                not 0 <= int(row["shard_index"]) < shard_counts[view][row["shard"]]):
            raise ValueError(f"Curated shard index outside published receipt: {ecg_id}")
        selected.append({
            "ecg_id": ecg_id, "source": row["source"], "view": view,
            "shard_path": str(shard.relative_to(ROOT)), "shard_index": row["shard_index"],
            "signal_sha256": digest, "source_samples": row["source_samples"],
            "window_start": row["window_start"], "window_samples": row["window_samples"],
            "sampling_rate_hz": row["sampling_rate_hz"], "units": row["units"],
            "lead_order": row["lead_order"], "qc_flags": row["qc_flags"],
            "patient_id": row["patient_id"],
            "patient_identity_known": "false", "patient_independent_eval_eligible": "false",
            "label_scope": "ssl_only", "endpoint_supervised_eligible": "false",
        })
        seen_signals.add(digest)
        seen_ids.add(ecg_id)
    if len(selected) != len(seen_signals) or len(selected) != len(seen_ids):
        raise ValueError("Curated SSL IDs or waveforms are not unique")
    if len(selected) + len(omitted) != len(all_rows):
        raise ValueError("Curated SSL accounting mismatch")
    if any(r["ecg_id"] in rail_ids for r in selected):
        raise ValueError("Rail-hit ECG leaked into curated SSL candidate")
    if any(r["label_scope"] != "ssl_only" or r["endpoint_supervised_eligible"] != "false"
           for r in selected):
        raise ValueError("Curated SSL candidate has an endpoint label")
    counts = {
        "strict_input_rows": len(strict_rows), "centered_input_rows": len(crop_rows),
        "selected_total": len(selected),
        "selected_strict": sum(r["view"] == "strict_10s" for r in selected),
        "selected_centered": sum(r["view"] == "cpsc_ssl_center_crop" for r in selected),
        "omitted_total": len(omitted),
        "omitted_by_reason": dict(sorted(Counter(r["reason"] for r in omitted).items())),
        "rail_unique_ecg_ids": len(rail_ids),
        "duplicate_signal_count_after_curation": 0,
        "duplicate_ecg_id_count_after_curation": 0,
    }
    return selected, omitted, counts


def exclusions_by_source(metadata: dict, directory: Path) -> dict[str, dict[str, int]]:
    prepared = Path(metadata["input_prepared_dir"])
    if sha256(prepared / "metadata.json") != metadata["input_metadata_sha256"]:
        raise ValueError("Challenge audit metadata changed since materialization")
    excluded = rows(prepared / "exclusions.csv")
    if sha256(prepared / "exclusions.csv") != metadata["input_exclusions_sha256"]:
        raise ValueError("Challenge exclusions changed since materialization")
    if len(excluded) != metadata["excluded_records"]:
        raise ValueError("Challenge exclusion count mismatch")
    grouped: dict[str, Counter] = defaultdict(Counter)
    for item in excluded:
        grouped[item["source"]][item["reason"]] += 1
    return {source: dict(sorted(counts.items())) for source, counts in sorted(grouped.items())}


def code_view(directory: Path, sample_size: int, seed: int) -> tuple[dict, dict]:
    metadata = json.loads((directory / "metadata.json").read_text())
    if metadata.get("limit_per_archive") is not None or metadata.get("canonical_500hz_10s_eligible") is not False:
        raise ValueError("CODE output is a limited or unexpectedly canonicalized view")
    for name in ("manifest.csv", "exclusions.csv", "demographics.csv", "source_labels.csv"):
        if sha256(directory / name) != metadata["output_sha256"][name]:
            raise ValueError(f"CODE output checksum mismatch: {name}")
    manifest = rows(directory / "manifest.csv")
    exclusions = rows(directory / "exclusions.csv")
    demographics = rows(directory / "demographics.csv")
    labels = rows(directory / "source_labels.csv")
    if len(manifest) != metadata["counts"]["accepted"] or len(exclusions) + len(manifest) != metadata["counts"]["audited"]:
        raise ValueError("CODE accepted and excluded rows do not match receipt")
    if len(demographics) != len(manifest) or len(labels) != len(manifest):
        raise ValueError("CODE linked tables do not match manifest length")
    if sum(bool(r["qc_flags"]) for r in manifest) != metadata["counts"]["review_flagged"]:
        raise ValueError("CODE review-flag count mismatch")
    identifiers = [r["exam_id"] for r in manifest]
    if len(set(identifiers)) != len(identifiers) or identifiers != [r["exam_id"] for r in demographics] or identifiers != [r["exam_id"] for r in labels]:
        raise ValueError("CODE exam IDs are duplicated or linked tables are misaligned")
    age = [float(r["age"]) for r in demographics if r["age"] and 0 <= float(r["age"]) <= 120]
    patient_counts = Counter(r["patient_id"] for r in manifest)
    edge_counts = Counter((int(r["edge_zero_left"]), int(r["edge_zero_right"])) for r in manifest)
    hdf5_file = directory / "exams_part0_native.hdf5"
    sampled_max_abs, sampled_min_std = [], []
    with h5py.File(hdf5_file, "r") as handle:
        signals, ids = handle["tracings"], handle["exam_id"]
        if len(signals) != len(manifest) or signals.shape[1:] != (4096, 12) or len(ids) != len(signals):
            raise ValueError("CODE native HDF5 shape/count mismatch")
        for index in deterministic_indices(len(manifest), sample_size, seed):
            i = int(index)
            trace = np.asarray(signals[i])
            if str(int(ids[i])) != manifest[i]["exam_id"]:
                raise ValueError("CODE native HDF5 exam ID mismatch")
            digest = hashlib.sha256(np.ascontiguousarray(trace, dtype="<f4").tobytes()).hexdigest()
            if digest != manifest[i]["signal_sha256"] or not np.isfinite(trace).all():
                raise ValueError(f"CODE sampled waveform hash mismatch: {manifest[i]['exam_id']}")
            sampled_max_abs.append(float(np.max(np.abs(trace))))
            sampled_min_std.append(float(np.std(trace, axis=0).min()))
    diagnosis = {field: sum(r[field] == "True" for r in labels) for field in
                 ("1dAVb", "RBBB", "LBBB", "SB", "ST", "AF", "normal_ecg")}
    summary = {
        "directory": str(directory.resolve().relative_to(ROOT)),
        "manifest_sha256": metadata["output_sha256"]["manifest.csv"],
        "audited_records": metadata["counts"]["audited"],
        "accepted_records": len(manifest), "excluded_records": len(exclusions),
        "exclusions": dict(sorted(Counter(r["reason"] for r in exclusions).items())),
        "distinct_patients": len(patient_counts),
        "patients_with_multiple_exams": sum(n > 1 for n in patient_counts.values()),
        "max_exams_per_patient": max(patient_counts.values()),
        "age_years": quantiles(age), "age_known": len(age),
        "age_18_to_30": sum(18 <= value <= 30 for value in age),
        "sex_values": dict(sorted(Counter(r["is_male"] for r in demographics).items())),
        "released_flag_positive_counts": diagnosis,
        "qc_flags": flag_counts(manifest),
        "edge_zero_top12": [[list(pair), count] for pair, count in edge_counts.most_common(12)],
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


def make_figure(path: Path, strict: dict, crop: dict, code: dict, code_figure: dict) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    names = ["Georgia", "CPSC 2018", "CPSC Extra"]
    keys = ["georgia", "cpsc_2018", "cpsc_2018_extra"]
    accepted = [strict["source_counts"][key]["accepted"] for key in keys]
    exclusions = strict["exclusions_by_source"]
    duration = [exclusions.get(key, {}).get("duration_contract", 0) for key in keys]
    other = [sum(exclusions.get(key, {}).values()) - value for key, value in zip(keys, duration)]
    ax = axes[0, 0]
    ax.bar(names, accepted, label="Accepted", color="#2664a5")
    ax.bar(names, duration, bottom=accepted, label="Duration excluded", color="#d3a649")
    ax.bar(names, other, bottom=np.asarray(accepted) + duration, label="Other excluded", color="#be6b62")
    ax.set_title("Strict 10 s audit outcomes")
    ax.set_ylabel("Recordings")
    ax.legend(fontsize=8)
    ax.tick_params(axis="x", rotation=12)

    ax = axes[0, 1]
    for name, summary, color in (("Strict 10 s", strict, "#2664a5"),
                                 ("CPSC centered", crop, "#b86b32")):
        # Source medians are plotted rather than pooling groups with different sizes.
        x = [summary["source_counts"][key]["max_abs_mv"]["median"]
             for key in keys if key in summary["source_counts"]]
        labels = [key.replace("_", " ") for key in keys if key in summary["source_counts"]]
        ax.scatter(labels, x, label=name, color=color, s=55)
    ax.set_title("Median peak absolute amplitude by source")
    ax.set_ylabel("mV; full accepted manifests")
    ax.tick_params(axis="x", rotation=12)
    ax.legend(fontsize=8)

    ax = axes[1, 0]
    ax.hist(code_figure["ages"], bins=np.arange(0, 111, 5), color="#487b74", edgecolor="white")
    ax.set_title("CODE part 0 accepted ages")
    ax.set_xlabel("Years")
    ax.set_ylabel("Exams")

    ax = axes[1, 1]
    pair_counts = [("0 / 0", code["edge_zero_dominant_0_0"]),
                   ("581 / 581", code["edge_zero_dominant_581_581"]),
                   ("Other", code["edge_zero_other"])]
    ax.bar([item[0] for item in pair_counts], [item[1] for item in pair_counts],
           color=["#487b74", "#97ada0", "#d3a649"])
    ax.set_title("CODE exact-zero edge runs")
    ax.set_xlabel("Left / right samples; not proven duration")
    ax.set_ylabel("Exams")
    fig.suptitle("Processed ECG exploratory analysis; no clinical quality or endpoint labels inferred", fontsize=12)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def make_review_figure(path: Path, directory: Path, data: list[dict[str, str]]) -> None:
    """Show representative flagged leads for human inspection, without adjudication."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    high = sorted((r for r in data if "amplitude_over_10mV_review" in r["qc_flags"]),
                  key=lambda r: -float(r["max_abs_mv"]))
    selected_high = []
    for row in high:
        if row["source"] not in {r["source"] for r in selected_high}:
            selected_high.append(row)
        if len(selected_high) == 2:
            break
    flat = [r for r in data if "near_flat_lead_review" in r["qc_flags"]]
    selected_flat = []
    for row in flat:
        if row["source"] not in {r["source"] for r in selected_flat}:
            selected_flat.append(row)
        if len(selected_flat) == 2:
            break
    selected = [(row, "peak") for row in selected_high] + [(row, "flat") for row in selected_flat]
    if len(selected) != 4:
        raise ValueError("Expected two high-amplitude and two near-flat review examples")
    lead_names = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]
    fig, axes = plt.subplots(2, 2, figsize=(13, 6), constrained_layout=True)
    time = np.arange(5000) / 500
    for ax, (row, kind) in zip(axes.flat, selected):
        signal = np.load(directory / row["shard"], mmap_mode="r", allow_pickle=False)[int(row["shard_index"])]
        if kind == "peak":
            lead = int(np.unravel_index(np.argmax(np.abs(signal)), signal.shape)[0])
            ax.plot(time, signal[lead], lw=.8, color="#a24b43", label=lead_names[lead])
        else:
            lead = int(signal.std(axis=1).argmin())
            ax.plot(time, signal[lead], lw=.8, color="#a24b43", label=f"{lead_names[lead]} near-flat")
            ax.plot(time, signal[1], lw=.6, alpha=.6, color="#2664a5", label="II context")
        ax.set_title(f"{row['ecg_id']} — {kind} review")
        ax.set_xlabel("Seconds")
        ax.set_ylabel("mV")
        ax.legend(fontsize=8, loc="upper right")
    fig.suptitle("Selected review flags; displayed traces are not clinical diagnoses")
    fig.savefig(path, dpi=160)
    plt.close(fig)


def run(strict_dir: Path, crop_dir: Path, code_dir: Path, output_dir: Path,
        sample_size: int = 128, seed: int = 42) -> dict:
    if sample_size < 1:
        raise ValueError("sample_size must be positive")
    if output_dir.resolve().is_relative_to((ROOT / "data/raw").resolve()):
        raise ValueError("EDA output cannot be inside data/raw")
    strict, strict_rows = challenge_view(strict_dir, sample_size, seed)
    crop, crop_rows = challenge_view(crop_dir, sample_size, seed + 1)
    if strict["policy"] != "strict_10s" or crop["policy"] != "ssl_center_crop":
        raise ValueError("Unexpected Challenge view policies")
    strict["exclusions_by_source"] = exclusions_by_source(
        json.loads((strict_dir / "metadata.json").read_text()), strict_dir)
    crop["exclusions_by_source"] = exclusions_by_source(
        json.loads((crop_dir / "metadata.json").read_text()), crop_dir)
    for directory, summary in ((strict_dir, strict), (crop_dir, crop)):
        metadata = json.loads((directory / "metadata.json").read_text())
        for source, provenance in metadata["official_provenance"].items():
            accepted = summary["source_counts"].get(source, {}).get("accepted", 0)
            excluded = sum(summary["exclusions_by_source"].get(source, {}).values())
            if accepted + excluded != provenance["candidate_records"]:
                raise ValueError(f"Challenge source candidate count mismatch: {source}")
    flagged_strict = review_flagged_waveforms(strict_dir, strict_rows)
    flagged_crop = review_flagged_waveforms(crop_dir, crop_rows)
    rail_overlay, rail_summary = rail_exclusion_overlay(
        strict_rows, crop_rows, flagged_strict, flagged_crop)
    curated, curated_omitted, curated_counts = curated_ssl_rows(
        strict_dir, crop_dir, strict_rows, crop_rows, rail_overlay)
    strict["flagged_waveforms_inspected"] = len(flagged_strict)
    strict["rail_value_examples"] = [r for r in flagged_strict
                                     if r["samples_at_positive_32_767mv"] or r["samples_at_negative_32_768mv"]]
    crop["flagged_waveforms_inspected"] = len(flagged_crop)
    strict_ids = {r["ecg_id"] for r in strict_rows}
    crop_ids = {r["ecg_id"] for r in crop_rows}
    strict_hashes = {r["signal_sha256"] for r in strict_rows}
    crop_hashes = {r["signal_sha256"] for r in crop_rows}
    record_overlap = strict_ids & crop_ids
    signal_overlap = strict_hashes & crop_hashes
    if len(record_overlap) != strict["overlap_by_record_flagged"] or len(signal_overlap) != strict["overlap_by_signal_flagged"]:
        raise ValueError("Strict manifest overlap flags differ from recomputed overlap")
    if len(record_overlap) != crop["overlap_by_record_flagged"] or len(signal_overlap) != crop["overlap_by_signal_flagged"]:
        raise ValueError("Crop manifest overlap flags differ from recomputed overlap")
    code, code_figure = code_view(code_dir, sample_size, seed + 2)
    result = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "eda_source_sha256": sha256(Path(__file__)),
        "sampling": {"sample_size_per_view": sample_size, "seed": seed},
        "strict": strict, "center_crop": crop, "code_part0": code,
        "recommended_rail_exclusions": rail_summary,
        "curated_ssl_candidate": curated_counts,
        "challenge_cross_view": {
            "same_ecg_id": len(record_overlap), "same_canonical_signal": len(signal_overlap),
            "unique_ecg_ids_across_views": len(strict_ids | crop_ids),
            "unique_canonical_signals_across_views": len(strict_hashes | crop_hashes),
            "interpretation": "Two overlapping views, not independent cohorts; exact matching only."},
        "limitations": [
            "Challenge record IDs are not verified patient identities; patient-separated evaluation is unavailable.",
            "Challenge source diagnosis codes are unmapped to the PTB-XL endpoint.",
            "CODE native units and original duration remain unresolved; exact-zero edges are descriptive only.",
            "CODE part 0 is one of 18 archives, not the complete release.",
            "Waveform sample hashes validate sampled storage rows; Challenge full-manifest QC summaries originate from materialization."],
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    with (output_dir / "challenge_review_flags.csv").open("w", newline="", encoding="utf-8") as handle:
        fieldnames = ("view", "ecg_id", "source", "qc_flags", "peak_positive_mv",
                      "peak_negative_mv", "samples_abs_over_10mv",
                      "samples_at_positive_32_767mv", "samples_at_negative_32_768mv",
                      "minimum_lead_std_mv")
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows({"view": "strict_10s", **item} for item in flagged_strict)
        writer.writerows({"view": "cpsc_ssl_center_crop", **item} for item in flagged_crop)
    with (output_dir / "training_exclusion_overlay.csv").open("w", newline="", encoding="utf-8") as handle:
        fields = ("view", "ecg_id", "source", "signal_sha256", "decision", "reason",
                  "positive_32767_samples", "negative_32768_samples", "wfdb_header_sha256")
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rail_overlay)
    curated_path = output_dir / "challenge_ssl_curated_manifest.csv"
    with curated_path.open("w", newline="", encoding="utf-8") as handle:
        fields = ("ecg_id", "source", "view", "shard_path", "shard_index", "signal_sha256",
                  "source_samples", "window_start", "window_samples", "sampling_rate_hz",
                  "units", "lead_order", "qc_flags", "patient_id", "patient_identity_known",
                  "patient_independent_eval_eligible", "label_scope", "endpoint_supervised_eligible")
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(curated)
    omitted_path = output_dir / "challenge_ssl_curated_omissions.csv"
    with omitted_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=("view", "ecg_id", "signal_sha256", "reason"))
        writer.writeheader()
        writer.writerows(curated_omitted)
    curated_receipt = {
        "status": "candidate_ssl_only_not_scheduled",
        "source_sha256": sha256(Path(__file__)),
        "strict_manifest_sha256": strict["manifest_sha256"],
        "centered_manifest_sha256": crop["manifest_sha256"],
        "rail_overlay_sha256": sha256(output_dir / "training_exclusion_overlay.csv"),
        "curated_manifest_sha256": sha256(curated_path),
        "curated_omissions_sha256": sha256(omitted_path),
        "counts": curated_counts,
        "data_contract": "[12,5000] float32 physical mV, 500 Hz, original unfiltered materialized shards",
        "label_contract": "ssl_only; no endpoint labels and no patient-independent evaluation eligibility",
        "patient_identity": "Challenge patient IDs unavailable; patient_id is a record surrogate",
        "storage": "manifest references existing verified shard/index pairs; arrays are not copied or modified",
    }
    (output_dir / "challenge_ssl_curated_receipt.json").write_text(
        json.dumps(curated_receipt, indent=2, sort_keys=True) + "\n")
    make_figure(output_dir / "overview.png", strict, crop, code, code_figure)
    make_review_figure(output_dir / "review_examples.png", strict_dir, strict_rows)
    return result


def main() -> None:
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
