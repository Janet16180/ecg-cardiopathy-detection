"""Publish a manifest-only clean CODE-15% table: padding trimmed, 10 s kept, scored and deduplicated.

The waveforms are streamed from the export archive one HDF5 part at a time, after checking the archive
against ``CODE_SHA256SUMS``; each extracted part must hold the number of tracings recorded by the all-parts
audit, and is deleted before the next. Per-part scores are
cached under ``outputs/data_quality/code15_clean_v1/`` so an interrupted run resumes.
"""

from __future__ import annotations

import argparse
import json
from multiprocessing import Pool
from pathlib import Path

import h5py
import pandas as pd

from ecg_experiment import code15_clean
from ecg_experiment.eda import code15
from ecg_experiment.files import sha256_file, write_json_atomic

ROOT = Path(__file__).resolve().parents[2]
AUDIT = ROOT / "data/processed/code15_quality/all_parts_manifest_v1/metadata.json"
ARCHIVE_SUMS = ROOT / "outputs/data_export/CODE_SHA256SUMS"
SOURCES = ("ecg_experiment/code15_clean.py", "ecg_experiment/eda/code15.py", "ecg_experiment/ecg_quality.py",
           "scripts/data/build_code15_clean.py")
CHUNK = 500


def score_parts(cache: Path, workers: int) -> pd.DataFrame:
    """
    Score every stored tracing of all 18 parts, reusing cached parts.

    Parameters
    ----------
    cache : Path
        Directory receiving one ``part{N}.parquet`` per part.
    workers : int
        Number of scoring processes.

    Returns
    -------
    pd.DataFrame
        One row per stored tracing.

    Raises
    ------
    ValueError
        If the archive differs from its checksum or a part from the audited tracing count.
    """
    audited = {item["member"]: item["hdf5_records"] for item in json.loads(AUDIT.read_text())["archives"]}
    cache.mkdir(parents=True, exist_ok=True)
    parts = []
    if len(list(cache.glob("part*.parquet"))) < len(audited):
        expected = ARCHIVE_SUMS.read_text().split()[0]
        if sha256_file(code15.EXPORT_ARCHIVE) != expected:
            raise ValueError("CODE-15 export archive differs from CODE_SHA256SUMS")
        parts = code15.extracted_parts()
    for part, path in parts:
        target = cache / f"part{part}.parquet"
        if target.exists():
            continue
        with h5py.File(path, "r") as handle:
            count = len(handle["exam_id"])
        if count != audited[path.name]:
            raise ValueError(f"Extracted part differs from the audit: {path.name}")
        items = [(str(path), part, start, min(start + CHUNK, count)) for start in range(0, count, CHUNK)]
        with Pool(workers) as pool:
            rows = [row for chunk in pool.map(code15_clean.score_chunk, items) for row in chunk]
        pd.DataFrame(rows).to_parquet(target)
        print(f"CODE-15 part {part}: {count} tracings scored", flush=True)
    tables = [pd.read_parquet(path) for path in sorted(cache.glob("part*.parquet"))]
    return pd.concat(tables, ignore_index=True)


def clean_rows(scores: pd.DataFrame) -> pd.DataFrame:
    """
    Join the exam table, resolve duplicates and decide training use.

    Parameters
    ----------
    scores : pd.DataFrame
        Output of ``score_parts``.

    Returns
    -------
    pd.DataFrame
        One row per exam of ``exams.csv``, indexed by ``exam_id``.

    Raises
    ------
    ValueError
        If an exam of ``exams.csv`` has no tracing or more than one.
    """
    exams = code15.load_exams()
    stored = scores[scores["exam_id"].isin(exams.index)].set_index("exam_id")
    if stored.index.duplicated().any() or len(stored) != len(exams):
        raise ValueError("exams.csv and the stored tracings do not match one to one")
    rows = exams[["patient_id", "age", "is_male"]].join(stored.drop(columns=[
        column for column in stored if column.startswith("range_")]))
    rows["hdf5_member"] = "exams_part" + rows["part"].astype(str) + ".hdf5"
    rows["ten_seconds"] = rows["active_samples"] >= code15_clean.TEN_SECONDS
    rows["exclusion_reasons"] = code15_clean.exclusion_reasons(rows)
    rows["review_flags"] = code15_clean.review_flags(rows)
    signal = rows[rows["active_samples"] > 0]
    rows["duplicate_status"] = code15_clean.duplicate_status(signal).reindex(rows.index).fillna("no_signal")
    rows["use_training"] = (rows["ten_seconds"] & (rows["exclusion_reasons"] == "")
                            & rows["duplicate_status"].isin(["unique", "kept"]))
    rows["window_start"] = rows["edge_zero_left"]
    return rows


def build(output: Path, cache: Path, workers: int) -> dict[str, object]:
    """Score all parts, write ``rows.csv`` and ``metadata.json`` and return the metadata."""
    if output.exists():
        raise ValueError(f"Destination exists: {output}")
    scores = score_parts(cache, workers)
    rows = clean_rows(scores)
    ranges = scores.loc[scores["exam_id"].isin(rows.index[rows["ten_seconds"]]),
                        ["exam_id", *[column for column in scores if column.startswith("range_")]]]
    ranges.to_parquet(cache / "amplitude_ranges.parquet")

    stage = output.with_name(output.name + ".partial")
    stage.mkdir(parents=True)
    columns = ["patient_id", "age", "is_male", "part", "hdf5_member", "storage_index", "edge_zero_left",
               "edge_zero_right", "active_samples", "ten_seconds", "window_start", "native_sha256",
               "window_sha256", "exclusion_stored", "exclusion_halved", "exclusion_reasons", "review_flags",
               "duplicate_status", "use_training"]
    rows[columns].to_csv(stage / "rows.csv")
    ten = rows[rows["ten_seconds"]]
    usable = rows[rows["use_training"]]
    metadata = {
        "complete": True, "exams": len(rows), "patients": int(rows["patient_id"].nunique()),
        "stored_tracings": len(scores),
        "filler_rows_dropped": int((~scores["exam_id"].isin(rows.index)).sum()),
        "window": "first 10 s of the observed span, resampled 400 to 500 Hz (polyphase 5/4 over the "
                  "whole span), canonical lead order, stored units",
        "amplitude_unit": "unresolved; the policy is applied at 1.0 and 0.5 times the stored values and an "
                          "exam must pass at both",
        "active_samples": {"zero": int((rows["active_samples"] == 0).sum()),
                           "under_10s": int(((rows["active_samples"] > 0) & ~rows["ten_seconds"]).sum()),
                           "ten_seconds_or_more": int(rows["ten_seconds"].sum()),
                           "exactly_4096": int((rows["active_samples"] == 4096).sum())},
        "ten_second_exams": {"exams": len(ten), "patients": int(ten["patient_id"].nunique())},
        "exclusion_reason_counts": {name: ten[column].str.split(";").explode().loc[lambda v: v != ""]
                                    .value_counts().to_dict()
                                    for name, column in (("stored", "exclusion_stored"),
                                                         ("halved", "exclusion_halved"),
                                                         ("either", "exclusion_reasons"))},
        "excluded_ten_second_exams": int((ten["exclusion_reasons"] != "").sum()),
        "review_flag_counts": ten["review_flags"].str.split(";").explode().loc[lambda v: v != ""]
        .value_counts().to_dict(),
        "duplicate_status": rows["duplicate_status"].value_counts().to_dict(),
        "duplicate_status_ten_seconds": ten["duplicate_status"].value_counts().to_dict(),
        "training": {"exams": len(usable), "patients": int(usable["patient_id"].nunique())},
        "audit_metadata_sha256": sha256_file(AUDIT),
        "exams_csv_sha256": sha256_file(code15.RAW_DIR / "exams.csv"),
        "archive": str(code15.EXPORT_ARCHIVE.relative_to(ROOT)),
        "archive_sha256": ARCHIVE_SUMS.read_text().split()[0],
        "rows_sha256": sha256_file(stage / "rows.csv"),
        "amplitude_ranges_sha256": sha256_file(cache / "amplitude_ranges.parquet"),
        "source_sha256": {name: sha256_file(ROOT / name) for name in SOURCES},
    }
    write_json_atomic(stage / "metadata.json", metadata, sort_keys=True)
    stage.rename(output)
    return metadata


def main() -> None:
    """Parse arguments and build the manifest."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "data/processed/code15_clean_v1")
    parser.add_argument("--cache-dir", type=Path, default=ROOT / "outputs/data_quality/code15_clean_v1")
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    metadata = build(args.output_dir, args.cache_dir, args.workers)
    summary = ("exams", "active_samples", "ten_second_exams", "excluded_ten_second_exams", "duplicate_status",
               "training")
    print(json.dumps({key: metadata[key] for key in summary}, indent=2))


if __name__ == "__main__":
    main()
