"""Publish a manifest-only clean SPH table: duplicates resolved, quality policy applied, labels mapped."""

from __future__ import annotations

import argparse
import json
from multiprocessing import Pool
from pathlib import Path

import pandas as pd

from ecg_experiment import sph
from ecg_experiment.ecg_quality import EXCLUSION_REASONS, REVIEW_FLAGS
from ecg_experiment.files import sha256_file, write_json_atomic

ROOT = Path(__file__).resolve().parents[2]
SOURCES = ("ecg_experiment/sph.py", "ecg_experiment/ecg_quality.py", "scripts/data/build_sph_clean.py")


def build(output: Path, workers: int) -> dict[str, object]:
    """Score every record, resolve duplicates and write ``rows.csv`` and ``metadata.json``."""
    if output.exists():
        raise ValueError(f"Destination exists: {output}")
    small_files = sph.verify_small_files()
    frame = sph.table()
    raw = pd.read_csv(sph.METADATA, dtype=str).set_index("ECG_ID")["AHA_Code"]
    with Pool(workers) as pool:
        scored = pool.map(sph.assess_window, list(frame.index), chunksize=64)
    quality = pd.DataFrame(scored).set_index("ecg_id")

    rows = frame.join(raw.rename("aha_code")).join(quality)
    rows["exclusion_reasons"] = rows[list(EXCLUSION_REASONS)].apply(
        lambda row: ";".join(name for name in EXCLUSION_REASONS if row[name]), axis=1)
    rows["review_flags"] = rows[list(REVIEW_FLAGS)].apply(
        lambda row: ";".join(name for name in REVIEW_FLAGS if row[name]), axis=1)
    rows["duplicate_status"] = sph.duplicate_status(rows["signal_sha256"], rows["aha_code"])
    rows["use_evaluation"] = rows["duplicate_status"].isin(["unique", "kept"])
    rows["use_training"] = rows["use_evaluation"] & (rows["exclusion_reasons"] == "")
    rows = rows.drop(columns=[*EXCLUSION_REASONS, *REVIEW_FLAGS, "samples"])

    stage = output.with_name(output.name + ".partial")
    stage.mkdir(parents=True)
    rows.to_csv(stage / "rows.csv")
    metadata = {
        "complete": True, "records": len(rows), "patients": int(rows["patient_id"].nunique()),
        "window": "first 10 s (5,000 samples at 500 Hz), mV, canonical lead order",
        "duplicate_status": rows["duplicate_status"].value_counts().to_dict(),
        "exclusion_reason_counts": rows["exclusion_reasons"].str.split(";").explode()
        .loc[lambda values: values != ""].value_counts().to_dict(),
        "evaluation": {"records": int(rows["use_evaluation"].sum()),
                       "primary": rows.loc[rows["use_evaluation"], "primary"].value_counts(dropna=False)
                       .rename(str).to_dict()},
        "training": {"records": int(rows["use_training"].sum()),
                     "primary": rows.loc[rows["use_training"], "primary"].value_counts(dropna=False)
                     .rename(str).to_dict()},
        "small_files_md5": small_files, "records_sha256": sph.records_sha256(list(frame.index)),
        "rows_sha256": sha256_file(stage / "rows.csv"),
        "source_sha256": {name: sha256_file(ROOT / name) for name in SOURCES},
    }
    write_json_atomic(stage / "metadata.json", metadata, sort_keys=True)
    stage.rename(output)
    return metadata


def main() -> None:
    """Parse arguments and build the manifest."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "data/processed/sph_clean_v1")
    parser.add_argument("--workers", type=int, default=6)
    args = parser.parse_args()
    metadata = build(args.output_dir, args.workers)
    summary = ("records", "duplicate_status", "evaluation", "training")
    print(json.dumps({key: metadata[key] for key in summary}, indent=2))


if __name__ == "__main__":
    main()
