"""Summarize CODE-15 release annotations without copying waveforms."""

from __future__ import annotations

import hashlib
import json
import os

import pandas as pd

from . import ROOT
from .files import sha256_file, write_json_atomic

LABELS = ("1dAVb", "RBBB", "LBBB", "SB", "ST", "AF")
SOURCE = ROOT / "data/raw/code-15pct/zenodo-4916206/exams.csv"
RECEIPT = ROOT / "data/acquisition/code_15pct.json"
OUTPUT = ROOT / "outputs/code15_label_groups"


def run() -> dict[str, object]:
    """Export source-positive IDs and counts for six released diagnoses."""
    receipt = json.loads(RECEIPT.read_text())
    expected = next(item["checksum"].removeprefix("md5:") for item in receipt["verified_files"]
                    if item["name"] == "exams.csv")
    with SOURCE.open("rb") as handle:
        actual = hashlib.file_digest(handle, "md5").hexdigest()  # noqa: S324 - upstream checksum
    if actual != expected:
        raise RuntimeError("CODE-15 metadata differs from the verified download")

    fields = ["exam_id", "patient_id", "trace_file", "normal_ecg", *LABELS]
    frame = pd.read_csv(SOURCE, usecols=fields)
    if frame["exam_id"].isna().any() or frame["exam_id"].duplicated().any():
        raise RuntimeError("CODE-15 exam IDs are missing or duplicated")
    if frame["patient_id"].isna().any() or frame[list(LABELS)].isna().any().any():
        raise RuntimeError("CODE-15 patient IDs or diagnosis flags are missing")
    positives = frame.loc[frame[list(LABELS)].any(axis=1)].copy()
    positives["source_labels"] = positives[list(LABELS)].apply(
        lambda row: ";".join(label for label in LABELS if row[label]), axis=1
    )
    OUTPUT.mkdir(parents=True, exist_ok=True)
    target = OUTPUT / "source_positive_ids.csv.gz"
    temporary = target.with_name(target.name + ".tmp")
    positives[["exam_id", "patient_id", "trace_file", "source_labels"]].to_csv(
        temporary, index=False, compression="gzip"
    )
    os.replace(temporary, target)
    report: dict[str, object] = {
        "source": "CODE-15% exams.csv release metadata, all 18 parts",
        "exams": len(frame),
        "patients": int(frame["patient_id"].nunique()),
        "source_positive_exams_any_six": len(positives),
        "source_positive_patients_any_six": int(positives["patient_id"].nunique()),
        "source_label_counts": {label: int(frame[label].sum()) for label in LABELS},
        "source_normal_ecg_count": int(frame["normal_ecg"].sum()),
        "multi_label_exams": int((frame[list(LABELS)].sum(axis=1) > 1).sum()),
        "metadata_md5": actual,
        "metadata_sha256": sha256_file(SOURCE),
        "identified_source_positives": str(target.relative_to(ROOT)),
        "identified_source_positives_sha256": sha256_file(target),
    }
    write_json_atomic(OUTPUT / "report.json", report)
    return report
