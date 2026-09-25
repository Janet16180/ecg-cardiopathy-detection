#!/usr/bin/env python3
"""Read-only, reproducible inventory for the public ECG data quality review."""

from __future__ import annotations

import csv
import json
import random
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import wfdb

from ecg_experiment.provenance import utc_now

ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "outputs/data_quality/local_audit.json"
LEADS = {"I", "II", "III", "AVR", "AVL", "AVF", "V1", "V2", "V3", "V4", "V5", "V6"}
SEED = 42
SAMPLE_SIZE = 64
MAX_PLAUSIBLE_AGE = 120
NEAR_FLAT_STD_MV = 0.01
TOP_DIAGNOSIS_CODES = 12
HEADER_SOURCES = {
    "georgia": ROOT / "data/raw/challenge-2020/1.0.2/training/georgia",
    "cpsc_2018": ROOT / "data/raw/challenge-2020/1.0.2/training/cpsc_2018",
    "cpsc_2018_extra": ROOT / "data/raw/challenge-2020/1.0.2/training/cpsc_2018_extra",
    "chapman_shaoxing": ROOT / "data/raw/challenge-2021/1.0.3/training/chapman_shaoxing",
}


def age_summary(values: list[float]) -> dict[str, Any]:
    """
    Summarize known ages.

    Parameters
    ----------
    values : list[float]
        Plausible ages in years.

    Returns
    -------
    dict[str, Any]
        Count, median and percentage aged 18 to 30; ``None`` when empty.
    """
    return {"known": len(values), "median": float(np.median(values)) if values else None,
            "age_18_to_30_percent": round(100 * sum(18 <= a <= 30 for a in values) / len(values), 2)
            if values else None}


def optional_age(value: str) -> float | None:
    """
    Parse an age, discarding missing or implausible values.

    Parameters
    ----------
    value : str
        Raw age text.

    Returns
    -------
    float | None
        Age in years within [0, 120], otherwise ``None``.
    """
    try:
        age = float(value)
    except (ValueError, TypeError):
        return None
    return age if np.isfinite(age) and 0 <= age <= MAX_PLAUSIBLE_AGE else None


def csv_summary(path: Path, patient_field: str, age_field: str,
                extra_fields: tuple[str, ...]) -> dict[str, Any]:
    """
    Summarize a metadata table's records, patients, ages and selected fields.

    Parameters
    ----------
    path : Path
        CSV metadata table.
    patient_field : str
        Column with patient identifiers.
    age_field : str
        Column with ages.
    extra_fields : tuple[str, ...]
        Columns whose nonempty counts and True/False values are reported.

    Returns
    -------
    dict[str, Any]
        Summary counts.
    """
    count, patients, ages = 0, set(), []
    present = Counter()
    values = Counter()
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            count += 1
            if row.get(patient_field):
                patients.add(row[patient_field])
            age = optional_age(row.get(age_field, ""))
            if age is not None:
                ages.append(age)
            for field in extra_fields:
                value = row.get(field, "").strip()
                present[field] += bool(value)
                values[(field, value)] += bool(value)
    return {"records": count, "unique_patient_ids": len(patients),
            "ages": age_summary(ages), "nonempty_fields": dict(present),
            "selected_values": {f: {v: n for (field, v), n in values.items()
                                     if field == f and v in ("True", "False")}
                                for f in extra_fields}}


def _count_header(path: Path, counts: Counter, diagnoses: Counter, ages: list[float]) -> None:
    """Add one WFDB header's contract checks and comments to the running tallies."""
    lines = path.read_text(encoding="utf-8").splitlines()
    fields = lines[0].split()
    leads, fs, samples = int(fields[1]), int(float(fields[2])), int(fields[3])
    duration = samples / fs
    names = {line.split()[-1].upper() for line in lines[1:1 + leads]}
    units = ["/mV" in line.split()[2] for line in lines[1:1 + leads]]
    comments = {line.split(":", 1)[0].strip("# "): line.split(":", 1)[1].strip()
                for line in lines if line.startswith("#") and ":" in line}
    counts["parsed_headers"] += 1
    counts["12_leads"] += leads == 12 and names == LEADS
    counts["500_hz"] += fs == 500
    counts["mV_units"] += len(units) == 12 and all(units)
    counts["duration_under_10_s"] += duration < 10
    counts["duration_exactly_10_s"] += duration == 10
    counts["duration_over_10_s"] += duration > 10
    counts["label_present"] += bool(comments.get("Dx") and comments["Dx"] != "Unknown")
    for code in comments.get("Dx", "").split(","):
        if code.strip() and code.strip() != "Unknown":
            diagnoses[code.strip()] += 1
    counts["sex_present"] += comments.get("Sex") in ("Male", "Female")
    age = optional_age(comments.get("Age", ""))
    if age is not None:
        ages.append(age)


def header_summary(root: Path, require_pair: bool = True) -> tuple[dict[str, Any], list[Path]]:
    """
    Summarize every WFDB header below a directory.

    Parameters
    ----------
    root : Path
        Directory searched recursively for ``.hea`` files.
    require_pair : bool
        Return only headers whose ``.mat`` waveform exists.

    Returns
    -------
    tuple[dict[str, Any], list[Path]]
        Summary counts, and headers available for waveform sampling.
    """
    headers = sorted(root.rglob("*.hea"))
    counts = Counter()
    diagnoses = Counter()
    ages = []
    available = []
    for path in headers:
        # A malformed header can fail partway; its partial tallies are kept, as before.
        try:
            _count_header(path, counts, diagnoses, ages)
        except (IndexError, ValueError, OSError):
            counts["malformed_headers"] += 1
            continue
        if not require_pair or path.with_suffix(".mat").is_file():
            available.append(path)
    counts["headers_found"] = len(headers)
    counts["complete_pairs"] = len(available)
    return {"counts": dict(counts), "ages": age_summary(ages),
            "distinct_diagnosis_codes": len(diagnoses),
            "top_diagnosis_codes": diagnoses.most_common(TOP_DIAGNOSIS_CODES)}, available


def _count_waveform(path: Path, counts: Counter, amplitudes: list[float]) -> None:
    """Add one decoded waveform's finiteness, flatness and amplitude to the tallies."""
    signal, _ = wfdb.rdsamp(str(path.with_suffix("")))
    counts["decoded"] += 1
    finite = bool(np.isfinite(signal).all())
    counts["finite"] += finite
    if finite:
        std = signal.std(axis=0)
        counts["any_near_flat_lead"] += bool(np.any(std < NEAR_FLAT_STD_MV))
        counts["all_near_flat_leads"] += bool(np.all(std < NEAR_FLAT_STD_MV))
        amplitudes.append(float(np.max(np.abs(signal))))
    counts["12_leads"] += signal.shape[1] == 12


def waveform_sample(paths: list[Path]) -> dict[str, Any]:
    """
    Decode a fixed random sample of records and screen their signals.

    Parameters
    ----------
    paths : list[Path]
        Candidate header paths.

    Returns
    -------
    dict[str, Any]
        Sample counts and amplitude percentiles.
    """
    rng = random.Random(SEED)
    selected = rng.sample(paths, min(SAMPLE_SIZE, len(paths)))
    counts = Counter()
    amplitudes = []
    for path in selected:
        try:
            _count_waveform(path, counts, amplitudes)
        except (ValueError, OSError, IndexError):
            counts["decode_error"] += 1
    return {"sample_size": len(selected), "seed": SEED, "counts": dict(counts),
            "max_absolute_mV_median": float(np.median(amplitudes)) if amplitudes else None,
            "max_absolute_mV_p95": float(np.percentile(amplitudes, 95)) if amplitudes else None,
            "near_flat_rule": ("screening flag: any lead standard deviation < 0.01 mV; "
                               "not a clinical quality label")}


def mimic_sample_paths() -> list[Path]:
    """
    List the raw header paths of the audited 40k MIMIC pool.

    Returns
    -------
    list[Path]
        One path per manifest row.
    """
    manifest = ROOT / "data/processed/mimic_ssl_40k_cpc/ssl_manifest.csv"
    with manifest.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    return [ROOT / "data/raw/mimic-iv-ecg/1.0" / row["filename_hr"] for row in rows]


def ptbxl_summary() -> dict[str, Any]:
    """
    Summarize PTB-XL metadata and a sample of its 500 Hz waveform pairs.

    Returns
    -------
    dict[str, Any]
        Metadata summary with local pair count and waveform sample.
    """
    summary = csv_summary(ROOT / "data/raw/ptb-xl/1.0.3/ptbxl_database.csv", "patient_id", "age",
                          ("scp_codes", "report", "validated_by_human", "second_opinion"))
    # PTB-XL uses .dat pairs; the Challenge sources use .mat pairs.
    records = ROOT / "data/raw/ptb-xl/1.0.3/records500"
    paths = [path for path in records.rglob("*.hea") if path.with_suffix(".dat").is_file()]
    summary["local_500_hz_pairs"] = len(paths)
    summary["waveform_sample"] = waveform_sample(paths)
    return summary


def build_report() -> dict[str, Any]:
    """
    Build the full local data quality inventory.

    Returns
    -------
    dict[str, Any]
        Report keyed by source.
    """
    report = {"generated_at_utc": utc_now(),
              "sample_size_per_source": SAMPLE_SIZE,
              "sampling": "deterministic simple random sample of complete local waveform pairs, seed 42",
              "ptbxl": ptbxl_summary(),
              "code_15pct": csv_summary(ROOT / "data/raw/code-15pct/zenodo-4916206/exams.csv",
                                        "patient_id", "age", ("normal_ecg", "AF", "RBBB", "LBBB", "death"))}
    for name, root in HEADER_SOURCES.items():
        summary, paths = header_summary(root)
        summary["waveform_sample"] = waveform_sample(paths)
        report[name] = summary
    report["mimic_iv_ecg"] = {
        "audited_40k_pool": json.loads((ROOT / "data/processed/mimic_ssl_40k_cpc/metadata.json").read_text()),
        "waveform_sample": waveform_sample(mimic_sample_paths()),
    }
    return report


def main() -> None:
    """Write the inventory report and print its path."""
    report = build_report()
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(OUTPUT)


if __name__ == "__main__":
    main()
