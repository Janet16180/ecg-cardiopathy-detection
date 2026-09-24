#!/usr/bin/env python3
"""Read-only, reproducible inventory for the public ECG data quality review."""

from __future__ import annotations

import csv
import json
import random
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import wfdb


ROOT = Path(__file__).resolve().parents[2]
LEADS = {"I", "II", "III", "AVR", "AVL", "AVF", "V1", "V2", "V3", "V4", "V5", "V6"}
SEED = 42
SAMPLE_SIZE = 64


def age_summary(values: list[float]) -> dict:
    return {"known": len(values), "median": float(np.median(values)) if values else None,
            "age_18_to_30_percent": round(100 * sum(18 <= a <= 30 for a in values) / len(values), 2)
            if values else None}


def optional_age(value: str) -> float | None:
    try:
        age = float(value)
        return age if np.isfinite(age) and 0 <= age <= 120 else None
    except (ValueError, TypeError):
        return None


def csv_summary(path: Path, patient_field: str, age_field: str,
                extra_fields: tuple[str, ...]) -> dict:
    count, patients, ages = 0, set(), []
    present = Counter()
    values = Counter()
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
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


def header_summary(root: Path, require_pair: bool = True) -> tuple[dict, list[Path]]:
    headers = sorted(root.rglob("*.hea"))
    counts = Counter()
    diagnoses = Counter()
    ages = []
    available = []
    for path in headers:
        try:
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
            if not require_pair or path.with_suffix(".mat").is_file():
                available.append(path)
        except (IndexError, ValueError, OSError):
            counts["malformed_headers"] += 1
    counts["headers_found"] = len(headers)
    counts["complete_pairs"] = len(available)
    return {"counts": dict(counts), "ages": age_summary(ages),
            "distinct_diagnosis_codes": len(diagnoses),
            "top_diagnosis_codes": diagnoses.most_common(12)}, available


def waveform_sample(paths: list[Path]) -> dict:
    rng = random.Random(SEED)
    selected = rng.sample(paths, min(SAMPLE_SIZE, len(paths)))
    counts = Counter()
    amplitudes = []
    for path in selected:
        try:
            signal, fields = wfdb.rdsamp(str(path.with_suffix("")))
            counts["decoded"] += 1
            finite = bool(np.isfinite(signal).all())
            counts["finite"] += finite
            if finite:
                std = signal.std(axis=0)
                counts["any_near_flat_lead"] += bool(np.any(std < 0.01))
                counts["all_near_flat_leads"] += bool(np.all(std < 0.01))
                amplitudes.append(float(np.max(np.abs(signal))))
            counts["12_leads"] += signal.shape[1] == 12
        except (ValueError, OSError, IndexError):
            counts["decode_error"] += 1
    return {"sample_size": len(selected), "seed": SEED, "counts": dict(counts),
            "max_absolute_mV_median": float(np.median(amplitudes)) if amplitudes else None,
            "max_absolute_mV_p95": float(np.percentile(amplitudes, 95)) if amplitudes else None,
            "near_flat_rule": "screening flag: any lead standard deviation < 0.01 mV; not a clinical quality label"}


def mimic_sample_paths() -> list[Path]:
    manifest = ROOT / "data/processed/mimic_ssl_40k_cpc/ssl_manifest.csv"
    with manifest.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    return [ROOT / "data/raw/mimic-iv-ecg/1.0" / row["filename_hr"] for row in rows]


def main() -> None:
    report = {"generated_at_utc": datetime.now(timezone.utc).isoformat(),
              "sample_size_per_source": SAMPLE_SIZE,
              "sampling": "deterministic simple random sample of complete local waveform pairs, seed 42",
              "ptbxl": csv_summary(ROOT / "data/raw/ptb-xl/1.0.3/ptbxl_database.csv",
                                   "patient_id", "age", ("scp_codes", "report", "validated_by_human", "second_opinion")),
              "code_15pct": csv_summary(ROOT / "data/raw/code-15pct/zenodo-4916206/exams.csv",
                                        "patient_id", "age", ("normal_ecg", "AF", "RBBB", "LBBB", "death"))}
    sources = {
        "ptbxl": ROOT / "data/raw/ptb-xl/1.0.3/records500",
        "georgia": ROOT / "data/raw/challenge-2020/1.0.2/training/georgia",
        "cpsc_2018": ROOT / "data/raw/challenge-2020/1.0.2/training/cpsc_2018",
        "cpsc_2018_extra": ROOT / "data/raw/challenge-2020/1.0.2/training/cpsc_2018_extra",
        "chapman_shaoxing": ROOT / "data/raw/challenge-2021/1.0.3/training/chapman_shaoxing",
    }
    # PTB-XL uses .dat pairs; the others use .mat pairs.
    for name, root in sources.items():
        if name == "ptbxl":
            paths = [path for path in root.rglob("*.hea") if path.with_suffix(".dat").is_file()]
            report[name]["local_500_hz_pairs"] = len(paths)
            report[name]["waveform_sample"] = waveform_sample(paths)
        else:
            summary, paths = header_summary(root)
            summary["waveform_sample"] = waveform_sample(paths)
            report[name] = summary
    mimic_paths = mimic_sample_paths()
    report["mimic_iv_ecg"] = {
        "audited_40k_pool": json.loads((ROOT / "data/processed/mimic_ssl_40k_cpc/metadata.json").read_text()),
        "waveform_sample": waveform_sample(mimic_paths),
    }
    output = ROOT / "outputs/data_quality/local_audit.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(output)


if __name__ == "__main__":
    main()
