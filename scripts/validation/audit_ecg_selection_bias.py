#!/usr/bin/env python3
"""Compare age/sex retention before and after public ECG quality policies."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pandas as pd

from scripts.data.prepare_code15 import ROOT, metadata_rows


CHALLENGE = ROOT / "data/raw/challenge-2020/1.0.2/training"
STRICT = ROOT / "data/processed/public_ecg_quality/strict_10s"
CENTERED = ROOT / "data/processed/public_ecg_quality/cpsc_ssl_center_crop"
CODE = ROOT / "data/processed/code15_quality/part0_native"
OUTPUT = ROOT / "outputs/data_quality/selection_bias.json"
SOURCES = {"georgia": "georgia", "cpsc_2018": "cpsc_2018",
           "cpsc_2018_extra": "cpsc_2018_extra"}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def age_band(value: str | None) -> str:
    try:
        age = float(value) if value else float("nan")
    except ValueError:
        return "unknown"
    if not 0 <= age <= 120:
        return "unknown"
    if age < 18:
        return "under_18"
    if age <= 30:
        return "18_to_30"
    if age <= 50:
        return "31_to_50"
    if age <= 70:
        return "51_to_70"
    return "over_70"


def sex_value(value: str | None) -> str:
    value = (value or "").strip().lower()
    if value in {"male", "true"}:
        return "male"
    if value in {"female", "false"}:
        return "female"
    return "unknown"


def header_demographics(path: Path) -> tuple[str, str]:
    comments = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith("#") and ":" in line:
            key, value = line[1:].split(":", 1)
            comments[key.strip()] = value.strip()
    return age_band(comments.get("Age")), sex_value(comments.get("Sex"))


def aggregate(items: list[tuple[str, str, str, str]]) -> dict:
    """Rows are (source, age band, sex, outcome)."""
    frame = pd.DataFrame(items, columns=("source", "age_band", "sex", "outcome"))
    result = {}
    for source, subset in frame.groupby("source", sort=True):
        groups = {}
        for field, column in (("age_bands", "age_band"), ("sex", "sex")):
            counts = pd.crosstab(subset[column], subset["outcome"])
            groups[field] = {}
            for category, row in counts.iterrows():
                outcomes = {name: int(count) for name, count in row.items() if count}
                total = sum(outcomes.values())
                accepted = outcomes.pop("accepted", 0)
                groups[field][category] = {
                    "total": total, "accepted": accepted,
                    "retained_percent": round(100 * accepted / total, 2),
                    "other_outcomes": outcomes,
                }
        result[source] = {"records": len(subset),
                          "accepted": int((subset["outcome"] == "accepted").sum()),
                          **groups}
    return result


def main() -> None:
    strict_manifest = read_csv(STRICT / "manifest.csv")
    strict_exclusions = read_csv(STRICT / "exclusions.csv")
    strict_outcome = {row["ecg_id"]: "accepted" for row in strict_manifest}
    for row in strict_exclusions:
        if row["ecg_id"] in strict_outcome:
            raise ValueError("Duplicate strict outcome")
        strict_outcome[row["ecg_id"]] = row["reason"]
    centered_manifest = read_csv(CENTERED / "manifest.csv")
    centered_exclusions = read_csv(CENTERED / "exclusions.csv")
    centered_outcome = {row["ecg_id"]: "accepted" for row in centered_manifest}
    for row in centered_exclusions:
        if row["ecg_id"] in centered_outcome:
            raise ValueError("Duplicate centered outcome")
        centered_outcome[row["ecg_id"]] = row["reason"]
    strict_rows, centered_rows = [], []
    for source, directory in SOURCES.items():
        for header in (CHALLENGE / directory).rglob("*.hea"):
            ecg_id = f"{source}:{header.stem}"
            age, sex = header_demographics(header)
            if ecg_id not in strict_outcome:
                raise ValueError(f"Missing strict outcome: {ecg_id}")
            strict_rows.append((source, age, sex, strict_outcome[ecg_id]))
            if source != "georgia":
                if ecg_id not in centered_outcome:
                    raise ValueError(f"Missing centered outcome: {ecg_id}")
                centered_rows.append((source, age, sex, centered_outcome[ecg_id]))
    if len(strict_rows) != len(strict_outcome) or len(centered_rows) != len(centered_outcome):
        raise ValueError("Challenge demographic and policy rows differ")
    code_metadata = metadata_rows(ROOT / "data/raw/code-15pct/zenodo-4916206/exams.csv")
    code_manifest = read_csv(CODE / "manifest.csv")
    code_exclusions = read_csv(CODE / "exclusions.csv")
    code_outcome = {int(row["exam_id"]): "accepted" for row in code_manifest}
    for row in code_exclusions:
        exam_id = int(row["exam_id"])
        if exam_id in code_outcome:
            raise ValueError("Duplicate CODE outcome")
        code_outcome[exam_id] = row["reason"]
    code_rows = []
    for exam_id, outcome in code_outcome.items():
        row = code_metadata.get(exam_id)
        code_rows.append(("code_part0", age_band(row["age"] if row else None),
                          sex_value(row["is_male"] if row else None), outcome))
    result = {"challenge_strict": aggregate(strict_rows),
              "challenge_centered": aggregate(centered_rows),
              "code_part0_native": aggregate(code_rows),
              "interpretation": "Exam/record-level retention by reported age/sex, not patient-level prevalence or causal selection effect; strict CPSC duration policy drives many exclusions."}
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(OUTPUT)


if __name__ == "__main__":
    main()
