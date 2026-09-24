#!/usr/bin/env python3
"""Compare age/sex retention before and after public ECG quality policies."""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pandas as pd

from ecg_experiment.code15 import metadata_rows
from ecg_experiment.files import read_csv

ROOT = Path(__file__).resolve().parents[2]
CHALLENGE = ROOT / "data/raw/challenge-2020/1.0.2/training"
STRICT = ROOT / "data/processed/public_ecg_quality/strict_10s"
CENTERED = ROOT / "data/processed/public_ecg_quality/cpsc_ssl_center_crop"
CODE = ROOT / "data/processed/code15_quality/part0_native"
CODE_METADATA = ROOT / "data/raw/code-15pct/zenodo-4916206/exams.csv"
OUTPUT = ROOT / "outputs/data_quality/selection_bias.json"
SOURCES = {"georgia": "georgia", "cpsc_2018": "cpsc_2018",
           "cpsc_2018_extra": "cpsc_2018_extra"}
MAX_AGE = 120
Row = tuple[str, str, str, str]


def age_band(value: str | None) -> str:
    """
    Map a reported age to a coarse band.

    Parameters
    ----------
    value : str | None
        Reported age; missing, unparsable or implausible values are unknown.

    Returns
    -------
    str
        Band name, or ``"unknown"``.
    """
    try:
        age = float(value) if value else float("nan")
    except ValueError:
        return "unknown"
    if not 0 <= age <= MAX_AGE:
        return "unknown"

    band = "over_70"
    if age < 18:
        band = "under_18"
    elif age <= 30:
        band = "18_to_30"
    elif age <= 50:
        band = "31_to_50"
    elif age <= 70:
        band = "51_to_70"
    return band


def sex_value(value: str | None) -> str:
    """
    Normalize reported sex from Challenge headers or CODE ``is_male`` flags.

    Parameters
    ----------
    value : str | None
        ``Male``/``Female`` or ``True``/``False``, in any case.

    Returns
    -------
    str
        ``"male"``, ``"female"`` or ``"unknown"``.
    """
    value = (value or "").strip().lower()
    sex = "unknown"
    if value in {"male", "true"}:
        sex = "male"
    elif value in {"female", "false"}:
        sex = "female"
    return sex


def header_demographics(path: Path) -> tuple[str, str]:
    """
    Read the age band and sex from a WFDB header's comment lines.

    Parameters
    ----------
    path : Path
        Challenge ``.hea`` file.

    Returns
    -------
    tuple[str, str]
        Age band and sex category.
    """
    comments = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith("#") and ":" in line:
            key, value = line[1:].split(":", 1)
            comments[key.strip()] = value.strip()
    return age_band(comments.get("Age")), sex_value(comments.get("Sex"))


def aggregate(items: list[Row]) -> dict[str, Any]:
    """
    Summarize retention by source, age band and sex.

    Parameters
    ----------
    items : list[tuple[str, str, str, str]]
        Rows of (source, age band, sex, outcome); outcome is ``"accepted"`` or
        an exclusion reason.

    Returns
    -------
    dict[str, Any]
        Per-source totals with per-category retention and other outcomes.
    """
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


def outcomes(directory: Path, description: str, column: str = "ecg_id",
             key: Callable[[str], Any] = str) -> dict[Any, str]:
    """
    Map every audited record to ``"accepted"`` or its exclusion reason.

    Parameters
    ----------
    directory : Path
        Prepared directory with ``manifest.csv`` and ``exclusions.csv``.
    description : str
        Name used in the duplicate-outcome error message.
    column : str
        Record ID column in both tables.
    key : Callable[[str], Any]
        Converts the ID column value to the map key.

    Returns
    -------
    dict[Any, str]
        Outcome keyed by record ID.

    Raises
    ------
    ValueError
        If a record is both accepted and excluded, or excluded twice.
    """
    outcome = {key(row[column]): "accepted" for row in read_csv(directory / "manifest.csv")}
    for row in read_csv(directory / "exclusions.csv"):
        record_id = key(row[column])
        if record_id in outcome:
            raise ValueError(f"Duplicate {description} outcome")
        outcome[record_id] = row["reason"]
    return outcome


def challenge_rows(strict_outcome: dict[str, str],
                   centered_outcome: dict[str, str]) -> tuple[list[Row], list[Row]]:
    """
    Pair each Challenge header's demographics with its strict and centered outcome.

    Parameters
    ----------
    strict_outcome : dict[str, str]
        Outcome of every record under the strict ten-second policy.
    centered_outcome : dict[str, str]
        Outcome of every CPSC record under the centered-crop policy.

    Returns
    -------
    tuple[list[Row], list[Row]]
        Strict and centered aggregation rows.

    Raises
    ------
    ValueError
        If headers and policy outcomes do not cover the same records.
    """
    strict_rows, centered_rows = [], []
    for source, directory in SOURCES.items():
        for header in (CHALLENGE / directory).rglob("*.hea"):
            ecg_id = f"{source}:{header.stem}"
            age, sex = header_demographics(header)
            if ecg_id not in strict_outcome:
                raise ValueError(f"Missing strict outcome: {ecg_id}")
            strict_rows.append((source, age, sex, strict_outcome[ecg_id]))
            if source == "georgia":
                continue
            if ecg_id not in centered_outcome:
                raise ValueError(f"Missing centered outcome: {ecg_id}")
            centered_rows.append((source, age, sex, centered_outcome[ecg_id]))
    if len(strict_rows) != len(strict_outcome) or len(centered_rows) != len(centered_outcome):
        raise ValueError("Challenge demographic and policy rows differ")
    return strict_rows, centered_rows


def code_rows() -> list[Row]:
    """
    Pair each audited CODE-15% part-0 exam with its reported demographics.

    Returns
    -------
    list[Row]
        Aggregation rows for the native part-0 preparation.
    """
    metadata = metadata_rows(CODE_METADATA)
    rows = []
    for exam_id, outcome in outcomes(CODE, "CODE", "exam_id", int).items():
        row = metadata.get(exam_id)
        rows.append(("code_part0", age_band(row["age"] if row else None),
                     sex_value(row["is_male"] if row else None), outcome))
    return rows


def main() -> None:
    """Write the selection-bias report and print its path."""
    strict_rows, centered_rows = challenge_rows(outcomes(STRICT, "strict"),
                                                outcomes(CENTERED, "centered"))
    result = {"challenge_strict": aggregate(strict_rows),
              "challenge_centered": aggregate(centered_rows),
              "code_part0_native": aggregate(code_rows()),
              "interpretation": ("Exam/record-level retention by reported age/sex, not patient-level "
                                 "prevalence or causal selection effect; strict CPSC duration policy "
                                 "drives many exclusions.")}
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(OUTPUT)


if __name__ == "__main__":
    main()
