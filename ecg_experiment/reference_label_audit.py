"""Audit public PTB reasoning annotations without interpreting quarantined cases."""

from __future__ import annotations

import ast
import csv
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ecg_experiment.eda.ptbxl import diagnostic_classes, project_label, superclass_label

TIME_RANGE = re.compile(r"\[\s*(\d+(?:\.\d+)?)s\s*-\s*(\d+(?:\.\d+)?)s\s*\]")
GROUNDING_TYPES = {"lead_grounding", "wave_grounding", "measurement_grounding"}


def integer_identifier(value: object) -> int | None:
    """Accept integer identifiers without truncating floats or accepting booleans.

    Parameters
    ----------
    value : object
        Untrusted JSON identifier.

    Returns
    -------
    int or None
        Nonnegative integer, or None for malformed or missing identifiers.
    """
    if type(value) is int and value >= 0:
        return value
    if isinstance(value, str) and re.fullmatch(r"[0-9]+", value):
        return int(value)
    return None


def identifier_table(path: Path) -> pd.DataFrame:
    """Read identifiers and folds, without clinical metadata.

    Parameters
    ----------
    path : Path
        PTB metadata CSV.

    Returns
    -------
    pd.DataFrame
        Unique ECG identifiers indexed to patient and fold.
    """
    table = pd.read_csv(path, usecols=["ecg_id", "patient_id", "strat_fold"])
    if table.ecg_id.duplicated().any() or table.isna().any().any():
        raise ValueError("Metadata identifiers must be unique and complete")
    if not table.strat_fold.isin(range(1, 11)).all():
        raise ValueError("Unexpected PTB fold")
    return table.astype(int).set_index("ecg_id")


def question_rows(data: dict[str, Any]) -> list[dict[str, Any]]:
    """Flatten one permitted case into question dictionaries.

    Parameters
    ----------
    data : dict
        Reasoning payload of a permitted-fold case only.

    Returns
    -------
    list of dict
        Initial, criterion, finding, grounding and diagnostic questions.
    """
    questions = [data["initial_diagnostic_question"]]
    for step in data["reasoning"]:
        for key, value in step.items():
            questions.extend(value if key == "ecg_grounding" else [value])
    return questions


def grounding_summary(questions: list[dict[str, Any]]) -> dict[str, Any]:
    """Validate answer consistency and summarize a single grounding type.

    Parameters
    ----------
    questions : list of dict
        Permitted grounding questions of a single type.

    Returns
    -------
    dict
        Counts, exact option-zero accuracy and uniform-position expectation.
    """
    invalid = empty = multi = single = first = 0
    expected = 0.0
    widths: list[float] = []
    endpoints: list[float] = []
    bad_ranges = 0
    positions: Counter[int] = Counter()
    option_counts: Counter[int] = Counter()
    for question in questions:
        options, answer, indices = (question.get(k) for k in ("options", "answer", "answer_idx"))
        valid = isinstance(options, list) and bool(options) and isinstance(answer, list)
        valid = valid and isinstance(indices, list)
        if not valid:
            invalid += 1
            continue
        if not all(isinstance(value, str) for value in [*options, *answer]):
            invalid += 1
            continue
        valid = all(type(i) is int and 0 <= i < len(options) for i in indices)
        if not valid or len(set(indices)) != len(indices) or [options[i] for i in indices] != answer:
            invalid += 1
            continue
        option_counts[len(options)] += 1
        empty += len(indices) == 0
        multi += len(indices) > 1
        if len(indices) == 1:
            single += 1
            first += indices[0] == 0
            expected += 1 / len(options)
            positions[indices[0]] += 1
        if question.get("question_type") == "wave_grounding":
            for value in answer:
                match = TIME_RANGE.fullmatch(value)
                if match is None:
                    bad_ranges += 1
                    continue
                start, stop = map(float, match.groups())
                if not 0 <= start < stop <= 10:
                    bad_ranges += 1
                    continue
                widths.append(stop - start)
                endpoints.extend((start, stop))
    accuracy = first / single if single else None
    chance = expected / single if single else None
    return {
        "questions": len(questions),
        "invalid_answer_schema": invalid,
        "empty_answers": empty,
        "multiple_answers": multi,
        "single_answers": single,
        "option_counts": dict(option_counts),
        "answer_positions_single_only": dict(positions),
        "option_zero_correct": first,
        "option_zero_exact_accuracy": accuracy,
        "uniform_position_expectation": chance,
        "shortcut_flag": bool(single and accuracy - chance >= 0.10),
        "invalid_selected_time_ranges": bad_ranges,
        "selected_time_ranges": len(widths),
        "width_seconds_percentiles": np.percentile(widths, [0, 25, 50, 75, 100]).tolist() if widths else [],
        "endpoints_on_10ms_grid": sum(abs(x * 100 - round(x * 100)) < 1e-7 for x in endpoints),
        "endpoints": len(endpoints),
    }


def audit_reasoning(path: Path, identifiers: pd.DataFrame) -> dict[str, Any]:
    """Count identities globally and inspect clinical payloads only in folds 1–8.

    Parameters
    ----------
    path : Path
        Official public PTB JSONL, never the MIMIC file.
    identifiers : pd.DataFrame
        Output of ``identifier_table``.

    Returns
    -------
    dict
        Aggregate audit without case, patient or waveform payloads.
    """
    cases: Counter[int] = Counter()
    ecgs: Counter[int] = Counter()
    safe_ecgs: Counter[int] = Counter()
    folds: Counter[int] = Counter()
    diagnoses: Counter[str] = Counter()
    polarities: Counter[str] = Counter()
    types: Counter[str] = Counter()
    patients: set[int] = set()
    safe_patients: set[int] = set()
    grounding: dict[str, list[dict[str, Any]]] = defaultdict(list)
    unknown = no_grounding = safe_cases = total = invalid_case = invalid_ecg = 0
    with path.open() as stream:
        for line in stream:
            total += 1
            case = json.loads(line)
            meta = case["metadata"]
            if meta["data_source"] != "ptbxl":
                raise ValueError("Only the public PTB annotation source is permitted")
            case_id = integer_identifier(meta.get("id"))
            ecg = integer_identifier(meta.get("ecg_id"))
            invalid_case += case_id is None
            invalid_ecg += ecg is None or ecg == 0
            if case_id is not None:
                cases[case_id] += 1
            if case_id is None or ecg is None or ecg == 0:
                continue
            ecgs[ecg] += 1
            if ecg not in identifiers.index:
                unknown += 1
                continue
            patient, fold = identifiers.loc[ecg, ["patient_id", "strat_fold"]].astype(int)
            folds[int(fold)] += 1
            patients.add(int(patient))
            if fold > 8:
                continue
            safe_cases += 1
            safe_ecgs[ecg] += 1
            safe_patients.add(int(patient))
            diagnoses[str(meta["target_dx"])] += 1
            if type(meta["dx_label"]) is not bool:
                raise ValueError("Diagnosis polarity must be boolean")
            polarities[str(meta["dx_label"]).lower()] += 1
            rows = question_rows(case["data"])
            types.update(question["question_type"] for question in rows)
            grounded = [q for q in rows if q["question_type"] in GROUNDING_TYPES]
            no_grounding += not grounded
            for question in grounded:
                grounding[question["question_type"]].append(question)
    return {
        "total_cases_identifiers_only": total,
        "invalid_case_identifiers": invalid_case,
        "invalid_ecg_identifiers": invalid_ecg,
        "duplicate_case_ids": sum(v - 1 for v in cases.values()),
        "unknown_ecg_cases": unknown,
        "unique_ecgs_identifiers_only": len(ecgs),
        "unique_patients_identifiers_only": len(patients),
        "cases_by_fold_identifiers_only": dict(sorted(folds.items())),
        "quarantined_cases": sum(v for k, v in folds.items() if k > 8),
        "safe_cases": safe_cases,
        "safe_unique_ecgs": len(safe_ecgs),
        "safe_unique_patients": len(safe_patients),
        "safe_repeat_cases_beyond_unique_ecg": safe_cases - len(safe_ecgs),
        "safe_max_cases_per_ecg": max(safe_ecgs.values(), default=0),
        "safe_diagnosis_cases": dict(sorted(diagnoses.items())),
        "safe_diagnosis_polarities": dict(polarities),
        "safe_question_types": dict(sorted(types.items())),
        "safe_cases_without_grounding": no_grounding,
        "safe_grounding": {key: grounding_summary(grounding[key]) for key in sorted(GROUNDING_TYPES)},
    }


def audit_local_labels(metadata: Path, statements: Path) -> dict[str, Any]:
    """Reproduce descriptive label findings with an independent rule check.

    Parameters
    ----------
    metadata : Path
        Local PTB metadata; only folds 1–8 code dictionaries are interpreted.
    statements : Path
        SCP code schema.

    Returns
    -------
    dict
        Aggregate counts and exact agreement of the two label implementations.
    """
    classes = diagnostic_classes(pd.read_csv(statements, index_col=0))
    records: list[dict[str, Any]] = []
    with metadata.open() as stream:
        for row in csv.DictReader(stream):
            if int(row["strat_fold"]) > 8:
                continue
            codes = ast.literal_eval(row["scp_codes"])
            supers = {classes[code] for code in codes if code in classes}
            abnormal = bool(supers - {"NORM"})
            standard = 1.0 if abnormal else (0.0 if supers == {"NORM"} else np.nan)
            strict = standard if "NORM" not in codes else (0.0 if set(codes) <= {"NORM", "SR"} else np.nan)
            expected = np.array([standard, strict])
            observed = np.array([superclass_label(codes, classes), project_label(codes, classes)])
            if not np.array_equal(expected, observed, equal_nan=True):
                raise ValueError("Existing labels disagree with the independent audit")
            records.append(
                {
                    "patient": int(float(row["patient_id"])),
                    "codes": set(codes),
                    "abnormal": abnormal,
                    "supers": supers,
                    "standard": standard,
                    "strict": strict,
                }
            )
    return {
        "folds": "1-8",
        "records": len(records),
        "patients": len({r["patient"] for r in records}),
        "label_integrity_exact": True,
        "standard_labels": label_counts([r["standard"] for r in records]),
        "strict_project_labels": label_counts([r["strict"] for r in records]),
        "sr_total": sum("SR" in r["codes"] for r in records),
        "sr_with_diagnostic_abnormal": sum("SR" in r["codes"] and r["abnormal"] for r in records),
        "norm_total": sum("NORM" in r["codes"] for r in records),
        "norm_with_diagnostic_abnormal": sum("NORM" in r["codes"] and r["abnormal"] for r in records),
        "norm_with_extra_codes": sum(
            "NORM" in r["codes"] and bool(r["codes"] - {"NORM", "SR"}) for r in records
        ),
        "multiple_abnormal_superclasses": sum(len(r["supers"] - {"NORM"}) >= 2 for r in records),
    }


def label_counts(labels: list[float]) -> dict[str, int]:
    """Count positive, negative and undefined labels without changing their meaning.

    Parameters
    ----------
    labels : list of float
        Binary labels and NaNs.

    Returns
    -------
    dict
        Three exhaustive category counts.
    """
    values = np.asarray(labels)
    return {
        "positive": int((values == 1).sum()),
        "negative": int((values == 0).sum()),
        "undefined": int(np.isnan(values).sum()),
    }
