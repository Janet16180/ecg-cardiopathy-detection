"""Training-free physical reference matching within supplied ECG candidate windows."""

from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import wfdb
from scipy.signal import find_peaks

from ecg_experiment.intervals import patient_groups, patient_resample
from ecg_experiment.reference_label_audit import TIME_RANGE, integer_identifier

SEEDS = (20261018, 20261019, 20261020)
METHODS = (
    "morphology",
    "duration",
    "center",
    "qrs_phase",
    "amplitude",
    "energy",
    "wrong_window",
    "shuffled_class",
    "option_zero",
    "uniform",
)


def digest_integer(text: str) -> int:
    """Return the integer SHA256 digest of UTF-8 text."""
    return int(hashlib.sha256(text.encode()).hexdigest(), 16)


def inner_fold(patient: int) -> int:
    """Assign one patient to a deterministic inner fold."""
    return digest_integer(f"ntt02_inner:{patient}") % 3


def eligibility_table(metadata: Path) -> pd.DataFrame:
    """Read identity/path metadata and quarantine whole patients before content access.

    Parameters
    ----------
    metadata : Path
        Original PTB-XL metadata CSV.

    Returns
    -------
    pd.DataFrame
        Identity rows with a training-only eligibility flag.
    """
    columns = ["ecg_id", "patient_id", "strat_fold", "filename_lr"]
    table = pd.read_csv(metadata, usecols=columns)
    if table[columns[:3]].isna().any().any() or table.ecg_id.duplicated().any():
        raise ValueError("Invalid metadata identities")
    for name in columns[:3]:
        if not (table[name] == table[name].astype(int)).all():
            raise ValueError("Noninteger source identity")
        table[name] = table[name].astype(int)
    if not table.strat_fold.isin(range(1, 11)).all():
        raise ValueError("Invalid source fold")
    closed_patients = set(table.loc[table.strat_fold > 8, "patient_id"])
    table["eligible"] = [
        patient not in closed_patients
        and digest_integer(f"ssl_synthetic_partition_v1:20261018:{patient}") % 10000 < 8000
        for patient in table.patient_id
    ]
    return table.set_index("ecg_id")


def timestamp_options(question: dict[str, Any]) -> np.ndarray | None:
    """Parse distinct absolute-time candidates and validate their released answer."""
    options = question["options"]
    indices = question["answer_idx"]
    if len(indices) != 1 or type(indices[0]) is not int or not 0 <= indices[0] < len(options):
        raise ValueError("Wave question must have one valid answer index")
    if question["answer"] != [options[indices[0]]]:
        raise ValueError("Released option and answer disagree")
    matches = [TIME_RANGE.fullmatch(option) for option in options]
    if not all(matches):
        return None
    intervals = np.array([list(map(float, match.groups())) for match in matches])
    if not np.all((intervals[:, 0] >= 0) & (intervals[:, 1] > intervals[:, 0]) & (intervals[:, 1] <= 10)):
        raise ValueError("Invalid absolute timestamp")
    if len(intervals) < 2 or len(set(map(tuple, intervals))) != len(intervals):
        return None
    return intervals


def case_questions(case: dict[str, Any], ecg: int, row: pd.Series) -> tuple[list[dict], Counter]:
    """Extract positive-finding typed references from an already eligible case."""
    counts: Counter = Counter()
    result = []
    if case["metadata"]["dx_label"] is not True:
        counts["nonpositive_diagnosis_cases"] += 1
        return result, counts
    for step_index, step in enumerate(case["data"]["reasoning"]):
        waves = [q for q in step["ecg_grounding"] if q["question_type"] == "wave_grounding"]
        if step["finding_identification"]["answer"] != "Yes":
            counts["nonaffirmative_finding_wave_questions"] += len(waves)
            continue
        criterion = step["criterion_selection"]["answer"]
        if not isinstance(criterion, str) or not criterion.strip():
            raise ValueError("Missing named criterion")
        leads = [q["answer"] for q in step["ecg_grounding"] if q["question_type"] == "lead_grounding"]
        for wave_index, question in enumerate(waves):
            intervals = timestamp_options(question)
            if intervals is None:
                counts["nonabsolute_or_nondistinct_wave_questions"] += 1
                continue
            key = criterion + "\n" + question["question"]
            result.append(
                {
                    "ecg_id": ecg,
                    "patient_id": int(row.patient_id),
                    "case_id": int(case["metadata"]["id"]),
                    "step": step_index,
                    "wave": wave_index,
                    "diagnosis": case["metadata"]["target_dx"],
                    "criterion": criterion,
                    "question": question["question"],
                    "finding_polarity": "Yes",
                    "class_key": key,
                    "coordinate_type": "absolute_seconds",
                    "intervals": intervals.tolist(),
                    "answer": question["answer_idx"][0],
                    "step_lead_answers_unlinked": leads,
                    "waveform": str(row.filename_lr),
                    "units": "mV",
                    "sampling_hz": 100,
                }
            )
    return result, counts


def read_questions(annotations: Path, table: pd.DataFrame) -> tuple[list[dict], dict]:
    """Check patient eligibility before interpreting diagnosis, reasoning or questions."""
    questions = []
    counts: Counter = Counter()
    with annotations.open() as stream:
        for line in stream:
            case = json.loads(line)
            meta = case["metadata"]
            if meta["data_source"] != "ptbxl":
                raise ValueError("Only public PTB input is allowed")
            ecg = integer_identifier(meta.get("ecg_id"))
            if ecg is None or ecg not in table.index:
                raise ValueError("Unknown ECG identity")
            row = table.loc[ecg]
            if not row.eligible:
                counts["ineligible_cases_identity_only"] += 1
                continue
            counts["eligible_train_cases"] += 1
            items, removed = case_questions(case, ecg, row)
            counts.update(removed)
            questions.extend(items)
    counts["positive_absolute_questions_before_cap"] = len(questions)
    kept = []
    per_patient_class: Counter = Counter()
    for question in sorted(questions, key=lambda q: (q["case_id"], q["step"], q["wave"])):
        pair = (question["patient_id"], question["class_key"])
        per_patient_class[pair] += 1
        if per_patient_class[pair] <= 3:
            question["qid"] = len(kept)
            kept.append(question)
    counts["capped_questions_removed"] = len(questions) - len(kept)
    return kept, dict(counts)


def class_groups(questions: list[dict]) -> tuple[dict[str, list[dict]], list[dict]]:
    """Apply the prospective patient-count rule without reading waveform features."""
    groups: dict[str, list[dict]] = defaultdict(list)
    for question in questions:
        groups[question["class_key"]].append(question)
    eligible = {}
    inventory = []
    for key, rows in sorted(groups.items()):
        patients = sorted({q["patient_id"] for q in rows})
        outside = [sum(inner_fold(patient) != fold for patient in patients) for fold in range(3)]
        allowed = len(patients) >= 12 and min(outside) >= 3
        inventory.append(
            {
                "class_key": key,
                "questions": len(rows),
                "patients": len(patients),
                "outside_inner_fold_patients": outside,
                "eligible": allowed,
                "reason": "eligible" if allowed else "insufficient_independent_patients",
            }
        )
        if allowed:
            eligible[key] = rows
    if len(eligible) < 2:
        raise ValueError("Need at least two eligible classes for shuffled-reference control")
    return eligible, inventory


def unit(vector: np.ndarray) -> np.ndarray:
    """Normalize a vector, retaining zero vectors as zero."""
    return vector / max(float(np.linalg.norm(vector)), 1e-12)


def qrs_centers(signal: np.ndarray, fs: int = 100) -> np.ndarray:
    """Propose QRS centers from smoothed all-lead derivative energy without labels."""
    derivative = np.diff(signal, axis=0, prepend=signal[:1])
    energy = np.mean(derivative**2, axis=1)
    energy = np.convolve(energy, np.ones(5) / 5, mode="same")
    peaks, _ = find_peaks(energy, distance=int(0.25 * fs), prominence=max(np.std(energy), 1e-12))
    return peaks / fs


def interval_features(
    signal: np.ndarray, intervals: np.ndarray, peaks: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Pool masked waveforms and physical width/center/QRS/amplitude/energy descriptors."""
    vectors, scalars = [], []
    for start, stop in intervals:
        first, last = int(round(start * 100)), int(round(stop * 100))
        segment = signal[first:last]
        if len(segment) < 2:
            raise ValueError("Window shorter than two native samples")
        grid = np.linspace(0, len(segment) - 1, 32)
        pooled = np.stack([np.interp(grid, np.arange(len(segment)), lead) for lead in segment.T])
        vectors.append(unit(pooled.ravel()))
        center = (start + stop) / 2
        phase = float(np.min(np.abs(peaks - center))) if len(peaks) else 10.0
        scalars.append(
            [
                np.log(stop - start),
                center,
                phase,
                np.sqrt(np.mean(segment**2)),
                np.sqrt(np.mean(np.diff(segment, axis=0) ** 2)),
            ]
        )
    return np.array(vectors), np.array(scalars)


def attach_features(groups: dict[str, list[dict]], base: Path) -> dict[str, str]:
    """Read each permitted physical recording once and bind waveform input bytes."""
    from ecg_experiment.files import sha256_file
    from ecg_experiment.paths import to_stored

    cache = {}
    hashes = {}
    for rows in groups.values():
        for question in rows:
            name = question["waveform"]
            if name not in cache:
                path = base / name
                signal, fields = wfdb.rdsamp(str(path))
                if fields["fs"] != 100 or signal.shape != (1000, 12) or set(fields["units"]) != {"mV"}:
                    raise ValueError("Expected physical 12-lead 100 Hz ten-second mV signal")
                if not np.isfinite(signal).all():
                    raise ValueError("Nonfinite permitted waveform")
                signal = signal - np.median(signal, axis=0)
                cache[name] = (signal, qrs_centers(signal))
                for suffix in (".hea", ".dat"):
                    source = path.with_suffix(suffix)
                    hashes[to_stored(source)] = sha256_file(source)
            signal, peaks = cache[name]
            question["features"], question["scalars"] = interval_features(
                signal,
                np.asarray(question["intervals"]),
                peaks,
            )
    return hashes


def support_patients(rows: list[dict], key: str, fold: int, seed: int, budget: int) -> list[int]:
    """Return a nested deterministic reference draw outside the target patient fold."""
    patients = {q["patient_id"] for q in rows if inner_fold(q["patient_id"]) != fold}
    order = sorted(patients, key=lambda patient: digest_integer(f"{seed}:{key}:{patient}"))
    if len(order) < budget:
        raise ValueError("Insufficient out-of-fold reference patients")
    return order[:budget]


def prototype(rows: list[dict], patients: list[int], wrong: bool = False) -> tuple[np.ndarray, np.ndarray]:
    """Average masked instances within patients and then across reference patients."""
    vectors, scalars = [], []
    for patient in patients:
        instances = [q for q in rows if q["patient_id"] == patient]
        indices = [
            next(i for i in range(len(q["intervals"])) if i != q["answer"]) if wrong else q["answer"]
            for q in instances
        ]
        vectors.append(np.mean([q["features"][i] for q, i in zip(instances, indices, strict=True)], axis=0))
        scalars.append(np.mean([q["scalars"][i] for q, i in zip(instances, indices, strict=True)], axis=0))
    return unit(np.mean(vectors, axis=0)), np.median(scalars, axis=0)


def overlap(first: list[float], second: list[float]) -> float:
    """Return temporal intersection over union of two positive-width intervals."""
    intersection = max(0.0, min(first[1], second[1]) - max(first[0], second[0]))
    return intersection / (max(first[1], second[1]) - min(first[0], second[0]))


def score_groups(groups: dict[str, list[dict]]) -> tuple[list[dict], list[dict]]:
    """Evaluate fixed masked prototypes and controls on patient-disjoint inner folds."""
    predictions, manifests = [], []
    keys = sorted(groups)
    for class_index, key in enumerate(keys):
        rows = groups[key]
        swapped_key = keys[(class_index + 1) % len(keys)]
        for fold in range(3):
            targets = [q for q in rows if inner_fold(q["patient_id"]) == fold]
            for seed in SEEDS:
                for budget in (1, 3):
                    patients = support_patients(rows, key, fold, seed, budget)
                    swapped = support_patients(groups[swapped_key], swapped_key, fold, seed, budget)
                    vector, scalar = prototype(rows, patients)
                    wrong, _ = prototype(rows, patients, wrong=True)
                    shuffled, _ = prototype(groups[swapped_key], swapped)
                    manifests.append(
                        {
                            "class_key": key,
                            "fold": fold,
                            "seed": seed,
                            "budget": budget,
                            "patients": patients,
                            "shuffled_patients": swapped,
                            "shuffled_class": swapped_key,
                        }
                    )
                    for question in targets:
                        candidates = question["features"]
                        chosen = {
                            "morphology": int(np.argmax(candidates @ vector)),
                            "wrong_window": int(np.argmax(candidates @ wrong)),
                            "shuffled_class": int(np.argmax(candidates @ shuffled)),
                            "option_zero": 0,
                        }
                        for column, method in enumerate(
                            ("duration", "center", "qrs_phase", "amplitude", "energy")
                        ):
                            chosen[method] = int(
                                np.argmin(np.abs(question["scalars"][:, column] - scalar[column]))
                            )
                        truth = question["answer"]
                        intervals = question["intervals"]
                        predictions.append(
                            {
                                "qid": question["qid"],
                                "patient_id": question["patient_id"],
                                "class_key": key,
                                "fold": fold,
                                "seed": seed,
                                "budget": budget,
                                "answer": truth,
                                "chosen": chosen,
                                "accuracy": {
                                    **{m: float(i == truth) for m, i in chosen.items()},
                                    "uniform": 1 / len(intervals),
                                },
                                "iou": {
                                    **{m: overlap(intervals[i], intervals[truth]) for m, i in chosen.items()},
                                    "uniform": float(
                                        np.mean([overlap(i, intervals[truth]) for i in intervals])
                                    ),
                                },
                            }
                        )
    return predictions, manifests


def patient_macro(rows: list[dict], metric: str, method: str) -> tuple[np.ndarray, np.ndarray]:
    """Average repeat questions and reference draws within each patient."""
    values: dict[int, list[float]] = defaultdict(list)
    for row in rows:
        values[row["patient_id"]].append(row[metric][method])
    patients = np.array(sorted(values))
    return patients, np.array([np.mean(values[p]) for p in patients])


def paired_interval(patients: np.ndarray, differences: np.ndarray) -> dict[str, Any]:
    """Use shared whole-patient bootstrap draws for paired macro-accuracy differences."""
    groups = patient_groups(patients)
    rng = np.random.default_rng(20261018)
    draws = [float(np.mean(differences[patient_resample(groups, rng)])) for _ in range(2000)]
    low, high = np.percentile(draws, [2.5, 97.5])
    return {
        "difference": float(np.mean(differences)),
        "ci_low": float(low),
        "ci_high": float(high),
        "draws": 2000,
        "seed": 20261018,
        "patients": len(patients),
    }


def summarize(predictions: list[dict], inventory: list[dict]) -> dict[str, Any]:
    """Compute all prospective endpoints without fitting or post-score selection."""
    result: dict[str, Any] = {"class_inventory": inventory, "budgets": {}}
    for budget in (1, 3):
        rows = [row for row in predictions if row["budget"] == budget]
        summary = {
            metric: {m: float(patient_macro(rows, metric, m)[1].mean()) for m in METHODS}
            for metric in ("accuracy", "iou")
        }
        summary["draw_accuracy"] = {
            str(seed): {
                m: float(patient_macro([r for r in rows if r["seed"] == seed], "accuracy", m)[1].mean())
                for m in METHODS
            }
            for seed in SEEDS
        }
        summary["by_class"] = {
            key: {
                m: float(patient_macro([r for r in rows if r["class_key"] == key], "accuracy", m)[1].mean())
                for m in METHODS
            }
            for key in sorted({r["class_key"] for r in rows})
        }
        result["budgets"][str(budget)] = summary
    rows = [row for row in predictions if row["budget"] == 3]
    patients, morphology = patient_macro(rows, "accuracy", "morphology")
    result["paired_comparisons_budget3"] = {
        method: paired_interval(patients, morphology - patient_macro(rows, "accuracy", method)[1])
        for method in METHODS
        if method != "morphology"
    }
    primary = result["paired_comparisons_budget3"]["duration"]
    accuracy = result["budgets"]["3"]["accuracy"]
    result["advance_gate_passed"] = primary["ci_low"] > 0 and all(
        accuracy["morphology"] > accuracy[method] for method in METHODS if method != "morphology"
    )
    result["primary"] = primary
    result["patients"] = len(patients)
    result["unique_questions"] = len({r["qid"] for r in rows})
    result["candidate_oracle_exact_answer"] = 1.0
    return result
