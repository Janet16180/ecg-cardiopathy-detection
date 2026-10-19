"""Independent six-array mode audit; no ECG, checkpoint, inference or production metric imports."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import time
from pathlib import Path
from typing import Any

for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_name] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import numpy as np  # noqa: E402

from ecg_experiment.intervals import patient_groups, patient_resample  # noqa: E402

BASE = "outputs/ssl_synthetic_localization"
STUDY = BASE + "/counterfactual_factorial_v1"
MODE = STUDY + "/mode_diagnostic_v1"
PARENT_AUDIT = STUDY + "/independent_audit_v1/summary.json"
PARENT_EVAL = STUDY + "/evaluation_v2/summary.json"
PARENT_SHA = "74b02f60ac0d77bf071bc99e3ae6fcf460fd8ef0dfba21b9626f5d67646d6c04"
EVAL_SHA = "3f4555614180442c1eb82c5e5b20ca4137d53ae35697421bfa4337cf4f588d28"
CONFIG = "configs/ssl_synthetic_factorial_mode_diagnostic_v1.json"
ARMS = ("RB", "SB", "RP", "SP")
MODES = ("focal", "repeated")
PHASES = ("baseline", "edited", "change")
FIELDS = (
    "full_grid_ap",
    "lead_ap",
    "temporal_ap",
    "oracle_temporal_ap",
    "top_1_precision",
    "top_1_recall",
    "top_5_precision",
    "top_5_recall",
    "top_10_precision",
    "top_10_recall",
    "mass_T",
    "mass_U",
    "mass_O",
    "candidate_minus_outside",
)
ARRAYS = {STUDY + "/evaluation_v2/probabilities_" + arm + ".npy" for arm in ARMS}
ARRAYS |= {STUDY + "/preparation/" + name + ".npy" for name in ("support", "candidate")}
SEED, DRAWS = 20261028, 2000


def permitted(name: str) -> bool:
    """Decide access before opening a file, with exactly six numerical payloads."""
    path = Path(name)
    if path.is_absolute() or ".." in path.parts or "\\" in name:
        raise ValueError("Nonrelative path rejected")
    if name in ARRAYS:
        return True
    return not name.startswith("data/") and path.suffix in {
        ".json",
        ".jsonl",
        ".py",
        ".md",
        ".toml",
        ".lock",
        ".yaml",
        ".yml",
        ".txt",
    }


def array_digest(value: np.ndarray) -> str:
    """Hash contiguous native array bytes independently of receipt helpers."""
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def bytes_under(path: Path) -> int:
    """Count storage through metadata without reading file contents."""
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


class Audit:
    """Independent identity, comparison and deadline accounting."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self.cpu, self.wall = time.process_time(), time.perf_counter()
        self.ceiling = 118.0
        self.checks = 0
        self.mismatches: list[str] = []
        self.bindings: dict[str, str] = {}
        self.verified: dict[str, str] = {}
        self.authorities: dict[str, str] = {}
        self.exceptions: dict[str, dict] = {}

    def live(self) -> None:
        """Stop before another unit of work if the CPU or wall allowance is exhausted."""
        if max(time.process_time() - self.cpu, time.perf_counter() - self.wall) >= self.ceiling:
            raise TimeoutError("Independent mode audit CPU/wall allowance exhausted")

    def check(self, label: str, condition: bool, fatal: bool = False) -> None:
        """Count a comparison and fail closed when its identity requirement is mandatory."""
        self.live()
        self.checks += 1
        if not condition:
            self.mismatches.append(label)
            if fatal:
                raise ValueError("Independent audit integrity violation: " + label)

    def compare(self, label: str, expected: Any, actual: Any, tolerance: float = 1e-12) -> None:
        """Compare complete nested schemas and numerical leaves at the declared tolerance."""
        if isinstance(expected, dict):
            self.check(label + ":keys", isinstance(actual, dict) and expected.keys() == actual.keys())
            if isinstance(actual, dict):
                for key in expected.keys() & actual.keys():
                    self.compare(label + ":" + str(key), expected[key], actual[key], tolerance)
        elif isinstance(expected, list):
            self.check(label + ":length", isinstance(actual, list) and len(expected) == len(actual))
            if isinstance(actual, list):
                for i, (left, right) in enumerate(zip(expected, actual, strict=False)):
                    self.compare(label + ":" + str(i), left, right, tolerance)
        elif isinstance(expected, float):
            self.check(
                label,
                isinstance(actual, (int, float))
                and math.isclose(expected, actual, rel_tol=0, abs_tol=tolerance),
            )
        else:
            self.check(label, type(expected) is type(actual) and expected == actual)

    def digest(self, name: str) -> str:
        """Hash only allowlisted arrays or permitted source and metadata files."""
        if not permitted(name):
            raise ValueError("Forbidden payload rejected before hashing")
        h = hashlib.sha256()
        with (self.root / name).open("rb") as stream:
            while True:
                self.live()
                chunk = stream.read(1024 * 1024)
                if not chunk:
                    break
                h.update(chunk)
        return h.hexdigest()

    def declare(self, name: str, digest: str) -> None:
        """Require a unique byte identity for every declared artifact."""
        self.check("binding_conflict", name not in self.bindings or self.bindings[name] == digest, True)
        self.bindings[name] = digest

    def receipt(self, name: str) -> dict:
        """Read an allowed JSON receipt and collect its source/input/output bindings."""
        self.check("receipt_access", permitted(name) and Path(name).suffix == ".json", True)
        self.declare(name, self.digest(name))
        value = json.loads((self.root / name).read_text())
        for kind in ("source_sha256", "input_sha256"):
            for key, digest in value.get(kind, {}).items():
                self.declare(key, digest)
        for key, digest in value.get("output_sha256", {}).items():
            full = key if key.startswith("outputs/") else (Path(name).parent / key).as_posix()
            self.declare(full, digest)
        return value

    def verify(self) -> None:
        """Freshly verify permitted bytes and defer all other payloads to anchored authority."""
        for name, expected in self.bindings.items():
            self.live()
            if permitted(name):
                actual = self.digest(name)
                self.check("fresh_file_digest", actual == expected, True)
                self.verified[name] = actual
            else:
                self.check("deferred_authority", self.authorities.get(name) == expected, True)
                self.exceptions[name] = {
                    "expected_sha256": expected,
                    "authority_receipt": PARENT_AUDIT,
                    "payload_read": False,
                    "whole_payload_read": False,
                }


def average_precision(truth: np.ndarray, score: np.ndarray) -> float:
    """Independent exact-tie AP using cumulative positives at descending score-group endpoints."""
    labels, values = np.asarray(truth, bool).ravel(), np.asarray(score).ravel()
    positive = int(labels.sum())
    if labels.shape != values.shape or not positive or not np.isfinite(values).all():
        raise ValueError("Invalid AP input")
    order = np.argsort(-values, kind="stable")
    sorted_values = values[order]
    ends = np.r_[np.where(sorted_values[1:] != sorted_values[:-1])[0], len(values) - 1]
    cumulative = np.cumsum(labels[order], dtype=np.int64)[ends]
    additions = np.diff(np.r_[0, cumulative])
    return float(np.sum(additions / positive * cumulative / (ends + 1)))


def coverage(truth: np.ndarray, scores: np.ndarray, budget: int) -> tuple[float, float]:
    """Independently allocate a fixed area with fractional inclusion of every boundary tie."""
    labels, values = truth.ravel(), scores.ravel()
    if not 0 < budget <= len(values) or not labels.any():
        raise ValueError("Invalid fixed-area input")
    cutoff = np.sort(values)[len(values) - budget]
    high, equal = values > cutoff, values == cutoff
    fraction = (budget - np.count_nonzero(high)) / np.count_nonzero(equal)
    hits = np.count_nonzero(labels & high) + fraction * np.count_nonzero(labels & equal)
    return float(hits / budget), float(hits / np.count_nonzero(labels))


def geometry(truth: np.ndarray, candidate: np.ndarray) -> dict:
    """Derive support dimensions and attainable area recall from masks, not predictions."""
    if (
        truth.dtype != bool
        or candidate.dtype != bool
        or truth.shape != (12, 2500)
        or candidate.shape != truth.shape
    ):
        raise ValueError("Native mask geometry/dtype changed")
    if (
        np.any(truth & ~candidate)
        or not truth.any()
        or not (candidate & ~truth).any()
        or not (~candidate).any()
    ):
        raise ValueError("Invalid T/U/O partition")
    lead_truth, temporal_truth = np.any(truth, axis=1), np.any(truth, axis=0)
    if lead_truth.all() or temporal_truth.all() or not np.all(candidate.sum(axis=1) > 0):
        raise ValueError("Projection targets/candidate cells require both classes")
    count = int(np.count_nonzero(truth))
    budgets = {str(percent): math.ceil(truth.size * percent / 100) for percent in (1, 5, 10)}
    return {
        "support_cells": count,
        "candidate_cells": int(np.count_nonzero(candidate)),
        "support_prevalence": count / truth.size,
        "positive_leads": int(lead_truth.sum()),
        "positive_times": int(temporal_truth.sum()),
        "area_budgets": budgets,
        "attainable_recall": {key: min(1.0, value / count) for key, value in budgets.items()},
    }


def map_fields(truth: np.ndarray, candidate: np.ndarray, scores: np.ndarray, geo: dict) -> dict:
    """Compute all14 fields without sklearn or production diagnostic calculators."""
    values = np.asarray(scores, dtype=np.float64)
    if values.shape != truth.shape or not np.isfinite(values).all() or np.any((values < 0) | (values > 1)):
        raise ValueError("Invalid saved probability map")
    mass = float(values.sum())
    if mass <= 0:
        raise ValueError("Undefined zero probability mass")
    active = np.flatnonzero(truth.any(axis=1))
    # Explicit per-lead selection preserves the specified candidate-conditioned projection.
    lead_scores = np.array(
        [values[i, candidate[i]].sum(dtype=np.float64) / int(candidate[i].sum()) for i in range(12)]
    )
    result = {
        "full_grid_ap": average_precision(truth, values),
        "lead_ap": average_precision(truth.any(axis=1), lead_scores),
        "temporal_ap": average_precision(truth.any(axis=0), values.sum(axis=0) / 12),
        "oracle_temporal_ap": float(np.mean([average_precision(truth[i], values[i]) for i in active])),
        "mass_T": float(values[truth].sum() / mass),
        "mass_U": float(values[candidate & ~truth].sum() / mass),
        "mass_O": float(values[~candidate].sum() / mass),
        "candidate_minus_outside": float(
            values[candidate].sum() / candidate.sum() - values[~candidate].sum() / (~candidate).sum()
        ),
    }
    for key, budget in geo["area_budgets"].items():
        result["top_" + key + "_precision"], result["top_" + key + "_recall"] = coverage(
            truth, values, budget
        )
    if set(result) != set(FIELDS):
        raise ValueError("Independent field schema changed")
    return result


def draws() -> np.ndarray:
    """Build the exact shared patient-resampling schedule from its frozen seed."""
    rng = np.random.default_rng(SEED)
    groups = patient_groups(np.arange(128))
    return np.asarray([patient_resample(groups, rng) for _ in range(DRAWS)])


def interval(values: np.ndarray, indices: np.ndarray) -> dict:
    """Independently reduce paired patient draws to a linear percentile interval."""
    values = np.asarray(values, dtype=np.float64)
    if values.shape != (128,) or not np.isfinite(values).all():
        raise ValueError("Every patient is required")
    means = np.array([values[index].mean() for index in indices])
    lower, upper = np.quantile(means, [0.025, 0.975], method="linear")
    return {
        "difference": float(values.mean()),
        "ci_low": float(lower),
        "ci_high": float(upper),
        "subjects": 128,
        "cases": 128,
        "draws": DRAWS,
        "skipped_draws": 0,
        "seed": SEED,
    }


def describe(values: list) -> dict:
    """Summarize immutable geometry without patient filtering."""
    vector = np.asarray(values, dtype=np.float64)
    return {"mean": float(vector.mean()), "minimum": float(vector.min()), "maximum": float(vector.max())}


def aggregate(rows: list[dict], audit: Audit) -> dict:
    """Reconstruct every prespecified arm/mode/phase/field and paired initialization contrast."""
    resampling = draws()
    result = {"statistics": {}, "paired_loss_contrasts": {}, "geometry": {}}
    vectors = {}
    for arm in ARMS:
        result["statistics"][arm] = {}
        for mode in MODES:
            selected = sorted(
                [row for row in rows if row["arm"] == arm and row["mode"] == mode],
                key=lambda row: row["subject"],
            )
            audit.check(
                "complete_patient_alignment", [row["subject"] for row in selected] == list(range(128)), True
            )
            result["statistics"][arm][mode] = {}
            for phase in PHASES:
                result["statistics"][arm][mode][phase] = {}
                for field in FIELDS:
                    audit.live()
                    values = np.array([row[phase][field] for row in selected])
                    vectors[arm, mode, phase, field] = values
                    result["statistics"][arm][mode][phase][field] = interval(values, resampling)
            if arm == "RB":
                keys = (
                    "support_cells",
                    "candidate_cells",
                    "support_prevalence",
                    "positive_leads",
                    "positive_times",
                )
                result["geometry"][mode] = {
                    "patients": 128,
                    "fields": {key: describe([row["geometry"][key] for row in selected]) for key in keys},
                    **{
                        kind: {
                            area: describe([row["geometry"][kind][area] for row in selected])
                            for area in ("1", "5", "10")
                        }
                        for kind in ("area_budgets", "attainable_recall")
                    },
                }
    for initial in ("R", "S"):
        result["paired_loss_contrasts"][initial] = {}
        for mode in MODES:
            result["paired_loss_contrasts"][initial][mode] = {}
            for phase in PHASES:
                result["paired_loss_contrasts"][initial][mode][phase] = {}
                for field in FIELDS:
                    audit.live()
                    difference = (
                        vectors[initial + "P", mode, phase, field]
                        - vectors[initial + "B", mode, phase, field]
                    )
                    result["paired_loss_contrasts"][initial][mode][phase][field] = interval(
                        difference, resampling
                    )
    return result


def run(root: Path, main_sha: str, script_sha: str) -> dict:  # noqa: C901
    """Verify one completed source-bound main receipt, then independently score its six saved arrays."""
    audit = Audit(root)
    destination = root / MODE / "independent_audit_v1"
    if (destination / "summary.json").exists() or (destination / "failure.json").exists():
        raise FileExistsError("Independent mode audit refuses overwrite")
    executed = MODE + "/independent_audit_v1/audit_script.py"
    audit.check("frozen_script_location", Path(__file__) == root / executed, True)
    audit.check("frozen_script_bytes", audit.digest(executed) == script_sha, True)
    audit.check("exact_parent_audit", audit.digest(PARENT_AUDIT) == PARENT_SHA, True)
    audit.check("exact_parent_evaluation", audit.digest(PARENT_EVAL) == EVAL_SHA, True)
    audit.check("exact_main_receipt", audit.digest(MODE + "/summary.json") == main_sha, True)
    parent = audit.receipt(PARENT_AUDIT)
    evaluation = audit.receipt(PARENT_EVAL)
    main = audit.receipt(MODE + "/summary.json")
    audit.check("completed_parent", parent["all_passed"] and parent["mismatch_count"] == 0, True)
    audit.check("parent_bound_to_evaluation", parent["evaluation_summary_sha256"] == EVAL_SHA, True)
    audit.check("completed_mode_main", main["diagnostic_completed"], True)
    audit.ceiling = (
        min(
            120.0,
            240.0 - main["cpu_seconds"],
            240.0 - main["wall_seconds"],
            14400.0 - main["cumulative_cpu_seconds"],
        )
        - 2.0
    )
    audit.live()
    audit.authorities.update(parent["input_sha256"])
    for name, item in parent["deferred_payload_scope_exceptions"].items():
        digest = item["expected_sha256"]
        audit.check(
            "parent_authority_conflict",
            name not in audit.authorities or audit.authorities[name] == digest,
            True,
        )
        audit.authorities[name] = digest
    for key, values in parent["independently_recomputed_complete_summary"].items():
        audit.compare("parent_complete:" + key, values, evaluation[key])
    audit.compare(
        "parent_thresholds",
        parent["independently_recomputed_thresholds"],
        evaluation["calibration"]["thresholds"],
    )
    audit.check(
        "parent12_pass_preserved",
        len(evaluation["decision"]["gates"]) == 12
        and all(evaluation["decision"]["gates"].values())
        and evaluation["decision"]["engineering_replication_admitted"],
        True,
    )
    required = [
        STUDY + "/preparation/cases.json",
        STUDY + "/evaluation_v2/calibration.json",
        STUDY + "/evaluation_v2/local_case_metrics.json",
        MODE + "/local_case_statistics.json",
        CONFIG,
    ]
    for name in [*required, *ARRAYS]:
        audit.check("required_bound_input", name in audit.bindings, True)
    audit.declare(executed, script_sha)
    audit.verify()
    config = audit.receipt(CONFIG)
    audit.compare(
        "frozen_metric_config",
        {
            "bootstrap_seed": SEED,
            "bootstrap_draws": DRAWS,
            "metric_fields": 14,
            "area_fractions": [0.01, 0.05, 0.1],
            "arms": list(ARMS),
            "modes": list(MODES),
        },
        {
            key: config[key]
            for key in (
                "bootstrap_seed",
                "bootstrap_draws",
                "metric_fields",
                "area_fractions",
                "arms",
                "modes",
            )
        },
    )
    cases = audit.receipt(STUDY + "/preparation/cases.json")["cases"]
    calibration = audit.receipt(STUDY + "/evaluation_v2/calibration.json")
    parent_rows = audit.receipt(STUDY + "/evaluation_v2/local_case_metrics.json")["responses"]
    stored = audit.receipt(MODE + "/local_case_statistics.json")["rows"]
    by_case = {case["case_index"]: case for case in cases}
    audit.check(
        "all_new_case_keys",
        len(cases) == len(by_case) == 768 and set(by_case) == set(range(2000, 2768)),
        True,
    )
    declared = {(row["arm"], row["global_case"]): row for row in stored}
    audit.check("all1024_unique_stored_rows", len(stored) == len(declared) == 1024, True)
    prior_rows = {(row["method"], row["case"]): row for row in parent_rows}
    audit.check("all1024_parent_rows", len(parent_rows) == len(prior_rows) == 1024, True)
    audit.compare("calibration_metadata", calibration, evaluation["calibration"])
    audit.compare(
        "main_parent_reproduction",
        {
            "complete_scientific_blocks": 6,
            "thresholds": calibration["thresholds"],
            "parent_decision": evaluation["decision"],
            "parent_audit_all_passed": True,
            "parent_case_rows": 2816,
            "parent_response_rows": 1024,
        },
        main["reproduction"],
    )
    audit.check(
        "main_complete_rows",
        main["progress"]["completed"]
        and main["progress"]["rows"] == 1024
        and main["progress"]["scoring_started"]
        and main["progress"]["scoring_completed"],
        True,
    )
    maps = {
        arm: np.load(
            root / STUDY / ("evaluation_v2/probabilities_" + arm + ".npy"), mmap_mode="r", allow_pickle=False
        )
        for arm in ARMS
    }
    support, candidate = [
        np.load(root / STUDY / ("preparation/" + name + ".npy"), mmap_mode="r", allow_pickle=False)
        for name in ("support", "candidate")
    ]
    for array in maps.values():
        audit.check("map_geometry", array.shape == (768, 12, 2500) and array.dtype == np.float32, True)
    for array in (support, candidate):
        audit.check("mask_geometry", array.shape == (768, 12, 2500) and array.dtype == bool, True)
    calculated_rows = []
    for subject in range(128):
        audit.live()
        for offset, mode in enumerate(MODES):
            index = 2256 + 4 * subject + offset
            case, local = by_case[index], index - 2000
            control = 2256 + 4 * subject + (3 if case["nuisance"] else 2)
            counterpart = by_case[control]
            audit.check(
                "positive_metadata_identity",
                case["role"] == "evaluation"
                and case["target"]
                and case["mode"] == mode
                and case["role_subject_index"] == subject
                and case["mask_row"] == local
                and case["matched_control_case_index"] == control,
                True,
            )
            audit.check(
                "same_patient_control",
                not counterpart["target"]
                and counterpart["role"] == "evaluation"
                and counterpart["role_subject_index"] == subject
                and counterpart["patient_id"] == case["patient_id"]
                and case["baseline_sha256"] == counterpart["edited_sha256"]
                and case["nuisance_parameters"] == counterpart["nuisance_parameters"],
                True,
            )
            truth, prior = support[local], candidate[local]
            audit.check("native_support_hash", array_digest(truth) == case["support_sha256"], True)
            audit.check("native_candidate_hash", array_digest(prior) == case["candidate_sha256"], True)
            geo = geometry(truth, prior)
            for arm in ARMS:
                phases = {
                    "baseline": map_fields(truth, prior, maps[arm][control - 2000], geo),
                    "edited": map_fields(truth, prior, maps[arm][local], geo),
                }
                phases["change"] = {key: phases["edited"][key] - phases["baseline"][key] for key in FIELDS}
                row = {
                    "arm": arm,
                    "subject": subject,
                    "mode": mode,
                    "global_case": index,
                    "map_row": local,
                    "control_case": control,
                    "geometry": geo,
                    **phases,
                }
                audit.compare("all_case_geometry_and14_fields", row, declared[arm, index])
                old = prior_rows[arm, index]
                for phase, prefix in (("baseline", "baseline_site"), ("edited", "edited")):
                    for field, parent_field in (
                        ("full_grid_ap", "average_precision"),
                        ("top_1_precision", "top_one_percent_precision"),
                        ("top_1_recall", "top_one_percent_recall"),
                    ):
                        audit.compare(
                            "existing_parent_endpoint", phases[phase][field], old[prefix + "_" + parent_field]
                        )
                calculated_rows.append(row)
    result_blocks = aggregate(calculated_rows, audit)
    for name, values in result_blocks.items():
        audit.compare("complete_mode_summary:" + name, values, main[name])
    audit.compare("all14_fields", list(FIELDS), main["metric_fields"])
    audit.compare("all3_phases", list(PHASES), main["phases"])
    audit.compare(
        "prior_CPU_exact_authority",
        parent["cumulative_cpu_seconds"],
        main["prior_cumulative_cpu_seconds"],
        1e-9,
    )
    audit.compare(
        "main_CPU_charged_once",
        parent["cumulative_cpu_seconds"] + main["cpu_seconds"],
        main["cumulative_cpu_seconds"],
        1e-9,
    )
    audit.compare("GPU_unchanged", parent["global_gpu_wall_seconds"], main["global_gpu_wall_seconds"], 1e-9)
    audit.check("main_budget", max(main["cpu_seconds"], main["wall_seconds"]) < 120, True)
    audit.check(
        "no_new_advancement_or_payload_action",
        main["new_advancement_rule"] is None
        and main["parent_decision_preserved"]
        and not any(
            main[key]
            for key in (
                "raw_waveform_reads",
                "processed_waveform_reads",
                "checkpoint_payload_reads",
                "model_inferences",
                "optimizer_updates",
                "gpu_used",
            )
        ),
        True,
    )
    audit.verify()
    audit.live()
    family_bytes, whole_bytes = bytes_under(root / MODE), bytes_under(root / BASE)
    audit.check("mode_family_storage", family_bytes + 1024**2 < 16 * 1024**2, True)
    audit.check("whole_stream_storage", whole_bytes + 1024**2 < 4 * 1024**3, True)
    cpu, wall = time.process_time() - audit.cpu, time.perf_counter() - audit.wall
    result = {
        "all_passed": not audit.mismatches,
        "mismatch_count": len(audit.mismatches),
        "mismatches": audit.mismatches,
        "comparison_checks": audit.checks,
        "main_summary_sha256": main_sha,
        "parent_audit_sha256": PARENT_SHA,
        "source_script_sha256": script_sha,
        "source_sha256": {executed: script_sha},
        "input_sha256": audit.verified,
        "deferred_payload_scope_exceptions": audit.exceptions,
        "independently_recomputed_complete_summary": result_blocks,
        "case_rows_checked": len(calculated_rows),
        "metric_fields": list(FIELDS),
        "cpu_seconds": cpu,
        "wall_seconds": wall,
        "main_plus_audit_cpu_seconds": main["cpu_seconds"] + cpu,
        "main_plus_audit_wall_seconds": main["wall_seconds"] + wall,
        "prior_cumulative_cpu_seconds": main["cumulative_cpu_seconds"],
        "cumulative_cpu_seconds": main["cumulative_cpu_seconds"] + cpu,
        "global_gpu_wall_seconds": main["global_gpu_wall_seconds"],
        "mode_bytes_before_summary": family_bytes,
        "stream_bytes_before_summary": whole_bytes,
        "raw_waveform_reads": 0,
        "processed_waveform_reads": 0,
        "checkpoint_payload_reads": 0,
        "model_inferences": 0,
        "optimizer_updates": 0,
        "gpu_used": False,
        "numerical_threads": 1,
        "parent_decision_preserved": True,
        "limitations": [
            "No waveform regeneration or new clinical truth; only saved-output mathematics.",
            "Patient intervals condition on fixed fitted models; no training-seed inference.",
        ],
    }
    with (destination / "summary.json").open("x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    return result


def self_test() -> None:  # noqa: C901
    """Artificial-only tests for ranking, projections, scope, exact ties and paired uncertainty."""
    if not math.isclose(average_precision(np.array([1, 0, 1, 0], bool), np.array([1, 0.8, 0.8, 0.1])), 5 / 6):
        raise AssertionError("Tied AP")
    precision, recall = coverage(np.array([True] * 50 + [False] * 50), np.ones(100), 1)
    if precision != 0.5 or recall != 0.01:
        raise AssertionError("Fractional top-area ties")
    truth = np.zeros((12, 2500), bool)
    truth[0, 100:200] = True
    prior = np.zeros_like(truth)
    prior[:, 50:250] = True
    score = np.full(truth.shape, 0.1, np.float32)
    score[truth] = 0.9
    geo = geometry(truth, prior)
    values = map_fields(truth, prior, score, geo)
    if set(values) != set(FIELDS) or any(
        values[key] != 1 for key in ("full_grid_ap", "lead_ap", "temporal_ap", "oracle_temporal_ap")
    ):
        raise AssertionError("Independent projections")
    if not math.isclose(values["mass_T"] + values["mass_U"] + values["mass_O"], 1):
        raise AssertionError("Probability allocation")
    outside_decoy = score.copy()
    outside_decoy[1, ~prior[1]] = 0.99
    if map_fields(truth, prior, outside_decoy, geo)["lead_ap"] != 1:
        raise AssertionError("Lead projection must ignore outside-candidate decoys")
    unequal_truth = truth.copy()
    unequal_truth[1, 400:600] = True
    wide_prior = np.zeros_like(truth)
    wide_prior[:, 50:650] = True
    unequal_score = score.copy()
    unequal_score[1] = 0.9
    unequal_score[1, unequal_truth[1]] = 0.1
    oracle = map_fields(unequal_truth, wide_prior, unequal_score, geometry(unequal_truth, wide_prior))
    if not math.isclose(oracle["oracle_temporal_ap"], 0.54):
        raise AssertionError("Oracle temporal AP must weight true leads equally")
    broad_truth = np.zeros_like(truth)
    broad_truth[:6, 100:600] = True
    if geometry(broad_truth, wide_prior)["attainable_recall"] != {"1": 0.1, "5": 0.5, "10": 1.0}:
        raise AssertionError("Broad-support coverage ceilings")
    if geo["attainable_recall"]["1"] != 1 or geo["area_budgets"]["10"] != 3000:
        raise AssertionError("Geometry ceilings")
    try:
        map_fields(truth, prior, np.zeros_like(score), geo)
    except ValueError:
        pass
    else:
        raise AssertionError("Zero-mass guard")
    for name in (STUDY + "/preparation/backgrounds.npy", STUDY + "/training/RB.pt", "data/raw/clinical.json"):
        if permitted(name):
            raise AssertionError("Forbidden scope")
    if len(ARRAYS) != 6 or not all(permitted(name) for name in ARRAYS):
        raise AssertionError("Six-array scope")
    value = interval(np.full(128, 0.25), draws())
    if any(value[key] != 0.25 for key in ("difference", "ci_low", "ci_high")):
        raise AssertionError("Paired interval")
    print("Artificial independent mode audit checks passed; no study payloads opened")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--root", type=Path, default=Path("/tmp/ecg-wt-ssl-synthetic"))
    parser.add_argument("--main-sha")
    parser.add_argument("--script-sha")
    arguments = parser.parse_args()
    if arguments.self_test:
        self_test()
    else:
        if not arguments.main_sha or not arguments.script_sha:
            parser.error("Execution requires coordinator-frozen main and script digests")
        start_cpu, start_wall = time.process_time(), time.perf_counter()
        try:
            receipt = run(arguments.root, arguments.main_sha, arguments.script_sha)
        except Exception as error:
            path = arguments.root / MODE / "independent_audit_v1/failure.json"
            if path.parent.exists() and not path.exists():
                with path.open("x") as stream:
                    json.dump(
                        {
                            "all_passed": False,
                            "failure_type": type(error).__name__,
                            "failure_detail": str(error),
                            "source_script_sha256": arguments.script_sha,
                            "main_summary_sha256": arguments.main_sha,
                            "cpu_seconds": time.process_time() - start_cpu,
                            "wall_seconds": time.perf_counter() - start_wall,
                            "raw_waveform_reads": 0,
                            "processed_waveform_reads": 0,
                            "checkpoint_payload_reads": 0,
                            "model_inferences": 0,
                            "optimizer_updates": 0,
                            "gpu_used": False,
                        },
                        stream,
                        indent=2,
                    )
                    stream.write("\n")
            raise
        print(
            json.dumps(
                {
                    key: receipt[key]
                    for key in (
                        "all_passed",
                        "mismatch_count",
                        "comparison_checks",
                        "cpu_seconds",
                        "wall_seconds",
                    )
                }
            )
        )
