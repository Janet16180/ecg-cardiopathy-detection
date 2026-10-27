"""Descriptive counterfactual response from frozen saved synthetic probability maps only."""

from __future__ import annotations

import json
import os
import subprocess
import time
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from ecg_experiment import ROOT
from ecg_experiment.files import sha256_file
from ecg_experiment.paths import to_stored
from ecg_experiment.ssl_synthetic_data import load_bank
from ecg_experiment.ssl_synthetic_evaluation import (
    ARMS,
    _merge_hashes,
    _read_receipt,
    verify_inference_hashes,
)
from ecg_experiment.ssl_synthetic_metrics import paired_subject_difference, positive_map_metrics
from ecg_experiment.ssl_synthetic_preparation import BASE, write_new_json

ORIGINAL = BASE / "pilot_v1/evaluation"
DESTINATION = BASE / "pilot_v1/counterfactual_audit_v1"
NEW_SOURCES = (
    "ecg_experiment/ssl_synthetic_counterfactual.py",
    "scripts/validation/audit_ssl_synthetic_counterfactual.py",
    "tests/test_ssl_synthetic_counterfactual.py",
    "docs/ssl-synthetic-localization-counterfactual.md",
)
RESPONSE_METRICS = (
    "edited_average_precision",
    "baseline_site_average_precision",
    "average_precision_change",
    "mean_probability_change_T",
    "mean_probability_change_U",
    "mean_probability_change_O",
    "selectivity_contrast",
    "edited_top_one_percent_precision",
    "baseline_site_top_one_percent_precision",
    "top_one_percent_precision_change",
    "edited_top_one_percent_recall",
    "baseline_site_top_one_percent_recall",
    "top_one_percent_recall_change",
)


def require_original_audit(audit: dict[str, Any], original_digest: str) -> None:
    """Require exact independent original-metric reproduction before the new comparison.

    Parameters
    ----------
    audit : dict
        Independently executed audit receipt, not a model-performance claim.
    original_digest : str
        Current original evaluation summary SHA-256.
    """
    path = to_stored(ORIGINAL / "summary.json")
    if (
        not audit["all_passed"]
        or audit["mismatch_count"] != 0
        or audit["input_sha256"].get(path) != original_digest
    ):
        raise ValueError("Original independent reproduction is absent, failed or bound to another evaluation")


def match_counterparts(bank: dict[str, np.ndarray], metadata: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Verify the complete latent-matched byte-identical counterpart set before diagnostics.

    Parameters
    ----------
    bank : dict
        Original synthetic arrays, never real waveform payloads.
    metadata : list of dict
        Original label-only latent and nuisance identities.

    Returns
    -------
    list of dict
        Positive-case/control indices, subject/mode and exact region denominators.
    """
    subjects = np.unique(bank["subjects"])
    if len(subjects) != 256 or len(metadata) != 1024:
        raise ValueError("Expected every original subject and all four variants")
    expected = Counter(
        {
            ("positive", "focal"): 1,
            ("positive", "persistent"): 1,
            ("sham", "focal"): 1,
            ("nuisance", "focal"): 1,
        }
    )
    for subject in subjects:
        indices = np.flatnonzero(bank["subjects"] == subject)
        variants = Counter(zip(bank["kinds"][indices].tolist(), bank["modes"][indices].tolist(), strict=True))
        if variants != expected:
            raise ValueError("Missing or duplicated original subject variant")
    pairs = []
    for index in np.flatnonzero(bank["kinds"] == "positive"):
        positive = metadata[index]
        subject = int(bank["subjects"][index])
        kind = "nuisance" if positive["nuisance"] else "sham"
        candidates = np.flatnonzero((bank["subjects"] == subject) & (bank["kinds"] == kind))
        if len(candidates) != 1:
            raise ValueError("Missing or duplicate latent-matched control")
        control = int(candidates[0])
        other = metadata[control]
        _check_pair_identity(positive, other, subject)
        if bank["baselines"][index].tobytes() != bank["signals"][control].tobytes():
            raise ValueError("Baseline is not byte-identical to its saved observed control")
        truth = bank["support"][index]
        delta = bank["signals"][index].astype(np.float64) - bank["baselines"][index].astype(np.float64)
        if not np.array_equal(truth, np.abs(delta) > 0.005):
            raise ValueError("Original physical-floor support changed")
        counts = region_counts(truth, bank["prior"][index])
        pairs.append(
            {
                "case": int(index),
                "control_case": control,
                "subject": subject,
                "mode": str(bank["modes"][index]),
                "nuisance": positive["nuisance"],
                **counts,
            }
        )
    if len(pairs) != 512 or any(
        sorted(row["mode"] for row in pairs if row["subject"] == int(subject)) != ["focal", "persistent"]
        for subject in subjects
    ):
        raise ValueError("Positive variants cannot be omitted or duplicated")
    return pairs


def _check_pair_identity(positive: dict[str, Any], control: dict[str, Any], subject: int) -> None:
    if positive["subject"] != subject or control["subject"] != subject or positive["seed"] != control["seed"]:
        raise ValueError("Counterpart latent identity differs")
    if (
        positive["parameters"]["latent"] != control["parameters"]["latent"]
        or positive["parameters"]["nuisance"] != control["parameters"]["nuisance"]
        or positive["parameters"]["profile"] != control["parameters"]["profile"]
    ):
        raise ValueError("Counterpart background, nuisance or profile differs")


def region_counts(truth: np.ndarray, prior: np.ndarray) -> dict[str, int]:
    """Require disjoint nonempty target, unedited candidate and outside-candidate regions.

    Parameters
    ----------
    truth, prior : np.ndarray
        Exact intervention support and original intervention-independent candidate domain.

    Returns
    -------
    dict
        T/U/O cell counts, each strictly positive.
    """
    if truth.shape != prior.shape or np.any(truth & ~prior):
        raise ValueError("Target and candidate regions do not form the declared partition")
    counts = {
        "T_cells": int(truth.sum()),
        "U_cells": int((prior & ~truth).sum()),
        "O_cells": int((~prior).sum()),
    }
    if min(counts.values()) <= 0:
        raise ValueError("Empty counterfactual region; cases cannot be selectively removed")
    return counts


def response_metrics(
    truth: np.ndarray, prior: np.ndarray, edited: np.ndarray, baseline: np.ndarray
) -> dict[str, float]:
    """Measure same-mask site agreement and exact map response, without fitting or inference.

    Parameters
    ----------
    truth, prior : np.ndarray
        Fixed original support/candidate maps.
    edited, baseline : np.ndarray
        Saved same-arm probability maps for the edited input and byte-identical counterpart.

    Returns
    -------
    dict
        T/U/O mean changes, selectivity and unchanged full-map/tied-area metric differences.
    """
    region_counts(truth, prior)
    if edited.shape != truth.shape or baseline.shape != truth.shape:
        raise ValueError("Saved maps and original regions must share coordinates")
    if not np.isfinite(edited).all() or not np.isfinite(baseline).all():
        raise ValueError("Nonfinite saved counterfactual map")
    delta = edited.astype(np.float64) - baseline.astype(np.float64)
    target = float(delta[truth].mean())
    unedited = float(delta[prior & ~truth].mean())
    result = {
        "mean_probability_change_T": target,
        "mean_probability_change_U": unedited,
        "mean_probability_change_O": float(delta[~prior].mean()),
        "selectivity_contrast": target - unedited,
    }
    first, second = positive_map_metrics(truth, edited), positive_map_metrics(truth, baseline)
    for name in ("average_precision", "top_one_percent_precision", "top_one_percent_recall"):
        result[f"edited_{name}"] = first[name]
        result[f"baseline_site_{name}"] = second[name]
        result[f"{name}_change"] = first[name] - second[name]
    return result


def _interval(subjects: np.ndarray, values: np.ndarray, config: dict[str, Any]) -> dict[str, Any]:
    interval = paired_subject_difference(
        subjects,
        values,
        np.zeros_like(values),
        draws=config["bootstrap_draws"],
        seed=config["bootstrap_seed"],
    )
    return {"mean": interval.pop("difference"), **interval}


def summarize_responses(rows: list[dict[str, Any]], config: dict[str, Any]) -> dict[str, Any]:
    """Macro-average variants within subjects and retain shared bootstrap draws for every metric.

    Parameters
    ----------
    rows : list of dict
        All arm/positive-case response diagnostics for one profile.
    config : dict
        Original 2,000-draw/seed bootstrap configuration.

    Returns
    -------
    dict
        Arm estimates/intervals by mode and exploratory paired S-minus-R changes.
    """
    arms, differences = {}, {}
    for mode in ("all", "focal", "persistent"):
        selected = [row for row in rows if mode == "all" or row["mode"] == mode]
        by_arm = {
            arm: sorted(
                (row for row in selected if row["method"] == arm),
                key=lambda row: (row["subject"], row["case"]),
            )
            for arm in ARMS
        }
        for arm, cases in by_arm.items():
            subjects = np.array([row["subject"] for row in cases])
            arms.setdefault(arm, {})[mode] = {
                name: _interval(subjects, np.array([row[name] for row in cases]), config)
                for name in RESPONSE_METRICS
            }
        if [(row["subject"], row["case"]) for row in by_arm["S"]] != [
            (row["subject"], row["case"]) for row in by_arm["R"]
        ]:
            raise ValueError("Exploratory paired denominators disagree")
        differences[mode] = {
            name: paired_subject_difference(
                np.array([row["subject"] for row in by_arm["S"]]),
                np.array([row[name] for row in by_arm["S"]]),
                np.array([row[name] for row in by_arm["R"]]),
                draws=config["bootstrap_draws"],
                seed=config["bootstrap_seed"],
            )
            for name in (
                "average_precision_change",
                "selectivity_contrast",
                "mean_probability_change_T",
                "mean_probability_change_U",
                "mean_probability_change_O",
            )
        }
    return {"arms": arms, "exploratory_s_minus_r": differences}


def nuisance_metrics(sham: np.ndarray, nuisance: np.ndarray, threshold: float) -> dict[str, Any]:
    """Compare nuisance and sham maps at the original fixed recording threshold.

    Parameters
    ----------
    sham, nuisance : np.ndarray
        Existing same-latent zero-target probability maps.
    threshold : float
        Original validation-frozen arm threshold, never refitted.

    Returns
    -------
    dict
        Mean absolute full-map response, original flags and signed flag change.
    """
    if sham.shape != nuisance.shape or not np.isfinite(sham).all() or not np.isfinite(nuisance).all():
        raise ValueError("Invalid nuisance/sham saved maps")
    first, second = bool(sham.max() > threshold), bool(nuisance.max() > threshold)
    return {
        "mean_absolute_nuisance_change": float(np.abs(nuisance.astype(np.float64) - sham).mean()),
        "sham_flag": first,
        "nuisance_flag": second,
        "flag_change": int(second) - int(first),
    }


def _profile_rows(
    bank: dict[str, np.ndarray],
    maps: dict[str, np.ndarray],
    pairs: list[dict[str, Any]],
    thresholds: dict[str, float],
    original_rows: list[dict[str, Any]],
) -> tuple[list, list]:
    lookup = {(row["method"], row["case"]): row for row in original_rows}
    rows, controls = [], []
    for pair in pairs:
        index, control = pair["case"], pair["control_case"]
        for arm in ARMS:
            metrics = response_metrics(
                bank["support"][index], bank["prior"][index], maps[arm][index], maps[arm][control]
            )
            for name in ("average_precision", "top_one_percent_precision", "top_one_percent_recall"):
                if metrics[f"edited_{name}"] != lookup[(arm, index)][name]:
                    raise ValueError("Original edited-map metric failed exact reproduction")
            rows.append({**pair, "method": arm, **metrics})
    for subject in np.unique(bank["subjects"]):
        sham = int(np.flatnonzero((bank["subjects"] == subject) & (bank["kinds"] == "sham"))[0])
        noisy = int(np.flatnonzero((bank["subjects"] == subject) & (bank["kinds"] == "nuisance"))[0])
        for arm in ARMS:
            controls.append(
                {
                    "subject": int(subject),
                    "method": arm,
                    **nuisance_metrics(maps[arm][sham], maps[arm][noisy], thresholds[arm]),
                }
            )
    return rows, controls


def _summarize_controls(rows: list[dict[str, Any]], config: dict[str, Any]) -> dict[str, Any]:
    result = {}
    for arm in ARMS:
        selected = [row for row in rows if row["method"] == arm]
        subjects = np.array([row["subject"] for row in selected])
        result[arm] = {
            name: _interval(subjects, np.array([row[name] for row in selected], dtype=float), config)
            for name in ("mean_absolute_nuisance_change", "flag_change")
        }
        result[arm]["sham_flag_count"] = sum(row["sham_flag"] for row in selected)
        result[arm]["nuisance_flag_count"] = sum(row["nuisance_flag"] for row in selected)
    return result


def _admission(original_audit: Path) -> tuple[dict, dict, dict]:
    original = _read_receipt(ORIGINAL / "summary.json")
    audit = _read_receipt(original_audit)
    require_original_audit(audit, sha256_file(ORIGINAL / "summary.json"))
    if original["decision"]["engineering_advancement_passed"]:
        raise ValueError("Original failed pilot verdict changed")
    sources = dict(original["source_sha256"])
    _merge_hashes(sources, {name: sha256_file(ROOT / name) for name in NEW_SOURCES})
    _merge_hashes(sources, audit.get("source_sha256", {}))
    inputs = dict(original["input_sha256"])
    _merge_hashes(inputs, audit.get("input_sha256", {}))
    _merge_hashes(
        inputs, {to_stored(ORIGINAL / name): digest for name, digest in original["output_sha256"].items()}
    )
    _merge_hashes(
        inputs,
        {
            to_stored(ORIGINAL / "summary.json"): sha256_file(ORIGINAL / "summary.json"),
            to_stored(original_audit): sha256_file(original_audit),
        },
    )
    verify_inference_hashes(sources | inputs)
    return original, sources, inputs


def _saved_inputs() -> tuple[dict, dict, dict]:
    banks, probabilities, pairings = {}, {}, {}
    for profile in ("B", "A"):
        directory = ORIGINAL / f"{profile}_bank"
        banks[profile] = load_bank(directory)
        metadata = json.loads((directory / "metadata.json").read_text())
        pairings[profile] = match_counterparts(banks[profile], metadata)
        probabilities[profile] = {
            arm: np.load(ORIGINAL / f"{profile}_prob_{arm}.npy", mmap_mode="r", allow_pickle=False)
            for arm in ARMS
        }
        for values in probabilities[profile].values():
            if values.shape != (1024, 12, 2500) or values.dtype != np.float32:
                raise ValueError("Saved probability grid/dtype changed")
            if not np.isfinite(values).all() or values.min() < 0 or values.max() > 1:
                raise ValueError("Invalid saved probability values")
    return banks, probabilities, pairings


def audit_counterfactual(original_audit: Path) -> dict[str, Any]:
    """Execute one source-bound existing-map diagnostic after exact original-metric reproduction.

    Parameters
    ----------
    original_audit : Path
        Passed independent original reproduction receipt binding the exact original summary.

    Returns
    -------
    dict
        Descriptive audit only; original calibration/advancement and model states are untouched.
    """
    if DESTINATION.exists():
        raise FileExistsError("Counterfactual audit must be new; failures cannot be silently restarted")
    wall, cpu = time.perf_counter(), time.process_time()
    sources = {name: sha256_file(ROOT / name) for name in NEW_SOURCES}
    inputs = {}
    progress = {"verified_positive_pairs": 0, "completed_profiles": []}
    DESTINATION.mkdir(parents=True, exist_ok=False)
    try:
        if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
            raise ValueError("Counterfactual audit requires CUDA hidden and no model inference")
        thread_names = ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS")
        threads = {name: os.environ.get(name) for name in thread_names}
        if any(value != "1" for value in threads.values()):
            raise ValueError("Counterfactual audit requires one numerical thread")
        original, sources, inputs = _admission(original_audit)
        config = json.loads((ROOT / "configs/ssl_synthetic_pilot_v1.json").read_text())
        reproduction = json.loads(original_audit.read_text())
        prior_cpu = original["total_preparation_training_evaluation_cpu_seconds"] + reproduction.get(
            "cpu_seconds", 0
        )
        cpu_deadline = cpu + config["cpu_seconds_ceiling"] - prior_cpu
        if time.process_time() >= cpu_deadline:
            raise TimeoutError("Cumulative protocol CPU ceiling exhausted before the diagnostic")
        banks, probabilities, pairings = _saved_inputs()
        progress["verified_positive_pairs"] = sum(map(len, pairings.values()))
        write_new_json(
            DESTINATION / "admission.json",
            {
                "created_at_utc": datetime.now(UTC).isoformat(),
                "source_sha256": sources,
                "input_sha256": inputs,
                "progress": progress,
            },
        )
        summaries = {}
        for profile in ("B", "A"):
            if time.process_time() >= cpu_deadline:
                raise TimeoutError("Cumulative protocol CPU ceiling exhausted")
            prior_rows = json.loads((ORIGINAL / f"{profile}_local_case_metrics.json").read_text())["rows"]
            rows, controls = _profile_rows(
                banks[profile],
                probabilities[profile],
                pairings[profile],
                original["calibration"]["thresholds"],
                prior_rows,
            )
            write_new_json(
                DESTINATION / f"{profile}_local_case_responses.json", {"rows": rows, "controls": controls}
            )
            summaries[profile] = {
                "subjects": 256,
                "positive_cases": len(pairings[profile]),
                "arm_positive_case_comparisons": len(rows),
                "arm_control_comparisons": len(controls),
                "region_cell_counts": {
                    name: dict(
                        zip(
                            ("minimum", "median", "maximum"),
                            np.quantile([pair[name] for pair in pairings[profile]], [0, 0.5, 1]).tolist(),
                            strict=True,
                        )
                    )
                    for name in ("T_cells", "U_cells", "O_cells")
                },
                **summarize_responses(rows, config),
                "nuisance_controls": _summarize_controls(controls, config),
            }
            progress["completed_profiles"].append(profile)
        verify_inference_hashes(sources | inputs)
        if time.process_time() >= cpu_deadline:
            raise TimeoutError("Cumulative protocol CPU ceiling exhausted")
        output_hashes = {path.name: sha256_file(path) for path in DESTINATION.iterdir() if path.is_file()}
        size = sum(path.stat().st_size for path in DESTINATION.iterdir() if path.is_file())
        if size > 99 * 1024**2:
            raise ValueError("Counterfactual output ceiling/reserve exhausted")
        summary = {
            "audit_completed": True,
            "created_at_utc": datetime.now(UTC).isoformat(),
            "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
            "source_sha256": sources,
            "input_sha256": inputs,
            "output_sha256": output_hashes,
            "progress": progress,
            "profiles": summaries,
            "wall_seconds": time.perf_counter() - wall,
            "cpu_seconds": time.process_time() - cpu,
            "prior_protocol_and_reproduction_cpu_seconds": prior_cpu,
            "cumulative_cpu_seconds": prior_cpu + time.process_time() - cpu,
            "new_output_bytes_before_summary": size,
            "new_output_ceiling_bytes": 100 * 1024**2,
            "original_advancement_passed": False,
            "new_advancement_rule": None,
            "new_model_inferences": 0,
            "model_checkpoints_deserialized": 0,
            "gpu_used": False,
            "numerical_thread_environment": threads,
            "real_waveform_payloads_opened": 0,
            "interpretation": "Spent-development descriptive diagnostic",
        }
        write_new_json(DESTINATION / "summary.json", summary)
        return summary
    except Exception as error:
        write_new_json(
            DESTINATION / "failure.json",
            {
                "audit_completed": False,
                "created_at_utc": datetime.now(UTC).isoformat(),
                "source_sha256": sources,
                "input_sha256": inputs,
                "progress": progress,
                "failure_type": type(error).__name__,
                "failure_detail": str(error),
                "wall_seconds": time.perf_counter() - wall,
                "cpu_seconds": time.process_time() - cpu,
            },
        )
        raise
