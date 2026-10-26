"""Matched-patient-union controls for contextual reference contribution consistency."""

from __future__ import annotations

import json
import math
import subprocess
import time
from itertools import combinations
from pathlib import Path

import numpy as np

from ecg_experiment import ROOT
from ecg_experiment.clustering.bank_inputs import committed_sources, read_bound
from ecg_experiment.clustering.native_cache import runtime_fingerprint
from ecg_experiment.clustering.reference_geometry import check_budget
from ecg_experiment.files import sha256_file, write_json_atomic
from ecg_experiment.localization.contextual_margin import summary
from ecg_experiment.localization.contribution_inputs import replay_cached_inputs
from ecg_experiment.localization.reference_contribution_stability import (
    aggregate_patients,
    contextual_geometry,
    family_metrics,
    fixed_contributions,
    nested_error,
    pair_metrics,
    prototypes_and_manifest,
)
from ecg_experiment.localization.reference_contribution_stability import (
    primary_decision as predecessor_decision,
)
from ecg_experiment.native_reference_ntt05 import unit_vector
from ecg_experiment.paths import to_stored
from ecg_experiment.same_wave_ntt04 import FINDINGS

PREVIOUS_SHA = "009eacb3c72b0dcfe85e155a9036542cc504e788ddd7179e097860fe4f4e8f64"
METHODS = ("A", "B", "C")
PAIRS = (("A", "B"), ("A", "C"), ("B", "C"))
SOURCES = (
    "docs/research/localization/ntt10-matched-union/protocol.md",
    "ecg_experiment/localization/matched_union_control.py",
    "scripts/experiments/localization/run_ntt10.py",
    "tests/localization/test_matched_union_control.py",
)


def array_error(actual: np.ndarray, expected: np.ndarray) -> float:
    """Check exact array shape/null masks and return finite maximum numerical error.

    Parameters
    ----------
    actual, expected : np.ndarray
        Reconstructed and source-bound predecessor arrays.

    Returns
    -------
    float
        Maximum absolute difference on finite entries.
    """
    if (
        actual.shape != expected.shape
        or np.isinf(actual).any()
        or np.isinf(expected).any()
        or not np.array_equal(np.isnan(actual), np.isnan(expected))
    ):
        raise ValueError("Predecessor array shape or null mask differs")
    finite = np.isfinite(actual)
    return float(np.max(np.abs(actual[finite].astype(float) - expected[finite]))) if finite.any() else 0.0


def replay09(data: dict, receipt: dict, output: Path, deadline: float) -> tuple:
    """Reproduce all predecessor prototypes, decompositions, pairs and patient summaries.

    Parameters
    ----------
    data, receipt : dict
        Role-verified original cached inputs and bound NTT-09 receipt.
    output : Path
        Local destination for the replayed frozen contrast/bank manifest.
    deadline : float
        Monotonic execution deadline.

    Returns
    -------
    tuple
        Replay diagnostics, prototypes, fixed contrasts, exposure and original geometry.
    """
    base = ROOT / "outputs/ntt09_reference_contribution_stability"
    prototypes, contrasts, exposure = prototypes_and_manifest(data, output)
    manifest = json.loads((output / "fixed_manifest_local.json").read_text())
    maximum = nested_error(manifest, json.loads((base / "fixed_manifest_local.json").read_text()))
    patients, geometries = [], []
    with np.load(base / "contributions_local.npz", allow_pickle=False) as saved:
        for key, values in prototypes.items():
            maximum = max(maximum, array_error(values, saved[key + "_prototypes"]))
        for index, target in enumerate(data["selection"]["targets"]):
            check_budget(deadline)
            geometry = contextual_geometry(data, target)
            geometries.append(geometry)
            contrast = contrasts[str(index)]
            patient = {"target_index": index, "ecg_id": target["ecg_id"], "patient_id": target["patient_id"]}
            for family in ("single", "ensemble"):
                result = fixed_contributions(
                    geometry, prototypes[family], (contrast["winner"], contrast["runner"])
                )
                pairs, arrays = pair_metrics(result["beat"])
                patient[family] = {"pairs": pairs, "metrics": family_metrics(result, geometry)}
                for key, values in result.items():
                    if isinstance(values, np.ndarray):
                        maximum = max(maximum, array_error(values, saved[f"target_{index}_{family}_{key}"]))
                for key, values in arrays.items():
                    maximum = max(maximum, array_error(values, saved[f"target_{index}_{family}_pairs_{key}"]))
            for key, values in geometry.items():
                if isinstance(values, np.ndarray):
                    maximum = max(maximum, array_error(values, saved[f"target_{index}_geometry_{key}"]))
            patients.append(patient)
    errors = [
        maximum,
        nested_error(patients, json.loads((base / "patients_local.json").read_text())),
        nested_error(aggregate_patients(patients), receipt["aggregate"]),
        nested_error(predecessor_decision(patients), receipt["decision"]),
        nested_error(exposure, receipt["support_exposure"]),
    ]
    if len(patients) != 60 or max(errors) > 1e-12:
        raise ValueError("NTT-09 exact predecessor replay failed")
    return (
        {
            "patients": 60,
            "decompositions": 3600,
            "bank_pairs": 76200,
            "pair_metrics": 4,
            "maximum_array_error": maximum,
            "maximum_complete_summary_error": max(errors[1:]),
            "all_null_masks_exact": True,
            "complete_primary_reproduced": True,
            "native_predecessor_replay": data["replay"],
        },
        prototypes,
        contrasts,
        exposure,
        geometries,
    )


def union_prototypes(vectors: np.ndarray, bank_indices: np.ndarray) -> dict:
    """Build three matched-union prototypes and analytically reconstruct bank weighting.

    Parameters
    ----------
    vectors : np.ndarray
        Canonically ordered unique unit patient vectors.
    bank_indices : np.ndarray
        Five banks with three distinct indices each, whose union covers every vector.

    Returns
    -------
    dict
        A/B/C prototypes, analytic A, normalized weights, occurrence counts and bank norms.
    """
    if (
        vectors.ndim != 2
        or not np.isfinite(vectors).all()
        or not np.allclose(np.linalg.norm(vectors, axis=1), 1, atol=1e-12, rtol=0)
        or bank_indices.shape != (5, 3)
        or not np.issubdtype(bank_indices.dtype, np.integer)
        or np.any(bank_indices < 0)
        or np.any(bank_indices >= len(vectors))
        or any(len(np.unique(row)) != 3 for row in bank_indices)
        or not np.array_equal(np.unique(bank_indices), np.arange(len(vectors)))
    ):
        raise ValueError("Expected five complete banks over the identical finite unit-vector union")
    sums = vectors[bank_indices].sum(axis=1)
    norms = np.linalg.norm(sums, axis=1)
    if np.any(norms <= 0) or not np.isfinite(norms).all():
        raise ValueError("Undefined bank normalization")
    counts = np.bincount(bank_indices.ravel(), minlength=len(vectors))
    presence = np.zeros((5, len(vectors)))
    np.put_along_axis(presence, bank_indices, 1, axis=1)
    alpha = (presence / norms[:, None]).sum(axis=0)
    weights = {
        "A": alpha / alpha.sum(),
        "B": np.full(len(vectors), 1 / len(vectors)),
        "C": counts / counts.sum(),
    }
    prototypes = {
        "A": unit_vector((sums / norms[:, None]).mean(axis=0)),
        "B": unit_vector(vectors.sum(axis=0)),
        "C": unit_vector(np.sum(counts[:, None] * vectors, axis=0)),
    }
    analytic = unit_vector(np.sum(alpha[:, None] * vectors, axis=0))
    error = float(np.max(np.abs(prototypes["A"] - analytic)))
    if error > 1e-12:
        raise ValueError("Analytic bank-normalization coefficient identity failed")
    return {
        "prototypes": prototypes,
        "analytic_A": analytic,
        "weights": weights,
        "occurrence_counts": counts,
        "bank_norms": norms,
        "maximum_analytic_error": error,
    }


def union_diagnostics(signatures: list, prototypes_b: np.ndarray) -> dict:
    """Count repeated union signatures and require identical canonical uniform prototypes.

    Parameters
    ----------
    signatures : list
        Ten blocks of three canonical patient-ID union tuples.
    prototypes_b : np.ndarray
        Uniform-union unit prototypes in the same block/finding order.

    Returns
    -------
    dict
        Aggregate-safe duplicate counts and exact prototype-identity checks.
    """
    findings = {}
    for finding in range(3):
        identical = [
            (a, b) for a, b in combinations(range(10), 2) if signatures[a][finding] == signatures[b][finding]
        ]
        if any(not np.array_equal(prototypes_b[a, finding], prototypes_b[b, finding]) for a, b in identical):
            raise ValueError("Identical canonical unions produced different uniform prototypes")
        findings[str(finding)] = {
            "unique_unions": len({row[finding] for row in signatures}),
            "identical_union_block_pairs": len(identical),
        }
    return {
        "per_finding": findings,
        "unique_three_finding_union_triples": len(set(signatures)),
        "identical_union_triple_block_pairs": sum(
            signatures[a] == signatures[b] for a, b in combinations(range(10), 2)
        ),
        "repeated_union_B_prototypes_exact": True,
    }


def matched_prototypes(data: dict, previous: dict, output: Path) -> tuple:
    """Freeze exact support unions, normalized weights and all matched prototype methods.

    Parameters
    ----------
    data, previous : dict
        Verified original features and replayed NTT-09 prototypes.
    output : Path
        Local manifest destination.

    Returns
    -------
    tuple
        Matched prototypes, analytic A, local coefficient records and aggregate diagnostics.
    """
    manifest = json.loads((output / "fixed_manifest_local.json").read_text())
    blocks = manifest["banks"]["banks"]
    arrays = {method: [] for method in METHODS}
    analytic, records, signatures, maximum = [], [], [], 0.0
    for block in range(10):
        current = {method: [] for method in METHODS}
        current_analytic, current_signatures = [], []
        for finding_index, finding in enumerate(FINDINGS):
            banks = [row["supports"][finding] for row in blocks[block * 5 : block * 5 + 5]]
            entries = {row["patient_id"]: row for bank in banks for row in bank}
            ordered = sorted(entries.values(), key=lambda row: (int(row["patient_id"]), int(row["ecg_id"])))
            indices = {row["patient_id"]: i for i, row in enumerate(ordered)}
            vectors = np.stack(
                [unit_vector(data["features"][row["ecg_id"]]["native_qrs"]) for row in ordered]
            )
            bank_indices = np.array([[indices[row["patient_id"]] for row in bank] for bank in banks])
            result = union_prototypes(vectors, bank_indices)
            original_error = float(
                np.max(np.abs(result["prototypes"]["A"] - previous["ensemble"][block, finding_index]))
            )
            analytic_original_error = float(
                np.max(np.abs(result["analytic_A"] - previous["ensemble"][block, finding_index]))
            )
            if max(original_error, analytic_original_error) > 1e-12:
                raise ValueError("Matched-union A differs from original NTT-09 ensemble")
            maximum = max(maximum, original_error, analytic_original_error, result["maximum_analytic_error"])
            for method in METHODS:
                current[method].append(result["prototypes"][method])
            current_analytic.append(result["analytic_A"])
            current_signatures.append(tuple(row["patient_id"] for row in ordered))
            records.append(
                {
                    "block": block,
                    "finding_index": finding_index,
                    "supports": ordered,
                    "bank_indices": bank_indices.tolist(),
                    "bank_norms": result["bank_norms"].tolist(),
                    "occurrence_counts": result["occurrence_counts"].tolist(),
                    "normalized_weights": {key: value.tolist() for key, value in result["weights"].items()},
                }
            )
        for method in METHODS:
            arrays[method].append(np.stack(current[method]))
        analytic.append(np.stack(current_analytic))
        signatures.append(tuple(current_signatures))
    arrays = {key: np.stack(value) for key, value in arrays.items()}
    arrays["A"] = previous["ensemble"]
    diagnostics = {
        "maximum_prototype_identity_error": maximum,
        "duplicate_unions": union_diagnostics(signatures, arrays["B"]),
        "weight_effective_support": {
            method: summary(
                np.array([1 / np.square(row["normalized_weights"][method]).sum() for row in records])
            )
            for method in METHODS
        },
        "weight_effective_support_fraction": {
            method: summary(
                np.array(
                    [
                        1 / np.square(row["normalized_weights"][method]).sum() / len(row["supports"])
                        for row in records
                    ]
                )
            )
            for method in METHODS
        },
        "bank_norms": summary(np.array([value for row in records for value in row["bank_norms"]])),
    }
    write_json_atomic(
        output / "matched_union_manifest_local.json",
        {
            "records": records,
            "diagnostics": diagnostics,
            "fixed_contrasts_from": "fixed_manifest_local.json",
            "unique_support_people_matched": True,
            "equal_computation_budget": False,
        },
    )
    return arrays, np.stack(analytic), records, diagnostics


def paired_effect(patients: list[dict], first: str, second: str) -> dict:
    """Summarize signed patient effects without averaging an undefined patient away.

    Parameters
    ----------
    patients : list of dict
        Complete patient/method pair summaries.
    first, second : str
        Prespecified subtraction order.

    Returns
    -------
    dict
        Full patient-difference distribution, materiality counts and coverage.
    """
    differences = []
    for row in patients:
        left, right = (row[method]["pairs"]["positive_quarter_jaccard"] for method in (first, second))
        differences.append(
            left["mean"] - right["mean"] if left["mean"] is not None and right["mean"] is not None else np.nan
        )
    values = np.asarray(differences)
    finite = values[np.isfinite(values)]
    defined = len(patients) == 60 and len(finite) == 60
    distribution = summary(values)
    distribution["mean"] = math.fsum(finite) / len(finite) if len(finite) else None
    coverage = {
        method: len(patients) == 60
        and all(
            row[method]["pairs"]["positive_quarter_jaccard"]["count"] == 45
            and row[method]["pairs"]["positive_quarter_jaccard"]["defined"] >= 43
            for row in patients
        )
        for method in (first, second)
    }
    return {
        "first": first,
        "second": second,
        "all_60_patient_differences_defined": defined,
        "mean_patient_difference": distribution["mean"] if defined else None,
        "distribution": distribution,
        "per_method_every_patient_coverage_passed": coverage,
        "patient_counts": {
            "positive": int(np.sum(finite > 0)),
            "negative": int(np.sum(finite < 0)),
            "zero": int(np.sum(finite == 0)),
            "at_least_plus_0.05": int(np.sum(finite >= 0.05)),
            "at_most_minus_0.05": int(np.sum(finite <= -0.05)),
            "inside_resolution_band": int(np.sum(np.abs(finite) < 0.05)),
            "undefined": int(len(values) - len(finite)),
        },
    }


def primary_decision(patients: list[dict]) -> dict:
    """Classify the symmetric finite-design A-minus-B effect with frozen coverage rules.

    Parameters
    ----------
    patients : list of dict
        All target-level method summaries.

    Returns
    -------
    dict
        Assessability, operational category and complete signed effect.
    """
    effect = paired_effect(patients, "A", "B")
    assessable = effect["all_60_patient_differences_defined"] and all(
        effect["per_method_every_patient_coverage_passed"].values()
    )
    category = "unassessable"
    if assessable:
        value = effect["mean_patient_difference"]
        category = "no_material_mean_difference_at_operational_resolution"
        if value >= 0.05:
            category = "A_material_conditional_consistency_advantage"
        elif value <= -0.05:
            category = "B_material_conditional_consistency_advantage"
    return {
        "assessable": assessable,
        "category": category,
        "resolution": 0.05,
        "effect": effect,
        "population_equivalence_or_clinical_adequacy_established": False,
    }


def between_methods(first: dict, second: dict, active: np.ndarray) -> tuple[dict, dict]:
    """Compare the same blocks' sets, ranks, margins and signed/absolute maps.

    Parameters
    ----------
    first, second : dict
        Complete ten-block fixed-contrast outputs from a prespecified method pair.
    active : np.ndarray
        Original unique active-token mask.

    Returns
    -------
    tuple
        Within-patient summaries and complete private same-block metric arrays.
    """
    values = {
        name: []
        for name in ("signed_rank", "absolute_rank", "positive_quarter_jaccard", "absolute_quarter_jaccard")
    }
    for block in range(10):
        _, pairs = pair_metrics(np.stack([first["beat"][block], second["beat"][block]]))
        for key in values:
            values[key].append(pairs[key][0])
    arrays = {key: np.array(value) for key, value in values.items()}
    arrays["signed_margin_difference"] = first["margins"] - second["margins"]
    arrays["absolute_margin_difference"] = np.abs(arrays["signed_margin_difference"])
    arrays["strict_margin_sign_reversal"] = ((first["margins"] > 0) & (second["margins"] < 0)) | (
        (first["margins"] < 0) & (second["margins"] > 0)
    )
    arrays["zero_margin_transition"] = (first["margins"] == 0) != (second["margins"] == 0)
    arrays["both_margins_zero"] = (first["margins"] == 0) & (second["margins"] == 0)
    for level in ("beat", "token"):
        left, right = first[level], second[level]
        if level == "token":
            left, right = left[:, active], right[:, active]
        arrays[level + "_signed_L1_difference"] = np.abs(left - right).sum(axis=1)
        left_mass, right_mass = np.abs(left).sum(axis=1), np.abs(right).sum(axis=1)
        defined = (left_mass > 0) & (right_mass > 0)
        tv = np.full(len(left), np.nan)
        tv[defined] = 0.5 * np.abs(
            np.abs(left[defined]) / left_mass[defined, None]
            - np.abs(right[defined]) / right_mass[defined, None]
        ).sum(axis=1)
        arrays[level + "_normalized_absolute_TV"] = tv
    stats = {key: summary(value) for key, value in arrays.items()}
    for key in ("strict_margin_sign_reversal", "zero_margin_transition", "both_margins_zero"):
        stats[key]["event_count"] = int(np.sum(arrays[key]))
    return stats, arrays


def aggregate_methods(patients: list[dict], prototypes: dict) -> dict:
    """Aggregate equal-patient method and same-block comparisons with explicit null counts.

    Parameters
    ----------
    patients : list of dict
        Complete private patient summaries for A/B/C.
    prototypes : dict
        Ten unit finding prototypes per method.

    Returns
    -------
    dict
        Aggregate-safe family, map, prototype and patient-effect summaries.
    """
    methods = {
        method: aggregate_patients([{"single": row[method], "ensemble": row[method]} for row in patients])[
            "ensemble"
        ]
        for method in METHODS
    }
    comparisons = {}
    for first, second in PAIRS:
        key = first + ":" + second
        comparisons[key] = {
            name: {
                "patient_mean": summary(
                    np.array(
                        [
                            row["between"][key][name]["mean"]
                            if row["between"][key][name]["mean"] is not None
                            else np.nan
                            for row in patients
                        ]
                    )
                ),
                "undefined_blocks": sum(row["between"][key][name]["undefined"] for row in patients),
            }
            for name in patients[0]["between"][key]
        }
        comparisons[key]["prototype_cosine"] = summary(
            np.clip(np.sum(prototypes[first] * prototypes[second], axis=2), -1, 1).ravel()
        )
        comparisons[key]["prototype_L2_difference"] = summary(
            np.linalg.norm(prototypes[first] - prototypes[second], axis=2).ravel()
        )
        comparisons[key]["margin_event_counts"] = {
            name: sum(row["between"][key][name]["event_count"] for row in patients)
            for name in ("strict_margin_sign_reversal", "zero_margin_transition", "both_margins_zero")
        }
    return {
        "methods": methods,
        "matching_block_comparisons": comparisons,
        "secondary_patient_effects": {
            "C:B": paired_effect(patients, "C", "B"),
            "A:C": paired_effect(patients, "A", "C"),
        },
    }


def evaluate_patients(
    data: dict,
    prototypes: dict,
    analytic: np.ndarray,
    contrasts: dict,
    geometries: list[dict],
    output: Path,
    deadline: float,
) -> tuple:
    """Evaluate all declared methods and analytic identities without target exclusions.

    Parameters
    ----------
    data, prototypes, contrasts : dict
        Verified targets, matched unit prototypes and unchanged full-nine contrasts.
    analytic : np.ndarray
        Analytic coefficient reconstruction of A.
    geometries : list of dict
        Replayed original contextual pooling geometry.
    output : Path
        Local progress destination.
    deadline : float
        Monotonic execution deadline.

    Returns
    -------
    tuple
        Complete patient records, private arrays and maximum numerical errors.
    """
    patients, arrays = [], {method + "_prototypes": value for method, value in prototypes.items()}
    arrays["analytic_A_prototypes"] = analytic
    maximum, analytic_error = 0.0, 0.0
    for index, target in enumerate(data["selection"]["targets"]):
        check_budget(deadline)
        geometry, contrast = geometries[index], contrasts[str(index)]
        ordered = (contrast["winner"], contrast["runner"])
        patient = {"target_index": index, "ecg_id": target["ecg_id"], "patient_id": target["patient_id"]}
        results = {}
        for method in METHODS:
            result = fixed_contributions(geometry, prototypes[method], ordered)
            results[method] = result
            pairs, pair_arrays = pair_metrics(result["beat"])
            patient[method] = {"pairs": pairs, "metrics": family_metrics(result, geometry)}
            maximum = max(maximum, result["maximum_sum_error"])
            arrays.update(
                {
                    f"target_{index}_{method}_{key}": value
                    for key, value in result.items()
                    if isinstance(value, np.ndarray)
                }
            )
            arrays.update(
                {f"target_{index}_{method}_pairs_{key}": value for key, value in pair_arrays.items()}
            )
        check = fixed_contributions(geometry, analytic, ordered)
        analytic_error = max(
            analytic_error,
            *(array_error(check[key], results["A"][key]) for key in ("beat", "token", "margins")),
        )
        if analytic_error > 1e-10:
            raise ValueError("Analytic A contribution identity failed")
        patient["between"] = {}
        for first, second in PAIRS:
            key = first + ":" + second
            stats, values = between_methods(results[first], results[second], geometry["active_tokens"])
            patient["between"][key] = stats
            arrays.update({f"target_{index}_between_{key}_{name}": value for name, value in values.items()})
        arrays.update(
            {
                f"target_{index}_geometry_{key}": value
                for key, value in geometry.items()
                if isinstance(value, np.ndarray)
            }
        )
        patients.append(patient)
        write_json_atomic(output / "progress.json", {"targets_completed": index + 1, "total": 60})
    return (
        patients,
        arrays,
        {"maximum_contribution_sum_error": maximum, "maximum_analytic_A_contribution_error": analytic_error},
    )


def run(output: Path) -> dict:
    """Execute one reviewed, source-frozen matched-union study using local caches only.

    Parameters
    ----------
    output : Path
        New local execution directory.

    Returns
    -------
    dict
        Complete source-bound result, including unassessable or unfavorable classifications.
    """
    started = time.perf_counter()
    deadline = started + 1200
    inputs = {}
    previous_path = ROOT / "outputs/ntt09_reference_contribution_stability/result.json"
    previous = read_bound(previous_path, PREVIOUS_SHA, inputs)
    status = json.loads(previous_path.with_name("status.json").read_text())
    if (
        previous["status"] != "completed"
        or status.get("status") != "completed"
        or status.get("result_sha256") != PREVIOUS_SHA
    ):
        raise ValueError("NTT-09 completion binding differs")
    data = replay_cached_inputs()
    inputs.update(data["inputs"])
    sources = committed_sources(SOURCES, previous["source_sha256"])
    replay, old_prototypes, contrasts, exposure, geometries = replay09(data, previous, output, deadline)
    prototypes, analytic, _, diagnostics = matched_prototypes(data, old_prototypes, output)
    manifest_sha = sha256_file(output / "matched_union_manifest_local.json")
    patients, arrays, errors = evaluate_patients(
        data, prototypes, analytic, contrasts, geometries, output, deadline
    )
    np.savez_compressed(output / "contributions_local.npz", **arrays)
    write_json_atomic(output / "patients_local.json", patients)
    write_json_atomic(output / "selection_local.json", data["selection"])
    write_json_atomic(output / "coordinates_local.json", data["coordinates"])
    if sha256_file(output / "matched_union_manifest_local.json") != manifest_sha:
        raise ValueError("Frozen matched-union manifest changed")
    check_budget(deadline)
    result = {
        "study": "NTT-10",
        "status": "completed",
        "decision": primary_decision(patients),
        "aggregate": aggregate_methods(patients, prototypes),
        "prototype_diagnostics": diagnostics,
        "source_exposure": exposure,
        "unique_support_people_matched": True,
        "equal_computation_budget": False,
        **errors,
        "predecessor_reproduction": replay,
        "source_sha256": sources,
        "input_sha256": inputs,
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "runtime": runtime_fingerprint(),
        "seconds": time.perf_counter() - started,
        "counts": {
            "patients": 60,
            "support_patients": 27,
            "blocks": 10,
            "methods": 3,
            "new_decompositions": 1800,
            "analytic_A_decompositions": 600,
            "pairs_per_method_per_patient": 45,
            "within_method_pairs": 8100,
            "matching_block_method_pairs": 1800,
            "failed_decompositions": 0,
        },
        "waveform_reads": 0,
        "new_encoder_inferences": 0,
        "new_clinical_annotations": 0,
        "clinical_localization_evaluated": False,
        "closed_tests_read": False,
        "artifact_sha256": {
            to_stored(path): sha256_file(path)
            for path in output.iterdir()
            if path.name not in ("result.json", "status.json", "failure.json", "progress.json")
        },
    }
    write_json_atomic(output / "result.json", result)
    return result
