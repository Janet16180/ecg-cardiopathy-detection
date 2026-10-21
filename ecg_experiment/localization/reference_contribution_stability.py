"""Finite-pool reference averaging for fixed-contrast contextual contribution maps."""

from __future__ import annotations

import json
import math
import subprocess
import time
from pathlib import Path

import numpy as np
from scipy.stats import rankdata

from ecg_experiment import ROOT
from ecg_experiment.clustering.bank_inputs import committed_sources, read_bound
from ecg_experiment.clustering.bank_robustness import bank_prototypes, support_draws
from ecg_experiment.clustering.native_cache import runtime_fingerprint
from ecg_experiment.clustering.reference_geometry import check_budget
from ecg_experiment.files import sha256_file, write_json_atomic
from ecg_experiment.localization.contextual_margin import mass_summary, occupancy, summary
from ecg_experiment.localization.contribution_inputs import cached_tokens, replay_cached_inputs
from ecg_experiment.localization.contribution_study import evaluate_target, patient_summaries
from ecg_experiment.localization.native_study import regions_from_saved
from ecg_experiment.native_reference_ntt05 import unit_vector
from ecg_experiment.paths import to_stored
from ecg_experiment.same_wave_ntt04 import FINDINGS

PREDECESSORS = {
    "ntt08_contextual_contributions": "0d137e4f8618a99f2d6994179b9ff5b1b3de0d82e309fd6ab4d915d5a3001058",
    "cluster04_bank_robustness": "c672be6b2fc9fe6d8c14d974179f04345381b068349df611d1656723e0427baa",
}
SOURCES = (
    "docs/research/localization/ntt09-reference-contribution-stability/protocol.md",
    "ecg_experiment/localization/reference_contribution_stability.py",
    "scripts/experiments/localization/run_ntt09.py",
    "tests/localization/test_reference_contribution_stability.py",
)


def nested_error(actual: object, expected: object) -> float:
    """Compare complete predecessor structures while requiring exact identities and counts.

    Parameters
    ----------
    actual, expected : object
        JSON-compatible reconstructed and saved nested values.

    Returns
    -------
    float
        Maximum absolute floating error; structural/integer differences raise.
    """
    if isinstance(expected, dict):
        if not isinstance(actual, dict) or actual.keys() != expected.keys():
            raise ValueError("Predecessor dictionary fields differ")
        return max((nested_error(actual[key], value) for key, value in expected.items()), default=0.0)
    if isinstance(expected, list):
        if not isinstance(actual, list) or len(actual) != len(expected):
            raise ValueError("Predecessor list denominator differs")
        return max((nested_error(a, b) for a, b in zip(actual, expected, strict=True)), default=0.0)
    if isinstance(expected, float):
        if not np.isfinite(float(actual)) or not np.isfinite(expected):
            raise ValueError("Nonfinite predecessor numerical value")
        return abs(float(actual) - expected)
    if actual != expected:
        raise ValueError("Predecessor discrete value differs")
    return 0.0


def replay08(data: dict, receipt: dict, deadline: float) -> dict:
    """Reconstruct all frozen original contributions, interventions and patient summaries.

    Parameters
    ----------
    data, receipt : dict
        Role-verified cached inputs and hash-bound NTT-08 result.
    deadline : float
        Monotonic execution deadline.

    Returns
    -------
    dict
        Full replay counts and maximum errors.
    """
    base = ROOT / "outputs/ntt08_contextual_contributions"
    evaluations, consistency = [], []
    maximum_array = 0.0
    with np.load(base / "contributions_local.npz", allow_pickle=False) as saved:
        for index, target in enumerate(data["selection"]["targets"]):
            check_budget(deadline)
            arrays = {}
            rows, pairs, _ = evaluate_target(data, target, index, arrays)
            evaluations.extend(rows)
            consistency.append(pairs)
            for key, values in arrays.items():
                if values.shape != saved[key].shape or not np.isfinite(values).all():
                    raise ValueError("Original contribution array shape differs")
                maximum_array = max(maximum_array, float(np.max(np.abs(values.astype(float) - saved[key]))))
    errors = [
        nested_error(evaluations, json.loads((base / "evaluations_local.json").read_text())),
        nested_error(consistency, json.loads((base / "bank_consistency_local.json").read_text())),
        nested_error(patient_summaries(evaluations, consistency), receipt["aggregate"]),
    ]
    if len(evaluations) != 180 or max(maximum_array, *errors) > 1e-12:
        raise ValueError("NTT-08 exact replay failed")
    return {
        "evaluations": 180,
        "patients": 60,
        "maximum_array_error": maximum_array,
        "maximum_summary_error": max(errors),
        "original_aggregate_reproduced": True,
        "native_predecessor_replay": data["replay"],
    }


def contextual_geometry(data: dict, target: dict) -> dict:
    """Reconstruct the fixed pooling geometry without selecting any contrast.

    Parameters
    ----------
    data, target : dict
        Verified original tokens, regions and one unchanged target identity.

    Returns
    -------
    dict
        Occupancy, unit beats, patient vector and unique-token projection weights.
    """
    ecg = target["ecg_id"]
    tokens = cached_tokens(data["tokens"][ecg], ecg)
    regions = regions_from_saved(data["coordinates"][str(ecg)]["qrs"])
    weights = occupancy(tokens, regions)
    values = np.asarray(tokens.values, dtype=float)
    means = weights @ values
    norms = np.linalg.norm(means, axis=1)
    if np.any(norms <= 0) or not np.isfinite(norms).all():
        raise ValueError("Undefined frozen region normalization")
    beats = means / norms[:, None]
    total = beats.sum(axis=0)
    denominator = float(np.linalg.norm(total))
    if denominator <= 0 or not np.isfinite(denominator):
        raise ValueError("Undefined aggregate normalization")
    patient = total / denominator
    if np.max(np.abs(patient - data["features"][ecg]["native_qrs"])) > 1e-12:
        raise ValueError("Fixed geometry descriptor differs")
    return {
        "beats": beats,
        "denominator": denominator,
        "patient": patient,
        "tokens": values,
        "occupancy": weights,
        "region_norms": norms,
        "active_tokens": np.any(weights > 0, axis=0),
        "token_coefficients": (weights / norms[:, None]).sum(axis=0),
    }


def fixed_contributions(geometry: dict, prototypes: np.ndarray, contrast: tuple[int, int]) -> dict:
    """Project all banks through the same ordered contrast, preserving negative margins.

    Parameters
    ----------
    geometry : dict
        Original occupancy, beat vectors and normalization factors.
    prototypes : np.ndarray
        Banks by findings by feature unit prototype vectors.
    contrast : tuple of int
        Frozen full-nine-pool winner and runner-up indices.

    Returns
    -------
    dict
        Signed beat/token contributions, scores, margins and reconstruction errors.
    """
    prototypes = np.asarray(prototypes, dtype=float)
    if (
        prototypes.ndim != 3
        or prototypes.shape[1] != 3
        or not np.isfinite(prototypes).all()
        or not np.allclose(np.linalg.norm(prototypes, axis=2), 1, atol=1e-12, rtol=0)
    ):
        raise ValueError("Expected finite unit finding prototypes")
    winner, runner = contrast
    if winner == runner or min(contrast) < 0 or max(contrast) >= 3:
        raise ValueError("Invalid fixed ordered contrast")
    delta = prototypes[:, winner] - prototypes[:, runner]
    beat = delta @ geometry["beats"].T / geometry["denominator"]
    token = (delta @ geometry["tokens"].T) * geometry["token_coefficients"] / geometry["denominator"]
    scores = np.clip(prototypes @ geometry["patient"], -1, 1)
    margins = scores[:, winner] - scores[:, runner]
    error = max(
        float(np.max(np.abs(beat.sum(axis=1) - margins))), float(np.max(np.abs(token.sum(axis=1) - margins)))
    )
    if error > 1e-10 or not np.isfinite(beat).all() or not np.isfinite(token).all():
        raise ValueError("Fixed-contrast contribution reconstruction failed")
    return {"beat": beat, "token": token, "scores": scores, "margins": margins, "maximum_sum_error": error}


def top_sets(values: np.ndarray, positive: bool) -> np.ndarray:
    """Return deterministic signed-positive or absolute top-quarter membership.

    Parameters
    ----------
    values : np.ndarray
        Banks by beats contributions.
    positive : bool
        Select strictly positive signed ranks when true, absolute ranks otherwise.

    Returns
    -------
    np.ndarray
        Boolean banks-by-beats membership with original-index tie order.
    """
    count = math.ceil(values.shape[1] / 4)
    ranks = np.argsort(-values if positive else -np.abs(values), axis=1, kind="stable")[:, :count]
    result = np.zeros(values.shape, dtype=bool)
    np.put_along_axis(result, ranks, True, axis=1)
    return result & (values > 0) if positive else result


def rank_pairs(values: np.ndarray, pairs: tuple[np.ndarray, np.ndarray]) -> np.ndarray:
    """Compute average-rank Spearman pairs, retaining constant-rank comparisons as NaN.

    Parameters
    ----------
    values : np.ndarray
        Banks by beats finite contributions.
    pairs : tuple of np.ndarray
        Upper-triangle bank indices in fixed lexicographic order.

    Returns
    -------
    np.ndarray
        Signed-rank correlations for all pairs, including undefined entries.
    """
    ranks = rankdata(values, axis=1)
    centered = ranks - ranks.mean(axis=1, keepdims=True)
    norms = np.linalg.norm(centered, axis=1)
    denominator = norms[:, None] * norms[None, :]
    matrix = np.divide(
        centered @ centered.T, denominator, out=np.full(denominator.shape, np.nan), where=denominator > 0
    )
    return matrix[pairs]


def pair_metrics(values: np.ndarray) -> tuple[dict, dict]:
    """Describe every bank pair with fixed denominators and explicit empty-set handling.

    Parameters
    ----------
    values : np.ndarray
        Banks by beats fixed-contrast contributions.

    Returns
    -------
    tuple
        Complete metric summaries and private pair arrays.
    """
    pairs = np.triu_indices(len(values), 1)
    arrays = {"signed_rank": rank_pairs(values, pairs), "absolute_rank": rank_pairs(np.abs(values), pairs)}
    for name, positive in (("positive_quarter_jaccard", True), ("absolute_quarter_jaccard", False)):
        membership = top_sets(values, positive).astype(int)
        intersections = membership @ membership.T
        sizes = membership.sum(axis=1)
        unions = sizes[:, None] + sizes[None, :] - intersections
        matrix = np.divide(intersections, unions, out=np.full(unions.shape, np.nan), where=unions > 0)
        arrays[name] = matrix[pairs]
    return {name: summary(array) for name, array in arrays.items()}, arrays


def family_metrics(result: dict, geometry: dict) -> dict:
    """Summarize equal-bank signed masses, ties and signed/absolute map variability.

    Parameters
    ----------
    result, geometry : dict
        Complete fixed-contrast outputs and original active-token geometry.

    Returns
    -------
    dict
        Private patient-level scalar summaries without discarding finite banks.
    """
    metrics = {
        "negative_margin_fraction": float(np.mean(result["margins"] < 0)),
        "zero_margin_fraction": float(np.mean(result["margins"] == 0)),
        "minimum_margin": float(result["margins"].min()),
        "maximum_margin": float(result["margins"].max()),
        "mean_positive_quarter_size": float(top_sets(result["beat"], True).sum(axis=1).mean()),
        "empty_positive_sets": int(np.sum(~top_sets(result["beat"], True).any(axis=1))),
    }
    for level in ("beat", "token"):
        values = result[level] if level == "beat" else result[level][:, geometry["active_tokens"]]
        masses = [mass_summary(row) for row in values]
        metrics[f"{level}_zero_absolute_mass_banks"] = sum(row["absolute_mass"] == 0 for row in masses)
        for name in ("positive_mass", "negative_mass", "absolute_mass", "cancellation", "effective_fraction"):
            finite = [row[name] for row in masses if row[name] is not None]
            metrics[f"{level}_{name}"] = float(np.mean(finite)) if finite else None
            metrics[f"{level}_{name}_undefined_banks"] = len(masses) - len(finite)
        for fraction in ("0.1", "0.25", "0.5"):
            metrics[f"{level}_top_absolute_share_{fraction}"] = float(
                np.mean([row["top_absolute_shares"][fraction] for row in masses])
            )
        for absolute, label in ((False, "signed"), (True, "absolute")):
            maps = np.abs(values) if absolute else values
            metrics[f"{level}_{label}_mean_population_std"] = float(maps.std(axis=0, ddof=0).mean())
            ordered = np.sort(maps, axis=1)[:, ::-1]
            count = math.ceil(values.shape[1] / 4)
            metrics[f"{level}_{label}_any_rank_tie_banks"] = int(
                np.sum(np.any(np.diff(ordered, axis=1) == 0, axis=1))
            )
            metrics[f"{level}_{label}_quarter_boundary_tie_banks"] = (
                int(np.sum(ordered[:, count - 1] == ordered[:, count])) if count < values.shape[1] else 0
            )
    return metrics


def primary_decision(patients: list[dict]) -> dict:
    """Apply the prospective all-patient effect and per-patient coverage conditions.

    Parameters
    ----------
    patients : list of dict
        Each target's complete single/ensemble pair summaries.

    Returns
    -------
    dict
        Signed all-patient effect, coverage and fixed decision conditions.
    """
    differences = []
    for patient in patients:
        single = patient["single"]["pairs"]["positive_quarter_jaccard"]
        ensemble = patient["ensemble"]["pairs"]["positive_quarter_jaccard"]
        differences.append(
            ensemble["mean"] - single["mean"]
            if single["mean"] is not None and ensemble["mean"] is not None
            else None
        )
    all_defined = len(patients) == 60 and all(value is not None for value in differences)
    effect = float(np.mean(differences)) if all_defined else None
    conditions = {
        "all_60_patient_differences_defined": all_defined,
        "every_patient_single_coverage_at_least_1164_of_1225": len(patients) == 60
        and all(row["single"]["pairs"]["positive_quarter_jaccard"]["defined"] >= 1164 for row in patients),
        "every_patient_ensemble_coverage_at_least_43_of_45": len(patients) == 60
        and all(row["ensemble"]["pairs"]["positive_quarter_jaccard"]["defined"] >= 43 for row in patients),
        "mean_patient_improvement_at_least_0.10": effect is not None and effect >= 0.10,
    }
    return {
        "conditional_highlight_stability_benefit": all(conditions.values()),
        "conditions": conditions,
        "mean_patient_jaccard_difference": effect,
        "patient_difference_distribution": summary(
            np.array([value if value is not None else np.nan for value in differences])
        ),
    }


def aggregate_patients(patients: list[dict]) -> dict:
    """Create only equal-patient scalar summaries with explicit undefined counts.

    Parameters
    ----------
    patients : list of dict
        All 60 private patient records and their full bank-family summaries.

    Returns
    -------
    dict
        Aggregate-safe scalar distributions and pair coverage.
    """
    result = {}
    for family in ("single", "ensemble"):
        metrics = {
            name: summary(
                np.array(
                    [
                        row[family]["metrics"][name] if row[family]["metrics"][name] is not None else np.nan
                        for row in patients
                    ]
                )
            )
            for name in patients[0][family]["metrics"]
        }
        pairs = {
            name: {
                "patient_mean": summary(
                    np.array(
                        [
                            row[family]["pairs"][name]["mean"]
                            if row[family]["pairs"][name]["mean"] is not None
                            else np.nan
                            for row in patients
                        ]
                    )
                ),
                "defined_pairs_per_patient": summary(
                    np.array([row[family]["pairs"][name]["defined"] for row in patients])
                ),
                "all_pairs_per_patient": 1225 if family == "single" else 45,
            }
            for name in patients[0][family]["pairs"]
        }
        result[family] = {"patient_metrics": metrics, "pair_metrics": pairs}
    return result


def normalized_ensemble_prototypes(single: np.ndarray) -> np.ndarray:
    """Normalize the equal-bank prototype mean separately for every finding.

    Parameters
    ----------
    single : np.ndarray
        Ordered unit bank prototypes; the bank count is a multiple of five.

    Returns
    -------
    np.ndarray
        Unit finding prototypes for consecutive five-bank blocks.
    """
    if len(single) % 5:
        raise ValueError("Ensemble blocks require complete groups of five")
    return np.stack(
        [
            [unit_vector(single[i : i + 5, finding].mean(axis=0)) for finding in range(3)]
            for i in range(0, len(single), 5)
        ]
    )


def prototypes_and_manifest(data: dict, output: Path) -> tuple[dict, dict, dict]:
    """Freeze all banks, normalized ensembles and full-nine contrasts before comparisons.

    Parameters
    ----------
    data : dict
        Verified original support and target features.
    output : Path
        Local manifest destination.

    Returns
    -------
    tuple
        Unit prototypes, fixed patient contrasts and identity-only support-exposure counts.
    """
    manifest = json.loads((ROOT / "outputs/cluster04_bank_robustness/bank_manifest_local.json").read_text())
    if manifest["banks"] != support_draws(data["selection"]):
        raise ValueError("Original 50 support draws differ")
    features = {str(key): value for key, value in data["features"].items()}
    single = np.stack([bank_prototypes(draw, features) for draw in manifest["banks"]])
    ensemble = normalized_ensemble_prototypes(single)
    full = []
    for finding in FINDINGS:
        rows = {
            row["patient_id"]: row
            for group in data["selection"]["support_draws"]
            if group["finding"] == finding
            for row in group["supports"]
        }
        if len(rows) != 9:
            raise ValueError("Full-pool reference denominator differs")
        full.append(
            unit_vector(
                np.mean(
                    [
                        unit_vector(data["features"][row["ecg_id"]]["native_qrs"])
                        for _, row in sorted(rows.items())
                    ],
                    axis=0,
                )
            )
        )
    full = np.stack(full)
    contrasts = {}
    for index, target in enumerate(data["selection"]["targets"]):
        scores = np.clip(full @ data["features"][target["ecg_id"]]["native_qrs"], -1, 1)
        order = np.argsort(-scores, kind="stable")
        contrasts[str(index)] = {
            "target_index": index,
            "ecg_id": target["ecg_id"],
            "winner": int(order[0]),
            "runner": int(order[1]),
            "scores": scores.tolist(),
            "winner_runner_tied": bool(scores[order[0]] == scores[order[1]]),
        }
    distinct = {
        "single": [
            [len({row["patient_id"] for row in bank["supports"][finding]}) for finding in FINDINGS]
            for bank in manifest["banks"]
        ],
        "ensemble": [
            [
                len(
                    {
                        row["patient_id"]
                        for bank in manifest["banks"][i : i + 5]
                        for row in bank["supports"][finding]
                    }
                )
                for finding in FINDINGS
            ]
            for i in range(0, 50, 5)
        ],
    }
    exposure = {
        "single_support_slots_per_finding": 3,
        "ensemble_support_slots_per_finding": 15,
        "shared_contrast_support_patients_per_finding": 9,
        "single_distinct_supports_per_finding": summary(np.array(distinct["single"]).ravel()),
        "ensemble_distinct_supports_per_finding": summary(np.array(distinct["ensemble"]).ravel()),
        "ensemble_distinct_supports_all_findings": summary(np.array(distinct["ensemble"]).sum(axis=1)),
        "equal_support_or_computation_budget_comparison": False,
    }
    write_json_atomic(
        output / "fixed_manifest_local.json",
        {
            "banks": manifest,
            "contrasts": contrasts,
            "distinct_support_counts": distinct,
            "support_exposure": exposure,
        },
    )
    return {"single": single, "ensemble": ensemble, "full_pool": full}, contrasts, exposure


def run(output: Path) -> dict:
    """Execute one frozen cache-only study after exact predecessor reconstruction.

    Parameters
    ----------
    output : Path
        New local execution directory.

    Returns
    -------
    dict
        Source-bound completed study with complete patient coverage.
    """
    started = time.perf_counter()
    deadline = started + 1200
    inputs = {}
    receipts = {
        name: read_bound(ROOT / "outputs" / name / "result.json", digest, inputs)
        for name, digest in PREDECESSORS.items()
    }
    for name, receipt in receipts.items():
        status = json.loads((ROOT / "outputs" / name / "status.json").read_text())
        if (
            receipt["status"] != "completed"
            or status.get("status") != "completed"
            or status.get("result_sha256") != PREDECESSORS[name]
        ):
            raise ValueError("Predecessor completion differs")
    data = replay_cached_inputs()
    inputs.update(data["inputs"])
    inherited = {
        key: value for receipt in receipts.values() for key, value in receipt["source_sha256"].items()
    }
    sources = committed_sources(SOURCES, inherited)
    reproduction = replay08(data, receipts["ntt08_contextual_contributions"], deadline)
    prototypes, contrasts, exposure = prototypes_and_manifest(data, output)
    manifest_sha = sha256_file(output / "fixed_manifest_local.json")
    arrays = {
        "single_prototypes": prototypes["single"],
        "ensemble_prototypes": prototypes["ensemble"],
        "full_pool_prototypes": prototypes["full_pool"],
    }
    patients, maximum = [], 0.0
    for index, target in enumerate(data["selection"]["targets"]):
        check_budget(deadline)
        geometry = contextual_geometry(data, target)
        contrast = contrasts[str(index)]
        patient = {"target_index": index, "ecg_id": target["ecg_id"], "patient_id": target["patient_id"]}
        for family in ("single", "ensemble"):
            result = fixed_contributions(
                geometry, prototypes[family], (contrast["winner"], contrast["runner"])
            )
            pairs, pair_arrays = pair_metrics(result["beat"])
            patient[family] = {"pairs": pairs, "metrics": family_metrics(result, geometry)}
            maximum = max(maximum, result["maximum_sum_error"])
            arrays.update(
                {
                    f"target_{index}_{family}_{key}": value
                    for key, value in result.items()
                    if isinstance(value, np.ndarray)
                }
            )
            arrays.update(
                {f"target_{index}_{family}_pairs_{key}": value for key, value in pair_arrays.items()}
            )
        arrays.update(
            {
                f"target_{index}_geometry_{key}": value
                for key, value in geometry.items()
                if isinstance(value, np.ndarray)
            }
        )
        patients.append(patient)
        write_json_atomic(output / "progress.json", {"targets_completed": index + 1, "total": 60})
    decision = primary_decision(patients)
    np.savez_compressed(output / "contributions_local.npz", **arrays)
    write_json_atomic(output / "patients_local.json", patients)
    write_json_atomic(output / "selection_local.json", data["selection"])
    write_json_atomic(output / "coordinates_local.json", data["coordinates"])
    if sha256_file(output / "fixed_manifest_local.json") != manifest_sha:
        raise ValueError("Frozen contrast/support manifest changed")
    check_budget(deadline)
    result = {
        "study": "NTT-09",
        "status": "completed",
        "decision": decision,
        "support_exposure": exposure,
        "aggregate": aggregate_patients(patients),
        "maximum_contribution_sum_error": maximum,
        "predecessor_reproduction": reproduction,
        "source_sha256": sources,
        "input_sha256": inputs,
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "runtime": runtime_fingerprint(),
        "seconds": time.perf_counter() - started,
        "counts": {
            "patients": 60,
            "support_patients": 27,
            "single_banks": 50,
            "ensembles": 10,
            "new_decompositions": 3600,
            "single_pairs_per_patient": 1225,
            "ensemble_pairs_per_patient": 45,
            "failed_decompositions": 0,
        },
        "full_pool_winner_runner_ties": sum(row["winner_runner_tied"] for row in contrasts.values()),
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
