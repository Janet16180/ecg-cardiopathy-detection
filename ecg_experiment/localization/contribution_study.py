"""Prospective cached contextual-contribution execution and patient-first aggregate summaries."""

from __future__ import annotations

import subprocess
import time
from pathlib import Path

import numpy as np

from ecg_experiment import ROOT
from ecg_experiment.clustering.bank_inputs import committed_sources
from ecg_experiment.clustering.native_cache import runtime_fingerprint
from ecg_experiment.files import sha256_file, write_json_atomic
from ecg_experiment.localization.contextual_margin import (
    FRACTIONS,
    bank_consistency,
    decomposition,
    intervention_summary,
    mass_summary,
    summary,
)
from ecg_experiment.localization.contribution_inputs import cached_tokens, replay_cached_inputs
from ecg_experiment.localization.jepa_native import CONTEXT_SCOPE, EMBEDDING_SPACE
from ecg_experiment.localization.native_study import regions_from_saved
from ecg_experiment.paths import to_stored
from ecg_experiment.same_wave_ntt04 import SEEDS

SOURCES = (
    "docs/research/localization/ntt08-contextual-contributions/protocol.md",
    "docs/research/localization/ntt08-contextual-contributions/edge-cases.md",
    "ecg_experiment/localization/contextual_margin.py",
    "ecg_experiment/localization/contribution_inputs.py",
    "ecg_experiment/localization/contribution_study.py",
    "ecg_experiment/clustering/bank_inputs.py",
    "ecg_experiment/clustering/native_cache.py",
    "ecg_experiment/clustering01.py",
    "scripts/experiments/localization/run_ntt08.py",
    "tests/localization/test_contextual_margin.py",
)


def metric_values(result: dict, intervention: dict, lead_contributions: dict) -> dict:
    """Collect scalar behavior descriptors for aggregation within each patient first."""
    beat = mass_summary(result["beat_contributions"])
    token = mass_summary(result["token_contributions"][result["active_tokens"]])
    metrics = {
        "margin": result["margin"],
        "beat_count": beat["count"],
        "active_token_count": token["count"],
        "single_deletion_spearman": intervention["contribution_drop_spearman"],
        "single_negative_drop_fraction": intervention["single_negative_drops"] / beat["count"],
        "negative_beat_contribution_fraction": float(np.mean(result["beat_contributions"] < 0)),
    }
    metrics.update({f"lead_{lead}_signed_contribution": value for lead, value in lead_contributions.items()})
    for level, values in (("beat", beat), ("token", token)):
        for name in (
            "positive_mass",
            "negative_mass",
            "absolute_mass",
            "cancellation",
            "effective_count",
            "effective_fraction",
        ):
            metrics[f"{level}_{name}"] = values[name]
        for fraction, value in values["top_absolute_shares"].items():
            metrics[f"{level}_top_absolute_share_{fraction}"] = value
    for fraction, values in intervention["fractions"].items():
        metrics[f"deletion_selected_drop_{fraction}"] = values["selected_margin_drop"]
        metrics[f"deletion_random_mean_drop_{fraction}"] = values["random_drop"]["mean"]
        metrics[f"deletion_selected_minus_random_{fraction}"] = values["selected_minus_random_mean"]
    return metrics


def mean_defined(values: list[float | None]) -> float | None:
    """Average defined values without replacing missing interventions by zero."""
    finite = [value for value in values if value is not None and np.isfinite(value)]
    return float(np.mean(finite)) if finite else None


def patient_summaries(evaluations: list[dict], consistency: list[dict]) -> dict:
    """Produce public scalar summaries after equal-patient averaging across original banks."""
    patients = sorted({row["patient_id"] for row in evaluations})
    metrics = {}
    for name in evaluations[0]["metrics"]:
        per_patient = [
            mean_defined([row["metrics"][name] for row in evaluations if row["patient_id"] == patient])
            for patient in patients
        ]
        metrics[name] = summary(np.array(per_patient, dtype=float))
        metrics[name]["undefined_bank_evaluations"] = sum(row["metrics"][name] is None for row in evaluations)
    same_contrast = sum(row["same_contrast"] for row in consistency)
    pair_counts = sum(len(row["pairs"]) for row in consistency)
    bank_metrics = {}
    for name in ("signed_rank_spearman", "top_quarter_jaccard"):
        per_patient = [
            mean_defined([pair.get(name) for pair in row["pairs"] if pair["same_contrast"]])
            for row in consistency
        ]
        bank_metrics[name] = summary(np.array(per_patient, dtype=float))
        bank_metrics[name]["undefined_eligible_pairs"] = sum(
            pair.get(name) is None for row in consistency for pair in row["pairs"] if pair["same_contrast"]
        )
    return {
        "patients": len(patients),
        "bank_target_evaluations": len(evaluations),
        "patient_first_metrics": metrics,
        "bank_consistency": {
            "all_pairs": pair_counts,
            "same_ordered_contrast_pairs": same_contrast,
            "changed_ordered_contrast_pairs": pair_counts - same_contrast,
            "patients_with_any_same_contrast_pair": sum(row["same_contrast"] > 0 for row in consistency),
            "metrics": bank_metrics,
        },
        "undefined_single_deletions": sum(
            row["intervention"]["single_undefined_drops"] for row in evaluations
        ),
        "undefined_selected_subsets": sum(
            values["selected_margin_drop"] is None
            for row in evaluations
            for values in row["intervention"]["fractions"].values()
        ),
        "undefined_random_subsets": sum(
            values["random_drop"]["undefined"]
            for row in evaluations
            for values in row["intervention"]["fractions"].values()
        ),
        "clinical_localization_evaluated": False,
        "actual_waveform_intervention_performed": False,
    }


def evaluate_target(data: dict, target: dict, index: int, arrays: dict) -> tuple[list[dict], dict, dict]:
    """Evaluate all three frozen banks for one admitted target without correctness filtering."""
    ecg = target["ecg_id"]
    tokens = cached_tokens(data["tokens"][ecg], ecg)
    regions = regions_from_saved(data["coordinates"][str(ecg)]["qrs"])
    evaluations, results = [], []
    errors = dict.fromkeys(
        ("pooling", "descriptor", "scores", "beat_sum", "token_sum", "beat_token_sum"), 0.0
    )
    for bank_index, seed in enumerate(SEEDS):
        result = decomposition(tokens, regions, data["banks"][seed].prototypes)
        results.append(result)
        expected = next(row for row in data["predictions"] if row["ecg_id"] == ecg and row["seed"] == seed)
        original_order = np.argsort(-np.asarray(expected["scores"]["native_qrs"]), kind="stable")
        if (result["winner"], result["runner"]) != tuple(original_order[:2]):
            raise ValueError("Original winner/runner contrast changed during decomposition")
        current = {
            "pooling": result["pooling_maximum_error"],
            "descriptor": float(
                np.max(np.abs(result["patient_vector"] - data["features"][ecg]["native_qrs"]))
            ),
            "scores": float(np.max(np.abs(result["scores"] - expected["scores"]["native_qrs"]))),
            "beat_sum": abs(float(result["beat_contributions"].sum()) - result["margin"]),
            "token_sum": abs(float(result["token_contributions"].sum()) - result["margin"]),
            "beat_token_sum": float(
                np.max(np.abs(result["beat_token_contributions"].sum(axis=1) - result["beat_contributions"]))
            ),
        }
        errors = {name: max(errors[name], current[name]) for name in errors}
        intervention = intervention_summary(result, ecg, seed)
        lead_contributions = {
            lead: float(result["token_contributions"][[lead in group for group in tokens.token_leads]].sum())
            for lead in ("V1", "V6")
        }
        evaluations.append(
            {
                "ecg_id": ecg,
                "patient_id": target["patient_id"],
                "target_index": index,
                "seed": seed,
                "winner": result["winner"],
                "runner": result["runner"],
                "margin": result["margin"],
                "aggregate_norm": result["aggregate_norm"],
                "lead_contributions": lead_contributions,
                "beat_mass": mass_summary(result["beat_contributions"]),
                "token_mass": mass_summary(result["token_contributions"][result["active_tokens"]]),
                "intervention": intervention,
                "metrics": metric_values(result, intervention, lead_contributions),
                "errors": current,
            }
        )
        for name, value in result.items():
            if isinstance(value, np.ndarray):
                arrays[f"target_{index}_bank_{bank_index}_{name}"] = value
    consistency = {"patient_id": target["patient_id"], **bank_consistency(results)}
    return evaluations, consistency, errors


def coordinate_artifact(data: dict) -> list[dict]:
    """Bind each private array index to unchanged original region and unique token coordinates."""
    records = []
    for index, target in enumerate(data["selection"]["targets"]):
        ecg = target["ecg_id"]
        tokens = cached_tokens(data["tokens"][ecg], ecg)
        records.append(
            {
                "target_index": index,
                "ecg_id": ecg,
                "record_frame": tokens.record_frame,
                "duration_seconds": tokens.duration_seconds,
                "context_scope": tokens.context_scope,
                "embedding_space": tokens.embedding_space,
                "qrs_regions": data["coordinates"][str(ecg)]["qrs"],
                "token_intervals_seconds": tokens.intervals.tolist(),
                "token_leads": [list(leads) for leads in tokens.token_leads],
                "bank_order": list(SEEDS),
            }
        )
    return records


def run(output: Path) -> dict:
    """Execute the single source-frozen cached study and preserve all declared observations."""
    started = time.perf_counter()
    deadline = started + 1200
    data = replay_cached_inputs()
    sources = committed_sources(SOURCES, data["receipt"]["source_sha256"])
    evaluations, consistency, arrays = [], [], {}
    maximum = dict.fromkeys(
        ("pooling", "descriptor", "scores", "beat_sum", "token_sum", "beat_token_sum"), 0.0
    )
    for index, target in enumerate(data["selection"]["targets"]):
        if time.perf_counter() >= deadline:
            raise TimeoutError("Prospective contextual contribution budget exceeded")
        rows, pairs, errors = evaluate_target(data, target, index, arrays)
        evaluations.extend(rows)
        consistency.append(pairs)
        maximum = {name: max(maximum[name], errors[name]) for name in maximum}
        write_json_atomic(output / "progress.json", {"targets_completed": index + 1, "total": 60})
    gates = {
        "all_180_original_evaluations_retained": len(evaluations) == 180,
        "all_87_original_descriptors_reconstructed": data["replay"]["maximum_descriptor_error"] <= 1e-12,
        "original_scores_and_descriptor_exact": max(
            maximum[name] for name in ("pooling", "descriptor", "scores")
        )
        <= 1e-12,
        "signed_beat_and_token_sums_exact": max(
            maximum[name] for name in ("beat_sum", "token_sum", "beat_token_sum")
        )
        <= 1e-10,
    }
    write_json_atomic(output / "evaluations_local.json", evaluations)
    write_json_atomic(output / "bank_consistency_local.json", consistency)
    write_json_atomic(output / "selection_local.json", data["selection"])
    write_json_atomic(output / "coordinates_local.json", coordinate_artifact(data))
    np.savez_compressed(output / "contributions_local.npz", **arrays)
    artifacts = {
        to_stored(path): sha256_file(path)
        for path in sorted(output.iterdir())
        if path.name not in {"progress.json", "status.json"}
    }
    aggregate = patient_summaries(evaluations, consistency)
    if time.perf_counter() >= deadline:
        raise TimeoutError("Prospective contribution writing budget exceeded")
    result = {
        "study": "NTT-08",
        "status": "completed",
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "source_sha256": sources,
        "input_sha256": data["inputs"],
        "artifact_sha256": artifacts,
        "runtime": runtime_fingerprint(),
        "predecessor_reproduction": data["replay"],
        "primary_engineering_gate_passed": all(gates.values()),
        "gates": gates,
        "maximum_errors": maximum,
        "aggregate": aggregate,
        "deletion_fractions": list(FRACTIONS),
        "random_subsets_per_fraction": 100,
        "embedding_space": EMBEDDING_SPACE,
        "context_scope": CONTEXT_SCOPE,
        "waveform_reads": 0,
        "new_encoder_inferences": 0,
        "new_clinical_annotations": 0,
        "clinical_localization_evaluated": False,
        "closed_tests_read": False,
        "seconds": time.perf_counter() - started,
    }
    write_json_atomic(output / "result.json", result)
    if not all(gates.values()):
        raise ValueError("Prospective contextual decomposition engineering gate failed")
    return result
