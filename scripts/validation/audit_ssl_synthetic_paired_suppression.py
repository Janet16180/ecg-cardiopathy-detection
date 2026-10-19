"""Independently audit saved four-arm suppression evidence with zero neural operations."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import resource
import signal
import struct
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from ecg_experiment.ssl_synthetic_pretrained_saved_audit import (
    checkpoint_digest,
    noise_for_seed,
    require,
    verify_case,
)
from ecg_experiment.ssl_synthetic_randomized_saved_audit import exact_tree, reconstruct_case, verify_optimizer
from ecg_experiment.ssl_synthetic_replication_saved_audit import (
    hotspot_rows,
    initialization,
    panel_controls,
    panel_metrics,
    replication_summary,
    schedule_for_seed,
)
from ecg_experiment.ssl_synthetic_suppression_saved_audit import (
    ARMS,
    control_top_values,
    objective_components,
    suppression_summary,
    verify_components,
)
from ecg_experiment.synthetic_audit_io import compare_values, file_sha256, read_json, relative_file

RELATIVE = "outputs/ssl_synthetic_localization/generated_paired_suppression_v1"
CONFIG = "configs/ssl_synthetic_paired_suppression_v1.json"
ADMISSION = "configs/ssl_synthetic_paired_suppression_admission_v1.json"
CORE_HASHES = dict(
    zip(
        (
            "ecg_experiment/ssl_synthetic_suppression_saved_audit.py",
            "ecg_experiment/ssl_synthetic_replication_saved_audit.py",
            "ecg_experiment/ssl_synthetic_randomized_saved_audit.py",
            "ecg_experiment/ssl_synthetic_quality_saved_audit.py",
            "ecg_experiment/ssl_synthetic_pretrained_saved_audit.py",
            "ecg_experiment/ssl_synthetic_control_diagnosis_audit.py",
            "ecg_experiment/synthetic_audit_io.py",
            "ecg_experiment/intervals.py",
            "docs/ssl-synthetic-localization-paired-suppression-independent-audit-v1.md",
            "tests/test_ssl_synthetic_suppression_saved_audit.py",
        ),
        (
            "55aa6f1cec9edc12ccdc61216376cebd70c6dab8ac7fd382da396ac14ef27ee7",
            "13d3c6c5f9efd70e01a0565a9bde9285556799bad2450744690074a6d5eebdc6",
            "8a2c6a019691181a7b805b50da8215a7a16406fa23f8643a6b1075f6ecb53aad",
            "14d095ae8e54bf18da09d8b455d06a43f1fe740d7f3b2d3e7f701851cd89d1e6",
            "650eb6463bfe48aef307c86ee36145fa302dc3563da373ebff2abedffcef7c09",
            "219c4cd8a8ee2ee298b9dac49ce7ea11bc1a736d2b1500640999a7a6e455ce4b",
            "f9fa5ebff570c0c99ec7a74ecdea393bdab782060d11c6855da662cd3cc0555a",
            "b45103664e8c2e063b90c8223b64e7bbe6394e54796e0613538bd0b53dfe99af",
            "dd835b44359cb29ae93663bc46c3d79537c02d9d1eeb5857b5bca6d06054ba8e",
            "a7a3f8c3551a0e086dd040f279c3ea30d907b955f9717a193f0860ccf9628d6b",
        ),
        strict=True,
    )
)


def load_array(path: Path, shape: tuple[int, ...], dtype: Any) -> np.ndarray:
    """Open one saved array read-only and enforce its exact native geometry."""
    value = np.load(path, mmap_mode="r", allow_pickle=False)
    require(value.shape == shape and value.dtype == dtype, f"native geometry {path.name}")
    return value


def load_state(path: Path, reference: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Strictly load complete finite six-tensor model values on CPU."""
    state = torch.load(path, map_location="cpu", weights_only=True)
    checkpoint_digest(state)
    require(list(state) == list(reference), "model names and registration order")
    require(all(state[name].shape == reference[name].shape for name in reference), "model parameter shapes")
    return state


def generation_evidence(
    study: Path, parents: dict[str, Path], main: dict[str, Any], guard: Any
) -> dict[str, Any]:
    """Regenerate input noise and reconstruct every fresh array without generator execution."""
    public = parents["generator"]
    checkpoint = torch.load(public / "pytorch_model.bin", map_location="cpu", weights_only=True)
    model_digest, parameters = checkpoint_digest(checkpoint)
    require(
        model_digest == read_json(public / "summary.json")["model_state_sha256"],
        "released named tensor identity",
    )
    identity = {"initial_state_sha256": model_digest, "final_state_sha256": model_digest, "unchanged": True}
    compare_values(identity, read_json(study / "generator_state_identity.json"), "saved generator identity")
    profile = main["profile"]["generation"]
    require(
        profile["initial_state_sha256"] == profile["final_state_sha256"] == model_digest
        and profile["model_state_unchanged"] is True,
        "profile generator identity",
    )
    old_noise = load_array(public / "noise.npy", (4, 1, 8, 5000), np.float32)
    for index in range(4):
        require(
            np.array_equal(old_noise[index], noise_for_seed(2026102901 + index)), "original probe noise seed"
        )
        verify_case(
            np.load(public / f"case_{index}_raw.npy", allow_pickle=False),
            np.load(public / f"case_{index}_author_export.npy", allow_pickle=False),
            np.load(public / f"case_{index}_canonical12.npy", allow_pickle=False),
        )
    cpu = read_json(study / "public_generator_closure.json")
    compare_values(
        {
            "all_exact": True,
            "cases": [dict.fromkeys(("raw", "author_export", "canonical12"), True) for _ in range(4)],
            "forwards": 4,
            "model_state_sha256": model_digest,
        },
        cpu,
        "source-bound four CPU closures",
    )
    raw = load_array(study / "fresh_raw.npy", (64, 8, 5000), np.float32)
    noise = load_array(study / "fresh_noise.npy", (64, 1, 8, 5000), np.float32)
    observed = load_array(study / "fresh_observed.npy", (64, 4, 12, 5000), np.float32)
    support = load_array(study / "fresh_support.npy", (64, 4, 12, 5000), bool)
    strict = load_array(study / "fresh_strict_support.npy", (64, 2, 12, 5000), bool)
    rows = [json.loads(line) for line in (study / "fresh_cases.jsonl").read_text().splitlines()]
    require(len(rows) == 64, "complete fresh case count")
    old_observed = load_array(parents["learning"] / "observed.npy", (128, 4, 12, 5000), np.float32)
    old_shams = {hashlib.sha256(old_observed[index, 0].tobytes()).hexdigest() for index in range(128)}
    replication_observed = load_array(
        parents["replication"] / "fresh_observed.npy", (64, 4, 12, 5000), np.float32
    )
    old_shams.update(
        hashlib.sha256(replication_observed[index, 0].tobytes()).hexdigest() for index in range(64)
    )
    shams, raw_hashes, checks = set(), set(), 0
    for offset, index in enumerate(range(2000, 2064)):
        guard()
        require(np.array_equal(noise[offset], noise_for_seed(2026105000 + offset)), "fresh input RNG bytes")
        require(
            bool(np.isfinite(raw[offset]).all()) and bool((raw[offset].std(axis=1) > 1e-6).all()),
            "finite nonflat native raw",
        )
        expected_observed, expected_strict, expected_support, row = reconstruct_case(raw[offset], index)
        require(bool((expected_observed[0].std(axis=1) > 1e-6).all()), "nonflat canonical sham")
        for expected, actual in (
            (expected_observed, observed[offset]),
            (expected_strict, strict[offset]),
            (expected_support, support[offset]),
        ):
            require(expected.tobytes() == actual.tobytes(), "fresh final array reconstruction bytes")
        checks += compare_values(row, rows[offset], f"fresh case {index}")
        sham = hashlib.sha256(observed[offset, 0].tobytes()).hexdigest()
        raw_hash = hashlib.sha256(raw[offset].tobytes()).hexdigest()
        require(
            sham not in shams and sham not in old_shams and raw_hash not in raw_hashes,
            "fresh exact nonduplication",
        )
        shams.add(sham)
        raw_hashes.add(raw_hash)
        if offset < 3:
            for name, value in (
                ("raw", raw[offset]),
                ("observed", observed[offset]),
                ("strict_support", strict[offset]),
                ("support", support[offset]),
            ):
                require(
                    np.array_equal(
                        value,
                        np.load(
                            study / "generation_profile" / f"original_{offset}_{name}.npy", allow_pickle=False
                        ),
                    ),
                    "bank first-three saved profile equality",
                )
            checks += compare_values(
                row,
                read_json(study / "generation_profile" / f"original_{offset}_case.json"),
                "saved generation profile metadata",
            )
    for name in ("raw", "observed", "strict_support", "support"):
        original = np.load(study / "generation_profile" / f"original_0_{name}.npy", allow_pickle=False)
        replay = np.load(study / "generation_profile" / f"replay_0_{name}.npy", allow_pickle=False)
        require(
            original.dtype == replay.dtype
            and original.shape == replay.shape
            and original.tobytes() == replay.tobytes(),
            "saved generator same-input replay bytes",
        )
    checks += compare_values(
        rows[0], read_json(study / "generation_profile/replay_0_case.json"), "saved generator replay metadata"
    )
    compare_values(
        {
            "fresh_sham_sha256": sorted(shams),
            "fresh_raw_sha256": sorted(raw_hashes),
            "old_sham_count": len(old_shams),
            "fresh_count": 64,
            "disjoint": True,
            "first_three_profile_exact": True,
        },
        read_json(study / "fresh_uniqueness.json"),
        "saved uniqueness",
    )
    return {
        "model_state_sha256": model_digest,
        "parameters": parameters,
        "fresh_cases": 64,
        "fresh_shams_unique": True,
        "fresh_shams_disjoint_from_all192_old": True,
        "case_scalar_checks": checks,
        "saved_generator_replay_exact": True,
    }


def parents_from_config(root: Path, config: dict[str, Any], admission: dict[str, Any]) -> dict[str, Path]:
    """Confine the three exact old artifact roots declared by the two prospective configs."""
    return {
        label: relative_file(root, row["root"])
        for label, row in {**config["fixed_parents"], "replication": admission["replication"]}.items()
    }


def bindings(
    source_root: Path,
    study: Path,
    parents: dict[str, Path],
    main: dict[str, Any],
    config: dict[str, Any],
    admission: dict[str, Any],
    guard: Any,
) -> int:
    """Verify complete source snapshots and strictly allowed old/new artifact catalogs."""
    allowed = {
        label: row["sha256"]
        for label, row in {**config["fixed_parents"], "replication": admission["replication"]}.items()
    }
    require(main["input_sha256"] == allowed, "exact three-parent input allowlist")
    count = 0
    for label, catalog in allowed.items():
        for name, expected in catalog.items():
            require(
                file_sha256(relative_file(parents[label], name), guard) == expected,
                "allowed predecessor binding",
            )
            count += 1
    for name, expected in main["source_sha256"].items():
        for directory in (source_root, study / "source_snapshot"):
            require(file_sha256(relative_file(directory, name), guard) == expected, "source/snapshot binding")
            count += 1
    for kind in ("generated_input_sha256", "output_sha256"):
        for name, expected in main[kind].items():
            require(
                file_sha256(relative_file(study, name), guard) == expected, "immutable new payload binding"
            )
            count += 1
    return count


def predecessor_closure(study: Path, parents: dict[str, Path], guard: Any) -> dict[str, Any]:
    """Independently close all replication primary/conditional/range/gate metadata from saved rows."""
    fits = []
    for identifier in (1, 2, 3):
        guard()
        fits.append(
            {
                arm: read_json(parents["replication"] / f"fit_{identifier}" / arm / "fresh_metrics.json")[
                    "cases"
                ]
                for arm in ("supervised", "permuted")
            }
        )
    aggregate = replication_summary(fits)
    checks = compare_values(
        aggregate,
        read_json(parents["replication"] / "summary.json")["aggregate"],
        "independent predecessor replication",
    )
    checks += compare_values(
        {"all_exact": True, "aggregate": aggregate},
        read_json(study / "predecessor_closure.json"),
        "saved immediate closure",
    )
    for directory in parents.values():
        audit = read_json(directory / "independent_audit_v1/summary.json")
        require(
            audit["all_passed"] is True and audit["mismatch_count"] == 0, "passed predecessor technical audit"
        )
    return {"metric_scalar_checks": checks, "aggregate": aggregate}


def training_design(study: Path, parent: Path, main: dict[str, Any]) -> None:
    """Recheck unchanged original TRAIN-only targets, class weight and fixed spatial prior."""
    support = load_array(parent / "support.npy", (128, 4, 12, 5000), bool)
    positives = int(np.count_nonzero(support[:80]))
    require(not support[:80, :2].any(), "zero TRAIN control masks")
    weight = min(20.0, (support[:80].size - positives) / positives)
    expected = {
        "positive_weight": weight,
        "positive_cells": positives,
        "negative_cells": int(support[:80].size - positives),
        "train_backgrounds": 80,
        "evaluation_backgrounds": 48,
    }
    compare_values(expected, read_json(study / "training_design.json"), "unchanged TRAIN design")
    compare_values(expected, read_json(parent / "training_design.json"), "parent TRAIN design")
    require(main["positive_weight"] == weight, "same capped positive class weight")
    prior = load_array(parent / "spatial_prior.npy", (12, 5000), np.float64)
    require(
        np.array_equal(prior, support[:80, 2:].mean(axis=(0, 1), dtype=np.float64)), "TRAIN-only prior bytes"
    )


def states_and_schedules(study: Path, parent: Path, guard: Any) -> dict[str, Any]:
    """Check common starts, four-arm final Adam evidence and exact B replay for every fit."""
    result = {}
    for fit in (1, 2, 3):
        guard()
        directory = study / f"fit_{fit}"
        reference = initialization(2026103800 + fit)
        common = load_state(directory / "initial_state.pt", reference)
        require(exact_tree(common, reference), "declared common initialization bytes")
        require(
            exact_tree(
                common,
                torch.load(parent / f"fit_{fit}/initial_state.pt", map_location="cpu", weights_only=True),
            ),
            "matched replication initialization",
        )
        schedule = load_array(directory / "training_schedule.npy", (400, 8), np.int64)
        expected = schedule_for_seed(2026103900 + fit)
        require(
            np.array_equal(schedule, expected)
            and np.array_equal(
                schedule, np.load(parent / f"fit_{fit}/training_schedule.npy", allow_pickle=False)
            ),
            "same complete paired schedule",
        )
        require(
            np.array_equal(
                np.load(directory / "background_schedule.npy", allow_pickle=False), expected[:, [0, 4]] // 4
            ),
            "same background order",
        )
        compare_values(
            {"fit_id": fit, "initialization_seed": 2026103800 + fit, "schedule_seed": 2026103900 + fit},
            read_json(directory / "fit_design.json"),
            "fixed fit design",
        )
        arms = {}
        for arm in ARMS:
            guard()
            require(
                exact_tree(load_state(directory / arm / "initial_state.pt", reference), common),
                "untouched common arm start",
            )
            final = load_state(directory / arm / "final_state.pt", reference)
            optimizer = torch.load(directory / arm / "optimizer.pt", map_location="cpu", weights_only=True)
            verify_optimizer(optimizer, final, 400)
            require(not exact_tree(common, final), "scientific arm parameters changed")
            losses = load_array(directory / arm / "losses.npy", (400,), np.float64)
            components = load_array(directory / arm / "loss_components.npy", (400, 5), np.float64)
            verify_components(components, losses, arm)
            if arm == "B":
                previous = parent / f"fit_{fit}/supervised"
                require(
                    np.array_equal(losses, np.load(previous / "losses.npy", allow_pickle=False)),
                    "baseline all400 losses exact",
                )
                require(
                    exact_tree(
                        final, torch.load(previous / "final_state.pt", map_location="cpu", weights_only=True)
                    ),
                    "baseline full final model exact",
                )
                require(
                    exact_tree(
                        optimizer,
                        torch.load(previous / "optimizer.pt", map_location="cpu", weights_only=True),
                    ),
                    "baseline complete Adam exact",
                )
            arms[arm] = {
                "final_state_sha256": checkpoint_digest(final)[0],
                "adam_steps": 400,
                "loss_count": 400,
            }
        compare_values(
            dict.fromkeys(
                ("all_exact", "all_400_losses_exact", "final_model_exact", "complete_adam_exact"), True
            ),
            read_json(directory / "baseline_closure.json"),
            "saved B baseline closure",
        )
        result[str(fit)] = {"initial_state_sha256": checkpoint_digest(common)[0], "arms": arms}
    return result


def profile_recovery(study: Path, parent: Path, weight: float, guard: Any) -> dict[str, Any]:
    """Check full saved recovery and independently score each captured original first objective."""
    reference = initialization(2026103801)
    schedule = schedule_for_seed(2026103901)
    support = load_array(parent / "support.npy", (128, 4, 12, 5000), bool)[:80].reshape(320, 12, 5000)
    result, objective_evidence = {}, {}
    for arm in ARMS:
        guard()
        directory = study / "fit_1/profile" / arm
        first = load_state(directory / "step1_model.pt", reference)
        verify_optimizer(
            torch.load(directory / "step1_optimizer.pt", map_location="cpu", weights_only=True), first, 1
        )
        original_losses = load_array(directory / "original/losses.npy", (3,), np.float64)
        replay_losses = load_array(directory / "replay/losses.npy", (2,), np.float64)
        original = load_array(directory / "original/loss_components.npy", (3, 5), np.float64)
        replay = load_array(directory / "replay/loss_components.npy", (2, 5), np.float64)
        verify_components(original, original_losses, arm)
        verify_components(replay, replay_losses, arm)
        require(
            np.array_equal(original[1:], replay) and np.array_equal(original_losses[1:], replay_losses),
            "all recovered objective components and total losses exact",
        )
        for step in (2, 3):
            old = load_state(directory / f"original_step{step}_model.pt", reference)
            new = load_state(directory / f"replay_step{step}_model.pt", reference)
            require(exact_tree(old, new), "complete recovered profile model exact")
            old_adam = torch.load(
                directory / f"original_step{step}_optimizer.pt", map_location="cpu", weights_only=True
            )
            new_adam = torch.load(
                directory / f"replay_step{step}_optimizer.pt", map_location="cpu", weights_only=True
            )
            verify_optimizer(old_adam, old, step)
            verify_optimizer(new_adam, new, step)
            require(exact_tree(old_adam, new_adam), "complete recovered profile Adam exact")
        indices = load_array(directory / "step1_case_indices.npy", (8,), np.int64)
        require(np.array_equal(indices, schedule[0]), "first captured TRAIN batch identities")
        logits = load_array(directory / "step1_logits.npy", (8, 12, 5000), np.float32)
        independent = objective_components(logits, support[indices], weight, arm)
        error = float(np.max(np.abs(independent - original[0])))
        require(error <= 1e-6, "independent captured objective components absolute tolerance")
        summary = read_json(directory / "summary.json")
        for key, value in (
            ("losses", original_losses.tolist()),
            ("replay_losses", replay_losses.tolist()),
            ("components", original.tolist()),
            ("replay_components", replay.tolist()),
        ):
            compare_values(value, summary[key], "profile complete saved component rows", tolerance=0)
        require(
            summary["trajectory_state_exact"] == [True, True]
            and summary["exact_replay"] is True
            and summary["recurring_persistence_included"] is True,
            "full-path exact profile flags",
        )
        result[arm] = summary
        objective_evidence[arm] = {
            "maximum_absolute_error": error,
            "absolute_tolerance": 1e-6,
            "independent_components": independent.tolist(),
            "saved_components": original[0].tolist(),
        }
    return {"profiles": result, "first_objective": objective_evidence}


def metric_evidence(study: Path, parent: Path, main: dict[str, Any], guard: Any) -> dict[str, Any]:
    """Independently score all twelve maps and every primary/factorial/conditional endpoint."""
    observed = load_array(study / "fresh_observed.npy", (64, 4, 12, 5000), np.float32)
    support = load_array(study / "fresh_support.npy", (64, 4, 12, 5000), bool)
    prior = load_array(parent / "spatial_prior.npy", (12, 5000), np.float64)
    fits, checks = [], 0
    for fit in (1, 2, 3):
        record = {"fit_id": fit}
        for arm in ARMS:
            guard()
            probabilities = load_array(
                study / f"fit_{fit}" / arm / "fresh_probabilities.npy", (64, 4, 12, 5000), np.float32
            )
            rows = panel_metrics(probabilities, observed, support, prior, list(range(2000, 2064)))
            hotspots = hotspot_rows(probabilities, support, list(range(2000, 2064)))
            means = {
                "sham_nuisance_persistence": float(
                    np.mean([row["sham_nuisance_persistence"] for row in hotspots])
                )
            }
            for mode in ("focal", "repeated"):
                means[mode] = {
                    key: float(np.mean([row[mode][key] for row in hotspots])) for key in hotspots[0][mode]
                }
            metrics = {
                "cases": rows,
                "controls": panel_controls(probabilities),
                "hotspots": {"cases": hotspots, "means": means},
                "top600_control": control_top_values(probabilities),
            }
            checks += compare_values(
                metrics,
                read_json(study / f"fit_{fit}" / arm / "fresh_metrics.json"),
                "all independent arm metrics",
            )
            record[arm] = metrics
        fits.append(record)
    aggregate = suppression_summary(fits)
    checks += compare_values(aggregate, main["aggregate"], "all independent suppression aggregates")
    return {"metric_scalar_checks": checks, "scientific": aggregate}


def verified_accounting(
    study: Path, parents: dict[str, Path], main: dict[str, Any], guard: Any
) -> dict[str, Any]:
    """Capture actual once-only main costs before later corruption checks can fail."""
    path = study / "prospective_ledger.json"
    ledger = read_json(path)
    require(
        ledger["receipt"] == "summary.json"
        and ledger["receipt_sha256"] == file_sha256(study / "summary.json", guard),
        "main receipt/terminal binding",
    )
    prior_path = parents["replication"] / "independent_audit_v1/prospective_ledger.json"
    require(
        file_sha256(prior_path, guard)
        == ledger["prior_ledger_sha256"]
        == "f17f0c112a3eeed8b5f46be389c2a06ff8dcb8c2dd6169b1833a398f20a0d264",
        "immediate audited accounting anchor",
    )
    prior = read_json(prior_path)
    for field, literal in (("cpu_seconds", 190.681380118), ("gpu_wall_seconds", 178.56192114995793)):
        previous = prior[f"current_machine_cumulative_{field}"]
        require(
            previous == literal == ledger[f"prior_current_machine_{field}"], "literal verified previous costs"
        )
        require(np.isfinite(ledger[field]) and ledger[field] >= 0, "finite main self cost")
        require(
            ledger[f"current_machine_cumulative_{field}"] == previous + ledger[field],
            "once-only cumulative addition",
        )
    require(
        ledger["cpu_seconds"] >= main["cpu_seconds"]
        and ledger["gpu_wall_seconds"] == main["gpu_lock_wall_seconds"],
        "main self-cost summary consistency",
    )
    return {
        "main_cpu_seconds": ledger["cpu_seconds"],
        "main_wall_seconds": ledger["wall_seconds"],
        "main_gpu_wall_seconds": ledger["gpu_wall_seconds"],
        "main_ledger_sha256": file_sha256(path, guard),
        "current_machine_cpu_through_main": ledger["current_machine_cumulative_cpu_seconds"],
        "current_machine_gpu_wall_through_main": ledger["current_machine_cumulative_gpu_wall_seconds"],
    }


def main_resources(
    study: Path, parents: dict[str, Path], main: dict[str, Any], config: dict[str, Any], guard: Any
) -> dict[str, Any]:
    """Check complete allocations, full-path forecasts, terminal ceilings and actual final bytes."""
    costs = verified_accounting(study, parents, main, guard)
    profile = main["profile"]
    compare_values(profile, read_json(study / "combined_profile.json"), "saved combined profile")
    compare_values(
        profile["generation"], read_json(study / "generation_profile.json"), "saved generation profile"
    )
    generation, training = profile["generation"], profile["training"]
    require(set(training) == set(ARMS), "all four profiled objectives")
    require(
        len(generation["original_timings"]) == 3 and generation["first_input_exact_replay"] is True,
        "three generation originals and exact replay",
    )
    for kind in ("cpu", "wall"):
        generator = max(row[f"{kind}_seconds"] for row in generation["original_timings"][1:])
        update = max(row["timings"][step][f"{kind}_seconds"] for row in training.values() for step in (1, 2))
        inference = max(row["inference"][f"{kind}_seconds"] for row in training.values())
        compare_values(
            {"generation": generator, "update": update, "inference": inference},
            profile[f"recurrent_{kind}_seconds"],
            "worst recurring full-path timing",
        )
        future = 128 * generator + 9600 * update + 768 * inference + 90
        require(
            abs(profile["spent"][f"{kind}_seconds"] + future - profile[f"forecast_{kind}_seconds"]) <= 1e-12,
            "cold-once combined forecast",
        )
        require(0 < profile[f"forecast_{kind}_seconds"] < 1800, "CPU/wall admission forecast")
        if kind == "wall":
            require(
                abs(
                    profile["spent_gpu_lock_wall_seconds"]
                    + future
                    - profile["forecast_gpu_lock_wall_seconds"]
                )
                <= 1e-12,
                "GPU-lock forecast arithmetic",
            )
    require(0 < profile["forecast_gpu_lock_wall_seconds"] < 1200, "GPU-lock forecast cap")
    require(
        profile["peak_allocated_bytes"] < 2 * 1024**3 and 0 < profile["forecast_saved_bytes"] < 1024**3,
        "profile memory/bytes gate",
    )
    require(
        profile["profile_evidence_forecast_reserve_bytes"] == 8 * 1024**2
        and profile["captured_profile_logits_already_in_current_bytes"] is True,
        "prospective captured-evidence reserve",
    )
    counts = {
        "cpu_generator_attempts": 4,
        "cpu_generator_forwards": 4,
        "gpu_generator_profile_attempts": 4,
        "gpu_generator_profile_forwards": 4,
        "gpu_generator_bank_attempts": 64,
        "gpu_generator_bank_forwards": 64,
        "training_profile": {
            "profile_update_attempts": 20,
            "profile_updates": 20,
            "profile_inference_attempts": 4,
            "profile_inferences": 4,
        },
        "fits": {},
    }
    for fit in (1, 2, 3):
        values = {}
        for arm in ARMS:
            values[f"{arm}_update_attempts"] = values[f"{arm}_updates"] = 400
            values[f"{arm}_fresh_inference_attempts"] = values[f"{arm}_fresh_inferences"] = 32
        counts["fits"][str(fit)] = values
    compare_values(counts, main["counts"], "exact attempted/completed operation counts")
    expected = {
        "cpu_generator_forwards": 4,
        "gpu_generator_profile_forwards": 4,
        "gpu_generator_bank_forwards": 64,
        "profile_optimizer_updates": 20,
        "profile_inference_batches": 4,
        "baseline_replay_optimizer_updates": 1200,
        "new_objective_optimizer_updates": 3600,
        "fresh_inference_batches": 384,
        "old_panel_inference_batches": 0,
        "hyperparameter_search_updates": 0,
    }
    compare_values(expected, main["operation_totals"], "all authorized operations")
    compare_values(expected, config["allowed_operations"], "prospective operation allocation")
    compare_values(list(range(2000, 2064)), main["completed_fresh_indices"], "all fixed new backgrounds")
    metadata = main["gpu_metadata"]
    require(
        metadata["free_bytes_at_admission"] >= 2 * 1024**3 and metadata["peak_allocated_bytes"] < 2 * 1024**3,
        "actual GPU memory ceiling",
    )
    require(metadata["torch_threads"] == metadata["interop_threads"] == 1, "one intraop/interop thread")
    require(
        metadata["deterministic"] is True
        and metadata["matmul_tf32"] is False
        and metadata["cudnn_tf32"] is False,
        "deterministic full float32 path",
    )
    require(
        main["runtime_environment"]["CUBLAS_WORKSPACE_CONFIG"] == ":4096:8",
        "CUDA reproducibility environment",
    )
    for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        require(main["runtime_environment"][key] == "1", "single production numerical thread")
    ledger = read_json(study / "prospective_ledger.json")
    require(
        all(
            ledger[key] is True
            for key in (
                "prior_accounting_verified",
                "cpu_budget_passed",
                "wall_budget_passed",
                "gpu_budget_passed",
            )
        ),
        "actual terminal main flags",
    )
    require(
        main["cpu_seconds"] <= costs["main_cpu_seconds"] < 1800
        and main["wall_seconds"] <= costs["main_wall_seconds"] < 1800,
        "actual terminal CPU/wall caps",
    )
    require(0 < costs["main_gpu_wall_seconds"] < 1200, "actual shared GPU-lock ceiling")
    files = [path for path in study.rglob("*") if path.is_file() and "independent_audit_v1" not in path.parts]
    size = sum(path.stat().st_size for path in files)
    require(size < 1024**3, "actual main output-byte ceiling")
    require(
        size - (study / "prospective_ledger.json").stat().st_size
        == ledger["output_bytes_before_terminal_ledger"],
        "actual bytes before main terminal ledger",
    )
    require(
        size - (study / "prospective_ledger.json").stat().st_size - (study / "summary.json").stat().st_size
        == main["output_bytes_before_receipt"],
        "actual main bytes before summary",
    )
    catalog = {
        path.relative_to(study).as_posix()
        for path in files
        if path.parent != study or path.name not in {"summary.json", "prospective_ledger.json"}
    }
    require(catalog == set(main["output_sha256"]), "complete immutable output-byte catalog")
    return {
        **costs,
        "main_bytes": size,
        "prior_costs_added_once": True,
        "inference_profile_includes_map_persistence": False,
    }


def figures(study: Path, names: list[str]) -> None:
    """Verify frozen all-arm figure catalogs and file geometry without rerendering."""
    expected = {
        "factorial_diagnostics.png": (2160, 1080),
        "fixed_focal_factorial.png": (2800, 2000),
        "fixed_repeated_factorial.png": (2800, 2000),
    }
    require(names == list(expected), "fixed all-arm figure catalog")
    for name, dimensions in expected.items():
        with (study / name).open("rb") as stream:
            header = stream.read(24)
        require(
            header[:8] == b"\x89PNG\r\n\x1a\n" and struct.unpack(">II", header[16:24]) == dimensions,
            "fixed PNG geometry",
        )


def write_ledger(destination: Path, receipt: str, clock: tuple[float, float], costs: dict[str, Any]) -> None:
    """Record success/failure cost once after writing and hashing its receipt."""
    digest = file_sha256(destination / receipt)
    cpu, wall = time.process_time() - clock[0], time.perf_counter() - clock[1]
    size = sum(path.stat().st_size for path in destination.rglob("*") if path.is_file())
    ledger = {
        "receipt": receipt,
        "sha256": digest,
        "cpu_seconds": cpu,
        "wall_seconds": wall,
        "gpu_wall_seconds": 0.0,
        "main_ledger_sha256": costs.get("main_ledger_sha256"),
        "main_cpu_seconds": costs.get("main_cpu_seconds"),
        "stage_main_plus_audit_cpu_seconds": costs["main_cpu_seconds"] + cpu if costs else None,
        "current_machine_cumulative_cpu_seconds": costs["current_machine_cpu_through_main"] + cpu
        if costs
        else None,
        "current_machine_cumulative_gpu_wall_seconds": costs.get("current_machine_gpu_wall_through_main"),
        "audit_cpu_budget_passed": cpu < 300,
        "audit_wall_budget_passed": wall < 300,
        "audit_bytes_before_terminal_ledger": size,
        "historical_costs_verified": False,
        "measurement_endpoint": "after audit receipt write/hash, before terminal ledger serialization",
    }
    with (destination / "prospective_ledger.json").open("x") as stream:
        json.dump(ledger, stream, indent=2, allow_nan=False)
        stream.write("\n")
    require(
        sum(path.stat().st_size for path in destination.rglob("*") if path.is_file()) < 32 * 1024**2,
        "terminal audit byte cap",
    )
    require(cpu < 300 and wall < 300, "terminal audit time caps")


def run(root: Path, source_root: Path, main_sha: str, script_sha: str) -> dict[str, Any]:
    """Execute one source-frozen copied check with no model/optimizer operations."""
    clock = time.process_time(), time.perf_counter()
    study = root / RELATIVE
    destination = study / "independent_audit_v1"
    require(Path(__file__).absolute() == destination / "audit_script.py", "execute copied admitted checker")
    require(
        not (destination / "summary.json").exists() and not (destination / "failure.json").exists(),
        "audit already attempted",
    )
    costs: dict[str, Any] = {}
    receipt = "summary.json"

    def guard() -> None:
        require(time.process_time() - clock[0] < 298, "audit CPU deadline")
        require(time.perf_counter() - clock[1] < 298, "audit elapsed deadline")
        require(
            sum(path.stat().st_size for path in destination.rglob("*") if path.is_file()) < 31 * 1024**2,
            "audit storage reserve",
        )

    def interrupt(signum: int, frame: Any) -> None:
        raise TimeoutError(f"audit timer {signum}")

    for timer, signum in ((signal.ITIMER_REAL, signal.SIGALRM), (signal.ITIMER_PROF, signal.SIGPROF)):
        require(signal.getitimer(timer) == (0, 0), "exclusive audit timer")
        signal.signal(signum, interrupt)
        signal.setitimer(timer, 298)
    try:
        require(os.environ.get("CUDA_VISIBLE_DEVICES") == "", "CUDA hidden")
        for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
            require(os.environ.get(key) == "1", "one numerical audit thread")
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        require(file_sha256(Path(__file__), guard) == script_sha, "admitted copied checker bytes")
        require(len(CORE_HASHES) >= 8, "frozen complete independent arithmetic sources")
        for name, expected in CORE_HASHES.items():
            require(file_sha256(source_root / name, guard) == expected, "frozen audit arithmetic dependency")
        require(file_sha256(study / "summary.json", guard) == main_sha, "admitted main summary bytes")
        require(
            file_sha256(source_root / CONFIG, guard)
            == "12732cb09c545cf09d1f8f133acd5ec48f96b2eb48aefc9be691ad2d3654b083",
            "frozen scientific config",
        )
        require(
            file_sha256(source_root / ADMISSION, guard)
            == "d8076d042fe665ac95e80ef93ac50e6b650acdd355273ea4771d2d1b07158b89",
            "prospective identity/evidence admission",
        )
        main, config, admission = (
            read_json(study / "summary.json"),
            read_json(source_root / CONFIG),
            read_json(source_root / ADMISSION),
        )
        parents = parents_from_config(root, config, admission)
        costs = verified_accounting(study, parents, main, guard)
        count = bindings(source_root, study, parents, main, config, admission, guard)
        costs = main_resources(study, parents, main, config, guard)
        closure = predecessor_closure(study, parents, guard)
        generation = generation_evidence(study, parents, main, guard)
        training_design(study, parents["learning"], main)
        states = states_and_schedules(study, parents["replication"], guard)
        profiles = profile_recovery(study, parents["learning"], main["positive_weight"], guard)
        compare_values(
            profiles["profiles"], main["profile"]["training"], "complete profile recovery embedding"
        )
        compare_values(
            {
                identifier: {arm: row["final_state_sha256"] for arm, row in value["arms"].items()}
                for identifier, value in states.items()
            },
            main["final_state_sha256"],
            "all final named tensor digests",
        )
        require([row["fit_id"] for row in main["fit_designs"]] == [1, 2, 3], "all fixed fit identities")
        for design in main["fit_designs"]:
            require(
                design["initial_state_sha256"] == states[str(design["fit_id"])]["initial_state_sha256"],
                "initial named tensor identities",
            )
        metrics = metric_evidence(study, parents["learning"], main, guard)
        integrity = dict.fromkeys(
            (
                "immediate_replication_aggregate_exact",
                "public_CPU_replay_exact",
                "generator_profile_and_bank_replay_exact",
                "public_generator_unchanged",
                "fresh64_qualified_disjoint_all192",
                "four_objective_profile_recoveries_exact",
                "all_three_BCE_replays_exact",
                "all_operations_exact",
                "resources",
            ),
            True,
        )
        compare_values(integrity, main["integrity_gates"], "all integrity predicates")
        compare_values(
            {
                str(index): dict.fromkeys(
                    ("all_exact", "all_400_losses_exact", "final_model_exact", "complete_adam_exact"), True
                )
                for index in (1, 2, 3)
            },
            main["baseline_closures"],
            "full exact B baseline closure embedding",
        )
        require(
            main["all_passed"] == (metrics["scientific"]["scientific_passed"] and all(integrity.values())),
            "overall scientific/integrity conjunction",
        )
        compare_values(
            ["bce", "hinge", "consistency", "control", "total"],
            main["loss_component_columns"],
            "exact component meaning/order",
        )
        for key in (
            "old_panel_inference_batches",
            "hyperparameter_search_updates",
            "real_waveform_reads",
            "external_api_calls",
        ):
            require(main[key] == 0, f"zero {key}")
        require(
            main["clinical_claim"] is False and main["physical_units_verified"] is False, "claim boundaries"
        )
        figures(study, main["figures"])
        require(
            bindings(source_root, study, parents, main, config, admission, guard) == count,
            "all consumed bytes unchanged",
        )
        require(file_sha256(study / "summary.json", guard) == main_sha, "main receipt unchanged")
        require(
            file_sha256(study / "prospective_ledger.json", guard) == costs["main_ledger_sha256"],
            "main terminal ledger unchanged",
        )
        require(file_sha256(Path(__file__), guard) == script_sha, "checker unchanged")
        for name, expected in CORE_HASHES.items():
            require(file_sha256(source_root / name, guard) == expected, "frozen independent source unchanged")
        guard()
        result = {
            "all_passed": True,
            "mismatch_count": 0,
            "main_sha256": main_sha,
            "script_sha256": script_sha,
            "core_sha256": CORE_HASHES,
            "bindings_verified": count,
            "predecessor": closure,
            "generation": generation,
            "states": states,
            "profile_objective_evidence": profiles["first_objective"],
            **metrics,
            "resources": costs,
            "independent_model_forwards": 0,
            "independent_optimizer_updates": 0,
            "saved_profile_recovery_verified": True,
            "training_runtime_source_bound": True,
            "gpu_used": False,
            "raw_real_ecg_reads": 0,
            "clinical_claim": False,
            "peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
            "cpu_seconds": time.process_time() - clock[0],
            "wall_seconds": time.perf_counter() - clock[1],
        }
        with (destination / receipt).open("x") as stream:
            json.dump(result, stream, indent=2, allow_nan=False)
            stream.write("\n")
        return result
    except Exception as error:
        receipt = "failure.json"
        with (destination / receipt).open("x") as stream:
            json.dump(
                {
                    "all_passed": False,
                    "failure_type": type(error).__name__,
                    "error": str(error),
                    "main_sha256": main_sha,
                    "script_sha256": script_sha,
                    "cpu_seconds": time.process_time() - clock[0],
                    "wall_seconds": time.perf_counter() - clock[1],
                },
                stream,
                indent=2,
                allow_nan=False,
            )
            stream.write("\n")
        raise
    finally:
        write_ledger(destination, receipt, clock, costs)
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.setitimer(signal.ITIMER_PROF, 0)


def main() -> None:
    """Parse authorized roots and explicit admitted receipt/source identities."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--main-sha", required=True)
    parser.add_argument("--script-sha", required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.root.absolute(), args.source_root.absolute(), args.main_sha, args.script_sha)))


if __name__ == "__main__":
    main()
