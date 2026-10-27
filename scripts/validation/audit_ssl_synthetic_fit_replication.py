"""Independently verify retained fit-replication arrays, states and arithmetic without inference."""

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

from ecg_experiment.ssl_synthetic_control_diagnosis_audit import (
    seed_values,
)
from ecg_experiment.ssl_synthetic_control_diagnosis_audit import (
    summarize_rows as diagnosis_summary,
)
from ecg_experiment.ssl_synthetic_pretrained_saved_audit import (
    checkpoint_digest,
    noise_for_seed,
    require,
    verify_case,
)
from ecg_experiment.ssl_synthetic_randomized_saved_audit import (
    exact_tree,
    reconstruct_case,
    verify_optimizer,
)
from ecg_experiment.ssl_synthetic_randomized_saved_audit import (
    summarize_rows as learning_summary,
)
from ecg_experiment.ssl_synthetic_replication_saved_audit import (
    contrast_matrices,
    hotspot_rows,
    initialization,
    panel_controls,
    panel_metrics,
    replication_summary,
    resampled_summaries,
    schedule_for_seed,
)
from ecg_experiment.synthetic_audit_io import compare_values, file_sha256, read_json, relative_file

RELATIVE = "outputs/ssl_synthetic_localization/generated_fit_replication_v1"
CONFIG = "configs/ssl_synthetic_fit_replication_v1.json"
ARMS = ("supervised", "permuted")
CORE_HASHES = dict(
    zip(
        (
            "ecg_experiment/ssl_synthetic_replication_saved_audit.py",
            "ecg_experiment/ssl_synthetic_randomized_saved_audit.py",
            "ecg_experiment/ssl_synthetic_quality_saved_audit.py",
            "ecg_experiment/ssl_synthetic_pretrained_saved_audit.py",
            "ecg_experiment/ssl_synthetic_control_diagnosis_audit.py",
            "ecg_experiment/synthetic_audit_io.py",
            "ecg_experiment/intervals.py",
            "docs/ssl-synthetic-localization-fit-replication-independent-audit-v1.md",
            "tests/test_ssl_synthetic_replication_saved_audit.py",
        ),
        (
            "13d3c6c5f9efd70e01a0565a9bde9285556799bad2450744690074a6d5eebdc6",
            "8a2c6a019691181a7b805b50da8215a7a16406fa23f8643a6b1075f6ecb53aad",
            "14d095ae8e54bf18da09d8b455d06a43f1fe740d7f3b2d3e7f701851cd89d1e6",
            "650eb6463bfe48aef307c86ee36145fa302dc3563da373ebff2abedffcef7c09",
            "219c4cd8a8ee2ee298b9dac49ce7ea11bc1a736d2b1500640999a7a6e455ce4b",
            "f9fa5ebff570c0c99ec7a74ecdea393bdab782060d11c6855da662cd3cc0555a",
            "b45103664e8c2e063b90c8223b64e7bbe6394e54796e0613538bd0b53dfe99af",
            "06e98aa97dd0fbb8c0d572f7f852deccd76eb2d6fb934035af5e40fc23ad2aa8",
            "41cd3f22f9017b7641e25d43017d37f9387288a43e2824f59fbac38b222514ab",
        ),
        strict=True,
    )
)


def load_array(path: Path, shape: tuple[int, ...], dtype: Any) -> np.ndarray:
    """Open one saved array read-only and enforce its exact native geometry."""
    value = np.load(path, mmap_mode="r", allow_pickle=False)
    require(value.shape == shape and value.dtype == dtype, f"native geometry {path.name}")
    return value


def parent_roots(root: Path, config: dict[str, Any]) -> dict[str, Path]:
    """Identify the three explicitly admitted predecessor directories."""
    require(set(config["parents"]) == {"generator", "learning", "diagnosis"}, "three allowed parents")
    return {name: relative_file(root, value["root"]) for name, value in config["parents"].items()}


def bindings(
    source_root: Path,
    study: Path,
    parents: dict[str, Path],
    main: dict[str, Any],
    config: dict[str, Any],
    guard: Any,
) -> int:
    """Bind all sources, declared parent inputs and generated artifacts to exact bytes."""
    count = 0
    require(set(main["input_sha256"]) == set(parents), "exact input parent labels")
    for label, parent in parents.items():
        require(main["input_sha256"][label] == config["parents"][label]["sha256"], "literal parent allowlist")
        for name, expected in main["input_sha256"][label].items():
            require(
                file_sha256(relative_file(parent, name), guard) == expected, f"parent binding {label}/{name}"
            )
            count += 1
    for name, expected in main["source_sha256"].items():
        for directory in (source_root, study / "source_snapshot"):
            require(file_sha256(relative_file(directory, name), guard) == expected, f"source binding {name}")
            count += 1
    for key in ("generated_input_sha256", "output_sha256"):
        for name, expected in main[key].items():
            require(file_sha256(relative_file(study, name), guard) == expected, f"new output binding {name}")
            count += 1
    return count


def predecessor_closure(study: Path, parents: dict[str, Path], guard: Any) -> dict[str, Any]:
    """Recompute full old learning endpoints and the three latest diagnostic primaries."""
    parent = parents["learning"]
    observed = load_array(parent / "observed.npy", (128, 4, 12, 5000), np.float32)
    support = load_array(parent / "support.npy", (128, 4, 12, 5000), bool)
    prior = load_array(parent / "spatial_prior.npy", (12, 5000), np.float64)
    require(
        np.array_equal(prior, support[:80, 2:].mean(axis=(0, 1), dtype=np.float64)),
        "unchanged training-only prior",
    )
    maps, rows, checks = {}, {}, 0
    for arm in ARMS:
        guard()
        maps[arm] = load_array(parent / arm / "evaluation_probabilities.npy", (48, 4, 12, 5000), np.float32)
        rows[arm] = panel_metrics(maps[arm], observed[80:], support[80:], prior, list(range(80, 128)))
        saved = read_json(parent / arm / "evaluation_metrics.json")
        checks += compare_values(rows[arm], saved["cases"], "old complete case metrics")
        checks += compare_values(panel_controls(maps[arm]), saved["controls"], "old controls")
    scientific = learning_summary(rows["supervised"], rows["permuted"])
    old_main = read_json(parent / "summary.json")
    for key, value in scientific.items():
        checks += compare_values(value, old_main[key], f"old {key}")
    checks += compare_values(
        {"all_exact": True, "metric_case_rows": 96, "counts": old_main["counts"], **scientific},
        read_json(study / "predecessor_closure.json"),
        "saved old learning closure",
    )
    diagnostic_rows = []
    for offset, index in enumerate(range(80, 128)):
        guard()
        diagnostic_rows.append(
            seed_values(
                {arm: maps[arm][offset] for arm in ARMS}, observed[index], support[index], prior, index
            )
        )
    primary = diagnosis_summary(diagnostic_rows)["primary"]
    checks += compare_values(
        primary, read_json(parents["diagnosis"] / "summary.json")["primary"], "old diagnostic primaries"
    )
    checks += compare_values(
        {"all_exact": True, "primary": primary},
        read_json(study / "diagnostic_primary_closure.json"),
        "saved diagnostic closure",
    )
    for directory in parents.values():
        require(
            read_json(directory / "independent_audit_v1/summary.json")["all_passed"] is True,
            "passed parent audit",
        )
        require(
            read_json(directory / "independent_audit_v1/summary.json")["mismatch_count"] == 0,
            "zero parent audit mismatches",
        )
    return {"metric_scalar_checks": checks, "learning": scientific, "diagnostic_primary": primary}


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
    shams, raw_hashes, checks = set(), set(), 0
    for offset, index in enumerate(range(1000, 1064)):
        guard()
        require(np.array_equal(noise[offset], noise_for_seed(2026104000 + offset)), "fresh input RNG bytes")
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
        "fresh_shams_disjoint_from_all128_old": True,
        "case_scalar_checks": checks,
        "saved_generator_replay_exact": True,
    }


def load_state(path: Path, reference: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Strictly load complete finite six-tensor model values on CPU."""
    state = torch.load(path, map_location="cpu", weights_only=True)
    checkpoint_digest(state)
    require(list(state) == list(reference), "model names and registration order")
    require(all(state[name].shape == reference[name].shape for name in reference), "model parameter shapes")
    return state


def states_and_schedules(study: Path, parent: Path, guard: Any) -> dict[str, Any]:
    """Check every saved initialization, training plan, Adam count and exact old-pair replay."""
    result = {}
    for fit in range(4):
        guard()
        directory = study / f"fit_{fit}"
        seed = 2026103501 if fit == 0 else 2026103800 + fit
        schedule_seed = 2026103502 if fit == 0 else 2026103900 + fit
        reference = initialization(seed)
        common = load_state(directory / "initial_state.pt", reference)
        require(exact_tree(common, reference), "declared initialization RNG bytes")
        if fit == 0:
            require(
                exact_tree(
                    common, torch.load(parent / "initial_state.pt", map_location="cpu", weights_only=True)
                ),
                "old common initialization exact",
            )
        schedule = load_array(directory / "training_schedule.npy", (400, 8), np.int64)
        expected_schedule = schedule_for_seed(schedule_seed)
        require(np.array_equal(schedule, expected_schedule), "declared independent training schedule")
        require(
            np.array_equal(
                np.load(directory / "background_schedule.npy", allow_pickle=False),
                expected_schedule[:, [0, 4]] // 4,
            ),
            "paired background schedule",
        )
        compare_values(
            {"fit_id": fit, "initialization_seed": seed, "schedule_seed": schedule_seed},
            read_json(directory / "fit_design.json"),
            "frozen fit design",
        )
        evidence = {"initial_state_sha256": checkpoint_digest(common)[0], "arms": {}}
        closure = {}
        for arm in ARMS:
            guard()
            initial = load_state(directory / arm / "initial_state.pt", reference)
            require(exact_tree(initial, common), "matched untouched scientific initialization")
            final = load_state(directory / arm / "final_state.pt", reference)
            require(not exact_tree(initial, final), "scientific model changed")
            optimizer = torch.load(directory / arm / "optimizer.pt", map_location="cpu", weights_only=True)
            verify_optimizer(optimizer, final, 400)
            losses = load_array(directory / arm / "losses.npy", (400,), np.float64)
            require(
                bool(np.isfinite(losses).all()) and bool((losses >= 0).all()),
                "finite nonnegative complete losses",
            )
            if fit == 0:
                require(
                    np.array_equal(losses, np.load(parent / arm / "losses.npy", allow_pickle=False)),
                    "old replay all400 losses exact",
                )
                require(
                    exact_tree(
                        final,
                        torch.load(parent / arm / "final_state.pt", map_location="cpu", weights_only=True),
                    ),
                    "old full final model exact",
                )
                require(
                    exact_tree(
                        optimizer,
                        torch.load(parent / arm / "optimizer.pt", map_location="cpu", weights_only=True),
                    ),
                    "old full Adam exact",
                )
                closure[arm] = dict.fromkeys(
                    ("all_400_losses_exact", "final_model_exact", "complete_adam_exact"), True
                )
            evidence["arms"][arm] = {
                "final_state_sha256": checkpoint_digest(final)[0],
                "adam_steps": 400,
                "loss_count": 400,
            }
        result[str(fit)] = evidence
        if fit == 0:
            compare_values(
                {"all_exact": True, "arms": closure},
                read_json(directory / "old_training_closure.json"),
                "saved old training closure",
            )
    require(
        len({value["initial_state_sha256"] for value in result.values()}) == 4,
        "four distinct fit initializations",
    )
    return result


def profile_recovery(study: Path, guard: Any) -> dict[str, Any]:
    """Verify retained full-model, full-Adam and loss recovery at both original steps."""
    reference = initialization(2026103501)
    summaries = {}
    for arm in ARMS:
        guard()
        directory = study / "fit_0/profile" / arm
        step1 = load_state(directory / "step1_model.pt", reference)
        verify_optimizer(
            torch.load(directory / "step1_optimizer.pt", map_location="cpu", weights_only=True), step1, 1
        )
        summary = read_json(directory / "summary.json")
        require(summary["losses"][1:] == summary["replay_losses"], "profile loss recovery exactly")
        require(
            summary["trajectory_state_exact"] == [True, True] and summary["exact_replay"] is True,
            "source-bound profile recovery flags",
        )
        for step in (2, 3):
            original = load_state(directory / f"original_step{step}_model.pt", reference)
            replay = load_state(directory / f"replay_step{step}_model.pt", reference)
            require(exact_tree(original, replay), "complete recovered profile model bytes")
            first = torch.load(
                directory / f"original_step{step}_optimizer.pt", map_location="cpu", weights_only=True
            )
            second = torch.load(
                directory / f"replay_step{step}_optimizer.pt", map_location="cpu", weights_only=True
            )
            verify_optimizer(first, original, step)
            verify_optimizer(second, replay, step)
            require(exact_tree(first, second), "complete recovered profile Adam bytes")
        summaries[arm] = summary
    return summaries


def metric_evidence(study: Path, parent: Path, main: dict[str, Any], guard: Any) -> dict[str, Any]:
    """Independently rescore every saved panel and the complete crossed primary decision."""
    old_observed = load_array(parent / "observed.npy", (128, 4, 12, 5000), np.float32)
    old_support = load_array(parent / "support.npy", (128, 4, 12, 5000), bool)
    prior = load_array(parent / "spatial_prior.npy", (12, 5000), np.float64)
    fresh_observed = load_array(study / "fresh_observed.npy", (64, 4, 12, 5000), np.float32)
    fresh_support = load_array(study / "fresh_support.npy", (64, 4, 12, 5000), bool)
    fresh_fits, conditional, spent_conditional, absolute, checks = [], {}, {}, {}, 0
    for fit in range(4):
        arms = {}
        absolute[str(fit)] = {}
        for panel in ("fresh",) if fit == 0 else ("fresh", "spent"):
            count = 64 if panel == "fresh" else 48
            indices = list(range(1000, 1064)) if panel == "fresh" else list(range(80, 128))
            observed, support = (
                (fresh_observed, fresh_support) if panel == "fresh" else (old_observed[80:], old_support[80:])
            )
            rows_by_arm = {}
            absolute[str(fit)][panel] = {}
            for arm in ARMS:
                guard()
                directory = study / f"fit_{fit}" / arm
                probabilities = load_array(
                    directory / f"{panel}_probabilities.npy", (count, 4, 12, 5000), np.float32
                )
                rows = panel_metrics(probabilities, observed, support, prior, indices)
                hotspots = hotspot_rows(probabilities, support, indices)
                hotspot_means = {
                    "sham_nuisance_persistence": float(
                        np.mean([row["sham_nuisance_persistence"] for row in hotspots])
                    )
                }
                for mode in ("focal", "repeated"):
                    hotspot_means[mode] = {
                        key: float(np.mean([row[mode][key] for row in hotspots])) for key in hotspots[0][mode]
                    }
                metrics = {
                    "cases": rows,
                    "controls": panel_controls(probabilities),
                    "hotspots": {"cases": hotspots, "means": hotspot_means},
                }
                checks += compare_values(
                    metrics, read_json(directory / f"{panel}_metrics.json"), "full independent panel metrics"
                )
                rows_by_arm[arm] = rows
                absolute[str(fit)][panel][arm] = {
                    mode: {key: float(np.mean([row[mode][key] for row in rows])) for key in rows[0][mode]}
                    for mode in ("focal", "repeated")
                }
            if panel == "fresh":
                arms = rows_by_arm
            else:
                spent_conditional[str(fit)] = learning_summary(
                    rows_by_arm["supervised"], rows_by_arm["permuted"]
                )["contrasts"]
        conditional[str(fit)] = resampled_summaries(contrast_matrices([arms]), False)
        if fit:
            fresh_fits.append(arms)
    aggregate = replication_summary(fresh_fits)
    expected = {
        "aggregate": aggregate,
        "conditional_fresh_contrasts": conditional,
        "conditional_spent_contrasts": spent_conditional,
        "absolute_mode_means": absolute,
        "fit0_spent_reference_contrasts": read_json(parent / "summary.json")["contrasts"],
        "fit0_spent_maps_reused_without_inference": True,
    }
    for key, value in expected.items():
        checks += compare_values(value, main[key], f"replication summary {key}")
    return {"metric_scalar_checks": checks, "scientific": aggregate}


def main_resources(
    root: Path,
    study: Path,
    parents: dict[str, Path],
    main: dict[str, Any],
    config: dict[str, Any],
    guard: Any,
) -> dict[str, Any]:
    """Check full-path forecasts, exact operation allocations and once-only terminal costs."""
    profile = main["profile"]
    compare_values(profile, read_json(study / "combined_profile.json"), "combined profile")
    generation, training = profile["generation"], profile["training"]
    compare_values(generation, read_json(study / "generation_profile.json"), "generation profile embedding")
    require(
        len(generation["original_timings"]) == 3 and generation["first_input_exact_replay"] is True,
        "three generation profiles plus replay",
    )
    for kind in ("cpu", "wall"):
        generator = max(row[f"{kind}_seconds"] for row in generation["original_timings"][1:])
        update = max(row["timings"][step][f"{kind}_seconds"] for row in training.values() for step in (1, 2))
        inference = max(row["inference"][f"{kind}_seconds"] for row in training.values())
        compare_values(
            {"generation": generator, "update": update, "inference": inference},
            profile[f"recurrent_{kind}_seconds"],
            "worst recurrent timings",
        )
        future = 128 * generator + 6400 * update + 800 * inference + 90
        require(
            abs(profile["spent"][f"{kind}_seconds"] + future - profile[f"forecast_{kind}_seconds"]) <= 1e-12,
            "cold-once combined forecast",
        )
        require(0 < profile[f"forecast_{kind}_seconds"] < 1800, "CPU/wall forecast admission")
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
    require(0 < profile["forecast_gpu_lock_wall_seconds"] < 1200, "GPU forecast admission")
    require(
        0 < profile["forecast_saved_bytes"] < 1024**3 and profile["peak_allocated_bytes"] < 2 * 1024**3,
        "profile byte/memory admission",
    )
    counts = {
        "cpu_generator_attempts": 4,
        "cpu_generator_forwards": 4,
        "gpu_generator_profile_attempts": 4,
        "gpu_generator_profile_forwards": 4,
        "gpu_generator_bank_attempts": 64,
        "gpu_generator_bank_forwards": 64,
        "training_profile": {
            "profile_update_attempts": 10,
            "profile_updates": 10,
            "profile_inference_attempts": 2,
            "profile_inferences": 2,
        },
        "fits": {},
    }
    for fit in range(4):
        values = {}
        for arm in ARMS:
            values[arm + "_update_attempts"] = values[arm + "_updates"] = 400
            values[arm + "_fresh_inference_attempts"] = values[arm + "_fresh_inferences"] = 32
            values[arm + "_spent_inference_attempts"] = values[arm + "_spent_inferences"] = (
                0 if fit == 0 else 24
            )
        counts["fits"][str(fit)] = values
    compare_values(counts, main["counts"], "all attempted/completed operations")
    compare_values(
        {
            "cpu_generator_forwards": 4,
            "gpu_generator_forwards": 68,
            "profile_optimizer_updates": 10,
            "old_replay_optimizer_updates": 800,
            "new_fit_optimizer_updates": 2400,
            "profile_localizer_inference_batches": 2,
            "fresh_localizer_inference_batches": 256,
            "spent_localizer_inference_batches": 144,
            "model_selection_updates": 0,
        },
        main["operation_totals"],
        "complete operations",
    )
    compare_values(main["operation_totals"], config["allowed_operations"], "prospective operation allocation")
    compare_values(list(range(1000, 1064)), main["completed_fresh_indices"], "complete fresh seeds")
    metadata = main["gpu_metadata"]
    require(
        metadata["free_bytes_at_admission"] >= 2 * 1024**3 and metadata["peak_allocated_bytes"] < 2 * 1024**3,
        "GPU memory limits",
    )
    require(metadata["torch_threads"] == metadata["interop_threads"] == 1, "scientific single threads")
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
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        require(main["runtime_environment"][name] == "1", "production numerical thread settings")
    ledger_path = study / "prospective_ledger.json"
    ledger = read_json(ledger_path)
    require(
        ledger["receipt"] == "summary.json"
        and ledger["receipt_sha256"] == file_sha256(study / "summary.json", guard),
        "main terminal receipt binding",
    )
    require(0 <= main["cpu_seconds"] <= ledger["cpu_seconds"] < 1800, "actual terminal main CPU")
    require(0 <= main["wall_seconds"] <= ledger["wall_seconds"] < 1800, "actual terminal main wall")
    require(
        ledger["gpu_wall_seconds"] == main["gpu_lock_wall_seconds"] and 0 < ledger["gpu_wall_seconds"] < 1200,
        "actual charged GPU lock",
    )
    require(
        all(
            ledger[key] is True
            for key in (
                "cpu_budget_passed",
                "wall_budget_passed",
                "gpu_budget_passed",
                "prior_accounting_verified",
            )
        ),
        "main terminal flags",
    )
    prior_path = parents["diagnosis"] / "independent_audit_v1/prospective_ledger.json"
    require(
        file_sha256(prior_path, guard)
        == ledger["prior_ledger_sha256"]
        == "a2e2c31b6eda8a5aaed2498e69f6ee2bed97bd90d6b8b3d133e6c15a0f617f77",
        "latest cumulative ledger anchor",
    )
    prior = read_json(prior_path)
    for field, literal in (("cpu_seconds", 44.99976453399999), ("gpu_wall_seconds", 38.433311437256634)):
        previous = prior[f"current_machine_cumulative_{field}"]
        require(previous == literal == ledger[f"prior_current_machine_{field}"], "literal previous costs")
        require(
            ledger[f"current_machine_cumulative_{field}"] == previous + ledger[field],
            "once-only cumulative addition",
        )
    files = [path for path in study.rglob("*") if path.is_file() and "independent_audit_v1" not in path.parts]
    final_bytes = sum(path.stat().st_size for path in files)
    require(
        final_bytes - ledger_path.stat().st_size == ledger["output_bytes_before_terminal_ledger"],
        "main bytes before terminal ledger",
    )
    require(
        final_bytes - ledger_path.stat().st_size - (study / "summary.json").stat().st_size
        == main["output_bytes_before_receipt"],
        "main bytes before receipt",
    )
    require(final_bytes < 1024**3, "actual final main output cap")
    catalog = {
        path.relative_to(study).as_posix()
        for path in files
        if path.name not in {"summary.json", "prospective_ledger.json"} or path.parent != study
    }
    require(catalog == set(main["output_sha256"]), "complete main output catalog")
    return {
        "main_cpu_seconds": ledger["cpu_seconds"],
        "main_wall_seconds": ledger["wall_seconds"],
        "main_gpu_wall_seconds": ledger["gpu_wall_seconds"],
        "main_ledger_sha256": file_sha256(ledger_path, guard),
        "main_bytes": final_bytes,
        "current_machine_cpu_through_main": ledger["current_machine_cumulative_cpu_seconds"],
        "current_machine_gpu_wall_through_main": ledger["current_machine_cumulative_gpu_wall_seconds"],
        "prior_costs_added_once": True,
    }


def training_inputs(study: Path, parent: Path, main: dict[str, Any]) -> None:
    """Recheck unchanged training truth, donor rotation and positive class weight."""
    support = load_array(parent / "support.npy", (128, 4, 12, 5000), bool)
    train = support[:80]
    donors = load_array(study / "donor_map.npy", (80,), np.int64)
    require(np.array_equal(donors, (np.arange(80) + 1) % 80), "train-only donor derangement")
    require(
        np.array_equal(donors, np.load(parent / "donor_map.npy", allow_pickle=False)),
        "unchanged donor mapping",
    )
    positives = int(np.count_nonzero(train))
    require(
        positives == np.count_nonzero(train[donors, 2:]) and not train[:, :2].any(),
        "same-mode equal-count targets and zero controls",
    )
    weight = min(20.0, float((train.size - positives) / positives))
    design = {
        "positive_weight": weight,
        "positive_cells": positives,
        "negative_cells": int(train.size - positives),
        "train_backgrounds": 80,
        "evaluation_backgrounds": 48,
    }
    compare_values(design, read_json(parent / "training_design.json"), "old unchanged training design")
    compare_values(design, read_json(study / "training_design.json"), "saved unchanged training design")
    require(weight == main["positive_weight"], "training-only class weight")


def figures(study: Path, names: list[str]) -> None:
    """Bind fixed figure names and PNG geometry without rerendering numerical scores."""
    expected = {
        "fit_diagnostics.png": (2160, 1080),
        "fixed_focal_maps.png": (2200, 2200),
        "fixed_repeated_maps.png": (2200, 2200),
    }
    require(names == list(expected), "fixed figure catalog")
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


def verified_accounting(
    study: Path, parents: dict[str, Path], main: dict[str, Any], guard: Any
) -> dict[str, Any]:
    """Capture verified costs before a later artifact-corruption check can fail."""
    path = study / "prospective_ledger.json"
    ledger = read_json(path)
    require(
        ledger["receipt"] == "summary.json"
        and ledger["receipt_sha256"] == file_sha256(study / "summary.json", guard),
        "early receipt binding",
    )
    prior_path = parents["diagnosis"] / "independent_audit_v1/prospective_ledger.json"
    require(
        file_sha256(prior_path, guard)
        == ledger["prior_ledger_sha256"]
        == "a2e2c31b6eda8a5aaed2498e69f6ee2bed97bd90d6b8b3d133e6c15a0f617f77",
        "early prior accounting anchor",
    )
    prior = read_json(prior_path)
    for field in ("cpu_seconds", "gpu_wall_seconds"):
        previous = prior[f"current_machine_cumulative_{field}"]
        require(previous == ledger[f"prior_current_machine_{field}"], "early prior costs")
        require(np.isfinite(ledger[field]) and ledger[field] >= 0, "early finite main self cost")
        require(
            ledger[f"current_machine_cumulative_{field}"] == previous + ledger[field],
            "early once-only addition",
        )
    require(
        ledger["cpu_seconds"] >= main["cpu_seconds"]
        and ledger["gpu_wall_seconds"] == main["gpu_lock_wall_seconds"],
        "early self-cost summary agreement",
    )
    return {
        "main_cpu_seconds": ledger["cpu_seconds"],
        "main_wall_seconds": ledger["wall_seconds"],
        "main_gpu_wall_seconds": ledger["gpu_wall_seconds"],
        "main_ledger_sha256": file_sha256(path, guard),
        "current_machine_cpu_through_main": ledger["current_machine_cumulative_cpu_seconds"],
        "current_machine_gpu_wall_through_main": ledger["current_machine_cumulative_gpu_wall_seconds"],
    }


def run(root: Path, source_root: Path, main_sha: str, script_sha: str) -> dict[str, Any]:
    """Run exactly one admitted copied checker with no neural execution."""
    clock = time.process_time(), time.perf_counter()
    study = root / RELATIVE
    destination = study / "independent_audit_v1"
    require(Path(__file__).absolute() == destination / "audit_script.py", "execute admitted copied checker")
    require(
        not (destination / "summary.json").exists() and not (destination / "failure.json").exists(),
        "audit already attempted",
    )
    costs: dict[str, Any] = {}
    receipt = "summary.json"

    def guard() -> None:
        require(time.process_time() - clock[0] < 298, "audit CPU deadline")
        require(time.perf_counter() - clock[1] < 298, "audit wall deadline")
        require(
            sum(path.stat().st_size for path in destination.rglob("*") if path.is_file()) < 31 * 1024**2,
            "audit output reserve",
        )

    def interrupt(signum: int, frame: Any) -> None:
        raise TimeoutError(f"audit timer {signum}")

    for timer, sig in ((signal.ITIMER_REAL, signal.SIGALRM), (signal.ITIMER_PROF, signal.SIGPROF)):
        require(signal.getitimer(timer) == (0, 0), "exclusive audit timer")
        signal.signal(sig, interrupt)
        signal.setitimer(timer, 298)
    try:
        require(os.environ.get("CUDA_VISIBLE_DEVICES") == "", "CUDA hidden")
        for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
            require(os.environ.get(key) == "1", "single audit numerical thread")
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        require(file_sha256(Path(__file__), guard) == script_sha, "admitted checker bytes")
        require(len(CORE_HASHES) >= 7, "complete frozen independent arithmetic source catalog")
        for name, expected in CORE_HASHES.items():
            require(file_sha256(source_root / name, guard) == expected, "frozen audit source")
        require(file_sha256(study / "summary.json", guard) == main_sha, "admitted main summary")
        require(
            file_sha256(source_root / CONFIG, guard)
            == "83094ae866b5481fd82704c6031215de1fba680a855cc7b43215298ae3f6a318",
            "frozen main prospective config",
        )
        main, config = read_json(study / "summary.json"), read_json(source_root / CONFIG)
        parents = parent_roots(root, config)
        costs = verified_accounting(study, parents, main, guard)
        count = bindings(source_root, study, parents, main, config, guard)
        costs = main_resources(root, study, parents, main, config, guard)
        closure = predecessor_closure(study, parents, guard)
        generation = generation_evidence(study, parents, main, guard)
        training_inputs(study, parents["learning"], main)
        states = states_and_schedules(study, parents["learning"], guard)
        recovery = profile_recovery(study, guard)
        compare_values(recovery, main["profile"]["training"], "saved full recovery embedding")
        compare_values(
            {
                identifier: {arm: value["final_state_sha256"] for arm, value in fit["arms"].items()}
                for identifier, fit in states.items()
            },
            main["trained"]["final_state_sha256"],
            "all final named tensor identities",
        )
        for design in main["fit_designs"]:
            require(
                design["initial_state_sha256"] == states[str(design["fit_id"])]["initial_state_sha256"],
                "initial named tensor identities",
            )
        require(
            len(main["fit_designs"]) == 4
            and [row["fit_id"] for row in main["fit_designs"]] == list(range(4)),
            "all fixed fit identities",
        )
        metrics = metric_evidence(study, parents["learning"], main, guard)
        integrity = dict.fromkeys(
            (
                "old_learning_metrics_exact",
                "immediate_diagnosis_primary_exact",
                "old_public_cpu_generation_exact",
                "generator_profile_replay_exact",
                "generator_state_unchanged",
                "fresh_64_qualified_unique_and_disjoint",
                "bank_first_three_profile_exact",
                "training_profile_recovery_exact",
                "old_400_step_pair_exact",
                "all_operations_exact",
                "resources",
            ),
            True,
        )
        compare_values(integrity, main["integrity_gates"], "full integrity conjunction")
        require(
            main["all_passed"] == (metrics["scientific"]["scientific_passed"] and all(integrity.values())),
            "scientific/integrity overall conjunction",
        )
        for key in (
            "reserved_successor_backgrounds_generated",
            "model_selection_updates",
            "real_waveform_reads",
            "external_api_calls",
        ):
            require(main[key] == 0, f"zero {key}")
        require(
            main["clinical_claim"] is False and main["physical_units_verified"] is False, "claim boundaries"
        )
        figures(study, main["figures"])
        require(bindings(source_root, study, parents, main, config, guard) == count, "all bindings unchanged")
        require(file_sha256(study / "summary.json", guard) == main_sha, "main receipt unchanged")
        require(
            file_sha256(study / "prospective_ledger.json", guard) == costs["main_ledger_sha256"],
            "main terminal ledger unchanged",
        )
        require(file_sha256(Path(__file__), guard) == script_sha, "checker unchanged")
        for name, expected in CORE_HASHES.items():
            require(file_sha256(source_root / name, guard) == expected, "audit core unchanged")
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
    """Parse explicit source/output roots and admitted receipt identities."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--main-sha", required=True)
    parser.add_argument("--script-sha", required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.root.absolute(), args.source_root.absolute(), args.main_sha, args.script_sha)))


if __name__ == "__main__":
    main()
