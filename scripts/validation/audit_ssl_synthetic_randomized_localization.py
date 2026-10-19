"""Independently audit saved randomized-support learning without model inference."""

from __future__ import annotations

import argparse
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

from ecg_experiment.ssl_synthetic_pretrained_saved_audit import checkpoint_digest, require
from ecg_experiment.ssl_synthetic_quality_saved_audit import coherence_values, primary_values
from ecg_experiment.ssl_synthetic_randomized_saved_audit import (
    control_rows,
    exact_tree,
    expected_schedule,
    initial_state,
    reconstruct_case,
    score_rows,
    summarize_rows,
    verify_optimizer,
)
from ecg_experiment.synthetic_audit_io import compare_values, file_sha256, read_json, relative_file

RELATIVE = "outputs/ssl_synthetic_localization/randomized_localization_v1"
PARENT = "outputs/ssl_synthetic_localization/generated_quality_v2"
CORE_HASHES = dict(
    zip(
        (
            "ecg_experiment/ssl_synthetic_randomized_saved_audit.py",
            "ecg_experiment/synthetic_audit_io.py",
            "ecg_experiment/ssl_synthetic_pretrained_saved_audit.py",
            "ecg_experiment/ssl_synthetic_quality_saved_audit.py",
            "ecg_experiment/intervals.py",
        ),
        (
            "8a2c6a019691181a7b805b50da8215a7a16406fa23f8643a6b1075f6ecb53aad",
            "f9fa5ebff570c0c99ec7a74ecdea393bdab782060d11c6855da662cd3cc0555a",
            "650eb6463bfe48aef307c86ee36145fa302dc3563da373ebff2abedffcef7c09",
            "14d095ae8e54bf18da09d8b455d06a43f1fe740d7f3b2d3e7f701851cd89d1e6",
            "b45103664e8c2e063b90c8223b64e7bbe6394e54796e0613538bd0b53dfe99af",
        ),
        strict=True,
    )
)


def verify_bindings(source_root: Path, study: Path, parent: Path, main: dict[str, Any], guard: Any) -> int:
    """Verify every allowed source, immutable fitting input and output binding.

    Parameters
    ----------
    source_root, study, parent : Path
        Authorized source and artifact roots.
    main : dict
        Admitted main summary.
    guard : callable
        Deadline and byte guard.

    Returns
    -------
    int
        Checked binding count.
    """
    count = 0
    for name, expected in main["source_sha256"].items():
        for path in (relative_file(source_root, name), relative_file(study / "source_snapshot", name)):
            require(file_sha256(path, guard) == expected, f"source binding {name}")
            count += 1
    for directory, catalog in (
        (parent, main["input_sha256"]),
        (study, main["bank_sha256"]),
        (study, main["output_sha256"]),
    ):
        for name, expected in catalog.items():
            require(
                file_sha256(relative_file(directory, name), guard) == expected, f"artifact binding {name}"
            )
            count += 1
    return count


def verify_bank(study: Path, parent: Path, main: dict[str, Any], guard: Any) -> dict[str, Any]:
    """Rebuild randomized data, split/schedule and immediate predecessor closure.

    Parameters
    ----------
    study, parent : Path
        New bank and admitted parent native bank.
    main : dict
        Admitted main metadata.
    guard : callable
        Audit resource guard.

    Returns
    -------
    dict[str, Any]
        Reconstructed data integrity and class-weight evidence.
    """
    raw = np.load(parent / "raw.npy", mmap_mode="r", allow_pickle=False)
    arrays = {
        name: np.load(study / f"{name}.npy", mmap_mode="r", allow_pickle=False)
        for name in ("observed", "support", "strict_support")
    }
    for name, shape, dtype in (
        ("observed", (128, 4, 12, 5000), np.float32),
        ("support", (128, 4, 12, 5000), bool),
        ("strict_support", (128, 2, 12, 5000), bool),
    ):
        require(arrays[name].shape == shape and arrays[name].dtype == dtype, f"{name} geometry")
    require(raw.shape == (128, 8, 5000) and raw.dtype == np.float32, "source bank geometry")
    rows = [json.loads(line) for line in (study / "cases.jsonl").read_text().splitlines()]
    require(len(rows) == 128, "complete case rows")
    checks, coherent = 0, []
    for index in range(128):
        guard()
        observed, strict, support, row = reconstruct_case(raw[index], index)
        require(observed.tobytes() == arrays["observed"][index].tobytes(), "observed reconstruction bytes")
        require(
            strict.tobytes() == arrays["strict_support"][index].tobytes(), "strict mask reconstruction bytes"
        )
        require(support.tobytes() == arrays["support"][index].tobytes(), "support reconstruction bytes")
        require(not bool(support[:2].any()), "zero control target masks")
        row.update(sham_exact=True, limb_equations=True)
        checks += compare_values(row, rows[index], f"case{index}")
        coherent.append(coherence_values(observed[0, [0, 1, 6, 7, 8, 9, 10, 11]], index))
    predecessor_primary = primary_values(coherent)
    compare_values(predecessor_primary, read_json(parent / "summary.json")["primary"], "parent primary")
    closure = read_json(study / "predecessor_closure.json")
    compare_values(predecessor_primary, closure["primary"], "saved predecessor closure")
    require(
        closure["all_exact"] is True and closure["cases"] == 128 and closure["new_generator_forwards"] == 0,
        "predecessor complete closure",
    )
    schedule = np.load(study / "training_schedule.npy", allow_pickle=False)
    expected = expected_schedule()
    require(schedule.dtype == np.int64 and np.array_equal(schedule, expected), "400 matched schedule batches")
    background = np.load(study / "background_schedule.npy", allow_pickle=False)
    require(np.array_equal(background, expected[:, [0, 4]] // 4), "background schedule closure")
    donors = np.load(study / "donor_map.npy", allow_pickle=False)
    require(np.array_equal(donors, (np.arange(80) + 1) % 80), "training-only target derangement")
    train = arrays["support"][:80]
    positives = int(np.count_nonzero(train))
    permuted_positives = int(np.count_nonzero(train[donors, 2:]))
    require(
        positives == permuted_positives and not bool(train[:, :2].any()), "equal control target collections"
    )
    weight = float(min(20, (train.size - positives) / positives))
    require(weight == main["positive_weight"], "training-only positive weight")
    compare_values(
        {
            "positive_weight": weight,
            "positive_cells": positives,
            "negative_cells": int(train.size - positives),
            "train_backgrounds": 80,
            "evaluation_backgrounds": 48,
        },
        read_json(study / "training_design.json"),
        "training design",
    )
    return {
        "case_scalar_checks": checks,
        "complete_reconstruction": True,
        "schedule_exact": True,
        "train_evaluation_disjoint": True,
        "positive_train_cells": positives,
        "negative_train_cells": int(train.size - positives),
        "positive_weight": weight,
        "predecessor_primary": predecessor_primary,
    }


def load_state(path: Path, reference: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Strictly load and validate a complete finite model parameter state.

    Parameters
    ----------
    path : Path
        Admitted saved model state.
    reference : dict
        CPU-generated common initialization shape contract.

    Returns
    -------
    dict
        Finite CPU parameter tensors in registration order.
    """
    state = torch.load(path, map_location="cpu", weights_only=True)
    checkpoint_digest(state)
    require(list(state) == list(reference), "exact model parameter names/order")
    require(all(state[name].shape == reference[name].shape for name in reference), "model parameter shapes")
    return state


def verify_states(study: Path, guard: Any) -> dict[str, Any]:
    """Validate matched initialization, final Adam evidence and complete profile recovery.

    Parameters
    ----------
    study : Path
        Model/checkpoint output directory.
    guard : callable
        Audit resource guard.

    Returns
    -------
    dict[str, Any]
        Model digests and independently verified saved recovery identities.
    """
    reference = initial_state()
    common = load_state(study / "initial_state.pt", reference)
    require(exact_tree(reference, common), "frozen initialization seed bytes")
    result = {"initial_state_sha256": checkpoint_digest(common)[0], "arms": {}}
    for arm in ("supervised", "permuted"):
        guard()
        initial = load_state(study / arm / "initial_state.pt", reference)
        require(exact_tree(common, initial), "identical fresh arm initialization")
        final = load_state(study / arm / "final_state.pt", reference)
        optimizer = torch.load(study / arm / "optimizer.pt", map_location="cpu", weights_only=True)
        verify_optimizer(optimizer, final, 400)
        losses = np.load(study / arm / "losses.npy", allow_pickle=False)
        require(losses.dtype == np.float64 and losses.shape == (400,), "400 native saved losses")
        require(bool(np.isfinite(losses).all()) and bool((losses >= 0).all()), "finite nonnegative losses")
        require(not exact_tree(common, final), "parameters changed by scientific fit")
        profile = study / "profile" / arm
        step_one = load_state(profile / "step1_model.pt", reference)
        first_optimizer = torch.load(profile / "step1_optimizer.pt", map_location="cpu", weights_only=True)
        verify_optimizer(first_optimizer, step_one, 1)
        summary = read_json(profile / "summary.json")
        require(summary["losses"][1:] == summary["replay_losses"], "saved recovery losses exactly equal")
        require(
            summary["trajectory_state_exact"] == [True, True] and summary["exact_replay"] is True,
            "source-bound recovery verdicts",
        )
        for step in (2, 3):
            original = load_state(profile / f"original_step{step}_model.pt", reference)
            replay = load_state(profile / f"replay_step{step}_model.pt", reference)
            require(exact_tree(original, replay), "complete profile model recovery bytes")
            original_optimizer = torch.load(
                profile / f"original_step{step}_optimizer.pt", map_location="cpu", weights_only=True
            )
            replay_optimizer = torch.load(
                profile / f"replay_step{step}_optimizer.pt", map_location="cpu", weights_only=True
            )
            verify_optimizer(original_optimizer, original, step)
            verify_optimizer(replay_optimizer, replay, step)
            require(exact_tree(original_optimizer, replay_optimizer), "complete optimizer recovery bytes")
        result["arms"][arm] = {
            "final_state_sha256": checkpoint_digest(final)[0],
            "adam_steps": 400,
            "loss_count": 400,
            "saved_step2_and_step3_recovery_exact": True,
        }
    return result


def verify_metrics(study: Path, main: dict[str, Any], guard: Any) -> dict[str, Any]:
    """Independently score maps and the full prospective scientific decision.

    Parameters
    ----------
    study : Path
        Saved maps and target bank.
    main : dict
        Bound main summary.
    guard : callable
        Audit resource guard.

    Returns
    -------
    dict[str, Any]
        Recomputed full scientific summary and checked scalar count.
    """
    observed = np.load(study / "observed.npy", mmap_mode="r", allow_pickle=False)[80:]
    support = np.load(study / "support.npy", mmap_mode="r", allow_pickle=False)[80:]
    all_support = np.load(study / "support.npy", mmap_mode="r", allow_pickle=False)
    expected_prior = np.mean(all_support[:80, 2:], axis=(0, 1), dtype=np.float64)
    prior = np.load(study / "spatial_prior.npy", allow_pickle=False)
    require(prior.dtype == np.float64 and prior.shape == (12, 5000), "training spatial prior geometry")
    require(np.array_equal(prior, expected_prior), "training-only spatial prior")
    arm_rows, checks = {}, 0
    for arm in ("supervised", "permuted"):
        guard()
        probability = np.load(study / arm / "evaluation_probabilities.npy", mmap_mode="r", allow_pickle=False)
        rows = score_rows(probability, observed, support, prior)
        saved = read_json(study / arm / "evaluation_metrics.json")
        checks += compare_values(rows, saved["cases"], f"{arm} case metrics")
        checks += compare_values(control_rows(probability), saved["controls"], f"{arm} controls")
        original = read_json(study / arm / "metrics.json")
        without_prior = [
            {
                **row,
                **{
                    mode: {key: value for key, value in row[mode].items() if key != "spatial_prior_ap"}
                    for mode in ("focal", "repeated")
                },
            }
            for row in rows
        ]
        checks += compare_values(without_prior, original["cases"], f"{arm} original case metrics")
        checks += compare_values(control_rows(probability), original["controls"], f"{arm} original controls")
        arm_rows[arm] = rows
    summary = summarize_rows(arm_rows["supervised"], arm_rows["permuted"])
    for key, value in summary.items():
        checks += compare_values(value, main[key], f"scientific/{key}")
    return {"scientific": summary, "metric_scalar_checks": checks}


def verify_resources(
    root: Path, study: Path, parent: Path, main: dict[str, Any], guard: Any
) -> dict[str, Any]:
    """Recompute complete-path forecasts, update counts and current-machine ledger arithmetic.

    Parameters
    ----------
    root, study, parent : Path
        Output host and authorized stage roots.
    main : dict
        Bound main summary.
    guard : callable
        Audit deadline guard.

    Returns
    -------
    dict[str, Any]
        Terminal stage and predecessor costs and bytes.
    """
    profile = read_json(study / "profile.json")
    compare_values(profile, main["profile"], "profile embedding")
    arms = profile["arms"]
    require(set(arms) == {"supervised", "permuted"}, "both profiled arms")
    recurrent, inference = {}, {}
    for kind in ("cpu", "wall"):
        recurrent[kind] = max(
            row["timings"][step][f"{kind}_seconds"] for row in arms.values() for step in (1, 2)
        )
        inference[kind] = max(row["inference"][f"{kind}_seconds"] for row in arms.values())
        forecast = profile["spent"][f"{kind}_seconds"] + 1600 * recurrent[kind] + 96 * inference[kind] + 60
        require(abs(forecast - profile[f"forecast_{kind}_seconds"]) <= 1e-12, "whole-stage profile forecast")
        require(forecast < 1800, "whole-stage admission")
    gpu_forecast = (
        profile["spent_gpu_lock_wall_seconds"] + 1600 * recurrent["wall"] + 96 * inference["wall"] + 60
    )
    require(
        abs(gpu_forecast - profile["forecast_gpu_lock_wall_seconds"]) <= 1e-12, "GPU-lock profile forecast"
    )
    require(gpu_forecast < 1200 and profile["peak_allocated_bytes"] < 2 * 1024**3, "GPU profile admission")
    require(profile["forecast_saved_bytes"] < 512 * 1024**2, "profile output forecast")
    counts = {
        "profile_update_attempts": 10,
        "profile_updates": 10,
        "profile_inference_attempts": 2,
        "profile_inferences": 2,
    }
    for arm in ("supervised", "permuted"):
        counts[arm + "_update_attempts"] = counts[arm + "_updates"] = 400
        counts[arm + "_inference_attempts"] = counts[arm + "_inferences"] = 24
        compare_values(arms[arm], read_json(study / "profile" / arm / "summary.json"), "profile arm metadata")
    compare_values(counts, main["counts"], "all profile/scientific update and inference counts")
    metadata = main["gpu_metadata"]
    require(metadata["free_bytes_at_admission"] >= 2 * 1024**3, "GPU admission free bytes")
    require(metadata["peak_allocated_bytes"] < 2 * 1024**3, "actual GPU peak allocation")
    require(
        metadata["deterministic"] is True
        and metadata["matmul_tf32"] is False
        and metadata["cudnn_tf32"] is False,
        "deterministic float32 GPU path",
    )
    require(metadata["torch_threads"] == metadata["interop_threads"] == 1, "single scientific threads")
    require(0 < main["gpu_lock_wall_seconds"] < 1200, "charged complete GPU lock")
    require(0 < main["cpu_seconds"] < 1800 and 0 < main["wall_seconds"] < 1800, "scientific time caps")
    ledger = read_json(study / "prospective_ledger.json")
    require(
        ledger["receipt"] == "summary.json"
        and ledger["receipt_sha256"] == file_sha256(study / "summary.json", guard),
        "main terminal ledger binding",
    )
    require(main["cpu_seconds"] <= ledger["cpu_seconds"] < 1800, "terminal main CPU")
    require(main["wall_seconds"] <= ledger["wall_seconds"] < 1800, "terminal main wall")
    require(ledger["gpu_wall_seconds"] == main["gpu_lock_wall_seconds"], "terminal GPU cost")
    prior_path = parent / "independent_audit_v1/prospective_ledger.json"
    require(
        file_sha256(prior_path, guard) == ledger["prior_quality_audit_ledger_sha256"],
        "prior additive ledger binding",
    )
    prior = read_json(prior_path)
    prior_cpu = prior["current_machine_cumulative_cpu_seconds"]
    prior_gpu = prior["current_machine_cumulative_gpu_wall_seconds"]
    require(
        ledger["prior_current_machine_cpu_seconds"] == prior_cpu
        and ledger["prior_current_machine_gpu_wall_seconds"] == prior_gpu,
        "current-machine prior costs",
    )
    require(
        ledger["current_machine_cumulative_cpu_seconds"] == prior_cpu + ledger["cpu_seconds"],
        "additive CPU ledger",
    )
    require(
        ledger["current_machine_cumulative_gpu_wall_seconds"] == prior_gpu + ledger["gpu_wall_seconds"],
        "additive GPU ledger",
    )
    actual = sum(
        path.stat().st_size
        for path in study.rglob("*")
        if path.is_file() and "independent_audit_v1" not in path.parts
    )
    receipts = sum((study / name).stat().st_size for name in ("summary.json", "prospective_ledger.json"))
    require(actual - receipts == main["output_bytes_before_receipt"], "main exact bytes before receipt")
    require(actual < 512 * 1024**2, "actual main byte cap")
    return {
        "main_cpu_seconds": ledger["cpu_seconds"],
        "main_wall_seconds": ledger["wall_seconds"],
        "main_gpu_wall_seconds": ledger["gpu_wall_seconds"],
        "main_bytes": actual,
        "main_ledger_sha256": file_sha256(study / "prospective_ledger.json", guard),
        "current_machine_cpu_through_main": ledger["current_machine_cumulative_cpu_seconds"],
        "current_machine_gpu_wall_through_main": ledger["current_machine_cumulative_gpu_wall_seconds"],
        "profile_counts": {"optimizer_updates": 10, "training_inference_batches": 2},
        "main_counts": {"optimizer_updates": 800, "evaluation_inference_batches": 48},
    }


def write_terminal_ledger(
    destination: Path, receipt_name: str, clock: tuple[float, float], costs: dict[str, Any]
) -> None:
    """Charge own audit terminal cost once after writing and hashing its receipt.

    Parameters
    ----------
    destination : Path
        Audit output directory.
    receipt_name : str
        Success/failure receipt filename.
    clock : tuple
        CPU and monotonic start values.
    costs : dict
        Verified main/prior ledger arithmetic, absent if admission failed.
    """
    digest = file_sha256(destination / receipt_name)
    cpu, wall = time.process_time() - clock[0], time.perf_counter() - clock[1]
    with (destination / "prospective_ledger.json").open("x") as stream:
        json.dump(
            {
                "receipt": receipt_name,
                "sha256": digest,
                "cpu_seconds": cpu,
                "wall_seconds": wall,
                "gpu_wall_seconds": 0.0,
                "main_cpu_seconds": costs.get("main_cpu_seconds"),
                "stage_main_plus_audit_cpu_seconds": costs["main_cpu_seconds"] + cpu if costs else None,
                "current_machine_cumulative_cpu_seconds": costs["current_machine_cpu_through_main"] + cpu
                if costs
                else None,
                "current_machine_cumulative_gpu_wall_seconds": costs.get(
                    "current_machine_gpu_wall_through_main"
                ),
                "historical_costs_verified": False,
                "audit_budget_passed": cpu < 180 and wall < 180,
                "measurement_endpoint": "after audit receipt write/hash, before ledger serialization",
            },
            stream,
            indent=2,
            allow_nan=False,
        )
        stream.write("\n")


def verify_figures(study: Path, files: list[str]) -> None:
    """Verify fixed first-eight and diagnostic figure artifact geometry.

    Parameters
    ----------
    study : Path
        Authorized artifact root.
    files : list[str]
        Main declared figure catalog.
    """
    expected = {
        "first_eight_focal.png": (2200, 2200),
        "first_eight_repeated.png": (2200, 2200),
        "learning_diagnostics.png": (1920, 480),
    }
    require(files == list(expected), "fixed figure catalog")
    for name, shape in expected.items():
        with (study / name).open("rb") as stream:
            header = stream.read(24)
        require(header[:8] == b"\x89PNG\r\n\x1a\n", "figure PNG signature")
        require(struct.unpack(">II", header[16:24]) == shape, "figure dimensions")


def run(root: Path, source_root: Path, main_sha: str, script_sha: str) -> dict[str, Any]:
    """Run one admitted frozen copied audit without inference or optimizer updates.

    Parameters
    ----------
    root, source_root : Path
        Authorized output host and source worktree.
    main_sha, script_sha : str
        Explicit coordinator-admitted SHA-256 values.

    Returns
    -------
    dict[str, Any]
        Independent arithmetic and saved-training-integrity report.
    """
    clock = time.process_time(), time.perf_counter()
    study, parent = root / RELATIVE, root / PARENT
    destination = study / "independent_audit_v1"
    require(Path(__file__).absolute() == destination / "audit_script.py", "execute exact copied checker")
    require(
        not (destination / "summary.json").exists() and not (destination / "failure.json").exists(),
        "audit previously attempted",
    )
    costs: dict[str, Any] = {}
    receipt_name = "summary.json"

    def guard() -> None:
        require(time.process_time() - clock[0] < 178, "audit CPU deadline")
        require(time.perf_counter() - clock[1] < 178, "audit wall deadline")
        actual = sum(path.stat().st_size for path in destination.rglob("*") if path.is_file())
        require(actual < 15 * 1024**2, "audit byte reserve")

    def interrupt(signum: int, frame: Any) -> None:
        raise TimeoutError(f"audit timer {signum}")

    for timer, sig in ((signal.ITIMER_REAL, signal.SIGALRM), (signal.ITIMER_PROF, signal.SIGPROF)):
        require(signal.getitimer(timer) == (0, 0), "exclusive audit timer")
        signal.signal(sig, interrupt)
        signal.setitimer(timer, 178)
    try:
        require(os.environ.get("CUDA_VISIBLE_DEVICES") == "", "CUDA hidden")
        for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
            require(os.environ.get(key) == "1", "single numerical thread")
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        require(file_sha256(Path(__file__), guard) == script_sha, "admitted checker bytes")
        require(len(CORE_HASHES) >= 5, "complete independent arithmetic source catalog")
        for name, expected in CORE_HASHES.items():
            require(file_sha256(source_root / name, guard) == expected, "frozen independent arithmetic core")
        require(file_sha256(study / "summary.json", guard) == main_sha, "admitted main bytes")
        main = read_json(study / "summary.json")
        count = verify_bindings(source_root, study, parent, main, guard)
        costs = verify_resources(root, study, parent, main, guard)
        bank = verify_bank(study, parent, main, guard)
        states = verify_states(study, guard)
        require(
            states["initial_state_sha256"] == main["initial_state_sha256"], "initial named tensor identity"
        )
        compare_values(
            {arm: value["final_state_sha256"] for arm, value in states["arms"].items()},
            main["final_state_sha256"],
            "final named tensor identities",
        )
        metrics = verify_metrics(study, main, guard)
        integrity_names = (
            "parent_exact",
            "all_128_supports_qualified",
            "matched_initial_states",
            "profile_exact",
            "profile_updates_10",
            "profile_inferences_2",
            "main_updates_400_each",
            "evaluation_batches_24_each",
            "resources",
        )
        compare_values(dict.fromkeys(integrity_names, True), main["integrity_gates"], "integrity conjunction")
        require(
            main["all_passed"]
            == (metrics["scientific"]["scientific_passed"] and all(main["integrity_gates"].values())),
            "overall conjunction",
        )
        for key in ("new_generator_forwards", "real_waveform_reads", "external_api_calls"):
            require(main[key] == 0, f"zero {key}")
        require(
            main["clinical_claim"] is False and main["physical_units_verified"] is False, "claim boundaries"
        )
        verify_figures(study, main["figures"])
        require(verify_bindings(source_root, study, parent, main, guard) == count, "all bindings unchanged")
        require(file_sha256(study / "summary.json", guard) == main_sha, "main unchanged")
        require(
            file_sha256(study / "prospective_ledger.json", guard) == costs["main_ledger_sha256"],
            "terminal main ledger unchanged",
        )
        require(file_sha256(Path(__file__), guard) == script_sha, "checker unchanged")
        for name, expected in CORE_HASHES.items():
            require(file_sha256(source_root / name, guard) == expected, "core unchanged")
        guard()
        actual_study_bytes = sum(path.stat().st_size for path in study.rglob("*") if path.is_file())
        require(actual_study_bytes + 1024**2 < 528 * 1024**2, "combined main/audit byte reserve")
        result = {
            "all_passed": True,
            "mismatch_count": 0,
            "bank": bank,
            "states": states,
            **metrics,
            "resources": costs,
            "bindings_verified": count,
            "study_bytes_before_audit_receipt": actual_study_bytes,
            "main_sha256": main_sha,
            "script_sha256": script_sha,
            "core_sha256": CORE_HASHES,
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
        with (destination / receipt_name).open("x") as stream:
            json.dump(result, stream, indent=2, allow_nan=False)
            stream.write("\n")
        return result
    except Exception as error:
        receipt_name = "failure.json"
        with (destination / receipt_name).open("x") as stream:
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
            )
            stream.write("\n")
        raise
    finally:
        write_terminal_ledger(destination, receipt_name, clock, costs)
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.setitimer(signal.ITIMER_PROF, 0)


def main() -> None:
    """Parse explicit main and copied-script bindings."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--main-sha", required=True)
    parser.add_argument("--script-sha", required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.root.absolute(), args.source_root.absolute(), args.main_sha, args.script_sha)))


if __name__ == "__main__":
    main()
