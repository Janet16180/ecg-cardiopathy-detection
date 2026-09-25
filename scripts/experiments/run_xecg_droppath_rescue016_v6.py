#!/usr/bin/env python3
"""Run the frozen seed-43, one-epoch xECG DropPath mechanism screen."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import signal
import time
from pathlib import Path

import numpy as np
import torch

from ecg_experiment.files import sha256_file, write_json_atomic
from ecg_experiment.gpu import gpu_lock
from ecg_experiment.reproducibility import cpu_state, flat_rng_state
from ecg_experiment.xecg_rescue_analysis import development_metrics, paired_patient_bootstrap
from ecg_experiment.xecg_rescue_training import (
    cpu_nested,
    one_epoch,
    predict,
    restore_checkpoint,
    same_nested,
    save_checkpoint,
    sequential_replay,
)
from ecg_experiment.xecg_rescue_v6 import (
    EXECUTION_EPOCHS,
    SCHEDULER_HORIZON_EPOCHS,
    SEED,
    UPDATES,
    build_model_seed43,
    decisions,
    projected_cost,
    warmup_multipliers,
)
from scripts.experiments import run_xecg_droppath_rescue016 as v5
from scripts.experiments.run_xecg_probe_finetune016 import CachedECGs

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs/experiment016_droppath_rescue_v6"
V5 = ROOT / "outputs/experiment016_droppath_rescue_v5"
V5_MAP = ROOT / "outputs/experiment_queue_016_rescue_profile_v5/sources.json"
V5_MAP_SHA = "e0fa710b871695c3f43f77489d9f3769f6255d8261e728d392c37be73f1934ac"
ARMS = ("legacy", "residual", "off")
V6_SOURCES = (
    "docs/experiment-016-droppath-rescue-v6.md",
    "ecg_experiment/xecg_rescue_v6.py",
    "scripts/experiments/run_xecg_droppath_rescue016_v6.py",
    "tests/test_xecg_droppath_rescue_v6.py",
)
V5_RECEIPTS = ("check.json", "probe.json", "probe.npz", "diagnostic.json", "profile.json", "cost_gate.json")


def evidence_and_inputs() -> dict:
    """Bind the new scope to unchanged v5 bytes and the exact clean cohort."""
    if sha256_file(V5_MAP) != V5_MAP_SHA:
        raise ValueError("Frozen v5 source map changed")
    source_map = json.loads(V5_MAP.read_text())
    changed = [name for name, digest in source_map.items() if sha256_file(ROOT / name) != digest]
    if changed:
        raise ValueError(f"Frozen v5 source/input bytes changed: {changed}")
    receipts = {name: sha256_file(V5 / name) for name in V5_RECEIPTS}
    v5_check = json.loads((V5 / "check.json").read_text())
    v5_probe = json.loads((V5 / "probe.json").read_text())
    v5_diag = json.loads((V5 / "diagnostic.json").read_text())
    v5_profile = json.loads((V5 / "profile.json").read_text())
    v5_gate = json.loads((V5 / "cost_gate.json").read_text())
    if not all(item["fingerprint"] == v5_check["fingerprint"] for item in (v5_probe, v5_diag, v5_profile)):
        raise ValueError("V5 evidence fingerprints differ")
    if v5_probe["sha256"] != receipts["probe.npz"] or v5_gate["passed"]:
        raise ValueError("V5 probe or expected two-epoch rejection differs")
    for arm in ARMS:
        row = v5_profile["arms"][arm]
        if (
            row["epoch"]["updates"] != UPDATES
            or row["epoch"]["seen"] != 15359
            or not row["model_optimizer_scheduler_rng_roundtrip"]
            or not row["resumed_next_update_same_device"]
        ):
            raise ValueError(f"V5 full-path profile correctness missing: {arm}")
    status = json.loads((ROOT / "outputs/experiment_queue_016_rescue_profile_v5/status.json").read_text())
    if status["state"] != "failed" or "stage profile exited 1" not in status["reason"]:
        raise ValueError("V5 predecessor is not the expected stopped cost-gate profile")
    inputs = v5.frozen_inputs()
    if inputs["fingerprint"] != v5_check["fingerprint"]:
        raise ValueError("Current clean inputs differ from v5 profile")
    if (
        inputs["fingerprint"]["train_records"] != 15359
        or inputs["fingerprint"]["development_records"] != 1306
    ):
        raise ValueError("Current clean cohort size differs")
    inputs["v5_fingerprint"] = inputs["fingerprint"]
    inputs["fingerprint"] = {
        "v5": inputs["v5_fingerprint"],
        "v6_sources": {name: sha256_file(ROOT / name) for name in V6_SOURCES},
        "optimization_seed": SEED,
        "execution_epochs": EXECUTION_EPOCHS,
        "scheduler_horizon_epochs": SCHEDULER_HORIZON_EPOCHS,
        "updates": UPDATES,
    }
    inputs["evidence"] = {
        "v5_source_map_sha256": V5_MAP_SHA,
        "v5_receipt_sha256": receipts,
        "v5_profile_status": status,
        "unchanged_runtime_module_sha256": {
            name: source_map[name]
            for name in (
                "ecg_experiment/xecg.py",
                "ecg_experiment/xecg_droppath_rescue.py",
                "ecg_experiment/xecg_rescue_training.py",
                "ecg_experiment/xecg_rescue_analysis.py",
                "third_party/xecg-deps/xlstm/blocks/xlstm_block.py",
                "uv.lock",
            )
        },
        "source_behavior_difference": [
            "optimization seed 42 to 43 after identical model/probe loading",
            "execution endpoint 480 to 240 updates; two-epoch scheduler prefix unchanged",
            "new v6 output/result identity and inherited-evidence cost gate",
        ],
    }
    return inputs


def probe_logits(inputs: dict) -> np.ndarray:
    """Use the exact frozen v5 C=.01 coefficients without refitting."""
    with np.load(V5 / "probe.npz") as probe:
        result = inputs["development_features"].astype(np.float64) @ probe["raw_weight"] + float(
            probe["raw_bias"]
        )
    return result


def cpu_start_checks(inputs: dict) -> dict:
    """Prove equal initial tensors and isolated seed-43 mask/order streams."""
    reference, *rest = v5.build_model(v5.RELEASE, V5 / "probe.npz", "legacy", "cpu", 15359)
    original = cpu_state(reference)
    del reference, rest
    gc.collect()
    expected_rng = torch.Generator(device="cpu").manual_seed(SEED).get_state()
    expected_order = torch.randperm(15359, generator=torch.Generator().manual_seed(SEED))
    mask_states = {}
    starts = {}
    for arm in ARMS:
        model, mask, optimizer, scheduler, permutation = build_model_seed43(
            v5.RELEASE, V5 / "probe.npz", arm, "cpu", 15359
        )
        state = cpu_state(model)
        if state.keys() != original.keys() or not all(
            torch.equal(state[key], original[key]) for key in original
        ):
            raise RuntimeError(f"V6 {arm} initial model differs from released/probe v5 start")
        if not torch.equal(mask.get_rng_state(), expected_rng) or not torch.equal(
            permutation.get_state(), expected_rng
        ):
            raise RuntimeError(f"V6 {arm} has a seed-42 sampler leak")
        order = torch.randperm(15359, generator=permutation)
        if not torch.equal(order, expected_order):
            raise RuntimeError(f"V6 {arm} sampled record order differs")
        if scheduler.get_last_lr() != [0.0 for _ in optimizer.param_groups]:
            raise RuntimeError("Initial warmup LR must be zero")
        if [scheduler.lr_lambdas[0](step) for step in range(UPDATES)] != warmup_multipliers():
            raise RuntimeError("First-epoch LR vector differs from historical two-epoch prefix")
        if not torch.equal(torch.get_rng_state(), expected_rng):
            raise RuntimeError("Global training RNG is not seed 43")
        start_hash = hashlib.sha256()
        for value in state.values():
            start_hash.update(value.numpy().tobytes())
        starts[arm] = start_hash.hexdigest()
        mask_states[arm] = hashlib.sha256(mask.get_rng_state().numpy().tobytes()).hexdigest()
        del model, mask, optimizer, scheduler, permutation, state, order
        gc.collect()
    if len(set(starts.values())) != 1 or mask_states["legacy"] != mask_states["residual"]:
        raise RuntimeError("Paired arms do not share equal starts/mask streams")
    return {
        "initial_tensor_sha256": starts,
        "mask_rng_sha256": mask_states,
        "permutation_first64_sha256": hashlib.sha256(expected_order[:64].numpy().tobytes()).hexdigest(),
        "warmup_first": warmup_multipliers()[0],
        "warmup_last_applied": warmup_multipliers()[-1],
        "scheduler_horizon_epochs": SCHEDULER_HORIZON_EPOCHS,
        "execution_epochs": EXECUTION_EPOCHS,
        "updates": UPDATES,
    }


def check() -> dict:
    """Run fresh CPU/identity checks and reference the unchanged probe/diagnostic."""
    started = time.monotonic()
    test_receipt = json.loads((OUT / "preflight_tests.json").read_text())
    if not test_receipt["passed"] or test_receipt["seconds"] < 0:
        raise ValueError("Focused v6 tests did not pass")
    inputs = evidence_and_inputs()
    starts = cpu_start_checks(inputs)
    labels = np.array([int(row["target"]) for row in inputs["development"]])
    patients = np.array([row["patient_id"] for row in inputs["development"]])
    logits = probe_logits(inputs)
    np.save(OUT / "probe_logits.npy", logits)
    metrics = development_metrics(labels, patients, logits)
    if abs(metrics["auroc"] - 0.9619404113151374) > 1e-12:
        raise RuntimeError("Frozen clean probe AUROC differs")
    receipt = {
        "fingerprint": inputs["fingerprint"],
        "inherited_evidence": inputs["evidence"],
        "initialization": starts,
        "probe": {
            "sha256": sha256_file(V5 / "probe.npz"),
            "logits_sha256": sha256_file(OUT / "probe_logits.npy"),
            "metrics": metrics,
        },
        "cpu_verification_seconds": time.monotonic() - started,
        "preflight_tests_seconds": test_receipt["seconds"],
        "preflight_tests_sha256": sha256_file(OUT / "preflight_tests.json"),
        "status": "passed_no_training",
    }
    write_json_atomic(OUT / "check.json", receipt)
    return receipt


def bridge_snapshot(checkpoint: Path, fingerprint: dict, arm: str, data: CachedECGs, device: str) -> dict:
    """Restore and execute update two, releasing model before the next replay."""
    model, mask, optimizer, scheduler, permutation = build_model_seed43(
        v5.RELEASE, V5 / "probe.npz", arm, device, len(data)
    )
    saved = restore_checkpoint(checkpoint, fingerprint, model, mask, optimizer, scheduler, permutation)
    if saved["updates"] != 1 or saved["next_index"] != 64:
        raise RuntimeError("Bridge checkpoint is not positioned after update one")
    result = one_epoch(
        model,
        mask,
        data,
        device,
        optimizer,
        scheduler,
        permutation,
        0,
        checkpoint=None,
        fingerprint=fingerprint,
        initial=saved,
        max_updates=1,
    )
    if result["updates"] != 2 or not math.isfinite(result["mean_loss"]):
        raise RuntimeError("Bridge resumed update is invalid")
    snapshot = {
        "model": cpu_state(model),
        "optimizer": cpu_nested(optimizer.state_dict()),
        "scheduler": cpu_nested(scheduler.state_dict()),
        "mask_rng": mask.get_rng_state(),
        "permutation_rng": permutation.get_state(),
        "global_rng": cpu_nested(flat_rng_state()),
        "batch_order_sha256": hashlib.sha256(saved["order"][64:128].numpy().tobytes()).hexdigest(),
    }
    del model, mask, optimizer, scheduler, permutation, saved
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()
        if torch.cuda.memory_allocated() > 256 * 1024**2:
            raise RuntimeError("Bridge replay retained active GPU model")
    return snapshot


def bridge(inputs: dict, device: str, preparation_seconds: float = 0.0) -> dict:
    """Bounded seed-43 GPU bridge; it is compatibility, never cost extrapolation."""
    started = time.monotonic()
    check_receipt = json.loads((OUT / "check.json").read_text())
    if check_receipt["fingerprint"] != inputs["fingerprint"]:
        raise ValueError("Fresh CPU check differs from bridge sources")
    if device != "cuda" or "V100" not in torch.cuda.get_device_name(0):
        raise RuntimeError("V6 bridge requires the same V100 device as v5")
    with gpu_lock(device, blocking=False):
        signal.alarm(300)
        try:
            data = CachedECGs(inputs["views"], inputs["index"], inputs["train"])
            result = {
                "fingerprint": inputs["fingerprint"],
                "device": torch.cuda.get_device_name(0),
                "arms": {},
            }
            mask_states = {}
            for arm in ARMS:
                model, mask, optimizer, scheduler, permutation = build_model_seed43(
                    v5.RELEASE, V5 / "probe.npz", arm, device, len(data)
                )
                order = torch.randperm(len(data), generator=permutation)
                source = {"fingerprint": inputs["fingerprint"], "arm": arm, "stage": "bridge"}
                first = one_epoch(
                    model,
                    mask,
                    data,
                    device,
                    optimizer,
                    scheduler,
                    permutation,
                    0,
                    checkpoint=None,
                    fingerprint=source,
                    initial={"order": order, "next_index": 0, "updates": 0},
                    max_updates=1,
                )
                if first["updates"] != 1 or first["seen"] != 64 or not math.isfinite(first["mean_loss"]):
                    raise RuntimeError("Bridge first update is invalid")
                path = OUT / f"bridge_{arm}.pt"
                saved = save_checkpoint(
                    path, source, model, mask, optimizer, scheduler, permutation, 0, 64, order, 1
                )
                if not same_nested(cpu_state(model), saved["model"]) or not same_nested(
                    cpu_nested(optimizer.state_dict()), saved["optimizer"]
                ):
                    raise RuntimeError("Bridge CPU model/optimizer checkpoint mismatch")
                if (
                    not same_nested(scheduler.state_dict(), saved["scheduler"])
                    or not torch.equal(mask.get_rng_state(), saved["mask_rng"])
                    or not torch.equal(permutation.get_state(), saved["permutation_rng"])
                ):
                    raise RuntimeError("Bridge scheduler/RNG checkpoint mismatch")
                mask_states[arm] = hashlib.sha256(mask.get_rng_state().numpy().tobytes()).hexdigest()
                del model, mask, optimizer, scheduler, permutation, saved, order
                gc.collect()
                torch.cuda.empty_cache()
                replay = sequential_replay(
                    lambda path=path, source=source, arm=arm: bridge_snapshot(
                        path, source, arm, data, device
                    ),
                    device,
                )
                result["arms"][arm] = {
                    "first_update": first,
                    "checkpoint_cpu_roundtrip": True,
                    "resumed_next_update": replay,
                    "mask_rng_sha256_after_first_update": mask_states[arm],
                }
                path.unlink()
                if time.monotonic() - started > 300:
                    raise TimeoutError("V6 bridge exceeded the frozen 300-second limit")
            if mask_states["legacy"] != mask_states["residual"]:
                raise RuntimeError("Legacy/Residual bridge mask streams diverged")
            result["bridge_seconds"] = time.monotonic() - started
            result["inherited_v5_profile_sha256"] = inputs["evidence"]["v5_receipt_sha256"]["profile.json"]
            write_json_atomic(OUT / "bridge.json", result)
        finally:
            signal.alarm(0)
    gate = projected_cost(
        check_receipt["preflight_tests_seconds"]
        + check_receipt["cpu_verification_seconds"]
        + preparation_seconds
        + result["bridge_seconds"],
        [],
    )
    gate["v6_cost_components_seconds"] = {
        "preflight_tests": check_receipt["preflight_tests_seconds"],
        "cpu_check": check_receipt["cpu_verification_seconds"],
        "pre_bridge_identity_check": preparation_seconds,
        "gpu_bridge": result["bridge_seconds"],
    }
    gate["bridge_receipt_sha256"] = sha256_file(OUT / "bridge.json")
    write_json_atomic(OUT / "cost_gate.json", gate)
    if not gate["passed"]:
        raise RuntimeError(f"V6 inherited-profile gate failed: {gate['projected_total_seconds']:.1f}s")
    return result


def verified_gate(inputs: dict) -> dict:
    """Require the exact bridge/check identity and a passing frozen cost gate."""
    check_receipt = json.loads((OUT / "check.json").read_text())
    bridge_receipt = json.loads((OUT / "bridge.json").read_text())
    gate = json.loads((OUT / "cost_gate.json").read_text())
    if (
        check_receipt["fingerprint"] != inputs["fingerprint"]
        or bridge_receipt["fingerprint"] != inputs["fingerprint"]
    ):
        raise ValueError("V6 check/bridge inputs differ")
    if gate["bridge_receipt_sha256"] != sha256_file(OUT / "bridge.json") or not gate["passed"]:
        raise ValueError("V6 bridge/gate is absent or failed")
    return gate


def train_arm(inputs: dict, arm: str, device: str) -> dict:  # noqa: C901 - resume invariants
    """Run exactly update indices 0..239, with an exact resumable checkpoint."""
    started = time.monotonic()
    directory = OUT / arm
    directory.mkdir(parents=True, exist_ok=True)
    receipt_path = directory / "complete.json"
    identity = {"source": inputs["fingerprint"], "arm": arm, "stage": "train"}
    if receipt_path.is_file():
        receipt = json.loads(receipt_path.read_text())
        if receipt["fingerprint"] != identity:
            raise ValueError("Existing arm receipt has another source identity")
        for name, digest in receipt["sha256"].items():
            if sha256_file(directory / name) != digest:
                raise ValueError(f"Existing {arm} artifact changed: {name}")
        return receipt
    data = CachedECGs(inputs["views"], inputs["index"], inputs["train"])
    dev = CachedECGs(inputs["views"], inputs["index"], inputs["development"])
    labels = np.array([int(row["target"]) for row in inputs["development"]])
    patients = np.array([row["patient_id"] for row in inputs["development"]])
    model, mask, optimizer, scheduler, permutation = build_model_seed43(
        v5.RELEASE, V5 / "probe.npz", arm, device, len(data)
    )
    baseline = cpu_state(model)
    initial_blocks = [
        [parameter.detach().float().cpu().clone() for parameter in block.parameters()]
        for block in model.backbone.core.model.blocks
    ]
    checkpoint = directory / "resume.pt"
    history_path = directory / "history.json"
    trajectory_path = directory / "trajectory.json"
    if checkpoint.is_file():
        saved = restore_checkpoint(checkpoint, identity, model, mask, optimizer, scheduler, permutation)
        history = json.loads(history_path.read_text())
        trajectory = json.loads(trajectory_path.read_text())
        if saved["epoch"] != 0 or saved["updates"] > UPDATES or len(history) != 1:
            raise RuntimeError("Invalid one-epoch resume position/history")
    else:
        saved = None
        probe = np.load(OUT / "probe_logits.npy")
        np.save(directory / "development_logits_epoch0.npy", probe)
        history = [{"epoch": 0, **development_metrics(labels, patients, probe)}]
        trajectory = [v5.trajectory_point(model, inputs, device, initial_blocks, "initial")]
        write_json_atomic(history_path, history)
        write_json_atomic(trajectory_path, trajectory)

    def first_update(updated_model: torch.nn.Module, gradient_norm: float) -> None:
        state = cpu_state(updated_model)
        if not all(torch.equal(state[key], baseline[key]) for key in baseline):
            raise RuntimeError("Zero-LR first update changed model tensors")
        point = v5.trajectory_point(
            updated_model, inputs, device, initial_blocks, "first_update", gradient_norm
        )
        point["parameters_identical_to_initial"] = True
        trajectory.append(point)
        write_json_atomic(trajectory_path, trajectory)

    if saved is None or saved["updates"] < UPDATES:
        training = one_epoch(
            model,
            mask,
            data,
            device,
            optimizer,
            scheduler,
            permutation,
            0,
            checkpoint=checkpoint,
            fingerprint=identity,
            initial=saved,
            after_update=first_update if saved is None else None,
        )
    else:
        training = {
            "epoch": 1,
            "updates": saved["updates"],
            "seen": len(data),
            "recovered_from_epoch_end_checkpoint": True,
            "mask_draws": saved["mask_draws"],
        }
    if training["updates"] != UPDATES or training["epoch"] != 1:
        raise RuntimeError("V6 training attempted an invalid endpoint")
    if scheduler.last_epoch != UPDATES:
        raise RuntimeError("V6 scheduler crossed the wrong update count")
    logits = predict(model, dev, device)
    np.save(directory / "development_logits_epoch1.npy", logits)
    history.append({"epoch": 1, "training": training, **development_metrics(labels, patients, logits)})
    trajectory.append(v5.trajectory_point(model, inputs, device, initial_blocks, "epoch_1"))
    write_json_atomic(history_path, history)
    write_json_atomic(trajectory_path, trajectory)
    if [row["epoch"] for row in history] != [0, 1] or len(trajectory) != 3:
        raise RuntimeError("V6 initial, first-update, final diagnostics are incomplete")
    if (directory / "development_logits_epoch2.npy").exists():
        raise RuntimeError("Forbidden epoch-two artifact exists")
    names = (
        "history.json",
        "trajectory.json",
        "resume.pt",
        "development_logits_epoch0.npy",
        "development_logits_epoch1.npy",
    )
    receipt = {
        "fingerprint": identity,
        "sha256": {name: sha256_file(directory / name) for name in names},
        "execution_epochs": EXECUTION_EPOCHS,
        "scheduler_horizon_epochs": SCHEDULER_HORIZON_EPOCHS,
        "final_epoch": 1,
        "optimizer_updates": UPDATES,
        "record_exposures": 15359,
        "complete_pass_seconds": time.monotonic() - started,
    }
    write_json_atomic(receipt_path, receipt)
    return receipt


def train(inputs: dict, device: str) -> None:
    """Train one arm at a time and recheck the inherited ceiling after each."""
    if device != "cuda" or "V100" not in torch.cuda.get_device_name(0):
        raise RuntimeError("V6 training requires the profiled V100")
    gate = verified_gate(inputs)
    verification_seconds = gate["new_verification_seconds"]
    completed = []
    with gpu_lock(device, blocking=False):
        for arm in ARMS:
            arm_receipt = train_arm(inputs, arm, device)
            completed.append(arm_receipt["complete_pass_seconds"])
            updated = projected_cost(verification_seconds, completed)
            updated["last_completed_arm"] = arm
            write_json_atomic(OUT / "runtime_gate.json", updated)
            print(json.dumps({"arm": arm, "actual_pass_seconds": completed[-1], "gate": updated}), flush=True)
            if not updated["passed"]:
                raise RuntimeError(f"V6 actual runtime gate failed after {arm}; stop before next arm")
            gc.collect()
            torch.cuda.empty_cache()


def report(inputs: dict) -> dict:
    """Analyze only the frozen after-update-240 development endpoint."""
    started = time.monotonic()
    gate = verified_gate(inputs)
    latest_gate = json.loads((OUT / "runtime_gate.json").read_text())
    if not latest_gate["passed"] or latest_gate["remaining_profiled_passes"] != 0:
        raise RuntimeError("All three one-epoch arms must complete within cost gate")
    labels = np.array([int(row["target"]) for row in inputs["development"]])
    patients = np.array([row["patient_id"] for row in inputs["development"]])
    arm_metrics = {}
    histories = {}
    predictions = {}
    arm_receipts = {}
    for arm in ARMS:
        directory = OUT / arm
        receipt = json.loads((directory / "complete.json").read_text())
        if receipt["fingerprint"] != {"source": inputs["fingerprint"], "arm": arm, "stage": "train"}:
            raise ValueError("Arm receipt belongs to another source")
        if (
            receipt["optimizer_updates"] != UPDATES
            or receipt["record_exposures"] != 15359
            or receipt["final_epoch"] != 1
        ):
            raise ValueError("Arm has an invalid one-epoch endpoint")
        for name, digest in receipt["sha256"].items():
            if sha256_file(directory / name) != digest:
                raise ValueError(f"Arm artifact changed: {arm}/{name}")
        if (directory / "development_logits_epoch2.npy").exists():
            raise ValueError("An epoch-two arm artifact is forbidden")
        histories[arm] = json.loads((directory / "history.json").read_text())
        if [row["epoch"] for row in histories[arm]] != [0, 1]:
            raise ValueError("Arm history has another epoch count")
        arm_metrics[arm] = histories[arm][1]
        predictions[arm] = np.load(directory / "development_logits_epoch1.npy")
        arm_receipts[arm] = sha256_file(directory / "complete.json")
    probe = json.loads((OUT / "check.json").read_text())["probe"]["metrics"]
    bootstrap = paired_patient_bootstrap(labels, patients, predictions)
    inherited = json.loads((V5 / "diagnostic.json").read_text())
    shift = {arm: inherited["arms"][arm]["mean_logit_shift"] for arm in ARMS}
    decision = decisions(arm_metrics, probe, bootstrap, shift)
    primary_interval = bootstrap["contrasts"]["residual_minus_legacy"]["interval_95"]
    result = {
        "fingerprint": inputs["fingerprint"],
        "scope": "one optimization seed, one warmup epoch, development annotation proxy; no calibration/test",
        "records": {
            "train": 15359,
            "development": len(labels),
            "development_patients": len(np.unique(patients)),
        },
        "optimization_seed": SEED,
        "execution_epochs": EXECUTION_EPOCHS,
        "scheduler_horizon_epochs": SCHEDULER_HORIZON_EPOCHS,
        "updates": UPDATES,
        "probe_metrics": probe,
        "arm_metrics": arm_metrics,
        "histories": histories,
        "paired_patient_bootstrap": bootstrap,
        "inherited_v5_initial_logit_shifts": shift,
        "decision": decision,
        "arm_receipt_sha256": arm_receipts,
        "verification_cost_gate": gate,
        "actual_runtime_gate": latest_gate,
        "report_seconds": time.monotonic() - started,
    }
    write_json_atomic(OUT / "report.json", result)
    lines = [
        "# Experiment 016 v6: one-epoch DropPath mechanism screen",
        "",
        "Seed-43, full-label clean cohort, 240-update warmup prefix; development annotation proxy only.",
        "The v5 seed-42 profile scores were inspected before this protocol. "
        "This seed-43 result uses the same development cohort, so it is not independent confirmation.",
        "",
        "| Arm | AUROC | AP | BCE | Patient-fold sensitivity | Patient-fold specificity |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for name, row in (("Frozen probe", probe), *((arm, arm_metrics[arm]) for arm in ARMS)):
        lines.append(
            f"| {name} | {row['auroc']:.5f} | {row['average_precision']:.5f} | {row['bce']:.5f} | "
            f"{row['mean_fold_sensitivity']:.5f} | {row['mean_fold_specificity']:.5f} |"
        )
    lines += [
        "",
        f"Primary Residual − Legacy AUROC: {decision['residual_minus_legacy']:+.5f}; "
        f"paired 95% patient bootstrap interval {primary_interval}.",
        f"Secondary Off − Legacy: {decision['off_minus_legacy']:+.5f}; "
        f"Residual − Off: {decision['residual_minus_off']:+.5f}.",
        f"Valid paired patient draws: {bootstrap['valid']} / {bootstrap['requested']}.",
        f"Mechanistic point screen: {decision['mechanistic_support_point_screen']}. "
        f"{decision['mechanistic_interpretation']}.",
        f"Practical rescue arms: {decision['practical_rescue_arms']}; "
        f"preferred if any: {decision['preferred_if_any']}.",
        "Intervals condition on one trained model per arm; training-seed uncertainty remains unmeasured.",
        "This one-epoch screen cannot establish longer-budget fine-tuning or clinical utility. "
        "No calibration/test was evaluated.",
        "",
    ]
    (OUT / "report.md").write_text("\n".join(lines))
    return result


def main() -> None:
    """Dispatch one immutable v6 stage with its expected device."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("check", "bridge", "train", "report"), required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    args = parser.parse_args()
    torch.set_num_threads(1)
    OUT.mkdir(parents=True, exist_ok=True)
    if args.stage == "check":
        if args.device != "cpu":
            raise ValueError("Fresh check must run on CPU")
        receipt = check()
        print(
            json.dumps({"stage": "016_v6_check", "seconds": receipt["cpu_verification_seconds"]}), flush=True
        )
        return
    before_identity = time.monotonic()
    inputs = evidence_and_inputs()
    identity_seconds = time.monotonic() - before_identity
    if args.stage == "bridge":
        receipt = bridge(inputs, args.device, identity_seconds)
        print(json.dumps({"stage": "016_v6_bridge", "seconds": receipt["bridge_seconds"]}), flush=True)
    elif args.stage == "train":
        train(inputs, args.device)
    else:
        report(inputs)


if __name__ == "__main__":
    main()
