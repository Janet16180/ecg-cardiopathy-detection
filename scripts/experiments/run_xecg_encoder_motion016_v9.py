#!/usr/bin/env python3
"""Verified moving-versus-frozen xECG encoder intervention on clean ECGs."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

from ecg_experiment.files import sha256_file, write_json_atomic, write_npz_atomic
from ecg_experiment.gpu import gpu_lock
from ecg_experiment.reproducibility import cpu_state, flat_rng_state
from ecg_experiment.xecg_encoder_motion_v9 import (
    ARMS,
    DEVELOPMENT_RECORDS,
    DIAGNOSTIC_UPDATES,
    EFFECTIVE_BATCH,
    TRAIN_RECORDS,
    UPDATES,
    build_model,
    encoder_equal,
    encoder_state,
    fixed_diagnostic,
    infer_features,
    model_identity,
    train_update,
)
from ecg_experiment.xecg_readout_v7 import fixed_probe
from ecg_experiment.xecg_rescue_analysis import development_metrics
from ecg_experiment.xecg_rescue_training import (
    compare_replay_snapshots,
    cpu_nested,
    predict,
    restore_checkpoint,
    same_nested,
    save_checkpoint,
)
from scripts.experiments import run_xecg_droppath_rescue016 as v5
from scripts.experiments import run_xecg_droppath_rescue016_v6 as v6
from scripts.experiments.run_xecg_probe_finetune016 import CachedECGs

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs/experiment016_encoder_motion_v9"
V8_MAP = ROOT / "outputs/experiment_queue_016_head_mechanism_v8_full/sources.json"
V8_MAP_SHA = "2619aea36784cb4cd9eb1f2203348a9ce6e36e260712fcb2772b521e2d96417f"
V8_COMPLETION = (
    ROOT / "outputs/experiment_queue_016_head_mechanism_v8_full/job/priority_queue_completion.json"
)
V8_MANIFEST_SHA = "ccc4c3322047695f5582787c7aa51f27e57b9eace82ede01c7cd22a8b3cb07f5"
V7 = ROOT / "outputs/experiment016_frozen_readout_audit_v7"
V6 = ROOT / "outputs/experiment016_droppath_rescue_v6"
V9_SOURCES = (
    "docs/experiment-016-encoder-motion-v9.md",
    "ecg_experiment/xecg_encoder_motion_v9.py",
    "scripts/experiments/run_xecg_encoder_motion016_v9.py",
    "tests/test_xecg_encoder_motion_v9.py",
)


def inputs() -> dict:
    """Bind one clean cohort and every frozen v5–v8 source/result identity."""
    if sha256_file(V8_MAP) != V8_MAP_SHA:
        raise ValueError("Frozen v8 source map changed")
    for name, expected in json.loads(V8_MAP.read_text()).items():
        if sha256_file(ROOT / name) != expected:
            raise ValueError(f"Frozen predecessor source/input changed: {name}")
    completion = json.loads(V8_COMPLETION.read_text())
    if completion["queue_manifest_sha256"] != V8_MANIFEST_SHA:
        raise ValueError("V8 completion belongs to another manifest")
    for name, expected in completion["artifacts"].items():
        if sha256_file(ROOT / name) != expected:
            raise ValueError(f"Frozen v8 completion artifact changed: {name}")
    data = v6.evidence_and_inputs()
    if len(data["train"]) != TRAIN_RECORDS or len(data["development"]) != DEVELOPMENT_RECORDS:
        raise ValueError("Clean cohort differs from v6/v7")
    data["fingerprint"] = {
        "v6_cohort": data["fingerprint"],
        "v8_source_map_sha256": V8_MAP_SHA,
        "v8_completion_sha256": sha256_file(V8_COMPLETION),
        "v8_report_sha256": sha256_file(ROOT / "outputs/experiment016_head_mechanism_v8/report.json"),
        "v6_probe_sha256": sha256_file(v6.V5 / "probe.npz"),
        "v7_reference_logits_sha256": sha256_file(V7 / "released_refit_logits.npy"),
        "v9_sources": {name: sha256_file(ROOT / name) for name in V9_SOURCES},
        "environment": {
            "python": sys.version.split()[0],
            "torch": str(torch.__version__),
            "numpy": np.__version__,
            "uv_lock_sha256": sha256_file(ROOT / "uv.lock"),
        },
    }
    return data


def datasets(data: dict) -> tuple[CachedECGs, CachedECGs, CachedECGs, np.ndarray, np.ndarray]:
    """Build the exact clean training, development and fixed 128-training subsets."""
    train = CachedECGs(data["views"], data["index"], data["train"])
    development = CachedECGs(data["views"], data["index"], data["development"])
    positive = [row for row in data["train"] if row["target"] == "1"][:64]
    negative = [row for row in data["train"] if row["target"] == "0"][:64]
    diagnostic_rows = positive + negative
    if len(positive) != 64 or len(negative) != 64:
        raise ValueError("Fixed diagnostic lacks 64 records per class")
    diagnostic = CachedECGs(data["views"], data["index"], diagnostic_rows)
    original = {row["ecg_id"]: data["train_features"][index] for index, row in enumerate(data["train"])}
    release = np.stack([original[row["ecg_id"]] for row in diagnostic_rows])
    labels = np.asarray([int(row["target"]) for row in diagnostic_rows])
    return train, development, diagnostic, release, labels


def check() -> dict:
    """Check common released start, optimizer/scheduler groups and first logits on CPU."""
    started = time.monotonic()
    data = inputs()
    train, _, diagnostic, release, labels = datasets(data)
    if len(train) != TRAIN_RECORDS or labels.shape != (128,):
        raise ValueError("CPU diagnostic data shape differs")
    with np.load(v6.V5 / "probe.npz") as probe:
        weight, bias = probe["raw_weight"], float(probe["raw_bias"])
    starts = {}
    for arm in ARMS:
        model, optimizer, scheduler, permutation = build_model(
            v5.RELEASE, v6.V5 / "probe.npz", arm, 16090, "cpu"
        )
        starts[arm] = {
            "model_sha256": model_identity(model),
            "group_names": [group["name"] for group in optimizer.param_groups],
            "base_lrs": scheduler.base_lrs,
            "first64_sha256": hashlib.sha256(
                torch.randperm(TRAIN_RECORDS, generator=permutation)[:64].numpy().tobytes()
            ).hexdigest(),
            "encoder_requires_grad": all(
                parameter.requires_grad for parameter in model.backbone.parameters()
            ),
            "dropout_rates": model.backbone.core.dropout_rates,
        }
        features, logits = infer_features(model, diagnostic, "cpu")
        if not np.allclose(features, release, atol=2e-4, rtol=2e-5):
            raise RuntimeError("Fresh released features differ from frozen feature cache")
        expected = release.astype(np.float64) @ weight + bias
        if not np.allclose(logits, expected, atol=2e-4, rtol=2e-5):
            raise RuntimeError("Fresh initial probe-head logits differ")
        del model, optimizer, scheduler, permutation
        gc.collect()
    if (
        starts["M"]["model_sha256"] != starts["F"]["model_sha256"]
        or starts["M"]["first64_sha256"] != starts["F"]["first64_sha256"]
    ):
        raise RuntimeError("Matched arms start from different weights or order")
    for index, name in enumerate(starts["F"]["group_names"]):
        if name == "binary_head":
            if starts["F"]["base_lrs"][index] != 1e-3:
                raise RuntimeError("Frozen arm head LR changed")
        elif starts["F"]["base_lrs"][index] != 0:
            raise RuntimeError("Frozen arm encoder scheduler base LR changed")
    receipt = {
        "fingerprint": data["fingerprint"],
        "arms": starts,
        "train_records": TRAIN_RECORDS,
        "development_records": DEVELOPMENT_RECORDS,
        "development_patients": len({row["patient_id"] for row in data["development"]}),
        "fixed_training_diagnostic_ids": [
            row["ecg_id"]
            for row in (
                [row for row in data["train"] if row["target"] == "1"][:64]
                + [row for row in data["train"] if row["target"] == "0"][:64]
            )
        ],
        "seconds": time.monotonic() - started,
        "status": "passed_no_training",
    }
    write_json_atomic(OUT / "check.json", receipt)
    return receipt


def _verify_checkpoint(
    path: Path,
    source: dict,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LambdaLR,
    permutation: torch.Generator,
    order: torch.Tensor,
    next_index: int,
    updates: int,
) -> dict:
    """Require exact CPU serialization of model, optimizer, schedule and RNG."""
    mask = model.backbone.core.drop_path
    loaded = save_checkpoint(
        path, source, model, mask, optimizer, scheduler, permutation, 0, next_index, order, updates
    )
    if not same_nested(cpu_nested(optimizer.state_dict()), loaded["optimizer"]):
        raise RuntimeError("Checkpoint AdamW state changed on CPU serialization")
    if not same_nested(cpu_nested(scheduler.state_dict()), loaded["scheduler"]):
        raise RuntimeError("Checkpoint scheduler changed on CPU serialization")
    if not same_nested(permutation.get_state(), loaded["permutation_rng"]):
        raise RuntimeError("Checkpoint permutation RNG changed on CPU serialization")
    if not same_nested(cpu_nested(flat_rng_state()), {key: loaded[key] for key in flat_rng_state()}):
        raise RuntimeError("Checkpoint global RNG changed on CPU serialization")
    if not same_nested(order, loaded["order"]):
        raise RuntimeError("Checkpoint current permutation changed on CPU serialization")
    return loaded


def _resume_next_snapshot(
    path: Path, source: dict, arm: str, seed: int, data: CachedECGs, ecg_ids: np.ndarray, device: str
) -> dict:
    """Restore one full model then execute update 241 for sequential replay."""
    model, optimizer, scheduler, permutation = build_model(v5.RELEASE, v6.V5 / "probe.npz", arm, seed, device)
    mask = model.backbone.core.drop_path
    saved = restore_checkpoint(path, source, model, mask, optimizer, scheduler, permutation)
    if saved["updates"] != UPDATES or saved["next_index"] != TRAIN_RECORDS:
        raise RuntimeError("Replay checkpoint is not the completed first epoch")
    order = torch.randperm(TRAIN_RECORDS, generator=permutation)
    record = train_update(
        model, optimizer, scheduler, data, order[:EFFECTIVE_BATCH], ecg_ids, device, UPDATES
    )
    snapshot = {
        "model": cpu_state(model),
        "optimizer": cpu_nested(optimizer.state_dict()),
        "scheduler": cpu_nested(scheduler.state_dict()),
        "mask_rng": mask.get_rng_state(),
        "permutation_rng": permutation.get_state(),
        "global_rng": cpu_nested(flat_rng_state()),
        "batch_ids_sha256": record["batch_ids_sha256"],
        "updates": UPDATES + 1,
    }
    del model, optimizer, scheduler, permutation, mask, saved
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()
        if torch.cuda.memory_allocated() > 256 * 1024**2:
            raise RuntimeError("Replay retained an active CUDA model")
    return snapshot


def _replay(
    path: Path, source: dict, arm: str, seed: int, data: CachedECGs, ecg_ids: np.ndarray, device: str
) -> dict:
    """Run two resumed updates sequentially under v5's tight floating tolerance."""
    first = _resume_next_snapshot(path, source, arm, seed, data, ecg_ids, device)
    second = _resume_next_snapshot(path, source, arm, seed, data, ecg_ids, device)
    return compare_replay_snapshots(first, second)


def _fit_final_probe(directory: Path, features: np.ndarray, data: dict, arm: str) -> dict:
    """Fit fixed train-only C=.01 readout and verify saved native joint logits."""
    train_labels = np.asarray([int(row["target"]) for row in data["train"]])
    probe = fixed_probe(features[:TRAIN_RECORDS], train_labels, features[TRAIN_RECORDS:])
    np.save(directory / "refit_logits.npy", probe["development_logits"])
    write_npz_atomic(
        directory / "refit_coefficients.npz",
        weight=probe["weight"],
        bias=np.array(probe["bias"]),
        mean=probe["mean"],
        scale=probe["scale"],
        coefficient=probe["coefficient"],
        intercept=np.array(probe["intercept"]),
    )
    receipt = {
        "C": 0.01,
        "iterations": probe["iterations"],
        "train_only_scaler": True,
        "converged": probe["iterations"] < 3000,
    }
    if arm == "F":
        reference_features = np.concatenate((data["train_features"], data["development_features"]))
        difference = float(np.max(np.abs(features - reference_features)))
        if not np.allclose(features, reference_features, atol=2e-4, rtol=2e-5):
            raise RuntimeError(f"Frozen encoder features differ from released cache: {difference}")
        reference = np.load(V6 / "probe_logits.npy")
        logit_difference = float(np.max(np.abs(probe["development_logits"] - reference)))
        if not np.allclose(probe["development_logits"], reference, atol=2e-4, rtol=2e-5):
            raise RuntimeError(f"Frozen refit differs from clean reference: {logit_difference}")
        receipt["released_feature_max_abs_difference"] = difference
        receipt["released_probe_logit_max_abs_difference"] = logit_difference
    return receipt


def run_arm(data: dict, arm: str, seed: int, device: str, mode: str) -> dict:  # noqa: C901
    """Run one full matched pipeline, retaining update logs/checkpoints/features/probe."""
    if device != "cuda" or "V100" not in torch.cuda.get_device_name(0):
        raise RuntimeError("V9 full pipeline requires the V100")
    directory = OUT / mode / arm
    directory.mkdir(parents=True, exist_ok=True)
    complete_path = directory / "complete.json"
    source = {"fingerprint": data["fingerprint"], "arm": arm, "seed": seed, "mode": mode}
    if complete_path.exists():
        completed = json.loads(complete_path.read_text())
        if completed["source"] != source:
            raise ValueError("Completed arm has another source identity")
        for name, digest in completed["sha256"].items():
            if sha256_file(directory / name) != digest:
                raise ValueError(f"Completed arm artifact changed: {name}")
        return completed
    started = time.monotonic()
    train, development, diagnostic, released_diag, diagnostic_labels = datasets(data)
    ecg_ids = np.asarray([int(row["ecg_id"]) for row in data["train"]], dtype=np.int64)
    model, optimizer, scheduler, permutation = build_model(v5.RELEASE, v6.V5 / "probe.npz", arm, seed, device)
    baseline_encoder = encoder_state(model)
    baseline_head = {name: value.detach().clone() for name, value in model.head.state_dict().items()}
    initial_model_sha = model_identity(model)
    with np.load(v6.V5 / "probe.npz") as saved_probe:
        old_weight = saved_probe["raw_weight"].copy()
        old_bias = float(saved_probe["raw_bias"])
    checkpoint = directory / "resume.pt"
    log_path = directory / "updates.jsonl"
    diagnostic_path = directory / "diagnostics.json"
    if checkpoint.exists():
        saved = restore_checkpoint(
            checkpoint, source, model, model.backbone.core.drop_path, optimizer, scheduler, permutation
        )
        if saved["epoch"] != 0 or saved["updates"] > UPDATES:
            raise RuntimeError("Invalid one-epoch resume position")
        order = saved["order"]
        next_index = int(saved["next_index"])
        updates = int(saved["updates"])
        records = [json.loads(line) for line in log_path.read_text().splitlines()]
        records = [record for record in records if record["updates_completed"] <= updates]
        log_path.write_text("".join(json.dumps(row) + "\n" for row in records))
        diagnostics = json.loads(diagnostic_path.read_text())
        diagnostics = [row for row in diagnostics if row["update"] <= updates]
    else:
        order = torch.randperm(TRAIN_RECORDS, generator=permutation)
        next_index = 0
        updates = 0
        records = []
        diagnostics = [
            fixed_diagnostic(
                model,
                diagnostic,
                device,
                diagnostic_labels,
                released_diag,
                old_weight,
                old_bias,
                baseline_encoder,
                baseline_head,
                0,
            )
        ]
        write_json_atomic(diagnostic_path, diagnostics)
    if len(order) != TRAIN_RECORDS or not torch.equal(torch.sort(order).values, torch.arange(TRAIN_RECORDS)):
        raise RuntimeError("Training permutation does not cover clean rows")
    if updates != len(records):
        raise RuntimeError("Checkpoint and per-update log count differ")
    next_position = next_index
    with log_path.open("a", buffering=1) as log:
        try:
            for start in range(next_index, TRAIN_RECORDS, EFFECTIVE_BATCH):
                group = order[start : start + EFFECTIVE_BATCH]
                record = train_update(model, optimizer, scheduler, train, group, ecg_ids, device, updates)
                updates += 1
                next_position = min(start + EFFECTIVE_BATCH, TRAIN_RECORDS)
                records.append(record)
                log.write(json.dumps(record) + "\n")
                if arm == "F" and not encoder_equal(model, baseline_encoder):
                    raise RuntimeError("Frozen encoder parameter/buffer changed after update")
                if updates == 1 and record["head_update_norm"] != 0:
                    raise RuntimeError("Zero-LR first update changed head parameters")
                if updates == 2:
                    if record["head_update_norm"] <= 0:
                        raise RuntimeError("First nonzero-LR step did not update head")
                    if arm == "M" and encoder_equal(model, baseline_encoder):
                        raise RuntimeError("Moving encoder did not change at first nonzero step")
                if updates in DIAGNOSTIC_UPDATES:
                    diagnostics.append(
                        fixed_diagnostic(
                            model,
                            diagnostic,
                            device,
                            diagnostic_labels,
                            released_diag,
                            old_weight,
                            old_bias,
                            baseline_encoder,
                            baseline_head,
                            updates,
                        )
                    )
                    write_json_atomic(diagnostic_path, diagnostics)
                if updates % 40 == 0 or next_position == TRAIN_RECORDS:
                    log.flush()
                    _verify_checkpoint(
                        checkpoint,
                        source,
                        model,
                        optimizer,
                        scheduler,
                        permutation,
                        order,
                        next_position,
                        updates,
                    )
                    print(json.dumps({"mode": mode, "arm": arm, "updates": updates}), flush=True)
        except BaseException:
            if updates > 0:
                log.flush()
                _verify_checkpoint(
                    checkpoint,
                    source,
                    model,
                    optimizer,
                    scheduler,
                    permutation,
                    order,
                    next_position,
                    updates,
                )
                write_json_atomic(diagnostic_path, diagnostics)
            raise
    if updates != UPDATES or next_position != TRAIN_RECORDS or len(records) != UPDATES:
        raise RuntimeError("Full 15,359-record 240-update epoch incomplete")
    if scheduler.last_epoch != UPDATES or len(diagnostics) != len(DIAGNOSTIC_UPDATES):
        raise RuntimeError("Warmup scheduler or fixed trajectory endpoint differs")
    if arm == "F" and not encoder_equal(model, baseline_encoder):
        raise RuntimeError("Frozen encoder changed at epoch end")
    dev_logits = predict(model, development, device)
    np.save(directory / "joint_logits.npy", dev_logits)
    features, native_logits = infer_features(model, train, device)
    dev_features, native_dev_logits = infer_features(model, development, device)
    features = np.concatenate((features, dev_features))
    np.save(directory / "features.npy", features)
    if not np.allclose(native_dev_logits, dev_logits, atol=2e-4, rtol=2e-5):
        raise RuntimeError("Final native development pass changed logits")
    with torch.no_grad():
        weight = model.head.weight.detach().float().cpu().numpy().reshape(-1)
        bias = float(model.head.bias.detach().float().cpu().item())
    affine = features[TRAIN_RECORDS:].astype(np.float64) @ weight.astype(np.float64) + bias
    affine_difference = float(np.max(np.abs(affine - dev_logits)))
    if not np.allclose(affine, dev_logits, atol=2e-4, rtol=2e-5):
        raise RuntimeError("Native joint-head logits differ from extracted features")
    probe_receipt = _fit_final_probe(directory, features, data, arm)
    del model, optimizer, scheduler, permutation, baseline_encoder, baseline_head
    gc.collect()
    torch.cuda.empty_cache()
    replay = _replay(checkpoint, source, arm, seed, train, ecg_ids, device)
    if not replay["exact_fields"] or replay["atol"] != 1e-8 or replay["rtol"] != 1e-6:
        raise RuntimeError("Resumed-update replay criterion changed")
    receipt = {
        "source": source,
        "arm": arm,
        "mode": mode,
        "seed": seed,
        "initial_model_sha256": initial_model_sha,
        "initial_order_sha256": hashlib.sha256(order.numpy().tobytes()).hexdigest(),
        "updates": UPDATES,
        "record_exposures": TRAIN_RECORDS,
        "last_batch_records": records[-1]["records"],
        "final_encoder_bitwise_unchanged": diagnostics[-1]["encoder_bitwise_unchanged"],
        "first_backward": records[0],
        "first_nonzero_update": records[1],
        "per_update_log_rows": len(records),
        "fixed_diagnostic_updates": [row["update"] for row in diagnostics],
        "native_affine_joint_logit_max_abs_difference": affine_difference,
        "probe": probe_receipt,
        "replay": replay,
        "peak_gpu_allocated_bytes": torch.cuda.max_memory_allocated(),
        "peak_gpu_reserved_bytes": torch.cuda.max_memory_reserved(),
        "complete_pipeline_seconds": time.monotonic() - started,
        "sha256": {
            name: sha256_file(directory / name)
            for name in (
                "resume.pt",
                "updates.jsonl",
                "diagnostics.json",
                "joint_logits.npy",
                "features.npy",
                "refit_logits.npy",
                "refit_coefficients.npz",
            )
        },
    }
    write_json_atomic(complete_path, receipt)
    return receipt


def _compare_common_update(moving: dict, frozen: dict, description: str) -> None:
    """Verify common batch, gradient and head-step behavior before encoder motion."""
    for name in ("batch_ids_sha256", "records", "unused_gradient_parameters"):
        if moving[name] != frozen[name]:
            raise RuntimeError(f"{description} identity differs: {name}")
    for name in (
        "mean_training_bce",
        "preclip_global_norm",
        "preclip_encoder_norm",
        "preclip_head_norm",
        "clip_factor",
        "head_update_norm",
    ):
        if not math.isclose(
            moving[name], frozen[name], abs_tol=1e-5, rel_tol=1e-6
        ):
            raise RuntimeError(f"{description} numerical value differs: {name}")


def compare_profile_arms(receipts: dict[str, dict]) -> dict:
    """Require exact common first update and the single intended encoder intervention."""
    moving, frozen = receipts["M"], receipts["F"]
    if (
        moving["initial_model_sha256"] != frozen["initial_model_sha256"]
        or moving["initial_order_sha256"] != frozen["initial_order_sha256"]
    ):
        raise RuntimeError("Profile arms differ in initial tensors or batch order")
    _compare_common_update(moving["first_backward"], frozen["first_backward"], "First backward")
    if (
        moving["first_nonzero_update"]["head_update_norm"] <= 0
        or frozen["first_nonzero_update"]["head_update_norm"] <= 0
    ):
        raise RuntimeError("First nonzero step did not update both heads")
    _compare_common_update(moving["first_nonzero_update"], frozen["first_nonzero_update"],
                           "First nonzero-update")
    if moving["final_encoder_bitwise_unchanged"] or not frozen["final_encoder_bitwise_unchanged"]:
        raise RuntimeError("Profile encoder motion intervention failed")
    return {
        "common_initial_tensors": True,
        "common_batch_order": True,
        "first_backward_agreement": True,
        "moving_encoder_changed": True,
        "frozen_encoder_bitwise_unchanged": True,
    }


def profile(data: dict, device: str, preparation_seconds: float) -> dict:
    """Run two complete seed16090 pipelines then apply the frozen two-hour gate."""
    started = time.monotonic()
    checked = json.loads((OUT / "check.json").read_text())
    if checked["fingerprint"] != data["fingerprint"]:
        raise ValueError("V9 profile differs from CPU check")
    if device != "cuda" or "V100" not in torch.cuda.get_device_name(0):
        raise RuntimeError("V9 profile requires V100")
    receipts = {}
    with gpu_lock(device, blocking=False):
        for arm in ARMS:
            receipts[arm] = run_arm(data, arm, 16090, device, "profile")
            gc.collect()
            torch.cuda.empty_cache()
    correctness = compare_profile_arms(receipts)
    p = max(row["complete_pipeline_seconds"] for row in receipts.values())
    measured = checked["seconds"] + preparation_seconds + time.monotonic() - started
    projected = measured + 1.25 * (2 * p + 300)
    gate = {
        "fingerprint": data["fingerprint"],
        "P_seconds": p,
        "measured_preparation_plus_both_profile_pipelines_seconds": measured,
        "projected_total_seconds": projected,
        "ceiling_seconds": 7200,
        "passed": projected <= 7200,
        "profile_correctness": correctness,
    }
    write_json_atomic(
        OUT / "profile.json",
        {
            "fingerprint": data["fingerprint"],
            "profile_seed": 16090,
            "arms": {
                arm: {
                    "complete_pipeline_seconds": row["complete_pipeline_seconds"],
                    "completion_sha256": sha256_file(OUT / "profile" / arm / "complete.json"),
                    "peak_gpu_allocated_bytes": row["peak_gpu_allocated_bytes"],
                    "peak_gpu_reserved_bytes": row["peak_gpu_reserved_bytes"],
                }
                for arm, row in receipts.items()
            },
            "correctness": correctness,
            "nonselected_performance": True,
        },
    )
    write_json_atomic(OUT / "cost_gate.json", gate)
    if not gate["passed"]:
        raise RuntimeError(f"V9 two-hour profile gate failed: {projected:.1f}s")
    return gate


def train(data: dict, device: str) -> dict:
    """Run fresh seed46 M then F, updating the remaining two-hour gate per arm."""
    gate = json.loads((OUT / "cost_gate.json").read_text())
    if gate["fingerprint"] != data["fingerprint"] or not gate["passed"]:
        raise ValueError("Production requires the verified passing profile gate")
    if device != "cuda" or "V100" not in torch.cuda.get_device_name(0):
        raise RuntimeError("V9 production requires V100")
    receipts = {}
    with gpu_lock(device, blocking=False):
        for arm in ARMS:
            receipts[arm] = run_arm(data, arm, 46, device, "production")
            completed = sum(row["complete_pipeline_seconds"] for row in receipts.values())
            remaining = len(ARMS) - len(receipts)
            projected = (
                gate["measured_preparation_plus_both_profile_pipelines_seconds"]
                + completed
                + 1.25 * (remaining * gate["P_seconds"] + 300)
            )
            runtime = {
                "fingerprint": data["fingerprint"],
                "completed_arms": list(receipts),
                "measured_production_pipeline_seconds": completed,
                "projected_total_seconds": projected,
                "ceiling_seconds": 7200,
                "passed": projected <= 7200,
            }
            write_json_atomic(OUT / "runtime_gate.json", runtime)
            if not runtime["passed"]:
                raise RuntimeError(f"V9 runtime gate failed after {arm}")
            gc.collect()
            torch.cuda.empty_cache()
    if (
        receipts["M"]["initial_model_sha256"] != receipts["F"]["initial_model_sha256"]
        or receipts["M"]["initial_order_sha256"] != receipts["F"]["initial_order_sha256"]
    ):
        raise RuntimeError("Production arms differ in initial tensors or order")
    result = {
        "fingerprint": data["fingerprint"],
        "seed": 46,
        "arms": {arm: sha256_file(OUT / "production" / arm / "complete.json") for arm in ARMS},
    }
    write_json_atomic(OUT / "production.json", result)
    return result


def paired_bootstrap(labels: np.ndarray, patients: np.ndarray, logits: dict[str, np.ndarray]) -> dict:
    """Use 2,000 seed16019 common whole-patient draws for all fixed contrasts."""
    unique, inverse = np.unique(patients, return_inverse=True)
    groups = [np.flatnonzero(inverse == index) for index in range(len(unique))]
    contrasts = {
        "F_joint_minus_M_joint": lambda auc: auc["F_joint"] - auc["M_joint"],
        "M_refit_minus_F_refit": lambda auc: auc["M_refit"] - auc["F_refit"],
        "gap_M": lambda auc: auc["M_refit"] - auc["M_joint"],
        "gap_F": lambda auc: auc["F_refit"] - auc["F_joint"],
        "gap_difference": lambda auc: (auc["M_refit"] - auc["M_joint"]) - (auc["F_refit"] - auc["F_joint"]),
    }
    rng = np.random.default_rng(16019)
    draws = {name: [] for name in contrasts}
    invalid = 0
    for _ in range(2000):
        sample = np.concatenate([groups[index] for index in rng.integers(len(groups), size=len(groups))])
        if len(np.unique(labels[sample])) != 2:
            invalid += 1
            continue
        auc = {name: roc_auc_score(labels[sample], values[sample]) for name, values in logits.items()}
        for name, evaluate in contrasts.items():
            draws[name].append(float(evaluate(auc)))
    auc = {name: roc_auc_score(labels, values) for name, values in logits.items()}
    return {
        "seed": 16019,
        "requested": 2000,
        "valid": 2000 - invalid,
        "single_class_invalid": invalid,
        "contrasts": {
            name: {
                "observed": float(evaluate(auc)),
                "interval_95": np.quantile(draws[name], [0.025, 0.975]).tolist(),
            }
            for name, evaluate in contrasts.items()
        },
    }


def report(data: dict) -> dict:
    """Analyze only the paired completed seed46 endpoints under frozen gates."""
    started = time.monotonic()
    production = json.loads((OUT / "production.json").read_text())
    runtime = json.loads((OUT / "runtime_gate.json").read_text())
    if (
        production["fingerprint"] != data["fingerprint"]
        or set(production["arms"]) != set(ARMS)
        or not runtime["passed"]
    ):
        raise ValueError("Incomplete or incompatible production pair")
    labels = np.asarray([int(row["target"]) for row in data["development"]])
    patients = np.asarray([row["patient_id"] for row in data["development"]])
    logits = {"released_probe": np.load(V6 / "probe_logits.npy")}
    arm_hashes = {}
    for arm in ARMS:
        directory = OUT / "production" / arm
        receipt = json.loads((directory / "complete.json").read_text())
        if receipt["source"] != {
            "fingerprint": data["fingerprint"],
            "arm": arm,
            "seed": 46,
            "mode": "production",
        }:
            raise ValueError("Production arm identity differs")
        if (
            receipt["updates"] != UPDATES
            or receipt["record_exposures"] != TRAIN_RECORDS
            or receipt["per_update_log_rows"] != UPDATES
        ):
            raise ValueError("Production arm endpoint/logging incomplete")
        for name, digest in receipt["sha256"].items():
            if sha256_file(directory / name) != digest:
                raise ValueError(f"Production arm artifact changed: {arm}/{name}")
        logits[f"{arm}_joint"] = np.load(directory / "joint_logits.npy")
        logits[f"{arm}_refit"] = np.load(directory / "refit_logits.npy")
        arm_hashes[arm] = sha256_file(directory / "complete.json")
    if any(len(values) != DEVELOPMENT_RECORDS or not np.isfinite(values).all() for values in logits.values()):
        raise ValueError("Development logits are incomplete or nonfinite")
    metrics = {name: development_metrics(labels, patients, values) for name, values in logits.items()}
    bootstrap = paired_bootstrap(labels, patients, logits)
    primary = bootstrap["contrasts"]["F_joint_minus_M_joint"]
    refit_delta = bootstrap["contrasts"]["M_refit_minus_F_refit"]
    gap_delta = bootstrap["contrasts"]["gap_difference"]
    harm = primary["observed"] >= 0.005
    gap_supported = harm and refit_delta["observed"] >= -0.002 and gap_delta["observed"] >= 0.005
    result = {
        "fingerprint": data["fingerprint"],
        "scope": "one seed46 one warmup epoch; repeatedly inspected development annotation proxy",
        "records": {
            "train": TRAIN_RECORDS,
            "development": DEVELOPMENT_RECORDS,
            "development_patients": len(np.unique(patients)),
        },
        "optimization_seed": 46,
        "updates_per_arm": UPDATES,
        "metrics": metrics,
        "bootstrap": bootstrap,
        "decision": {
            "F_joint_minus_M_joint": primary["observed"],
            "encoder_update_harm_point_screen": harm,
            "primary_interval_excludes_zero": primary["interval_95"][0] > 0,
            "M_refit_minus_F_refit": refit_delta["observed"],
            "gap_difference": gap_delta["observed"],
            "readout_gap_interpretation_supported": gap_supported,
            "retain_released_probe": True,
        },
        "arm_receipt_sha256": arm_hashes,
        "runtime_gate": runtime,
        "report_seconds": time.monotonic() - started,
        "calibration_test_evaluated": False,
    }
    write_json_atomic(OUT / "report.json", result)
    lines = [
        "# Experiment 016 v9: matched encoder-update intervention",
        "",
        "Development-only one-seed46 one-epoch comparison; no calibration/test.",
        "",
        "| Readout | AUROC | AP | BCE | Patient-fold sensitivity | Patient-fold specificity |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for name, row in metrics.items():
        lines.append(
            f"| {name} | {row['auroc']:.5f} | {row['average_precision']:.5f} | "
            f"{row['bce']:.5f} | {row['mean_fold_sensitivity']:.5f} | "
            f"{row['mean_fold_specificity']:.5f} |"
        )
    lines += [
        "",
        f"Primary F−M joint AUROC: {primary['observed']:+.6f}; paired interval {primary['interval_95']}.",
        f"M−F refit: {refit_delta['observed']:+.6f}; gap difference: {gap_delta['observed']:+.6f}.",
        f"Harm screen: {harm}; readout-gap interpretation: {gap_supported}.",
        f"Paired patient draws: {bootstrap['valid']}/{bootstrap['requested']} valid.",
        "The result conditions on one encoder optimization seed and inspected development patients.",
        "Full encoder gradients/clipping were retained in frozen F; no GPU-efficient shortcut was used.",
        "",
    ]
    (OUT / "report.md").write_text("\n".join(lines))
    return result


def main() -> None:
    """Run a single verified v9 stage with no automatic successor."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("check", "profile", "train", "report"), required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    args = parser.parse_args()
    torch.set_num_threads(1)
    OUT.mkdir(parents=True, exist_ok=True)
    if args.stage == "check":
        if args.device != "cpu":
            raise ValueError("CPU check requires CPU")
        result = check()
    else:
        started = time.monotonic()
        data = inputs()
        if args.stage == "profile":
            result = profile(data, args.device, time.monotonic() - started)
        elif args.stage == "train":
            result = train(data, args.device)
        else:
            result = report(data)
    print(
        json.dumps(
            {
                "stage": args.stage,
                "status": "complete",
                "projected_total_seconds": result.get("projected_total_seconds"),
                "decision": result.get("decision"),
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
