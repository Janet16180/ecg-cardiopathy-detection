"""Versioned seed-47 xECG replication; matched training loop from V9."""

from __future__ import annotations

import gc
import hashlib
import json
import time
from collections.abc import Callable
from pathlib import Path

import numpy as np
import torch

from ecg_experiment.files import sha256_file, write_json_atomic
from ecg_experiment.xecg_encoder_motion_v9 import (
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
from scripts.experiments import run_xecg_encoder_motion016_v9 as v9

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/experiment016_encoder_motion_replication_v14"
v5 = v9.v5
v6 = v9.v6
datasets = v9.datasets
_verify_checkpoint = v9._verify_checkpoint
_replay = v9._replay
_fit_final_probe = v9._fit_final_probe
predict = v9.predict


def run_arm(  # noqa: C901 - copied scientific loop with only explicit identity/cost guards
    data: dict,
    arm: str,
    seed: int,
    device: str,
    mode: str,
    cost_guard: Callable[[str, int | None, float | None], None] | None = None,
) -> dict:
    """Run one full matched pipeline, retaining update logs/checkpoints/features/probe."""
    if seed != 47 or mode != "production":
        raise ValueError("V14 production requires fresh seed 47 and production output identity")
    if device != "cuda" or "V100" not in torch.cuda.get_device_name(0):
        raise RuntimeError("V9 full pipeline requires the V100")
    directory = OUT / mode / arm
    directory.mkdir(parents=True, exist_ok=True)
    complete_path = directory / "complete.json"
    source = {"fingerprint": data["fingerprint"], "arm": arm, "seed": seed, "mode": mode}
    if complete_path.exists() or (directory / "resume.pt").exists():
        raise RuntimeError("V14 production must start fresh; retry and resume are forbidden")
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
    block_started = started
    with log_path.open("x", buffering=1) as log:
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
                    if cost_guard is not None:
                        block_seconds = time.monotonic() - block_started
                        cost_guard("checkpoint", updates, block_seconds)
                        block_started = time.monotonic()
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
    if cost_guard is not None:
        cost_guard("tail_start", updates, None)
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
    if cost_guard is not None:
        cost_guard("tail_complete", updates, time.monotonic() - block_started)
    return receipt
