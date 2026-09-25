#!/usr/bin/env python3
"""Staged, development-only 016 clean-cohort DropPath rescue."""

from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.metadata
import json
import sys
import time
from collections.abc import Callable
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, Dataset

from ecg_experiment.files import read_csv, sha256_file, write_json_atomic, write_npz_atomic
from ecg_experiment.gpu import gpu_lock
from ecg_experiment.reproducibility import cpu_state, flat_rng_state
from ecg_experiment.xecg_rescue_analysis import development_metrics, paired_patient_bootstrap
from ecg_experiment.xecg_rescue_training import (
    build_model,
    cpu_nested,
    one_epoch,
    predict,
    restore_checkpoint,
    sequential_replay,
)
from scripts.experiments.run_xecg_probe_finetune016 import (
    CachedECGs,
    affine_from_standardized,
    cache_fingerprint,
    checkpoint_fingerprint,
    load_manifests,
    manifest_dir,
    source_tree_sha256,
)

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs/experiment016_droppath_rescue_v5"
HISTORICAL = ROOT / "outputs/experiment016_xecg_probe_finetune/features"
CLEAN = ROOT / "outputs/data_quality/clean_rerun_preflight_v1"
MANIFEST = ROOT / "data/processed/ptbxl"
CACHE = ROOT / "data/processed/ptbxl/xecg_views"
RAW = ROOT / "data/raw/ptb-xl/1.0.3"
RELEASE = ROOT / "third_party/checkpoints/xecg"
ARMS = ("legacy", "residual", "off")
SOURCE_FILES = (
    "docs/experiment-016-droppath-rescue.md",
    "docs/experiment-016-droppath-rescue-v5.md",
    "scripts/experiments/run_xecg_droppath_rescue016.py",
    "ecg_experiment/xecg_droppath_rescue.py",
    "ecg_experiment/xecg_rescue_training.py",
    "ecg_experiment/xecg_rescue_analysis.py",
    "scripts/experiments/run_xecg_probe_finetune016.py",
    "ecg_experiment/xecg.py",
    "ecg_experiment/evaluation.py",
    "ecg_experiment/files.py",
    "ecg_experiment/reproducibility.py",
    "ecg_experiment/gpu.py",
    "tests/test_xecg_droppath_rescue.py",
    "third_party/checkpoints/xecg/xECG.py",
    "third_party/xecg-deps/xlstm/blocks/xlstm_block.py",
)


def environment_metadata() -> dict[str, str]:
    """Return primitive strings safe for restricted checkpoint loading."""
    return {
        "python": sys.version.split()[0],
        "torch": str(torch.__version__),
        "numpy": np.__version__,
        "scikit_learn": importlib.metadata.version("scikit-learn"),
        "xlstm": "2.0.4 (vendored source tree hashed above)",
        "uv_lock_sha256": sha256_file(ROOT / "uv.lock"),
        "pyproject_sha256": sha256_file(ROOT / "pyproject.toml"),
    }


def frozen_inputs() -> dict:  # noqa: C901 - split and hash validations are one boundary
    """Verify original split, clean overlay, 100-Hz cache and pooled features."""
    started = time.monotonic()
    rows, development, _, manifests = load_manifests(MANIFEST, "full")
    full = rows["labeled_train"]
    overlay = read_csv(CLEAN / "labels_fraction1.csv")
    expected = {row["record_id"]: row for row in overlay}
    train = [row for row in full if f"ptbxl:{row['ecg_id']}" in expected]
    if len(full) != 15360 or len(train) != 15359 or len(expected) != 15359:
        raise ValueError("Clean xECG training cohort differs from frozen 15,359")
    excluded = {row["ecg_id"] for row in full} - {row["ecg_id"] for row in train}
    if excluded != {"12722"}:
        raise ValueError(f"Unexpected clean xECG exclusion: {excluded}")
    for row in train:
        clean = expected[f"ptbxl:{row['ecg_id']}"]
        if clean["patient_id"] != f"ptbxl:{row['patient_id']}" or clean["target"] != row["target"]:
            raise ValueError("Clean labels or patient identities differ")
    preflight = json.loads((CLEAN / "receipt.json").read_text())
    if preflight["output_sha256"]["labels_fraction1.csv"] != sha256_file(CLEAN / "labels_fraction1.csv"):
        raise ValueError("Clean preflight label receipt changed")
    views, index, cache = cache_fingerprint(CACHE, RAW, manifest_dir(MANIFEST, "full"))
    if any(int(row["ecg_id"]) not in index for row in train + development):
        raise ValueError("Clean training/development ECG is absent from the verified cache")
    receipt = json.loads((HISTORICAL / "receipt.json").read_text())
    if receipt["sha256"] != {
        name: sha256_file(HISTORICAL / name) for name in ("features.npy", "ecg_ids.npy")
    }:
        raise ValueError("Historical released xECG feature cache hash changed")
    if receipt["fingerprint"]["checkpoint_sha256"] != checkpoint_fingerprint(RELEASE) or receipt[
        "fingerprint"
    ]["cache_sha256"] != {"metadata_sha256": cache["metadata_sha256"], "views_sha256": cache["views_sha256"]}:
        raise ValueError("Historical features were built from another release or cache")
    features = np.load(HISTORICAL / "features.npy", mmap_mode="r")
    ids = np.load(HISTORICAL / "ecg_ids.npy")
    original_ids = np.array([int(row["ecg_id"]) for row in full + development])
    if features.shape != (len(original_ids), 1024) or not np.array_equal(ids, original_ids):
        raise ValueError("Historical feature identity/order differs")
    selected = np.array([i for i, row in enumerate(full) if row["ecg_id"] != "12722"])
    clean_features = np.asarray(features[selected], dtype=np.float32)
    dev_features = np.asarray(features[len(full) :], dtype=np.float32)
    if not np.isfinite(clean_features).all() or not np.isfinite(dev_features).all():
        raise ValueError("Nonfinite released xECG pooled features")
    source_hashes = {name: sha256_file(ROOT / name) for name in SOURCE_FILES}
    fingerprint = {
        "sources": source_hashes,
        "manifest": manifests,
        "clean_labels_sha256": sha256_file(CLEAN / "labels_fraction1.csv"),
        "clean_receipt_sha256": sha256_file(CLEAN / "receipt.json"),
        "feature_receipt_sha256": sha256_file(HISTORICAL / "receipt.json"),
        "feature_hashes": receipt["sha256"],
        "cache_hashes": {key: cache[key] for key in ("metadata_sha256", "views_sha256")},
        "release_hashes": checkpoint_fingerprint(RELEASE),
        "xlstm_source_sha256": source_tree_sha256(ROOT / "third_party/xecg-deps/xlstm"),
        "environment": environment_metadata(),
        "train_records": len(train),
        "development_records": len(development),
        "excluded_id": "12722",
        "seed": 42,
        "epochs": 2,
    }
    return {
        "train": train,
        "development": development,
        "views": views,
        "index": index,
        "train_features": clean_features,
        "development_features": dev_features,
        "fingerprint": fingerprint,
        "preparation_seconds": time.monotonic() - started,
    }


def fit_clean_probe(inputs: dict) -> dict:
    """Refit historical C=.01 on clean training features, with no search."""
    path = OUT / "probe.npz"
    receipt_path = OUT / "probe.json"
    if receipt_path.is_file():
        saved = json.loads(receipt_path.read_text())
        if saved["fingerprint"] != inputs["fingerprint"] or saved["sha256"] != sha256_file(path):
            raise ValueError("Existing clean probe differs from frozen inputs")
        return saved
    if path.exists():
        raise FileExistsError("Unreceipted clean probe exists")
    started = time.monotonic()
    x = inputs["train_features"].astype(np.float64)
    y = np.array([int(row["target"]) for row in inputs["train"]])
    scaler = StandardScaler().fit(x)
    model = LogisticRegression(C=0.01, solver="lbfgs", max_iter=3000, random_state=42)
    model.fit(scaler.transform(x), y)
    dev = inputs["development_features"].astype(np.float64)
    native = model.decision_function(scaler.transform(dev))
    weight, bias = affine_from_standardized(
        model.coef_[0], float(model.intercept_[0]), scaler.mean_, scaler.scale_
    )
    folded = dev @ weight + bias
    if not np.allclose(native, folded, rtol=1e-10, atol=1e-9):
        raise RuntimeError("Clean frozen probe affine logits differ")
    write_npz_atomic(
        path,
        raw_weight=weight,
        raw_bias=np.array(bias),
        mean=scaler.mean_,
        scale=scaler.scale_,
        coefficient=model.coef_[0],
        intercept=model.intercept_[0],
    )
    labels = np.array([int(row["target"]) for row in inputs["development"]])
    saved = {
        "fingerprint": inputs["fingerprint"],
        "C": 0.01,
        "iterations": int(model.n_iter_[0]),
        "development_auroc": float(roc_auc_score(labels, native)),
        "max_affine_logit_difference": float(np.max(np.abs(native - folded))),
        "seconds": time.monotonic() - started,
        "sha256": sha256_file(path),
    }
    write_json_atomic(receipt_path, saved)
    return saved


def diagnostic_rows(inputs: dict) -> list[dict[str, str]]:
    """Choose the first fixed 64 clean positive and negative training ECGs."""
    positives = [row for row in inputs["train"] if row["target"] == "1"][:64]
    negatives = [row for row in inputs["train"] if row["target"] == "0"][:64]
    if len(positives) != 64 or len(negatives) != 64:
        raise ValueError("Diagnostic set lacks 64 records in each class")
    return positives + negatives


@torch.inference_mode()
def diagnostic_pass(model: nn.Module, dataset: Dataset, device: str) -> tuple[np.ndarray, np.ndarray]:
    """Return pooled features and logits without consuming training gradients."""
    logits, features = [], []
    for signal, _ in DataLoader(dataset, batch_size=16, shuffle=False, num_workers=0):
        pooled, _ = model.backbone(signal.to(device))
        features.append(pooled.float().cpu().numpy())
        logits.append(model.head(pooled).squeeze(-1).float().cpu().numpy())
    return np.concatenate(logits), np.concatenate(features)


def diagnostic(inputs: dict, device: str) -> dict:  # noqa: C901 - paired arms/seeds share checks
    """Measure paired initial train/eval mode disturbance before updates."""
    path = OUT / "diagnostic.json"
    if path.is_file():
        result = json.loads(path.read_text())
        if result["fingerprint"] != inputs["fingerprint"]:
            raise ValueError("Diagnostic belongs to other inputs")
        return result
    started = time.monotonic()
    rows = diagnostic_rows(inputs)
    data = CachedECGs(inputs["views"], inputs["index"], rows)
    labels = np.array([int(row["target"]) for row in rows])
    result = {"fingerprint": inputs["fingerprint"], "ecg_ids": [row["ecg_id"] for row in rows], "arms": {}}
    reference = None
    with np.load(OUT / "probe.npz") as probe:
        feature_by_id = {row["ecg_id"]: inputs["train_features"][i] for i, row in enumerate(inputs["train"])}
        cached = np.stack([feature_by_id[row["ecg_id"]] for row in rows])
        expected_logits = cached.astype(np.float64) @ probe["raw_weight"] + float(probe["raw_bias"])
    for arm in ARMS:
        model, mask, _, _, _ = build_model(RELEASE, OUT / "probe.npz", arm, device, len(inputs["train"]))
        model.eval()
        eval_logits, eval_features = diagnostic_pass(model, data, device)
        if reference is None:
            reference = eval_logits
            difference = float(np.max(np.abs(eval_logits - expected_logits)))
            if not np.allclose(eval_logits, expected_logits, rtol=2e-5, atol=2e-4):
                raise RuntimeError(f"Fresh release logits differ from cached frozen probe: {difference}")
            result["initial_probe_logit_max_difference"] = difference
        elif not np.allclose(reference, eval_logits, rtol=0, atol=1e-6):
            raise RuntimeError("Initial inference logits differ between arms")
        draws = []
        for seed in range(16000, 16008):
            model.train()
            mask.generator.manual_seed(seed)
            train_logits, train_features = diagnostic_pass(model, data, device)
            cosine = np.sum(eval_features * train_features, axis=1) / (
                np.linalg.norm(eval_features, axis=1) * np.linalg.norm(train_features, axis=1) + 1e-12
            )
            draws.append(
                {
                    "seed": seed,
                    "bce": float(np.mean(np.logaddexp(0, train_logits) - labels * train_logits)),
                    "auroc": float(roc_auc_score(labels, train_logits)),
                    "mean_abs_logit_shift": float(np.mean(np.abs(train_logits - eval_logits))),
                    "max_abs_feature_shift": float(np.max(np.abs(train_features - eval_features))),
                    "median_feature_cosine": float(np.median(cosine)),
                    "mean_feature_norm": float(np.mean(np.linalg.norm(train_features, axis=1))),
                }
            )
        if arm == "off" and any(
            row["mean_abs_logit_shift"] > 2e-4 or row["max_abs_feature_shift"] > 2e-4 for row in draws
        ):
            raise RuntimeError("Off train/eval initial features/logits differ beyond FP32 tolerance")
        if arm == "residual":
            result["blockwise"] = blockwise_check(model, data, device)
        result["arms"][arm] = {
            "eval_bce": float(np.mean(np.logaddexp(0, eval_logits) - labels * eval_logits)),
            "eval_auroc": float(roc_auc_score(labels, eval_logits)),
            "eval_mean_feature_norm": float(np.mean(np.linalg.norm(eval_features, axis=1))),
            "draws": draws,
            "mean_logit_shift": float(np.mean([d["mean_abs_logit_shift"] for d in draws])),
        }
        del model
        if device == "cuda":
            torch.cuda.empty_cache()
    result["seconds"] = time.monotonic() - started
    write_json_atomic(path, result)
    return result


@torch.inference_mode()
def blockwise_check(model: nn.Module, dataset: Dataset, device: str) -> list[dict]:
    """Check exact conditional means of each complete block on captured inputs."""
    model.eval()
    blocks = model.backbone.core.model.blocks
    captured = {}

    def capture(index: int) -> Callable:
        def hook(_module: nn.Module, arguments: tuple[torch.Tensor, ...]) -> None:
            captured[index] = arguments[0].detach().clone()

        return hook

    hooks = [block.register_forward_pre_hook(capture(index)) for index, block in enumerate(blocks)]
    signal, _ = next(iter(DataLoader(dataset, batch_size=16, shuffle=False, num_workers=0)))
    model(signal.to(device))
    for hook in hooks:
        hook.remove()
    rows = []
    for index, block in enumerate(blocks):
        x = captured[index]
        complete = block(x)
        probability = float(model.backbone.core.dropout_rates[index])
        if probability:
            keep = 1 - probability
            legacy_mean = keep * (complete / keep) + probability * x
            residual_mean = keep * (x + (complete - x) / keep) + probability * x
        else:
            legacy_mean = residual_mean = complete
        legacy_error = float((legacy_mean - (complete + probability * x)).abs().max())
        residual_error = float((residual_mean - complete).abs().max())
        if legacy_error > 5e-5 or residual_error > 5e-5:
            raise RuntimeError(f"Block {index} conditional mean formula failed")
        rows.append(
            {
                "block": index,
                "drop_probability": probability,
                "input_norm_mean": float(x.flatten(1).norm(dim=1).mean()),
                "complete_norm_mean": float(complete.flatten(1).norm(dim=1).mean()),
                "legacy_conditional_mean_shift_norm": float(
                    (legacy_mean - complete).flatten(1).norm(dim=1).mean()
                ),
                "residual_conditional_mean_shift_norm": float(
                    (residual_mean - complete).flatten(1).norm(dim=1).mean()
                ),
                "legacy_formula_max_error": legacy_error,
                "residual_formula_max_error": residual_error,
            }
        )
    return rows


def _same_optimizer(left: dict, right: dict) -> bool:  # noqa: C901 - nested state comparison
    """Compare nested optimizer dictionaries after same-device reload."""
    if left.keys() != right.keys():
        return False
    for key in left:
        a, b = left[key], right[key]
        if isinstance(a, dict):
            if not isinstance(b, dict) or not _same_optimizer(a, b):
                return False
        elif isinstance(a, list):
            if not isinstance(b, list) or len(a) != len(b):
                return False
            for item_a, item_b in zip(a, b, strict=True):
                if isinstance(item_a, dict):
                    if not _same_optimizer(item_a, item_b):
                        return False
                elif item_a != item_b:
                    return False
        elif isinstance(a, torch.Tensor):
            if not isinstance(b, torch.Tensor) or not torch.equal(a.cpu(), b.cpu()):
                return False
        elif a != b:
            return False
    return True


def profile(inputs: dict, device: str) -> dict:  # noqa: C901 - full pass and replay checks
    """Time each complete arm epoch, development pass, and checkpoint reload."""
    path = OUT / "profile.json"
    if path.exists():
        saved = json.loads(path.read_text())
        if saved["fingerprint"] != inputs["fingerprint"]:
            raise ValueError("Profile belongs to other inputs")
        return saved
    if not (OUT / "diagnostic.json").is_file():
        raise FileNotFoundError("Run correctness diagnostic before full-path profile")
    train = CachedECGs(inputs["views"], inputs["index"], inputs["train"])
    dev = CachedECGs(inputs["views"], inputs["index"], inputs["development"])
    labels = np.array([int(row["target"]) for row in inputs["development"]])
    result = {"fingerprint": inputs["fingerprint"], "arms": {}}
    for arm in ARMS:
        started = time.monotonic()
        if device == "cuda":
            torch.cuda.reset_peak_memory_stats()
        model, mask, optimizer, scheduler, permutation = build_model(
            RELEASE, OUT / "probe.npz", arm, device, len(train)
        )
        checkpoint = OUT / f"profile_{arm}.pt"
        arm_fingerprint = {"source": inputs["fingerprint"], "arm": arm, "stage": "profile"}
        epoch = one_epoch(
            model,
            mask,
            train,
            device,
            optimizer,
            scheduler,
            permutation,
            0,
            checkpoint=checkpoint,
            fingerprint=arm_fingerprint,
        )
        logits = predict(model, dev, device)
        saved_state = cpu_state(model)
        saved_optimizer = optimizer.state_dict()
        copy, copy_mask, copy_optimizer, copy_scheduler, copy_permutation = build_model(
            RELEASE, OUT / "probe.npz", arm, device, len(train)
        )
        restored = restore_checkpoint(
            checkpoint, arm_fingerprint, copy, copy_mask, copy_optimizer, copy_scheduler, copy_permutation
        )
        if restored["epoch"] != 0 or restored["next_index"] != len(train) or restored["updates"] != 240:
            raise RuntimeError("Profile exact update position failed")
        reloaded_state = cpu_state(copy)
        if not all(torch.equal(saved_state[name], reloaded_state[name]) for name in saved_state):
            raise RuntimeError("Profile model roundtrip failed")
        if not _same_optimizer(saved_optimizer, copy_optimizer.state_dict()):
            raise RuntimeError("Profile optimizer roundtrip failed")
        if (
            scheduler.state_dict() != copy_scheduler.state_dict()
            or not torch.equal(mask.get_rng_state(), copy_mask.get_rng_state())
            or not torch.equal(permutation.get_state(), copy_permutation.get_state())
        ):
            raise RuntimeError("Profile scheduler or isolated RNG roundtrip failed")
        # Preserve the first restored model; release the active epoch model and
        # optimizer before constructing or updating the next replay model.
        first_ready = (copy, copy_mask, copy_optimizer, copy_scheduler, copy_permutation)
        del model, mask, optimizer, scheduler, permutation, saved_state, saved_optimizer
        del copy, copy_mask, copy_optimizer, copy_scheduler, copy_permutation, reloaded_state
        gc.collect()
        if device == "cuda":
            torch.cuda.empty_cache()
        released_allocated = []

        def run_next_update(
            arm_name: str = arm,
            checkpoint_path: Path = checkpoint,
            arm_source: dict = arm_fingerprint,
            released: list[int] = released_allocated,
        ) -> dict:
            nonlocal first_ready
            if first_ready is None:
                active = build_model(RELEASE, OUT / "probe.npz", arm_name, device, len(train))
                restore_checkpoint(checkpoint_path, arm_source, *active)
            else:
                active, first_ready = first_ready, None
            current_model, current_mask, current_optimizer, current_scheduler, current_permutation = active
            order_generator = torch.Generator().set_state(current_permutation.get_state())
            first_batch = torch.randperm(len(train), generator=order_generator)[:64]
            batch_order_sha256 = hashlib.sha256(first_batch.numpy().tobytes()).hexdigest()
            one_epoch(
                current_model,
                current_mask,
                train,
                device,
                current_optimizer,
                current_scheduler,
                current_permutation,
                1,
                checkpoint=None,
                fingerprint=arm_source,
                max_updates=1,
            )
            snapshot = {
                "model": cpu_state(current_model),
                "optimizer": cpu_nested(current_optimizer.state_dict()),
                "scheduler": cpu_nested(current_scheduler.state_dict()),
                "mask_rng": current_mask.get_rng_state(),
                "permutation_rng": current_permutation.get_state(),
                "global_rng": cpu_nested(flat_rng_state()),
                "batch_order_sha256": batch_order_sha256,
            }
            del active, current_model, current_mask, current_optimizer
            del current_scheduler, current_permutation
            gc.collect()
            if device == "cuda":
                torch.cuda.empty_cache()
                released.append(torch.cuda.memory_allocated())
                if released[-1] > 256 * 1024**2:
                    raise RuntimeError("Replay retained active GPU model/optimizer references")
            return snapshot

        replay_check = sequential_replay(run_next_update, device)
        if device == "cuda":
            torch.cuda.synchronize()
        result["arms"][arm] = {
            "complete_pass_seconds": time.monotonic() - started,
            "epoch": epoch,
            "development_auroc": float(roc_auc_score(labels, logits)),
            "checkpoint_sha256": sha256_file(checkpoint),
            "model_optimizer_scheduler_rng_roundtrip": True,
            "resumed_next_update_same_device": True,
            "resumed_next_update_bitwise": replay_check["tensors_with_numeric_difference"] == 0,
            "resumed_next_update_numeric_check": replay_check,
            "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated() if device == "cuda" else None,
            "peak_cuda_reserved_bytes": torch.cuda.max_memory_reserved() if device == "cuda" else None,
            "released_cuda_allocated_bytes_after_replays": released_allocated,
        }
        print(json.dumps({"stage": "016_rescue_profile", "arm": arm, **result["arms"][arm]}), flush=True)
        checkpoint.unlink()
        if device == "cuda":
            torch.cuda.empty_cache()
    result["peak_cuda_memory_bytes"] = torch.cuda.max_memory_allocated() if device == "cuda" else None
    write_json_atomic(path, result)
    return result


def cost_gate(inputs: dict) -> dict:
    """Apply the frozen two-hour equation to measured full-path profiles."""
    diagnostic_receipt = json.loads((OUT / "diagnostic.json").read_text())
    probe_receipt = json.loads((OUT / "probe.json").read_text())
    profile_receipt = json.loads((OUT / "profile.json").read_text())
    check_receipt = json.loads((OUT / "check.json").read_text())
    for receipt in (diagnostic_receipt, probe_receipt, profile_receipt, check_receipt):
        if receipt["fingerprint"] != inputs["fingerprint"]:
            raise ValueError("Cost receipt belongs to different frozen inputs")
    passes = [profile_receipt["arms"][arm]["complete_pass_seconds"] for arm in ARMS]
    measured = (
        check_receipt["preparation_seconds"]
        + probe_receipt["seconds"]
        + diagnostic_receipt["seconds"]
        + sum(passes)
    )
    # Profile includes development pass, one checkpoint and same-device replay.
    remaining = 6 * max(passes) + 180.0
    projected = measured + 1.25 * remaining
    gate = {
        "measured_preparation_and_profiles_seconds": measured,
        "slow_complete_pass_seconds": max(passes),
        "remaining_six_passes_plus_report_seconds": remaining,
        "projected_total_seconds": projected,
        "ceiling_seconds": 7200,
        "passed": projected <= 7200,
        "formula": "measured_preparation_and_profiles + 1.25*(six_slowest_complete_passes + 180s report)",
    }
    write_json_atomic(OUT / "cost_gate.json", gate)
    if not gate["passed"]:
        raise RuntimeError(f"016 rescue exceeds two-hour gate: {projected:.1f}s")
    return gate


def trajectory_point(
    model: nn.Module,
    inputs: dict,
    device: str,
    initial_blocks: list,
    phase: str,
    grad_norm: float | None = None,
) -> dict:
    """Track fixed train records and parameter drift without model selection."""
    rows = diagnostic_rows(inputs)
    data = CachedECGs(inputs["views"], inputs["index"], rows)
    prior_mode = model.training
    model.eval()
    logits, features = diagnostic_pass(model, data, device)
    if prior_mode:
        model.train()
    feature_by_id = {
        row["ecg_id"]: inputs["train_features"][index] for index, row in enumerate(inputs["train"])
    }
    release = np.stack([feature_by_id[row["ecg_id"]] for row in rows])
    cosine = np.sum(release * features, axis=1) / (
        np.linalg.norm(release, axis=1) * np.linalg.norm(features, axis=1) + 1e-12
    )
    with np.load(OUT / "probe.npz") as probe:
        frozen_logits = features.astype(np.float64) @ probe["raw_weight"] + float(probe["raw_bias"])
        release_logits = release.astype(np.float64) @ probe["raw_weight"] + float(probe["raw_bias"])
    blocks = model.backbone.core.model.blocks
    relative_changes = {}
    for index, block in enumerate(blocks):
        changed, baseline = 0.0, 0.0
        for parameter, original in zip(block.parameters(), initial_blocks[index], strict=True):
            current = parameter.detach().float().cpu()
            changed += float(torch.sum((current - original) ** 2))
            baseline += float(torch.sum(original**2))
        relative_changes[f"block_{index}"] = (changed / max(baseline, 1e-24)) ** 0.5
    return {
        "phase": phase,
        "head_weight_norm": float(model.head.weight.detach().norm()),
        "head_bias": float(model.head.bias.detach()),
        "block_relative_parameter_change": relative_changes,
        "gradient_norm_before_clip": grad_norm,
        "median_feature_cosine_to_release": float(np.median(cosine)),
        "mean_abs_logit_shift_under_frozen_probe": float(np.mean(np.abs(frozen_logits - release_logits))),
        "mean_current_logit": float(np.mean(logits)),
    }


def train_arm(inputs: dict, device: str, arm: str) -> dict:  # noqa: C901 - resume states
    """Run the fixed two-epoch arm, resuming only from exact-update receipt."""
    directory = OUT / arm
    directory.mkdir(parents=True, exist_ok=True)
    receipt_path = directory / "complete.json"
    fingerprint = {"source": inputs["fingerprint"], "arm": arm, "stage": "train"}
    if receipt_path.exists():
        receipt = json.loads(receipt_path.read_text())
        if receipt["fingerprint"] != fingerprint:
            raise ValueError("Completed arm has another fingerprint")
        for name, digest in receipt["sha256"].items():
            if digest != sha256_file(directory / name):
                raise ValueError("Completed arm artifact hash changed")
        return receipt
    train = CachedECGs(inputs["views"], inputs["index"], inputs["train"])
    dev = CachedECGs(inputs["views"], inputs["index"], inputs["development"])
    labels = np.array([int(row["target"]) for row in inputs["development"]])
    patients = np.array([row["patient_id"] for row in inputs["development"]])
    model, mask, optimizer, scheduler, permutation = build_model(
        RELEASE, OUT / "probe.npz", arm, device, len(train)
    )
    checkpoint = directory / "resume.pt"
    history_path = directory / "history.json"
    trajectory_path = directory / "trajectory.json"
    history = json.loads(history_path.read_text()) if history_path.exists() else []
    trajectory = json.loads(trajectory_path.read_text()) if trajectory_path.exists() else []
    initial_blocks = [
        [parameter.detach().float().cpu().clone() for parameter in block.parameters()]
        for block in model.backbone.core.model.blocks
    ]
    saved = None
    if checkpoint.exists():
        saved = restore_checkpoint(checkpoint, fingerprint, model, mask, optimizer, scheduler, permutation)
        completed_epoch = int(saved["epoch"]) + 1
        if saved["next_index"] == len(train) and len(history) == completed_epoch:
            # A crash between the epoch checkpoint and metrics receipt is recoverable.
            logits = predict(model, dev, device)
            np.save(directory / f"development_logits_epoch{completed_epoch}.npy", logits)
            history.append(
                {
                    "epoch": completed_epoch,
                    "training": {
                        "recovered_from_epoch_end_checkpoint": True,
                        "updates": int(saved["updates"]),
                    },
                    **development_metrics(labels, patients, logits),
                }
            )
            trajectory.append(
                trajectory_point(model, inputs, device, initial_blocks, f"epoch_{completed_epoch}")
            )
            write_json_atomic(history_path, history)
            write_json_atomic(trajectory_path, trajectory)
    else:
        # The cached released-feature probe is the exact no-update reference.
        with np.load(OUT / "probe.npz") as probe:
            epoch0 = inputs["development_features"].astype(np.float64) @ probe["raw_weight"] + float(
                probe["raw_bias"]
            )
        np.save(directory / "development_logits_epoch0.npy", epoch0)
        history = [{"epoch": 0, **development_metrics(labels, patients, epoch0)}]
        trajectory = [trajectory_point(model, inputs, device, initial_blocks, "initial")]
        write_json_atomic(history_path, history)
        write_json_atomic(trajectory_path, trajectory)
    first_epoch = 0 if saved is None else int(saved["epoch"])
    if saved is not None and saved["next_index"] == len(train):
        first_epoch += 1
    for epoch in range(first_epoch, 2):
        partial = (
            saved
            if saved is not None and saved["epoch"] == epoch and saved["next_index"] < len(train)
            else None
        )

        def first_update(updated_model: nn.Module, norm: float) -> None:
            trajectory.append(
                trajectory_point(updated_model, inputs, device, initial_blocks, "first_update", norm)
            )
            write_json_atomic(trajectory_path, trajectory)

        epoch_result = one_epoch(
            model,
            mask,
            train,
            device,
            optimizer,
            scheduler,
            permutation,
            epoch,
            checkpoint=checkpoint,
            fingerprint=fingerprint,
            initial=partial,
            after_update=first_update if epoch == 0 else None,
        )
        logits = predict(model, dev, device)
        np.save(directory / f"development_logits_epoch{epoch + 1}.npy", logits)
        history.append(
            {"epoch": epoch + 1, "training": epoch_result, **development_metrics(labels, patients, logits)}
        )
        trajectory.append(trajectory_point(model, inputs, device, initial_blocks, f"epoch_{epoch + 1}"))
        write_json_atomic(history_path, history)
        write_json_atomic(trajectory_path, trajectory)
        print(
            json.dumps(
                {"stage": "016_rescue_train", "arm": arm, "epoch": epoch + 1, "auroc": history[-1]["auroc"]}
            ),
            flush=True,
        )
        saved = None
    if len(history) != 3 or [row["epoch"] for row in history] != [0, 1, 2]:
        raise RuntimeError("Fixed epoch-0/1/2 outcomes are incomplete")
    files = ["history.json", "trajectory.json", "resume.pt"] + [
        f"development_logits_epoch{epoch}.npy" for epoch in range(3)
    ]
    receipt = {
        "fingerprint": fingerprint,
        "sha256": {name: sha256_file(directory / name) for name in files},
        "final_epoch": 2,
        "optimizer_updates": 480,
        "record_exposures": 30718,
    }
    write_json_atomic(receipt_path, receipt)
    return receipt


def report(inputs: dict) -> dict:
    """Compute prespecified final contrasts and exploratory decision gates."""
    labels = np.array([int(row["target"]) for row in inputs["development"]])
    patients = np.array([row["patient_id"] for row in inputs["development"]])
    history = {}
    final = {}
    for arm in ARMS:
        receipt = json.loads((OUT / arm / "complete.json").read_text())
        if receipt["fingerprint"] != {"source": inputs["fingerprint"], "arm": arm, "stage": "train"}:
            raise ValueError("Arm completion belongs to another experiment")
        for name, digest in receipt["sha256"].items():
            if digest != sha256_file(OUT / arm / name):
                raise ValueError("Arm artifact hash changed before reporting")
        history[arm] = json.loads((OUT / arm / "history.json").read_text())
        final[arm] = np.load(OUT / arm / "development_logits_epoch2.npy")
    bootstrap = paired_patient_bootstrap(labels, patients, final)
    probe_auc = history["legacy"][0]["auroc"]
    residual_legacy = history["residual"][2]["auroc"] - history["legacy"][2]["auroc"]
    off_legacy = history["off"][2]["auroc"] - history["legacy"][2]["auroc"]
    residual_off = history["residual"][2]["auroc"] - history["off"][2]["auroc"]
    diag = json.loads((OUT / "diagnostic.json").read_text())
    support = (
        residual_legacy >= 0.005
        and diag["arms"]["residual"]["mean_logit_shift"] < diag["arms"]["legacy"]["mean_logit_shift"]
    )
    passing = []
    for arm, advantage in (("residual", residual_legacy), ("off", off_legacy)):
        sensitivity_harm = (
            history["legacy"][0]["mean_fold_sensitivity"] - history[arm][2]["mean_fold_sensitivity"]
        )
        if advantage >= 0.005 and history[arm][2]["auroc"] >= probe_auc - 0.002 and sensitivity_harm <= 0.005:
            passing.append(arm)
    preferred = None
    if passing:
        preferred = (
            "residual" if "residual" in passing and ("off" not in passing or residual_off >= 0.002) else "off"
        )
    result = {
        "fingerprint": inputs["fingerprint"],
        "records": {
            "train": 15359,
            "development": len(labels),
            "development_patients": len(np.unique(patients)),
        },
        "history": history,
        "final_epoch2_auc_contrasts": {
            "residual_minus_legacy": residual_legacy,
            "off_minus_legacy": off_legacy,
            "residual_minus_off": residual_off,
        },
        "paired_patient_bootstrap": bootstrap,
        "decision": {
            "correction_supports_mechanism_screen": support,
            "partial_if_below_probe": support and history["residual"][2]["auroc"] < probe_auc - 0.002,
            "practical_rescue_arms": passing,
            "preferred_for_seed43_replication": preferred,
        },
        "scope": "development diagnostic-annotation proxy only; no calibration/test",
    }
    write_json_atomic(OUT / "report.json", result)
    lines = [
        "# Experiment 016 DropPath rescue",
        "",
        "Seed-42 clean-cohort, development-only mechanistic screen; no calibration or test evaluation.",
        "",
        (
            f"Frozen clean probe AUROC: {probe_auc:.5f} on {len(labels)} development ECGs "
            f"/ {len(np.unique(patients))} patients."
        ),
        "",
        "| Arm | Epoch 0 | Epoch 1 | Epoch 2 | Epoch-2 sensitivity |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for arm in ARMS:
        rows = history[arm]
        lines.append(
            f"| {arm} | {rows[0]['auroc']:.5f} | {rows[1]['auroc']:.5f} | "
            f"{rows[2]['auroc']:.5f} | {rows[2]['mean_fold_sensitivity']:.5f} |"
        )
    lines += [
        "",
        (
            f"Final Residual−Legacy AUROC: {residual_legacy:+.5f}; "
            f"Off−Legacy: {off_legacy:+.5f}; Residual−Off: {residual_off:+.5f}."
        ),
        (
            f"Mechanism support screen: {support}. Practical rescue arms: {passing}. "
            f"Preferred recipe for a separately costed seed-43 replication: {preferred}."
        ),
        "",
        (
            "Paired 2,000-draw whole-patient bootstrap intervals are in `report.json`; "
            "these do not measure training-seed uncertainty."
        ),
        "The endpoint is an ECG diagnostic annotation proxy and does not establish clinical validity.",
        "",
    ]
    (OUT / "report.md").write_text("\n".join(lines))
    return result


def main() -> None:
    """Execute one frozen stage with verified inputs and a shared GPU lock."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage", choices=("check", "diagnostic", "profile", "train", "report"), required=True
    )
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    args = parser.parse_args()
    torch.set_num_threads(1)
    OUT.mkdir(parents=True, exist_ok=True)
    inputs = frozen_inputs()
    if args.stage == "check":
        probe = fit_clean_probe(inputs)
        write_json_atomic(
            OUT / "check.json",
            {
                "fingerprint": inputs["fingerprint"],
                "preparation_seconds": inputs["preparation_seconds"],
                "clean_probe_auc": probe["development_auroc"],
                "status": "passed_cpu_preflight_no_training",
            },
        )
        print(
            json.dumps({"stage": "016_rescue_check", "clean_probe_auc": probe["development_auroc"]}),
            flush=True,
        )
        return
    if not (OUT / "check.json").is_file():
        raise FileNotFoundError("CPU check must complete before GPU stages")
    if args.stage in {"diagnostic", "profile", "train"} and args.device != "cuda":
        raise ValueError("Real V100 CUDA is required for GPU stages")
    if args.stage == "report":
        report(inputs)
        return
    with gpu_lock(args.device, blocking=False):
        if args.stage == "diagnostic":
            diagnostic(inputs, args.device)
        elif args.stage == "profile":
            profile(inputs, args.device)
            cost_gate(inputs)
        else:
            cost_gate(inputs)
            for arm in ARMS:
                train_arm(inputs, args.device, arm)


if __name__ == "__main__":
    main()
