"""Profile, train, and score a fresh matched 25k GRU/KDA/CKDA CPC study.

No stage runs automatically. The profile is training-only and gates the
complete study before the first comparative update.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np
import torch
from scipy.special import expit
from sklearn.metrics import average_precision_score, roc_auc_score
from threadpoolctl import threadpool_limits
from torch.utils.data import DataLoader

from ecg_experiment.cpc_delta_memory import matched_initial_models
from ecg_experiment.cpc_local_readout import fit_head, replay_logits
from ecg_experiment.cpc_pool import Pool, PoolDataset
from ecg_experiment.cpc_scaling_cached import CachedCPCDataset
from ecg_experiment.cpc_scaling_readout import development_examples, training_examples
from ecg_experiment.delta_memory25k011 import (
    ARMS,
    EXPOSURES,
    ORDER_SEED,
    SELECTION_SEED,
    SUBSET_SIZE,
    frozen_stream,
    paired_patient_auc,
)
from ecg_experiment.files import read_csv, sha256_file, sha256_json, write_json_atomic, write_torch_atomic
from ecg_experiment.gpu import gpu_lock
from ecg_experiment.reproducibility import capture_rng_state, restore_rng_state, seed_everything
from ecg_experiment.training import checked_step

ROOT = Path(__file__).resolve().parents[2]
COHORT = ROOT / "data/processed/sampled_100k_plus_labels_v1"
CACHE = ROOT / "data/processed/sampled_100k_cpc_cache_v1"
PTB_CACHE = ROOT / "data/processed/cpc_pool_40k"
NORMALIZATION = ROOT / "outputs/experiment004_cpc_40k/normalization.json"
OUTPUT = ROOT / "outputs/experiment011_delta_memory_25k_v1"
MODEL_SEED = 9001
BATCH_SIZE = 128
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 0.01
PROFILE_BATCHES = 24
PROFILE_READOUT_RECORDS = 128
CHECKPOINT_INTERVAL = 100
CEILING_SECONDS = 7200
CODE_FILES = (
    "ecg_experiment/cpc.py",
    "ecg_experiment/cpc_delta_memory.py",
    "ecg_experiment/cpc_subset25.py",
    "ecg_experiment/delta_memory25k011.py",
    "ecg_experiment/cpc_scaling_cached.py",
    "ecg_experiment/cpc_scaling_readout.py",
    "ecg_experiment/cpc_local_readout.py",
    "ecg_experiment/cpc_pool.py",
    "ecg_experiment/gpu.py",
    "ecg_experiment/files.py",
    "ecg_experiment/reproducibility.py",
    "ecg_experiment/training.py",
    "scripts/experiments/run_delta_memory25k011.py",
    "docs/experiment-011-delta-memory-25k.md",
)


def _hashes(paths: dict[str, Path]) -> dict[str, str]:
    """Hash each source and artifact without writing to the dataset."""
    return {name: sha256_file(path) for name, path in paths.items()}


def prepare() -> tuple[np.ndarray, dict[str, Any], float]:
    """Verify exact 25k selection and complete cache before GPU work."""
    started = time.monotonic()
    rows = read_csv(COHORT / "train_manifest.csv", required=("record_id", "source", "split"))
    if (len(rows) != EXPOSURES or any(row["split"] != "train" for row in rows)
            or len({row["record_id"] for row in rows}) != EXPOSURES):
        raise ValueError("Frozen source manifest changed")
    selected, order, counts = frozen_stream([row["source"] for row in rows])
    cache_complete = json.loads((CACHE / "complete.json").read_text())
    paths = {
        "cohort_manifest": COHORT / "train_manifest.csv",
        "cohort_metadata": COHORT / "metadata.json",
        "cache_receipt": CACHE / "complete.json",
        "cache_verification": ROOT / "outputs/experiment018_cpc_data_scaling_v2/cache_verification.json",
        "cache_signals": CACHE / "signals.npy",
        "ptb_receipt": PTB_CACHE / "complete.json",
        "ptb_rows": PTB_CACHE / "rows.csv",
        "ptb_ids": PTB_CACHE / "ecg_ids.npy",
        "ptb_signals": PTB_CACHE / "signals.npy",
        "normalization": NORMALIZATION,
        "clean_full_labels": ROOT / "outputs/data_quality/clean_rerun_preflight_v1/labels_fraction1.csv",
        "clean_limited_labels": ROOT / "outputs/data_quality/clean_rerun_preflight_v1/labels_fraction0.1.csv",
        "heldout_references": ROOT / "outputs/data_quality/clean_rerun_preflight_v1/heldout_references.csv",
        **{name: ROOT / name for name in CODE_FILES},
    }
    hashes = _hashes(paths)
    if hashes["cache_signals"] != cache_complete["signals_sha256"]:
        raise ValueError("25k source cache SHA-256 mismatch")
    if (hashes["cohort_manifest"] != cache_complete["identity"]["source_manifest_sha256"]
            or hashes["cohort_metadata"] != cache_complete["identity"]["source_metadata_sha256"]):
        raise ValueError("25k cache and source manifest identity disagree")
    verification = json.loads((ROOT / "outputs/experiment018_cpc_data_scaling_v2"
                               / "cache_verification.json").read_text())
    if (verification["status"] != "passed"
            or verification["cache_sha256"] != hashes["cache_signals"]
            or verification["record_count"] != EXPOSURES):
        raise ValueError("25k source cache independent verification changed")
    ptb_complete = json.loads((PTB_CACHE / "complete.json").read_text())
    for name, key in (("ptb_signals", "signals_sha256"),
                      ("ptb_rows", "rows_sha256"), ("ptb_ids", "ecg_ids_sha256")):
        if hashes[name] != ptb_complete[key]:
            raise ValueError(f"Historical PTB cache changed: {name}")
    pool = Pool(PTB_CACHE)
    norm = json.loads(NORMALIZATION.read_text())
    expected_normalization_source = {
        "train_ids_sha256": sha256_json([row["ecg_id"] for row in pool.train_rows]),
        "rows_sha256": hashes["ptb_rows"],
        "signals_sha256": hashes["ptb_signals"],
        "method": "global per-lead mean and population std, training waveforms only",
    }
    if norm["source"] != expected_normalization_source:
        raise ValueError("Historical training-only normalization source changed")
    identity = {
        "hashes": hashes,
        "selected_indices_sha256": sha256_json(selected.tolist()),
        "exposure_order_sha256": sha256_json(order.tolist()),
        "source_counts": counts,
        "selection_seed": SELECTION_SEED,
        "order_seed": ORDER_SEED,
        "model_seed": MODEL_SEED,
        "subset_size": SUBSET_SIZE,
        "record_exposures_per_arm": EXPOSURES,
        "batch_size": BATCH_SIZE,
        "learning_rate": LEARNING_RATE,
        "weight_decay": WEIGHT_DECAY,
        "arms": list(ARMS),
        "readout": "512-feature mean/max, fixed C=0.01 train-only logistic, full and limited labels",
    }
    return order, identity, time.monotonic() - started


def _stage_selected(dataset: CachedCPCDataset, order: np.ndarray,
                    *, device: str = "cuda", chunk_size: int = 128,
                    ) -> tuple[torch.Tensor, np.ndarray]:
    """Stage only the frozen 25k rows in exact normalized float32 precision."""
    if chunk_size < 1 or len(order) != EXPOSURES:
        raise ValueError("Invalid staging chunk or exposure count")
    selected = np.unique(order)
    if len(selected) != SUBSET_SIZE or selected[0] < 0 or selected[-1] >= len(dataset):
        raise ValueError("Frozen selected cache rows changed")
    positions = np.full(len(dataset), -1, dtype=np.int32)
    positions[selected] = np.arange(SUBSET_SIZE, dtype=np.int32)
    staged_order = positions[order]
    if np.any(staged_order < 0):
        raise ValueError("Exposure order includes an unstaged cache row")
    shape = (SUBSET_SIZE, *dataset.signals.shape[1:])
    staged = torch.empty(shape, dtype=torch.float32, device=device)
    for start in range(0, SUBSET_SIZE, chunk_size):
        indices = selected[start:start + chunk_size]
        block = np.array(dataset.signals[indices], dtype=np.float32, copy=True)
        block -= dataset.mean
        block /= dataset.std
        staged[start:start + len(indices)] = torch.from_numpy(block).to(device)
    return staged, staged_order


def _staged_batches(staged: torch.Tensor, order: np.ndarray) -> Iterator[torch.Tensor]:
    """Yield the fixed exposure stream from staged device rows."""
    for start in range(0, len(order), BATCH_SIZE):
        indices = torch.as_tensor(order[start:start + BATCH_SIZE],
                                  dtype=torch.long, device=staged.device)
        yield staged[indices]


def _model(name: str) -> torch.nn.Module:
    """Reconstruct matched fresh initialization for one arm."""
    if name not in ARMS:
        raise ValueError(f"Unknown arm: {name}")
    seed_everything(MODEL_SEED)
    return matched_initial_models(MODEL_SEED)[name].cuda()


def _update(model: torch.nn.Module, optimizer: torch.optim.Optimizer,
            signals: torch.Tensor, name: str) -> float:
    """Apply one finite ordinary CPC gradient step."""
    loss, _ = model(signals)
    checked_step(loss, model, optimizer, f"011 {name}")
    return float(loss.detach())


def _ptb_examples() -> tuple[Pool, list[Any], set[str], list[Any]]:
    """Load only clean PTB training and development identities."""
    pool = Pool(PTB_CACHE)
    train, limited = training_examples(pool)
    dev = development_examples(pool, train)
    return pool, train, limited, dev


@torch.inference_mode()
def _features(pool: Pool, examples: list[Any], model: torch.nn.Module) -> np.ndarray:
    """Extract the shared 512-coordinate pooled context feature."""
    norm = json.loads(NORMALIZATION.read_text())
    mean = np.asarray(norm["mean"], dtype=np.float32)
    std = np.asarray(norm["std"], dtype=np.float32)
    data = PoolDataset(pool, [row.cache_row for row in examples], mean, std)
    batches = DataLoader(data, batch_size=BATCH_SIZE, shuffle=False, num_workers=4,
                         pin_memory=True, drop_last=False)
    model.eval()
    chunks = []
    for signals, _, _ in batches:
        _, contexts = model.encoder(signals.cuda(non_blocking=True))
        chunks.append(model.encoder.pooled(contexts).float().cpu().numpy())
    features = np.concatenate(chunks, axis=0)
    if features.shape != (len(examples), 512) or not np.isfinite(features).all():
        raise ValueError("Invalid frozen PTB features")
    return features


def profile(dataset: CachedCPCDataset, order: np.ndarray, identity: dict[str, Any],
            preflight_seconds: float) -> None:
    """Measure all three real update and readout paths before training."""
    output = OUTPUT / "profile_only"
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Profile output already exists")
    output.mkdir(parents=True, exist_ok=True)
    pool, train, _, _ = _ptb_examples()
    results = {}
    with gpu_lock("cuda", blocking=False):
        started = time.monotonic()
        staged, staged_order = _stage_selected(dataset, order)
        torch.cuda.synchronize()
        staging_seconds = time.monotonic() - started
        for name in ARMS:
            model = _model(name).train()
            optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE,
                                          weight_decay=WEIGHT_DECAY)
            torch.cuda.reset_peak_memory_stats()
            started = time.monotonic()
            losses = [_update(model, optimizer, signals, name)
                      for signals in _staged_batches(
                          staged, staged_order[:PROFILE_BATCHES * BATCH_SIZE])]
            torch.cuda.synchronize()
            update_seconds = time.monotonic() - started
            checkpoint = output / f"{name}_profile.pt"
            checkpoint_started = time.monotonic()
            write_torch_atomic(checkpoint, {"model": model.state_dict(),
                                            "optimizer": optimizer.state_dict(),
                                            "updates": len(losses)})
            saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
            if saved["updates"] != len(losses) or any(
                not torch.equal(t.cpu(), saved["model"][key])
                for key, t in model.state_dict().items()
            ):
                raise ValueError(f"{name} profile checkpoint roundtrip failed")
            checkpoint_seconds = time.monotonic() - checkpoint_started
            started = time.monotonic()
            features = _features(pool, train[:PROFILE_READOUT_RECORDS], model)
            torch.cuda.synchronize()
            extraction_seconds = time.monotonic() - started
            results[name] = {
                "profile_updates": len(losses), "last_profile_loss": losses[-1],
                "update_seconds": update_seconds,
                "checkpoint_seconds": checkpoint_seconds,
                "profile_readout_records": len(features),
                "readout_seconds": extraction_seconds,
                "peak_gpu_bytes": torch.cuda.max_memory_allocated(),
                "parameters": sum(p.numel() for p in model.parameters()),
                "checkpoint_sha256": sha256_file(checkpoint),
                "checkpoint_roundtrip": True,
            }
            print(json.dumps({"stage": "profile", "arm": name, **results[name]}), flush=True)
            del model, optimizer
            gc.collect()
    total_updates = math.ceil(EXPOSURES / BATCH_SIZE)
    total_readout_records = len(train) + 1306
    profile_work_seconds = sum(
        value["update_seconds"] + value["checkpoint_seconds"] + value["readout_seconds"]
        for value in results.values()
    )
    projected = (3 * (preflight_seconds + staging_seconds)
                 + profile_work_seconds + sum(
        1.5 * total_updates * value["update_seconds"] / value["profile_updates"]
        + 1.5 * total_readout_records * value["readout_seconds"]
        / value["profile_readout_records"]
        + math.ceil(total_updates / CHECKPOINT_INTERVAL) * value["checkpoint_seconds"]
        for value in results.values()) + 900)
    receipt = {"identity": identity, "arms": results,
               "preflight_seconds_per_stage": preflight_seconds,
               "selected_staging_seconds_per_stage": staging_seconds,
               "staged_bytes": staged.numel() * staged.element_size(),
               "profile_work_seconds": profile_work_seconds,
               "projected_complete_seconds": projected,
               "ceiling_seconds": CEILING_SECONDS,
               "gate_passed": projected <= CEILING_SECONDS,
               "method": (
                   "three full-input hashes and selected-pool staging passes, "
                   "measured profile work, 1.5x all SSL updates "
                   "and PTB extraction, checkpoint writes, 900s CPU/report reserve"
               )}
    write_json_atomic(OUTPUT / "profile.json", receipt)
    print(json.dumps({"stage": "profile_complete", "gate_passed": receipt["gate_passed"],
                      "projected_complete_seconds": projected}), flush=True)


def _check_profile(identity: dict[str, Any]) -> None:
    """Require the exact matching and passed real-device cost gate."""
    receipt = json.loads((OUTPUT / "profile.json").read_text())
    if receipt["identity"] != identity or not receipt["gate_passed"]:
        raise ValueError("Matching passed 011 full-path profile required")


def train(dataset: CachedCPCDataset, order: np.ndarray, identity: dict[str, Any]) -> None:
    """Train each arm with exact batch-boundary recovery and no dev selection."""
    _check_profile(identity)
    total = math.ceil(EXPOSURES / BATCH_SIZE)
    with gpu_lock("cuda", blocking=False):
        staged, staged_order = _stage_selected(dataset, order)
        for name in ARMS:
            arm_dir = OUTPUT / name
            arm_dir.mkdir(parents=True, exist_ok=True)
            complete = arm_dir / "complete.json"
            checkpoint = arm_dir / "latest.pt"
            if complete.exists():
                saved_complete = json.loads(complete.read_text())
                if (saved_complete["identity"] != identity
                        or saved_complete["completed_updates"] != total
                        or saved_complete["record_exposures"] != EXPOSURES
                        or saved_complete["profile_sha256"] != sha256_file(OUTPUT / "profile.json")
                        or saved_complete["checkpoint_sha256"] != sha256_file(checkpoint)):
                    raise ValueError(f"{name} completed artifact identity changed")
                continue
            model = _model(name).train()
            optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE,
                                          weight_decay=WEIGHT_DECAY)
            completed = 0
            loss_sum = 0.0
            if checkpoint.exists():
                saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
                if saved["identity"] != identity or saved["arm"] != name:
                    raise ValueError(f"{name} resume identity changed")
                completed = saved["completed_updates"]
                loss_sum = saved["loss_sum"]
                model.load_state_dict(saved["model"], strict=True)
                optimizer.load_state_dict(saved["optimizer"])
                restore_rng_state(saved["rng"])
            if not 0 <= completed <= total:
                raise ValueError("Invalid saved update count")
            started = time.monotonic()
            for offset, signals in enumerate(
                _staged_batches(staged, staged_order[completed * BATCH_SIZE:]), start=1
            ):
                loss_sum += _update(model, optimizer, signals, name)
                current = completed + offset
                if current % CHECKPOINT_INTERVAL == 0 or current == total:
                    torch.cuda.synchronize()
                    write_torch_atomic(checkpoint, {
                        "identity": identity, "arm": name, "completed_updates": current,
                        "loss_sum": loss_sum, "model": model.state_dict(),
                        "optimizer": optimizer.state_dict(), "rng": capture_rng_state(),
                    })
                    print(json.dumps({"stage": "train", "arm": name,
                                      "completed_updates": current,
                                      "total_updates": total,
                                      "elapsed_seconds": time.monotonic() - started}), flush=True)
            write_json_atomic(complete, {"identity": identity, "arm": name,
                                         "completed_updates": total,
                                         "record_exposures": EXPOSURES,
                                         "mean_cpc_loss": loss_sum / total,
                                         "checkpoint_sha256": sha256_file(checkpoint),
                                         "profile_sha256": sha256_file(OUTPUT / "profile.json")})
            del model, optimizer
            gc.collect()


def readout(identity: dict[str, Any]) -> None:
    """Fit fixed train-only logistic heads and score development only."""
    _check_profile(identity)
    checkpoint_hashes = {}
    for name in ARMS:
        checkpoint = OUTPUT / name / "latest.pt"
        complete = json.loads((OUTPUT / name / "complete.json").read_text())
        checkpoint_hash = sha256_file(checkpoint)
        if (complete["identity"] != identity
                or complete["completed_updates"] != math.ceil(EXPOSURES / BATCH_SIZE)
                or complete["record_exposures"] != EXPOSURES
                or complete["profile_sha256"] != sha256_file(OUTPUT / "profile.json")
                or complete["checkpoint_sha256"] != checkpoint_hash):
            raise ValueError(f"{name} training artifact changed")
        checkpoint_hashes[name] = checkpoint_hash
    result_path = OUTPUT / "result.json"
    if result_path.exists():
        previous = json.loads(result_path.read_text())
        if (previous["status"] != "complete_development_only"
                or previous["identity"] != identity
                or previous["checkpoint_sha256"] != checkpoint_hashes
                or previous["profile_sha256"] != sha256_file(OUTPUT / "profile.json")
                or previous["predictions_sha256"]
                != sha256_file(OUTPUT / "development_predictions.npz")
                or any(previous["feature_sha256"][name]
                       != sha256_file(OUTPUT / f"{name}_features.npy") for name in ARMS)):
            raise ValueError("Existing development result artifact identity changed")
        print(json.dumps({"stage": "readout_already_complete"}), flush=True)
        return
    pool, train, limited, dev = _ptb_examples()
    examples = train + dev
    features: dict[str, np.ndarray] = {}
    with gpu_lock("cuda", blocking=False):
        for name in ARMS:
            checkpoint = OUTPUT / name / "latest.pt"
            model = _model(name).eval()
            saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
            if (saved["identity"] != identity or saved["arm"] != name
                    or saved["completed_updates"] != math.ceil(EXPOSURES / BATCH_SIZE)):
                raise ValueError(f"{name} checkpoint identity changed")
            model.load_state_dict(saved["model"], strict=True)
            features[name] = _features(pool, examples, model)
            np.save(OUTPUT / f"{name}_features.npy", features[name])
    targets = np.asarray([row.target for row in dev], dtype=np.int64)
    patients = np.asarray([row.patient_id for row in dev])
    scores: dict[str, Any] = {}
    predictions: dict[str, dict[str, np.ndarray]] = {}
    with threadpool_limits(limits=1):
        for budget in ("full", "limited"):
            indices = (np.arange(len(train)) if budget == "full" else
                       np.asarray([i for i, row in enumerate(train)
                                   if row.record_id in limited], dtype=np.int64))
            labels = np.asarray([train[i].target for i in indices], dtype=np.int64)
            scores[budget] = {}
            predictions[budget] = {}
            for name in ARMS:
                head = fit_head(features[name][indices].astype(np.float64), labels)
                dev_x = features[name][len(train):].astype(np.float64)
                probability = expit(replay_logits(dev_x, head))
                direct = head["model"].predict_proba(head["scaler"].transform(dev_x))[:, 1]
                if not np.allclose(probability, direct, rtol=0, atol=1e-10):
                    raise ValueError(f"{name} logistic head replay failed")
                predictions[budget][name] = probability
                scores[budget][name] = {
                    "auroc": float(roc_auc_score(targets, probability)),
                    "average_precision": float(average_precision_score(targets, probability)),
                    "classifier_iterations": head["n_iter"],
                    "training_labels": len(indices),
                }
        contrasts = {budget: paired_patient_auc(targets, patients, prediction)
                     for budget, prediction in predictions.items()}
    predictions_path = OUTPUT / "development_predictions.npz"
    np.savez_compressed(predictions_path,
                        record_ids=np.asarray([row.record_id for row in dev]),
                        patient_ids=patients, targets=targets,
                        **{f"{budget}_{name}": values[name]
                           for budget, values in predictions.items() for name in ARMS})
    result = {
        "status": "complete_development_only", "identity": identity,
        "training_labels_full": len(train), "training_labels_limited": len(limited),
        "development_records": len(dev),
        "development_patients": len(set(patients)),
        "scores": scores, "contrasts": contrasts,
        "feature_sha256": {name: sha256_file(OUTPUT / f"{name}_features.npy") for name in ARMS},
        "checkpoint_sha256": checkpoint_hashes,
        "predictions_sha256": sha256_file(predictions_path),
        "profile_sha256": sha256_file(OUTPUT / "profile.json"),
        "calibration_test_evaluated": False,
    }
    write_json_atomic(OUTPUT / "result.json", result)
    print(json.dumps({"stage": "readout_complete", "scores": scores,
                      "contrasts": contrasts}), flush=True)


def main() -> None:
    """Run one explicitly selected verified Experiment 011 stage."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("check", "profile", "train", "readout"), required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    order, identity, preflight_seconds = prepare()
    if args.stage == "check":
        print(json.dumps({"stage": "check", "identity_sha256": sha256_json(identity),
                          "preflight_seconds": preflight_seconds}), flush=True)
        return
    if not torch.cuda.is_available():
        raise RuntimeError("Real V100 profile and training require CUDA")
    dataset = CachedCPCDataset(CACHE, NORMALIZATION, verify_hash=False)
    if len(dataset) != EXPOSURES:
        raise ValueError("Verified 25k source cache row count changed")
    if args.stage == "profile":
        profile(dataset, order, identity, preflight_seconds)
    elif args.stage == "train":
        train(dataset, order, identity)
    else:
        readout(identity)


if __name__ == "__main__":
    main()
