"""Frozen development-only readout for Experiment 012's three fresh encoders."""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from scipy.special import expit
from sklearn.metrics import average_precision_score, roc_auc_score
from threadpoolctl import threadpool_limits
from torch.utils.data import DataLoader

from ecg_experiment.cpc import CPCEncoder
from ecg_experiment.cpc_local_readout import fit_head, replay_logits
from ecg_experiment.cpc_pool import Pool, PoolDataset
from ecg_experiment.cpc_scaling_readout import (
    BASE,
    CLEAN,
    VALIDATION,
    Example,
    development_examples,
    training_examples,
)
from ecg_experiment.cpc_temporal_hybrid import TemporalHybridEncoder
from ecg_experiment.files import sha256_file, sha256_json, write_json_atomic
from ecg_experiment.gpu import gpu_lock

ROOT = Path(__file__).resolve().parents[2]
POOL_DIR = ROOT / "data/processed/cpc_pool_40k"
TRAINING = ROOT / "outputs/experiment012_temporal_hybrid"
OUTPUT = ROOT / "outputs/experiment012_temporal_hybrid_readout"
ARMS = ("gru", "mixed", "local")
PROFILE_RECORDS = 512
CEILING_SECONDS = 7_200
BOOTSTRAP_SEED = 12045
BOOTSTRAP_DRAWS = 2_000


def identity(pool: Pool) -> dict[str, Any]:
    """Bind the clean readout cache, completed encoders and executable sources."""
    paths = {
        "pool_complete": POOL_DIR / "complete.json",
        "pool_signals": POOL_DIR / "signals.npy",
        "pool_rows": POOL_DIR / "rows.csv",
        "pool_ids": POOL_DIR / "ecg_ids.npy",
        "normalization": BASE / "normalization.json",
        "full_labels": CLEAN / "labels_fraction1.csv",
        "limited_labels": CLEAN / "labels_fraction0.1.csv",
        "heldout_references": VALIDATION,
        "training_profile": TRAINING / "profile.json",
        "model_source": ROOT / "ecg_experiment/cpc_temporal_hybrid.py",
        "cpc_source": ROOT / "ecg_experiment/cpc.py",
        "pool_source": ROOT / "ecg_experiment/cpc_pool.py",
        "readout_source": ROOT / "ecg_experiment/cpc_scaling_readout.py",
        "head_source": ROOT / "ecg_experiment/cpc_local_readout.py",
        "runner_source": Path(__file__),
        "protocol": ROOT / "docs/experiment-012-temporal-hybrid.md",
    }
    for arm in ARMS:
        paths[f"{arm}_completion"] = TRAINING / arm / "complete.json"
        paths[f"{arm}_checkpoint"] = TRAINING / arm / "latest.pt"
    hashes = {name: sha256_file(path) for name, path in paths.items()}
    for filename, key, expected in (
        ("signals.npy", "pool_signals", "signals_sha256"),
        ("rows.csv", "pool_rows", "rows_sha256"),
        ("ecg_ids.npy", "pool_ids", "ecg_ids_sha256"),
    ):
        if hashes[key] != pool.metadata[expected]:
            raise ValueError(f"Historical PTB cache changed: {filename}")
    profile = json.loads((TRAINING / "profile.json").read_text())
    if not profile["gate_passed"]:
        raise ValueError("Experiment 012 training cost gate did not pass")
    for source, key in (("ecg_experiment/cpc_temporal_hybrid.py", "model_source"),
                        ("ecg_experiment/cpc.py", "cpc_source"),
                        ("ecg_experiment/cpc_pool.py", "pool_source"),
                        ("ecg_experiment/cpc_scaling_readout.py", "readout_source"),
                        ("ecg_experiment/cpc_local_readout.py", "head_source"),
                        ("scripts/experiments/run_cpc_temporal_hybrid012_readout.py",
                         "runner_source"),
                        ("docs/experiment-012-temporal-hybrid.md", "protocol")):
        if profile["identity"]["source_sha256"][source] != hashes[key]:
            raise ValueError(f"Training source changed before readout: {source}")
    for arm in ARMS:
        complete = json.loads((TRAINING / arm / "complete.json").read_text())
        if (complete["checkpoint_sha256"] != hashes[f"{arm}_checkpoint"]
                or complete["profile_sha256"] != hashes["training_profile"]
                or complete["identity"] != profile["identity"]
                or complete["completed_updates"] != 902
                or complete["record_exposures"] != 115_359):
            raise ValueError(f"Incomplete or changed training artifact: {arm}")
    return {
        "sha256": hashes,
        "feature_width": 512,
        "classifier": "StandardScaler plus L2 logistic C=0.01 lbfgs max_iter=5000 tol=1e-8 seed=42",
        "bootstrap_seed": BOOTSTRAP_SEED,
        "bootstrap_draws": BOOTSTRAP_DRAWS,
        "full_labels": 15_359,
        "limited_labels": 1_518,
        "development_records": 1_306,
        "development_patients": 1_173,
    }


def load_encoder(arm: str) -> CPCEncoder | TemporalHybridEncoder:
    """Load only one completed arm's encoder and freeze all parameters."""
    if arm not in ARMS:
        raise ValueError("Unknown Experiment 012 arm")
    saved = torch.load(TRAINING / arm / "latest.pt", map_location="cpu", weights_only=False)
    profile = json.loads((TRAINING / "profile.json").read_text())
    if (saved["identity"] != profile["identity"] or saved["arm"] != arm
            or saved["completed_updates"] != 902):
        raise ValueError(f"Experiment 012 checkpoint state mismatch: {arm}")
    state = {key.removeprefix("encoder."): value for key, value in saved["model"].items()
             if key.startswith("encoder.")}
    model = CPCEncoder() if arm == "gru" else TemporalHybridEncoder(local_control=arm == "local")
    model.load_state_dict(state, strict=True)
    return model.cuda().eval().requires_grad_(False)


@torch.inference_mode()
def extract(pool: Pool, examples: list[Example],
            model: CPCEncoder | TemporalHybridEncoder) -> np.ndarray:
    """Extract unchanged normalized, half-averaged mean/max CPC features."""
    normalization = json.loads((BASE / "normalization.json").read_text())
    mean = np.asarray(normalization["mean"], dtype=np.float32)
    std = np.asarray(normalization["std"], dtype=np.float32)
    dataset = PoolDataset(pool, [example.cache_row for example in examples], mean, std)
    batches = DataLoader(dataset, batch_size=128, shuffle=False, num_workers=4,
                         pin_memory=True, drop_last=False)
    chunks = []
    for signals, _, _ in batches:
        _, contexts = model(signals.cuda(non_blocking=True))
        chunks.append(model.pooled(contexts).float().cpu().numpy())
    features = np.concatenate(chunks, axis=0)
    if features.shape != (len(examples), 512) or not np.isfinite(features).all():
        raise ValueError("Invalid frozen Experiment 012 features")
    return features


def profile(pool: Pool, run_identity: dict[str, Any], preflight_seconds: float) -> None:
    """Gate three-arm extraction with real training-only V100 work."""
    if (OUTPUT / "profile.json").exists():
        raise FileExistsError("Experiment 012 readout profile already exists")
    train, _ = training_examples(pool)
    arms = {}
    with gpu_lock("cuda", blocking=False):
        for arm in ARMS:
            model = load_encoder(arm)
            torch.cuda.reset_peak_memory_stats()
            start = time.monotonic()
            features = extract(pool, train[:PROFILE_RECORDS], model)
            torch.cuda.synchronize()
            arms[arm] = {
                "seconds": time.monotonic() - start,
                "feature_shape": list(features.shape),
                "peak_gpu_bytes": torch.cuda.max_memory_allocated(),
            }
            del model
            torch.cuda.empty_cache()
    projected = (2 * preflight_seconds + sum(row["seconds"] for row in arms.values())
                 + 1.5 * math.ceil((len(train) + 1306) / PROFILE_RECORDS)
                 * sum(row["seconds"] for row in arms.values()) + 900)
    result = {
        "identity": run_identity,
        "arms": arms,
        "preflight_seconds_per_launch": preflight_seconds,
        "projected_total_seconds": projected,
        "ceiling_seconds": CEILING_SECONDS,
        "gate_passed": projected <= CEILING_SECONDS,
        "method": "two full PTB hashes, 1.5x three-arm extraction, 900s CPU/report reserve",
    }
    OUTPUT.mkdir(parents=True, exist_ok=True)
    write_json_atomic(OUTPUT / "profile.json", result)
    print(json.dumps({"stage": "profile", "gate_passed": result["gate_passed"],
                      "projected_total_seconds": projected}), flush=True)


def paired_difference(dev: list[Example], mixed: np.ndarray,
                      local: np.ndarray) -> dict[str, Any]:
    """Compute a 2,000-draw paired patient AUROC difference."""
    patients = np.asarray([row.patient_id for row in dev])
    labels = np.asarray([row.target for row in dev], dtype=np.int64)
    unique = np.unique(patients)
    positions = [np.flatnonzero(patients == patient) for patient in unique]
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    draws = []
    invalid = 0
    for _ in range(BOOTSTRAP_DRAWS):
        sampled = rng.integers(len(unique), size=len(unique))
        index = np.concatenate([positions[i] for i in sampled])
        if np.unique(labels[index]).size < 2:
            invalid += 1
            continue
        draws.append(roc_auc_score(labels[index], mixed[index])
                     - roc_auc_score(labels[index], local[index]))
    if not draws:
        raise ValueError("No valid patient bootstrap draws")
    return {
        "difference": float(roc_auc_score(labels, mixed) - roc_auc_score(labels, local)),
        "paired_patient_ci95": np.quantile(draws, [0.025, 0.975]).tolist(),
        "valid_draws": len(draws),
        "invalid_draws": invalid,
    }


def run(pool: Pool, run_identity: dict[str, Any]) -> None:
    """Fit fixed heads and compare mixed versus local on development patients."""
    receipt = json.loads((OUTPUT / "profile.json").read_text())
    if receipt["identity"] != run_identity or not receipt["gate_passed"]:
        raise ValueError("Matching passed Experiment 012 readout gate required")
    if (OUTPUT / "result.json").exists():
        raise FileExistsError("Experiment 012 development result already exists")
    train, limited = training_examples(pool)
    dev = development_examples(pool, train)
    examples = train + dev
    features = {}
    elapsed = {}
    with gpu_lock("cuda", blocking=False):
        for arm in ARMS:
            model = load_encoder(arm)
            start = time.monotonic()
            features[arm] = extract(pool, examples, model)
            torch.cuda.synchronize()
            elapsed[arm] = time.monotonic() - start
            del model
            torch.cuda.empty_cache()
    scores: dict[str, dict[str, Any]] = {}
    predictions: dict[str, dict[str, np.ndarray]] = {}
    y_dev = np.asarray([row.target for row in dev], dtype=np.int64)
    with threadpool_limits(limits=1):
        for budget in ("full", "limited"):
            indices = (np.arange(len(train)) if budget == "full"
                       else np.asarray([i for i, row in enumerate(train)
                                        if row.record_id in limited], dtype=np.int64))
            y_train = np.asarray([train[index].target for index in indices], dtype=np.int64)
            scores[budget] = {}
            predictions[budget] = {}
            for arm in ARMS:
                head = fit_head(features[arm][indices].astype(np.float64), y_train)
                dev_features = features[arm][len(train):].astype(np.float64)
                probabilities = expit(replay_logits(dev_features, head))
                direct = head["model"].predict_proba(head["scaler"].transform(dev_features))[:, 1]
                if not np.allclose(probabilities, direct, rtol=0, atol=1e-10):
                    raise ValueError("Fixed logistic head replay mismatch")
                predictions[budget][arm] = probabilities
                scores[budget][arm] = {
                    "auroc": float(roc_auc_score(y_dev, probabilities)),
                    "average_precision": float(average_precision_score(y_dev, probabilities)),
                    "training_labels": len(indices),
                    "classifier_iterations": head["n_iter"],
                }
        contrasts = {budget: paired_difference(dev, scores["mixed"], scores["local"])
                     for budget, scores in predictions.items()}
    feature_hashes = {}
    for arm in ARMS:
        path = OUTPUT / f"{arm}_features.npy"
        np.save(path, features[arm])
        feature_hashes[arm] = sha256_file(path)
    prediction_path = OUTPUT / "development_predictions.npz"
    np.savez_compressed(prediction_path,
                        record_ids=np.asarray([row.record_id for row in dev]),
                        patient_ids=np.asarray([row.patient_id for row in dev]),
                        targets=y_dev,
                        **{f"{budget}_{arm}": values[arm]
                           for budget, values in predictions.items() for arm in ARMS})
    result = {
        "status": "complete_development_only",
        "identity": run_identity,
        "profile_sha256": sha256_file(OUTPUT / "profile.json"),
        "feature_sha256": feature_hashes,
        "predictions_sha256": sha256_file(prediction_path),
        "scores": scores,
        "mixed_minus_local": contrasts,
        "extraction_seconds": elapsed,
        "development_records": len(dev),
        "development_patients": len({row.patient_id for row in dev}),
        "calibration_test_evaluated": False,
    }
    write_json_atomic(OUTPUT / "result.json", result)
    print(json.dumps({"stage": "complete", "scores": scores,
                      "mixed_minus_local": contrasts}), flush=True)


def main() -> None:
    """Check, profile, or run the locked development-only readout."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("check", "profile", "run"), required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    start = time.monotonic()
    pool = Pool(POOL_DIR)
    run_identity = identity(pool)
    preflight_seconds = time.monotonic() - start
    print(json.dumps({"stage": "check", "identity_sha256": sha256_json(run_identity)}),
          flush=True)
    if args.stage == "check":
        return
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable")
    if args.stage == "profile":
        profile(pool, run_identity, preflight_seconds)
    else:
        run(pool, run_identity)


if __name__ == "__main__":
    main()
