"""Frozen PTB development readout for the 25k CPC continuation."""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

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
    Example,
    development_examples,
    paired_intervals,
    training_examples,
)
from ecg_experiment.files import sha256_file, write_json_atomic
from ecg_experiment.gpu import gpu_lock

ROOT = Path(__file__).resolve().parents[2]
POOL_DIR = ROOT / "data/processed/cpc_pool_40k"
TRAINING = ROOT / "outputs/experiment019_cpc_25k_v1"
PRIOR = ROOT / "outputs/experiment018_cpc_data_scaling_readout_v1"
OUTPUT = ROOT / "outputs/experiment019_cpc_25k_readout_v1"
PROFILE_RECORDS = 512
CEILING_SECONDS = 7_200


def identity(pool: Pool) -> dict[str, str | float | int]:
    """Hash the historical readout, new checkpoint and executable sources."""
    paths = {
        "pool_complete": POOL_DIR / "complete.json",
        "pool_signals": POOL_DIR / "signals.npy",
        "pool_rows": POOL_DIR / "rows.csv",
        "pool_ids": POOL_DIR / "ecg_ids.npy",
        "normalization": BASE / "normalization.json",
        "training_completion": TRAINING / "complete.json",
        "training_checkpoint": TRAINING / "latest.pt",
        "training_audit": TRAINING / "training_audit.json",
        "prior_result": PRIOR / "result.json",
        "prior_predictions": PRIOR / "development_predictions.npz",
        "model_source": ROOT / "ecg_experiment/cpc.py",
        "readout_source": ROOT / "ecg_experiment/cpc_scaling_readout.py",
        "head_source": ROOT / "ecg_experiment/cpc_local_readout.py",
        "runner_source": Path(__file__),
        "project": ROOT / "pyproject.toml",
        "lock": ROOT / "uv.lock",
    }
    hashes = {name: sha256_file(path) for name, path in paths.items()}
    for filename, key, expected in (
        ("signals.npy", "pool_signals", "signals_sha256"),
        ("rows.csv", "pool_rows", "rows_sha256"),
        ("ecg_ids.npy", "pool_ids", "ecg_ids_sha256"),
    ):
        if hashes[key] != pool.metadata[expected]:
            raise ValueError(f"Historical PTB cache changed: {filename}")
    completion = json.loads((TRAINING / "complete.json").read_text())
    audit = json.loads((TRAINING / "training_audit.json").read_text())
    prior = json.loads((PRIOR / "result.json").read_text())
    if (completion["checkpoint_sha256"] != hashes["training_checkpoint"]
            or audit["checkpoint_sha256"] != hashes["training_checkpoint"]
            or audit["status"] != "passed_training_only"):
        raise ValueError("25k checkpoint does not match its training audit")
    if (prior["development_predictions_sha256"] != hashes["prior_predictions"]
            or prior["status"] != "complete_development_only"):
        raise ValueError("Prior frozen predictions changed")
    return {**hashes, "classifier_C": 0.01, "bootstrap_seed": 18045,
            "full_labels": 15359, "limited_labels": 1518,
            "development_records": 1306}


def load_encoder() -> CPCEncoder:
    """Load only the audited 25k encoder in evaluation mode."""
    saved = torch.load(TRAINING / "latest.pt", map_location="cpu", weights_only=False)
    state = {key.removeprefix("encoder."): value for key, value in saved["model"].items()
             if key.startswith("encoder.")}
    model = CPCEncoder().cuda().eval()
    model.load_state_dict(state, strict=True)
    model.requires_grad_(False)
    return model


@torch.inference_mode()
def extract(pool: Pool, examples: list[Example], model: CPCEncoder) -> np.ndarray:
    """Use Experiment 018's exact normalized 512-feature operation."""
    normalization = json.loads((BASE / "normalization.json").read_text())
    mean = np.asarray(normalization["mean"], dtype=np.float32)
    std = np.asarray(normalization["std"], dtype=np.float32)
    dataset = PoolDataset(pool, [example.cache_row for example in examples], mean, std)
    loader = DataLoader(dataset, batch_size=128, shuffle=False, num_workers=4,
                        pin_memory=True, drop_last=False)
    chunks = []
    for signals, _, _ in loader:
        _, contexts = model(signals.cuda(non_blocking=True))
        chunks.append(CPCEncoder.pooled(contexts).float().cpu().numpy())
    features = np.concatenate(chunks, axis=0)
    if features.shape != (len(examples), 512) or not np.isfinite(features).all():
        raise ValueError("Invalid frozen 25k features")
    return features


def prior_predictions(dev: list[Example]) -> dict[str, dict[str, np.ndarray]]:
    """Match the saved 115k and initial predictions to exact development rows."""
    with np.load(PRIOR / "development_predictions.npz") as saved:
        if (not np.array_equal(saved["record_ids"], [row.record_id for row in dev])
                or not np.array_equal(saved["patient_ids"], [row.patient_id for row in dev])
                or not np.array_equal(saved["targets"], [row.target for row in dev])):
            raise ValueError("Prior development order or targets changed")
        return {budget: {"old": saved[f"{budget}_new"].copy(),
                         "initial": saved[f"{budget}_initial"].copy()}
                for budget in ("full", "limited")}


def profile(pool: Pool, run_identity: dict[str, str | float | int],
            preflight_seconds: float) -> None:
    """Gate full development extraction using training-only real V100 work."""
    train, _ = training_examples(pool)
    with gpu_lock("cuda", blocking=False):
        model = load_encoder()
        start = time.monotonic()
        features = extract(pool, train[:PROFILE_RECORDS], model)
        torch.cuda.synchronize()
        seconds = time.monotonic() - start
    projected = (2 * preflight_seconds + seconds
                 + 1.5 * math.ceil((len(train) + 1306) / PROFILE_RECORDS) * seconds + 900)
    receipt = {
        "identity": run_identity,
        "sampled_training_records": PROFILE_RECORDS,
        "feature_shape": list(features.shape),
        "preflight_seconds_per_launch": preflight_seconds,
        "profile_seconds": seconds,
        "projected_total_seconds": projected,
        "ceiling_seconds": CEILING_SECONDS,
        "gate_passed": projected <= CEILING_SECONDS,
        "method": "two full PTB cache preflights, 1.5x extraction, 900s CPU/report reserve",
    }
    OUTPUT.mkdir(parents=True, exist_ok=True)
    write_json_atomic(OUTPUT / "profile.json", receipt)
    print(json.dumps({"stage": "profile", "gate_passed": receipt["gate_passed"],
                      "projected_total_seconds": projected}), flush=True)


def run(pool: Pool, run_identity: dict[str, str | float | int]) -> None:
    """Fit fixed PTB heads and compare saved predictions by patient."""
    receipt = json.loads((OUTPUT / "profile.json").read_text())
    if receipt["identity"] != run_identity or not receipt["gate_passed"]:
        raise ValueError("Matching passed readout profile required")
    train, limited = training_examples(pool)
    dev = development_examples(pool, train)
    prior = prior_predictions(dev)
    with gpu_lock("cuda", blocking=False):
        model = load_encoder()
        start = time.monotonic()
        features = extract(pool, train + dev, model)
        torch.cuda.synchronize()
        seconds = time.monotonic() - start
    OUTPUT.mkdir(parents=True, exist_ok=True)
    features_path = OUTPUT / "features.npy"
    np.save(features_path, features)
    y_dev = np.asarray([row.target for row in dev], dtype=np.int64)
    scores: dict[str, dict[str, float | int]] = {}
    comparisons: dict[str, dict[str, np.ndarray]] = {}
    with threadpool_limits(limits=1):
        for budget in ("full", "limited"):
            indices = (np.arange(len(train)) if budget == "full"
                       else np.asarray([i for i, row in enumerate(train)
                                        if row.record_id in limited], dtype=np.int64))
            y_train = np.asarray([train[index].target for index in indices], dtype=np.int64)
            head = fit_head(features[indices].astype(np.float64), y_train)
            dev_features = features[len(train):].astype(np.float64)
            probabilities = expit(replay_logits(dev_features, head))
            direct = head["model"].predict_proba(head["scaler"].transform(dev_features))[:, 1]
            if not np.allclose(probabilities, direct, rtol=0, atol=1e-10):
                raise ValueError("25k logistic head replay mismatch")
            comparisons[budget] = {**prior[budget], "new": probabilities}
            scores[budget] = {
                "auroc": float(roc_auc_score(y_dev, probabilities)),
                "average_precision": float(average_precision_score(y_dev, probabilities)),
                "classifier_iterations": head["n_iter"],
                "training_labels": len(indices),
            }
        contrasts = paired_intervals(dev, comparisons)
    predictions_path = OUTPUT / "development_predictions.npz"
    np.savez_compressed(predictions_path,
                        record_ids=np.asarray([row.record_id for row in dev]),
                        patient_ids=np.asarray([row.patient_id for row in dev]),
                        targets=y_dev,
                        **{f"{budget}_25k": values["new"]
                           for budget, values in comparisons.items()})
    result = {
        "status": "complete_development_only",
        "identity": run_identity,
        "profile_sha256": sha256_file(OUTPUT / "profile.json"),
        "features_sha256": sha256_file(features_path),
        "predictions_sha256": sha256_file(predictions_path),
        "training_labels_full": len(train),
        "training_labels_limited": len(limited),
        "development_records": len(dev),
        "development_patients": len({row.patient_id for row in dev}),
        "extraction_seconds": seconds,
        "scores_25k": scores,
        "contrasts": {budget: {
            "25k_minus_115k": values["new_minus_old"],
            "25k_minus_initial": values["new_minus_initial"],
        } for budget, values in contrasts.items()},
        "calibration_test_evaluated": False,
    }
    write_json_atomic(OUTPUT / "result.json", result)
    print(json.dumps({"stage": "complete", "scores_25k": scores,
                      "contrasts": result["contrasts"]}), flush=True)


def main() -> None:
    """Profile or complete the frozen 25k development-only readout."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("profile", "run"), required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    pool = Pool(POOL_DIR)
    start = time.monotonic()
    run_identity = identity(pool)
    preflight_seconds = time.monotonic() - start
    if args.stage == "profile":
        profile(pool, run_identity, preflight_seconds)
    else:
        run(pool, run_identity)


if __name__ == "__main__":
    main()
