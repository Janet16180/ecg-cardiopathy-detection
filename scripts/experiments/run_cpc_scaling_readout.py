"""Profile and evaluate a fixed frozen readout of CPC data scaling."""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from threadpoolctl import threadpool_limits

from ecg_experiment.cpc_pool import Pool
from ecg_experiment.cpc_scaling_readout import (
    ARMS,
    BASE,
    CLEAN,
    ROOT,
    TRAINING,
    VALIDATION,
    development_examples,
    extract,
    fit_readouts,
    load_encoders,
    paired_intervals,
    training_examples,
)
from ecg_experiment.files import sha256_file, write_json_atomic
from ecg_experiment.gpu import gpu_lock

POOL_DIR = ROOT / "data/processed/cpc_pool_40k"
OUTPUT = ROOT / "outputs/experiment018_cpc_data_scaling_readout_v1"
PROFILE_RECORDS = 512
CEILING_SECONDS = 7200


def input_identity(pool: Pool) -> dict[str, str | float | int]:
    """Hash exact inputs and source code, including the historical cache."""
    paths = {
        "old_pool_complete": POOL_DIR / "complete.json",
        "old_pool_signals": POOL_DIR / "signals.npy",
        "old_pool_rows": POOL_DIR / "rows.csv",
        "old_pool_ids": POOL_DIR / "ecg_ids.npy",
        "historical_normalization": BASE / "normalization.json",
        "initial_encoder": BASE / "cpc_ssl/encoder.pt",
        "old_encoder_checkpoint": TRAINING / "old/latest.pt",
        "new_encoder_checkpoint": TRAINING / "new/latest.pt",
        "training_audit": TRAINING / "training_audit.json",
        "clean_full_labels": CLEAN / "labels_fraction1.csv",
        "clean_limited_labels": CLEAN / "labels_fraction0.1.csv",
        "development_manifest": VALIDATION,
        "encoder_source": ROOT / "ecg_experiment/cpc.py",
        "pool_source": ROOT / "ecg_experiment/cpc_pool.py",
        "head_source": ROOT / "ecg_experiment/cpc_local_readout.py",
        "readout_source": ROOT / "ecg_experiment/cpc_scaling_readout.py",
        "runner_source": Path(__file__),
        "project": ROOT / "pyproject.toml",
        "lock": ROOT / "uv.lock",
    }
    hashes = {name: sha256_file(path) for name, path in paths.items()}
    cache_hashes = (
        ("signals.npy", "signals_sha256", "old_pool_signals"),
        ("rows.csv", "rows_sha256", "old_pool_rows"),
        ("ecg_ids.npy", "ecg_ids_sha256", "old_pool_ids"),
    )
    for filename, metadata_key, identity_key in cache_hashes:
        if hashes[identity_key] != pool.metadata[metadata_key]:
            raise ValueError(f"Historical CPC cache changed: {filename}")
    audit = json.loads((TRAINING / "training_audit.json").read_text())
    if audit["status"] != "passed_training_only":
        raise ValueError("Training artifact audit did not pass")
    for arm in ("old", "new"):
        if hashes[f"{arm}_encoder_checkpoint"] != audit["arms"][arm]["checkpoint_sha256"]:
            raise ValueError(f"{arm} checkpoint changed after audit")
    return {**hashes, "classifier_C": 0.01, "bootstrap_seed": 18045,
            "full_labels": 15359, "limited_labels": 1518, "development_records": 1306}


def profile(pool: Pool, identity: dict[str, str | float | int], preflight_seconds: float) -> None:
    """Measure the actual loader and three V100 encoders on training ECGs."""
    train, _ = training_examples(pool)
    with gpu_lock("cuda", blocking=False):
        models = load_encoders()
        start = time.monotonic()
        features = extract(pool, train[:PROFILE_RECORDS], models)
        torch.cuda.synchronize()
        seconds = time.monotonic() - start
    full_records = len(train) + 1306
    projected = (2 * preflight_seconds + seconds
                 + 1.5 * math.ceil(full_records / PROFILE_RECORDS) * seconds + 900)
    receipt = {
        "identity": identity, "sampled_train_records": PROFILE_RECORDS,
        "feature_shapes": {arm: list(features[arm].shape) for arm in ARMS},
        "preflight_seconds_per_launch": preflight_seconds,
        "profile_seconds": seconds,
        "projected_total_seconds": projected,
        "ceiling_seconds": CEILING_SECONDS,
        "gate_passed": projected <= CEILING_SECONDS,
        "method": "two full-cache preflights, 1.5x complete extraction, 900s fit/bootstrap/report reserve",
    }
    OUTPUT.mkdir(parents=True, exist_ok=True)
    write_json_atomic(OUTPUT / "profile.json", receipt)
    print(json.dumps({"stage": "profile", "seconds": seconds,
                      "projected_total_seconds": projected,
                      "gate_passed": receipt["gate_passed"]}), flush=True)


def run(pool: Pool, identity: dict[str, str | float | int]) -> None:
    """Extract equal-width features and fit fixed development-only heads."""
    receipt = json.loads((OUTPUT / "profile.json").read_text())
    if receipt["identity"] != identity or not receipt["gate_passed"]:
        raise ValueError("Matching passed V100 profile required")
    train, limited = training_examples(pool)
    dev = development_examples(pool, train)
    with gpu_lock("cuda", blocking=False):
        models = load_encoders()
        started = time.monotonic()
        features = extract(pool, train + dev, models)
        torch.cuda.synchronize()
        extraction_seconds = time.monotonic() - started
    OUTPUT.mkdir(parents=True, exist_ok=True)
    feature_hashes = {}
    for arm in ARMS:
        path = OUTPUT / f"{arm}_features.npy"
        np.save(path, features[arm])
        feature_hashes[arm] = sha256_file(path)
    n_train = len(train)
    with threadpool_limits(limits=1):
        scores, predictions = fit_readouts(
            train, dev, limited,
            {arm: values[:n_train] for arm, values in features.items()},
            {arm: values[n_train:] for arm, values in features.items()},
        )
        intervals = paired_intervals(dev, predictions)
    prediction_path = OUTPUT / "development_predictions.npz"
    np.savez_compressed(prediction_path,
                        record_ids=np.asarray([row.record_id for row in dev]),
                        patient_ids=np.asarray([row.patient_id for row in dev]),
                        targets=np.asarray([row.target for row in dev]),
                        **{f"{budget}_{arm}": values for budget, arms in predictions.items()
                           for arm, values in arms.items()})
    result: dict[str, Any] = {
        "status": "complete_development_only",
        "identity": identity,
        "profile_sha256": sha256_file(OUTPUT / "profile.json"),
        "training_labels_full": len(train),
        "training_labels_limited": len(limited),
        "development_records": len(dev),
        "development_patients": len({row.patient_id for row in dev}),
        "extraction_seconds": extraction_seconds,
        "feature_sha256": feature_hashes,
        "development_predictions_sha256": sha256_file(prediction_path),
        "scores": scores,
        "contrasts": intervals,
        "calibration_test_evaluated": False,
    }
    write_json_atomic(OUTPUT / "result.json", result)
    print(json.dumps({"stage": "complete", "scores": scores,
                      "contrasts": intervals}), flush=True)


def main() -> None:
    """Run the frozen profile or development-only readout."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("profile", "run"), required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    if not torch.cuda.is_available():
        parser.error("V100 CUDA device required")
    pool = Pool(POOL_DIR)
    start = time.monotonic()
    identity = input_identity(pool)
    preflight_seconds = time.monotonic() - start
    if args.stage == "profile":
        profile(pool, identity, preflight_seconds)
    else:
        run(pool, identity)


if __name__ == "__main__":
    main()
