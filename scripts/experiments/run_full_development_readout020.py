"""Experiment 020: full-development readout of the starting CPC encoder with demographic baselines."""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, roc_auc_score
from threadpoolctl import threadpool_limits

from ecg_experiment.cpc import CPCEncoder
from ecg_experiment.cpc_input_audit import evenly_spaced_indices, historical_resample
from ecg_experiment.cpc_pool import Pool
from ecg_experiment.files import sha256_file, write_json_atomic
from ecg_experiment.full_development import (
    PROBE_DEVICE,
    cohorts,
    demographics,
    fit_logistic,
    patient_bootstrap,
    predict,
    ptb_table,
)
from ecg_experiment.gpu import gpu_lock
from ecg_experiment.waveforms import read_record

ROOT = Path(__file__).resolve().parents[2]
POOL_DIR = ROOT / "data/processed/cpc_pool_40k"
BASE = ROOT / "outputs/experiment004_cpc_40k"
PRIOR = ROOT / "outputs/experiment018_cpc_data_scaling_readout_v1"
PTB_RAW = ROOT / "data/raw/ptb-xl/1.0.3"
OUTPUT = ROOT / "outputs/experiment020_full_development_v1"
PROFILE_RECORDS = 512
CEILING_SECONDS = 3_600
PRIOR_AUROC = {"full": 0.9207935251300892, "limited": 0.9129638312209045}
SOURCES = (
    "ecg_experiment/full_development.py", "ecg_experiment/eda/ptbxl.py", "ecg_experiment/cpc.py",
    "ecg_experiment/cpc_input_audit.py", "ecg_experiment/cpc_pool.py", "ecg_experiment/waveforms.py",
    "scripts/experiments/run_full_development_readout020.py", "pyproject.toml", "uv.lock",
    "docs/experiment-020-full-development-readout.md",
)


def official_hashes() -> dict[str, str]:
    """Read the official PTB-XL SHA-256 manifest keyed by release-relative path."""
    lines = (PTB_RAW / "SHA256SUMS.txt").read_text().splitlines()
    return {path: digest for digest, path in (line.split(maxsplit=1) for line in lines)}


def identity(pool: Pool, added: pd.DataFrame) -> dict[str, object]:
    """Hash every input and check the added raw records against the official manifest."""
    paths = {
        "pool_signals": POOL_DIR / "signals.npy", "pool_rows": POOL_DIR / "rows.csv",
        "pool_ids": POOL_DIR / "ecg_ids.npy", "encoder": BASE / "cpc_ssl/encoder.pt",
        "normalization": BASE / "normalization.json", "prior_result": PRIOR / "result.json",
        "prior_predictions": PRIOR / "development_predictions.npz",
    }
    hashes = {name: sha256_file(path) for name, path in paths.items()}
    for key, expected in (("pool_signals", "signals_sha256"), ("pool_rows", "rows_sha256"),
                          ("pool_ids", "ecg_ids_sha256")):
        if hashes[key] != pool.metadata[expected]:
            raise ValueError(f"Historical PTB cache changed: {key}")
    official = official_hashes()
    for stem in added["filename_hr"]:
        for suffix in (".hea", ".dat"):
            if sha256_file(PTB_RAW / f"{stem}{suffix}") != official[f"{stem}{suffix}"]:
                raise ValueError(f"PTB-XL raw file differs from the official manifest: {stem}{suffix}")
    sources = {name: sha256_file(ROOT / name) for name in SOURCES}
    return {**hashes, "sources": sources, "added_raw_records_verified": len(added)}


def raw_signal(stem: str) -> np.ndarray:
    """Convert one raw 500 Hz PTB-XL record with the historical CPC transform."""
    return historical_resample(read_record(PTB_RAW, stem))


def check_transform(pool: Pool, train: pd.DataFrame, original: pd.DataFrame) -> int:
    """Require the historical transform to reproduce 32 cached rows bit for bit."""
    checked = 0
    for frame in (train, original):
        for position in evenly_spaced_indices(len(frame), 16):
            row = frame.iloc[position]
            cached = pool.signals[pool.index[str(row["ecg_id"])]]
            if not np.array_equal(raw_signal(row["filename_hr"]), cached):
                raise ValueError(f"Historical transform does not reproduce {row.name}")
            checked += 1
    return checked


def signals(pool: Pool, frame: pd.DataFrame) -> np.ndarray:
    """Read cached 250 Hz signals where available and convert the rest from the raw files."""
    rows = []
    for row in frame.itertuples():
        index = pool.index.get(str(row.ecg_id))
        cached = index is not None and pool.rows[index]["source"] == "ptbxl"
        rows.append(np.array(pool.signals[index]) if cached else raw_signal(row.filename_hr))
    return np.stack(rows)


def load_encoder() -> CPCEncoder:
    """Load the unchanged starting CPC encoder with no trainable parameters."""
    saved = torch.load(BASE / "cpc_ssl/encoder.pt", map_location="cpu", weights_only=True)
    model = CPCEncoder().cuda().eval()
    model.load_state_dict(saved["encoder"], strict=True)
    model.requires_grad_(False)
    return model


@torch.inference_mode()
def extract(model: CPCEncoder, batch: np.ndarray) -> np.ndarray:
    """Normalize with the historical statistics and return pooled 512 features."""
    normalization = json.loads((BASE / "normalization.json").read_text())
    mean = np.asarray(normalization["mean"], dtype=np.float32)[:, None]
    std = np.asarray(normalization["std"], dtype=np.float32)[:, None]
    chunks = []
    for start in range(0, len(batch), 128):
        normalized = torch.from_numpy((batch[start:start + 128] - mean) / std).cuda()
        _, contexts = model(normalized)
        chunks.append(CPCEncoder.pooled(contexts).float().cpu().numpy())
    features = np.concatenate(chunks)
    if features.shape != (len(batch), 512) or not np.isfinite(features).all():
        raise ValueError("Invalid CPC features")
    return features


def score(y: np.ndarray, probabilities: np.ndarray) -> dict[str, float | int]:
    """AUROC, average precision and class counts."""
    return {"auroc": float(roc_auc_score(y, probabilities)),
            "average_precision": float(average_precision_score(y, probabilities)),
            "records": len(y), "positives": int(y.sum())}


def integrity(dev: pd.DataFrame, probabilities: dict[str, np.ndarray]) -> dict[str, float]:
    """Reproduce Experiment 018's saved initial-encoder predictions on the original rows."""
    original = dev[dev["original"]]
    with np.load(PRIOR / "development_predictions.npz") as saved:
        order = pd.Index(original.index).get_indexer(saved["record_ids"])
        if (order < 0).any() or len(order) != len(original):
            raise ValueError("Original development rows differ from Experiment 018")
        result = {}
        for budget in ("full", "limited"):
            mine = probabilities[budget][dev["original"].to_numpy()][order]
            difference = float(np.abs(mine - saved[f"{budget}_initial"]).max())
            auroc = roc_auc_score(saved["targets"], mine)
            if difference > 1e-6 or abs(auroc - PRIOR_AUROC[budget]) > 1e-6:
                raise ValueError(f"Integrity check failed for {budget}: {difference}, {auroc}")
            result[f"{budget}_max_abs_difference"] = difference
    return result


def heads(train: pd.DataFrame, dev: pd.DataFrame, features: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Fit every prespecified head on training rows and score all development rows."""
    median_age = float(train["age"].median())
    inputs = {
        "cpc": (features["train"], features["development"]),
        "age_sex": (demographics(train, median_age), demographics(dev, median_age)),
    }
    inputs["cpc_age_sex"] = tuple(np.hstack([cpc, demo]) for cpc, demo in zip(
        inputs["cpc"], inputs["age_sex"], strict=True))
    project = train["target"].notna().to_numpy()
    limited = train["limited"].to_numpy()
    standard = train["standard"].notna().to_numpy()
    plans = {
        "cpc_project_full": ("cpc", project, "target"),
        "cpc_project_limited": ("cpc", limited, "target"),
        "age_sex_project": ("age_sex", project, "target"),
        "cpc_standard": ("cpc", standard, "standard"),
        "age_sex_standard": ("age_sex", standard, "standard"),
        "cpc_age_sex_standard": ("cpc_age_sex", standard, "standard"),
    }
    predictions = {}
    for name, (kind, mask, label) in plans.items():
        x_train, x_dev = inputs[kind]
        head = fit_logistic(x_train[mask], train.loc[mask, label].to_numpy(dtype=np.int64))
        predictions[name] = predict(head, x_dev)
    is_device = (train["device"] == PROBE_DEVICE).to_numpy(dtype=np.int64)
    predictions["device_probe"] = predict(fit_logistic(features["train"], is_device), features["development"])
    return predictions


def analyses(dev: pd.DataFrame, predictions: dict[str, np.ndarray]) -> dict[str, object]:
    """Compute the prespecified scores, contrasts and device breakdown."""
    defined = dev["standard"].notna().to_numpy()
    frame = dev[defined]
    y = frame["standard"].to_numpy(dtype=np.int64)
    patients = frame["patient_id"].to_numpy()
    by_name = {name: values[defined] for name, values in predictions.items()}
    subsets = {"full": np.ones(len(frame), bool), "original": frame["original"].to_numpy(),
               "added": ~frame["original"].to_numpy()}
    standard_scores = {
        name: {subset: score(y[mask], by_name[name][mask]) for subset, mask in subsets.items()}
        for name in ("cpc_project_full", "cpc_standard", "age_sex_standard", "cpc_age_sex_standard")
    }
    contrasts = {
        "cpc_standard_minus_age_sex_standard": patient_bootstrap(
            patients, y, by_name["cpc_standard"], by_name["age_sex_standard"]),
        "cpc_age_sex_minus_cpc_standard": patient_bootstrap(
            patients, y, by_name["cpc_age_sex_standard"], by_name["cpc_standard"]),
        "cpc_standard_minus_cpc_project_full": patient_bootstrap(
            patients, y, by_name["cpc_standard"], by_name["cpc_project_full"]),
    }
    devices = {}
    for device, rows in frame.groupby("device"):
        mask = frame.index.isin(rows.index)
        counts = np.bincount(y[mask], minlength=2)
        if mask.sum() >= 50 and counts.min() >= 10:
            devices[device] = score(y[mask], by_name["cpc_standard"][mask])
    probe_truth = (dev["device"] == PROBE_DEVICE).to_numpy(dtype=np.int64)
    sinus = frame["sinus_rate"].to_numpy() & (y == 0) & ~frame["original"].to_numpy()
    return {
        "standard_label": standard_scores,
        "contrasts": contrasts,
        "devices_cpc_standard": devices,
        "device_probe": score(probe_truth, predictions["device_probe"]),
        "sinus_rate_negatives_added": int(sinus.sum()),
        "cpc_project_full_without_sinus_rate_negatives": score(
            y[~sinus], by_name["cpc_project_full"][~sinus]),
        "undefined_standard_label": int((~defined).sum()),
    }


def profile(pool: Pool, groups: dict[str, pd.DataFrame], run_identity: dict[str, object],
            preflight_seconds: float) -> None:
    """Time real extraction on training ECGs and gate the full run."""
    batch = signals(pool, groups["train"].iloc[:PROFILE_RECORDS])
    with gpu_lock("cuda", blocking=False):
        model = load_encoder()
        start = time.monotonic()
        extract(model, batch)
        torch.cuda.synchronize()
        seconds = time.monotonic() - start
    total = len(groups["train"]) + len(groups["development"])
    projected = 2 * preflight_seconds + 1.5 * math.ceil(total / PROFILE_RECORDS) * seconds + 900
    receipt = {"identity": run_identity, "profile_seconds": seconds, "preflight_seconds": preflight_seconds,
               "projected_total_seconds": projected, "ceiling_seconds": CEILING_SECONDS,
               "gate_passed": projected <= CEILING_SECONDS}
    OUTPUT.mkdir(parents=True, exist_ok=True)
    write_json_atomic(OUTPUT / "profile.json", receipt)
    print(json.dumps({"stage": "profile", "gate_passed": receipt["gate_passed"],
                      "projected_total_seconds": projected}), flush=True)


def run(pool: Pool, groups: dict[str, pd.DataFrame], run_identity: dict[str, object]) -> None:
    """Extract features, fit the heads and write the development-only result."""
    receipt = json.loads((OUTPUT / "profile.json").read_text())
    if receipt["identity"] != run_identity or not receipt["gate_passed"]:
        raise ValueError("A matching passed profile is required")
    train, dev = groups["train"], groups["development"]
    with gpu_lock("cuda", blocking=False):
        model = load_encoder()
        start = time.monotonic()
        features = {name: extract(model, signals(pool, frame)) for name, frame in
                    (("train", train), ("development", dev))}
        seconds = time.monotonic() - start
    np.savez_compressed(OUTPUT / "features.npz", **features)
    with threadpool_limits(limits=1):
        predictions = heads(train, dev, features)
        checks = integrity(dev, {"full": predictions["cpc_project_full"],
                                 "limited": predictions["cpc_project_limited"]})
        results = analyses(dev, predictions)
    np.savez_compressed(OUTPUT / "development_predictions.npz", record_ids=dev.index.to_numpy(),
                        patient_ids=dev["patient_id"].to_numpy(), **predictions)
    result = {
        "status": "complete_development_only", "identity": run_identity, "integrity": checks,
        "extraction_seconds": seconds, "training_records": len(train),
        "development_records": len(dev), "development_patients": int(dev["patient_id"].nunique()),
        "added_records": int((~dev["original"]).sum()), "closed_calibration_records": len(groups["closed"]),
        "features_sha256": sha256_file(OUTPUT / "features.npz"),
        "predictions_sha256": sha256_file(OUTPUT / "development_predictions.npz"),
        "calibration_test_evaluated": False, **results,
    }
    write_json_atomic(OUTPUT / "result.json", result)
    print(json.dumps({"stage": "complete", "integrity": checks}), flush=True)


def main() -> None:
    """Profile or run the frozen development-only readout."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("profile", "run"), required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    pool = Pool(POOL_DIR)
    groups = cohorts(ptb_table())
    start = time.monotonic()
    dev = groups["development"]
    run_identity = identity(pool, dev[~dev["original"]])
    run_identity["transform_rows_checked"] = check_transform(
        pool, groups["train"], dev[dev["original"]])
    preflight_seconds = time.monotonic() - start
    if args.stage == "profile":
        profile(pool, groups, run_identity, preflight_seconds)
    else:
        run(pool, groups, run_identity)


if __name__ == "__main__":
    main()
