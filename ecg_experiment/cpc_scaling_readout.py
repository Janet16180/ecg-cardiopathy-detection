"""Matched frozen-feature readout for the CPC data-scaling study."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from scipy.special import expit
from sklearn.metrics import average_precision_score, roc_auc_score
from torch.utils.data import DataLoader

from .cpc import CPCEncoder
from .cpc_local_readout import fit_head, replay_logits
from .cpc_pool import Pool, PoolDataset
from .files import read_csv

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "outputs/experiment004_cpc_40k"
TRAINING = ROOT / "outputs/experiment018_cpc_data_scaling_v3"
CLEAN = ROOT / "outputs/data_quality/clean_rerun_preflight_v1"
VALIDATION = CLEAN / "heldout_references.csv"
ARMS = ("initial", "old", "new")


@dataclass(frozen=True)
class Example:
    """One verified PTB cache row and its diagnostic-annotation proxy."""

    record_id: str
    patient_id: str
    target: int
    cache_row: dict[str, str]


def _example(pool: Pool, record_id: str, patient_id: str,
             target: str | int, split: str) -> Example:
    """Join a labeled PTB row to the historical CPC cache by identity."""
    ecg_id = record_id.removeprefix("ptbxl:")
    row = pool.rows[pool.index[ecg_id]]
    if (row["source"] != "ptbxl" or row["split"] != split
            or f"ptbxl:{row['patient_id']}" != patient_id):
        raise ValueError(f"PTB cache identity mismatch: {record_id}")
    label = int(target)
    if label not in (0, 1):
        raise ValueError("Nonbinary PTB target")
    return Example(record_id, patient_id, label, row)


def training_examples(pool: Pool) -> tuple[list[Example], set[str]]:
    """Load clean full labels and verify the fixed limited-label subset."""
    full_rows = read_csv(CLEAN / "labels_fraction1.csv")
    limited_rows = read_csv(CLEAN / "labels_fraction0.1.csv")
    if len(full_rows) != 15359 or len(limited_rows) != 1518:
        raise ValueError("Unexpected clean training label count")
    examples = [_example(pool, row["record_id"], row["patient_id"],
                         row["target"], "train") for row in full_rows]
    by_id = {row.record_id: row for row in examples}
    limited = {row["record_id"] for row in limited_rows}
    if len(by_id) != len(examples) or len(limited) != len(limited_rows) or not limited <= by_id.keys():
        raise ValueError("Duplicate or unaligned training labels")
    for row in limited_rows:
        example = by_id[row["record_id"]]
        if example.patient_id != row["patient_id"] or example.target != int(row["target"]):
            raise ValueError("Limited/full label disagreement")
    examples.sort(key=lambda row: pool.index[row.cache_row["ecg_id"]])
    return examples, limited


def development_examples(pool: Pool, train: list[Example]) -> list[Example]:
    """Select only development references and verify patient separation."""
    rows = [row for row in read_csv(VALIDATION) if row["split"] == "development"]
    if len(rows) != 1306:
        raise ValueError("Unexpected development count")
    examples = [_example(pool, row["record_id"], row["patient_id"],
                         row["target"], "validation")
                for row in rows]
    if len({row.record_id for row in examples}) != len(examples):
        raise ValueError("Duplicate development ECG")
    if {row.patient_id for row in train} & {row.patient_id for row in examples}:
        raise ValueError("Train/development patient overlap")
    if len({row.patient_id for row in examples}) != 1173:
        raise ValueError("Unexpected development patient count")
    examples.sort(key=lambda row: pool.index[row.cache_row["ecg_id"]])
    return examples


def load_encoders() -> dict[str, CPCEncoder]:
    """Load the three frozen encoders with no trainable parameters."""
    initial = torch.load(BASE / "cpc_ssl/encoder.pt", map_location="cpu", weights_only=True)
    states: dict[str, dict[str, torch.Tensor]] = {"initial": initial["encoder"]}
    for arm in ("old", "new"):
        checkpoint = torch.load(TRAINING / arm / "latest.pt", map_location="cpu", weights_only=False)
        states[arm] = {key.removeprefix("encoder."): value
                       for key, value in checkpoint["model"].items()
                       if key.startswith("encoder.")}
    models = {}
    for arm in ARMS:
        model = CPCEncoder().cuda().eval()
        model.load_state_dict(states[arm], strict=True)
        model.requires_grad_(False)
        models[arm] = model
    return models


@torch.inference_mode()
def extract(pool: Pool, examples: list[Example], models: dict[str, CPCEncoder]) -> dict[str, np.ndarray]:
    """Extract equal-width mean/max context features from identical ECGs."""
    norm = json.loads((BASE / "normalization.json").read_text())
    mean = np.asarray(norm["mean"], dtype=np.float32)
    std = np.asarray(norm["std"], dtype=np.float32)
    data = PoolDataset(pool, [row.cache_row for row in examples], mean, std)
    loader = DataLoader(data, batch_size=128, shuffle=False, num_workers=4,
                        pin_memory=True, drop_last=False)
    chunks: dict[str, list[np.ndarray]] = {arm: [] for arm in ARMS}
    for signal, _, _ in loader:
        signal = signal.cuda(non_blocking=True)
        for arm in ARMS:
            _, contexts = models[arm](signal)
            feature = CPCEncoder.pooled(contexts)
            chunks[arm].append(feature.float().cpu().numpy())
    features = {arm: np.concatenate(chunks[arm], axis=0) for arm in ARMS}
    if any(value.shape != (len(examples), 512) or not np.isfinite(value).all()
           for value in features.values()):
        raise ValueError("Malformed or nonfinite CPC readout features")
    return features


def fit_readouts(
    train: list[Example], dev: list[Example], limited: set[str],
    train_features: dict[str, np.ndarray], dev_features: dict[str, np.ndarray],
) -> tuple[dict[str, Any], dict[str, dict[str, np.ndarray]]]:
    """Fit fixed train-only logistic heads and score development patients."""
    results: dict[str, Any] = {}
    predictions: dict[str, dict[str, np.ndarray]] = {}
    y_dev = np.asarray([row.target for row in dev], dtype=np.int64)
    for budget in ("full", "limited"):
        indices = (np.arange(len(train)) if budget == "full"
                   else np.asarray([i for i, row in enumerate(train)
                                    if row.record_id in limited], dtype=np.int64))
        y_train = np.asarray([train[i].target for i in indices], dtype=np.int64)
        results[budget] = {}
        predictions[budget] = {}
        for arm in ARMS:
            head = fit_head(train_features[arm][indices].astype(np.float64), y_train)
            x_dev = dev_features[arm].astype(np.float64)
            logits = replay_logits(x_dev, head)
            probabilities = expit(logits)
            direct = head["model"].predict_proba(head["scaler"].transform(x_dev))[:, 1]
            if not np.allclose(probabilities, direct, rtol=0, atol=1e-10):
                raise ValueError("Saved-head logit replay mismatch")
            predictions[budget][arm] = probabilities
            results[budget][arm] = {
                "auroc": float(roc_auc_score(y_dev, probabilities)),
                "average_precision": float(average_precision_score(y_dev, probabilities)),
                "classifier_iterations": head["n_iter"],
                "training_labels": len(indices),
            }
    return results, predictions


def paired_intervals(
    dev: list[Example], predictions: dict[str, dict[str, np.ndarray]],
) -> dict[str, Any]:
    """Pair AUROC differences by development patient over 2,000 draws."""
    patients = np.asarray([row.patient_id for row in dev])
    unique = np.unique(patients)
    positions = [np.flatnonzero(patients == patient) for patient in unique]
    y = np.asarray([row.target for row in dev], dtype=np.int64)
    rng = np.random.default_rng(18045)
    contrasts = (("new_minus_old", "new", "old"),
                 ("new_minus_initial", "new", "initial"))
    draws = []
    invalid = 0
    for _ in range(2000):
        sampled = rng.integers(len(unique), size=len(unique))
        index = np.concatenate([positions[i] for i in sampled])
        if np.unique(y[index]).size < 2:
            invalid += 1
            continue
        draws.append(index)
    result: dict[str, Any] = {}
    for budget, scores in predictions.items():
        differences: dict[str, list[float]] = {name: [] for name, _, _ in contrasts}
        for index in draws:
            for name, left, right in contrasts:
                differences[name].append(float(
                    roc_auc_score(y[index], scores[left][index])
                    - roc_auc_score(y[index], scores[right][index])
                ))
        result[budget] = {name: {
            "difference": float(roc_auc_score(y, scores[left]) - roc_auc_score(y, scores[right])),
            "paired_patient_ci95": np.quantile(differences[name], [0.025, 0.975]).tolist(),
            "valid_draws": len(draws), "invalid_draws": invalid,
        } for name, left, right in contrasts}
    return result
