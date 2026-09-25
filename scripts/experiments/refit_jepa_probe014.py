"""Refit the limited-label JEPA probe when old probe training IDs lack a checksum."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

from ecg_experiment.evaluation import DEVELOPMENT_RECORDS, LIMITED_LABELS, PROBE_C_GRID, partition_validation
from ecg_experiment.files import read_csv, sha256_file, write_json_atomic, write_npz_atomic

ROOT = Path(__file__).resolve().parents[2]
LIMITED = ROOT / "data/processed/ptbxl/features_jepa_multiblock_union_seeds42_43_44"
MANIFEST = ROOT / "data/processed/ptbxl/seed42_fraction0.1"
OUTPUT = ROOT / "outputs/experiment014_jepa_cpc_fusion"
LIMITED_SHAPE = (7931, 768)
MAX_ITER = 3000
SEED = 42


def load_split_features() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int, int]:
    """
    Gather float64 JEPA features and labels for the exact seed-42 training and development rows.

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int, int]
        Training features and labels, development features and labels, and
        the training and development record counts.

    Raises
    ------
    ValueError
        If the cache is malformed or does not cover the fixed cohorts.
    """
    ids = np.load(LIMITED / "ecg_ids.npy")
    features = np.load(LIMITED / "features.npy", mmap_mode="r")
    if features.shape != LIMITED_SHAPE or len(ids) != len(features):
        raise ValueError("Malformed limited JEPA features")
    index = {int(identifier): i for i, identifier in enumerate(ids)}
    if len(index) != len(ids):
        raise ValueError("Duplicate limited JEPA ECG IDs")
    train = read_csv(MANIFEST / "labeled_train.csv")
    development, _ = partition_validation(read_csv(MANIFEST / "validation.csv"))
    if len(train) != LIMITED_LABELS or len(development) != DEVELOPMENT_RECORDS:
        raise ValueError("Fixed label/development cohorts changed")
    train_ids = [int(row["ecg_id"]) for row in train]
    dev_ids = [int(row["ecg_id"]) for row in development]
    if not set(train_ids + dev_ids).issubset(index) or len(set(train_ids)) != len(train):
        raise ValueError("Limited JEPA cache does not cover exact seed-42 labels")
    train_x = np.asarray(features[[index[i] for i in train_ids]], dtype=np.float64)
    dev_x = np.asarray(features[[index[i] for i in dev_ids]], dtype=np.float64)
    train_y = np.array([int(row["target"]) for row in train])
    dev_y = np.array([int(row["target"]) for row in development])
    return train_x, train_y, dev_x, dev_y, len(train), len(development)


def select_probe(train_scaled: np.ndarray, train_y: np.ndarray, dev_scaled: np.ndarray,
                 dev_y: np.ndarray) -> tuple[float, float, LogisticRegression, list[dict[str, float]]]:
    """
    Fit every regularization value and keep the best development AUROC; the first wins ties.

    Parameters
    ----------
    train_scaled : np.ndarray
        Standardized training features.
    train_y : np.ndarray
        Training labels.
    dev_scaled : np.ndarray
        Standardized development features.
    dev_y : np.ndarray
        Development labels.

    Returns
    -------
    tuple[float, float, LogisticRegression, list[dict[str, float]]]
        Best AUROC, its ``C``, its fitted model, and every candidate's score.

    Raises
    ------
    RuntimeError
        If no candidate produced a comparable AUROC.
    """
    choices, best_auc, best_c, best_model = [], -1.0, None, None
    for c in PROBE_C_GRID:
        model = LogisticRegression(C=c, max_iter=MAX_ITER, solver="lbfgs", random_state=SEED)
        model.fit(train_scaled, train_y)
        auc = float(roc_auc_score(dev_y, model.decision_function(dev_scaled)))
        choices.append({"C": c, "development_auroc": auc})
        if auc > best_auc:
            best_auc, best_c, best_model = auc, c, model
    if best_model is None:
        raise RuntimeError("No regularization candidate produced a development AUROC")
    return best_auc, best_c, best_model, choices


def input_fingerprints() -> dict[str, str]:
    """
    Digest the frozen inputs and this script.

    Returns
    -------
    dict[str, str]
        SHA-256 keyed by repository-relative path.
    """
    files = [LIMITED / "features.npy", LIMITED / "ecg_ids.npy", LIMITED / "metadata.json",
             MANIFEST / "labeled_train.csv", MANIFEST / "validation.csv", Path(__file__).resolve()]
    return {str(p.relative_to(ROOT)): sha256_file(p) for p in files}


def main() -> None:
    """
    Fit the matched ten-percent JEPA probe once, or verify the existing one.

    Raises
    ------
    ValueError
        If an existing probe differs from the frozen inputs.
    """
    model_path = OUTPUT / "matched_jepa_ten_percent.npz"
    receipt_path = OUTPUT / "matched_jepa_ten_percent.json"
    fingerprints = input_fingerprints()
    if receipt_path.exists():
        receipt = json.loads(receipt_path.read_text())
        if receipt["inputs_sha256"] != fingerprints or receipt["model_sha256"] != sha256_file(model_path):
            raise ValueError("Existing matched JEPA probe differs from frozen inputs")
        print(json.dumps({"status": "verified_completed", "receipt": str(receipt_path)}), flush=True)
        return
    train_x, train_y, dev_x, dev_y, train_count, development_count = load_split_features()
    scaler = StandardScaler().fit(train_x)
    auc, c, model, choices = select_probe(scaler.transform(train_x), train_y, scaler.transform(dev_x), dev_y)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    write_npz_atomic(model_path, mean=scaler.mean_, scale=scaler.scale_, coefficient=model.coef_,
                     intercept=model.intercept_)
    labeled_sha256 = sha256_file(MANIFEST / "labeled_train.csv")
    receipt: dict[str, Any] = {
        "inputs_sha256": fingerprints, "model_sha256": sha256_file(model_path),
        "source_budget": "seed42_fraction0.1", "exact_labeled_manifest_sha256": labeled_sha256,
        "train_ecg_ids_order_sha256": labeled_sha256,
        "train_count": train_count, "development_count": development_count,
        "policy": "StandardScaler labeled train only; LogisticRegression lbfgs max_iter=3000 seed42; "
                  "six-C development AUROC",
        "C": c, "development_auroc": auc, "choices": choices}
    write_json_atomic(receipt_path, receipt, sort_keys=True)
    print(json.dumps({"status": "complete", "C": c, "development_auroc": auc, "receipt": str(receipt_path)}),
          flush=True)


if __name__ == "__main__":
    main()
