"""Refit the limited-label JEPA probe when old probe training IDs lack a checksum."""

import json
import os
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

from ecg_experiment.data import read_manifest
from ecg_experiment.evaluation import partition_validation
from ecg_experiment.files import sha256_file, write_json_atomic


ROOT = Path(__file__).resolve().parents[2]
LIMITED = ROOT / "data/processed/ptbxl/features_jepa_multiblock_union_seeds42_43_44"


def main():
    manifest = ROOT / "data/processed/ptbxl/seed42_fraction0.1"
    output = ROOT / "outputs/experiment014_jepa_cpc_fusion"
    model_path = output / "matched_jepa_ten_percent.npz"
    receipt_path = output / "matched_jepa_ten_percent.json"
    files = [LIMITED / "features.npy", LIMITED / "ecg_ids.npy", LIMITED / "metadata.json",
             manifest / "labeled_train.csv", manifest / "validation.csv", Path(__file__).resolve()]
    fingerprints = {str(p.relative_to(ROOT)): sha256_file(p) for p in files}
    if receipt_path.exists():
        receipt = json.loads(receipt_path.read_text())
        if receipt["inputs_sha256"] != fingerprints or receipt["model_sha256"] != sha256_file(model_path):
            raise ValueError("Existing matched JEPA probe differs from frozen inputs")
        print(json.dumps({"status": "verified_completed", "receipt": str(receipt_path)}), flush=True)
        return
    ids = np.load(LIMITED / "ecg_ids.npy")
    features = np.load(LIMITED / "features.npy", mmap_mode="r")
    if features.shape != (7931, 768) or len(ids) != len(features):
        raise ValueError("Malformed limited JEPA features")
    index = {int(identifier): i for i, identifier in enumerate(ids)}
    if len(index) != len(ids):
        raise ValueError("Duplicate limited JEPA ECG IDs")
    train = read_manifest(manifest / "labeled_train.csv")
    development, _ = partition_validation(read_manifest(manifest / "validation.csv"))
    if len(train) != 1518 or len(development) != 1306:
        raise ValueError("Fixed label/development cohorts changed")
    train_ids = [int(row["ecg_id"]) for row in train]
    dev_ids = [int(row["ecg_id"]) for row in development]
    if not set(train_ids + dev_ids).issubset(index) or len(set(train_ids)) != len(train):
        raise ValueError("Limited JEPA cache does not cover exact seed-42 labels")
    train_x = np.asarray(features[[index[i] for i in train_ids]], dtype=np.float64)
    dev_x = np.asarray(features[[index[i] for i in dev_ids]], dtype=np.float64)
    train_y = np.array([int(row["target"]) for row in train])
    dev_y = np.array([int(row["target"]) for row in development])
    scaler = StandardScaler().fit(train_x)
    train_scaled = scaler.transform(train_x)
    dev_scaled = scaler.transform(dev_x)
    choices, best = [], None
    for c in (.001, .01, .1, 1., 10., 100.):
        model = LogisticRegression(C=c, max_iter=3000, solver="lbfgs", random_state=42)
        model.fit(train_scaled, train_y)
        auc = float(roc_auc_score(dev_y, model.decision_function(dev_scaled)))
        choices.append({"C": c, "development_auroc": auc})
        if best is None or auc > best[0]:
            best = (auc, c, model)
    output.mkdir(parents=True, exist_ok=True)
    temporary = model_path.with_name(model_path.name + f".tmp.{os.getpid()}")
    with temporary.open("wb") as handle:
        np.savez(handle, mean=scaler.mean_, scale=scaler.scale_,
                 coefficient=best[2].coef_, intercept=best[2].intercept_)
    os.replace(temporary, model_path)
    receipt = {"inputs_sha256": fingerprints, "model_sha256": sha256_file(model_path),
               "source_budget": "seed42_fraction0.1", "exact_labeled_manifest_sha256": sha256_file(manifest / "labeled_train.csv"),
               "train_ecg_ids_order_sha256": sha256_file(manifest / "labeled_train.csv"),
               "train_count": len(train), "development_count": len(development),
               "policy": "StandardScaler labeled train only; LogisticRegression lbfgs max_iter=3000 seed42; six-C development AUROC",
               "C": best[1], "development_auroc": best[0], "choices": choices}
    write_json_atomic(receipt_path, receipt, sort_keys=True)
    print(json.dumps({"status": "complete", "C": best[1], "development_auroc": best[0],
                      "receipt": str(receipt_path)}), flush=True)


if __name__ == "__main__":
    main()
