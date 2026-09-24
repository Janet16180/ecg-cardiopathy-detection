"""Experiment 014: patient-aligned cached JEPA/ordinary-CPC logit fusion."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold

from ecg_experiment.data import read_manifest
from ecg_experiment.evaluation import metrics, partition_validation, patient_bootstrap, select_threshold
from ecg_experiment.files import sha256_file

ROOT = Path(__file__).resolve().parents[2]
PROTOCOL = ROOT / "docs/experiment-014-fusion.md"
FULL = ROOT / "data/processed/pretrained/ecg-jepa-full-public"
LIMITED = ROOT / "data/processed/ptbxl/features_jepa_multiblock_union_seeds42_43_44"
CPC = ROOT / "outputs/experiment009_cpc_prediction_mismatch/features"
GRID = (0.0, 0.25, 0.5, 0.75, 1.0)


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    try:
        temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def rows_by_id(rows, name):
    index = {}
    for row in rows:
        identifier = int(row["ecg_id"])
        if identifier in index:
            raise ValueError(f"Duplicate ECG ID in {name}: {identifier}")
        index[identifier] = row
    return index


def patient_sets(parts):
    sets = {name: {str(row["patient_id"]) for row in rows} for name, rows in parts.items()}
    for name, values in sets.items():
        for other, other_values in sets.items():
            if name < other and values & other_values:
                raise ValueError(f"Patient overlap between {name} and {other}")


def check_alignment(parts, jepa_index, cpc_rows, budget):
    cpc_index = rows_by_id(cpc_rows, "CPC feature rows")
    if len(jepa_index) != len(set(jepa_index)):
        raise ValueError("Duplicate JEPA feature ECG IDs")
    patient_sets(parts)
    ids_by_partition = {}
    for name, rows in parts.items():
        ids = []
        for row in rows:
            identifier = int(row["ecg_id"])
            if identifier not in jepa_index or identifier not in cpc_index:
                raise ValueError(f"Missing {name} ECG {identifier} in {budget} features")
            cached = cpc_index[identifier]
            expected = "train" if name == "train" else "test" if name == "test" else "validation"
            if str(cached["patient_id"]) != str(row["patient_id"]) or cached["split"] != expected:
                raise ValueError(f"CPC patient/split mismatch for ECG {identifier}")
            if row["target"] not in ("0", "1"):
                raise ValueError(f"Nonbinary {name} label for ECG {identifier}")
            ids.append(identifier)
        if len(set(ids)) != len(ids):
            raise ValueError(f"Duplicate ECG ID in {name}")
        ids_by_partition[name] = ids
    names = list(ids_by_partition)
    for i, left in enumerate(names):
        for right in names[i + 1:]:
            if set(ids_by_partition[left]) & set(ids_by_partition[right]):
                raise ValueError(f"ECG overlap between {left} and {right}")
    return {int(row["ecg_id"]): (i, row) for i, row in enumerate(cpc_rows)}


def linear_logits(x, path):
    with np.load(path) as saved:
        mean, scale = saved["mean"], saved["scale"]
        coef, intercept = saved["coefficient"].reshape(-1), float(saved["intercept"].reshape(-1)[0])
    if x.shape[1] != len(mean) or len(mean) != len(scale) or len(coef) != len(mean):
        raise ValueError(f"Probe dimensions disagree with features: {path}")
    if np.any(scale <= 0) or not np.isfinite(mean).all() or not np.isfinite(coef).all():
        raise ValueError(f"Invalid saved probe: {path}")
    return ((np.asarray(x, dtype=np.float64) - mean) / scale) @ coef + intercept


def normalized_train_logits(logits, train_count):
    train = np.asarray(logits[:train_count], dtype=np.float64)
    mean, std = float(train.mean()), float(train.std(ddof=0))
    if not np.isfinite(mean) or not np.isfinite(std) or std <= 0:
        raise ValueError("Nonfinite or constant labeled-training logits")
    return (np.asarray(logits, dtype=np.float64) - mean) / std, {"mean": mean, "std": std, "training_records": train_count}


def development_folds(y, patients, seed=14042):
    groups = np.asarray([str(value) for value in patients])
    y = np.asarray(y, dtype=int)
    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
    fold = np.full(len(y), -1, dtype=int)
    for index, (training, heldout) in enumerate(splitter.split(np.zeros(len(y)), y, groups)):
        if len(set(y[training])) != 2 or len(set(y[heldout])) != 2:
            raise ValueError("Development fold lacks a class")
        fold[heldout] = index
    if np.any(fold < 0) or any(len(set(fold[groups == patient])) != 1 for patient in set(groups)):
        raise ValueError("Malformed patient-group folds")
    return fold


def dev_score(y, logit, fold):
    y = np.asarray(y, dtype=int)
    logit = np.asarray(logit, dtype=np.float64)
    fold = np.asarray(fold, dtype=int)
    if set(y) != {0, 1} or not np.isfinite(logit).all() or len(fold) != len(y):
        raise ValueError("Development labels or logits invalid")
    pred = np.zeros(len(y), dtype=bool)
    thresholds = {}
    for index in sorted(set(fold)):
        heldout = fold == index
        training = ~heldout
        if set(y[training]) != {0, 1} or set(y[heldout]) != {0, 1}:
            raise ValueError("Cross-fitted fold lacks a class")
        positive = np.sort(logit[training & (y == 1)])
        required = int(np.ceil(0.95 * len(positive) - 1e-12))
        threshold = float(positive[len(positive) - required])
        thresholds[str(index)] = threshold
        pred[heldout] = logit[heldout] >= threshold
    return {"auroc": float(roc_auc_score(y, logit)),
            "sensitivity": float(np.mean(pred[y == 1])),
            "specificity": float(np.mean(~pred[y == 0])),
            "crossfit_thresholds": thresholds}


def select(scores):
    return max(GRID, key=lambda a: (scores[a]["specificity"], scores[a]["auroc"], -abs(a - .5), -a))


def bootstrap_screen(y, je, cp, patients, fold, chosen, draws=300, seed=14042):
    groups = {}
    for i, patient in enumerate(patients):
        groups.setdefault(str(patient), []).append(i)
    group_indices = list(groups.values())
    rng = np.random.default_rng(seed)
    chosen_weights, gains = [], []
    for _ in range(draws):
        sample = np.concatenate([group_indices[i] for i in rng.integers(len(group_indices), size=len(group_indices))])
        try:
            scores = {a: dev_score(y[sample], a * je[sample] + (1 - a) * cp[sample], fold[sample]) for a in GRID}
        except ValueError:
            continue
        chosen_weights.append(select(scores))
        endpoint = max(scores[0.0]["specificity"], scores[1.0]["specificity"])
        gains.append(scores[chosen]["specificity"] - endpoint)
    if not chosen_weights:
        raise ValueError("No valid patient bootstrap draws")
    return {"seed": seed, "valid_draws": len(chosen_weights),
            "selected_weight_counts": {str(a): chosen_weights.count(a) for a in GRID},
            "interior_selection_fraction": float(np.mean(np.isin(chosen_weights, GRID[1:-1]))),
            "point_weight_positive_gain_fraction": float(np.mean(np.asarray(gains) > 0)),
            "point_weight_gain_ci95": [float(v) for v in np.quantile(gains, [.025, .975])]}


def gate(scores, chosen, bootstrap):
    endpoint = max(scores[0.0]["specificity"], scores[1.0]["specificity"])
    best_auc = max(scores[0.0]["auroc"], scores[1.0]["auroc"])
    adjacent = [a for a in GRID[1:-1] if abs(a - chosen) == .25]
    checks = {"interior_weight": chosen in GRID[1:-1],
              "specificity_gain_ge_0.02": scores[chosen]["specificity"] - endpoint >= .02,
              "auroc_loss_le_0.002": scores[chosen]["auroc"] >= best_auc - .002,
              "adjacent_interior_gain": any(scores[a]["specificity"] > endpoint for a in adjacent),
              "bootstrap_interior_ge_half": bootstrap["interior_selection_fraction"] >= .5,
              "bootstrap_gain_fraction_ge_0.75": bootstrap["point_weight_positive_gain_fraction"] >= .75}
    return {"pass": all(checks.values()), "checks": checks,
            "specificity_gain_vs_best_endpoint": scores[chosen]["specificity"] - endpoint,
            "auroc_difference_vs_best_endpoint": scores[chosen]["auroc"] - best_auc}


def verify_feature_hashes(jepa_dir):
    metadata = json.loads((jepa_dir / "metadata.json").read_text())
    cpc_metadata = json.loads((CPC / "metadata.json").read_text())
    if cpc_metadata["sha256"] != {name: sha256_file(CPC / name) for name in ("features.npy", "rows.csv")}:
        raise ValueError("CPC cache checksum differs from extraction receipt")
    if metadata["model"] != "ecg-jepa-multiblock" or metadata["feature_dimension"] != 768:
        raise ValueError("JEPA cache provenance differs")
    return metadata, cpc_metadata


def load_budget(budget):
    manifest = ROOT / "data/processed/ptbxl" / ("seed42_fraction1" if budget == "full" else "seed42_fraction0.1")
    jepa_dir = FULL if budget == "full" else LIMITED
    jepa_probe = ROOT / ("outputs/experiment002_public_labels/ecg-jepa_linear_seed42" if budget == "full" else "outputs/experiment001/ecg-jepa_linear_seed42")
    jepa_model = jepa_probe / "linear_model.npz" if budget == "full" else ROOT / "outputs/experiment014_jepa_cpc_fusion/matched_jepa_ten_percent.npz"
    jepa_config_path = jepa_probe / "config.json" if budget == "full" else ROOT / "outputs/experiment014_jepa_cpc_fusion/matched_jepa_ten_percent.json"
    cpc_probe = ROOT / f"outputs/experiment009_cpc_prediction_mismatch/ordinary_{budget}_seed42"
    jepa_meta, cpc_meta = verify_feature_hashes(jepa_dir)
    # Only training and development labels are loaded before the development gate.
    train = read_manifest(manifest / "labeled_train.csv")
    validation = read_manifest(manifest / "validation.csv")
    development, calibration = partition_validation(validation)
    if budget == "full":
        if sha256_file(manifest / "labeled_train.csv") != jepa_meta["manifest_sha256"]["labeled_train.csv"]:
            raise ValueError("Full JEPA probe label IDs differ from designated budget manifest")
    else:
        union = ROOT / "data/processed/ptbxl/probe_union_seeds42_43_44/labeled_train.csv"
        if sha256_file(union) != jepa_meta["manifest_sha256"]["labeled_train.csv"]:
            raise ValueError("Limited JEPA union extraction manifest changed")
        matched = json.loads(jepa_config_path.read_text())
        if matched["exact_labeled_manifest_sha256"] != sha256_file(manifest / "labeled_train.csv") or matched["model_sha256"] != sha256_file(jepa_model):
            raise ValueError("Matched limited JEPA probe provenance differs")
    if sha256_file(manifest / "validation.csv") != jepa_meta["manifest_sha256"]["validation.csv"]:
        raise ValueError("JEPA validation manifest changed")
    if sha256_file(manifest / "test.csv") != jepa_meta["manifest_sha256"]["test.csv"]:
        raise ValueError("JEPA test manifest changed")
    config = json.loads((cpc_probe / "config.json").read_text())
    if config["fingerprint"]["inputs"]["manifests"][budget]["labeled_train"] != sha256_file(manifest / "labeled_train.csv"):
        raise ValueError("CPC probe label IDs differ from designated budget manifest")
    if config["arm"] != "ordinary" or config["fingerprint"]["budget"] != budget:
        raise ValueError("Wrong CPC probe")
    with (CPC / "rows.csv").open(newline="") as handle:
        cpc_rows = list(csv.DictReader(handle))
    jepa_ids = np.load(jepa_dir / "ecg_ids.npy")
    if len(set(map(int, jepa_ids))) != len(jepa_ids):
        raise ValueError("Duplicate JEPA ECG IDs")
    jepa_index = {int(value): i for i, value in enumerate(jepa_ids)}
    jepa_x = np.load(jepa_dir / "features.npy", mmap_mode="r")
    cpc_x = np.load(CPC / "features.npy", mmap_mode="r")
    if jepa_x.shape != (jepa_meta["record_count"], 768) or len(jepa_ids) != len(jepa_x):
        raise ValueError("Malformed JEPA cache")
    if cpc_x.shape != (19126, 3, 512) or len(cpc_rows) != len(cpc_x):
        raise ValueError("Malformed CPC cache")
    parts = {"train": train, "development": development, "calibration": calibration}
    cpc_index = check_alignment(parts, jepa_index, cpc_rows, budget)
    ordered = train + development
    ji = [jepa_index[int(r["ecg_id"])] for r in ordered]
    ci = [cpc_index[int(r["ecg_id"])][0] for r in ordered]
    jepa_logit = linear_logits(jepa_x[ji], jepa_model)
    cpc_features = np.concatenate((cpc_x[ci, 0], cpc_x[ci, 1]), axis=1)
    cpc_logit = linear_logits(cpc_features, cpc_probe / "linear_model.npz")
    jepa_config = json.loads(jepa_config_path.read_text())
    split = len(train)
    y = np.array([int(r["target"]) for r in development])
    jepa_auc = float(roc_auc_score(y, jepa_logit[split:]))
    cpc_auc = float(roc_auc_score(y, cpc_logit[split:]))
    expected_jepa_auc = jepa_config["best_development_auroc"] if budget == "full" else jepa_config["development_auroc"]
    if abs(jepa_auc - expected_jepa_auc) > 1e-7:
        raise ValueError(f"JEPA saved probe does not reproduce development AUROC: {jepa_auc}")
    if abs(cpc_auc - config["best_development_auroc"]) > 1e-7:
        raise ValueError(f"CPC saved probe does not reproduce development AUROC: {cpc_auc}")
    jepa_norm, js = normalized_train_logits(jepa_logit, split)
    cpc_norm, cs = normalized_train_logits(cpc_logit, split)
    files = [PROTOCOL, Path(__file__), jepa_dir / "features.npy", jepa_dir / "ecg_ids.npy", jepa_dir / "metadata.json",
             CPC / "features.npy", CPC / "rows.csv", CPC / "metadata.json",
             jepa_model, jepa_config_path,
             cpc_probe / "linear_model.npz", cpc_probe / "selection.json", cpc_probe / "config.json",
             manifest / "labeled_train.csv", manifest / "validation.csv", manifest / "test.csv"]
    if budget == "ten_percent":
        files.append(ROOT / "data/processed/ptbxl/probe_union_seeds42_43_44/labeled_train.csv")
    return {"budget": budget, "manifest": manifest, "jepa_dir": jepa_dir, "jepa_probe": jepa_probe, "jepa_model": jepa_model,
            "cpc_probe": cpc_probe, "train": train, "development": development, "calibration": calibration,
            "jepa_index": jepa_index, "cpc_rows": cpc_rows, "jepa_x": jepa_x, "cpc_x": cpc_x,
            "jepa_dev": jepa_norm[split:], "cpc_dev": cpc_norm[split:], "jepa_stats": js, "cpc_stats": cs,
            "development_y": y, "hashes": {str(path.relative_to(ROOT)): sha256_file(path) for path in files},
            "reproduced_probe_dev_auroc": {"jepa": jepa_auc, "cpc": cpc_auc},
            "records": {"train": len(train), "development": len(development), "calibration": len(calibration)}}


def evaluate_if_pass(data, chosen):
    # Opening test rows and logits occurs only after the development gate passes.
    test = read_manifest(data["manifest"] / "test.csv")
    parts = {"train": data["train"], "development": data["development"],
             "calibration": data["calibration"], "test": test}
    cpc_index = check_alignment(parts, data["jepa_index"], data["cpc_rows"], data["budget"])
    rows = data["calibration"] + test
    ji = [data["jepa_index"][int(r["ecg_id"])] for r in rows]
    ci = [cpc_index[int(r["ecg_id"])][0] for r in rows]
    je = linear_logits(data["jepa_x"][ji], data["jepa_model"])
    cp_x = np.concatenate((data["cpc_x"][ci, 0], data["cpc_x"][ci, 1]), axis=1)
    cp = linear_logits(cp_x, data["cpc_probe"] / "linear_model.npz")
    je = (je - data["jepa_stats"]["mean"]) / data["jepa_stats"]["std"]
    cp = (cp - data["cpc_stats"]["mean"]) / data["cpc_stats"]["std"]
    fused = chosen * je + (1 - chosen) * cp
    ncal = len(data["calibration"])
    cal_y = np.array([int(r["target"]) for r in data["calibration"]])
    test_y = np.array([int(r["target"]) for r in test])
    calibrator = LogisticRegression(C=1e6, solver="lbfgs", max_iter=1000)
    calibrator.fit(fused[:ncal, None], cal_y)
    slope = float(calibrator.coef_[0, 0])
    if slope <= 0:
        raise ValueError("Nonpositive calibration slope")
    cal_p = calibrator.predict_proba(fused[:ncal, None])[:, 1]
    test_p = calibrator.predict_proba(fused[ncal:, None])[:, 1]
    threshold = select_threshold(cal_y, cal_p, .95)
    return {"records": {"calibration": ncal, "test": len(test)},
            "calibration": {"method": "Platt logistic", "slope": slope,
                            "intercept": float(calibrator.intercept_[0]), "threshold": threshold,
                            "sensitivity": metrics(cal_y, cal_p, threshold)["sensitivity"]},
            "test": metrics(test_y, test_p, threshold),
            "test_ci95_patient_bootstrap": patient_bootstrap(test_y, test_p,
                 [r["patient_id"] for r in test], threshold, repeats=500, seed=2026)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs/experiment014_jepa_cpc_fusion")
    args = parser.parse_args()
    start = time.monotonic()
    stamp = lambda: datetime.now(timezone.utc).isoformat()
    atomic_json(args.output_dir / "coordination.json", {"state": "running", "pid": os.getpid(),
                "started_at_utc": stamp(), "command": " ".join(sys.argv), "returncode": None})
    receipts = {}
    for budget in ("full", "ten_percent"):
        data = load_budget(budget)
        output = args.output_dir / f"{budget}.json"
        fingerprint = {"input_sha256": data["hashes"], "budget": budget,
                       "normalization": "labeled-training logits only", "alpha_grid": list(GRID)}
        if output.exists():
            saved = json.loads(output.read_text())
            if saved["fingerprint"] != fingerprint or saved.get("completed") is not True:
                raise ValueError(f"Existing {budget} receipt fingerprint differs")
            print(json.dumps({"budget": budget, "status": "verified_completed", "sha256": sha256_file(output)}), flush=True)
            receipts[budget] = {"path": str(output), "sha256": sha256_file(output)}
            continue
        y = data["development_y"]
        je, cp = data["jepa_dev"], data["cpc_dev"]
        fold = development_folds(y, [r["patient_id"] for r in data["development"]])
        fold_hash = hashlib.sha256(np.asarray(fold, dtype=np.int8).tobytes()).hexdigest()
        scores = {a: dev_score(y, a * je + (1 - a) * cp, fold) for a in GRID}
        chosen = select(scores)
        boot = bootstrap_screen(y, je, cp, [r["patient_id"] for r in data["development"]], fold, chosen)
        decision = gate(scores, chosen, boot)
        result = {"completed": True, "fingerprint": fingerprint, "budget": budget,
                  "model_description": "released ECG-JEPA frozen linear probe plus Experiment 009 frozen ordinary local/context CPC probe",
                  "records": data["records"], "train_logit_normalization": {"jepa": data["jepa_stats"], "cpc": data["cpc_stats"]},
                  "reproduced_probe_dev_auroc": data["reproduced_probe_dev_auroc"],
                  "development": {"grid": {str(a): scores[a] for a in GRID}, "selected_alpha": chosen,
                                  "folds": 5, "fold_seed": 14042, "fold_assignment_sha256": fold_hash,
                                  "patient_bootstrap": boot, "gate": decision},
                  "calibration_test_opened": decision["pass"]}
        if decision["pass"]:
            result["evaluation"] = evaluate_if_pass(data, chosen)
        atomic_json(output, result)
        receipts[budget] = {"path": str(output), "sha256": sha256_file(output)}
        print(json.dumps({"budget": budget, "gate": decision, "selected_alpha": chosen,
                          "seconds_elapsed": time.monotonic() - start, "sha256": sha256_file(output)}), flush=True)
    atomic_json(args.output_dir / "coordination.json", {"state": "complete", "pid": os.getpid(),
                "completed_at_utc": stamp(), "command": " ".join(sys.argv), "returncode": 0,
                "elapsed_seconds": time.monotonic() - start, "receipts": receipts})


if __name__ == "__main__":
    main()
