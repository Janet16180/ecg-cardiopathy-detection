"""Refit historical cached ECG probes on the verified clean PTB-XL labels."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

from .evaluation import DEVELOPMENT_RECORDS, PROBE_C_GRID, partition_validation
from .files import read_csv, sha256_file, write_json_atomic, write_npz_atomic
from .provenance import utc_now

ROOT = Path(__file__).resolve().parents[1]
PREFLIGHT = ROOT / "outputs/data_quality/clean_rerun_preflight_v1"
CPC_AUDIT = ROOT / "outputs/data_quality/clean_cpc_input_audit_v1/receipt.json"
CPC_CACHE = ROOT / "outputs/experiment009_cpc_prediction_mismatch/features"
PTB = ROOT / "data/processed/ptbxl"
JEPA = {
    "full": ROOT / "data/processed/pretrained/ecg-jepa-full-public",
    "ten_percent": ROOT / "data/processed/ptbxl/features_jepa_multiblock_union_seeds42_43_44",
}
FRACTIONS = {"full": "1", "ten_percent": "0.1"}
COUNTS = {"full": 15359, "ten_percent": 1518}
ORIGINAL_SCORES = {
    "full": {"jepa": 0.9597190943585723, "cpc": 0.936230012631018},
    "ten_percent": {"jepa": 0.9490608722832423, "cpc": 0.9244675372589409},
}


def clean_ecg_id(row: dict[str, str]) -> int:
    """Return the PTB-XL ECG ID from a clean-selection row."""
    source, value = row["record_id"].split(":", 1)
    if source != "ptbxl":
        raise ValueError("The cached probe rerun requires only PTB-XL labels")
    return int(value)


def validate_selection(clean: list[dict[str, str]], original: list[dict[str, str]]) -> list[dict[str, str]]:
    """Require exact label and patient agreement with the historical selection."""
    original_by_id = {int(row["ecg_id"]): row for row in original}
    if len(original_by_id) != len(original):
        raise ValueError("Duplicate original ECG ID")
    ids = [clean_ecg_id(row) for row in clean]
    if len(set(ids)) != len(ids):
        raise ValueError("Duplicate clean ECG ID")
    matched = []
    for row, identifier in zip(clean, ids, strict=True):
        old = original_by_id.get(identifier)
        if old is None or row["patient_id"] != f"ptbxl:{old['patient_id']}" or row["target"] != old["target"]:
            raise ValueError(f"Clean label identity differs for ECG {identifier}")
        matched.append(old)
    return matched


def verify_preflight() -> dict[str, Any]:
    """Verify clean selection files against the immutable preflight receipt."""
    receipt = json.loads((PREFLIGHT / "receipt.json").read_text())
    if (receipt["status"] != "passed_cpu_preflight_no_training"
            or receipt["label_counts"] != {"0.1": 1518, "1": 15359}
            or receipt["heldout_counts"] != {"development": 1306, "calibration": 564, "test": 1896}):
        raise ValueError("Clean preflight has an unexpected cohort or status")
    for name in ("labels_fraction1.csv", "labels_fraction0.1.csv"):
        if sha256_file(PREFLIGHT / name) != receipt["output_sha256"][name]:
            raise ValueError(f"Clean preflight file changed: {name}")
    if not CPC_AUDIT.is_file():
        raise ValueError("The clean CPC input audit is missing")
    return receipt


def input_hashes(budget: str) -> dict[str, str]:
    """Hash the selected manifests, caches, provenance receipts and code."""
    fraction = FRACTIONS[budget]
    manifest = PTB / f"seed42_fraction{fraction}"
    files = [
        PREFLIGHT / "receipt.json", PREFLIGHT / f"labels_fraction{fraction}.csv", CPC_AUDIT,
        ROOT / "docs/clean-cached-probe-rerun-v1.md", Path(__file__),
        ROOT / "scripts/experiments/run_clean_cached_probes.py",
        ROOT / "ecg_experiment/evaluation.py", ROOT / "ecg_experiment/files.py",
        manifest / "labeled_train.csv", manifest / "validation.csv",
        JEPA[budget] / "features.npy", JEPA[budget] / "ecg_ids.npy", JEPA[budget] / "metadata.json",
        CPC_CACHE / "features.npy", CPC_CACHE / "rows.csv", CPC_CACHE / "metadata.json",
        ROOT / "outputs/experiment014_jepa_cpc_fusion" / f"{budget}.json",
    ]
    return {str(path.relative_to(ROOT)): sha256_file(path) for path in files}


def verify_caches(budget: str, digests: dict[str, str]) -> tuple[np.ndarray, dict[int, int], np.ndarray,
                                                              dict[int, int]]:
    """Check cached feature identities against historical extraction evidence."""
    prior = json.loads((ROOT / "outputs/experiment014_jepa_cpc_fusion" / f"{budget}.json").read_text())
    previous = prior["fingerprint"]["input_sha256"]
    for directory, names in ((JEPA[budget], ("features.npy", "ecg_ids.npy", "metadata.json")),
                             (CPC_CACHE, ("features.npy", "rows.csv", "metadata.json"))):
        for name in names:
            key = str((directory / name).relative_to(ROOT))
            if digests[key] != previous[key]:
                raise ValueError(f"Historical feature cache changed: {key}")
    jepa_ids = [int(value) for value in np.load(JEPA[budget] / "ecg_ids.npy")]
    cpc_rows = read_csv(CPC_CACHE / "rows.csv")
    if len(set(jepa_ids)) != len(jepa_ids) or len({int(r["ecg_id"]) for r in cpc_rows}) != len(cpc_rows):
        raise ValueError("Duplicate feature ECG IDs")
    jepa_x = np.load(JEPA[budget] / "features.npy", mmap_mode="r")
    cpc_x = np.load(CPC_CACHE / "features.npy", mmap_mode="r")
    if jepa_x.shape != (len(jepa_ids), 768) or cpc_x.shape != (len(cpc_rows), 3, 512):
        raise ValueError("Historical cache shape changed")
    return jepa_x, dict(zip(jepa_ids, range(len(jepa_ids)), strict=True)), cpc_x, {
        int(row["ecg_id"]): i for i, row in enumerate(cpc_rows)}


def aligned_features(budget: str, train: list[dict[str, str]], development: list[dict[str, str]],
                     digests: dict[str, str]) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Join both cache representations to labeled train and development by ECG ID."""
    jepa_x, jepa_index, cpc_x, cpc_index = verify_caches(budget, digests)
    cpc_rows = read_csv(CPC_CACHE / "rows.csv")
    all_rows = train + development
    if len({row["ecg_id"] for row in all_rows}) != len(all_rows):
        raise ValueError("Training and development ECG IDs overlap")
    if {r["patient_id"] for r in train} & {r["patient_id"] for r in development}:
        raise ValueError("Training and development patients overlap")
    for position, row in enumerate(all_rows):
        identifier = int(row["ecg_id"])
        if identifier not in jepa_index or identifier not in cpc_index:
            raise ValueError(f"Missing cached features for ECG {identifier}")
        cpc_row = cpc_rows[cpc_index[identifier]]
        split = "train" if position < len(train) else "validation"
        if cpc_row["patient_id"] != row["patient_id"] or cpc_row["split"] != split:
            raise ValueError(f"CPC cache patient/split mismatch for ECG {identifier}")
    je = np.asarray(jepa_x[[jepa_index[int(r["ecg_id"])] for r in all_rows]], dtype=np.float64)
    cp_rows = cpc_x[[cpc_index[int(r["ecg_id"])] for r in all_rows]]
    cp = np.asarray(np.concatenate((cp_rows[:, 0], cp_rows[:, 1]), axis=1), dtype=np.float64)
    if not np.isfinite(je).all() or not np.isfinite(cp).all():
        raise ValueError("Cached features contain nonfinite values")
    count = len(train)
    return {"jepa": (je[:count], je[count:]), "cpc": (cp[:count], cp[count:])}


def fit_probe(train_x: np.ndarray, train_y: np.ndarray, dev_x: np.ndarray,
              dev_y: np.ndarray) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Select the historical six-C linear-probe grid on development AUROC."""
    scaler = StandardScaler().fit(train_x)
    scaled_train, scaled_dev = scaler.transform(train_x), scaler.transform(dev_x)
    best_auc, best_c, best_model = -1.0, None, None
    choices = []
    for c in PROBE_C_GRID:
        model = LogisticRegression(C=c, solver="lbfgs", max_iter=3000, random_state=42)
        model.fit(scaled_train, train_y)
        auc = float(roc_auc_score(dev_y, model.decision_function(scaled_dev)))
        choices.append({"C": c, "development_auroc": auc, "iterations": model.n_iter_.tolist()})
        if auc > best_auc:
            best_auc, best_c, best_model = auc, c, model
    if best_model is None:
        raise ValueError("No probe candidate fitted")
    arrays = {"mean": scaler.mean_, "scale": scaler.scale_, "coefficient": best_model.coef_,
              "intercept": best_model.intercept_}
    return arrays, {"C": best_c, "development_auroc": best_auc, "candidates": choices,
                    "tie_rule": "first C in ascending grid", "selection_split": "development only"}


def run_budget(budget: str, output: Path) -> dict[str, Any]:
    """Fit or verify both cached probes for one clean label budget."""
    digests = input_hashes(budget)
    original = read_csv(PTB / f"seed42_fraction{FRACTIONS[budget]}" / "labeled_train.csv")
    clean = read_csv(PREFLIGHT / f"labels_fraction{FRACTIONS[budget]}.csv")
    train = validate_selection(clean, original)
    if len(train) != COUNTS[budget]:
        raise ValueError("Clean label count changed")
    development, calibration = partition_validation(
        read_csv(PTB / f"seed42_fraction{FRACTIONS[budget]}" / "validation.csv"))
    if len(development) != DEVELOPMENT_RECORDS or len(calibration) != 564:
        raise ValueError("Held-out partition changed")
    if {r["patient_id"] for r in train} & {r["patient_id"] for r in calibration}:
        raise ValueError("Training/calibration patients overlap")
    if {r["patient_id"] for r in development} & {r["patient_id"] for r in calibration}:
        raise ValueError("Development/calibration patients overlap")
    features = aligned_features(budget, train, development, digests)
    train_y = np.asarray([int(row["target"]) for row in train])
    dev_y = np.asarray([int(row["target"]) for row in development])
    result: dict[str, Any] = {"budget": budget, "cohort": "clean_original_ptbxl_labels",
                              "train_count": len(train), "development_count": len(development),
                              "input_sha256": digests, "models": {}}
    for model_name, (train_x, dev_x) in features.items():
        model_path = output / f"{budget}_{model_name}.npz"
        receipt_path = output / f"{budget}_{model_name}.json"
        if receipt_path.exists():
            saved = json.loads(receipt_path.read_text())
            if saved["input_sha256"] != digests or saved["model_sha256"] != sha256_file(model_path):
                raise ValueError(f"Existing clean {budget} {model_name} probe changed")
            result["models"][model_name] = saved
            print(f"Verified existing {budget} {model_name} clean probe", flush=True)
            continue
        print(f"Fitting {budget} {model_name} clean probe", flush=True)
        started = time.monotonic()
        arrays, selection = fit_probe(train_x, train_y, dev_x, dev_y)
        write_npz_atomic(model_path, **arrays)
        receipt = {"model": model_name, "budget": budget, "cohort": result["cohort"],
                   "train_count": len(train), "development_count": len(development),
                   "input_sha256": digests, "model_sha256": sha256_file(model_path),
                   "selection": selection, "original_development_auroc": ORIGINAL_SCORES[budget][model_name],
                   "elapsed_seconds": time.monotonic() - started, "completed_at_utc": utc_now()}
        write_json_atomic(receipt_path, receipt, sort_keys=True)
        result["models"][model_name] = receipt
        print(f"Completed {budget} {model_name}: C={selection['C']}, "
              f"development AUROC={selection['development_auroc']:.6f}", flush=True)
    return result


def run(output: Path) -> dict[str, Any]:
    """Verify the clean preflight, then refit both models at both label budgets."""
    verify_preflight()
    output.mkdir(parents=True, exist_ok=True)
    results = {budget: run_budget(budget, output) for budget in FRACTIONS}
    summary = {"cohort": "clean_original_ptbxl_labels", "results": {
        budget: {model: {"C": receipt["selection"]["C"],
                         "development_auroc": receipt["selection"]["development_auroc"],
                         "original_development_auroc": receipt["original_development_auroc"]}
                 for model, receipt in result["models"].items()}
        for budget, result in results.items()}}
    write_json_atomic(output / "summary.json", summary, sort_keys=True)
    return summary
