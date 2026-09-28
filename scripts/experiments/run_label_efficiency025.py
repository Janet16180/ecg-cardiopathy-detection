"""Experiment 025: label efficiency of cached frozen encoders on PTB-XL development patients."""

from __future__ import annotations

import hashlib
import json
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score
from threadpoolctl import threadpool_limits

from ecg_experiment.files import sha256_file, write_json_atomic
from ecg_experiment.full_development import (
    cohorts,
    demographics,
    fit_logistic,
    patient_bootstrap,
    predict,
    ptb_table,
)
from ecg_experiment.label_efficiency import (
    draw_subset,
    fit_logistic_c,
    paired_summary,
    reading,
    select_c,
    smallest_budget,
    summarize,
)

ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "outputs/experiment025_label_efficiency_v1"
PRIOR = ROOT / "outputs/experiment020_full_development_v2"
CACHES = {
    "jepa": ROOT / "data/processed/pretrained/ecg-jepa-full-public",
    "xecg": ROOT / "outputs/experiment016_xecg_probe_finetune/features",
    "released_cpc": ROOT / "outputs/experiment004_cpc_40k/released_features",
}
JEPA_REFERENCE = ROOT / "outputs/experiment014_jepa_cpc_fusion/full.json"
ENCODERS = ("xecg", "jepa", "cpc", "released_cpc", "age_sex")
BUDGETS = (100, 250, 500, 1000, 2000, 4000)
READ_BUDGETS = ("250", "1000")
DRAWS = 20
SEED = 25025
BOOTSTRAP_DRAWS = 2000
WORKERS = 4
INTEGRITY_TOLERANCE = 1e-8
CONTRASTS = (("jepa", "cpc"), ("xecg", "cpc"), ("xecg", "jepa"), ("released_cpc", "cpc"),
             ("cpc", "age_sex"), ("jepa", "age_sex"), ("xecg", "age_sex"), ("released_cpc", "age_sex"))
SOURCES = (
    "ecg_experiment/label_efficiency.py", "ecg_experiment/full_development.py", "ecg_experiment/eda/ptbxl.py",
    "ecg_experiment/files.py", "scripts/experiments/run_label_efficiency025.py", "pyproject.toml", "uv.lock",
    "docs/experiment-025-label-efficiency.md",
)
INPUTS = {
    "cpc_features": PRIOR / "features.npz",
    "prior_result": PRIOR / "result.json",
    "prior_predictions": PRIOR / "development_predictions.npz",
    "jepa_features": CACHES["jepa"] / "features.npy",
    "jepa_ids": CACHES["jepa"] / "ecg_ids.npy",
    "jepa_metadata": CACHES["jepa"] / "metadata.json",
    "jepa_reference": JEPA_REFERENCE,
    "xecg_features": CACHES["xecg"] / "features.npy",
    "xecg_ids": CACHES["xecg"] / "ecg_ids.npy",
    "xecg_receipt": CACHES["xecg"] / "receipt.json",
    "released_cpc_features": CACHES["released_cpc"] / "features.npy",
    "released_cpc_ids": CACHES["released_cpc"] / "ecg_ids.npy",
    "released_cpc_metadata": CACHES["released_cpc"] / "metadata.json",
    "ptbxl_database": ROOT / "data/raw/ptb-xl/1.0.3/ptbxl_database.csv",
    "ptbxl_statements": ROOT / "data/raw/ptb-xl/1.0.3/scp_statements.csv",
    "clean_references": ROOT / "outputs/data_quality/clean_rerun_preflight_v1/heldout_references.csv",
    "clean_labels_full": ROOT / "outputs/data_quality/clean_rerun_preflight_v1/labels_fraction1.csv",
    "clean_labels_limited": ROOT / "outputs/data_quality/clean_rerun_preflight_v1/labels_fraction0.1.csv",
    "union_manifest": ROOT / "data/processed/training_union_500hz_v1/train_manifest.csv",
}


def identity() -> dict[str, Any]:
    """Hash every input and require the caches to match their extraction receipts."""
    hashes = {name: sha256_file(path) for name, path in INPUTS.items()}
    jepa = json.loads(JEPA_REFERENCE.read_text())["fingerprint"]["input_sha256"]
    xecg = json.loads((CACHES["xecg"] / "receipt.json").read_text())["sha256"]
    released = json.loads((CACHES["released_cpc"] / "metadata.json").read_text())["output_sha256"]
    expected = {
        "cpc_features": json.loads((PRIOR / "result.json").read_text())["features_sha256"],
        "jepa_features": jepa["data/processed/pretrained/ecg-jepa-full-public/features.npy"],
        "jepa_ids": jepa["data/processed/pretrained/ecg-jepa-full-public/ecg_ids.npy"],
        "jepa_metadata": jepa["data/processed/pretrained/ecg-jepa-full-public/metadata.json"],
        "xecg_features": xecg["features.npy"], "xecg_ids": xecg["ecg_ids.npy"],
        "released_cpc_features": released["features.npy"], "released_cpc_ids": released["ecg_ids.npy"],
    }
    for name, digest in expected.items():
        if hashes[name] != digest:
            raise ValueError(f"Cached input differs from its receipt: {name}")
    sources = {name: sha256_file(ROOT / name) for name in SOURCES}
    return {"inputs": hashes, "sources": sources, "verified_against_receipts": sorted(expected)}


def cached_ids(name: str) -> np.ndarray:
    """ECG IDs of a released cache as integers."""
    return np.load(CACHES[name] / "ecg_ids.npy").astype(np.int64)


def cached_rows(name: str, ecg_ids: pd.Series) -> np.ndarray:
    """Read only the requested ECG rows of a memory-mapped released cache."""
    ids = cached_ids(name)
    position = pd.Series(np.arange(len(ids)), index=ids)
    if not position.index.is_unique:
        raise ValueError(f"Duplicate ECG IDs in the {name} cache")
    features = np.load(CACHES[name] / "features.npy", mmap_mode="r")
    rows = np.asarray(features[position.loc[ecg_ids.to_numpy()].to_numpy()])
    if not np.isfinite(rows).all():
        raise ValueError(f"Nonfinite {name} features")
    return rows


def select_rows(train: pd.DataFrame, dev: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, int]]:
    """Training pool and evaluation rows with a standard label and features in every cache."""
    common = set.intersection(*(set(cached_ids(name).tolist()) for name in CACHES))
    defined = train["standard"].notna()
    cached = train["ecg_id"].isin(common)
    pool = train[defined & cached]
    original = dev[dev["original"]]
    evaluation = original[original["standard"].notna() & original["ecg_id"].isin(common)]
    if (pool["strat_fold"] > 8).any() or (evaluation["strat_fold"] != 9).any():
        raise ValueError("A selected row is outside training or development folds")
    counts = {
        "training_rows": len(train), "training_without_standard_label": int((~defined).sum()),
        "training_labeled_not_in_released_caches": int((defined & ~cached).sum()),
        "pool_records": len(pool), "pool_patients": int(pool["patient_id"].nunique()),
        "pool_positives": int(pool["standard"].sum()),
        "original_development_rows": len(original), "evaluation_records": len(evaluation),
        "evaluation_patients": int(evaluation["patient_id"].nunique()),
        "evaluation_positives": int(evaluation["standard"].sum()),
        "evaluation_dropped": len(original) - len(evaluation),
    }
    return pool, evaluation, counts


def load_features(train: pd.DataFrame, dev: pd.DataFrame, pool: pd.DataFrame,
                  evaluation: pd.DataFrame) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Frozen features of the pool and the evaluation rows for each encoder, joined by ECG ID."""
    with np.load(PRIOR / "features.npz") as saved:
        cpc_train, cpc_dev = saved["train"], saved["development"]
    if cpc_train.shape != (len(train), 512) or cpc_dev.shape != (len(dev), 512):
        raise ValueError("The Experiment 020 CPC features do not match the cohort")
    features = {"cpc": (cpc_train[train.index.get_indexer(pool.index)],
                        cpc_dev[dev.index.get_indexer(evaluation.index)])}
    for name in CACHES:
        features[name] = (cached_rows(name, pool["ecg_id"]), cached_rows(name, evaluation["ecg_id"]))
    return features


def integrity(train: pd.DataFrame, dev: pd.DataFrame) -> dict[str, float]:
    """Refit Experiment 020's ``cpc_standard`` head and reproduce its saved probabilities."""
    with np.load(PRIOR / "features.npz") as saved:
        cpc_train, cpc_dev = saved["train"], saved["development"]
    with np.load(PRIOR / "development_predictions.npz", allow_pickle=True) as saved:
        if not np.array_equal(saved["record_ids"], dev.index.to_numpy()):
            raise ValueError("Experiment 020 development order changed")
        prior = saved["cpc_standard"]
    mask = train["standard"].notna().to_numpy()
    head = fit_logistic(cpc_train[mask], train.loc[mask, "standard"].to_numpy(dtype=np.int64))
    difference = float(np.abs(predict(head, cpc_dev) - prior).max())
    if difference > INTEGRITY_TOLERANCE:
        raise ValueError(f"cpc_standard does not reproduce Experiment 020: {difference}")
    return {"cpc_standard_training_rows": int(mask.sum()), "cpc_standard_max_abs_difference": difference}


def draw_plans(pool: pd.DataFrame) -> list[tuple[str, int, int, np.ndarray]]:
    """Budget label, draw, seed and pool positions of every paired draw, then N = all."""
    patients = pool["patient_id"].to_numpy()
    y = pool["standard"].to_numpy(dtype=np.int64)
    plans = [(str(size), draw, SEED + draw, draw_subset(patients, y, size, SEED + draw))
             for size in BUDGETS for draw in range(DRAWS)]
    plans.append(("all", 0, SEED, np.arange(len(pool))))
    return plans


def design(name: str, features: tuple[np.ndarray, np.ndarray] | None, fitted: pd.DataFrame,
           positions: np.ndarray, evaluation: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Training and evaluation inputs of one encoder for one draw."""
    if name == "age_sex":
        median_age = float(fitted["age"].median())
        return demographics(fitted, median_age), demographics(evaluation, median_age)
    pool_x, evaluation_x = features
    return pool_x[positions], evaluation_x


def evaluate_encoder(name: str, features: tuple[np.ndarray, np.ndarray] | None, pool: pd.DataFrame,
                     evaluation: pd.DataFrame, plans: list[tuple[str, int, int, np.ndarray]]
                     ) -> tuple[list[dict[str, Any]], dict[str, np.ndarray], float]:
    """Fit both readouts of one encoder on every draw and score the evaluation rows."""
    start = time.monotonic()
    y_pool = pool["standard"].to_numpy(dtype=np.int64)
    patients = pool["patient_id"].to_numpy()
    y_eval = evaluation["standard"].to_numpy(dtype=np.int64)
    rows, final = [], {}
    with threadpool_limits(limits=1):
        for budget, draw, seed, positions in plans:
            fitted = pool.iloc[positions]
            x_train, x_eval = design(name, features, fitted, positions, evaluation)
            y_train = y_pool[positions]
            chosen, cv_means = select_c(x_train, y_train, patients[positions], seed)
            scores = {"primary": predict(fit_logistic(x_train, y_train), x_eval),
                      "secondary": predict(fit_logistic_c(x_train, y_train, chosen), x_eval)}
            c_values = {"primary": 0.01, "secondary": chosen}
            cv_text = {"primary": "", "secondary": json.dumps(cv_means)}
            subset_hash = hashlib.sha256("\n".join(fitted.index).encode()).hexdigest()
            for readout, probability in scores.items():
                rows.append({
                    "encoder": name, "readout": readout, "budget": budget, "draw": draw, "seed": seed,
                    "records": len(positions), "positives": int(y_train.sum()),
                    "patients": int(fitted["patient_id"].nunique()), "c": c_values[readout],
                    "cv_auroc": cv_text[readout],
                    "auroc": float(roc_auc_score(y_eval, probability)),
                    "average_precision": float(average_precision_score(y_eval, probability)),
                    "subset_sha256": subset_hash,
                })
            if budget == "all":
                final = scores
    return rows, final, time.monotonic() - start


def analyse(table: pd.DataFrame, final: dict[str, np.ndarray], evaluation: pd.DataFrame) -> dict[str, Any]:
    """Per-budget summaries, paired contrasts, bootstrap at N = all and the prespecified reading."""
    budgets = [str(size) for size in BUDGETS]
    per_budget, contrasts, chosen_c = {}, {}, {}
    for budget in budgets:
        frame = table[table["budget"] == budget]
        wide = {metric: frame.pivot(index="draw", columns="encoder", values=metric)
                for metric in ("auroc", "average_precision")}
        per_budget[budget] = {name: {metric: summarize(wide[metric][name].to_numpy()) for metric in wide}
                              for name in ENCODERS}
        contrasts[budget] = {f"{first}_minus_{second}": {
            metric: paired_summary(wide[metric][first].to_numpy(), wide[metric][second].to_numpy())
            for metric in wide} for first, second in CONTRASTS}
        counts = frame.groupby(["encoder", "c"]).size()
        chosen_c[budget] = {name: {str(c): int(n) for c, n in counts[name].items()} for name in ENCODERS}
    at_all = table[table["budget"] == "all"].set_index("encoder")
    full = {name: {"auroc": float(at_all.loc[name, "auroc"]),
                   "average_precision": float(at_all.loc[name, "average_precision"]),
                   "c": float(at_all.loc[name, "c"])} for name in ENCODERS}
    y = evaluation["standard"].to_numpy(dtype=np.int64)
    patients = evaluation["patient_id"].to_numpy()
    bootstrap = {f"{first}_minus_{second}": patient_bootstrap(
        patients, y, final[first], final[second], draws=BOOTSTRAP_DRAWS, seed=SEED)
        for first, second in CONTRASTS}
    readings = {budget: {f"{first}_minus_{second}": reading(
        first, second, contrasts[budget][f"{first}_minus_{second}"]["auroc"])
        for first, second in CONTRASTS} for budget in READ_BUDGETS}
    target = full["cpc"]["auroc"]
    reaches = {name: smallest_budget({int(b): per_budget[b][name]["auroc"]["mean"] for b in budgets}, target)
               for name in ENCODERS}
    return {"per_budget": per_budget, "contrasts": contrasts, "chosen_c": chosen_c, "all": full,
            "all_bootstrap": bootstrap, "reading": readings, "cpc_all_auroc": target,
            "smallest_budget_reaching_cpc_all": reaches}


def main() -> None:
    """Run the frozen development-only label-efficiency study once."""
    if (OUTPUT / "result.json").exists():
        raise FileExistsError("Experiment 025 v1 has already run")
    start = time.monotonic()
    run_identity = identity()
    groups = cohorts(ptb_table())
    train, dev = groups["train"], groups["development"]
    pool, evaluation, counts = select_rows(train, dev)
    with threadpool_limits(limits=1):
        checks = integrity(train, dev)
    print(json.dumps({"stage": "integrity", **checks, **counts}), flush=True)
    features = load_features(train, dev, pool, evaluation)
    plans = draw_plans(pool)
    with ProcessPoolExecutor(max_workers=WORKERS) as executor:
        futures = {name: executor.submit(evaluate_encoder, name, features.get(name), pool, evaluation, plans)
                   for name in ENCODERS}
        outputs = {name: future.result() for name, future in futures.items()}
    table = pd.DataFrame([row for rows, _, _ in outputs.values() for row in rows])
    OUTPUT.mkdir(parents=True, exist_ok=True)
    table.to_csv(OUTPUT / "draws.csv", index=False)
    finals = {readout: {name: final[readout] for name, (_, final, _) in outputs.items()}
              for readout in ("primary", "secondary")}
    np.savez_compressed(OUTPUT / "all_budget_predictions.npz", record_ids=evaluation.index.to_numpy(),
                        **{f"{readout}_{name}": values for readout, scores in finals.items()
                           for name, values in scores.items()})
    result = {
        "status": "complete_development_only", "identity": run_identity, "integrity": checks,
        "counts": counts, "prevalence": counts["pool_positives"] / counts["pool_records"],
        "budgets": list(BUDGETS), "draws": DRAWS, "seed": SEED, "bootstrap_draws": BOOTSTRAP_DRAWS,
        "readouts": {readout: analyse(table[table["readout"] == readout], finals[readout], evaluation)
                     for readout in ("primary", "secondary")},
        "encoder_seconds": {name: seconds for name, (_, _, seconds) in outputs.items()},
        "draws_sha256": sha256_file(OUTPUT / "draws.csv"),
        "predictions_sha256": sha256_file(OUTPUT / "all_budget_predictions.npz"),
        "total_seconds": time.monotonic() - start, "calibration_test_evaluated": False,
    }
    write_json_atomic(OUTPUT / "result.json", result)
    print(json.dumps({"stage": "complete", "total_seconds": result["total_seconds"]}), flush=True)


if __name__ == "__main__":
    main()
