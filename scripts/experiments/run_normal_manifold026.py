"""Experiment 026: one-class screening by distance from the normal-ECG manifold, development and SPH."""

from __future__ import annotations

import hashlib
import json
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from ecg_experiment.eda.ptbxl import diagnostic_classes, load_metadata, load_statements
from ecg_experiment.files import sha256_file, write_json_atomic
from ecg_experiment.full_development import cohorts, fit_logistic, predict, ptb_table
from ecg_experiment.normal_manifold import (
    bootstrap_metrics,
    contrast,
    fit_knn,
    fit_mahalanobis,
    knn_scores,
    mahalanobis_scores,
    metrics,
    patient_resamples,
)
from scripts.experiments import run_label_efficiency025 as label_efficiency

ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "outputs/experiment026_normal_manifold_v1"
EFFICIENCY = ROOT / "outputs/experiment025_label_efficiency_v1"
SPH_RUN = ROOT / "outputs/experiment022_sph_external_v3"
SPH_ROWS = ROOT / "data/processed/sph_clean_v1/rows.csv"
ENCODERS = ("jepa", "xecg", "cpc")
ONE_CLASS = ("mahalanobis", "knn")
SUPERCLASSES = ("MI", "STTC", "CD", "HYP")
SUBCLASS_MINIMUM = 20
PROBE_BUDGET = "100"
SEED = 26026
BOOTSTRAP_DRAWS = 2000
MARGIN = 0.05
PROBE_TOLERANCE = 1e-8
AUROC_TOLERANCE = 1e-10
SPH_COUNTS = {"records": 21008, "positives": 7190}
SPH_PROBES = {"jepa": "jepa_standard", "xecg": "xecg_standard", "cpc": "cpc_standard"}
SOURCES = (
    "ecg_experiment/normal_manifold.py", "ecg_experiment/label_efficiency.py",
    "ecg_experiment/full_development.py", "ecg_experiment/eda/ptbxl.py", "ecg_experiment/files.py",
    "scripts/experiments/run_normal_manifold026.py", "scripts/experiments/run_label_efficiency025.py",
    "pyproject.toml", "uv.lock", "docs/experiment-026-normal-manifold.md",
)
INPUTS = {
    "label_efficiency_result": EFFICIENCY / "result.json",
    "label_efficiency_draws": EFFICIENCY / "draws.csv",
    "label_efficiency_predictions": EFFICIENCY / "all_budget_predictions.npz",
    "sph_result": SPH_RUN / "result.json",
    "sph_features": SPH_RUN / "features.npz",
    "sph_rows": SPH_ROWS,
}


def identity() -> dict[str, Any]:
    """Hash every input and source; caches and prior outputs must match their receipts."""
    caches = label_efficiency.identity()
    hashes = {name: sha256_file(path) for name, path in INPUTS.items()}
    efficiency = json.loads(INPUTS["label_efficiency_result"].read_text())
    sph = json.loads(INPUTS["sph_result"].read_text())
    expected = {
        "label_efficiency_draws": efficiency["draws_sha256"],
        "label_efficiency_predictions": efficiency["predictions_sha256"],
        "sph_features": sph["outputs_sha256"]["features.npz"],
        "sph_rows": sph["identity"]["sph"]["rows"],
    }
    for name, digest in expected.items():
        if hashes[name] != digest:
            raise ValueError(f"Input differs from its receipt: {name}")
    return {"inputs": {**caches["inputs"], **hashes},
            "sources": {name: sha256_file(ROOT / name) for name in SOURCES},
            "verified_against_receipts": caches["verified_against_receipts"] + sorted(expected)}


def record_classes() -> pd.DataFrame:
    """Abnormal diagnostic superclasses and subclasses of every PTB-XL record, by ECG ID."""
    meta = load_metadata()
    statements = load_statements()
    classes = diagnostic_classes(statements)
    subclass_of = statements.loc[statements["diagnostic"] == 1, "diagnostic_subclass"].to_dict()
    return pd.DataFrame({
        "superclasses": meta["scp_codes"].apply(
            lambda codes: sorted({classes[code] for code in codes if code in classes} - {"NORM"})),
        "subclasses": meta["scp_codes"].apply(
            lambda codes: sorted({subclass_of[code] for code in codes if code in subclass_of} - {"NORM"})),
    }, index=meta.index.astype(int))


def has_class(frame: pd.DataFrame, column: str, name: str) -> np.ndarray:
    """Rows whose superclass or subclass list contains ``name``."""
    return frame[column].apply(lambda found: name in found).to_numpy()


def eligible_subclasses(evaluation: pd.DataFrame) -> list[str]:
    """List subclasses with at least ``SUBCLASS_MINIMUM`` development positives."""
    counts = pd.Series([name for found in evaluation["subclasses"] for name in found]).value_counts()
    return sorted(counts[counts >= SUBCLASS_MINIMUM].index)


def development_tasks(evaluation: pd.DataFrame, subclasses: list[str]) -> dict[str, dict[str, Any]]:
    """Rows, positives, reported scores and bootstrapped comparators of every development task."""
    norm = (evaluation["standard"] == 0).to_numpy()
    base = ("probe_all", "probe_100")
    tasks = {"binary": {"rows": np.ones(len(evaluation), dtype=bool),
                        "positive": (evaluation["standard"] == 1).to_numpy(),
                        "reported": base, "comparators": base}}
    for name in SUPERCLASSES:
        present = has_class(evaluation, "superclasses", name)
        alone = evaluation["superclasses"].apply(lambda found, name=name: found == [name]).to_numpy()
        held_out = f"probe_without_{name}"
        tasks[f"superclass:{name}"] = {"rows": present | norm, "positive": present,
                                       "reported": (*base, held_out), "comparators": (*base, held_out)}
        tasks[f"only:{name}"] = {"rows": alone | norm, "positive": alone,
                                 "reported": (*base, held_out), "comparators": (held_out,)}
    for name in subclasses:
        present = has_class(evaluation, "subclasses", name)
        tasks[f"subclass:{name}"] = {"rows": present | norm, "positive": present,
                                     "reported": base, "comparators": ("probe_all",)}
    return tasks


def load_sph() -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    """SPH evaluation rows with a primary label and their saved Experiment 022 features."""
    rows = pd.read_csv(SPH_ROWS, dtype={"ecg_id": str, "patient_id": str})
    selected = rows["use_evaluation"].astype(str).eq("True") & rows["primary"].notna()
    rows = rows[selected].reset_index(drop=True)
    if len(rows) != SPH_COUNTS["records"] or int(rows["primary"].sum()) != SPH_COUNTS["positives"]:
        raise ValueError("SPH evaluation counts changed")
    with np.load(INPUTS["sph_features"]) as saved:
        position = pd.Series(np.arange(len(saved["ecg_ids"])), index=saved["ecg_ids"])
        if not position.index.is_unique:
            raise ValueError("Duplicate SPH ECG IDs")
        selected = position.loc[rows["ecg_id"].to_numpy()].to_numpy()
        features = {name: saved[name][selected] for name in ENCODERS}
    for name, values in features.items():
        if not np.isfinite(values).all():
            raise ValueError(f"Nonfinite SPH {name} features")
    return rows, features


def probe_references(evaluation: pd.DataFrame) -> dict[str, dict[str, Any]]:
    """Experiment 025 N = all probabilities and N = 100 draw receipts of each encoder."""
    with np.load(INPUTS["label_efficiency_predictions"], allow_pickle=True) as saved:
        if not np.array_equal(saved["record_ids"], evaluation.index.to_numpy()):
            raise ValueError("Experiment 025 development order changed")
        probabilities = {name: saved[f"primary_{name}"] for name in ENCODERS}
    draws = pd.read_csv(INPUTS["label_efficiency_draws"], dtype={"budget": str})
    draws = draws[(draws["readout"] == "primary") & (draws["budget"] == PROBE_BUDGET)]
    return {name: {"probe_all": probabilities[name],
                   "draws": draws[draws["encoder"] == name].sort_values("draw").reset_index(drop=True)}
            for name in ENCODERS}


def fit_probes(x_pool: np.ndarray, x_eval: np.ndarray, pool: pd.DataFrame, evaluation: pd.DataFrame,
               plans: list[tuple[int, np.ndarray]], reference: dict[str, Any]
               ) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Refit the 025 probes, check them against their receipts, and fit the held-out-superclass probes."""
    y_pool = pool["standard"].to_numpy(dtype=np.int64)
    y_eval = evaluation["standard"].to_numpy(dtype=np.int64)
    probes = {"probe_all": predict(fit_logistic(x_pool, y_pool), x_eval)}
    difference = float(np.abs(probes["probe_all"] - reference["probe_all"]).max())
    if difference > PROBE_TOLERANCE:
        raise ValueError(f"probe_all does not reproduce Experiment 025: {difference}")
    draws, worst = [], 0.0
    for (seed, positions), (_, saved) in zip(plans, reference["draws"].iterrows(), strict=True):
        subset_hash = hashlib.sha256("\n".join(pool.index[positions]).encode()).hexdigest()
        if seed != saved["seed"] or subset_hash != saved["subset_sha256"]:
            raise ValueError(f"Draw {seed} differs from Experiment 025")
        probability = predict(fit_logistic(x_pool[positions], y_pool[positions]), x_eval)
        worst = max(worst, abs(metrics(y_eval, probability)["auroc"] - saved["auroc"]))
        draws.append(probability)
    if worst > AUROC_TOLERANCE:
        raise ValueError(f"probe_100 does not reproduce Experiment 025: {worst}")
    probes["probe_100"] = np.stack(draws)
    training = {}
    for name in SUPERCLASSES:
        keep = ~has_class(pool, "superclasses", name)
        probes[f"probe_without_{name}"] = predict(fit_logistic(x_pool[keep], y_pool[keep]), x_eval)
        training[name] = {"records": int(keep.sum()), "positives": int(y_pool[keep].sum())}
    checks = {"probe_all_max_abs_difference": difference, "probe_100_max_auroc_difference": worst,
              "probe_100_draws": len(draws), "probe_without_training": training}
    return probes, checks


def evaluate_task(y: np.ndarray, patients: np.ndarray, scores: dict[str, np.ndarray],
                  comparators: tuple[str, ...]) -> dict[str, Any]:
    """Point metrics of every score and paired bootstrap contrasts of each one-class score."""
    observed = {name: metrics(y, values) for name, values in scores.items()}
    resamples, invalid = patient_resamples(patients, y, BOOTSTRAP_DRAWS, SEED)
    resampled = bootstrap_metrics(y, {name: scores[name] for name in (*ONE_CLASS, *comparators)}, resamples)
    contrasts = {f"{score}_minus_{other}": contrast(observed, resampled, score, other)
                 for score in ONE_CLASS for other in comparators}
    return {"records": len(y), "positives": int(y.sum()), "patients": len(np.unique(patients)),
            "metrics": observed, "contrasts": contrasts, "invalid_draws": invalid}


def evaluate_sph(scores: dict[str, np.ndarray], sph: pd.DataFrame) -> dict[str, Any]:
    """AUROC and AP on the SPH primary label and per superclass versus negatives."""
    y = sph["primary"].to_numpy(dtype=np.int64)
    result = {"primary": {name: metrics(y, values) for name, values in scores.items()}, "superclasses": {}}
    for name in SUPERCLASSES:
        rows = ((sph[name] == 1) | (sph["primary"] == 0)).to_numpy()
        result["superclasses"][name] = {
            "records": int(rows.sum()), "positives": int((sph.loc[rows, name] == 1).sum()),
            **{score: metrics(sph.loc[rows, name].to_numpy(dtype=np.int64), values[rows])
               for score, values in scores.items()}}
    return result


def evaluate_encoder(name: str, x_pool: np.ndarray, x_eval: np.ndarray, x_sph: np.ndarray, pool: pd.DataFrame,
                     evaluation: pd.DataFrame, sph: pd.DataFrame, tasks: dict[str, dict[str, Any]],
                     plans: list[tuple[int, np.ndarray]], reference: dict[str, Any]) -> dict[str, Any]:
    """Probes, one-class scores, development tasks and the SPH readout of one encoder."""
    start = time.monotonic()
    with threadpool_limits(limits=1):
        probes, checks = fit_probes(x_pool, x_eval, pool, evaluation, plans, reference)
        print(json.dumps({"stage": f"probes:{name}", **checks}), flush=True)
        fit = (pool["standard"] == 0).to_numpy()
        mahalanobis_model = fit_mahalanobis(x_pool[fit])
        knn_model = fit_knn(x_pool[fit])
        development = {"mahalanobis": mahalanobis_scores(mahalanobis_model, x_eval),
                       "knn": knn_scores(knn_model, x_eval), **probes}
        external = {"mahalanobis": mahalanobis_scores(mahalanobis_model, x_sph),
                    "knn": knn_scores(knn_model, x_sph)}
        results = {}
        for task, spec in tasks.items():
            rows = spec["rows"]
            scores = {score: np.atleast_2d(development[score])[:, rows]
                      for score in (*ONE_CLASS, *spec["reported"])}
            patients = evaluation["patient_id"].to_numpy()[rows]
            results[task] = evaluate_task(spec["positive"][rows].astype(np.int64), patients, scores,
                                          spec["comparators"])
            print(json.dumps({"stage": f"{name}:{task}", "auroc": {
                score: round(values["auroc"], 4) for score, values in results[task]["metrics"].items()}}),
                flush=True)
        sph_result = evaluate_sph(external, sph)
    return {"checks": checks, "fit_records": int(fit.sum()),
            "pca_explained_variance": float(mahalanobis_model[1].explained_variance_ratio_.sum()),
            "tasks": results, "sph": sph_result, "development_scores": development, "sph_scores": external,
            "seconds": time.monotonic() - start}


def reading(results: dict[str, Any]) -> dict[str, Any]:
    """Apply the prespecified competitive and unseen-condition rules to every encoder and score."""
    readings = {}
    for name in ENCODERS:
        tasks = results[name]["tasks"]
        readings[name] = {}
        for score in ONE_CLASS:
            gap = tasks["binary"]["contrasts"][f"{score}_minus_probe_all"]["auroc"]["difference"]
            unseen = {group: tasks[f"superclass:{group}"]["contrasts"][f"{score}_minus_probe_without_{group}"]
                      ["auroc"]["ci_high"] >= 0 for group in SUPERCLASSES}
            alone = {group: tasks[f"only:{group}"]["contrasts"][f"{score}_minus_probe_without_{group}"]
                     ["auroc"]["ci_high"] >= 0 for group in SUPERCLASSES}
            verdict = []
            if gap >= -MARGIN:
                verdict.append("competitive")
            if any(unseen.values()):
                verdict.append("useful for unseen conditions: "
                               + ", ".join(group for group, useful in unseen.items() if useful))
            readings[name][score] = {"auroc_minus_probe_all": gap, "competitive": gap >= -MARGIN,
                                     "useful_for_unseen": unseen, "useful_for_unseen_only_sensitivity": alone,
                                     "verdict": "; ".join(verdict) or "not competitive"}
    return readings


def score_tables(evaluation: pd.DataFrame, sph: pd.DataFrame, results: dict[str, Any]
                 ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Per-row development and SPH scores of every encoder."""
    development = evaluation[["ecg_id", "patient_id", "standard"]].copy()
    development["superclasses"] = evaluation["superclasses"].str.join("+")
    development["subclasses"] = evaluation["subclasses"].str.join("+")
    external = sph[["ecg_id", "patient_id", "primary", *SUPERCLASSES]].copy()
    for name in ENCODERS:
        for score, values in results[name]["development_scores"].items():
            if score != "probe_100":
                development[f"{name}_{score}"] = values
        for score, values in results[name]["sph_scores"].items():
            external[f"{name}_{score}"] = values
    return development, external


def main() -> None:
    """Run the frozen development and SPH one-class study once."""
    if (OUTPUT / "result.json").exists():
        raise FileExistsError("Experiment 026 v1 has already run")
    start = time.monotonic()
    run_identity = identity()
    groups = cohorts(ptb_table())
    train, dev = groups["train"], groups["development"]
    pool, evaluation, counts = label_efficiency.select_rows(train, dev)
    classes = record_classes()
    pool, evaluation = pool.join(classes, on="ecg_id"), evaluation.join(classes, on="ecg_id")
    subclasses = eligible_subclasses(evaluation)
    tasks = development_tasks(evaluation, subclasses)
    features = label_efficiency.load_features(train, dev, pool, evaluation)
    sph, sph_features = load_sph()
    references = probe_references(evaluation)
    plans = [(seed, positions) for budget, _, seed, positions in label_efficiency.draw_plans(pool)
             if budget == PROBE_BUDGET]
    counts = {**counts, "fit_records": int((pool["standard"] == 0).sum()),
              "fit_patients": int(pool.loc[pool["standard"] == 0, "patient_id"].nunique()),
              "evaluation_norm": int((evaluation["standard"] == 0).sum()),
              "sph_records": len(sph), "sph_positives": int(sph["primary"].sum()),
              "sph_patients": int(sph["patient_id"].nunique())}
    print(json.dumps({"stage": "rows", **counts, "subclasses": subclasses}), flush=True)
    with ProcessPoolExecutor(max_workers=len(ENCODERS)) as executor:
        futures = {name: executor.submit(evaluate_encoder, name, *features[name], sph_features[name], pool,
                                         evaluation, sph, tasks, plans, references[name])
                   for name in ENCODERS}
        results = {name: future.result() for name, future in futures.items()}
    OUTPUT.mkdir(parents=True, exist_ok=True)
    development, external = score_tables(evaluation, sph, results)
    development.to_csv(OUTPUT / "development_scores.csv")
    external.to_csv(OUTPUT / "sph_scores.csv", index=False)
    sph_prior = json.loads(INPUTS["sph_result"].read_text())
    result = {
        "status": "complete_development_and_sph", "identity": run_identity, "counts": counts,
        "subclasses": subclasses, "seed": SEED, "bootstrap_draws": BOOTSTRAP_DRAWS, "margin": MARGIN,
        "encoders": {name: {key: value for key, value in output.items()
                            if key not in ("development_scores", "sph_scores")}
                     for name, output in results.items()},
        "reading": reading(results),
        "sph_probes_experiment022": {name: {"primary": sph_prior["by_label"]["primary"][head],
                                            "superclasses": sph_prior["superclasses"][head]}
                                     for name, head in SPH_PROBES.items()},
        "development_scores_sha256": sha256_file(OUTPUT / "development_scores.csv"),
        "sph_scores_sha256": sha256_file(OUTPUT / "sph_scores.csv"),
        "total_seconds": time.monotonic() - start, "calibration_test_evaluated": False,
    }
    write_json_atomic(OUTPUT / "result.json", result)
    print(json.dumps({"stage": "complete", "total_seconds": result["total_seconds"]}), flush=True)


if __name__ == "__main__":
    main()
