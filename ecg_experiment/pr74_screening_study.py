"""Fit fixed heads on all prespecified cohorts and report reused-development matrices."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

from . import ROOT
from .cpu_screening041 import HGB_CONFIG, RAW_FEATURES, sensitivity_difference
from .files import sha256_file, write_json_atomic, write_npz_atomic
from .full_development import fit_logistic, predict
from .pr74_screening_io import (
    CPU_PRIOR,
    PRIOR,
    binding,
    cohort_tables,
    completed,
    finish,
    read_json,
    stored,
    training_pool,
    verify_hashes,
)
from .pr74_screening_metrics import budget_threshold, point_metrics, threshold_sweep
from .pr74_screening_replay import evaluation_features, raw_features, site_data

STRICT = Path("outputs/cohort_strict_young_matched_20261028")
SEED = 20261028
DRAWS = 2000
ARM_LABELS = {
    "strict_young": "Strict ages 18–25, 770 training ECGs",
    "matched_general": "General ages, 770 matched training ECGs",
    "younger": "Age-preferred selection, up to 25,000 training ECGs",
    "all_age": "All-age selection, up to 25,000 training ECGs",
    "ptb_baseline": "Matched original PTB-only reference",
}


def strict_tables(input_root: Path, pool: pd.DataFrame) -> tuple[dict[str, pd.DataFrame], dict[str, str]]:
    """Require the separately completed strict/matched selection and exact candidate/stratum membership."""
    path = STRICT / "receipt.json"
    receipt = read_json(input_root / path)
    if receipt.get("complete") is not True or receipt.get("scope") != "strict_young_matched_cohorts":
        raise ValueError("Completed strict/matched cohort receipt is required")
    hashes = {stored(path): sha256_file(input_root / path), **receipt["artifacts"]}
    verify_hashes(input_root, hashes)
    verify_hashes(input_root, receipt["input_sha256"])
    verify_hashes(ROOT, receipt["recipe_sha256"])
    hashes.update(receipt["input_sha256"])
    candidates = pool.set_index("record_id")
    tables = {}
    for name, filename in (("strict_young", "strict_18_25.csv"),
                           ("matched_general", "general_matched_size.csv")):
        rows = pd.read_csv(input_root / STRICT / filename, dtype=str, keep_default_na=False)
        if rows["record_id"].duplicated().any() or not set(rows["record_id"]) <= set(candidates.index):
            raise ValueError("Strict/matched cohort leaves the verified candidate pool")
        expected = candidates.loc[rows["record_id"]]
        for column in ("target", "source", "label_evidence_rank", "signal_sha256", "split", "patient_id"):
            if rows[column].tolist() != expected[column].tolist():
                raise ValueError(f"Derived cohort admission changed: {column}")
        if name == "strict_young" and not pd.to_numeric(rows["age"]).between(18, 25).all():
            raise ValueError("Strict training cohort includes a record outside ages 18–25")
        rows["target"] = pd.to_numeric(rows["target"]).astype(int)
        tables[name] = rows
    keys = ["source", "label_evidence_rank", "target"]
    left = tables["strict_young"].groupby(keys).size()
    right = tables["matched_general"].groupby(keys).size()
    if not left.equals(right) or len(tables["strict_young"]) != 770:
        raise ValueError("Primary cohorts must preserve all 770 exact matching strata")
    return tables, hashes


def training_features(input_root: Path, output: Path,
                      pool: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray]:
    """Align cached PTB and newly extracted Challenge features on admitted training IDs only."""
    extracted = pd.read_csv(input_root / output / "extract/rows.csv", dtype=str, keep_default_na=False)
    challenge = pool[pool["source"] != "ptbxl"]
    if extracted["record_id"].tolist() != challenge["record_id"].tolist():
        raise ValueError("Challenge extracted IDs differ from the verified training union")
    for column in ("signal_sha256", "target", "split"):
        if extracted[column].tolist() != challenge[column].tolist():
            raise ValueError("Challenge extraction provenance changed")
    raw = pd.DataFrame(np.nan, index=pool["record_id"], columns=RAW_FEATURES)
    vectors = np.empty((len(pool), 1024), dtype=np.float32)
    challenge_index = pd.Index(pool["record_id"]).get_indexer(challenge["record_id"])
    raw.loc[challenge["record_id"]] = pd.read_csv(input_root / output / "extract/measurements.csv").to_numpy()
    vectors[challenge_index] = np.load(input_root / output / "extract/xecg.npy", mmap_mode="r")
    ptb = pool[pool["source"] == "ptbxl"]
    ecg_ids = ptb["record_id"].str.removeprefix("ptbxl:").tolist()
    raw.loc[ptb["record_id"]] = raw_features(input_root, ecg_ids).to_numpy()
    cached = pd.read_csv(input_root / PRIOR / "qualify/ptb_rows.csv", dtype={"ecg_id": str})
    cached_ids = pd.Index("ptbxl:" + cached["ecg_id"])
    positions = cached_ids.get_indexer(ptb["record_id"])
    if np.any(positions < 0):
        raise ValueError("An admitted PTB training feature is absent from the qualified cache")
    vectors[pd.Index(pool["record_id"]).get_indexer(ptb["record_id"])] = np.load(
        input_root / PRIOR / "qualify/ptb_xecg.npy", mmap_mode="r")[positions]
    if not np.isfinite(vectors).all() or raw.shape != (len(pool), 130):
        raise ValueError("Training feature schema/nonfinite encoder contract failed")
    return raw, vectors


def assert_evaluation_independence(arms: dict[str, pd.DataFrame], site: str, frame: pd.DataFrame) -> None:
    """Reject any observed train/evaluation patient or record identity before scoring a site."""
    prefix = "ptbxl:" if site == "ptb" else "sph:"
    records = set(prefix + frame["ecg_id"].astype(str))
    patients = set(frame["patient_id"])
    for name, rows in arms.items():
        known = set(rows.loc[rows["patient_id"].ne(""), "patient_id"])
        if set(rows["record_id"]) & records or known & patients:
            raise ValueError(f"Training/evaluation identity overlap in {name}/{site}")


def score_models(fitted: dict[str, Any], vectors: np.ndarray, raw: pd.DataFrame) -> dict[str, np.ndarray]:
    """Compute fixed component and equal-weight mixture scores without selecting a model."""
    hgb = fitted["hgb"].predict_proba(raw)[:, 1]
    xecg = predict(fitted["xecg"], vectors)
    return {"hgb": hgb, "xecg": xecg, "blend": 0.5 * hgb + 0.5 * xecg}


def evaluate_site(input_root: Path, site: str, fitted: dict[str, Any],
                  arms: dict[str, pd.DataFrame]) -> tuple[list[dict], dict[str, np.ndarray], pd.DataFrame]:
    """Apply frozen training recipes and calibrate only on the predeclared site's normal calibration rows."""
    frame, vectors, raw = evaluation_features(input_root, site)
    assert_evaluation_independence(arms, site, frame)
    y = frame["target"].to_numpy(dtype=int)
    held = frame["evaluation"].to_numpy()
    baseline = site_data(input_root, site)[1]
    scored = {name: score_models(model, vectors, raw) for name, model in fitted.items()
              if name != "ptb_baseline"}
    scored["ptb_baseline"] = {"hgb": baseline["hgb"], "xecg": baseline["xecg"],
                               "blend": 0.5 * baseline["hgb"] + 0.5 * baseline["xecg"]}
    records, arrays = [], {}
    for cohort, kinds in scored.items():
        for kind, scores in kinds.items():
            model_id = cohort + "_" + kind
            calibrated = budget_threshold(y[~held], scores[~held])
            threshold = calibrated["threshold"]
            metrics = point_metrics(y[held], scores[held], threshold)
            records.append({"model_id": model_id, "cohort": cohort, "kind": kind,
                            "evaluation": {"pool": site, "label": site.upper() + " reused development",
                                           "threshold": threshold, "comparison": ">",
                                           "calibration": calibrated,
                                           "metrics": metrics,
                                           "sweep": threshold_sweep(y[held], scores[held], threshold)}})
            arrays[model_id] = scores
    return records, arrays, frame


def primary_comparison(records: list[dict], arrays: dict[str, np.ndarray],
                       frame: pd.DataFrame) -> dict[str, Any]:
    """Apply the one prespecified practical paired-patient sensitivity decision."""
    points = {row["model_id"]: row["evaluation"] for row in records}
    held = frame["evaluation"].to_numpy()
    first, second = "strict_young_blend", "matched_general_blend"
    difference = sensitivity_difference(
        frame.loc[held, "patient_id"].to_numpy(), frame.loc[held, "target"].to_numpy(dtype=int),
        arrays[first][held] > points[first]["threshold"],
        arrays[second][held] > points[second]["threshold"], draws=DRAWS, seed=SEED,
    )
    gates = {"sensitivity_gain_at_least_0.02": difference["difference"] >= 0.02,
             "paired_95_lower_bound_positive": difference["ci_low"] > 0}
    return {"first": first, "second": second, "site": "sph", "sensitivity_difference": difference,
            "gates": gates, "development_advantage": all(gates.values()), "pipeline_promoted": False,
            "interpretation": "prespecified development advantage established" if all(gates.values())
            else "prespecified development advantage not established",
            "uncertainty_scope": "patient bootstrap conditional on fitted models and calibration sample"}


def aggregate_models(records: list[dict]) -> list[dict]:
    """Group only aggregate site summaries into the UI's explicit model schema."""
    grouped = {}
    for row in records:
        key = row["model_id"]
        if key not in grouped:
            grouped[key] = {"id": key, "cohort": row["cohort"],
                            "label": ARM_LABELS[row["cohort"]] + " · " + row["kind"], "evaluations": []}
        grouped[key]["evaluations"].append(row["evaluation"])
    return list(grouped.values())


def evaluate(input_root: Path, output: Path) -> dict[str, Any]:
    """Fit all prespecified cohorts before any new evaluation score and export every result."""
    receipt = binding()
    started = time.perf_counter()
    qualified, hashes = completed(input_root, output / "qualify")
    hashes.update(qualified["input_sha256"])
    extracted, bound = completed(input_root, output / "extract")
    hashes.update(bound)
    hashes.update(extracted["input_sha256"])
    verify_hashes(input_root, hashes)
    verify_hashes(ROOT, qualified["source_sha256"])
    verify_hashes(ROOT, extracted["source_sha256"])
    arms, bound = cohort_tables(input_root)
    hashes.update(bound)
    pool, _ = training_pool(input_root)
    strict, bound = strict_tables(input_root, pool)
    hashes.update(bound)
    arms = {**strict, **arms}
    raw, vectors = training_features(input_root, output, pool)
    fitted, timings = {}, {}
    for name, rows in arms.items():
        positions = pd.Index(pool["record_id"]).get_indexer(rows["record_id"])
        y = rows["target"].to_numpy(dtype=int)
        start = time.perf_counter()
        fitted[name] = {"hgb": HistGradientBoostingClassifier(**HGB_CONFIG).fit(raw.iloc[positions], y),
                        "xecg": fit_logistic(vectors[positions], y)}
        timings[name] = time.perf_counter() - start
        print(f"Fitted fixed heads for {name}: {len(rows)} training ECGs", flush=True)
    fitted["ptb_baseline"] = {
        "hgb": joblib.load(input_root / CPU_PRIOR / "fitted_models.joblib")["models"]["hgb"],
        "xecg": joblib.load(input_root / PRIOR / "evaluate/fitted_heads.joblib")["xecg"],
    }
    directory = input_root / output / "evaluate"
    directory.mkdir(parents=True, exist_ok=False)
    records, thresholds = [], {}
    primary = None
    for site in ("sph", "ptb"):
        rows, scores, frame = evaluate_site(input_root, site, fitted, arms)
        records.extend(rows)
        if site == "sph":
            primary = primary_comparison(rows, scores, frame)
        for row in rows:
            thresholds.setdefault(row["model_id"], {})[site] = row["evaluation"]["calibration"]
        write_npz_atomic(directory / (site + "_predictions.npz"), **scores,
                         ecg_id=frame["ecg_id"].to_numpy(dtype=str),
                         patient_id=frame["patient_id"].to_numpy(dtype=str),
                         target=frame["target"].to_numpy(dtype=int),
                         evaluation=frame["evaluation"].to_numpy())
    models = {row["model_id"]: {"cohort": row["cohort"], "kind": row["kind"]} for row in records}
    artifact = {"schema_version": 1, "complete": True, "scope": "reused_development",
                "score_interpretation": "uncalibrated", "default_model_id": "ptb_baseline_blend",
                "operating_point": {"normal_budget": 0.05, "comparison": ">",
                                    "selection": "calibration_normals_only"},
                "models": aggregate_models(records), "primary": primary,
                "training": {name: {"records": len(rows), "positive": int(rows["target"].sum()),
                                     "normal": int((rows["target"] == 0).sum())}
                             for name, rows in arms.items()},
                "limitations": ["Reused PTB/SPH development data; no untouched clinical test",
                                "No Italian student validation or clinical disease probability calibration",
                                "Slider sweeps are descriptive; operating points remain frozen",
                                "Patient independence cannot be established for sources without patient IDs"]}
    bundle = {"schema_version": 1, "features": list(RAW_FEATURES), "fitted": fitted,
              "models": models, "thresholds": thresholds}
    joblib.dump(bundle, directory / "model_bundle.joblib")
    write_json_atomic(directory / "screening_evaluation.json", artifact)
    write_json_atomic(directory / "results.json", {"primary": primary, "fit_seconds": timings,
                                                   "training": artifact["training"]})
    return finish(input_root, output, "evaluate", receipt, hashes,
                  {"primary": primary, "fit_seconds": timings, "wall_seconds": time.perf_counter() - started,
                   "model_scores_computed_on_closed_challenge_or_echonext_test": 0})
