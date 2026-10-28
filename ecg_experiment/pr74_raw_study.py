"""Run the fixed corrective comparison with fresh raw-derived dependencies."""

from __future__ import annotations

import gc
import hashlib
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import HistGradientBoostingClassifier

from . import ROOT
from .age_cohort_inputs import canonical_integral_id, parse_boolean
from .cpu_screening041 import HGB_CONFIG, RAW_FEATURES
from .cuda_runtime20261028 import configure_cuda_runtime
from .external_encoders import xecg_features, xecg_input
from .files import sha256_file, write_json_atomic, write_npz_atomic
from .full_development import fit_logistic
from .gpu import gpu_lock
from .pr74_fresh_baseline import finish_stage, source_binding, stage_inputs
from .pr74_screening_features import checkpoint_hashes, load_encoder, measurements, read_training_signal
from .pr74_screening_io import SPLIT_SHA256, completed, read_json, relative, stored, verify_hashes
from .pr74_screening_metrics import budget_threshold, point_metrics, threshold_sweep
from .pr74_screening_study import (
    aggregate_models,
    assert_evaluation_independence,
    primary_comparison,
    score_models,
)

COLD = Path("outputs/cohort25_recovery_20261028/cold_start")
COHORTS = COLD / "cohorts_25k"
STRICT = COLD / "strict_matched"
OUTPUT = Path("outputs/pr74_from_raw_validation_20261028")
PROTOCOL = Path("docs/pr74-raw-study-protocol-2026-10-28.md")


def study_binding() -> dict[str, Any]:
    """Capture all committed successor code before touching numerical inputs."""
    receipt = source_binding()
    paths = [
        ROOT / PROTOCOL,
        Path(__file__),
        ROOT / "scripts/experiments/run_pr74_raw_study.py",
        ROOT / "ecg_experiment/pr74_raw_inference.py",
    ]
    paths += sorted((ROOT / "tests").glob("test_pr74_raw_study*.py"))
    for path in paths:
        name = stored(path.relative_to(ROOT))
        committed = subprocess.check_output(["git", "show", "HEAD:" + name], cwd=ROOT)
        if hashlib.sha256(committed).hexdigest() != sha256_file(path):
            raise ValueError("Raw-study scientific source is not committed")
        receipt["source_sha256"][name] = sha256_file(path)
    receipt["protocol"] = {"path": str(PROTOCOL), "sha256": sha256_file(ROOT / PROTOCOL)}
    return receipt


def evidence(input_root: Path, directory: Path, scope: str) -> dict[str, str]:
    """Verify a completed new cohort receipt and its actual provenance fields."""
    path = relative(directory) / "receipt.json"
    receipt = read_json(input_root / path)
    if receipt.get("complete") is not True or receipt.get("scope") != scope:
        raise ValueError("Completed fresh cohort evidence required")
    if scope == "new_supervised_age_label_cohorts" and receipt.get("original_split_sha256") != SPLIT_SHA256:
        raise ValueError("Fresh cohorts must preserve the original Challenge split")
    hashes = {stored(path): sha256_file(input_root / path), **receipt["artifacts"], **receipt["input_sha256"]}
    verify_hashes(input_root, hashes)
    for key in ("source_sha256", "recipe_sha256"):
        verify_hashes(ROOT, receipt.get(key, {}))
    return hashes


def validate_manifest(rows: pd.DataFrame) -> pd.DataFrame:
    """Require explicit clean original-training roles and unique canonical identities."""
    if (
        not rows["split"].eq("train").all()
        or not parse_boolean(rows["quality_passed"]).all()
        or not rows["samples"].eq("5000").all()
        or not rows["window_start"].eq("0").all()
        or rows["record_id"].duplicated().any()
        or rows["signal_sha256"].duplicated().any()
    ):
        raise ValueError("Fresh manifest admission or unique identity gate failed")
    rows = rows.copy()
    target = pd.to_numeric(rows["target"], errors="raise")
    if not target.isin([0, 1]).all():
        raise ValueError("Fresh manifest target is not binary")
    rows["target"] = target.astype(int)
    return rows


def subset(rows: pd.DataFrame, pool: pd.DataFrame) -> pd.DataFrame:
    """Require every selected identity and admission field to match the verified candidate union."""
    rows = validate_manifest(rows)
    candidates = pool.set_index("record_id")
    if not set(rows["record_id"]) <= set(candidates.index):
        raise ValueError("Selected cohort leaves the fresh admitted candidate pool")
    selected = candidates.loc[rows["record_id"]]
    for key in ("source", "target", "label_evidence_rank", "signal_sha256", "patient_id", "split"):
        if rows[key].astype(str).tolist() != selected[key].astype(str).tolist():
            raise ValueError(f"Fresh selected admission changed: {key}")
    return rows


def cohorts(
    input_root: Path, broad: Path, strict: Path
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame], dict]:
    """Load explicit new cohort paths without historical cached role dependencies."""
    hashes = evidence(input_root, broad, "new_supervised_age_label_cohorts")
    hashes.update(evidence(input_root, strict, "strict_young_matched_cohorts"))

    def read(directory: Path, filename: str) -> pd.DataFrame:
        return pd.read_csv(input_root / directory / filename, dtype=str, keep_default_na=False)

    pool = (
        validate_manifest(read(broad, "candidates.csv"))
        .sort_values("record_id", kind="stable")
        .reset_index(drop=True)
    )
    names = {
        "strict_young": (strict, "strict_18_25.csv"),
        "matched_general": (strict, "general_matched_size.csv"),
        "younger": (broad, "younger.csv"),
        "all_age": (broad, "all_age.csv"),
    }
    arms = {name: subset(read(directory, filename), pool) for name, (directory, filename) in names.items()}
    first, second = arms["strict_young"], arms["matched_general"]
    keys = ["source", "label_evidence_rank", "target"]
    if (
        len(first) != 770
        or not first.groupby(keys).size().equals(second.groupby(keys).size())
        or not pd.to_numeric(first["age"], errors="raise").between(18, 25).all()
    ):
        raise ValueError("Primary fresh cohorts must preserve all 770 exact matching strata")
    return pool, arms, hashes


def challenge_rows(pool: pd.DataFrame) -> pd.DataFrame:
    """Exclude PTB reusable fresh baseline features from the new Challenge extraction union."""
    return pool.loc[pool["source"].ne("ptbxl")].reset_index(drop=True)


def profile(input_root: Path, output: Path, broad: Path, strict: Path) -> dict[str, Any]:
    """Profile the actual admitted raw training path before complete cold extraction."""
    receipt = study_binding()
    runtime = configure_cuda_runtime()
    pool, _, hashes = cohorts(input_root, broad, strict)
    hashes.update(checkpoint_hashes(input_root))
    rows = challenge_rows(pool)
    selected = rows.groupby("source", sort=True).head(16)
    directory = input_root / relative(output) / "profile"
    directory.mkdir(parents=True, exist_ok=False)
    with gpu_lock("cuda", blocking=False), torch.cuda.device(0):
        torch.set_num_threads(2)
        free, _ = torch.cuda.mem_get_info()
        if free < 4 * 1024**3:
            raise RuntimeError("Less than four GiB free for raw training profile")
        torch.cuda.reset_peak_memory_stats()
        model = load_encoder(input_root)
        started = time.perf_counter()
        signals = [read_training_signal(input_root, row) for _, row in selected.iterrows()]
        raw = measurements(signals)
        vectors = xecg_features(model, np.stack([xecg_input(signal) for signal in signals]))
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - started
        peak = torch.cuda.max_memory_allocated()
        del model
        gc.collect()
        torch.cuda.empty_cache()
    forecast = 2 * elapsed / len(selected) * len(rows) + 300
    gates = {
        "free_at_least_4gib": free >= 4 * 1024**3,
        "peak_under_3gib": peak < 3 * 1024**3,
        "forecast_under_2hours": forecast < 7200,
        "output_under_1gib": len(rows) * 5136 < 1024**3,
    }
    selected.to_csv(directory / "rows.csv", index=False)
    raw.to_csv(directory / "measurements.csv", index=False)
    np.save(directory / "xecg.npy", vectors)
    details = {
        "records": len(rows),
        "profile_records": len(selected),
        "wall_seconds": elapsed,
        "peak_gpu_bytes": peak,
        "initial_free_gpu_bytes": free,
        "forecast_seconds": forecast,
        "resource_gates": gates,
        "cuda_runtime": runtime,
    }
    if not all(gates.values()):
        raise RuntimeError("Raw training profile resource gate failed")
    return finish_stage(input_root, output, "profile", receipt, hashes, details)


def matching_extraction(rows: pd.DataFrame, expected: pd.DataFrame) -> None:
    """Reject cached feature rows from any differing source role or canonical training view."""
    for key in ("record_id", "signal_sha256", "target", "split", "source", "window_start", "samples"):
        if rows[key].astype(str).tolist() != expected[key].astype(str).tolist():
            raise ValueError(f"Training feature extraction differs from fresh candidates: {key}")


def extract(
    input_root: Path, output: Path, broad: Path, strict: Path, reuse: Path | None = None
) -> dict[str, Any]:
    """Extract raw admitted Challenge rows, or explicitly validate a verified warm extraction."""
    receipt = study_binding()
    pool, _, hashes = cohorts(input_root, broad, strict)
    rows = challenge_rows(pool)
    directory = input_root / relative(output) / "extract"
    if reuse is not None:
        previous, bound = completed(input_root, relative(reuse) / "extract")
        hashes.update(bound)
        hashes.update(previous["input_sha256"])
        verify_hashes(ROOT, previous["source_sha256"])
        verify_hashes(input_root, hashes)
        source = input_root / reuse / "extract"
        cached = pd.read_csv(source / "rows.csv", dtype=str, keep_default_na=False)
        matching_extraction(cached, rows)
        directory.mkdir(parents=True, exist_ok=False)
        for name in ("rows.csv", "measurements.csv", "xecg.npy"):
            shutil.copyfile(source / name, directory / name)
        return finish_stage(
            input_root,
            output,
            "extract",
            receipt,
            hashes,
            {
                "records": len(rows),
                "execution_mode": "explicit_verified_warm_reuse",
                "reused_extraction": str(reuse),
                "encoder_fitted": False,
            },
        )
    configure_cuda_runtime()
    profiled, bound = stage_inputs(input_root, output, "profile")
    hashes.update(bound)
    if not all(profiled["resource_gates"].values()):
        raise ValueError("Passing raw training resource profile required")
    hashes.update(checkpoint_hashes(input_root))
    directory.mkdir(parents=True, exist_ok=False)
    rows.to_csv(directory / "rows.csv", index=False)
    vectors = np.lib.format.open_memmap(
        directory / "xecg.npy", mode="w+", dtype=np.float32, shape=(len(rows), 1024)
    )
    raw = np.full((len(rows), 130), np.nan)
    started = time.perf_counter()
    with gpu_lock("cuda", blocking=False), torch.cuda.device(0):
        torch.set_num_threads(2)
        model = load_encoder(input_root)
        for start in range(0, len(rows), 16):
            if time.perf_counter() - started > 7200:
                raise TimeoutError("Raw training extraction exceeded two hours")
            part = rows.iloc[start : start + 16]
            signals = [read_training_signal(input_root, row) for _, row in part.iterrows()]
            raw[start : start + len(part)] = measurements(signals).to_numpy()
            vectors[start : start + len(part)] = xecg_features(
                model, np.stack([xecg_input(x) for x in signals])
            )
            if start % 512 == 0:
                vectors.flush()
                write_json_atomic(
                    directory / "progress.json",
                    {
                        "records": start + len(part),
                        "total": len(rows),
                        "wall_seconds": time.perf_counter() - started,
                    },
                )
        vectors.flush()
        del model
        gc.collect()
        torch.cuda.empty_cache()
    if not np.isfinite(vectors).all():
        raise ValueError("Nonfinite raw training encoder features")
    pd.DataFrame(raw, columns=RAW_FEATURES).to_csv(directory / "measurements.csv", index=False)
    return finish_stage(
        input_root,
        output,
        "extract",
        receipt,
        hashes,
        {
            "records": len(rows),
            "execution_mode": "fresh_raw_extraction",
            "encoder_fitted": False,
            "wall_seconds": time.perf_counter() - started,
        },
    )


def compare_aggregates(fresh: dict, published: dict) -> dict[str, Any]:
    """Check all reproduced points without claiming recovery of historical serialized bytes."""

    def points(artifact: dict) -> dict:
        return {
            (model["id"], evaluation["pool"]): evaluation
            for model in artifact["models"]
            for evaluation in model["evaluations"]
        }

    first, second = points(fresh), points(published)
    if first.keys() != second.keys():
        raise ValueError("Fresh aggregate model/site inventory differs from published evaluation")
    maximum = 0.0
    threshold_difference = 0.0
    for key, point in first.items():
        expected = second[key]
        for count in ("tp", "tn", "fp", "fn"):
            if point["metrics"][count] != expected["metrics"][count]:
                raise ValueError(f"Fresh confusion matrix differs: {key}/{count}")
        maximum = max(maximum, abs(point["metrics"]["auroc"] - expected["metrics"]["auroc"]))
        threshold_difference = max(threshold_difference, abs(point["threshold"] - expected["threshold"]))
        if point["calibration"]["flagged_normals"] != expected["calibration"]["flagged_normals"]:
            raise ValueError("Fresh frozen calibration count differs")
    if (
        maximum > 1e-9
        or fresh["primary"]["development_advantage"] != published["primary"]["development_advantage"]
    ):
        raise ValueError("Fresh AUROC or primary decision fails scientific parity")
    return {
        "scientific_parity_passed": True,
        "model_site_points": len(first),
        "all_confusion_matrices_exact": True,
        "maximum_absolute_auroc_difference": maximum,
        "maximum_absolute_threshold_difference": threshold_difference,
        "historical_serialized_artifacts_recovered": False,
    }


def verify(input_root: Path, output: Path) -> dict[str, Any]:
    """Bind numerical parity against the already published aggregate without extra model scoring."""
    receipt = study_binding()
    _, hashes = stage_inputs(input_root, output, "evaluate")
    published = Path("reports/pr74-corrective-evaluation-2026-10-28/aggregate.json")
    receipt["source_sha256"][str(published)] = sha256_file(ROOT / published)
    fresh = read_json(input_root / output / "evaluate/screening_evaluation.json")
    details = compare_aggregates(fresh, read_json(ROOT / published))
    directory = input_root / relative(output) / "verify"
    directory.mkdir(parents=True, exist_ok=False)
    write_json_atomic(directory / "parity.json", details)
    return finish_stage(input_root, output, "verify", receipt, hashes, details)


def feature_matrix(
    input_root: Path, output: Path, baseline: Path, pool: pd.DataFrame
) -> tuple[pd.DataFrame, np.ndarray]:
    """Align fresh PTB features and admitted Challenge features without any old cache path."""
    source = input_root / output / "extract"
    rows = pd.read_csv(source / "rows.csv", dtype=str, keep_default_na=False)
    expected = challenge_rows(pool)
    matching_extraction(rows, expected)
    raw = pd.DataFrame(np.nan, index=pool["record_id"], columns=RAW_FEATURES)
    vectors = np.empty((len(pool), 1024), dtype=np.float32)
    ids = pd.Index(pool["record_id"])
    raw.loc[rows["record_id"]] = pd.read_csv(source / "measurements.csv").to_numpy()
    vectors[ids.get_indexer(rows["record_id"])] = np.load(source / "xecg.npy", mmap_mode="r")
    fresh = input_root / baseline / "features"
    ptb = pd.read_csv(fresh / "ptb_rows.csv", dtype=str, keep_default_na=False)
    ptb_ids = pd.Index("ptbxl:" + ptb["ecg_id"].map(canonical_integral_id))
    admitted = pool.loc[pool["source"].eq("ptbxl")]
    indices = ptb_ids.get_indexer(admitted["record_id"])
    if (indices < 0).any():
        raise ValueError("Fresh PTB features omit an admitted training identity")
    selected = ptb.iloc[indices]
    if (
        not parse_boolean(selected["fit"]).all()
        or selected["signal_sha256"].tolist() != admitted["signal_sha256"].tolist()
        or selected["target"].tolist() != admitted["target"].astype(str).tolist()
    ):
        raise ValueError("Fresh PTB training features changed roles, windows or labels")
    raw.loc[admitted["record_id"]] = pd.read_csv(fresh / "ptb_measurements.csv").iloc[indices].to_numpy()
    vectors[ids.get_indexer(admitted["record_id"])] = np.load(fresh / "ptb_xecg.npy", mmap_mode="r")[indices]
    if raw.shape != (len(pool), 130) or not np.isfinite(vectors).all():
        raise ValueError("Fresh training features violate the fixed feature schema")
    return raw, vectors


def evaluation_features(
    input_root: Path, baseline: Path, site: str
) -> tuple[pd.DataFrame, np.ndarray, pd.DataFrame]:
    """Read fresh development inputs with their explicit patient and calibration roles."""
    source = input_root / baseline / "features"
    rows = pd.read_csv(source / (site + "_rows.csv"), dtype=str, keep_default_na=False)
    admitted = ~parse_boolean(rows["fit"]).to_numpy()
    frame = rows.loc[admitted].copy().reset_index(drop=True)
    frame["evaluation"] = parse_boolean(frame["evaluation"]).to_numpy()
    frame["target"] = pd.to_numeric(frame["target"]).astype(int)
    frame["patient_id"] = frame["patient_key"]
    raw = pd.read_csv(source / (site + "_measurements.csv")).loc[admitted].reset_index(drop=True)
    vectors = np.load(source / (site + "_xecg.npy"), mmap_mode="r")[admitted]
    return frame, vectors, raw


def evaluate_site(
    input_root: Path, baseline: Path, site: str, fitted: dict, arms: dict
) -> tuple[list[dict], dict, pd.DataFrame]:
    """Calibrate each fixed model only on site-specific normal calibration rows."""
    frame, vectors, raw = evaluation_features(input_root, baseline, site)
    assert_evaluation_independence(arms, site, frame)
    y, held = frame["target"].to_numpy(dtype=int), frame["evaluation"].to_numpy()
    records, arrays = [], {}
    for cohort, model in fitted.items():
        for kind, scores in score_models(model, vectors, raw).items():
            model_id = cohort + "_" + kind
            calibrated = budget_threshold(y[~held], scores[~held])
            threshold = calibrated["threshold"]
            evaluation = {
                "pool": site,
                "label": site.upper() + " reused development",
                "threshold": threshold,
                "comparison": ">",
                "calibration": calibrated,
                "metrics": point_metrics(y[held], scores[held], threshold),
                "sweep": threshold_sweep(y[held], scores[held], threshold),
            }
            records.append({"model_id": model_id, "cohort": cohort, "kind": kind, "evaluation": evaluation})
            arrays[model_id] = scores
    return records, arrays, frame


def evaluate(input_root: Path, output: Path, broad: Path, strict: Path, baseline: Path) -> dict[str, Any]:
    """Refit the registered four arms using solely verified fresh scientific baseline inputs."""
    receipt = study_binding()
    started = time.perf_counter()
    pool, arms, hashes = cohorts(input_root, broad, strict)
    _, bound = stage_inputs(input_root, output, "extract")
    hashes.update(bound)
    _, bound = stage_inputs(input_root, baseline, "features")
    hashes.update(bound)
    parity, bound = stage_inputs(input_root, baseline, "fit")
    hashes.update(bound)
    if parity.get("scientific_parity_passed") is not True:
        raise ValueError("Fresh baseline scientific parity must pass before corrective scoring")
    raw, vectors = feature_matrix(input_root, output, baseline, pool)
    fitted, timings = {}, {}
    for name, rows in arms.items():
        positions = pd.Index(pool["record_id"]).get_indexer(rows["record_id"])
        start = time.perf_counter()
        y = rows["target"].to_numpy(dtype=int)
        fitted[name] = {
            "hgb": HistGradientBoostingClassifier(**HGB_CONFIG).fit(raw.iloc[positions], y),
            "xecg": fit_logistic(vectors[positions], y),
        }
        timings[name] = time.perf_counter() - start
    fitted["ptb_baseline"] = joblib.load(input_root / baseline / "fit/fitted_models.joblib")
    directory = input_root / relative(output) / "evaluate"
    directory.mkdir(parents=True, exist_ok=False)
    records, thresholds, primary = [], {}, None
    for site in ("sph", "ptb"):
        rows, scores, frame = evaluate_site(input_root, baseline, site, fitted, arms)
        records.extend(rows)
        if site == "sph":
            primary = primary_comparison(rows, scores, frame)
        for row in rows:
            thresholds.setdefault(row["model_id"], {})[site] = row["evaluation"]["calibration"]
        write_npz_atomic(
            directory / (site + "_predictions.npz"),
            **scores,
            ecg_id=frame["ecg_id"].to_numpy(dtype=str),
            patient_id=frame["patient_id"].to_numpy(dtype=str),
            target=frame["target"].to_numpy(dtype=int),
            evaluation=frame["evaluation"].to_numpy(),
        )
    models = {row["model_id"]: {"cohort": row["cohort"], "kind": row["kind"]} for row in records}
    artifact = {
        "schema_version": 1,
        "complete": True,
        "scope": "reused_development",
        "score_interpretation": "uncalibrated",
        "default_model_id": "ptb_baseline_blend",
        "operating_point": {
            "normal_budget": 0.05,
            "comparison": ">",
            "selection": "calibration_normals_only",
        },
        "models": aggregate_models(records),
        "primary": primary,
        "training": {
            name: {
                "records": len(rows),
                "positive": int(rows["target"].sum()),
                "normal": int(rows["target"].eq(0).sum()),
            }
            for name, rows in arms.items()
        },
        "limitations": [
            "Reused PTB/SPH development data; no untouched clinical test",
            "No Italian student validation or clinical probability calibration",
            "Slider sweeps are descriptive; operating points remain frozen",
            "Patient independence unknown for Challenge sources without patient IDs",
        ],
    }
    joblib.dump(
        {
            "schema_version": 1,
            "features": list(RAW_FEATURES),
            "fitted": fitted,
            "models": models,
            "thresholds": thresholds,
        },
        directory / "model_bundle.joblib",
    )
    write_json_atomic(directory / "screening_evaluation.json", artifact)
    write_json_atomic(
        directory / "results.json",
        {"primary": primary, "fit_seconds": timings, "training": artifact["training"]},
    )
    return finish_stage(
        input_root,
        output,
        "evaluate",
        receipt,
        hashes,
        {
            "primary": primary,
            "fit_seconds": timings,
            "wall_seconds": time.perf_counter() - started,
            "model_scores_computed_on_closed_challenge_or_echonext_test": 0,
        },
    )
