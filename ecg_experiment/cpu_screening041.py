"""Prospective CPU comparison on cached ECG measurements, with an integrity gate."""

from __future__ import annotations

import json
import platform
import subprocess
import time
from importlib.metadata import version
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from ecg_experiment import ROOT
from ecg_experiment.eda.ptbxl import (
    PTBXL_DIR,
    diagnostic_classes,
    load_metadata,
    load_statements,
    superclass_label,
)
from ecg_experiment.files import sha256_file, write_json_atomic, write_npz_atomic
from ecg_experiment.intervals import (
    metric_intervals,
    paired_auroc_difference,
    patient_groups,
    two_class_resamples,
)
from ecg_experiment.paths import to_stored

SEED = 20261018
DRAWS = 2000
SUPERCLASSES = ("NORM", "MI", "STTC", "CD", "HYP")
LEADS = ("I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6")
RAW_FEATURES = (
    "noise_drift_frac",
    "noise_hf_frac",
    "noise_powerline_frac",
    "sp_low_rel",
    "sp_mid_rel",
    "sp_high_rel",
    "sp_entropy",
    "sp_centroid_hz",
    *(f"{measurement}_{lead}" for lead in LEADS for measurement in ("std", "ptp", "kurt")),
    "n_beats",
    "beat_similarity",
    "rr_mean_ms",
    "rr_sdnn_ms",
    "rr_rmssd_ms",
    "rr_range_rel",
    "hr_bpm",
    "qrs_dur_ms",
    "qt_ms",
    "pq_peak_ms",
    "p_rms_mv",
    "t_peak_rms_mv",
    "pr_ms",
    "p_amp_II",
    *(
        f"{measurement}_{lead}"
        for lead in LEADS
        for measurement in ("r_amp", "s_depth", "q_depth", "qrs_area", "st", "t_amp")
    ),
)
RAW_PATH = ROOT / "outputs/avance2_fusion/caracteristicas_primarias_ee39c8f6f7.csv"
SELECTED_PATH = ROOT / "outputs/avance2_fusion/ptbxl_caracteristicas_seleccionadas.csv"
BASELINE_PATH = ROOT / "outputs/avance3_baseline/baseline_logistica.joblib"
METRICS_PATH = ROOT / "outputs/avance3_baseline/baseline_metricas.json"
LR_CONFIG = {"C": 0.1, "max_iter": 5000, "tol": 1e-8}
HGB_CONFIG = {
    "max_iter": 200,
    "learning_rate": 0.05,
    "max_leaf_nodes": 15,
    "min_samples_leaf": 30,
    "l2_regularization": 10,
    "early_stopping": False,
    "random_state": SEED,
}


def role_masks(table: pd.DataFrame) -> dict[str, np.ndarray]:
    """Validate unique ECGs and independent patients, then identify the three roles."""
    if table[["ecg_id", "patient_id", "strat_fold", "target"]].isna().any().any():
        raise ValueError("missing identity, fold, or label")
    if table.ecg_id.duplicated().any() or not table.strat_fold.isin(range(1, 10)).all():
        raise ValueError("ECG identities must be unique and folds must be 1-9")
    if table.groupby("patient_id").strat_fold.nunique().max() != 1:
        raise ValueError("patients cross folds")
    roles = {
        "fit": table.strat_fold.le(7).to_numpy(),
        "calibration": table.strat_fold.eq(8).to_numpy(),
        "development": table.strat_fold.eq(9).to_numpy(),
    }
    for name, mask in roles.items():
        if set(table.loc[mask, "target"]) != {0, 1}:
            raise ValueError(f"{name} must contain exactly both binary classes")
    return roles


def validate_labels(table: pd.DataFrame, metadata: pd.DataFrame, classes: dict[str, str]) -> None:
    """Require Avance labels, superclasses and identities to match raw PTB metadata."""
    metadata = metadata.loc[metadata.strat_fold.le(9)]
    expected = metadata.scp_codes.apply(lambda codes: superclass_label(codes, classes))
    if set(table.ecg_id) != set(metadata.index[expected.notna()]):
        raise ValueError("selected development identities differ from labeled raw metadata")
    aligned = metadata.loc[table.ecg_id]
    for name in ("patient_id", "strat_fold"):
        if not np.array_equal(table[name].to_numpy(), aligned[name].to_numpy()):
            raise ValueError(f"raw metadata mismatch: {name}")
    if not np.array_equal(table.target.to_numpy(), expected.loc[table.ecg_id].to_numpy()):
        raise ValueError("target must equal the existing superclass label")
    for name in SUPERCLASSES:
        flags = aligned.scp_codes.apply(lambda codes, name=name: name in {classes.get(c) for c in codes})
        if not np.array_equal(table[name].to_numpy(), flags.to_numpy(dtype=int)):
            raise ValueError(f"raw superclass flag mismatch: {name}")


def read_selected(path: Path = SELECTED_PATH) -> pd.DataFrame:
    """Read only folds 1-9 into the experiment table; no test scores are computed."""
    return pd.concat(
        [chunk.loc[chunk.strat_fold.le(9)] for chunk in pd.read_csv(path, chunksize=4096)],
        ignore_index=True,
    )


def raw_measurements(table: pd.DataFrame, path: Path = RAW_PATH) -> pd.DataFrame:
    """Align the exact 130 raw numeric measurements and turn nonfinite values into NaN."""
    columns = pd.read_csv(path, nrows=0).columns.tolist()
    expected = ["ecg_id", "status", "implausible", "flat_leads", *RAW_FEATURES]
    if columns != expected or len(RAW_FEATURES) != 130:
        raise ValueError("raw schema must contain only the frozen 130 measurements and four identifiers")
    identifiers = set(table.ecg_id)
    chunks = [chunk.loc[chunk.ecg_id.isin(identifiers)] for chunk in pd.read_csv(path, chunksize=4096)]
    raw = pd.concat(chunks, ignore_index=True)
    if raw.ecg_id.duplicated().any() or set(raw.ecg_id) != identifiers:
        raise ValueError("raw ECG identities are missing or duplicated")
    values = (
        raw.set_index("ecg_id").loc[table.ecg_id, list(RAW_FEATURES)].apply(pd.to_numeric, errors="raise")
    )
    values = values.astype(float).replace([np.inf, -np.inf], np.nan)
    values.index = table.index
    return values


def baseline_integrity(
    table: pd.DataFrame, model_path: Path = BASELINE_PATH, metrics_path: Path = METRICS_PATH
) -> dict[str, Any]:
    """Reproduce saved and refitted fold-9 baseline scores before any prospective fit."""
    saved = joblib.load(model_path)
    receipt = json.loads(metrics_path.read_text())
    features = saved["features"]
    if features != receipt["caracteristicas"] or saved["C"] != 0.1:
        raise ValueError("baseline artifact and receipt feature order or C disagree")
    x = table[features]
    train = table.strat_fold.le(8).to_numpy()
    development = table.strat_fold.eq(9).to_numpy()
    y = table.target.to_numpy(dtype=int)
    saved_prob = saved["model"].predict_proba(x.loc[development])[:, 1]
    model = LogisticRegression(C=0.1, max_iter=5000).fit(x.loc[train], y[train])
    refit_prob = model.predict_proba(x.loc[development])[:, 1]
    if not np.allclose(saved_prob, refit_prob, atol=1e-12, rtol=0):
        raise ValueError("baseline saved/refitted predictions fail absolute tolerance 1e-12")
    observed = {}
    for name, scorer in (("AUROC", roc_auc_score), ("AUPRC", average_precision_score)):
        observed[name] = float(scorer(y[development], refit_prob))
        historical = float(receipt["validacion"][name])
        saved_score = float(scorer(y[development], saved_prob))
        if abs(observed[name] - historical) > 1e-12 or abs(saved_score - historical) > 1e-12:
            raise ValueError(f"baseline {name} fails historical tolerance 1e-12")
    return {
        "status": "passed",
        "validation": observed,
        "prediction_max_absolute_difference": float(np.max(np.abs(saved_prob - refit_prob))),
        "absolute_tolerance": 1e-12,
        "train_folds": list(range(1, 9)),
        "validation_fold": 9,
        "provenance_limit": (
            "Original recipe/notebook absent from current branch; notebook is in git avance3/baseline."
        ),
        "features": features,
    }


def calibrate_threshold(y: np.ndarray, probability: np.ndarray) -> dict[str, float | int | str]:
    """Freeze a conservative 5% normal referral threshold using the exact order statistic."""
    normals = np.asarray(probability)[np.asarray(y) == 0]
    if not len(normals) or not np.isfinite(normals).all():
        raise ValueError("calibration requires finite normal scores")
    allowed = int(np.floor(0.05 * len(normals)))
    threshold = float(np.sort(normals)[::-1][allowed])
    referred = int(np.count_nonzero(normals > threshold))
    return {
        "threshold": threshold,
        "normal_count": len(normals),
        "allowed_count": allowed,
        "referred_normal_count": referred,
        "normal_referral_rate": referred / len(normals),
        "comparator": ">",
        "order_index_descending": allowed,
    }


def sensitivity_difference(
    patients: np.ndarray,
    y: np.ndarray,
    first: np.ndarray,
    second: np.ndarray,
    draws: int = DRAWS,
    seed: int = SEED,
) -> dict[str, float | int]:
    """Bootstrap the paired sensitivity difference of already frozen referral decisions."""
    groups = patient_groups(patients)
    rng = np.random.default_rng(seed)
    values = [
        float(np.mean(first[rows][y[rows] == 1]) - np.mean(second[rows][y[rows] == 1]))
        for rows in two_class_resamples(groups, y, draws, rng)
    ]
    if not values:
        raise ValueError("all sensitivity bootstrap draws had a single class")
    low, high = np.percentile(values, [2.5, 97.5])
    return {
        "difference": float(np.mean(first[y == 1]) - np.mean(second[y == 1])),
        "ci_low": float(low),
        "ci_high": float(high),
        "draws": draws,
        "skipped_draws": draws - len(values),
        "seed": seed,
    }


def decision(auroc: dict[str, Any], sensitivity: float, normal_rate: float) -> dict[str, Any]:
    """Apply every prespecified development gate without rounding."""
    gates = {
        "auroc_gain_at_least_0.01": auroc["difference"] >= 0.01,
        "auroc_ci_low_positive": auroc["ci_low"] > 0,
        "sensitivity_difference_at_least_minus_0.02": sensitivity >= -0.02,
        "hgb_normal_referral_at_most_0.08": normal_rate <= 0.08,
    }
    return {"outcome": "advance_development" if all(gates.values()) else "do_not_advance", "gates": gates}


def arm_summary(
    table: pd.DataFrame,
    probability: np.ndarray,
    threshold: float,
    draws: int = DRAWS,
) -> dict[str, Any]:
    """Summarize fixed strict-threshold metrics and exploratory superclass misses."""
    y = table.target.to_numpy(dtype=int)
    summary = metric_intervals(
        y,
        probability,
        table.patient_id.to_numpy(),
        np.nextafter(threshold, np.inf),
        draws=draws,
        seed=SEED,
    )
    summary["threshold"] = threshold
    summary["comparator"] = ">"
    specificity = summary["metrics"]["specificity"]
    summary["normal_referral_rate"] = {
        "value": float(np.mean(probability[y == 0] > threshold)),
        "ci_low": 1 - specificity["ci_high"],
        "ci_high": 1 - specificity["ci_low"],
    }
    missed = (y == 1) & ~(probability > threshold)
    summary["exploratory_superclass_misses"] = {
        name: {
            "positive_count": int(np.count_nonzero((y == 1) & table[name].eq(1).to_numpy())),
            "missed_count": int(np.count_nonzero(missed & table[name].eq(1).to_numpy())),
        }
        for name in SUPERCLASSES[1:]
    }
    return summary


def fit_arms(values: pd.DataFrame, y: np.ndarray, fit: np.ndarray) -> tuple[dict[str, Any], dict[str, float]]:
    """Fit each fixed model once using only the primary fit role."""
    if values.loc[fit].isna().all().any():
        raise ValueError("HGB native NaN cannot fit an entirely missing column in installed scikit-learn")
    models = {
        "logistic": make_pipeline(
            SimpleImputer(strategy="median", add_indicator=True, keep_empty_features=True),
            StandardScaler(),
            LogisticRegression(**LR_CONFIG),
        ),
        "hgb": HistGradientBoostingClassifier(**HGB_CONFIG),
    }
    timings = {}
    for name, model in models.items():
        start = time.monotonic()
        model.fit(values.loc[fit], y[fit])
        timings[name] = time.monotonic() - start
    return models, timings


def provenance(protocol: Path) -> dict[str, Any]:
    """Pin inputs, executable sources, protocol and Git revision with portable paths."""
    files = [
        RAW_PATH,
        SELECTED_PATH,
        BASELINE_PATH,
        METRICS_PATH,
        PTBXL_DIR / "ptbxl_database.csv",
        PTBXL_DIR / "scp_statements.csv",
        protocol,
        Path(__file__),
        ROOT / "scripts/experiments/run_cpu_screening041.py",
        ROOT / "ecg_experiment/intervals.py",
        ROOT / "ecg_experiment/paths.py",
        ROOT / "ecg_experiment/files.py",
        ROOT / "ecg_experiment/eda/ptbxl.py",
        ROOT / "pyproject.toml",
        ROOT / "uv.lock",
    ]
    return {
        "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "files_sha256": {to_stored(path): sha256_file(path) for path in files},
        "protocol": to_stored(protocol),
        "versions": {
            "python": platform.python_version(),
            **{
                name: version(name)
                for name in ("numpy", "pandas", "scipy", "scikit-learn", "joblib", "threadpoolctl")
            },
        },
    }


def run_experiment(output: Path, protocol: Path, threads: int = 4) -> dict[str, Any]:
    """Execute the integrity-gated comparison and save local reproducible artifacts."""
    if not 1 <= threads <= 4:
        raise ValueError("threads must be between 1 and 4")
    output.mkdir(parents=True, exist_ok=False)
    start = time.monotonic()
    try:
        with threadpool_limits(limits=threads):
            result = _run(output, protocol, threads)
        result["total_seconds"] = time.monotonic() - start
        write_json_atomic(output / "result.json", result)
        return result
    except Exception as error:
        failure = {
            "status": "failed",
            "error_type": type(error).__name__,
            "error": str(error).replace(str(ROOT), "."),
            "elapsed_seconds": time.monotonic() - start,
            "test_scoring": False,
        }
        write_json_atomic(output / "failure.json", failure)
        raise


def _run(output: Path, protocol: Path, threads: int) -> dict[str, Any]:
    """Run validated roles, keeping the historical integrity gate before primary fitting."""
    sources = provenance(protocol)
    table = read_selected()
    roles = role_masks(table)
    validate_labels(table, load_metadata(), diagnostic_classes(load_statements()))
    with threadpool_limits(limits=1):
        integrity = baseline_integrity(table)
    integrity["threads"] = 1
    write_json_atomic(output / "integrity.json", integrity)
    values = raw_measurements(table)
    y = table.target.to_numpy(dtype=int)
    models, fit_seconds = fit_arms(values, y, roles["fit"])
    thresholds, summaries, predictions = {}, {}, {}
    timings = {"fit_seconds": fit_seconds, "predict_seconds": {}}
    evaluation = roles["calibration"] | roles["development"]
    evaluation_table = table.loc[evaluation].reset_index(drop=True)
    development = evaluation_table.strat_fold.eq(9).to_numpy()
    calibration = ~development
    for name, model in models.items():
        start = time.monotonic()
        probabilities = model.predict_proba(values.loc[evaluation])[:, 1]
        timings["predict_seconds"][name] = time.monotonic() - start
        thresholds[name] = calibrate_threshold(
            evaluation_table.target.to_numpy()[calibration], probabilities[calibration]
        )
        predictions[name] = probabilities
        summaries[name] = arm_summary(
            evaluation_table.loc[development], probabilities[development], thresholds[name]["threshold"]
        )
    dev_table = evaluation_table.loc[development]
    dev_y = dev_table.target.to_numpy(dtype=int)
    dev_patients = dev_table.patient_id.to_numpy()
    delta = paired_auroc_difference(
        dev_patients,
        dev_y,
        predictions["hgb"][development],
        predictions["logistic"][development],
        draws=DRAWS,
        seed=SEED,
    )
    sensitivity = sensitivity_difference(
        dev_patients,
        dev_y,
        predictions["hgb"][development] > thresholds["hgb"]["threshold"],
        predictions["logistic"][development] > thresholds["logistic"]["threshold"],
    )
    joblib.dump(
        {
            "models": models,
            "features": list(RAW_FEATURES),
            "thresholds": thresholds,
            "comparator": ">",
            "protocol": to_stored(protocol),
        },
        output / "fitted_models.joblib",
    )
    write_npz_atomic(
        output / "predictions.npz",
        ecg_id=evaluation_table.ecg_id.to_numpy(),
        patient_id=evaluation_table.patient_id.to_numpy(),
        strat_fold=evaluation_table.strat_fold.to_numpy(),
        target=evaluation_table.target.to_numpy(dtype=int),
        **predictions,
        **{name: evaluation_table[name].to_numpy(dtype=int) for name in SUPERCLASSES},
    )
    return {
        "status": "completed",
        "experiment": "041",
        "provenance": sources,
        "integrity": integrity,
        "config": {
            "threads": threads,
            "seed": SEED,
            "bootstrap_draws": DRAWS,
            "fit_folds": list(range(1, 8)),
            "calibration_fold": 8,
            "development_fold": 9,
            "logistic": LR_CONFIG,
            "hgb": HGB_CONFIG,
            "features": list(RAW_FEATURES),
            "imputer": {"strategy": "median", "add_indicator": True, "keep_empty_features": True},
            "scaler": "StandardScaler",
            "normal_referral_budget": 0.05,
            "nonfinite_policy": "NaN; no additional clipping",
            "test_scoring": False,
        },
        "counts": {
            name: {
                "ecgs": int(mask.sum()),
                "patients": int(table.loc[mask, "patient_id"].nunique()),
                "positive": int(y[mask].sum()),
                "normal": int(np.count_nonzero(y[mask] == 0)),
            }
            for name, mask in roles.items()
        },
        "thresholds": thresholds,
        "development": summaries,
        "paired_auroc_hgb_minus_logistic": delta,
        "paired_sensitivity_hgb_minus_logistic": sensitivity,
        "decision": decision(
            delta, sensitivity["difference"], summaries["hgb"]["normal_referral_rate"]["value"]
        ),
        "model_fit_status": {
            "logistic_iterations": models["logistic"].named_steps["logisticregression"].n_iter_.tolist(),
            "logistic_reached_iteration_cap": bool(
                np.any(models["logistic"].named_steps["logisticregression"].n_iter_ >= LR_CONFIG["max_iter"])
            ),
            "hgb_iterations": int(models["hgb"].n_iter_),
        },
        "timing": timings,
        "artifact_sha256": {
            to_stored(output / name): sha256_file(output / name)
            for name in ("fitted_models.joblib", "predictions.npz", "integrity.json")
        },
        "artifacts": {
            name: to_stored(output / name)
            for name in ("fitted_models.joblib", "predictions.npz", "integrity.json", "result.json")
        },
    }
