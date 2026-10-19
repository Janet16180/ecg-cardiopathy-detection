"""Run the unchanged Experiment 041 comparison on the verified historical cohort."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from ecg_experiment import ROOT
from ecg_experiment import cpu_screening041 as original
from ecg_experiment.files import sha256_file, write_json_atomic, write_npz_atomic
from ecg_experiment.paths import to_stored

AMENDMENT_PATH = ROOT / "docs/experiment-041-cpu-screening-execution-v2.md"


def read_quality(metadata: pd.DataFrame, path: Path = original.RAW_PATH) -> pd.DataFrame:
    """Read cache quality fields for development records without loading test measurements."""
    ids = set(metadata.index[metadata.strat_fold.le(9)])
    chunks = pd.read_csv(path, usecols=["ecg_id", "status", "flat_leads"], chunksize=4096)
    return pd.concat([chunk.loc[chunk.ecg_id.isin(ids)] for chunk in chunks], ignore_index=True)


def validate_historical_cohort(
    table: pd.DataFrame,
    metadata: pd.DataFrame,
    quality: pd.DataFrame,
    classes: dict[str, str],
) -> dict[str, Any]:
    """Require exactly diagnostic-labeled records without the historical flat-lead exclusion."""
    metadata = metadata.loc[metadata.strat_fold.le(9)]
    if quality.ecg_id.duplicated().any() or set(quality.ecg_id) != set(metadata.index):
        raise ValueError("quality IDs must uniquely cover all development metadata")
    quality = quality.set_index("ecg_id").loc[metadata.index]
    if quality[["status", "flat_leads"]].isna().any().any():
        raise ValueError("quality status and flat-lead counts must be present")
    if not quality.status.isin(["ok", "sin_latidos", "derivacion_plana"]).all():
        raise ValueError("unknown historical extraction status")
    counts = quality.flat_leads.to_numpy(dtype=float)
    if not np.isfinite(counts).all() or np.any((counts < 0) | (counts > 12) | (counts != np.floor(counts))):
        raise ValueError("invalid flat-lead count")
    excluded = quality.status.eq("derivacion_plana")
    if not np.array_equal(excluded.to_numpy(), counts > 0):
        raise ValueError("flat-lead status and counts disagree")
    original.validate_labels(table, metadata.loc[~excluded], classes)
    labeled = metadata.scp_codes.apply(lambda codes: original.superclass_label(codes, classes)).notna()
    return {
        "policy": "superclass label is defined and raw status != derivacion_plana; folds 1-9 only",
        "metadata_count": len(metadata),
        "labeled_count_before_quality": int(labeled.sum()),
        "quality_status_counts": {
            str(name): int(count) for name, count in quality.status.value_counts().items()
        },
        "excluded_labeled_count": int((excluded & labeled).sum()),
        "excluded_labeled_ecg_ids": [int(value) for value in metadata.index[excluded & labeled]],
        "selected_count": len(table),
        "retained_sin_latidos_count": int(
            np.count_nonzero(quality.loc[table.ecg_id, "status"] == "sin_latidos")
        ),
    }


def provenance(protocol: Path, amendment: Path) -> dict[str, Any]:
    """Bind the additive execution amendment and every original scientific helper."""
    sources = original.provenance(protocol)
    extra = [
        Path(__file__),
        ROOT / "scripts/experiments/run_cpu_screening041_v2.py",
        amendment,
        ROOT / "notebooks/Avance2.8.ipynb",
    ]
    sources["files_sha256"].update({to_stored(path): sha256_file(path) for path in extra})
    sources["execution_amendment"] = to_stored(amendment)
    return sources


def evaluate_models(
    table: pd.DataFrame,
    values: pd.DataFrame,
    roles: dict[str, np.ndarray],
    output: Path,
    protocol: Path,
) -> dict[str, Any]:
    """Use unchanged fitting, calibration, interval and decision helpers after the integrity gate."""
    models, fit_seconds = original.fit_arms(values, table.target.to_numpy(dtype=int), roles["fit"])
    evaluation = roles["calibration"] | roles["development"]
    evaluated = table.loc[evaluation].reset_index(drop=True)
    development = evaluated.strat_fold.eq(9).to_numpy()
    calibration = ~development
    thresholds, summaries, predictions, predict_seconds = {}, {}, {}, {}
    for name, model in models.items():
        start = time.monotonic()
        probabilities = model.predict_proba(values.loc[evaluation])[:, 1]
        predict_seconds[name] = time.monotonic() - start
        thresholds[name] = original.calibrate_threshold(
            evaluated.target.to_numpy()[calibration], probabilities[calibration]
        )
        predictions[name] = probabilities
        summaries[name] = original.arm_summary(
            evaluated.loc[development], probabilities[development], thresholds[name]["threshold"]
        )
    dev_y = evaluated.target.to_numpy(dtype=int)[development]
    patients = evaluated.patient_id.to_numpy()[development]
    delta = original.paired_auroc_difference(
        patients,
        dev_y,
        predictions["hgb"][development],
        predictions["logistic"][development],
        draws=original.DRAWS,
        seed=original.SEED,
    )
    sensitivity = original.sensitivity_difference(
        patients,
        dev_y,
        predictions["hgb"][development] > thresholds["hgb"]["threshold"],
        predictions["logistic"][development] > thresholds["logistic"]["threshold"],
    )
    joblib.dump(
        {
            "models": models,
            "features": list(original.RAW_FEATURES),
            "thresholds": thresholds,
            "comparator": ">",
            "protocol": to_stored(protocol),
        },
        output / "fitted_models.joblib",
    )
    write_npz_atomic(
        output / "predictions.npz",
        ecg_id=evaluated.ecg_id.to_numpy(),
        patient_id=evaluated.patient_id.to_numpy(),
        strat_fold=evaluated.strat_fold.to_numpy(),
        target=evaluated.target.to_numpy(dtype=int),
        **predictions,
        **{name: evaluated[name].to_numpy(dtype=int) for name in original.SUPERCLASSES},
    )
    iterations = models["logistic"].named_steps["logisticregression"].n_iter_
    return {
        "thresholds": thresholds,
        "development": summaries,
        "paired_auroc_hgb_minus_logistic": delta,
        "paired_sensitivity_hgb_minus_logistic": sensitivity,
        "decision": original.decision(
            delta, sensitivity["difference"], summaries["hgb"]["normal_referral_rate"]["value"]
        ),
        "model_fit_status": {
            "logistic_iterations": iterations.tolist(),
            "logistic_reached_iteration_cap": bool(np.any(iterations >= original.LR_CONFIG["max_iter"])),
            "hgb_iterations": int(models["hgb"].n_iter_),
        },
        "timing": {"fit_seconds": fit_seconds, "predict_seconds": predict_seconds},
    }


def run_experiment(
    output: Path, protocol: Path, amendment: Path = AMENDMENT_PATH, threads: int = 4
) -> dict[str, Any]:
    """Run the corrected cohort check and unchanged experiment, refusing output overwrites."""
    if not 1 <= threads <= 4:
        raise ValueError("threads must be between 1 and 4")
    output.mkdir(parents=True, exist_ok=False)
    start = time.monotonic()
    try:
        with threadpool_limits(limits=threads):
            result = _run(output, protocol, amendment, threads)
        result["total_seconds"] = time.monotonic() - start
        write_json_atomic(output / "result.json", result)
        return result
    except Exception as error:
        write_json_atomic(
            output / "failure.json",
            {
                "status": "failed",
                "error_type": type(error).__name__,
                "error": str(error).replace(str(ROOT), "."),
                "elapsed_seconds": time.monotonic() - start,
                "test_scoring": False,
            },
        )
        raise


def _run(output: Path, protocol: Path, amendment: Path, threads: int) -> dict[str, Any]:
    """Keep all cohort and saved-baseline integrity checks before either prospective fit."""
    sources = provenance(protocol, amendment)
    table = original.read_selected()
    roles = original.role_masks(table)
    metadata = original.load_metadata()
    cohort = validate_historical_cohort(
        table, metadata, read_quality(metadata), original.diagnostic_classes(original.load_statements())
    )
    write_json_atomic(output / "cohort.json", cohort)
    with threadpool_limits(limits=1):
        integrity = original.baseline_integrity(table)
    integrity["threads"] = 1
    write_json_atomic(output / "integrity.json", integrity)
    evaluation = evaluate_models(table, original.raw_measurements(table), roles, output, protocol)
    return {
        "status": "completed",
        "experiment": "041",
        "execution_version": 2,
        "provenance": sources,
        "cohort": cohort,
        "integrity": integrity,
        "config": {
            "threads": threads,
            "seed": original.SEED,
            "bootstrap_draws": original.DRAWS,
            "fit_folds": list(range(1, 8)),
            "calibration_fold": 8,
            "development_fold": 9,
            "logistic": original.LR_CONFIG,
            "hgb": original.HGB_CONFIG,
            "features": list(original.RAW_FEATURES),
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
                "positive": int(table.loc[mask, "target"].sum()),
                "normal": int(table.loc[mask, "target"].eq(0).sum()),
            }
            for name, mask in roles.items()
        },
        **evaluation,
        "artifact_sha256": {
            to_stored(output / name): sha256_file(output / name)
            for name in ("fitted_models.joblib", "predictions.npz", "integrity.json", "cohort.json")
        },
        "artifacts": {
            name: to_stored(output / name)
            for name in (
                "fitted_models.joblib",
                "predictions.npz",
                "integrity.json",
                "cohort.json",
                "result.json",
            )
        },
    }
