"""CPU-only prospective external transfer of frozen Experiment 041 models."""

from __future__ import annotations

import ast
import hashlib
import importlib.metadata
import io
import json
import platform
import subprocess
import time
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from ecg_experiment import ROOT, sph
from ecg_experiment.ecg_quality import EXCLUSION_REASONS, REVIEW_FLAGS, assess
from ecg_experiment.files import sha256_file, write_json_atomic
from ecg_experiment.intervals import paired_auroc_difference, threshold_metrics
from ecg_experiment.measurement_extractor042 import FEATURE_NAMES, PLAUSIBLE, extract_array, read_ecg
from ecg_experiment.paths import to_stored

SEED = 20261018
EXPECTED_RECORDS = 25770
OUTPUT = ROOT / "outputs/experiment042_measurement_transfer_v1"
RAW_CACHE = ROOT / "outputs/avance2_fusion/caracteristicas_primarias_ee39c8f6f7.csv"
PTB = ROOT / "data/raw/ptb-xl/1.0.3"
MODELS = ROOT / "outputs/experiment041_cpu_screening_v2/fitted_models.joblib"
PROTOCOL = ROOT / "docs/experiment-042-measurement-transfer.md"
SOURCES = [
    ROOT / "ecg_experiment/measurement_extractor042.py",
    ROOT / "ecg_experiment/measurement_screening042.py",
    ROOT / "scripts/experiments/run_measurement_screening042.py",
    ROOT / "ecg_experiment/cpu_screening041.py",
    ROOT / "ecg_experiment/cpu_screening041_v2.py",
    ROOT / "docs/experiment-041-cpu-screening-execution-v2.md",
    ROOT / "ecg_experiment/sph.py",
    ROOT / "ecg_experiment/ecg_quality.py",
    ROOT / "ecg_experiment/waveforms.py",
    ROOT / "scripts/experiments/run_cpu_screening041.py",
    ROOT / "scripts/experiments/run_cpu_screening041_v2.py",
    ROOT / "ecg_experiment/intervals.py",
    ROOT / "ecg_experiment/files.py",
    ROOT / "ecg_experiment/paths.py",
    ROOT / "notebooks/Avance2.8.ipynb",
]


def patient_role(patient_id: str) -> str:
    """Assign one patient to calibration or evaluation before scores exist.

    Parameters
    ----------
    patient_id : str
        Unmodified SPH patient identity.

    Returns
    -------
    str
        Deterministic role from the first SHA-256 byte.
    """
    first = hashlib.sha256(f"experiment042:20261018:{patient_id}".encode()).digest()[0]
    return "calibration" if first < 128 else "evaluation"


def feature_matrix(frame: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    """Select the exact frozen feature order, preserving missing measurements.

    Parameters
    ----------
    frame : pd.DataFrame
        Raw numeric measurements and optional identity columns.
    features : list[str]
        Ordered frozen model feature names.

    Returns
    -------
    pd.DataFrame
        Exactly 130 float columns, nonfinite values set to missing.

    Raises
    ------
    ValueError
        If the frozen feature schema is invalid.
    """
    if len(features) != 130 or len(set(features)) != 130:
        raise ValueError("Frozen model must contain 130 distinct raw features")
    return frame.loc[:, features].astype(np.float64).replace([np.inf, -np.inf], np.nan)


def serialized_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Round measurements through the original Avance CSV serialization.

    Parameters
    ----------
    frame : pd.DataFrame
        Extracted numeric features indexed by ECG identity.

    Returns
    -------
    pd.DataFrame
        The same %.6g representation used for the PTB-XL training cache.
    """
    return pd.read_csv(io.StringIO(frame.to_csv(float_format="%.6g")), index_col=0)


def notebook_extractor_hash(notebook: Path) -> str:
    """Bind the six original extraction functions and plausibility dictionary.

    Parameters
    ----------
    notebook : Path
        Original Avance notebook.

    Returns
    -------
    str
        Full SHA-256 whose expected ten-character prefix is ee39c8f6f7.
    """
    cells = json.loads(notebook.read_text())["cells"]
    text = "\n".join("".join(cells[index]["source"]) for index in (15, 20))
    nodes = {node.name: node for node in ast.parse(text).body if isinstance(node, ast.FunctionDef)}
    names = (
        "read_ecg",
        "detect_r_peaks",
        "median_beat",
        "delineate",
        "spectral_features",
        "extract_features",
    )
    original = "".join(ast.get_source_segment(text, nodes[name]) + "\n" for name in names)
    return hashlib.sha256((original + repr(PLAUSIBLE)).encode()).hexdigest()


def provenance(protocol: Path) -> dict[str, Any]:
    """Fingerprint immutable inputs common to each execution stage.

    Parameters
    ----------
    protocol : Path
        Committed prospective protocol.

    Returns
    -------
    dict[str, Any]
        Sources, protocol, environment and Git revision.
    """
    sources = {to_stored(path): sha256_file(path) for path in SOURCES}
    return {
        "protocol": {"path": to_stored(protocol), "sha256": sha256_file(protocol)},
        "source_sha256": sources,
        "git_revision": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "python": platform.python_version(),
        "versions": {
            name: importlib.metadata.version(name)
            for name in ("numpy", "pandas", "scipy", "scikit-learn", "joblib", "h5py", "threadpoolctl")
        },
        "threads": 1,
        "workers": 1,
        "seed": SEED,
        "scope": "SPH reused external development hospital; no final-test or clinical claim",
        "label_difference": "PTB NORM-only diagnostic superclass; SPH AHA code 1 alone is negative",
    }


def complete_stage(directory: Path, receipt: dict[str, Any], started: float) -> dict[str, Any]:
    """Write a completion receipt after hashing the stage artifacts.

    Parameters
    ----------
    directory : Path
        Exclusive stage directory.
    receipt : dict[str, Any]
        Stage provenance and summary.
    started : float
        Starting monotonic clock value.

    Returns
    -------
    dict[str, Any]
        Saved receipt, including measured wall duration.
    """
    receipt["wall_seconds"] = time.perf_counter() - started
    receipt["artifacts"] = {p.name: sha256_file(p) for p in sorted(directory.iterdir()) if p.is_file()}
    write_json_atomic(directory / "completion.json", receipt)
    return receipt


def verify_stage(directory: Path, protocol: Path) -> dict[str, Any]:
    """Require a completed unchanged stage with the same source and protocol.

    Parameters
    ----------
    directory : Path
        Stage directory containing completion.json.
    protocol : Path
        Current prospective protocol.

    Returns
    -------
    dict[str, Any]
        Verified completion receipt.

    Raises
    ------
    ValueError
        If an artifact, source or protocol changed.
    """
    receipt = json.loads((directory / "completion.json").read_text())
    for name, expected in receipt["artifacts"].items():
        if sha256_file(directory / name) != expected:
            raise ValueError(f"Stage artifact changed: {directory / name}")
    current = provenance(protocol)
    for key in ("source_sha256", "protocol"):
        if receipt[key] != current[key]:
            raise ValueError(f"Stage provenance changed: {key}")
    return receipt


def qualify(output: Path, protocol: Path) -> dict[str, Any]:
    """Reproduce 32 fold-one raw cache rows before external measurement use.

    Parameters
    ----------
    output : Path
        Experiment output root.
    protocol : Path
        Frozen prospective protocol.

    Returns
    -------
    dict[str, Any]
        Exact equality gate and maximum serialized numerical difference.

    Raises
    ------
    ValueError
        If source binding or any serialized numeric measurement disagrees.
    """
    started = time.perf_counter()
    directory = output / "qualification"
    directory.mkdir(parents=True, exist_ok=False)
    receipt = provenance(protocol)
    digest = notebook_extractor_hash(ROOT / "notebooks/Avance2.8.ipynb")
    if digest[:10] != "ee39c8f6f7":
        raise ValueError(f"Original notebook extractor source mismatch: {digest}")
    metadata_path = PTB / "ptbxl_database.csv"
    meta = pd.read_csv(metadata_path, index_col="ecg_id")
    cache = pd.read_csv(RAW_CACHE, index_col="ecg_id")
    ids = meta.index[(meta.strat_fold == 1) & meta.index.isin(cache.index)].sort_values()[:32]
    if len(ids) != 32:
        raise ValueError("Qualification requires exactly32 fold-one ECGs present in raw cache")
    features = [c for c in cache if c not in {"status", "implausible", "flat_leads"}]
    raw_paths = [PTB / meta.loc[ecg_id, "filename_hr"] for ecg_id in ids]
    rows = [extract_array(read_ecg(path)) for path in raw_paths]
    extracted = pd.DataFrame(rows, index=ids).reindex(columns=cache.columns)
    numeric = serialized_features(feature_matrix(extracted, features)).to_numpy()
    expected = feature_matrix(cache.loc[ids], features).to_numpy()
    status_same = all(
        extracted[column].fillna("").astype(str).equals(cache.loc[ids, column].fillna("").astype(str))
        for column in ("status", "implausible")
    )
    flat_same = np.array_equal(extracted.flat_leads.to_numpy(), cache.loc[ids, "flat_leads"].to_numpy())
    same = np.array_equal(numeric, expected, equal_nan=True) and status_same and flat_same
    missing_same = np.array_equal(np.isnan(numeric), np.isnan(expected))
    differences = np.abs(numeric - expected)
    maximum = float(np.nanmax(differences)) if np.isfinite(differences).any() else 0.0
    checks = {
        "passed": same,
        "serialized_max_abs_difference": maximum,
        "missingness_equal": missing_same,
        "status_plausibility_equal": status_same,
        "flat_leads_equal": flat_same,
        "records": len(ids),
        "features": len(features),
        "notebook_extractor_sha256": digest,
        "cache_sha256": sha256_file(RAW_CACHE),
        "metadata_sha256": sha256_file(metadata_path),
        "raw_inputs": {to_stored(Path(f"{p}.dat")): sha256_file(Path(f"{p}.dat")) for p in raw_paths},
    }
    write_json_atomic(directory / "integrity.json", checks)
    extracted.to_csv(directory / "reproduced_rows.csv", float_format="%.6g")
    if not same:
        raise ValueError(f"Raw cache equality failed; maximum absolute difference {maximum}")
    return complete_stage(directory, {**receipt, **checks}, started)


def extract_record(ecg_id: str) -> dict[str, Any]:
    """Read a SPH window once and extract measurements, quality and hash.

    Parameters
    ----------
    ecg_id : str
        Record identifier.

    Returns
    -------
    dict[str, Any]
        Unrounded features plus waveform hash and descriptive quality flags.
    """
    window = sph.read_window(ecg_id)
    reasons, flags = assess(window)
    features = extract_array(window.astype(np.float64).T)
    return {
        "ecg_id": ecg_id,
        **dict.fromkeys(FEATURE_NAMES, np.nan),
        **features,
        "signal_sha256": hashlib.sha256(window.tobytes()).hexdigest(),
        "quality_fail": bool(reasons),
        **{name: name in reasons for name in EXCLUSION_REASONS},
        **{name: name in flags for name in REVIEW_FLAGS},
    }


def sph_metadata() -> pd.DataFrame:
    """Load and verify SPH labels and assign patient roles before scoring.

    Returns
    -------
    pd.DataFrame
        Canonical labels, raw AHA strings and fixed patient roles.

    Raises
    ------
    ValueError
        If record identities, count or patient identities are invalid.
    """
    sph.verify_small_files()
    frame = sph.table().sort_index()
    if len(frame) != EXPECTED_RECORDS or not frame.index.is_unique or frame.patient_id.isna().any():
        raise ValueError("SPH metadata record count or identity failed")
    codes = pd.read_csv(sph.METADATA, dtype={"ECG_ID": str, "AHA_Code": str}).set_index("ECG_ID")
    frame["aha_codes"] = codes.AHA_Code
    frame["role"] = frame.patient_id.map(patient_role)
    return frame.drop(columns=["age", "male"])


def remaining_budget(output: Path) -> float:
    """Return seconds left in the full two-hour execution budget.

    Parameters
    ----------
    output : Path
        Experiment root containing completed stage receipts.

    Returns
    -------
    float
        Two hours minus the measured duration of every completed stage.

    Raises
    ------
    TimeoutError
        If completed stages consumed the full experiment budget.
    """
    spent = sum(json.loads(path.read_text())["wall_seconds"] for path in output.glob("*/completion.json"))
    remaining = 7200 - spent
    if remaining <= 0:
        raise TimeoutError("Experiment042 two-hour budget exhausted")
    return remaining


def extract_records(ecg_ids: list[str], deadline: float) -> pd.DataFrame:
    """Extract ordered records with a hard wall-budget check between windows.

    Parameters
    ----------
    ecg_ids : list[str]
        Ordered SPH record identities.
    deadline : float
        Latest allowed monotonic clock value.

    Returns
    -------
    pd.DataFrame
        Model-independent measurements indexed by record.

    Raises
    ------
    TimeoutError
        If the full experiment wall budget is exceeded.
    """
    rows = []
    for ecg_id in ecg_ids:
        if time.perf_counter() > deadline:
            raise TimeoutError("Experiment042 two-hour extraction budget exceeded")
        rows.append(extract_record(ecg_id))
    return pd.DataFrame(rows).set_index("ecg_id")


def prepare(output: Path, protocol: Path, profile_only: bool = False) -> dict[str, Any]:
    """Profile or prepare model-independent external measurements on one CPU.

    Parameters
    ----------
    output : Path
        Experiment output root.
    protocol : Path
        Frozen prospective protocol.
    profile_only : bool
        Extract only the first 64 lexical record identifiers for the resource gate.

    Returns
    -------
    dict[str, Any]
        Measured resource gate or completed full manifest receipt.

    Raises
    ------
    ValueError
        If an integrity or resource prerequisite fails.
    """
    qualification = verify_stage(output / "qualification", protocol)
    if not qualification["passed"]:
        raise ValueError("Extraction qualification has not passed")
    profile = None if profile_only else verify_stage(output / "profile", protocol)
    if profile is not None and not profile["resource_gate_passed"]:
        raise ValueError("Resource profile did not pass")
    started = time.perf_counter()
    deadline = started + remaining_budget(output)
    directory = output / ("profile" if profile_only else "preparation")
    directory.mkdir(parents=True, exist_ok=False)
    receipt = provenance(protocol)
    metadata = sph_metadata()
    metadata.loc[:, ["patient_id", "role"]].to_csv(directory / "patient_roles.csv")
    if not profile_only:
        evaluation_rows(metadata.assign(use_evaluation=True))
    selected = metadata.iloc[:64] if profile_only else metadata
    with threadpool_limits(limits=1):
        extracted = extract_records(selected.index.astype(str).to_list(), deadline)
    frame = selected.join(extracted, validate="one_to_one")
    frame["duplicate_status"] = sph.duplicate_status(frame.signal_sha256, frame.aha_codes)
    frame["use_evaluation"] = frame.duplicate_status.isin(["unique", "kept"])
    if not profile_only:
        evaluation_rows(frame)
    frame.to_csv(directory / "measurements.csv", float_format="%.6g")
    wall = time.perf_counter() - started
    size = (directory / "measurements.csv").stat().st_size
    forecast_wall = 2 * wall * EXPECTED_RECORDS / len(frame) + 120
    forecast_storage = 2 * size * EXPECTED_RECORDS / len(frame) + 32 * 1024**2
    receipt.update(
        {
            "records": len(frame),
            "metadata_sha256": sha256_file(sph.METADATA),
            "small_files_md5": sph.verify_small_files(),
            "raw_input_format": "SPH float16 mV; first5000 samples",
            "measurement_precision": "original %.6g CSV serialization",
            "quality_policy": "Every unique/kept labelled record retained regardless of technical quality",
            "patient_role_hash": "sha256('experiment042:20261018:'+patient_id).digest()[0]<128 calibration",
            "profile_overhead_seconds": 0.0 if profile is None else profile["wall_seconds"],
            "forecast_wall_seconds": forecast_wall,
            "forecast_storage_bytes": forecast_storage,
            "resource_gate_passed": bool(forecast_wall <= 3600 and forecast_storage <= 512 * 1024**2),
            "quality_fail_records": int(frame.quality_fail.sum()),
            "duplicate_counts": frame.duplicate_status.value_counts().to_dict(),
            "role_counts": frame.role.value_counts().to_dict(),
        }
    )
    return complete_stage(directory, receipt, started)


def evaluation_rows(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Select canonical target-labelled duplicate-resolved patient roles.

    Parameters
    ----------
    frame : pd.DataFrame
        Full prepared SPH manifest.

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame]
        Labelled calibration and evaluation rows with target alias.

    Raises
    ------
    ValueError
        If patient roles overlap or targets are not binary.
    """
    usable = frame.loc[frame.use_evaluation & frame.primary.notna()].copy()
    usable["target"] = usable.primary.astype(int)
    if not set(usable.target) <= {0, 1}:
        raise ValueError("Primary labels must be binary or missing")
    calibration = usable.loc[usable.role == "calibration"]
    evaluation = usable.loc[usable.role == "evaluation"]
    if not frame.role.equals(frame.patient_id.map(patient_role)):
        raise ValueError("Patient role table differs from frozen hash assignment")
    for role in (calibration, evaluation):
        if set(role.target) != {0, 1} or (role.target == 0).sum() < 100:
            raise ValueError("Each role requires both primary classes and at least 100 normals")
    if set(calibration.patient_id) & set(evaluation.patient_id):
        raise ValueError("Calibration and evaluation patients overlap")
    return calibration, evaluation


def evaluate(output: Path, protocol: Path, models_path: Path = MODELS) -> dict[str, Any]:
    """Compare frozen PTB-XL models after normal-only local recalibration.

    Parameters
    ----------
    output : Path
        Experiment output root with completed preparation.
    protocol : Path
        Committed prospective protocol.
    models_path : Path
        Frozen Experiment 041 model artifact; no refitting is performed.

    Returns
    -------
    dict[str, Any]
        Primary paired comparison, calibration, transport and decision.
    """
    from ecg_experiment.cpu_screening041 import (
        RAW_FEATURES,
        arm_summary,
        calibrate_threshold,
        sensitivity_difference,
    )

    if models_path.name != "fitted_models.joblib":
        raise ValueError("Only the bound fitted_models.joblib predecessor artifact may be scored")
    prepared = verify_stage(output / "preparation", protocol)
    started = time.perf_counter()
    deadline = started + remaining_budget(output)
    directory = output / "evaluation"
    directory.mkdir(parents=True, exist_ok=False)
    receipt = provenance(protocol)
    predecessor = verify_predecessor(models_path.parent)
    if sha256_file(models_path) != predecessor[to_stored(models_path)]:
        raise ValueError("Requested model differs from the verified predecessor artifact")
    models = joblib.load(models_path)
    if list(models["features"]) != list(RAW_FEATURES) or models["comparator"] != ">":
        raise ValueError("Frozen model schema or referral comparator differs")
    frame = pd.read_csv(
        output / "preparation/measurements.csv", index_col="ecg_id", dtype={"patient_id": str}
    )
    calibration, evaluation = evaluation_rows(frame)
    predictions = evaluation.loc[
        :, ["patient_id", "target", "MI", "STTC", "CD", "HYP", "quality_fail"]
    ].copy()
    thresholds, summaries, transport, ties = {}, {}, {}, {}
    for name in ("logistic", "hgb"):
        with threadpool_limits(limits=1):
            normal = calibration.loc[calibration.target == 0]
            calibration_scores = models["models"][name].predict_proba(
                feature_matrix(normal, models["features"])
            )[:, 1]
            probability = models["models"][name].predict_proba(
                feature_matrix(evaluation, models["features"])
            )[:, 1]
        thresholds[name] = calibrate_threshold(normal.target.to_numpy(), calibration_scores)
        threshold = thresholds[name]["threshold"]
        ties[name] = int(np.count_nonzero(calibration_scores == threshold))
        predictions[f"{name}_probability"] = probability
        predictions[f"{name}_referred"] = probability > threshold
        summaries[name] = arm_summary(evaluation, probability, threshold)
        frozen = models["thresholds"][name]["threshold"]
        transported = threshold_metrics(
            evaluation.target.to_numpy(), probability, np.nextafter(frozen, np.inf)
        )
        transport[name] = {
            "ptb_threshold": frozen,
            "comparator": ">",
            "sensitivity": transported["sensitivity"],
            "normal_referral_rate": 1 - transported["specificity"],
        }
        normal_predictions = normal.loc[:, ["patient_id", "target"]].copy()
        normal_predictions["probability"] = calibration_scores
        normal_predictions.to_csv(directory / f"{name}_normal_calibration.csv", float_format="%.17g")
        if time.perf_counter() > deadline:
            raise TimeoutError("Experiment042 two-hour evaluation budget exceeded")
    patients, y = evaluation.patient_id.to_numpy(), evaluation.target.to_numpy()
    primary = paired_auroc_difference(
        patients,
        y,
        predictions.hgb_probability.to_numpy(),
        predictions.logistic_probability.to_numpy(),
        draws=2000,
        seed=SEED,
    )
    sensitivity = sensitivity_difference(
        patients,
        y,
        predictions.hgb_referred.to_numpy(),
        predictions.logistic_referred.to_numpy(),
        draws=2000,
        seed=SEED,
    )
    hgb_normal_referral = float(predictions.loc[predictions.target == 0, "hgb_referred"].mean())
    checks = {
        "auroc_delta_at_least_0_01": primary["difference"] >= 0.01,
        "auroc_lower_bound_positive": primary["ci_low"] > 0,
        "sensitivity_delta_at_least_minus_0_02": sensitivity["difference"] >= -0.02,
        "hgb_normal_referral_at_most_0_08": hgb_normal_referral <= 0.08,
    }
    result = {
        "primary": primary,
        "sensitivity_difference": sensitivity,
        "arms": summaries,
        "local_thresholds": thresholds,
        "local_calibration_ties": ties,
        "ptb_threshold_transport": transport,
        "decision_checks": checks,
        "decision": "support_external_development" if all(checks.values()) else "do_not_advance_external",
        "counts": {
            "all_rows": len(frame),
            "calibration_labelled": len(calibration),
            "calibration_normals": int((calibration.target == 0).sum()),
            "calibration_positive": int((calibration.target == 1).sum()),
            "calibration_patients": int(calibration.patient_id.nunique()),
            "retained_unique_or_kept": int(frame.use_evaluation.sum()),
            "quality_failure_evaluation_rows": int(evaluation.quality_fail.sum()),
            "evaluation_labelled": len(evaluation),
            "evaluation_patients": int(evaluation.patient_id.nunique()),
            "evaluation_positive": int(y.sum()),
            "evaluation_negative": int((y == 0).sum()),
            "undefined_primary": int(frame.primary.isna().sum()),
        },
        "misses_by_superclass": {
            name: {
                code: int(
                    (
                        (predictions.target == 1)
                        & (predictions[code] == 1)
                        & ~predictions[f"{name}_referred"]
                    ).sum()
                )
                for code in sph.SUPERCLASSES
            }
            for name in ("logistic", "hgb")
        },
    }
    if time.perf_counter() > deadline:
        raise TimeoutError("Experiment042 two-hour evaluation budget exceeded")
    predictions.to_csv(directory / "predictions.csv", float_format="%.17g")
    write_json_atomic(directory / "results.json", result)
    receipt.update(
        {
            "predecessor_bindings": predecessor,
            "models": {"path": to_stored(models_path), "sha256": sha256_file(models_path)},
            "preparation_receipt_sha256": sha256_file(output / "preparation/completion.json"),
            "prepared_records": prepared["records"],
            "decision": result["decision"],
        }
    )
    return complete_stage(directory, receipt, started)


def verify_predecessor(directory: Path) -> dict[str, str]:
    """Bind the completed predecessor and its saved model and predictions.

    Parameters
    ----------
    directory : Path
        Experiment 041 artifact directory.

    Returns
    -------
    dict[str, str]
        Verified SHA-256 map of predecessor artifacts.

    Raises
    ------
    ValueError
        If the integrity gate is incomplete or an artifact differs.
    """
    result = json.loads((directory / "result.json").read_text())
    digests = {}
    for name in ("integrity.json", "fitted_models.joblib", "predictions.npz", "cohort.json"):
        path = directory / name
        actual = sha256_file(path)
        if result["artifact_sha256"].get(to_stored(path)) != actual:
            raise ValueError(f"Predecessor artifact changed or unbound: {name}")
        digests[to_stored(path)] = actual
    digests[to_stored(directory / "result.json")] = sha256_file(directory / "result.json")
    amendment = ROOT / "docs/experiment-041-cpu-screening-execution-v2.md"
    digests[to_stored(amendment)] = sha256_file(amendment)
    integrity = json.loads((directory / "integrity.json").read_text())
    if result.get("status") != "completed" or integrity.get("status") != "passed":
        raise ValueError("Predecessor integrity gate did not pass")
    return digests
