"""Build scientific PTB/SPH baseline inputs from raw data without historical model or feature caches."""

from __future__ import annotations

import gc
import hashlib
import subprocess
import time
from pathlib import Path
from typing import Any

import h5py
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
from .full_development import fit_logistic, predict
from .gpu import GPU_LOCK_PATH, gpu_lock
from .measurement_extractor042 import read_ecg
from .pr74_screening_features import checkpoint_hashes, load_encoder, measurements
from .pr74_screening_io import binding, read_json, relative, stored, verify_hashes
from .pr74_screening_metrics import budget_threshold, point_metrics
from .public_sources import signal_sha256

OUTPUT = Path("outputs/pr74_fresh_baseline_20261028")
ROLES = Path("outputs/cohort25_recovery_20261028/raw_roles")
PROTOCOL = Path("docs/pr74-fresh-baseline-protocol-2026-10-28.md")


def source_binding() -> dict[str, Any]:
    """Bind this committed cold pipeline separately from historical replay receipts."""
    receipt = binding()
    own = [ROOT / PROTOCOL, Path(__file__), ROOT / "scripts/experiments/run_pr74_fresh_baseline.py"]
    own.append(ROOT / "ecg_experiment/cuda_runtime20261028.py")
    own += sorted((ROOT / "tests").glob("test_pr74_fresh_baseline*.py"))
    for path in own:
        name = stored(path.relative_to(ROOT))
        committed = subprocess.check_output(["git", "show", "HEAD:" + name], cwd=ROOT)
        if hashlib.sha256(committed).hexdigest() != sha256_file(path):
            raise ValueError("Cold-bootstrap scientific source is not committed")
        receipt["source_sha256"][name] = sha256_file(path)
    receipt["protocol"] = {"path": str(PROTOCOL), "sha256": sha256_file(ROOT / PROTOCOL)}
    receipt["input_mode"] = "fresh_raw_scientific_reproduction"
    return receipt


def align_roles(rows: pd.DataFrame, evidence: pd.DataFrame, site: str) -> pd.DataFrame:
    """Bind scientific identities to raw pointers without coercing fractional administrative IDs."""
    if site == "ptb":
        rows["ecg_id"] = rows["ecg_id"].map(canonical_integral_id)
        evidence["ecg_id"] = evidence["ecg_id"].map(canonical_integral_id)
    evidence = evidence.set_index("ecg_id")
    if not evidence.index.is_unique or rows["ecg_id"].duplicated().any():
        raise ValueError("Raw role/evidence identities are duplicated")
    pointers = evidence.loc[rows["ecg_id"]].reset_index()
    for column in ("patient_id", "target"):
        if column not in pointers:
            continue
        first, second = rows[column], pointers[column]
        if column == "patient_id" and site == "ptb":
            first, second = first.map(canonical_integral_id), second.map(canonical_integral_id)
        elif column == "target":
            first, second = pd.to_numeric(first, errors="raise"), pd.to_numeric(second, errors="raise")
        if first.tolist() != second.tolist():
            raise ValueError("Raw pointer and scientific role identities disagree")
    for column in rows:
        pointers[column] = rows[column].to_numpy()
    pointers["site"] = site
    target = pd.to_numeric(pointers["target"], errors="raise")
    if not target.isin([0, 1]).all():
        raise ValueError("Undefined scientific baseline target")
    pointers["target"] = target.astype(int)
    return pointers


def assign_roles(pointers: pd.DataFrame, site: str) -> pd.DataFrame:
    """Enforce fixed fitting/calibration/development roles before opening feature waveforms."""
    if site == "ptb":
        folds = pd.to_numeric(pointers["strat_fold"])
        if not folds.between(1, 9).all():
            raise ValueError("PTB fold 10 cannot enter baseline features")
        if pointers.groupby("patient_id")["strat_fold"].nunique().max() != 1:
            raise ValueError("PTB patients cross scientific baseline folds")
        pointers["fit"] = folds.le(7)
        pointers["evaluation"] = folds.eq(9)
    else:
        if not pointers["role"].isin(["calibration", "evaluation"]).all():
            raise ValueError("Unknown SPH role")
        if pointers.groupby("patient_id")["role"].nunique().max() != 1:
            raise ValueError("SPH patients cross calibration/evaluation roles")
        pointers["fit"] = False
        pointers["evaluation"] = pointers["role"].eq("evaluation")
        if not pointers.loc[~pointers["evaluation"], "target"].eq(0).all():
            raise ValueError("SPH calibration scoring must contain normals only")
    return pointers


def role_inputs(input_root: Path, roles: Path) -> tuple[dict[str, pd.DataFrame], dict[str, str]]:
    """Read only fresh raw-derived role tables and their actual source/byte identity evidence."""
    path = relative(roles) / "receipt.json"
    receipt = read_json(input_root / path)
    if receipt.get("complete") is not True or receipt.get("scope") != "administrative_roles_raw_quality":
        raise ValueError("A completed fresh raw-role receipt is required")
    hashes = {stored(path): sha256_file(input_root / path), **receipt["artifacts"], **receipt["input_sha256"]}
    verify_hashes(input_root, hashes)
    verify_hashes(ROOT, receipt["source_sha256"])
    result = {}
    for site, administrative in (("ptb", "ptb_administrative.csv"), ("sph", "sph_all.csv")):
        rows = pd.read_csv(input_root / roles / (site + "_rows.csv"), dtype=str, keep_default_na=False)
        evidence = pd.read_csv(input_root / roles / administrative, dtype=str, keep_default_na=False)
        result[site] = assign_roles(align_roles(rows, evidence, site), site)
    if len(result["ptb"]) != 19229 or len(result["sph"]) != 17402:
        raise ValueError("Frozen scientific baseline role counts differ")
    return result, hashes


def raw_paths(input_root: Path, row: pd.Series) -> tuple[Path, Path | None]:
    """Resolve only admitted PTB/SPH pointers and reject reserved source roles."""
    path = input_root / relative(row["waveform_path"])
    source = row["source"]
    if str(row["window_start"]) != "0" or str(row["samples"]) != "5000":
        raise ValueError("Baseline requires the frozen first ten-second view")
    if source == "ptbxl":
        if row["backend"] != "ptbxl_wfdb" or str(row["strat_fold"]) not in {str(i) for i in range(1, 10)}:
            raise ValueError("Reserved PTB source or backend")
        raw_path = Path(str(path) + ".dat")
        header = Path(str(path) + ".hea")
    elif source == "sph" and row["backend"] == "sph_h5":
        raw_path, header = path, None
    else:
        raise ValueError("Only declared PTB/SPH development records are admitted")
    return raw_path, header


def verify_raw(raw_path: Path, header: Path | None, row: pd.Series) -> None:
    """Require both current raw bytes and the optional PTB header to match their fresh evidence."""
    if sha256_file(raw_path) != row["raw_waveform_sha256"]:
        raise ValueError("Baseline raw waveform changed")
    if header is not None and sha256_file(header) != row["raw_header_sha256"]:
        raise ValueError("Baseline source header changed")


def read_signal(input_root: Path, row: pd.Series) -> np.ndarray:
    """Verify precise raw bytes and canonical identity without requiring a historical cache."""
    raw_path, header = raw_paths(input_root, row)
    verify_raw(raw_path, header, row)
    source = row["source"]
    if source == "ptbxl":
        signal = read_ecg(input_root / row["waveform_path"]).T
    else:
        with h5py.File(raw_path, "r") as handle:
            signal = handle["ecg"][:, :5000].astype(np.float32).astype(np.float64)
    if signal.shape != (12, 5000) or not np.isfinite(signal).all():
        raise ValueError("Invalid baseline waveform view")
    if signal_sha256(signal) != row["signal_sha256"]:
        raise ValueError("Baseline canonical window identity changed")
    verify_raw(raw_path, header, row)
    return signal


def finish_stage(input_root: Path, output: Path, stage: str, receipt: dict[str, Any],
                  hashes: dict[str, str], details: dict[str, Any]) -> dict[str, Any]:
    """Publish new scientific completion evidence only after rechecking all captured inputs and source."""
    verify_hashes(input_root, hashes)
    verify_hashes(ROOT, receipt["source_sha256"])
    directory = input_root / relative(output) / stage
    artifacts = {stored(output / stage / path.name): sha256_file(path)
                 for path in directory.iterdir() if path.is_file() and path.name != "completion.json"}
    result = {**receipt, **details, "complete": True, "status": "completed", "stage": stage,
              "input_sha256": hashes, "artifacts": artifacts}
    write_json_atomic(directory / "completion.json", result)
    return result


def stage_inputs(input_root: Path, output: Path, stage: str) -> tuple[dict[str, Any], dict[str, str]]:
    """Require real newly generated stage receipts without interpreting them as historical artifacts."""
    path = relative(output) / stage / "completion.json"
    receipt = read_json(input_root / path)
    if (receipt.get("input_mode") != "fresh_raw_scientific_reproduction" or
            receipt.get("status") != "completed"):
        raise ValueError("Completed scientific cold-bootstrap stage required")
    hashes = {stored(path): sha256_file(input_root / path), **receipt["artifacts"], **receipt["input_sha256"]}
    verify_hashes(input_root, hashes)
    verify_hashes(ROOT, receipt["source_sha256"])
    return receipt, hashes


def profile(input_root: Path, output: Path, roles: Path = ROLES, device: int = 0) -> dict[str, Any]:
    """Gate full cold features on a measured raw-path profile, without fitting any model."""
    receipt = source_binding()
    cuda_runtime = configure_cuda_runtime()
    sites, hashes = role_inputs(input_root, roles)
    hashes.update(checkpoint_hashes(input_root))
    selected = [sites["ptb"].loc[sites["ptb"]["fit"]].head(32), sites["sph"].head(64)]
    rows = pd.concat(selected, ignore_index=True)
    directory = input_root / relative(output) / "profile"
    directory.mkdir(parents=True, exist_ok=False)
    lock = GPU_LOCK_PATH if device == 0 else Path("/tmp/ecg_project_gpu1.lock")
    with gpu_lock("cuda", blocking=False, path=lock), torch.cuda.device(device):
        torch.set_num_threads(2)
        free, _ = torch.cuda.mem_get_info()
        if free < 4 * 1024**3:
            raise RuntimeError("Less than four GiB free for the cold profile")
        torch.cuda.reset_peak_memory_stats()
        model = load_encoder(input_root)
        started = time.perf_counter()
        signals = [read_signal(input_root, row) for _, row in rows.iterrows()]
        measured = measurements(signals)
        vectors = xecg_features(model, np.stack([xecg_input(signal) for signal in signals]))
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - started
        peak = torch.cuda.max_memory_allocated()
        del model
        gc.collect()
        torch.cuda.empty_cache()
    total = sum(len(frame) for frame in sites.values())
    forecast = 2 * elapsed / len(rows) * total + 300
    footprint = total * (1024 * 4 + 130 * 8)
    gates = {"free_at_least_4gib": free >= 4 * 1024**3, "peak_under_3gib": peak < 3 * 1024**3,
             "forecast_under_2hours": forecast < 7200, "output_under_1gib": footprint < 1024**3}
    rows.to_csv(directory / "rows.csv", index=False)
    measured.to_csv(directory / "measurements.csv", index=False)
    np.save(directory / "xecg.npy", vectors)
    details = {"profile_records": len(rows), "records": total, "device": device,
               "wall_seconds": elapsed, "forecast_seconds": forecast, "peak_gpu_bytes": peak,
               "initial_free_gpu_bytes": free, "resource_gates": gates, "role_directory": str(roles)}
    details["cuda_runtime"] = cuda_runtime
    write_json_atomic(directory / "profile.json", details)
    if not all(gates.values()):
        raise RuntimeError("Cold raw feature resource profile failed")
    return finish_stage(input_root, output, "profile", receipt, hashes, details)


def features(input_root: Path, output: Path, roles: Path = ROLES, device: int = 0) -> dict[str, Any]:
    """Build complete PTB/SPH features from raw data under the passing resource gate."""
    receipt = source_binding()
    configure_cuda_runtime()
    profiled, hashes = stage_inputs(input_root, output, "profile")
    if not all(profiled["resource_gates"].values()) or profiled["device"] != device:
        raise ValueError("The actual selected GPU must pass the cold resource profile")
    sites, bound = role_inputs(input_root, roles)
    hashes.update(bound)
    directory = input_root / relative(output) / "features"
    directory.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    lock = GPU_LOCK_PATH if device == 0 else Path("/tmp/ecg_project_gpu1.lock")
    with gpu_lock("cuda", blocking=False, path=lock), torch.cuda.device(device):
        torch.set_num_threads(2)
        model = load_encoder(input_root)
        for site, rows in sites.items():
            vectors = np.lib.format.open_memmap(directory / (site + "_xecg.npy"), mode="w+",
                                                dtype=np.float32, shape=(len(rows), 1024))
            raw = np.full((len(rows), 130), np.nan)
            for start in range(0, len(rows), 16):
                if time.perf_counter() - started > 7200:
                    raise TimeoutError("Cold feature extraction exceeded its two-hour stage ceiling")
                part = rows.iloc[start:start + 16]
                signals = [read_signal(input_root, row) for _, row in part.iterrows()]
                raw[start:start + len(part)] = measurements(signals).to_numpy()
                vectors[start:start + len(part)] = xecg_features(
                    model, np.stack([xecg_input(signal) for signal in signals]))
                if start % 512 == 0:
                    vectors.flush()
                    progress = {"site": site, "records": start + len(part), "total": len(rows),
                                "wall_seconds": time.perf_counter() - started}
                    write_json_atomic(directory / "progress.json", progress)
                    print(f"Cold {site} features {start + len(part)}/{len(rows)}", flush=True)
            vectors.flush()
            rows.to_csv(directory / (site + "_rows.csv"), index=False)
            raw_path = directory / (site + "_measurements.csv")
            pd.DataFrame(raw, columns=RAW_FEATURES).to_csv(raw_path, index=False)
        del model
        gc.collect()
        torch.cuda.empty_cache()
    counts = {name: len(rows) for name, rows in sites.items()}
    return finish_stage(input_root, output, "features", receipt, hashes,
                        {"wall_seconds": time.perf_counter() - started, "device": device,
                         "role_directory": str(roles), "records": counts})


def fit(input_root: Path, output: Path) -> dict[str, Any]:
    """Refit the known PTB-only baseline from new cold features and check published development parity."""
    receipt = source_binding()
    _, hashes = stage_inputs(input_root, output, "features")
    source = input_root / output / "features"
    ptb = pd.read_csv(source / "ptb_rows.csv", dtype={"ecg_id": str, "patient_id": str})
    raw = pd.read_csv(source / "ptb_measurements.csv")
    vectors = np.load(source / "ptb_xecg.npy", mmap_mode="r")
    fitting = parse_boolean(ptb["fit"].astype(str)).to_numpy()
    y = ptb["target"].to_numpy(dtype=int)
    models = {"hgb": HistGradientBoostingClassifier(**HGB_CONFIG).fit(raw.loc[fitting], y[fitting]),
              "xecg": fit_logistic(vectors[fitting], y[fitting])}
    directory = input_root / relative(output) / "fit"
    directory.mkdir(parents=True, exist_ok=False)
    results = {}
    for site in ("ptb", "sph"):
        rows = pd.read_csv(source / (site + "_rows.csv"), dtype={"ecg_id": str, "patient_id": str})
        raw = pd.read_csv(source / (site + "_measurements.csv"))
        vectors = np.load(source / (site + "_xecg.npy"), mmap_mode="r")
        admitted = ~parse_boolean(rows["fit"].astype(str)).to_numpy()
        held = parse_boolean(rows.loc[admitted, "evaluation"].astype(str)).to_numpy()
        y = rows.loc[admitted, "target"].to_numpy(dtype=int)
        scores = {"hgb": models["hgb"].predict_proba(raw.loc[admitted])[:, 1],
                  "xecg": predict(models["xecg"], vectors[admitted])}
        scores["blend"] = 0.5 * scores["hgb"] + 0.5 * scores["xecg"]
        results[site] = {}
        for name, values in scores.items():
            calibrated = budget_threshold(y[~held], values[~held])
            results[site][name] = {"calibration": calibrated,
                                    "metrics": point_metrics(y[held], values[held], calibrated["threshold"])}
        write_npz_atomic(directory / (site + "_predictions.npz"), **scores, target=y, evaluation=held,
                         ecg_id=rows.loc[admitted, "ecg_id"].to_numpy(dtype=str),
                         patient_id=rows.loc[admitted, "patient_id"].to_numpy(dtype=str))
    expected = {"hgb": (2395, 6517, 333, 1189, 0.9079384857273202),
                "xecg": (2407, 6515, 335, 1177, 0.9131929011339939),
                "blend": (2460, 6504, 346, 1124, 0.9174945011079249)}
    for name, reference in expected.items():
        point = results["sph"][name]["metrics"]
        if tuple(point[key] for key in ("tp", "tn", "fp", "fn")) != reference[:4]:
            raise ValueError(f"Fresh baseline {name} confusion matrix failed scientific parity")
        if abs(point["auroc"] - reference[4]) > 1e-9:
            raise ValueError(f"Fresh baseline {name} AUROC failed scientific parity")
    joblib.dump(models, directory / "fitted_models.joblib")
    write_json_atomic(directory / "baseline_evaluation.json", results)
    return finish_stage(input_root, output, "fit", receipt, hashes,
                        {"scientific_parity_passed": True, "published_auroc_tolerance": 1e-9,
                         "historical_serialized_artifacts_recovered": False, "results": results})
