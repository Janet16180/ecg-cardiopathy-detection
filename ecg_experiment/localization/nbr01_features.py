"""NBR-01 per-record preparation, xECG encoding and scoring of the replaced variants."""

from __future__ import annotations

import gc
import hashlib
import time
from multiprocessing import get_context
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import torch
from threadpoolctl import threadpool_limits

from .. import normative_map062 as nm
from ..cpu_screening041 import RAW_FEATURES
from ..external_encoders import xecg_features, xecg_input
from ..files import write_json_atomic, write_npz_atomic
from ..full_development import predict
from ..gpu import gpu_lock
from ..heads01_features import lock_path
from ..pr74_fresh_baseline import read_signal
from ..pr74_screening_features import load_encoder, measurements
from . import nbr01_counterpart as nc

BUNDLE = Path("outputs/pr74_from_raw_validation_20261028/evaluate/model_bundle.joblib")
PREDICTIONS = Path("outputs/pr74_from_raw_validation_20261028/evaluate/sph_predictions.npz")
MODELS = ("all_age", "ptb_baseline")
KINDS = ("hgb", "xecg", "blend")
BATCH = 16
PLAN_KEYS = ("difference", "swap", "components", "labels", "knots", "heart_rate")

_STATE: dict[str, Any] = {}


def save_model(path: Path, model: nm.PopulationModel) -> None:
    """Write every field of a population model to one ``.npz`` file."""
    write_npz_atomic(path, coefficients=model.coefficients, scale=model.scale, basis=model.basis,
                     eigenvalues=model.eigenvalues, noise=np.array(model.noise), off_scale=model.off_scale)


def load_model(path: Path) -> nm.PopulationModel:
    """Read a population model written by ``save_model``."""
    with np.load(path) as data:
        return nm.PopulationModel(data["coefficients"], data["scale"], data["basis"], data["eigenvalues"],
                                  float(data["noise"]), data["off_scale"])


def covariates(row: pd.Series) -> tuple[float, float]:
    """
    Age and 062-coded sex of one SPH row.

    Parameters
    ----------
    row : pd.Series
        Frozen baseline row with ``age`` and ``male``.

    Returns
    -------
    tuple[float, float]
        Age in years (NaN if missing) and sex as PTB-XL codes it (0 male, 1 female).

    Raises
    ------
    ValueError
        If sex is missing.
    """
    if row["male"] not in ("0.0", "1.0"):
        raise ValueError(f"Missing sex for {row['ecg_id']}")
    age = float(row["age"]) if row["age"] != "" else float("nan")
    return age, 1.0 - float(row["male"])


def signal_digest(signals: np.ndarray) -> str:
    """SHA-256 of a float64 variant array, used to prove that two stages built the same inputs."""
    return hashlib.sha256(np.ascontiguousarray(signals, dtype=np.float64).tobytes()).hexdigest()


def prepare_record(input_root: Path, row: pd.Series, model: nm.PopulationModel) -> dict[str, Any] | None:
    """
    Plan, variant energies, variant measurements and the variant digest of one record.

    Parameters
    ----------
    input_root : Path
        Checkout holding ``data`` and ``outputs``.
    row : pd.Series
        Frozen SPH baseline row.
    model : nm.PopulationModel
        Refitted 062 rank-0 model.

    Returns
    -------
    dict[str, Any] or None
        Plan arrays plus ``energy`` (variants,), ``measurements`` (variants, 130) and ``digest``; None if
        no counterpart can be built.
    """
    raw = read_signal(input_root, row)
    age, sex = covariates(row)
    plan = nc.record_plan(raw, age, sex, model)
    if plan is None:
        return None
    signals, energy = nc.variant_signals(raw, plan, nc.variants())
    measured = measurements(list(signals)).to_numpy()
    return {**{key: plan[key] for key in PLAN_KEYS}, "energy": energy, "measurements": measured,
            "digest": signal_digest(signals)}


def _prepare_job(position: int) -> tuple[str, str, float]:
    torch.set_num_threads(1)
    row = _STATE["rows"].iloc[position]
    path = _STATE["work"] / "plans" / f"{row['ecg_id']}.npz"
    if path.exists():
        return row["ecg_id"], "done", 0.0
    started = time.perf_counter()
    with threadpool_limits(limits=1):
        prepared = prepare_record(_STATE["input_root"], row, _STATE["model"])
    if prepared is None:
        write_npz_atomic(path, status=np.array("excluded"))
        return row["ecg_id"], "excluded", time.perf_counter() - started
    write_npz_atomic(path, status=np.array("prepared"), **prepared)
    return row["ecg_id"], "prepared", time.perf_counter() - started


def prepare_all(input_root: Path, rows: pd.DataFrame, model: nm.PopulationModel, work: Path,
                workers: int) -> list[tuple[str, str, float]]:
    """
    Prepare every row in forked single-thread workers, skipping records already written.

    Parameters
    ----------
    input_root : Path
        Checkout holding inputs.
    rows : pd.DataFrame
        Selected SPH baseline rows.
    model : nm.PopulationModel
        Refitted rank-0 model.
    work : Path
        Local work directory; plans go to ``work/plans``.
    workers : int
        Processes.

    Returns
    -------
    list[tuple[str, str, float]]
        ECG ID, status and seconds of each record.
    """
    (work / "plans").mkdir(parents=True, exist_ok=True)
    _STATE.update(rows=rows.reset_index(drop=True), work=work, input_root=input_root, model=model)
    with get_context("fork").Pool(workers) as pool:
        return pool.map(_prepare_job, range(len(rows)), chunksize=1)


def load_plan(path: Path) -> dict[str, Any]:
    """Read a prepared plan, rejecting excluded records."""
    with np.load(path) as data:
        if str(data["status"]) != "prepared":
            raise ValueError(f"Record was not prepared: {path.name}")
        return {key: data[key] for key in data.files}


def encode(model: torch.nn.Module, inputs: np.ndarray) -> np.ndarray:
    """
    Pooled xECG features in full batches of sixteen, padding the last batch with its final input.

    Parameters
    ----------
    model : torch.nn.Module
        Output of ``load_encoder``.
    inputs : np.ndarray
        Float32 ``[records, 1000, 12]`` inputs.

    Returns
    -------
    np.ndarray
        Float32 ``[records, 1024]`` features.
    """
    extra = (-len(inputs)) % BATCH
    padded = np.concatenate([inputs, np.repeat(inputs[-1:], extra, axis=0)]) if extra else inputs
    return xecg_features(model, padded)[: len(inputs)]


def encode_records(input_root: Path, rows: pd.DataFrame, work: Path, device: int,
                   limit_seconds: float | None = None) -> dict[str, Any]:
    """
    Encode every variant of the prepared rows on one GPU under its project lock.

    Parameters
    ----------
    input_root : Path
        Checkout holding inputs.
    rows : pd.DataFrame
        Prepared SPH baseline rows of this shard.
    work : Path
        Local work directory; features go to ``work/pooled``.
    device : int
        CUDA device index.
    limit_seconds : float or None
        Stop after this many seconds (profile).

    Returns
    -------
    dict[str, Any]
        Records encoded, wall seconds, peak and initial free GPU bytes.

    Raises
    ------
    RuntimeError
        If fewer than 4 GiB are free on the device.
    ValueError
        If the rebuilt variants differ from the prepared ones.
    """
    (work / "pooled").mkdir(parents=True, exist_ok=True)
    items = nc.variants()
    done = 0
    with gpu_lock("cuda", blocking=True, path=lock_path(device)), torch.cuda.device(device):
        torch.set_num_threads(2)
        free, _ = torch.cuda.mem_get_info()
        if free < 4 * 1024**3:
            raise RuntimeError("Less than four GiB free on the selected GPU")
        torch.cuda.reset_peak_memory_stats()
        model = load_encoder(input_root)
        started = time.perf_counter()
        for _, row in rows.iterrows():
            path = work / "pooled" / f"{row['ecg_id']}.npy"
            if path.exists():
                continue
            plan = load_plan(work / "plans" / f"{row['ecg_id']}.npz")
            signals, _ = nc.variant_signals(read_signal(input_root, row), plan, items)
            if signal_digest(signals) != str(plan["digest"]):
                raise ValueError(f"Rebuilt variants differ from the prepared ones: {row['ecg_id']}")
            pooled = encode(model, np.stack([xecg_input(signal) for signal in signals]))
            temporary = path.with_suffix(".tmp.npy")
            np.save(temporary, pooled)
            temporary.replace(path)
            done += 1
            if done % 20 == 0:
                write_json_atomic(work / f"encode_progress_{device}.json",
                                  {"records": done, "wall_seconds": time.perf_counter() - started})
            if limit_seconds is not None and time.perf_counter() - started > limit_seconds:
                break
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - started
        peak = torch.cuda.max_memory_allocated()
        del model
        gc.collect()
        torch.cuda.empty_cache()
    return {"records": done, "wall_seconds": elapsed, "peak_gpu_bytes": int(peak),
            "initial_free_gpu_bytes": int(free), "device": device}


def score_records(input_root: Path, ecg_ids: list[str], work: Path) -> dict[str, np.ndarray]:
    """
    Score every variant of the prepared and encoded records with both frozen models.

    Parameters
    ----------
    input_root : Path
        Checkout holding the model bundle.
    ecg_ids : list[str]
        Prepared records in analysis order.
    work : Path
        Local work directory with ``plans`` and ``pooled``.

    Returns
    -------
    dict[str, np.ndarray]
        ``scores`` (records, variants, models, kinds) float64, ``energy`` (records, variants),
        ``measurements`` (records, variants, 130) and ``pooled`` (records, variants, 1024) float32.
    """
    bundle = joblib.load(input_root / BUNDLE)
    if list(bundle["features"]) != list(RAW_FEATURES):
        raise ValueError("Model bundle feature order differs from the extractor")
    plans = [load_plan(work / "plans" / f"{ecg_id}.npz") for ecg_id in ecg_ids]
    measured = np.stack([plan["measurements"] for plan in plans])
    energy = np.stack([plan["energy"] for plan in plans])
    pooled = np.stack([np.load(work / "pooled" / f"{ecg_id}.npy") for ecg_id in ecg_ids])
    records, count = measured.shape[:2]
    frame = pd.DataFrame(measured.reshape(records * count, -1), columns=list(RAW_FEATURES))
    flat = pooled.reshape(records * count, -1)
    scores = np.empty((records, count, len(MODELS), len(KINDS)))
    for position, name in enumerate(MODELS):
        fitted = bundle["fitted"][name]
        hgb = fitted["hgb"].predict_proba(frame)[:, 1]
        xecg = predict(fitted["xecg"], flat)
        blend = 0.5 * hgb + 0.5 * xecg
        scores[:, :, position] = np.stack([hgb, xecg, blend], axis=1).reshape(records, count, len(KINDS))
    return {"scores": scores, "energy": energy, "measurements": measured, "pooled": pooled}


def thresholds(input_root: Path) -> dict[str, float]:
    """Frozen SPH 5% normal-budget thresholds of every component and blend of both models."""
    bundle = joblib.load(input_root / BUNDLE)
    return {f"{name}_{kind}": float(bundle["thresholds"][f"{name}_{kind}"]["sph"]["threshold"])
            for name in MODELS for kind in KINDS}
