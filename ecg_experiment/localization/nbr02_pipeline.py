"""NBR-02 per-record preparation, encoding and scoring of the three new counterpart arms."""

from __future__ import annotations

import gc
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
from ..external_encoders import xecg_input
from ..files import write_json_atomic, write_npz_atomic
from ..full_development import predict
from ..gpu import gpu_lock
from ..heads01_features import lock_path
from ..pr74_fresh_baseline import read_signal
from ..pr74_screening_features import load_encoder, measurements
from . import nbr01_counterpart as nc
from . import nbr01_features as nf
from .nbr02_templates import NEW_ARMS, arm_differences

NBR01_PLANS = Path("/tmp/nbr01-work/full/plans")
PAIRS = (("P6", "P0"), ("C6", "C0"), ("C0", "P0"), ("C6", "P0"))

_STATE: dict[str, Any] = {}


def counterpart_variants() -> list[nc.Variant]:
    """Return NBR-01's 52 normal-counterpart variants, positions 1-52 of ``nbr01_counterpart.variants``."""
    items = nc.variants()[1:53]
    if any(item.source != "counterpart" for item in items):
        raise ValueError("Unexpected NBR-01 variant layout")
    return items


def arm_signals(raw: np.ndarray, plan: dict[str, Any],
                difference: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Build the replaced signals and energies of one arm, with NBR-01's masks, taper and regions.

    Parameters
    ----------
    raw : np.ndarray
        (12, 5000) signal in mV.
    plan : dict[str, Any]
        NBR-01 plan of the record (its ``components`` and ``swap`` are reused).
    difference : np.ndarray
        The arm's counterpart minus the record.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        (52, 12, 5000) signals and (52,) energies.
    """
    arm_plan = {"difference": difference, "swap": plan["swap"], "components": plan["components"]}
    return nc.variant_signals(raw, arm_plan, counterpart_variants())


def prepare_record(input_root: Path, row: pd.Series, models: dict[str, nm.PopulationModel]) -> dict[str, Any]:
    """
    Build the arm differences, template diagnostics, energies, measurements and digests of one record.

    Parameters
    ----------
    input_root : Path
        Checkout holding inputs.
    row : pd.Series
        NBR-01 prepared row.
    models : dict[str, nm.PopulationModel]
        Models of the arms P0, P6, C0 and C6.

    Returns
    -------
    dict[str, Any]
        Arrays to save for the record.

    Raises
    ------
    ValueError
        If 062 fails on a record NBR-01 prepared, or the knots differ from NBR-01's.
    """
    raw = read_signal(input_root, row)
    age, sex = nf.covariates(row)
    plan = nf.load_plan(NBR01_PLANS / f"{row['ecg_id']}.npz")
    arms = arm_differences(raw, age, sex, models)
    if arms is None or not np.array_equal(arms["knots"], plan["knots"]):
        raise ValueError(f"062 features differ from NBR-01 for {row['ecg_id']}")
    result: dict[str, Any] = {
        "p0_max_abs_difference_mv": np.array(float(np.max(np.abs(
            arms["P0"]["difference"].astype(np.float64) - plan["difference"].astype(np.float64)))))}
    for first, second in PAIRS:
        result[f"template_{first}_{second}"] = np.array(float(np.mean(np.abs(
            arms[first]["template"] - arms[second]["template"]))))
    for arm in NEW_ARMS:
        signals, energy = arm_signals(raw, plan, arms[arm]["difference"])
        result[f"{arm}_difference"] = arms[arm]["difference"]
        result[f"{arm}_energy"] = energy
        result[f"{arm}_measurements"] = measurements(list(signals)).to_numpy()
        result[f"{arm}_digest"] = np.array(nf.signal_digest(signals))
        result[f"{arm}_coordinates"] = arms[arm]["coordinates"]
        result[f"{arm}_clipped"] = np.array(arms[arm]["clipped"])
    return result


def _prepare_job(position: int) -> tuple[str, float]:
    torch.set_num_threads(1)
    row = _STATE["rows"].iloc[position]
    path = _STATE["work"] / "plans" / f"{row['ecg_id']}.npz"
    if path.exists():
        return row["ecg_id"], 0.0
    started = time.perf_counter()
    with threadpool_limits(limits=1):
        prepared = prepare_record(_STATE["input_root"], row, _STATE["models"])
    write_npz_atomic(path, **prepared)
    return row["ecg_id"], time.perf_counter() - started


def prepare_all(input_root: Path, rows: pd.DataFrame, models: dict[str, nm.PopulationModel], work: Path,
                workers: int) -> list[tuple[str, float]]:
    """Prepare every row in forked single-thread workers, skipping records already written."""
    (work / "plans").mkdir(parents=True, exist_ok=True)
    _STATE.update(rows=rows.reset_index(drop=True), work=work, input_root=input_root, models=models)
    with get_context("fork").Pool(workers) as pool:
        return pool.map(_prepare_job, range(len(rows)), chunksize=1)


def encode_records(input_root: Path, rows: pd.DataFrame, work: Path, device: int) -> dict[str, Any]:
    """
    Encode the 156 new variants of each prepared record on one GPU under its project lock.

    Parameters
    ----------
    input_root : Path
        Checkout holding inputs.
    rows : pd.DataFrame
        Rows of this shard.
    work : Path
        Local work directory with ``plans``; features go to ``pooled``.
    device : int
        CUDA device.

    Returns
    -------
    dict[str, Any]
        Records encoded, wall seconds, peak and initial free GPU bytes.

    Raises
    ------
    RuntimeError
        If fewer than 4 GiB are free.
    ValueError
        If rebuilt variants differ from the prepared ones.
    """
    (work / "pooled").mkdir(parents=True, exist_ok=True)
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
            raw = read_signal(input_root, row)
            plan = nf.load_plan(NBR01_PLANS / f"{row['ecg_id']}.npz")
            with np.load(work / "plans" / f"{row['ecg_id']}.npz") as prepared:
                inputs = []
                for arm in NEW_ARMS:
                    signals, _ = arm_signals(raw, plan, prepared[f"{arm}_difference"])
                    if nf.signal_digest(signals) != str(prepared[f"{arm}_digest"]):
                        raise ValueError(f"Rebuilt {arm} variants differ for {row['ecg_id']}")
                    inputs += [xecg_input(signal) for signal in signals]
            pooled = nf.encode(model, np.stack(inputs))
            temporary = path.with_suffix(".tmp.npy")
            np.save(temporary, pooled)
            temporary.replace(path)
            done += 1
            if done % 20 == 0:
                write_json_atomic(work / f"encode_progress_{device}.json",
                                  {"records": done, "wall_seconds": time.perf_counter() - started})
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
    Score the new arms' variants with both frozen models.

    Parameters
    ----------
    input_root : Path
        Checkout holding the model bundle.
    ecg_ids : list[str]
        Records in analysis order.
    work : Path
        Local work directory with ``plans`` and ``pooled``.

    Returns
    -------
    dict[str, np.ndarray]
        ``scores`` (records, arms, 52, models, kinds), ``energy`` (records, arms, 52) and the template
        diagnostics per record.
    """
    bundle = joblib.load(input_root / nf.BUNDLE)
    if list(bundle["features"]) != list(RAW_FEATURES):
        raise ValueError("Model bundle feature order differs from the extractor")
    count, arms = len(counterpart_variants()), len(NEW_ARMS)
    measured, energy, diagnostics = [], [], {}
    for ecg_id in ecg_ids:
        with np.load(work / "plans" / f"{ecg_id}.npz") as prepared:
            measured.append(np.stack([prepared[f"{arm}_measurements"] for arm in NEW_ARMS]))
            energy.append(np.stack([prepared[f"{arm}_energy"] for arm in NEW_ARMS]))
            for key in prepared.files:
                if key.startswith(("template_", "p0_")) or key.endswith("_clipped"):
                    diagnostics.setdefault(key, []).append(float(prepared[key]))
    pooled = np.stack([np.load(work / "pooled" / f"{ecg_id}.npy") for ecg_id in ecg_ids])
    records = len(ecg_ids)
    frame = pd.DataFrame(np.stack(measured).reshape(records * arms * count, -1), columns=list(RAW_FEATURES))
    flat = pooled.reshape(records * arms * count, -1)
    scores = np.empty((records, arms, count, len(nf.MODELS), len(nf.KINDS)))
    for position, name in enumerate(nf.MODELS):
        fitted = bundle["fitted"][name]
        hgb = fitted["hgb"].predict_proba(frame)[:, 1]
        xecg = predict(fitted["xecg"], flat)
        values = np.stack([hgb, xecg, 0.5 * hgb + 0.5 * xecg], axis=1)
        scores[:, :, :, position] = values.reshape(records, arms, count, len(nf.KINDS))
    return {"scores": scores, "energy": np.stack(energy),
            **{key: np.array(value) for key, value in diagnostics.items()}}
