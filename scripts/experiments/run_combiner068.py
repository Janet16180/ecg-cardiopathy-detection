"""
Experiment 068: learned unit combiner v2 with out-of-fold MIL evidence from a patient-level cross-fit.

Phases::

    run_combiner068 profile    # GPU timing of one cross-fit fold (lock), nothing evaluated
    run_combiner068 crossfit   # five-fold 065 mil_lse cross-fit on fit records (GPU, lock per fold)
    run_combiner068 features   # units and unit features of training and scored records (CPU, 8 processes)
    run_combiner068 fit        # CV selection, final combiners, maps (CPU)
    run_combiner068 score      # integrity, LocBench, strata, standardized noise, result.json

``--smoke`` uses 1,200 fit records, 60 training and 16 scored records and a separate folder; nothing is
scored. See ``docs/experiment-068-combiner-v2.md``.
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import shutil
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor
from multiprocessing import Pool
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import HistGradientBoostingClassifier
from threadpoolctl import threadpool_limits

from ecg_experiment import ROOT
from ecg_experiment import combination066 as cb
from ecg_experiment import combiner068 as cv
from ecg_experiment import mil065 as mil
from ecg_experiment import normative_map062 as nm
from ecg_experiment.ecgad import load_split
from ecg_experiment.eda.ptbxl import load_metadata, load_statements
from ecg_experiment.files import sha256_file, write_json_atomic
from ecg_experiment.gpu import gpu_lock
from ecg_experiment.locbench import (
    FAMILIES,
    REGION_RHYTHMS,
    _statement_sets,
    ecgad_record_metrics,
    load_benchmark,
    scored_ecg_ids,
)
from ecg_experiment.locbench_baselines061 import amplitude_map, exact_signal, residual_map
from ecg_experiment.locbench_cache import CACHE_DIR, open_cache
from ecg_experiment.locbench_data import PARTITION_PATH, REGION_TABLE, load_partition
from ecg_experiment.paths import to_stored
from ecg_experiment.provenance import git_head, utc_now

OUT = ROOT / "outputs/experiment068_combiner_v2"
SMOKE_OUT = ROOT / "outputs/experiment068_combiner_smoke"
WORK = Path(os.environ.get("ECG068_WORK", "/tmp/ecg068-work"))
WORK066 = Path(os.environ.get("ECG066_WORK", "/tmp/ecg066-work")) / "full"
MIL065 = ROOT / "outputs/experiment065_mil_v1"
OUT066 = ROOT / "outputs/experiment066_combination_v1"
LOCBENCH061 = ROOT / "outputs/experiment061_locbench_v1"
NORMATIVE062 = ROOT / "outputs/experiment062_normative_v1"
SEEDS065 = ("lse_65065", "lse_65066", "lse_65067")
BAR = "qrs_prior"
REFERENCES = {
    BAR: LOCBENCH061 / "tables" / BAR,
    "residual056": LOCBENCH061 / "tables/residual056",
    "units_learned": OUT066 / "locbench/units_learned/tables",
    "textbook": OUT066 / "locbench/textbook/tables",
}
LINEAR = ("units_v2", "units_v2_nolead", "units_v1_data")
MAPS = ("units_v2", "units_v2_nolead", "units_v2_hgb", "units_v1_data")
WORKERS = 8
CHUNK = 128
SMOKE_FIT = 1200
SMOKE_RECORDS = 16
SMOKE_TRAIN = 60
PROFILE_STEPS = 40
TOLERANCE = 1e-12
HGB = {
    "max_depth": 3,
    "learning_rate": 0.1,
    "max_iter": 200,
    "l2_regularization": 1.0,
    "min_samples_leaf": 100,
}
SOURCES = (
    "ecg_experiment/combiner068.py",
    "scripts/experiments/run_combiner068.py",
    "docs/experiment-068-combiner-v2.md",
    "ecg_experiment/combination066.py",
    "ecg_experiment/mil065.py",
    "ecg_experiment/normative_map062.py",
    "ecg_experiment/locbench.py",
)

_STATE: dict[str, Any] = {}


def folders(smoke: bool) -> tuple[Path, Path]:
    """Output folder and local work folder of the run."""
    out = SMOKE_OUT if smoke else OUT
    work = WORK / ("smoke" if smoke else "full")
    out.mkdir(parents=True, exist_ok=True)
    work.mkdir(parents=True, exist_ok=True)
    return out, work


def heads065() -> list[str]:
    """Return the 49 065 heads in network order."""
    return json.loads((MIL065 / "result.json").read_text())["heads"]


def forbidden_ids() -> set[int]:
    """ECGAD records (both halves) and evaluation-partition records."""
    partition = load_partition()
    evaluation = partition.ecg_id[partition.partition == "evaluation"]
    return set(load_split().ecg_id.astype(int)) | set(evaluation.astype(int))


def holdout_ids() -> set[int]:
    """ECGAD holdout records, used only to assert that none is mapped."""
    split = load_split()
    return set(split.ecg_id[split.partition != "development"].astype(int))


def fit_table(smoke: bool) -> pd.DataFrame:
    """Fit-partition records with their cross-fit fold (smoke: the first records only)."""
    partition = load_partition()
    fit = partition[partition.partition == "fit"].reset_index(drop=True)
    if forbidden_ids() & set(fit.ecg_id.astype(int)):
        raise ValueError("A fit record is an ECGAD or evaluation record")
    fit["fold"] = cb.patient_folds(fit.patient_id.to_numpy(), cv.FOLDS, cv.FOLD_SEED)
    return fit.iloc[:SMOKE_FIT].reset_index(drop=True) if smoke else fit


def training_records(fit: pd.DataFrame) -> pd.DataFrame:
    """
    Every fit record with at least one REGION_TABLE statement, its statements, fold and rhythm flag.

    Returns
    -------
    pd.DataFrame
        ``ecg_id``, ``patient_id``, ``fold``, ``statements`` (list), ``region_rhythm`` (passes the
        region-set rhythm rule).
    """
    meta, rhythm, _, _, _ = _statement_sets()
    table = fit[fit.region_statements != ""].copy()
    table["statements"] = table.region_statements.str.split(";")
    table["region_rhythm"] = [
        not any(c in rhythm and c not in REGION_RHYTHMS for c in meta.at[e, "scp_codes"])
        for e in table.ecg_id
    ]
    clean = set(fit.ecg_id[fit.clean_normal].astype(int))
    if clean & set(table.ecg_id.astype(int)):
        raise ValueError("A training record is a fit clean normal (062 reference)")
    columns = ["ecg_id", "patient_id", "fold", "statements", "region_rhythm"]
    return table[columns].reset_index(drop=True)


def family_counts(train: pd.DataFrame) -> dict[str, int]:
    """Training records per LocBench family (a record counts once in each of its families)."""
    counts: dict[str, int] = {}
    for statements in train.statements:
        for family in {FAMILIES[s] for s in statements}:
            counts[family] = counts.get(family, 0) + 1
    return dict(sorted(counts.items()))


def device_setup() -> str:
    """CUDA when visible, capped at 6 GB, deterministic cuDNN as in 065."""
    if not torch.cuda.is_available():
        return "cpu"
    total = torch.cuda.get_device_properties(0).total_memory
    torch.cuda.set_per_process_memory_fraction(min(1.0, 6e9 / total))
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    return "cuda"


def preprocessed(fit: pd.DataFrame, work: Path) -> np.ndarray:
    """065-preprocessed signals of the fit records, cached on local disk."""
    path, ids_path = work / "fit_preprocessed.npy", work / "fit_preprocessed_ids.npy"
    if path.exists() and np.array_equal(np.load(ids_path), fit.ecg_id.to_numpy()):
        return np.ascontiguousarray(np.load(path, mmap_mode="r"))
    raw, index = open_cache()
    row_of = dict(zip(index.ecg_id.astype(int), range(len(index)), strict=True))
    positions = np.array([row_of[int(e)] for e in fit.ecg_id])
    out = np.empty((len(positions), 12, mil.SAMPLES), np.float32)

    def run(chunk: np.ndarray) -> None:
        out[chunk] = mil.preprocess(raw[positions[chunk]])

    with ThreadPoolExecutor(WORKERS) as pool:
        list(pool.map(run, np.array_split(np.arange(len(positions)), max(1, len(positions) // 256))))
    np.save(path, out)
    np.save(ids_path, fit.ecg_id.to_numpy())
    return out


def labels(fit: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """065 statement labels and mask of the fit records for the 49 heads."""
    return mil.statement_labels(load_metadata(), load_statements(), fit.ecg_id.to_numpy(), heads065())


def fold_rows(fit: pd.DataFrame, fold: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Training, early-stopping and out-of-fold rows of one fold.

    The early-stopping rows are 065's held-out rule (15% of patients, seed 65065) applied to the patients
    of the training folds.
    """
    pool = np.flatnonzero(fit.fold.to_numpy() != fold)
    inner = mil.heldout_mask(fit.patient_id.to_numpy()[pool])
    return pool[~inner], pool[inner], np.flatnonzero(fit.fold.to_numpy() == fold)


def profile_phase() -> None:
    """Time training steps of fold 0 on the GPU; no held-out reading is made."""
    out, work = folders(False)
    fit = fit_table(False)
    signals = preprocessed(fit, work)
    y, mask = labels(fit)
    train_rows, _, _ = fold_rows(fit, 0)
    device = device_setup()
    torch.manual_seed(cv.SEED)
    model = mil.LeadMIL(y.shape[1], pooling="lse", cross_lead=False).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)
    rng = np.random.default_rng(cv.SEED)
    y_t, m_t = torch.from_numpy(y).to(device), torch.from_numpy(mask).to(device)
    with gpu_lock(device):
        torch.cuda.reset_peak_memory_stats()
        times = []
        for _ in range(PROFILE_STEPS):
            began = time.time()
            rows = np.sort(rng.choice(train_rows, 64, replace=False))
            x = mil.batch_tensor(signals, rows, device, mil.CROP, rng)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                bag, _, _ = model(x)
            loss = mil.masked_bce(bag.float(), y_t[rows], m_t[rows])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            torch.cuda.synchronize()
            times.append(time.time() - began)
        peak = torch.cuda.max_memory_allocated()
    step = float(np.median(times[5:]))
    profile = {
        "train_rows_fold0": int(len(train_rows)),
        "median_step_seconds": step,
        "peak_gpu_bytes": int(peak),
        "estimated_epoch_seconds": step * len(train_rows) / 64,
        "device": torch.cuda.get_device_name(0),
    }
    write_json_atomic(out / "gpu_profile.json", profile)
    print(json.dumps(profile, indent=1), flush=True)


def summarize_chunk(bag: np.ndarray, z: np.ndarray, heads: list[str]) -> dict[str, np.ndarray]:
    """Bag logits, lead means, rhythm evidence and combined map of a chunk of records (066 summaries)."""
    abnormal = np.array([k for k, h in enumerate(heads) if h != mil.REFERENCE_HEAD])
    rhythm = [heads.index(h) for h in cb.RHYTHM_HEADS]
    evidence = mil.finding_evidence(z, None)
    return {
        "bags": bag,
        "lead_means": cb.lead_means(evidence),
        "rhythm": evidence[:, rhythm].astype(np.float16),
        "combined": mil.combined_map(bag, evidence, abnormal, "sum"),
    }


def train_fold(
    fit: pd.DataFrame, signals: np.ndarray, y: np.ndarray, mask: np.ndarray, fold: int, smoke: bool
) -> tuple[mil.Fit, dict[str, np.ndarray], dict[str, int]]:
    """
    Train one cross-fit network and summarize its out-of-fold records.

    Returns
    -------
    tuple
        ``mil065.Fit`` and the out-of-fold summaries (``summarize_chunk`` keys plus ``ecg_id``).
    """
    heads = heads065()
    train_rows, valid_rows, oof_rows = fold_rows(fit, fold)
    device = device_setup()
    torch.manual_seed(cv.SEED)
    model = mil.LeadMIL(len(heads), pooling="lse", cross_lead=False).to(device)
    recipe = mil.Recipe(max_epochs=1) if smoke else mil.Recipe()
    parts = []
    with gpu_lock(device):
        result = mil.train(
            model, signals, y, mask, train_rows, valid_rows, recipe, device, cv.SEED, 5 if smoke else None
        )
        model.load_state_dict(result.state)
        for start in range(0, len(oof_rows), CHUNK):
            rows = oof_rows[start : start + CHUNK]
            bag, z, _ = mil.predict(model, signals, rows, device, keep_instances=True)
            parts.append(summarize_chunk(bag, z, heads))
    summary = {key: np.concatenate([p[key] for p in parts]) for key in parts[0]}
    summary["ecg_id"] = fit.ecg_id.to_numpy()[oof_rows]
    return (
        result,
        summary,
        {"train": len(train_rows), "early_stopping": len(valid_rows), "oof": len(oof_rows)},
    )


def crossfit_phase(smoke: bool) -> None:
    """Five-fold patient-level cross-fit of the 065 mil_lse network; out-of-fold summaries per fit record."""
    out, work = folders(smoke)
    fit = fit_table(smoke)
    signals = preprocessed(fit, work)
    y, mask = labels(fit)
    (work / "crossfit").mkdir(exist_ok=True)
    for fold in range(cv.FOLDS):
        path = work / "crossfit" / f"fold{fold}.npz"
        if path.exists():
            print(f"fold {fold} exists, skipping", flush=True)
            continue
        began = time.time()
        result, summary, counts = train_fold(fit, signals, y, mask, fold, smoke)
        torch.save(result.state, work / "crossfit" / f"weights_fold{fold}.pt")
        history = {
            "best_epoch": result.best_epoch,
            "best_early_stopping_macro_auroc": result.best_score,
            "rows": counts,
            "history": result.history,
            "seconds": time.time() - began,
        }
        write_json_atomic(out / f"crossfit_fold{fold}.json", history)
        np.savez(path, **summary)
        print(f"fold {fold}: best epoch {result.best_epoch}, {time.time() - began:.0f}s", flush=True)
    assemble_oof(fit, work)
    write_json_atomic(out / "crossfit_summary.json", crossfit_summary(fit, y, mask, work, smoke))


def assemble_oof(fit: pd.DataFrame, work: Path) -> None:
    """Put the per-fold out-of-fold summaries in fit-record order."""
    parts = [dict(np.load(work / "crossfit" / f"fold{f}.npz")) for f in range(cv.FOLDS)]
    ids = np.concatenate([p["ecg_id"] for p in parts])
    order = {int(e): k for k, e in enumerate(ids)}
    rows = np.array([order[int(e)] for e in fit.ecg_id])
    np.save(work / "oof_ids.npy", fit.ecg_id.to_numpy())
    for key in ("bags", "lead_means", "rhythm", "combined"):
        np.save(work / f"oof_{key}.npy", np.concatenate([p[key] for p in parts])[rows])


def probability_summary(bags: np.ndarray, abnormal: np.ndarray) -> dict[str, float]:
    """Distribution of non-NORM bag probabilities over records and heads."""
    p = 1.0 / (1.0 + np.exp(-bags[:, abnormal].astype(np.float64)))
    return {
        "records": int(len(p)),
        "mean": float(p.mean()),
        "q50": float(np.percentile(p, 50)),
        "q90": float(np.percentile(p, 90)),
        "q99": float(np.percentile(p, 99)),
        "mean_record_sum": float(p.sum(axis=1).mean()),
        "mean_record_max": float(p.max(axis=1).mean()),
    }


def crossfit_summary(fit: pd.DataFrame, y: np.ndarray, mask: np.ndarray, work: Path, smoke: bool) -> dict:
    """Out-of-fold detection AUROCs and the calibration gap against the 065 ensemble."""
    heads = heads065()
    abnormal = np.array([k for k, h in enumerate(heads) if h != mil.REFERENCE_HEAD])
    bags = np.load(work / "oof_bags.npy")
    heldout = mil.heldout_mask(load_partition().query("partition == 'fit'").patient_id.to_numpy())
    heldout = heldout[: len(fit)]
    ensemble = np.mean([np.load(MIL065 / f"heldout_bags_{name}.npz")["bag"] for name in SEEDS065], axis=0)
    ensemble_ids = np.load(MIL065 / f"heldout_bags_{SEEDS065[0]}.npz")["ecg_id"].astype(int)
    position = {int(e): k for k, e in enumerate(fit.ecg_id)}
    shared = np.array([position[e] for e in ensemble_ids if e in position], int)
    ens_rows = np.array([k for k, e in enumerate(ensemble_ids) if e in position], int)
    train = training_records(fit)
    train_rows = np.array([position[int(e)] for e in train.ecg_id])
    scored_bags = scored_mil()["bags"]
    gap_logit = bags[shared][:, abnormal] - ensemble[ens_rows][:, abnormal]
    summary = {
        "oof_macro_auroc_all_fit": mil.macro_auroc(y, mask, bags, mil.MIN_HELDOUT_POSITIVES)[0],
        "oof_macro_auroc_065_heldout_rows": mil.macro_auroc(
            y[heldout], mask[heldout], bags[heldout], mil.MIN_HELDOUT_POSITIVES
        )[0],
        "ensemble065_macro_auroc_065_heldout_rows": mil.macro_auroc(
            y[shared], mask[shared], ensemble[ens_rows], mil.MIN_HELDOUT_POSITIVES
        )[0],
        "per_head_oof_auroc": dict(
            zip(heads, np.nan_to_num(mil.macro_auroc(y, mask, bags, 5)[1], nan=-1).tolist(), strict=True)
        ),
        "calibration": {
            "oof_training_records": probability_summary(bags[train_rows], abnormal),
            "oof_065_heldout_rows": probability_summary(bags[shared], abnormal),
            "ensemble065_065_heldout_rows": probability_summary(ensemble[ens_rows], abnormal),
            "ensemble065_scored_records": probability_summary(scored_bags, abnormal),
            "logit_difference_oof_minus_ensemble_065_heldout": {
                "mean": float(gap_logit.mean()),
                "mean_abs": float(np.abs(gap_logit).mean()),
                "correlation": float(
                    np.corrcoef(bags[shared][:, abnormal].ravel(), ensemble[ens_rows][:, abnormal].ravel())[
                        0, 1
                    ]
                ),
            },
        },
        "folds": [
            json.loads((folders(smoke)[0] / f"crossfit_fold{f}.json").read_text()) for f in range(cv.FOLDS)
        ],
        "065_reference_heldout_macro_auroc": {
            name: json.loads((MIL065 / f"history_{name}.json").read_text())["best_heldout_macro_auroc"]
            for name in SEEDS065
        },
    }
    return summary


def scored_mil() -> dict[str, np.ndarray]:
    """065 three-seed summaries of the scored records from the 066 work folder, rows in scored order."""
    ids = np.load(WORK066 / "mil_ids.npy").astype(int)
    row = {e: k for k, e in enumerate(ids.tolist())}
    scored = scored_ecg_ids().astype(int)
    rows = np.array([row[e] for e in scored])
    return {"ecg_id": scored, "bags": np.load(WORK066 / "mil_bags.npy")[rows]}


def check_scored_mil() -> dict[str, Any]:
    """066's recomputed 065 bag logits of the scored records against 065's saved bags (exact)."""
    ids = np.load(WORK066 / "mil_ids.npy").astype(int)
    bags = np.load(WORK066 / "mil_bags.npy")
    row = {e: k for k, e in enumerate(ids.tolist())}
    largest, compared = 0.0, 0
    for set_name in ("ecgad_dev", "evaluation"):
        with np.load(MIL065 / f"bags_{set_name}.npz") as saved:
            for ecg_id, bag in zip(saved["ecg_id"].astype(int), saved["mil_lse"], strict=True):
                if ecg_id in row:
                    largest = max(largest, float(np.abs(bags[row[ecg_id]] - bag).max()))
                    compared += 1
    missing = set(scored_ecg_ids().astype(int)) - set(row)
    return {
        "compared": compared,
        "max_abs_difference": largest,
        "missing_scored": len(missing),
        "pass": bool(largest <= 1e-3 and not missing),
    }


def record_sets(smoke: bool) -> tuple[np.ndarray, pd.DataFrame]:
    """Scored records and combiner training records (smoke: the first 16 of each)."""
    scored = scored_ecg_ids()
    train = training_records(fit_table(smoke))
    if smoke:
        scored, train = scored[:SMOKE_RECORDS], train.iloc[:SMOKE_TRAIN].reset_index(drop=True)
    if holdout_ids() & set(scored.astype(int)):
        raise ValueError("An ECGAD holdout record would be mapped")
    if forbidden_ids() & set(train.ecg_id.astype(int)):
        raise ValueError("A training record is an ECGAD or evaluation record")
    return scored, train


def _mil_source(ids_path: Path, prefix: Path) -> dict[str, Any]:
    ids = np.load(ids_path).astype(int)
    source = {"row": dict(zip(ids.tolist(), range(len(ids)), strict=True))}
    for name in ("bags", "lead_means", "rhythm", "combined"):
        source[name] = np.load(f"{prefix}{name}.npy", mmap_mode="r")
    return source


def _features_init(work: str) -> None:
    threadpool_limits(limits=1)
    signals, index = open_cache()
    _STATE["signals"] = signals
    _STATE["row"] = dict(zip(index.ecg_id.astype(int), range(len(index)), strict=True))
    _STATE["index"] = index.set_index("ecg_id")
    _STATE["train"] = _mil_source(Path(work) / "oof_ids.npy", Path(work) / "oof_")
    _STATE["scored"] = _mil_source(WORK066 / "mil_ids.npy", WORK066 / "mil_")
    with np.load(OUT066 / "quality/q.npz") as q:
        _STATE["q_scored"] = dict(zip(q["scored_ids"].astype(int).tolist(), q["q_scored"], strict=True))
        _STATE["q_train"] = dict(zip(q["fit_ids"].astype(int).tolist(), q["q_fit_oof"], strict=True))
    norm_ids = np.load(NORMATIVE062 / "ecg_ids.npy")
    _STATE["norm_row"] = dict(zip(norm_ids.astype(int).tolist(), range(len(norm_ids)), strict=True))
    _STATE["norm_maps"] = np.load(NORMATIVE062 / "maps_normative_v1.npy", mmap_mode="r")
    with (NORMATIVE062 / "normative.pkl").open("rb") as handle:
        _STATE["normative"] = pickle.load(handle)  # noqa: S301
    with (LOCBENCH061 / "ub_references.pkl").open("rb") as handle:
        _STATE["references"] = pickle.load(handle)  # noqa: S301
    _STATE["heads"] = heads065()


def normative_for(ecg_id: int, signal: np.ndarray) -> np.ndarray:
    """Return the 062 ``normative_v1`` map: saved when present, else recomputed (066 rule)."""
    if ecg_id in _STATE["norm_row"]:
        return np.asarray(_STATE["norm_maps"][_STATE["norm_row"][ecg_id]], np.float64)
    meta = _STATE["index"].loc[ecg_id]
    maps, _ = nm.record_maps(signal, float(meta.age), float(meta.sex), _STATE["normative"])
    return maps[nm.PRIMARY].astype(np.float16).astype(np.float64)


def _features(task: tuple[int, str]) -> dict[str, Any]:
    """Units and raw unit means of one record; MIL evidence from the cross-fit or the 065 ensemble."""
    ecg_id, kind = task
    stored = _STATE["signals"][_STATE["row"][ecg_id]]
    raw = np.asarray(stored, np.float64)
    exact = exact_signal(stored)
    intervals = cb.delineated_components(raw)
    masks = cb.component_masks(intervals)
    heads = _STATE["heads"]
    abnormal = [k for k, h in enumerate(heads) if h != mil.REFERENCE_HEAD]
    source = _STATE[kind]
    m = source["row"][ecg_id]
    prob = 1.0 / (1.0 + np.exp(-np.asarray(source["bags"][m], np.float64)[abnormal]))
    means = np.asarray(source["lead_means"][m])[abnormal]
    rhythm = {h: np.asarray(source["rhythm"][m, j], np.float32) for j, h in enumerate(cb.RHYTHM_HEADS)}
    fields = cv.raw_fields([heads[k] for k in abnormal], prob, means, rhythm, masks)
    residual = residual_map(exact, _STATE["references"])
    fields["mil"] = mil.upsample(np.asarray(source["combined"][m], np.float32)).astype(np.float64)
    fields["normative"] = normative_for(ecg_id, raw)
    fields["residual"] = np.zeros_like(exact) if residual is None else residual
    fields["amplitude"] = amplitude_map(exact)
    units = cb.unit_table(intervals)
    unit_raw = (
        np.column_stack([cb.unit_means(units, fields[name]) for name in cv.RAW_MAPS])
        if len(units)
        else (np.zeros((0, len(cv.RAW_MAPS))))
    )
    q = _STATE["q_scored" if kind == "scored" else "q_train"][ecg_id]
    return {
        "ecg_id": ecg_id,
        "kind": kind,
        "beats": int(len(intervals)),
        "units": units,
        "unit_raw": unit_raw,
        "q": np.asarray(q, np.float64),
    }


def features_phase(smoke: bool) -> None:
    """Compute unit features of the training records (cross-fit MIL) and scored records (065 MIL)."""
    out, work = folders(smoke)
    began = time.time()
    scored, train = record_sets(smoke)
    tasks = [(int(e), "train") for e in train.ecg_id] + [(int(e), "scored") for e in scored]
    results: dict[str, dict[int, dict]] = {"train": {}, "scored": {}}
    with Pool(WORKERS, initializer=_features_init, initargs=(str(work),)) as pool:
        for k, result in enumerate(pool.imap_unordered(_features, tasks, chunksize=4)):
            results[result["kind"]][result["ecg_id"]] = result
            if (k + 1) % 500 == 0:
                print(f"features {k + 1}/{len(tasks)} {time.time() - began:.0f}s", flush=True)
    for kind, table in results.items():
        with (work / f"features_{kind}.pkl").open("wb") as handle:
            pickle.dump(table, handle, protocol=pickle.HIGHEST_PROTOCOL)
    receipt = {
        "train_records": len(results["train"]),
        "scored_records": len(results["scored"]),
        "scored_without_beats": [e for e, r in results["scored"].items() if r["beats"] == 0],
        "train_without_beats": [e for e, r in results["train"].items() if r["beats"] == 0],
        "seconds": time.time() - began,
    }
    write_json_atomic(out / "features_receipt.json", receipt)
    print(json.dumps(receipt), flush=True)


def load_features(work: Path, kind: str) -> dict[int, dict]:
    """Load the saved unit features of one record kind."""
    with (work / f"features_{kind}.pkl").open("rb") as handle:
        return pickle.load(handle)  # noqa: S301


def training_design(train: pd.DataFrame, features: dict[int, dict]) -> dict[str, Any]:
    """
    Full stacked design, union weak labels and folds of the training records with both classes.

    Returns
    -------
    dict
        ``x``, ``y``, ``starts``, ``folds`` (per record), ``ecg_ids``, ``patients`` and ``dropped``.
    """
    designs, positives, folds, ids, patients, dropped = [], [], [], [], [], []
    for record in train.itertuples(index=False):
        r = features[int(record.ecg_id)]
        y = (
            cv.union_labels(r["units"], record.statements, REGION_TABLE)
            if len(r["units"])
            else np.zeros(0, bool)
        )
        if not y.any() or y.all():
            dropped.append(int(record.ecg_id))
            continue
        designs.append(cv.unit_design(r["unit_raw"], r["q"], r["units"]))
        positives.append(y)
        folds.append(int(record.fold))
        ids.append(int(record.ecg_id))
        patients.append(int(record.patient_id))
    x, y, starts = cv.stack(designs, positives)
    return {
        "x": x,
        "y": y,
        "starts": starts,
        "folds": np.array(folds),
        "ecg_ids": np.array(ids),
        "patients": np.array(patients),
        "dropped": dropped,
    }


def cross_validate(data: dict[str, Any], cols: np.ndarray, l2: float) -> dict[str, float]:
    """Patient-grouped five-fold held-out listwise loss (data term) and within-record weak-label AUROC."""
    x, y, starts, folds = data["x"][:, cols], data["y"], data["starts"], data["folds"]
    losses, aurocs = [], []
    for fold in range(cv.FOLDS):
        train = np.flatnonzero(folds != fold)
        test = np.flatnonzero(folds == fold)
        w = cv.fit_listwise(*cv.subset(x, y, starts, train), l2)
        xt, yt, st = cv.subset(x, y, starts, test)
        losses.append(cv.listwise_loss(w, xt, yt, st, 0.0)[0] * len(test))
        aurocs.append(cv.within_record_auroc(xt @ w, yt, st) * len(test))
    n = len(folds)
    return {"loss": float(sum(losses) / n), "auroc": float(sum(aurocs) / n)}


def select_linear(data: dict[str, Any], variant: str) -> dict[str, Any]:
    """Choose (a, L2) for ``units_v2`` or L2 for the other linear variants by the lowest CV loss."""
    grid = cv.SOFT_GRID if variant == "units_v2" else (cv.SOFT_GRID[0],)
    rows = []
    for a in grid:
        cols = cv.columns(cv.feature_set("units_learned" if variant == "units_v1_data" else variant, a))
        for l2 in cv.L2_GRID:
            rows.append({"a": a, "l2": l2, **cross_validate(data, cols, l2)})
            print(f"{variant} a={a} l2={l2} {rows[-1]}", flush=True)
    best = min(rows, key=lambda r: (r["loss"], -r["l2"]))
    return {
        "grid": rows,
        "a": best["a"] if variant == "units_v2" else None,
        "l2": best["l2"],
        "cv_loss": best["loss"],
        "cv_auroc": best["auroc"],
    }


def hgb_model() -> HistGradientBoostingClassifier:
    """Build the nonlinear secondary: depth-3 gradient-boosted trees, pointwise logistic loss."""
    return HistGradientBoostingClassifier(**HGB, early_stopping=False, random_state=cv.SEED)


def hgb_cv(data: dict[str, Any], cols: np.ndarray) -> float:
    """Patient-grouped five-fold within-record weak-label AUROC of the boosted trees."""
    x, y, starts, folds = data["x"][:, cols], data["y"], data["starts"], data["folds"]
    total = 0.0
    for fold in range(cv.FOLDS):
        xa, ya, sa = cv.subset(x, y, starts, np.flatnonzero(folds != fold))
        model = hgb_model().fit(xa, ya, sample_weight=cv.record_weights(sa))
        xt, yt, st = cv.subset(x, y, starts, np.flatnonzero(folds == fold))
        total += cv.within_record_auroc(model.decision_function(xt), yt, st) * int((folds == fold).sum())
    return float(total / len(folds))


def weights066() -> np.ndarray:
    """066 ``units_learned`` weights in ``combination066.UNIT_FEATURES`` order."""
    saved = json.loads((OUT066 / "result.json").read_text())["maps"]["combiner_weights"]
    return np.array([saved[name] for name in cb.UNIT_FEATURES])


def fit_models(data: dict[str, Any]) -> dict[str, Any]:
    """Select and fit the linear variants and the boosted trees on all training records."""
    models: dict[str, Any] = {}
    for variant in LINEAR:
        choice = select_linear(data, variant)
        a = choice["a"] if choice["a"] is not None else cv.SOFT_GRID[0]
        names = cv.feature_set("units_learned" if variant == "units_v1_data" else variant, a)
        w = cv.fit_listwise(data["x"][:, cv.columns(names)], data["y"], data["starts"], choice["l2"])
        models[variant] = {
            **choice,
            "features": list(names),
            "weights": dict(zip(names, w.tolist(), strict=True)),
        }
    names = models["units_v2"]["features"]
    cols = cv.columns(tuple(names))
    began = time.time()
    hgb_auroc = hgb_cv(data, cols)
    trees = hgb_model().fit(data["x"][:, cols], data["y"], sample_weight=cv.record_weights(data["starts"]))
    models["units_v2_hgb"] = {
        "features": list(names),
        "params": HGB,
        "cv_auroc": hgb_auroc,
        "model": trees,
        "seconds": time.time() - began,
    }
    return models


def unit_scores(models: dict[str, Any], design: np.ndarray) -> dict[str, np.ndarray]:
    """Scores of every map's model on one record's full design, plus the 066 rebuild."""
    scores = {}
    for variant in LINEAR:
        names = tuple(models[variant]["features"])
        scores[variant] = design[:, cv.columns(names)] @ np.array(
            [models[variant]["weights"][n] for n in names]
        )
    if len(design):
        trees = models["units_v2_hgb"]
        scores["units_v2_hgb"] = trees["model"].decision_function(
            design[:, cv.columns(tuple(trees["features"]))]
        )
    else:
        scores["units_v2_hgb"] = np.zeros(0)
    scores["units_learned_rebuilt"] = design[:, cv.columns(cb.UNIT_FEATURES)] @ weights066()
    return scores


def _paint_init(work: str, names: tuple[str, ...]) -> None:
    threadpool_limits(limits=1)
    signals, index = open_cache()
    _STATE["signals"] = signals
    _STATE["row"] = dict(zip(index.ecg_id.astype(int), range(len(index)), strict=True))
    _STATE["maps"] = {
        name: np.lib.format.open_memmap(Path(work) / f"{name}.npy", mode="r+") for name in names
    }


def _paint(task: tuple[int, int, np.ndarray, dict[str, np.ndarray]]) -> int:
    position, ecg_id, units, scores = task
    amplitude = amplitude_map(exact_signal(_STATE["signals"][_STATE["row"][ecg_id]]))
    for name, values in scores.items():
        painted = cb.units_map(values, units, amplitude)
        if not np.isfinite(painted).all():
            raise ValueError(f"Nonfinite {name} map for {ecg_id}")
        _STATE["maps"][name][position] = painted
    return position


def descriptive_066(data: dict[str, Any]) -> dict[str, Any]:
    """Score the 066 weights on 068 training records outside 066's training set (weak-label AUROC)."""
    train066 = set(training066())
    keep = np.array([e not in train066 for e in data["ecg_ids"]])
    x, y, st = cv.subset(data["x"], data["y"], data["starts"], np.flatnonzero(keep))
    scores = x[:, cv.columns(cb.UNIT_FEATURES)] @ weights066()
    return {"records": int(keep.sum()), "auroc": cv.within_record_auroc(scores, y, st)}


def training066() -> list[int]:
    """066 combiner records: held-out 065 fit rows with one region statement and region-set rhythms."""
    fit = fit_table(False)
    train = training_records(fit)
    heldout = set(fit.ecg_id[mil.heldout_mask(fit.patient_id.to_numpy())].astype(int))
    rows = train[train.ecg_id.isin(heldout) & train.region_rhythm & (train.statements.apply(len) == 1)]
    return rows.ecg_id.astype(int).tolist()


def fit_phase(smoke: bool) -> None:
    """Fit the combiners on the training records and paint every declared map for the scored records."""
    out, work = folders(smoke)
    began = time.time()
    scored, train = record_sets(smoke)
    data = training_design(train, load_features(work, "train"))
    models = fit_models(data)
    features = load_features(work, "scored")
    names = (*MAPS, "units_learned_rebuilt")
    for name in names:
        np.lib.format.open_memmap(
            work / f"{name}.npy", mode="w+", dtype=np.float32, shape=(len(scored), 12, 5000)
        )
    tasks = []
    for k, ecg_id in enumerate(scored.astype(int).tolist()):
        r = features[ecg_id]
        design = cv.unit_design(r["unit_raw"], r["q"], r["units"]) if len(r["units"]) else np.zeros((0, 32))
        tasks.append((k, ecg_id, r["units"], unit_scores(models, design)))
    with Pool(WORKERS, initializer=_paint_init, initargs=(str(work), names)) as pool:
        list(pool.imap_unordered(_paint, tasks, chunksize=16))
    (work / "scored_ids.json").write_text(json.dumps([int(e) for e in scored]))
    files = {
        name: {"file": str(work / f"{name}.npy"), "sha256": sha256_file(work / f"{name}.npy")}
        for name in names
    }
    with (work / "hgb.pkl").open("wb") as handle:
        pickle.dump(models["units_v2_hgb"]["model"], handle)
    receipt = {
        "created": utc_now(),
        "git_head": git_head(ROOT),
        "training_candidates": int(len(train)),
        "training_records": int(len(data["ecg_ids"])),
        "training_patients": int(len(np.unique(data["patients"]))),
        "training_dropped": data["dropped"],
        "training_units": int(len(data["y"])),
        "training_positive_units": int(data["y"].sum()),
        "training_multi_statement": int((train.statements.apply(len) > 1).sum()),
        "training_failing_region_rhythm_rule": int((~train.region_rhythm).sum()),
        "training_families": family_counts(train),
        "training_statements": pd.Series([s for x in train.statements for s in x])
        .value_counts()
        .sort_index()
        .to_dict(),
        "models": {k: {n: v for n, v in m.items() if n != "model"} for k, m in models.items()},
        "weights066_on_new_records": None if smoke else descriptive_066(data),
        "files": files,
        "seconds": time.time() - began,
    }
    write_json_atomic(out / "fit_receipt.json", receipt)
    print(json.dumps({k: v for k, v in receipt.items() if k != "files"}, indent=1, default=str), flush=True)


def reproduce_bar() -> dict[str, Any]:
    """Rescore 061's saved ``qrs_prior`` ECGAD maps and match its per-record and mean A2 within 1e-12."""
    result = rescore_tables(LOCBENCH061 / "tables" / BAR)
    target = json.loads((LOCBENCH061 / "result.json").read_text())["bar"]["A2"]["value"]
    result["A2_saved"] = target
    result["pass"] = bool(result["pass"] and abs(result["A2"] - target) <= TOLERANCE)
    if not result["pass"]:
        raise ValueError(f"Bar not reproduced: {result}")
    return result


def rescore_tables(folder: Path) -> dict[str, Any]:
    """Rescore a saved tables folder's ECGAD maps and compare per-record A2 with its ``ecgad.csv``."""
    bench = load_benchmark()
    if (
        json.loads((folder / "meta.json").read_text())["ecgad_ecg_ids"]
        != bench.ecgad.ecg_id.astype(int).tolist()
    ):
        raise ValueError(f"ECGAD order of {folder} differs from the benchmark")
    maps = np.load(folder / "ecgad_maps.npy", mmap_mode="r")
    values = np.array(
        [
            ecgad_record_metrics(np.asarray(m, np.float32), y)["A2"]
            for m, y in zip(maps, bench.ecgad_labels, strict=True)
        ]
    )
    saved = pd.read_csv(folder / "ecgad.csv")["A2"].to_numpy()
    largest = float(np.nanmax(np.abs(values - saved)))
    same = bool(np.array_equal(np.isnan(values), np.isnan(saved)))
    return {
        "per_record_max_abs_difference": largest,
        "records": int((~np.isnan(values)).sum()),
        "A2": float(np.nanmean(values)),
        "pass": bool(same and largest <= TOLERANCE),
    }


def reproduce_066(work: Path) -> dict[str, Any]:
    """Rescore 066's saved ``units_learned`` tables and compare the rebuild from this run's features."""
    folder = REFERENCES["units_learned"]
    saved_a2 = json.loads((folder.parent / "summary.json").read_text())["metrics"]["A2"]["value"]
    tables = rescore_tables(folder)
    tables["A2_saved"] = saved_a2
    tables["pass"] = bool(tables["pass"] and abs(tables["A2"] - saved_a2) <= TOLERANCE)
    scored = json.loads((work / "scored_ids.json").read_text())
    if scored != json.loads((WORK066 / "scored_ids.json").read_text()):
        raise ValueError("Scored records differ from 066")
    rebuilt = np.load(work / "units_learned_rebuilt.npy", mmap_mode="r")
    original = np.load(WORK066 / "units_learned.npy", mmap_mode="r")
    largest = max(
        float(np.abs(rebuilt[k : k + 256] - original[k : k + 256]).max()) for k in range(0, len(scored), 256)
    )
    bench = load_benchmark()
    row = {e: k for k, e in enumerate(scored)}
    values = np.array(
        [
            ecgad_record_metrics(np.asarray(rebuilt[row[int(e)]], np.float32), y)["A2"]
            for e, y in zip(bench.ecgad.ecg_id, bench.ecgad_labels, strict=True)
        ]
    )
    saved = pd.read_csv(folder / "ecgad.csv")["A2"].to_numpy()
    rebuild_diff = float(np.nanmax(np.abs(values - saved)))
    rebuild = {
        "map_max_abs_difference": largest,
        "per_record_A2_max_abs_difference": rebuild_diff,
        "A2": float(np.nanmean(values)),
        "pass": bool(rebuild_diff <= TOLERANCE),
    }
    result = {"saved_tables": tables, "rebuild": rebuild, "pass": bool(tables["pass"] and rebuild["pass"])}
    if not result["pass"]:
        raise ValueError(f"066 units_learned not reproduced: {result}")
    return result


def run_score(maps: Path, ids: Path, name: str, target: Path, compare: list[Path]) -> dict[str, Any]:
    """Run the LocBench scoring CLI and return its summary."""
    command = [
        str(ROOT / ".venv/bin/python"),
        "-m",
        "scripts.experiments.run_locbench_score",
        "--maps",
        str(maps),
        "--ids",
        str(ids),
        "--name",
        name,
        "--out",
        str(target),
    ]
    for folder in compare:
        command += ["--compare", str(folder)]
    subprocess.run(command, check=True, cwd=ROOT)
    return json.loads((target / "summary.json").read_text())


def copy_references(out: Path) -> dict[str, Path]:
    """Copy the reference tables into this run's folder (the strata and noise CLIs write into them)."""
    copies = {}
    for name, source in REFERENCES.items():
        target = out / "ref" / name
        if not target.exists():
            shutil.copytree(source, target, ignore=shutil.ignore_patterns("strata.*", "noise_std.*"))
        copies[name] = target
    return copies


def decision(interval: dict[str, Any]) -> str:
    """Primary decision words from a paired interval."""
    if interval["ci_low"] > 0:
        return "localizes"
    if interval["ci_high"] < 0:
        return "below"
    return "matches"


def score_maps(out: Path, work: Path, refs: dict[str, Path]) -> dict[str, dict]:
    """Score every declared map with the LocBench CLI, with its paired comparisons."""
    base = [refs[BAR], refs["units_learned"], refs["textbook"], refs["residual056"]]
    extra = {"units_v2": ["units_v2_nolead", "units_v2_hgb", "units_v1_data"]}
    summaries = {}
    for name in ("units_v2_nolead", "units_v2_hgb", "units_v1_data", "units_v2"):
        compare = base + [out / "locbench" / other / "tables" for other in extra.get(name, [])]
        summaries[name] = run_score(
            work / f"{name}.npy", work / "scored_ids.json", name, out / "locbench" / name, compare
        )
        print(f"{name}: A2 {summaries[name]['metrics']['A2']['value']:.4f}", flush=True)
    return summaries


def run_secondary_clis(out: Path, refs: dict[str, Path]) -> None:
    """Strata (``qrs_prior`` reference first) and standardized noise readings for every map and reference."""
    tables = [out / "locbench" / name / "tables" for name in MAPS]
    others = [refs["units_learned"], refs["textbook"], refs["residual056"]]
    python = str(ROOT / ".venv/bin/python")
    strata = [
        python,
        "-m",
        "scripts.experiments.run_locbench_strata",
        str(refs[BAR]),
        *map(str, tables + others),
    ]
    subprocess.run(strata, check=True, cwd=ROOT)
    noise = [
        python,
        "-m",
        "scripts.experiments.run_locbench_noise",
        *map(str, tables + others),
        str(refs[BAR]),
    ]
    subprocess.run(noise, check=True, cwd=ROOT)


def separation(work: Path) -> dict[str, Any]:
    """Data-separation checks of every fitted quantity."""
    forbidden = forbidden_ids()
    fit = fit_table(False)
    oof = set(np.load(work / "oof_ids.npy").astype(int).tolist())
    train = set(training_records(fit).ecg_id.astype(int))
    clean = set(fit.ecg_id[fit.clean_normal].astype(int))
    scored = set(json.loads((work / "scored_ids.json").read_text()))
    checks = {
        "crossfit_records": len(oof),
        "crossfit_forbidden": len(oof & forbidden),
        "crossfit_outside_fit": len(oof - set(fit.ecg_id.astype(int))),
        "combiner_records": len(train),
        "combiner_forbidden": len(train & forbidden),
        "combiner_fit_clean_normals": len(train & clean),
        "mapped_ecgad_holdout": len(scored & holdout_ids()),
    }
    checks["pass"] = all(v == 0 for k, v in checks.items() if not k.endswith("_records"))
    if not checks["pass"]:
        raise ValueError(f"Data separation failed: {checks}")
    return checks


def readings(out: Path, refs: dict[str, Path], summaries: dict[str, dict]) -> tuple[dict, dict]:
    """Every map's LocBench metrics, strata and noise readings, and the references' strata and noise."""
    mine = {}
    for name in MAPS:
        folder = out / "locbench" / name / "tables"
        mine[name] = {
            "metrics": summaries[name]["metrics"],
            "paired": summaries[name]["paired"],
            "strata": json.loads((folder / "strata.json").read_text()),
            "noise_std": json.loads((folder / "noise_std.json").read_text()),
            "maps_sha256": summaries[name]["inputs"]["maps_sha256"],
        }
    refs_out = {
        name: {
            "metrics": json.loads((REFERENCES[name].parent / "summary.json").read_text())["metrics"]
            if (REFERENCES[name].parent / "summary.json").exists()
            else None,
            "strata": json.loads((folder / "strata.json").read_text()),
            "noise_std": json.loads((folder / "noise_std.json").read_text()),
        }
        for name, folder in refs.items()
    }
    return mine, refs_out


def score_phase() -> None:
    """Integrity checks, LocBench scoring of every map, strata and standardized noise, result.json."""
    out, work = folders(False)
    began = time.time()
    scored = np.array(json.loads((work / "scored_ids.json").read_text()), np.int64)
    if not np.array_equal(scored, scored_ecg_ids()):
        raise ValueError("Scored records changed")
    integrity = {
        "bar": reproduce_bar(),
        "units_learned066": reproduce_066(work),
        "scored_mil": check_scored_mil(),
    }
    integrity["separation"] = separation(work)
    refs = copy_references(out)
    summaries = score_maps(out, work, refs)
    run_secondary_clis(out, refs)
    mine, refs_out = readings(out, refs, summaries)
    v2 = summaries["units_v2"]["paired"]
    primary = v2[BAR]["A2"]
    a2 = summaries["units_v2"]["metrics"]["A2"]["value"]
    a2_066 = json.loads((REFERENCES["units_learned"].parent / "summary.json").read_text())["metrics"]["A2"][
        "value"
    ]
    decisions = {
        "primary": {
            "map": "units_v2",
            "against": BAR,
            "A2_difference": primary,
            "decision": decision(primary),
        },
        "units_v2_vs_units_learned": v2["units_learned"]["A2"],
        "units_v2_vs_textbook": v2["textbook"]["A2"],
        "units_v2_vs_units_v2_nolead": v2["units_v2_nolead"]["A2"],
        "units_v2_A2": a2,
        "units_learned_A2": a2_066,
        "add_units_v2_to_067": bool(a2 > a2_066),
    }
    result = {
        "experiment": "068",
        "protocol": "docs/experiment-068-combiner-v2.md",
        "protocol_sha256": sha256_file(ROOT / "docs/experiment-068-combiner-v2.md"),
        "created": utc_now(),
        "git_head": git_head(ROOT),
        "inputs": inputs_record(),
        "integrity": integrity,
        "gpu_profile": json.loads((out / "gpu_profile.json").read_text()),
        "crossfit": json.loads((out / "crossfit_summary.json").read_text()),
        "features": json.loads((out / "features_receipt.json").read_text()),
        "fit": json.loads((out / "fit_receipt.json").read_text()),
        "readings": mine,
        "reference_readings": refs_out,
        "decisions": decisions,
        "score_seconds": time.time() - began,
    }
    write_json_atomic(out / "result.json", result)
    print(json.dumps(decisions, indent=1), flush=True)


def inputs_record() -> dict[str, Any]:
    """Hashes of the inputs and sources of the run."""
    return {
        "partition": to_stored(PARTITION_PATH),
        "partition_sha256": sha256_file(PARTITION_PATH),
        "cache_dir": str(CACHE_DIR),
        "mil065_result_sha256": sha256_file(MIL065 / "result.json"),
        "mil065_weights_sha256": {n: sha256_file(MIL065 / f"weights_{n}.pt") for n in SEEDS065},
        "work066_mil_sha256": {
            n: sha256_file(WORK066 / f"mil_{n}.npy")
            for n in ("ids", "bags", "lead_means", "rhythm", "combined")
        },
        "quality066_sha256": sha256_file(OUT066 / "quality/q.npz"),
        "result066_sha256": sha256_file(OUT066 / "result.json"),
        "normative062_maps_sha256": sha256_file(NORMATIVE062 / "maps_normative_v1.npy"),
        "normative062_model_sha256": sha256_file(NORMATIVE062 / "normative.pkl"),
        "locbench061_result_sha256": sha256_file(LOCBENCH061 / "result.json"),
        "ub_references_sha256": sha256_file(LOCBENCH061 / "ub_references.pkl"),
        "sources_sha256": {path: sha256_file(ROOT / path) for path in SOURCES},
    }


def main() -> None:
    """Command-line entry point."""
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=["profile", "crossfit", "features", "fit", "score"])
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(WORKERS)
    if args.phase == "profile":
        profile_phase()
    elif args.phase == "crossfit":
        crossfit_phase(args.smoke)
    elif args.phase == "features":
        features_phase(args.smoke)
    elif args.phase == "fit":
        fit_phase(args.smoke)
    elif args.smoke:
        raise ValueError("The smoke run is not scored")
    else:
        score_phase()


if __name__ == "__main__":
    main()
