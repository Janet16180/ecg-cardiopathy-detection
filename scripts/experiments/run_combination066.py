"""
Experiment 066: factorized combination maps, lead-quality suppression and a learned unit combiner.

Phases::

    run_combination066 mil       # 065 bag logits and evidence for scored and combiner records (GPU, lock)
    run_combination066 quality   # lead-quality classifier on fit records (CPU)
    run_combination066 maps      # five declared maps and the combiner (CPU, 8 processes)
    run_combination066 score     # integrity checks, LocBench, strata, standardized noise, result.json
    run_combination066 separation  # record the data-separation checks in result.json

``--smoke`` uses 16 scored and 16 combiner records and a separate folder; nothing is scored. See
``docs/experiment-066-combination.md``.
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import shutil
import subprocess
import time
from multiprocessing import Pool
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from ecg_experiment import ROOT
from ecg_experiment import combination066 as cb
from ecg_experiment import mil065 as mil
from ecg_experiment import normative_map062 as nm
from ecg_experiment.ecgad import load_split
from ecg_experiment.eda.ptbxl import load_metadata, load_statements
from ecg_experiment.files import sha256_file, write_json_atomic
from ecg_experiment.gpu import gpu_lock
from ecg_experiment.locbench import (
    REGION_RHYTHMS,
    _noise_set,
    _statement_sets,
    auroc,
    ecgad_record_metrics,
    load_benchmark,
    scored_ecg_ids,
)
from ecg_experiment.locbench_baselines061 import amplitude_map, exact_signal, residual_map
from ecg_experiment.locbench_cache import CACHE_DIR, open_cache
from ecg_experiment.locbench_data import NOISE_COLUMNS, PARTITION_PATH, REGION_TABLE, load_partition
from ecg_experiment.paths import to_stored
from ecg_experiment.provenance import git_head, utc_now
from ecg_experiment.waveforms import LEADS

OUT = ROOT / "outputs/experiment066_combination_v1"
SMOKE_OUT = ROOT / "outputs/experiment066_combination_smoke"
WORK = Path(os.environ.get("ECG066_WORK", "/tmp/ecg066-work"))
MIL065 = ROOT / "outputs/experiment065_mil_v1"
LOCBENCH061 = ROOT / "outputs/experiment061_locbench_v1"
NORMATIVE062 = ROOT / "outputs/experiment062_normative_v1"
MIL_WEIGHTS = ("lse_65065", "lse_65066", "lse_65067")
BASELINES = ("random", "amplitude", "qrs_prior", "U_B", "residual056")
BAR = "qrs_prior"
LEXICOGRAPHIC = "qrs_lexicographic"
MAPS = ("factor", "textbook", "textbook_true", "factor_q", "units_learned")
PAINTED = ("factor", "textbook", "textbook_true", "factor_q")
WORKERS = 8
CHUNK = 128
SMOKE_RECORDS = 16
TOLERANCE = 1e-12
EVIDENCE_TOLERANCE = 2e-3
BAG_TOLERANCE = 1e-3
SOURCES = (
    "ecg_experiment/combination066.py",
    "scripts/experiments/run_combination066.py",
    "docs/experiment-066-combination.md",
    "ecg_experiment/mil065.py",
    "ecg_experiment/normative_map062.py",
    "ecg_experiment/morphology_boundaries.py",
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


def combiner_records() -> pd.DataFrame:
    """
    Select held-out 065 fit rows with one REGION_TABLE statement and only region-set rhythms.

    Returns
    -------
    pd.DataFrame
        ``ecg_id``, ``patient_id``, ``statement``.
    """
    partition = load_partition()
    fit = partition[partition.partition == "fit"].reset_index(drop=True)
    heldout = mil.heldout_mask(fit.patient_id.to_numpy())
    meta, rhythm, _, _, _ = _statement_sets()
    rows = []
    for record in fit[heldout].itertuples(index=False):
        codes = [code for code in record.region_statements.split(";") if code]
        if len(codes) != 1:
            continue
        if any(code in rhythm and code not in REGION_RHYTHMS for code in meta.at[record.ecg_id, "scp_codes"]):
            continue
        rows.append(
            {"ecg_id": int(record.ecg_id), "patient_id": int(record.patient_id), "statement": codes[0]}
        )
    table = pd.DataFrame(rows)
    if forbidden_ids() & set(table.ecg_id):
        raise ValueError("A combiner record is an ECGAD or evaluation record")
    clean = set(fit.ecg_id[fit.clean_normal].astype(int))
    if clean & set(table.ecg_id):
        raise ValueError("A combiner record is a fit clean normal (062 reference)")
    return table


def record_sets(smoke: bool) -> tuple[np.ndarray, pd.DataFrame]:
    """Scored records and combiner training records (smoke: the first 16 of each)."""
    scored = scored_ecg_ids()
    train = combiner_records()
    if smoke:
        scored, train = scored[:SMOKE_RECORDS], train.iloc[:SMOKE_RECORDS].reset_index(drop=True)
    if holdout_ids() & set(scored.astype(int)):
        raise ValueError("An ECGAD holdout record would be mapped")
    return scored, train


def device_setup() -> str:
    """CUDA when visible, capped at 6 GB, deterministic cuDNN as in 065."""
    if not torch.cuda.is_available():
        return "cpu"
    total = torch.cuda.get_device_properties(0).total_memory
    torch.cuda.set_per_process_memory_fraction(min(1.0, 6e9 / total))
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    return "cuda"


def load_networks(heads: int, device: str) -> list[mil.LeadMIL]:
    """Load the three 065 ``mil_lse`` networks from their saved weights."""
    networks = []
    for name in MIL_WEIGHTS:
        model = mil.LeadMIL(heads, pooling="lse", cross_lead=False)
        model.load_state_dict(torch.load(MIL065 / f"weights_{name}.pt", map_location="cpu"))
        networks.append(model.to(device).eval())
    return networks


def saved_065() -> tuple[dict[int, np.ndarray], dict[int, tuple[str, int]], dict[str, np.ndarray]]:
    """Load saved 065 bag logits and finding-map rows for the recomputation check."""
    bags, lookup, evidence = {}, {}, {}
    for set_name in ("ecgad_dev", "evaluation"):
        with np.load(MIL065 / f"bags_{set_name}.npz") as saved:
            bags.update(zip(saved["ecg_id"].astype(int).tolist(), saved["mil_lse"], strict=True))
        with np.load(MIL065 / f"finding_maps_{set_name}.npz") as saved:
            evidence[set_name] = saved["evidence"]
            for row, ecg_id in enumerate(saved["ecg_id"].astype(int).tolist()):
                lookup.setdefault(ecg_id, (set_name, row))
    return bags, lookup, evidence


def mil_phase(smoke: bool) -> None:
    """Recompute 065 bag logits and evidence summaries for every scored and combiner record."""
    out, work = folders(smoke)
    began = time.time()
    scored, train = record_sets(smoke)
    ecg_ids = np.unique(np.concatenate([scored, train.ecg_id.to_numpy()])).astype(np.int64)
    heads = heads065()
    abnormal = np.array([k for k, h in enumerate(heads) if h != mil.REFERENCE_HEAD])
    rhythm_index = [heads.index(h) for h in cb.RHYTHM_HEADS]
    raw, index = open_cache()
    row_of = dict(zip(index.ecg_id.astype(int), range(len(index)), strict=True))
    positions = np.array([row_of[int(e)] for e in ecg_ids])
    saved_bags, lookup, saved_evidence = saved_065()
    n, steps = len(ecg_ids), mil.SAMPLES // mil.STRIDE + 1
    bags = np.zeros((n, len(heads)), np.float32)
    means = np.zeros((n, len(heads), 12), np.float32)
    rhythm = np.zeros((n, len(rhythm_index), 12, steps), np.float16)
    combined = np.zeros((n, 12, steps), np.float32)
    bag_diff, evidence_diff, compared_bags, compared_evidence = 0.0, 0.0, 0, 0
    device = device_setup()
    with gpu_lock(device):
        networks = load_networks(len(heads), device)
        for start in range(0, n, CHUNK):
            chunk = np.arange(start, min(start + CHUNK, n))
            signals = mil.preprocess(raw[positions[chunk]])
            local = np.arange(len(chunk))
            outputs = [mil.predict(net, signals, local, device, keep_instances=True) for net in networks]
            bag = np.mean([o[0] for o in outputs], axis=0)
            evidence = mil.finding_evidence(np.mean([o[1] for o in outputs], axis=0), None)
            bags[chunk] = bag
            means[chunk] = cb.lead_means(evidence)
            rhythm[chunk] = evidence[:, rhythm_index].astype(np.float16)
            combined[chunk] = mil.combined_map(bag, evidence, abnormal, "sum")
            for j, ecg_id in enumerate(ecg_ids[chunk].tolist()):
                if ecg_id in saved_bags:
                    bag_diff = max(bag_diff, float(np.abs(bag[j] - saved_bags[ecg_id]).max()))
                    compared_bags += 1
                if ecg_id in lookup:
                    set_name, row = lookup[ecg_id]
                    reference = saved_evidence[set_name][row].astype(np.float32)
                    evidence_diff = max(evidence_diff, float(np.abs(evidence[j] - reference).max()))
                    compared_evidence += 1
            print(f"mil {chunk[-1] + 1}/{n}", flush=True)
    check = {
        "records": int(n),
        "device": device,
        "bag_compared": compared_bags,
        "bag_max_abs_difference": bag_diff,
        "evidence_compared": compared_evidence,
        "evidence_max_abs_difference": evidence_diff,
        "pass": bool(bag_diff <= BAG_TOLERANCE and evidence_diff <= EVIDENCE_TOLERANCE),
        "seconds": time.time() - began,
    }
    np.save(work / "mil_ids.npy", ecg_ids)
    np.save(work / "mil_bags.npy", bags)
    np.save(work / "mil_lead_means.npy", means)
    np.save(work / "mil_rhythm.npy", rhythm)
    np.save(work / "mil_combined.npy", combined)
    write_json_atomic(out / "mil_check.json", check)
    print(json.dumps(check), flush=True)
    if not check["pass"]:
        raise ValueError(f"065 recomputation differs from the saved outputs: {check}")


def _quality_init() -> None:
    threadpool_limits(limits=1)
    signals, index = open_cache()
    _STATE["signals"] = signals
    _STATE["row"] = dict(zip(index.ecg_id.astype(int), range(len(index)), strict=True))


def _quality_one(ecg_id: int) -> np.ndarray:
    return cb.quality_features(exact_signal(_STATE["signals"][_STATE["row"][int(ecg_id)]]))


def quality_rows(fit: pd.DataFrame, meta: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Training rows of the quality classifier.

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray]
        Record position in ``fit``, lead index and label of every row.
    """
    named = fit.noisy_leads.fillna("").str.split(";").apply(lambda leads: [lead for lead in leads if lead])
    no_note = meta.loc[fit.ecg_id, list(NOISE_COLUMNS)].isna().all(axis=1).to_numpy()
    records, leads, labels = [], [], []
    for position, (names, clean) in enumerate(zip(named, no_note, strict=True)):
        if 1 <= len(names) <= len(LEADS) - 1:
            for lead in names:
                records.append(position)
                leads.append(LEADS.index(lead))
                labels.append(1)
        elif clean and not names:
            for lead in range(len(LEADS)):
                records.append(position)
                leads.append(lead)
                labels.append(0)
    return np.array(records), np.array(leads), np.array(labels)


def quality_model() -> Any:
    """Build standard scaling plus unweighted L2 logistic regression."""
    return make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=5000))


def within_record_auroc(q: np.ndarray, named: list[list[str]]) -> float:
    """Mean within-record AUROC of q against the named leads, over records naming 1-11 leads."""
    values = [auroc(np.isin(LEADS, names), q[k]) for k, names in enumerate(named) if 1 <= len(names) <= 11]
    return float(np.mean(values))


def quality_phase(smoke: bool) -> None:
    """Fit the lead-quality classifier with patient-grouped out-of-fold predictions and score records."""
    out, work = folders(smoke)
    began = time.time()
    partition = load_partition()
    fit = partition[partition.partition == "fit"].reset_index(drop=True)
    if forbidden_ids() & set(fit.ecg_id.astype(int)):
        raise ValueError("A fit record is an ECGAD or evaluation record")
    scored, train = record_sets(smoke)
    if smoke:
        keep = (np.arange(len(fit)) < 600) | fit.ecg_id.isin(train.ecg_id).to_numpy()
        fit = fit[keep].reset_index(drop=True)
    meta = load_metadata()
    targets = np.unique(np.concatenate([fit.ecg_id.to_numpy(), scored])).astype(int)
    with Pool(WORKERS, initializer=_quality_init) as pool:
        features = np.stack(pool.map(_quality_one, targets.tolist(), chunksize=32))
    feature_of = dict(zip(targets.tolist(), range(len(targets)), strict=True))
    fit_features = features[[feature_of[int(e)] for e in fit.ecg_id]]
    records, leads, labels = quality_rows(fit, meta)
    x = fit_features[records, leads]
    folds = cb.patient_folds(fit.patient_id.to_numpy())
    oof = np.zeros((len(fit), len(LEADS)))
    row_oof = np.zeros(len(labels))
    for fold in range(cb.QUALITY_FOLDS):
        train_rows = folds[records] != fold
        model = quality_model().fit(x[train_rows], labels[train_rows])
        held = folds == fold
        oof[held] = model.predict_proba(fit_features[held].reshape(-1, x.shape[1]))[:, 1].reshape(-1, 12)
        row_oof[~train_rows] = model.predict_proba(x[~train_rows])[:, 1]
    final = quality_model().fit(x, labels)
    q_scored = final.predict_proba(features[[feature_of[int(e)] for e in scored]].reshape(-1, x.shape[1]))
    q_scored = q_scored[:, 1].reshape(-1, 12)
    named = [[lead for lead in text.split(";") if lead] for text in fit.noisy_leads.fillna("")]
    _, rhythm, diagnostic, form, _ = _statement_sets()
    noise_fit, _ = _noise_set(fit, meta, rhythm, diagnostic, form)
    noise_positions = fit.reset_index().set_index("ecg_id").loc[noise_fit.ecg_id, "index"].to_numpy()
    scaler, logistic = final.named_steps["standardscaler"], final.named_steps["logisticregression"]
    metrics = {
        "rows": int(len(labels)),
        "positive_rows": int(labels.sum()),
        "positive_records": int(len(np.unique(records[labels == 1]))),
        "negative_records": int(len(np.unique(records[labels == 0]))),
        "fit_records": int(len(fit)),
        "oof_pooled_lead_auroc": auroc(labels.astype(bool), row_oof),
        "oof_within_record_auroc_named_1_11": within_record_auroc(oof, named),
        "records_named_1_11": int(sum(1 <= len(n) <= 11 for n in named)),
        "oof_within_record_auroc_fit_noise_set": within_record_auroc(
            oof[noise_positions], [named[k] for k in noise_positions]
        ),
        "fit_noise_set_records": int(len(noise_positions)),
        "features": list(cb.QUALITY_FEATURES),
        "coefficients_standardized": dict(zip(cb.QUALITY_FEATURES, logistic.coef_[0].tolist(), strict=True)),
        "intercept": float(logistic.intercept_[0]),
        "feature_means": dict(zip(cb.QUALITY_FEATURES, scaler.mean_.tolist(), strict=True)),
        "feature_scales": dict(zip(cb.QUALITY_FEATURES, scaler.scale_.tolist(), strict=True)),
        "q_scored_mean": float(q_scored.mean()),
        "q_scored_share_above_half": float((q_scored > 0.5).mean()),
        "seconds": time.time() - began,
    }
    (out / "quality").mkdir(exist_ok=True)
    np.savez(
        out / "quality/q.npz",
        scored_ids=scored,
        q_scored=q_scored,
        fit_ids=fit.ecg_id.to_numpy(),
        q_fit_oof=oof,
    )
    with (out / "quality/model.pkl").open("wb") as handle:
        pickle.dump(final, handle)
    write_json_atomic(out / "quality/metrics.json", metrics)
    print(json.dumps(metrics, indent=1), flush=True)


def _maps_init(work: str, out: str, check_ids: list[int]) -> None:
    threadpool_limits(limits=1)
    _STATE["check_ids"] = set(check_ids)
    work_path, out_path = Path(work), Path(out)
    signals, index = open_cache()
    _STATE["signals"] = signals
    _STATE["row"] = dict(zip(index.ecg_id.astype(int), range(len(index)), strict=True))
    _STATE["index"] = index.set_index("ecg_id")
    ids = np.load(work_path / "mil_ids.npy")
    _STATE["mil_row"] = dict(zip(ids.astype(int).tolist(), range(len(ids)), strict=True))
    for name in ("bags", "lead_means", "rhythm", "combined"):
        _STATE[name] = np.load(work_path / f"mil_{name}.npy", mmap_mode="r")
    with np.load(out_path / "quality/q.npz") as q:
        _STATE["q"] = dict(zip(q["scored_ids"].astype(int).tolist(), q["q_scored"], strict=True))
        _STATE["q_fit"] = dict(zip(q["fit_ids"].astype(int).tolist(), q["q_fit_oof"], strict=True))
    truth = np.load(work_path / "true_labels.npz")
    _STATE["truth"] = dict(zip(truth["ecg_id"].astype(int).tolist(), truth["y"], strict=True))
    norm_ids = np.load(NORMATIVE062 / "ecg_ids.npy")
    _STATE["norm_row"] = dict(zip(norm_ids.astype(int).tolist(), range(len(norm_ids)), strict=True))
    _STATE["norm_maps"] = np.load(NORMATIVE062 / "maps_normative_v1.npy", mmap_mode="r")
    with (NORMATIVE062 / "normative.pkl").open("rb") as handle:
        _STATE["normative"] = pickle.load(handle)  # noqa: S301
    with (LOCBENCH061 / "ub_references.pkl").open("rb") as handle:
        _STATE["references"] = pickle.load(handle)  # noqa: S301
    _STATE["out_maps"] = {
        name: np.lib.format.open_memmap(work_path / f"{name}.npy", mode="r+") for name in PAINTED
    }
    _STATE["heads"] = heads065()


def normative_for(ecg_id: int, signal: np.ndarray) -> np.ndarray:
    """062 ``normative_v1`` map: the saved map when present, else recomputed with the saved model."""
    if ecg_id in _STATE["norm_row"]:
        return np.asarray(_STATE["norm_maps"][_STATE["norm_row"][ecg_id]], np.float64)
    meta = _STATE["index"].loc[ecg_id]
    maps, _ = nm.record_maps(signal, float(meta.age), float(meta.sex), _STATE["normative"])
    return maps[nm.PRIMARY].astype(np.float16).astype(np.float64)


def _record(task: tuple[int, int, bool]) -> dict[str, Any]:
    """Build the painted maps (scored records) and unit features (all records) of one record."""
    position, ecg_id, is_scored = task
    stored = _STATE["signals"][_STATE["row"][ecg_id]]
    raw = np.asarray(stored, np.float64)
    exact = exact_signal(stored)
    amplitude = amplitude_map(exact)
    intervals = cb.delineated_components(raw)
    masks = cb.component_masks(intervals)
    heads = _STATE["heads"]
    abnormal = [k for k, h in enumerate(heads) if h != mil.REFERENCE_HEAD]
    names = [heads[k] for k in abnormal]
    m = _STATE["mil_row"][ecg_id]
    prob = 1.0 / (1.0 + np.exp(-np.asarray(_STATE["bags"][m], np.float64)[abnormal]))
    means = np.asarray(_STATE["lead_means"][m])[abnormal]
    rhythm = {h: np.asarray(_STATE["rhythm"][m, j], np.float32) for j, h in enumerate(cb.RHYTHM_HEADS)}
    q = _STATE["q"][ecg_id] if is_scored else _STATE["q_fit"][ecg_id]
    factor = cb.factor_sum(names, prob, means, rhythm, masks)
    textbook = cb.textbook_sum(names, prob, masks)
    if is_scored:
        truth = np.asarray(_STATE["truth"][ecg_id], np.float64)[abnormal]
        built = {
            "factor": cb.finish(factor, amplitude),
            "textbook": cb.finish(textbook, amplitude),
            "textbook_true": cb.finish(cb.textbook_sum(names, truth, masks), amplitude),
            "factor_q": cb.finish(factor * cb.quality_weights(q)[:, None], amplitude),
        }
        for name, values in built.items():
            if not np.isfinite(values).all():
                raise ValueError(f"Nonfinite {name} map for {ecg_id}")
            _STATE["out_maps"][name][position] = values
    residual = residual_map(exact, _STATE["references"])
    residual = np.zeros_like(exact) if residual is None else residual
    fields = {
        "factor": factor / max(factor[:, 100:4900].max(), 1e-30),
        "textbook": textbook / max(textbook[:, 100:4900].max(), 1e-30),
        "mil": mil.upsample(np.asarray(_STATE["combined"][m], np.float32)).astype(np.float64),
        "normative": normative_for(ecg_id, raw),
        "residual": residual,
        "amplitude": amplitude,
    }
    units = cb.unit_table(intervals)
    unit_raw = (
        np.column_stack([cb.unit_means(units, fields[name]) for name in cb.UNIT_MAPS])
        if len(units)
        else (np.zeros((0, len(cb.UNIT_MAPS))))
    )
    return {
        "position": position,
        "ecg_id": ecg_id,
        "scored": is_scored,
        "beats": int(len(intervals)),
        "units": units,
        "unit_raw": unit_raw,
        "q": np.asarray(q, np.float64),
        "residual_constant": bool(not np.any(residual)),
        "residual_ecgad_check": residual.astype(np.float32) if ecg_id in _STATE["check_ids"] else None,
    }


def _paint_init(work: str) -> None:
    threadpool_limits(limits=1)
    signals, index = open_cache()
    _STATE["signals"] = signals
    _STATE["row"] = dict(zip(index.ecg_id.astype(int), range(len(index)), strict=True))
    _STATE["units_map"] = np.lib.format.open_memmap(Path(work) / "units_learned.npy", mode="r+")


def _paint(task: tuple[int, int, np.ndarray, np.ndarray]) -> int:
    position, ecg_id, units, scores = task
    amplitude = amplitude_map(exact_signal(_STATE["signals"][_STATE["row"][ecg_id]]))
    values = cb.units_map(scores, units, amplitude)
    if not np.isfinite(values).all():
        raise ValueError(f"Nonfinite units_learned map for {ecg_id}")
    _STATE["units_map"][position] = values
    _STATE["units_map"].flush()
    return position


def true_labels(ecg_ids: np.ndarray, work: Path) -> None:
    """Binary 065-rule PTB-XL statement labels of the scored records for ``textbook_true``."""
    y, mask = mil.statement_labels(load_metadata(), load_statements(), ecg_ids, heads065())
    np.savez(work / "true_labels.npz", ecg_id=ecg_ids, y=y * mask)


def check_residual(results: list[dict[str, Any]]) -> dict[str, Any]:
    """Recomputed ``residual056`` on ECGAD development records against 061's scored float32 maps."""
    folder = LOCBENCH061 / "tables/residual056"
    order = json.loads((folder / "meta.json").read_text())["ecgad_ecg_ids"]
    saved = np.load(folder / "ecgad_maps.npy", mmap_mode="r")
    position = {int(e): k for k, e in enumerate(order)}
    differences = [
        float(np.abs(r["residual_ecgad_check"] - saved[position[r["ecg_id"]]]).max())
        for r in results
        if r["residual_ecgad_check"] is not None
    ]
    return {"records": len(differences), "max_abs_difference": max(differences) if differences else None}


def maps_phase(smoke: bool) -> None:
    """Build the painted maps, fit the combiner on the training records and paint ``units_learned``."""
    out, work = folders(smoke)
    began = time.time()
    scored, train = record_sets(smoke)
    true_labels(scored, work)
    for name in (*PAINTED, "units_learned"):
        np.lib.format.open_memmap(
            work / f"{name}.npy", mode="w+", dtype=np.float32, shape=(len(scored), 12, 5000)
        )
    tasks = [(k, int(e), True) for k, e in enumerate(scored)] + [(-1, int(e), False) for e in train.ecg_id]
    split = load_split()
    ecgad = split.ecg_id[split.partition == "development"].astype(int).tolist()
    with Pool(WORKERS, initializer=_maps_init, initargs=(str(work), str(out), ecgad)) as pool:
        results = []
        for k, result in enumerate(pool.imap_unordered(_record, tasks, chunksize=4)):
            results.append(result)
            if (k + 1) % 500 == 0:
                print(f"maps {k + 1}/{len(tasks)} {time.time() - began:.0f}s", flush=True)
    residual_check = check_residual(results)
    by_id = {(r["ecg_id"], r["scored"]): r for r in results}
    statement = dict(zip(train.ecg_id.astype(int), train.statement, strict=True))
    designs, positives, dropped = [], [], []
    for ecg_id in train.ecg_id.astype(int):
        r = by_id[(ecg_id, False)]
        if len(r["units"]) == 0:
            dropped.append(ecg_id)
            continue
        y = cb.weak_labels(r["units"], statement[ecg_id], REGION_TABLE)
        if y.all() or not y.any():
            dropped.append(ecg_id)
            continue
        designs.append(cb.unit_design(r["unit_raw"], r["q"], r["units"]))
        positives.append(y)
    weights = cb.fit_combiner(designs, positives)
    paint_tasks = []
    for k, ecg_id in enumerate(scored.astype(int)):
        r = by_id[(ecg_id, True)]
        scores = (
            cb.unit_design(r["unit_raw"], r["q"], r["units"]) @ weights if len(r["units"]) else np.zeros(0)
        )
        paint_tasks.append((k, ecg_id, r["units"], scores))
    with Pool(WORKERS, initializer=_paint_init, initargs=(str(work),)) as pool:
        list(pool.imap_unordered(_paint, paint_tasks, chunksize=16))
    files = {}
    for name in MAPS:
        array = np.load(work / f"{name}.npy", mmap_mode="r")
        if not np.isfinite(array).all():
            raise ValueError(f"Nonfinite {name}")
        files[name] = {"file": str(work / f"{name}.npy"), "sha256": sha256_file(work / f"{name}.npy")}
    (work / "scored_ids.json").write_text(json.dumps([int(e) for e in scored]))
    scored_results = [by_id[(int(e), True)] for e in scored]
    train_loss, _ = cb.listwise_loss(weights, designs, positives)
    zero_loss, _ = cb.listwise_loss(np.zeros_like(weights), designs, positives)
    receipt = {
        "created": utc_now(),
        "git_head": git_head(ROOT),
        "scored_records": int(len(scored)),
        "combiner_candidates": int(len(train)),
        "combiner_records": int(len(designs)),
        "combiner_dropped": dropped,
        "combiner_statements": train.statement.value_counts().sort_index().to_dict(),
        "combiner_patients": int(train.patient_id.nunique()),
        "combiner_units": int(sum(len(y) for y in positives)),
        "combiner_positive_units": int(sum(int(y.sum()) for y in positives)),
        "combiner_weights": dict(zip(cb.UNIT_FEATURES, weights.tolist(), strict=True)),
        "combiner_train_loss": float(train_loss),
        "combiner_zero_loss": float(zero_loss),
        "records_without_beats": [int(r["ecg_id"]) for r in scored_results if r["beats"] == 0],
        "residual056_constant_records": int(sum(r["residual_constant"] for r in scored_results)),
        "residual056_check": residual_check,
        "median_beats": float(np.median([r["beats"] for r in scored_results])),
        "files": files,
        "seconds": time.time() - began,
    }
    write_json_atomic(out / "maps_receipt.json", receipt)
    print(json.dumps({k: v for k, v in receipt.items() if k != "files"}, indent=1), flush=True)


def reproduce_bar() -> dict[str, Any]:
    """Rescore 061's saved ``qrs_prior`` ECGAD maps and match its per-record and mean A2 within 1e-12."""
    bench = load_benchmark()
    folder = LOCBENCH061 / "tables" / BAR
    if (
        json.loads((folder / "meta.json").read_text())["ecgad_ecg_ids"]
        != bench.ecgad.ecg_id.astype(int).tolist()
    ):
        raise ValueError("061 ECGAD order differs from the benchmark")
    maps = np.load(folder / "ecgad_maps.npy", mmap_mode="r")
    values = np.array(
        [
            ecgad_record_metrics(np.asarray(m, np.float32), y)["A2"]
            for m, y in zip(maps, bench.ecgad_labels, strict=True)
        ]
    )
    saved = pd.read_csv(folder / "ecgad.csv")["A2"].to_numpy()
    if not np.array_equal(np.isnan(values), np.isnan(saved)):
        raise ValueError("Bar A2 records differ")
    largest = float(np.nanmax(np.abs(values - saved)))
    mean = float(np.nanmean(values))
    target = json.loads((LOCBENCH061 / "result.json").read_text())["bar"]["A2"]["value"]
    passed = largest <= TOLERANCE and abs(mean - target) <= TOLERANCE
    if not passed:
        raise ValueError(f"Bar not reproduced: {largest}, {mean} against {target}")
    return {
        "per_record_max_abs_difference": largest,
        "records": int((~np.isnan(values)).sum()),
        "A2": mean,
        "A2_saved": target,
        "pass": passed,
    }


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


def reproduce_065(out: Path, work: Path, scored: np.ndarray) -> dict[str, Any]:
    """Assemble 065's saved ``mil_lse`` maps for the scored records, score them and match 065's tables."""
    parts = [
        (np.load(MIL065 / f"maps_index_{s}.npy"), np.load(MIL065 / f"maps_mil_lse_{s}.npy", mmap_mode="r"))
        for s in ("ecgad_dev", "evaluation")
    ]
    lookup: dict[int, tuple[int, int]] = {}
    for set_index, (ids, _) in enumerate(parts):
        for position, ecg_id in enumerate(ids.astype(int).tolist()):
            lookup.setdefault(ecg_id, (set_index, position))
    path = work / "mil_lse065.npy"
    combined = np.lib.format.open_memmap(path, mode="w+", dtype=np.float32, shape=(len(scored), 12, 5000))
    for k, ecg_id in enumerate(scored.astype(int).tolist()):
        set_index, position = lookup[ecg_id]
        combined[k] = np.asarray(parts[set_index][1][position], np.float32)
    combined.flush()
    del combined
    target = out / "integrity/mil_lse065"
    summary = run_score(path, work / "scored_ids.json", "mil_lse065", target, [])
    mine = pd.read_csv(target / "tables/ecgad.csv")["A2"].to_numpy()
    saved = pd.read_csv(MIL065 / "locbench/mil_lse/tables/ecgad.csv")["A2"].to_numpy()
    largest = float(np.nanmax(np.abs(mine - saved)))
    saved_a2 = json.loads((MIL065 / "locbench/mil_lse/summary.json").read_text())["metrics"]["A2"]["value"]
    a2 = summary["metrics"]["A2"]["value"]
    passed = bool(
        np.array_equal(np.isnan(mine), np.isnan(saved))
        and largest <= TOLERANCE
        and abs(a2 - saved_a2) <= TOLERANCE
    )
    path.unlink()
    if not passed:
        raise ValueError(f"065 mil_lse not reproduced: {largest}, {a2} against {saved_a2}")
    return {"per_record_max_abs_difference": largest, "A2": a2, "A2_saved": saved_a2, "pass": passed}


def copy_references(out: Path) -> dict[str, Path]:
    """Copy the 061 baseline tables and 062 ``qrs_lexicographic`` tables into this run's folder."""
    sources = {name: LOCBENCH061 / "tables" / name for name in BASELINES}
    sources[LEXICOGRAPHIC] = NORMATIVE062 / "locbench" / LEXICOGRAPHIC / "tables"
    copies = {}
    for name, source in sources.items():
        target = out / "ref" / name
        if not target.exists():
            shutil.copytree(source, target, ignore=shutil.ignore_patterns("strata.*", "noise_std.*"))
        copies[name] = target
    return copies


def paired(summary: dict[str, Any], other: str, metric: str = "A2") -> dict[str, Any] | None:
    """Paired difference of one metric against a compared method, if present."""
    return summary["paired"].get(other, {}).get(metric)


def decision(interval: dict[str, Any]) -> str:
    """Primary decision words from a paired interval."""
    if interval["ci_low"] > 0:
        return "localizes"
    if interval["ci_high"] < 0:
        return "below"
    return "matches"


def score_phase() -> None:
    """Integrity checks, LocBench scoring of every map, strata and standardized noise, result.json."""
    out, work = folders(False)
    began = time.time()
    scored = np.array(json.loads((work / "scored_ids.json").read_text()), np.int64)
    if not np.array_equal(scored, scored_ecg_ids()):
        raise ValueError("Scored records changed")
    integrity = {"bar": reproduce_bar(), "mil_lse065": reproduce_065(out, work, scored)}
    integrity["mil_recomputation"] = json.loads((out / "mil_check.json").read_text())
    receipt = json.loads((out / "maps_receipt.json").read_text())
    integrity["residual056_recomputation"] = receipt["residual056_check"]
    refs = copy_references(out)
    base_compare = [refs[name] for name in (*BASELINES, LEXICOGRAPHIC)]
    extra = {
        "textbook_true": [],
        "textbook": [],
        "factor": ["textbook"],
        "factor_q": ["textbook", "factor"],
        "units_learned": ["textbook", "factor"],
    }
    summaries = {}
    for name in ("textbook_true", "textbook", "factor", "factor_q", "units_learned"):
        compare = base_compare + [out / "locbench" / other / "tables" for other in extra[name]]
        summaries[name] = run_score(
            work / f"{name}.npy", work / "scored_ids.json", name, out / "locbench" / name, compare
        )
        print(f"{name}: A2 {summaries[name]['metrics']['A2']['value']:.4f}", flush=True)
    tables = [out / "locbench" / name / "tables" for name in MAPS]
    strata_command = [
        str(ROOT / ".venv/bin/python"),
        "-m",
        "scripts.experiments.run_locbench_strata",
        str(refs[BAR]),
        *map(str, tables),
        str(refs["residual056"]),
        str(refs[LEXICOGRAPHIC]),
    ]
    subprocess.run(strata_command, check=True, cwd=ROOT)
    noise_command = [
        str(ROOT / ".venv/bin/python"),
        "-m",
        "scripts.experiments.run_locbench_noise",
        *map(str, tables),
        str(refs[BAR]),
        str(refs["residual056"]),
        str(refs[LEXICOGRAPHIC]),
    ]
    subprocess.run(noise_command, check=True, cwd=ROOT)
    readings = {}
    for name in MAPS:
        folder = out / "locbench" / name / "tables"
        readings[name] = {
            "metrics": summaries[name]["metrics"],
            "paired": summaries[name]["paired"],
            "strata": json.loads((folder / "strata.json").read_text()),
            "noise_std": json.loads((folder / "noise_std.json").read_text()),
            "maps_sha256": summaries[name]["inputs"]["maps_sha256"],
        }
    reference_readings = {
        name: {
            "strata": json.loads((refs[name] / "strata.json").read_text())["summary"],
            "noise_std": json.loads((refs[name] / "noise_std.json").read_text()),
        }
        for name in (BAR, "residual056", LEXICOGRAPHIC)
    }
    primary = paired(summaries["factor"], BAR)
    c1 = {name: readings[name]["noise_std"]["value"] for name in MAPS}
    q_vs_factor = paired(summaries["factor_q"], "factor")
    adopt_q = bool(c1["factor_q"] <= c1["factor"] - 0.05 and q_vs_factor["ci_low"] > -0.01)
    secondary = max(("factor_q", "units_learned"), key=lambda n: summaries[n]["metrics"]["A2"]["value"])
    decisions = {
        "primary": {"map": "factor", "against": BAR, "A2_difference": primary, "decision": decision(primary)},
        "factor_vs_textbook": paired(summaries["factor"], "textbook"),
        "factor_vs_qrs_lexicographic": paired(summaries["factor"], LEXICOGRAPHIC),
        "factor_q_vs_factor": q_vs_factor,
        "units_learned_vs_qrs_prior": paired(summaries["units_learned"], BAR),
        "noise": {"C1_std": c1, "C1_std_drop": c1["factor"] - c1["factor_q"], "adopt_suppression": adopt_q},
        "candidates_067": [BAR, LEXICOGRAPHIC, "factor", secondary],
    }
    result = {
        "experiment": "066",
        "protocol": "docs/experiment-066-combination.md",
        "protocol_sha256": sha256_file(ROOT / "docs/experiment-066-combination.md"),
        "created": utc_now(),
        "git_head": git_head(ROOT),
        "inputs": {
            "partition": to_stored(PARTITION_PATH),
            "partition_sha256": sha256_file(PARTITION_PATH),
            "cache_dir": str(CACHE_DIR),
            "mil065_result_sha256": sha256_file(MIL065 / "result.json"),
            "mil065_weights_sha256": {n: sha256_file(MIL065 / f"weights_{n}.pt") for n in MIL_WEIGHTS},
            "normative062_maps_sha256": sha256_file(NORMATIVE062 / "maps_normative_v1.npy"),
            "normative062_model_sha256": sha256_file(NORMATIVE062 / "normative.pkl"),
            "locbench061_result_sha256": sha256_file(LOCBENCH061 / "result.json"),
            "ub_references_sha256": sha256_file(LOCBENCH061 / "ub_references.pkl"),
            "sources_sha256": {path: sha256_file(ROOT / path) for path in SOURCES},
        },
        "integrity": integrity,
        "quality": json.loads((out / "quality/metrics.json").read_text()),
        "maps": receipt,
        "readings": readings,
        "reference_readings": reference_readings,
        "decisions": decisions,
        "score_seconds": time.time() - began,
    }
    write_json_atomic(out / "result.json", result)
    print(json.dumps(decisions, indent=1), flush=True)


def separation_phase() -> None:
    """Record the data-separation checks of the fitted quantities in ``result.json``."""
    out, _ = folders(False)
    forbidden = forbidden_ids()
    partition = load_partition()
    fit = partition[partition.partition == "fit"].reset_index(drop=True)
    heldout = set(fit.ecg_id[mil.heldout_mask(fit.patient_id.to_numpy())].astype(int))
    with np.load(out / "quality/q.npz") as q:
        quality_ids = set(q["fit_ids"].astype(int).tolist())
        scored = set(q["scored_ids"].astype(int).tolist())
    combiner = set(combiner_records().ecg_id.astype(int))
    clean = set(fit.ecg_id[fit.clean_normal].astype(int))
    checks = {
        "quality_records": len(quality_ids),
        "quality_forbidden": len(quality_ids & forbidden),
        "combiner_records": len(combiner),
        "combiner_forbidden": len(combiner & forbidden),
        "combiner_outside_065_heldout": len(combiner - heldout),
        "combiner_fit_clean_normals": len(combiner & clean),
        "mapped_ecgad_holdout": len(scored & holdout_ids()),
    }
    checks["pass"] = all(v == 0 for k, v in checks.items() if not k.endswith("_records"))
    result = json.loads((out / "result.json").read_text())
    result["integrity"]["separation"] = checks
    write_json_atomic(out / "result.json", result)
    print(json.dumps(checks), flush=True)
    if not checks["pass"]:
        raise ValueError(f"Data separation failed: {checks}")


def main() -> None:
    """Command-line entry point."""
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=["mil", "quality", "maps", "score", "separation"])
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(8)
    if args.phase == "mil":
        mil_phase(args.smoke)
    elif args.phase == "quality":
        quality_phase(args.smoke)
    elif args.phase == "maps":
        maps_phase(args.smoke)
    elif args.smoke:
        raise ValueError("The smoke run is not scored")
    elif args.phase == "score":
        score_phase()
    else:
        separation_phase()


if __name__ == "__main__":
    main()
