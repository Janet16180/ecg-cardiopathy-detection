"""
Experiment 070: replacement faithfulness test of the localization maps.

Phases::

    run_replacement070 select       # sets A (evaluation abnormal), N (evaluation normal), E (ECGAD dev)
    run_replacement070 train        # GPU: profile, then the independent detector (a) on fit records
    run_replacement070 prepare      # GPU for 065 bag logits, then 8 CPU processes: maps, counterparts, plans
    run_replacement070 score a      # GPU: detector logits of every pass (a or b)
    run_replacement070 analyze      # CPU: readings, intervals, Spearman, integrity, result.json
    run_replacement070 figures      # operator examples and the A2 scatter

``--smoke`` runs every phase on a few records into the smoke folders. See
``docs/experiment-070-replacement.md``.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from multiprocessing import get_context
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score
from threadpoolctl import threadpool_limits

from ecg_experiment import ROOT
from ecg_experiment import combination066 as cb
from ecg_experiment import holdout067 as ho
from ecg_experiment import mil065 as mil
from ecg_experiment import replacement070 as rp
from ecg_experiment.files import sha256_file, write_json_atomic
from ecg_experiment.gpu import gpu_lock
from ecg_experiment.locbench_baselines061 import exact_signal
from ecg_experiment.locbench_cache import open_cache
from ecg_experiment.locbench_data import load_partition
from ecg_experiment.provenance import git_head, utc_now
from ecg_experiment.waveforms import LEADS

OUT = ROOT / "outputs/experiment070_replacement_v1"
SMOKE_OUT = ROOT / "outputs/experiment070_replacement_smoke"
WORK = Path(os.environ.get("ECG070_WORK", "/tmp/ecg070-work"))
OUT061 = ROOT / "outputs/experiment061_locbench_v1"
OUT062 = ROOT / "outputs/experiment062_normative_v1"
OUT063 = ROOT / "outputs/experiment063_inpaint_v1"
OUT064 = ROOT / "outputs/experiment064_dino_knn_v1"
SETS = ("A", "N", "E")
SIZES = {"A": 1500, "N": 400}
SMOKE_SIZES = {"A": 6, "N": 3, "E": 3}
WORKERS = 8
CHUNK = 40
SAVED = ("normative_v1", "raw", "dino_dist", "mil_lse")
REBUILT = tuple(name for name in rp.MAPS if name not in SAVED)
DEVELOPMENT_TABLES = {
    **{name: OUT061 / "tables" / name for name in ("random", "amplitude", "qrs_prior", "U_B", "residual056")},
    "normative_v1": OUT062 / "locbench/normative_v1/tables",
    "qrs_lexicographic": OUT062 / "locbench/qrs_lexicographic/tables",
    "raw": OUT063 / "locbench/raw/tables",
    "dino_dist": OUT064 / "score/dino_dist/tables",
    "mil_lse": ho.MIL065 / "locbench/mil_lse/tables",
    **{name: ho.OUT066 / "locbench" / name / "tables" for name in ("factor", "textbook", "units_learned")},
    "units_v2": ho.OUT068 / "locbench/units_v2/tables",
}
PUBLISHED_A2 = {
    "random": 0.5001,
    "amplitude": 0.6978,
    "qrs_prior": 0.7419,
    "U_B": 0.6315,
    "residual056": 0.6796,
    "normative_v1": 0.6938,
    "qrs_lexicographic": 0.7530,
    "raw": 0.7091,
    "dino_dist": 0.7116,
    "mil_lse": 0.6624,
    "factor": 0.7247,
    "textbook": 0.7494,
    "units_learned": 0.7604,
    "units_v2": 0.7671,
}
MAP_TOLERANCE = 1e-5
BAG_TOLERANCE = 1e-3
DETECTOR_MINIMUM = 0.85
SOURCES = (
    "docs/experiment-070-replacement.md",
    "ecg_experiment/replacement070.py",
    "scripts/experiments/run_replacement070.py",
    "ecg_experiment/holdout067.py",
    "ecg_experiment/normative_map062.py",
    "ecg_experiment/combination066.py",
    "ecg_experiment/combiner068.py",
    "ecg_experiment/mil065.py",
    "ecg_experiment/locbench_baselines061.py",
    "ecg_experiment/locbench.py",
)

_STATE: dict[str, Any] = {}


def out_dir(smoke: bool) -> Path:
    """Output folder of a full or smoke run."""
    return SMOKE_OUT if smoke else OUT


def work_dir(name: str, smoke: bool) -> Path:
    """Work folder of one record set."""
    return WORK / ("smoke" if smoke else "full") / name


def read_records(name: str, smoke: bool) -> pd.DataFrame:
    """Read the records of one set as written by ``select``."""
    return pd.read_csv(out_dir(smoke) / f"records_{name}.csv")


def evaluation_saved_maps() -> dict[str, tuple[np.ndarray, dict[int, int]]]:
    """Memory-mapped saved evaluation maps and their row lookup, per saved map name."""
    manifest = json.loads((OUT063 / "maps_v1/evaluation_manifest.json").read_text())
    sources = {
        "normative_v1": (OUT062 / "maps_normative_v1.npy", np.load(OUT062 / "ecg_ids.npy")),
        "raw": (OUT063 / "maps_v1/evaluation_raw.npy", np.array(manifest["ecg_ids"])),
        "dino_dist": (OUT064 / "maps_dino_dist.npy", np.load(OUT064 / "ecg_ids.npy")),
        "mil_lse": (
            ho.MIL065 / "maps_mil_lse_evaluation.npy",
            np.load(ho.MIL065 / "maps_index_evaluation.npy"),
        ),
    }
    return {
        name: (np.load(path, mmap_mode="r"), {int(e): k for k, e in enumerate(ids)})
        for name, (path, ids) in sources.items()
    }


def development_saved_maps() -> dict[str, tuple[np.ndarray, dict[int, int]]]:
    """Memory-mapped scored development maps and their row lookup, for all 14 maps."""
    out = {}
    for name, folder in DEVELOPMENT_TABLES.items():
        order = json.loads((folder / "meta.json").read_text())["ecgad_ecg_ids"]
        out[name] = (
            np.load(folder / "ecgad_maps.npy", mmap_mode="r"),
            {int(e): k for k, e in enumerate(order)},
        )
    return out


def evaluation_references() -> dict[str, tuple[np.ndarray, dict[int, int]]]:
    """Open the saved evaluation maps of the rebuilt map names, where the files still exist."""
    out = {}
    for name in ("random", "amplitude", "qrs_prior", "U_B", "residual056"):
        path = OUT061 / "maps" / f"{name}.npy"
        ids = json.loads(Path(f"{path}.ids.json").read_text())
        out[name] = (np.load(path, mmap_mode="r"), {int(e): k for k, e in enumerate(ids)})
    out["qrs_lexicographic"] = (
        np.load(OUT062 / "maps_qrs_lexicographic.npy", mmap_mode="r"),
        {int(e): k for k, e in enumerate(np.load(OUT062 / "ecg_ids.npy"))},
    )
    for name, folder in (
        ("factor", "/tmp/ecg066-work/full"),
        ("textbook", "/tmp/ecg066-work/full"),
        ("units_learned", "/tmp/ecg066-work/full"),
        ("units_v2", "/tmp/ecg068-work/full"),
    ):
        path = Path(folder) / f"{name}.npy"
        if path.exists():
            ids = json.loads((Path(folder) / "scored_ids.json").read_text())
            out[name] = (np.load(path, mmap_mode="r"), {int(e): k for k, e in enumerate(ids)})
    return out


def select_phase(smoke: bool) -> None:
    """Draw sets A and N from the evaluation partition and list set E; no waveform is read."""
    partition = load_partition()
    evaluation = partition[partition["partition"] == "evaluation"].reset_index(drop=True)
    positive = rp.detection_positive(evaluation["ecg_id"].to_numpy())
    sizes = SMOKE_SIZES if smoke else SIZES
    sets = {
        "A": rp.one_per_patient(evaluation[positive], sizes["A"], rp.SEED),
        "N": rp.one_per_patient(evaluation[evaluation["clean_normal"]], sizes["N"], rp.NORMAL_SEED),
        "E": ho.split_records("development")[["ecg_id", "patient_id"]],
    }
    if smoke:
        sets["E"] = sets["E"].head(SMOKE_SIZES["E"])
    ecgad = set(ho.split_records("development").ecg_id) | set(ho.split_records("holdout").ecg_id)
    if ecgad & (set(sets["A"].ecg_id) | set(sets["N"].ecg_id)):
        raise ValueError("An ECGAD record entered set A or N")
    out = out_dir(smoke)
    out.mkdir(parents=True, exist_ok=True)
    for name, frame in sets.items():
        frame.to_csv(out / f"records_{name}.csv", index=False)
    summary = {name: {"records": len(f), "patients": int(f.patient_id.nunique())} for name, f in sets.items()}
    summary["eligible"] = {
        "A_records": int(positive.sum()),
        "A_patients": int(evaluation[positive].patient_id.nunique()),
        "N_records": int(evaluation["clean_normal"].sum()),
        "N_patients": int(evaluation[evaluation["clean_normal"]].patient_id.nunique()),
    }
    write_json_atomic(out / "select.json", summary)
    print(json.dumps(summary), flush=True)


def load_rows(ecg_ids: np.ndarray) -> tuple[np.ndarray, pd.DataFrame]:
    """Cache signals (float32 copies) and covariate rows of records."""
    signals, index = open_cache()
    row_of = dict(zip(index.ecg_id.astype(int), range(len(index)), strict=True))
    positions = np.array([row_of[int(e)] for e in ecg_ids])
    order = np.argsort(positions)
    stored = np.empty((len(positions), len(LEADS), rp.SAMPLES), np.float32)
    stored[order] = signals[positions[order]]
    return stored, index.iloc[positions].reset_index(drop=True)


def training_data() -> tuple[pd.DataFrame, np.ndarray]:
    """Fit records with a detection label and their labels."""
    partition = load_partition()
    fit = partition[partition["partition"] == "fit"].reset_index(drop=True)
    positive = rp.detection_positive(fit["ecg_id"].to_numpy())
    keep = positive | fit["clean_normal"].to_numpy()
    return fit[keep].reset_index(drop=True), positive[keep].astype(np.float32)


def device_setup() -> str:
    """065/067 device setup with TF32 disabled, so float32 logits are not rounded."""
    device = ho.device_setup()
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    return device


def profile_detector(signals: np.ndarray, y: np.ndarray, device: str, steps: int = 20) -> dict[str, Any]:
    """Time training steps and measure peak memory of detector (a)."""
    torch.manual_seed(0)
    detector = rp.Detector().to(device)
    optimizer = torch.optim.AdamW(detector.parameters(), lr=1e-3)
    rng = np.random.default_rng(0)
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    began = None
    for step in range(steps + 2):
        if step == 2:
            if device == "cuda":
                torch.cuda.synchronize()
            began = time.time()
        rows = rng.integers(0, len(signals), 64)
        logits = detector(rp.detector_crops(signals, rows, rng, device))
        loss = torch.nn.functional.binary_cross_entropy_with_logits(
            logits, torch.from_numpy(y[rows]).to(device)
        )
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    if device == "cuda":
        torch.cuda.synchronize()
    return {
        "parameters": int(sum(p.numel() for p in detector.parameters())),
        "seconds_per_step": (time.time() - began) / steps,
        "peak_gb": torch.cuda.max_memory_allocated() / 1e9 if device == "cuda" else 0.0,
    }


def train_phase(smoke: bool) -> None:
    """Profile and train detector (a) on fit records, then report its held-out and evaluation AUROC."""
    out = out_dir(smoke)
    records, y = training_data()
    held = mil.heldout_mask(records["patient_id"].to_numpy(), 0.15, rp.SEED)
    if smoke:
        keep = np.random.default_rng(0).permutation(len(records))[:600]
        records, y, held = records.iloc[keep].reset_index(drop=True), y[keep], held[keep]
    signals, _ = load_rows(records["ecg_id"].to_numpy())
    device = device_setup()
    with gpu_lock(device):
        profile = profile_detector(signals, y, device)
        print(json.dumps(profile), flush=True)
        write_json_atomic(out / "gpu_profile_a.json", profile)
        began = time.time()
        state, history = rp.train_detector(
            signals, y, np.flatnonzero(~held), np.flatnonzero(held), device, max_epochs=2 if smoke else 40
        )
        seconds = time.time() - began
    torch.save(state, out / "detector_a.pt")
    detector = rp.Detector().to(device)
    detector.load_state_dict(state)
    held_logits = rp.full_logits(detector, signals, np.flatnonzero(held), device)
    evaluation = evaluation_detection(detector, device, smoke)
    summary = {
        "records": len(records),
        "positives": int(y.sum()),
        "train_records": int((~held).sum()),
        "held_records": int(held.sum()),
        "held_patients": int(records.loc[held, "patient_id"].nunique()),
        "held_auroc": float(roc_auc_score(y[held], held_logits)),
        "best_epoch": int(np.argmax([h["held_auroc"] for h in history])),
        "epochs_run": len(history),
        "train_seconds": seconds,
        "history": history,
        "evaluation_detection": evaluation,
        "weights_sha256": sha256_file(out / "detector_a.pt"),
        "fit_ecgad_overlap": int(records.ecg_id.isin(ho.split_records("development").ecg_id).sum()),
    }
    write_json_atomic(out / "detector_a.json", summary)
    print(json.dumps({k: v for k, v in summary.items() if k != "history"}), flush=True)


def evaluation_detection(detector: torch.nn.Module, device: str, smoke: bool) -> dict[str, Any]:
    """Detector (a) AUROC on the evaluation detection set (description only)."""
    partition = load_partition()
    evaluation = partition[partition["partition"] == "evaluation"].reset_index(drop=True)
    positive = rp.detection_positive(evaluation["ecg_id"].to_numpy())
    keep = positive | evaluation["clean_normal"].to_numpy()
    ids, y = evaluation.loc[keep, "ecg_id"].to_numpy(), positive[keep]
    if smoke:
        ids, y = ids[:200], y[:200]
    signals, _ = load_rows(ids)
    with gpu_lock(device):
        logits = rp.full_logits(detector, signals, np.arange(len(ids)), device)
    return {"records": len(ids), "positives": int(y.sum()), "auroc": float(roc_auc_score(y, logits))}


def _quality_one(stored: np.ndarray) -> np.ndarray:
    return cb.quality_features(exact_signal(stored))


def _worker_init() -> None:
    threadpool_limits(limits=1)
    folder = _STATE["folder"]
    _STATE["out"] = {name: np.load(folder / f"{name}.npy", mmap_mode="r+") for name in _STATE["arrays"]}
    _STATE["saved"] = development_saved_maps() if _STATE["development"] else evaluation_saved_maps()
    _STATE["references"] = {} if _STATE["development"] else evaluation_references()


def _difference(new: np.ndarray, saved: np.ndarray) -> float:
    if saved.dtype == np.float16:
        new = new.astype(np.float16)
    return float(np.max(np.abs(new.astype(np.float64) - np.asarray(saved, np.float64))))


def _prepare_row(row: int) -> int:
    ecg_id = int(_STATE["ecg_ids"][row])
    stored = _STATE["signals"][row]
    built = ho.record_maps(
        stored,
        ecg_id,
        float(_STATE["age"][row]),
        float(_STATE["sex"][row]),
        {name: values[row] for name, values in _STATE["mil"].items()},
        _STATE["q"][row],
        np.zeros(len(_STATE["frozen"]["heads"]), np.float32),
        _STATE["frozen"],
    )
    saved = _STATE["saved"]
    references = saved if _STATE["development"] else _STATE["references"]
    for j, name in enumerate(REBUILT):
        if name in references and ecg_id in references[name][1]:
            reference = references[name][0][references[name][1][ecg_id]]
            _STATE["out"]["rebuild"][row, j] = _difference(built[name], reference)
    if _STATE["development"]:
        maps = np.stack([np.asarray(saved[n][0][saved[n][1][ecg_id]], np.float32) for n in rp.MAPS])
    else:
        maps = np.stack(
            [
                built[n] if n in REBUILT else np.asarray(saved[n][0][saved[n][1][ecg_id]], np.float32)
                for n in rp.MAPS
            ]
        )
    if not np.isfinite(maps).all():
        raise ValueError(f"Map of {ecg_id} is not finite")
    raw = np.asarray(stored, np.float64)
    difference = rp.counterpart_difference(
        raw, float(_STATE["age"][row]), float(_STATE["sex"][row]), _STATE["frozen"]["normative"]
    )
    if difference is None or not len(cb.delineated_components(raw)):
        return row
    plan = rp.record_plan(ecg_id, maps, rp.composition_labels(raw))
    out = _STATE["out"]
    out["difference"][row] = difference
    for name, values in plan.items():
        out[name][row] = values
    out["status"][row] = 1
    return row


ARRAYS = {
    "difference": ((len(LEADS), rp.SAMPLES), np.float32),
    "ranks": ((len(rp.MAPS), len(LEADS), rp.SCORED), np.uint16),
    "leads": ((len(rp.MAPS), rp.TOP_LEADS), np.int8),
    "inverse": ((rp.CONTROL_DRAWS, len(LEADS)), np.int8),
    "sources": ((rp.CONTROL_DRAWS, rp.SCORED), np.int16),
    "triplets": ((rp.LEAD_DRAWS, rp.TOP_LEADS), np.int8),
    "composition": ((len(rp.MAPS), len(rp.LABELS)), np.float32),
    "status": ((), np.int8),
    "rebuild": ((len(REBUILT),), np.float64),
}


def mil_inputs(
    name: str, records: pd.DataFrame, stored: np.ndarray, frozen: dict[str, Any]
) -> dict[str, Any]:
    """065 bag logits and evidence summaries on the GPU, checked against 065's saved bags."""
    device = device_setup()
    began = time.time()
    with gpu_lock(device):
        networks = ho.load_networks(len(frozen["heads"]), device)
        summaries = ho.mil_summaries(stored, networks, frozen["heads"], device)
        del networks
    saved_file = ho.MIL065 / ("bags_ecgad_dev.npz" if name == "E" else "bags_evaluation.npz")
    with np.load(saved_file) as saved:
        bag_of = dict(zip(saved["ecg_id"].astype(int).tolist(), saved["mil_lse"], strict=True))
    bags = np.stack([bag_of[int(e)] for e in records.ecg_id])
    return {
        "summaries": summaries,
        "seconds": time.time() - began,
        "bag_check": {
            "max_abs_difference": float(np.abs(summaries["bags"] - bags).max()),
            "tolerance": BAG_TOLERANCE,
        },
    }


def prepare_phase(smoke: bool) -> None:
    """Build maps, counterparts and plans for every set into the work folder."""
    frozen = ho.load_frozen()
    receipt = {"frozen_hashes": ho.frozen_hashes(), "sets": {}}
    for name in SETS:
        began = time.time()
        records = read_records(name, smoke)
        stored, covariates = load_rows(records.ecg_id.to_numpy())
        mil_out = mil_inputs(name, records, stored, frozen)
        with get_context("fork").Pool(WORKERS, initializer=threadpool_limits, initargs=(1,)) as pool:
            features = np.stack(pool.map(_quality_one, list(stored), chunksize=8))
        q = ho.lead_quality(features, frozen["quality"])
        folder = work_dir(name, smoke)
        folder.mkdir(parents=True, exist_ok=True)
        for array, (shape, dtype) in ARRAYS.items():
            values = np.lib.format.open_memmap(
                folder / f"{array}.npy", mode="w+", dtype=dtype, shape=(len(records), *shape)
            )
            values[:] = np.nan if array == "rebuild" else 0
            values.flush()
        _STATE.update(
            folder=folder,
            arrays=tuple(ARRAYS),
            development=name == "E",
            ecg_ids=records.ecg_id.to_numpy(),
            signals=stored,
            age=covariates.age.to_numpy(),
            sex=covariates.sex.to_numpy(),
            mil=mil_out["summaries"],
            q=q,
            frozen=frozen,
        )
        with get_context("fork").Pool(WORKERS, initializer=_worker_init) as pool:
            done = list(pool.imap_unordered(_prepare_row, range(len(records)), chunksize=4))
        if sorted(done) != list(range(len(records))):
            raise RuntimeError(f"Set {name}: not every record was prepared")
        abnormal = -mil_out["summaries"]["bags"][:, frozen["heads"].index(mil.REFERENCE_HEAD)]
        np.save(folder / "cpu_path_logits_b.npy", abnormal.astype(np.float32))
        receipt["sets"][name] = {
            "records": len(records),
            "mil_seconds": mil_out["seconds"],
            "bag_check": mil_out["bag_check"],
            "seconds": time.time() - began,
            "excluded": int((np.load(folder / "status.npy") == 0).sum()),
        }
        print(name, json.dumps(receipt["sets"][name]), flush=True)
    write_json_atomic(out_dir(smoke) / "prepare.json", receipt)


def load_detector(detector: str, smoke: bool, device: str) -> torch.nn.Module:
    """Detector (a) from its saved weights or the 065 ensemble as detector (b), in evaluation mode."""
    if detector == "a":
        model = rp.Detector()
        model.load_state_dict(torch.load(out_dir(smoke) / "detector_a.pt", map_location="cpu"))
    else:
        heads = ho.heads065()
        model = rp.MilDetector(
            ho.load_networks(len(heads), "cpu"), heads.index(mil.REFERENCE_HEAD), rp.filter_matrix()
        )
    return model.to(device).eval()


def plan_row(folder_arrays: dict[str, np.ndarray], row: int) -> dict[str, np.ndarray]:
    """One record's plan from the work arrays."""
    return {
        name: np.asarray(folder_arrays[name][row])
        for name in ("ranks", "leads", "inverse", "sources", "triplets")
    }


def score_set(
    name: str, detector_name: str, model: torch.nn.Module, device: str, smoke: bool
) -> dict[str, Any]:
    """Logits of every pass of every prepared record of one set, in chunks that each hold the GPU lock."""
    folder = work_dir(name, smoke)
    records = read_records(name, smoke)
    arrays = {
        n: np.load(folder / f"{n}.npy", mmap_mode="r")
        for n in ("ranks", "leads", "inverse", "sources", "triplets", "difference", "status")
    }
    status = np.asarray(arrays["status"])
    passes = rp.pass_layout(len(rp.MAPS))["passes"]
    path = folder / f"logits_{detector_name}.npy"
    done_path = folder / f"done_{detector_name}.npy"
    if path.exists() and done_path.exists():
        logits, done = np.load(path, mmap_mode="r+"), np.load(done_path)
    else:
        logits = np.lib.format.open_memmap(path, mode="w+", dtype=np.float32, shape=(len(records), passes))
        logits[:] = np.nan
        done = np.zeros(len(records), bool)
    stored, _ = load_rows(records.ecg_id.to_numpy())
    todo = np.flatnonzero((status == 1) & ~done)
    began, gpu_seconds = time.time(), 0.0
    for start in range(0, len(todo), CHUNK):
        rows = todo[start : start + CHUNK]
        with gpu_lock(device):
            locked = time.time()
            for row in rows:
                logits[row] = rp.record_logits(
                    model, stored[row], arrays["difference"][row], plan_row(arrays, row), device
                )
            gpu_seconds += time.time() - locked
        done[rows] = True
        logits.flush()
        np.save(done_path, done)
        print(
            f"{name} {detector_name}: {int(done.sum())}/{int((status == 1).sum())} records, "
            f"{time.time() - began:.0f} s",
            flush=True,
        )
    valid = np.flatnonzero(status == 1)
    with gpu_lock(device):
        originals = rp.full_logits(model, stored, valid, device, batch=32)
    null = np.abs(originals - np.asarray(logits)[valid, rp.pass_layout(len(rp.MAPS))["original"]])
    np.savez(
        out_dir(smoke) / f"logits_{detector_name}_{name}.npz",
        ecg_id=records.ecg_id.to_numpy(),
        status=status,
        logits=np.asarray(logits),
    )
    return {
        "records": int(len(valid)),
        "seconds": time.time() - began,
        "gpu_seconds": gpu_seconds,
        "null_check_max_abs_difference": float(null.max()) if len(null) else 0.0,
        "originals": originals.tolist(),
    }


def score_phase(detector_name: str, smoke: bool) -> None:
    """Score every set with one detector; profile three records first."""
    device = device_setup()
    model = load_detector(detector_name, smoke, device)
    receipt = {"detector": detector_name, "started": utc_now(), "sets": {}}
    for name in SETS:
        receipt["sets"][name] = score_set(name, detector_name, model, device, smoke)
        print(name, {k: v for k, v in receipt["sets"][name].items() if k != "originals"}, flush=True)
    receipt["finished"] = utc_now()
    write_json_atomic(out_dir(smoke) / f"score_{detector_name}.json", receipt)


def profile_phase(detector_name: str, smoke: bool) -> None:
    """Time ``record_logits`` on three prepared records of set A and report peak GPU memory."""
    device = device_setup()
    model = load_detector(detector_name, smoke, device)
    folder = work_dir("A", smoke)
    arrays = {
        n: np.load(folder / f"{n}.npy", mmap_mode="r")
        for n in ("ranks", "leads", "inverse", "sources", "triplets", "difference", "status")
    }
    records = read_records("A", smoke)
    rows = np.flatnonzero(np.asarray(arrays["status"]) == 1)[:3]
    stored, _ = load_rows(records.ecg_id.to_numpy()[rows])
    with gpu_lock(device):
        if device == "cuda":
            torch.cuda.reset_peak_memory_stats()
        times = []
        for k, row in enumerate(rows):
            began = time.time()
            rp.record_logits(model, stored[k], arrays["difference"][row], plan_row(arrays, row), device)
            if device == "cuda":
                torch.cuda.synchronize()
            times.append(time.time() - began)
    profile = {
        "detector": detector_name,
        "seconds_per_record": times,
        "passes": rp.pass_layout(len(rp.MAPS))["passes"],
        "peak_gb": torch.cuda.max_memory_allocated() / 1e9 if device == "cuda" else 0.0,
    }
    write_json_atomic(out_dir(smoke) / f"gpu_profile_score_{detector_name}.json", profile)
    print(json.dumps(profile), flush=True)


def analyze_phase(smoke: bool) -> None:
    """Raise until this phase is written (after the scoring code is tested)."""
    raise NotImplementedError(f"analyze {smoke} {git_head(ROOT)}")


def figures_phase(smoke: bool) -> None:
    """Raise until this phase is written (after the scoring code is tested)."""
    raise NotImplementedError(f"figures {smoke}")


def main() -> None:
    """Command-line entry point."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "phase", choices=("select", "train", "prepare", "profile", "score", "analyze", "figures")
    )
    parser.add_argument("detector", nargs="?", choices=("a", "b"), default="a")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    out_dir(args.smoke).mkdir(parents=True, exist_ok=True)
    if args.phase == "select":
        select_phase(args.smoke)
    elif args.phase == "train":
        train_phase(args.smoke)
    elif args.phase == "prepare":
        prepare_phase(args.smoke)
    elif args.phase == "profile":
        profile_phase(args.detector, args.smoke)
    elif args.phase == "score":
        score_phase(args.detector, args.smoke)
    elif args.phase == "analyze":
        analyze_phase(args.smoke)
    elif args.phase == "figures":
        figures_phase(args.smoke)


if __name__ == "__main__":
    main()
