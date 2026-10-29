"""
NBR-04 Amendment 1: the D6 arm, C6 on PTB-XL fold 9, the three-arm decision and the 2 x 2 factorial.

D6 and C6 use one whole-record template per record, exactly as NBR-02 (``nbr02_templates.arm_differences``
and ``nbr02_pipeline.arm_signals``), with the L6D model (D6) or NBR-02's C6 model.
"""

from __future__ import annotations

import gc
import time
from collections.abc import Callable
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
from ..files import read_json, write_json_atomic, write_npz_atomic
from ..full_development import predict
from ..gpu import gpu_lock
from ..heads01_features import lock_path
from ..intervals import patient_groups, patient_resample
from ..pr74_fresh_baseline import read_signal
from ..pr74_screening_features import load_encoder, measurements
from . import nbr01_analysis as na
from . import nbr01_features as nf
from . import nbr02_analysis as na2
from . import nbr02_pipeline as pipeline02
from . import nbr02_study as study02
from . import nbr02_templates as templates02
from . import nbr03_ptbxl as ptb
from . import nbr03_study as study03
from . import nbr04_study as study
from . import stages

WORK = study.WORK / "amendment"
NEW_ARMS = {"sph": ("D6",), "ptb": ("D6", "C6")}
DECISION_ARMS = ("L6", "D6", "L6D")
LEVEL = 100.0 - 5.0 / 3.0
FACTORIAL_LEVEL = 95.0
FACTORIAL_NAMES = ("selectivity", "territory", "top1_hit", "abnormal/component_flip")
CHECK_RECORDS = 100
_STATE: dict[str, Any] = {}


def models(input_root: Path) -> dict[str, nm.PopulationModel]:
    """Return the C6 and D6 (L6D basis) models."""
    arms = study.arm_models(input_root)
    return {"C6": arms["L6"], "D6": arms["L6D"]}


def record_arms(
    input_root: Path,
    row: pd.Series,
    dataset: str,
    arms: tuple[str, ...],
    plans: Path,
    covariates: Callable[[pd.Series], tuple[float, float]],
    fitted: dict[str, nm.PopulationModel],
) -> dict[str, Any]:
    """
    Whole-record differences, signals, diagnostics and measurements of the requested arms for one record.

    Raises
    ------
    ValueError
        If 062 features differ from the NBR plan, or a PTB row is outside fold 9.
    """
    if dataset == "ptb" and row["strat_fold"] != ptb.FOLD:
        raise ValueError("Only fold 9 may be read")
    raw = read_signal(input_root, row)
    age, sex = covariates(row)
    plan = nf.load_plan(plans / f"{row['ecg_id']}.npz")
    built = templates02.arm_differences(raw, age, sex, {arm: fitted[arm] for arm in arms})
    if built is None or not np.array_equal(built["knots"], plan["knots"]):
        raise ValueError(f"062 features differ from the NBR plan for {row['ecg_id']}")
    result: dict[str, Any] = {}
    for arm in arms:
        signals, energy = pipeline02.arm_signals(raw, plan, built[arm]["difference"])
        result.update(
            {
                f"{arm}_difference": built[arm]["difference"],
                f"{arm}_energy": energy,
                f"{arm}_measurements": measurements(list(signals)).to_numpy(),
                f"{arm}_digest": np.array(nf.signal_digest(signals)),
                f"{arm}_clipped": np.array(built[arm]["clipped"]),
                f"{arm}_coordinates": built[arm]["coordinates"],
            }
        )
    return result


def _prepare_job(job: tuple[str, int]) -> tuple[str, str, float]:
    torch.set_num_threads(1)
    dataset, position = job
    spec = _STATE["datasets"][dataset]
    row = spec["rows"].iloc[position]
    path = _STATE["work"] / dataset / "plans" / f"{row['ecg_id']}.npz"
    if path.exists():
        return dataset, row["ecg_id"], 0.0
    started = time.perf_counter()
    with threadpool_limits(limits=1):
        prepared = record_arms(
            _STATE["input_root"],
            row,
            dataset,
            NEW_ARMS[dataset],
            spec["plans"],
            spec["covariates"],
            _STATE["models"],
        )
    write_npz_atomic(path, **prepared)
    return dataset, row["ecg_id"], time.perf_counter() - started


def prepare_jobs(input_root: Path, jobs: list[tuple[str, int]], workers: int, work: Path) -> list:
    """Prepare (dataset, position) jobs in forked single-thread workers into ``work``."""
    specs = study.datasets(input_root)
    for name in specs:
        (work / name / "plans").mkdir(parents=True, exist_ok=True)
    _STATE.update(datasets=specs, input_root=input_root, models=models(input_root), work=work)
    with get_context("fork").Pool(workers) as pool:
        return pool.map(_prepare_job, jobs, chunksize=1)


def encode_rows(input_root: Path, jobs: list[tuple[str, int]], device: int, work: Path) -> dict[str, Any]:
    """Encode the new arms' variants of each job on one GPU under its lock, checking every digest."""
    specs = study.datasets(input_root)
    done = 0
    with gpu_lock("cuda", blocking=True, path=lock_path(device)), torch.cuda.device(device):
        torch.set_num_threads(2)
        free, _ = torch.cuda.mem_get_info()
        if free < 4 * 1024**3:
            raise RuntimeError("Less than four GiB free on the selected GPU")
        torch.cuda.reset_peak_memory_stats()
        encoder = load_encoder(input_root)
        started = time.perf_counter()
        for dataset, position in jobs:
            spec = specs[dataset]
            row = spec["rows"].iloc[position]
            path = work / dataset / "pooled" / f"{row['ecg_id']}.npy"
            if path.exists():
                continue
            path.parent.mkdir(parents=True, exist_ok=True)
            raw = read_signal(input_root, row)
            plan = nf.load_plan(spec["plans"] / f"{row['ecg_id']}.npz")
            inputs = []
            with np.load(work / dataset / "plans" / f"{row['ecg_id']}.npz") as prepared:
                for arm in NEW_ARMS[dataset]:
                    signals, _ = pipeline02.arm_signals(raw, plan, prepared[f"{arm}_difference"])
                    if nf.signal_digest(signals) != str(prepared[f"{arm}_digest"]):
                        raise ValueError(f"Rebuilt {arm} variants differ for {row['ecg_id']}")
                    inputs += [xecg_input(signal) for signal in signals]
            pooled = nf.encode(encoder, np.stack(inputs))
            temporary = path.with_suffix(".tmp.npy")
            np.save(temporary, pooled)
            temporary.replace(path)
            done += 1
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - started
        peak = torch.cuda.max_memory_allocated()
        del encoder
        gc.collect()
        torch.cuda.empty_cache()
    return {
        "records": done,
        "wall_seconds": elapsed,
        "peak_gpu_bytes": int(peak),
        "initial_free_gpu_bytes": int(free),
        "device": device,
    }


def check(input_root: Path, workers: int) -> dict[str, Any]:
    """Rebuild NBR-02's SPH C6 counterparts with this code path for 100 records (no score)."""
    receipt = study.binding(require_committed=False)
    spec = study.datasets(input_root)["sph"]
    fitted = models(input_root)
    worst = 0.0
    for _, row in spec["rows"].head(CHECK_RECORDS).iterrows():
        raw = read_signal(input_root, row)
        built = templates02.arm_differences(raw, *nf.covariates(row), {"C6": fitted["C6"]})
        with np.load(study02.WORK / "full/plans" / f"{row['ecg_id']}.npz") as previous:
            worst = max(
                worst,
                float(np.max(np.abs(built["C6"]["difference"].astype(float) - previous["C6_difference"]))),
            )
    details = {"records": CHECK_RECORDS, "c6_max_abs_difference_mv": worst}
    write_json_atomic(stages.stage_dir(input_root, study.OUTPUT, "amendment_check") / "check.json", details)
    if worst != 0.0:
        raise ValueError(f"C6 rebuild differs from NBR-02 by {worst} mV")
    return study.complete(input_root, "amendment_check", receipt, details)


def profile(input_root: Path, device: int, workers: int) -> dict[str, Any]:
    """Time preparation and encoding on 10 SPH and 10 fold-9 ECGs, without scoring."""
    receipt = study.binding(require_committed=False)
    work = WORK / "profile"
    jobs = [(name, position) for name in ("sph", "ptb") for position in range(10)]
    started = time.perf_counter()
    timings = prepare_jobs(input_root, jobs, workers, work)
    prepare_wall = time.perf_counter() - started
    encoded = encode_rows(input_root, jobs, device, work)
    details = {
        "records": len(jobs),
        "prepare_wall_seconds": prepare_wall,
        "prepare_cpu_seconds_per_record": float(np.mean([value for _, _, value in timings])),
        "encode": encoded,
        "encode_seconds_per_record": encoded["wall_seconds"] / max(encoded["records"], 1),
    }
    write_json_atomic(
        stages.stage_dir(input_root, study.OUTPUT, "amendment_profile") / "profile.json", details
    )
    return study.complete(input_root, "amendment_profile", receipt, details)


def prepare(input_root: Path, workers: int) -> dict[str, Any]:
    """Prepare D6 on both samples and C6 on fold 9."""
    receipt = study.binding()
    study.require(input_root, "amendment_check")
    started = time.perf_counter()
    timings = prepare_jobs(input_root, study.all_jobs(input_root), workers, WORK / "full")
    frame = pd.DataFrame(timings, columns=["dataset", "ecg_id", "seconds"])
    frame.to_csv(stages.stage_dir(input_root, study.OUTPUT, "amendment_prepare") / "prepare.csv", index=False)
    details = {
        "records": frame.groupby("dataset").size().to_dict(),
        "wall_seconds": time.perf_counter() - started,
    }
    return study.complete(input_root, "amendment_prepare", receipt, details)


def encode(input_root: Path, device: int, shard: int, shards: int) -> dict[str, Any]:
    """Encode one shard of both samples on one GPU."""
    receipt = study.binding()
    study.require(input_root, "amendment_prepare")
    result = encode_rows(input_root, study.all_jobs(input_root)[shard::shards], device, WORK / "full")
    stage = f"amendment_encode_{shard}"
    write_json_atomic(stages.stage_dir(input_root, study.OUTPUT, stage) / "encode.json", result)
    return study.complete(input_root, stage, receipt, {**result, "shard": shard, "shards": shards})


def score_dataset(input_root: Path, dataset: str, ecg_ids: list[str]) -> dict[str, np.ndarray]:
    """Score the new arms of one sample with both frozen models."""
    arms = NEW_ARMS[dataset]
    work = WORK / "full" / dataset
    bundle = joblib.load(input_root / nf.BUNDLE)
    count = len(pipeline02.counterpart_variants())
    measured, energy, clipped = [], [], []
    for ecg_id in ecg_ids:
        with np.load(work / "plans" / f"{ecg_id}.npz") as prepared:
            measured.append(np.stack([prepared[f"{arm}_measurements"] for arm in arms]))
            energy.append(np.stack([prepared[f"{arm}_energy"] for arm in arms]))
            clipped.append(np.array([int(prepared[f"{arm}_clipped"]) for arm in arms]))
    pooled = np.stack([np.load(work / "pooled" / f"{ecg_id}.npy") for ecg_id in ecg_ids])
    records = len(ecg_ids)
    frame = pd.DataFrame(
        np.stack(measured).reshape(records * len(arms) * count, -1), columns=list(RAW_FEATURES)
    )
    flat = pooled.reshape(records * len(arms) * count, -1)
    scores = np.empty((records, len(arms), count, len(nf.MODELS), len(nf.KINDS)))
    for position, name in enumerate(nf.MODELS):
        fitted = bundle["fitted"][name]
        hgb = fitted["hgb"].predict_proba(frame)[:, 1]
        xecg = predict(fitted["xecg"], flat)
        values = np.stack([hgb, xecg, 0.5 * hgb + 0.5 * xecg], axis=1)
        scores[:, :, :, position] = values.reshape(records, len(arms), count, len(nf.KINDS))
    return {"scores": scores, "energy": np.stack(energy), "clipped": np.stack(clipped)}


def score(input_root: Path) -> dict[str, Any]:
    """Score D6 on both samples and C6 on fold 9."""
    receipt = study.binding()
    study.require(input_root, "amendment_prepare")
    directory = stages.stage_dir(input_root, study.OUTPUT, "amendment_score")
    details = {}
    for name, spec in study.datasets(input_root).items():
        ids = spec["rows"]["ecg_id"].tolist()
        scored = score_dataset(input_root, name, ids)
        write_npz_atomic(
            directory / f"{name}_scores.npz",
            ecg_id=np.array(ids, dtype=str),
            arms=np.array(NEW_ARMS[name]),
            **scored,
        )
        details[name] = {"records": len(ids), "arms": list(NEW_ARMS[name])}
    return study.complete(input_root, "amendment_score", receipt, details)


def tables_for(input_root: Path, dataset: str) -> tuple[dict[str, dict[str, Any]], dict[str, np.ndarray]]:
    """NBR-01-layout tables of P0, C6, L6, D6 and L6D for one sample, and its P0 arrays."""
    spec = study.datasets(input_root)[dataset]
    base = dict(np.load(input_root / spec["scores"]))
    nbr04 = dict(np.load(input_root / study.OUTPUT / f"score/{dataset}_scores.npz"))
    amendment = dict(np.load(input_root / study.OUTPUT / f"amendment_score/{dataset}_scores.npz"))
    for other in (nbr04, amendment):
        if not np.array_equal(base["ecg_id"], other["ecg_id"]):
            raise ValueError(f"{dataset}: scores are misaligned")
    thresholds = nf.thresholds(input_root) if dataset == "sph" else ptb.thresholds(input_root)
    arrays = {
        "L6": (nbr04["scores"][:, 0], nbr04["energy"][:, 0]),
        "L6D": (nbr04["scores"][:, 1], nbr04["energy"][:, 1]),
    }
    for position, arm in enumerate(NEW_ARMS[dataset]):
        arrays[arm] = (amendment["scores"][:, position], amendment["energy"][:, position])
    if dataset == "sph":
        previous = dict(np.load(input_root / study02.OUTPUT / "score/scores.npz"))
        arrays["C6"] = (previous["scores"][:, 2], previous["energy"][:, 2])
    tables = {"P0": na.record_table(base["scores"], base["energy"], thresholds)}
    for arm in ("C6", "L6", "D6", "L6D"):
        tables[arm] = na.record_table(
            *na2.arm_arrays(base["scores"], base["energy"], *arrays[arm]), thresholds
        )
    return tables, base


def factorial(result: dict[str, float]) -> dict[str, float]:
    """Leave-region-out and direction-removal effects and their interaction for the factorial statistics."""
    found = {}
    for score_name in na2.SCORES:
        for name in FACTORIAL_NAMES:
            key = f"{score_name}/{name}"
            value = {arm: result.get(f"{arm}/{key}") for arm in ("C6", "L6", "D6", "L6D")}
            if any(item is None for item in value.values()):
                continue
            found[f"factorial/leave_region_out_without_removal/{key}"] = value["L6"] - value["C6"]
            found[f"factorial/leave_region_out_with_removal/{key}"] = value["L6D"] - value["D6"]
            found[f"factorial/removal_without_leave_out/{key}"] = value["D6"] - value["C6"]
            found[f"factorial/removal_with_leave_out/{key}"] = value["L6D"] - value["L6"]
            found[f"factorial/interaction/{key}"] = value["L6D"] - value["D6"] - value["L6"] + value["C6"]
    return found


def bootstrap(statistic: Callable[[np.ndarray], dict[str, float]], patients: np.ndarray) -> dict[str, dict]:
    """Point values with 98.33% and 95% whole-patient bootstrap intervals on the same 2,000 draws."""
    point = statistic(np.arange(len(patients)))
    rng = np.random.default_rng(na.SEED)
    resample = patient_groups(np.asarray(patients))
    values: dict[str, list[float]] = {key: [] for key in point}
    for _ in range(na.DRAWS):
        drawn = statistic(patient_resample(resample, rng))
        for key in point:
            value = drawn.get(key, float("nan"))
            if np.isfinite(value):
                values[key].append(value)
    result = {}
    for key, value in point.items():
        found = {"point": value, "valid_draws": len(values[key])}
        for level, suffix in ((LEVEL, ""), (FACTORIAL_LEVEL, "95")):
            tail = (100.0 - level) / 2
            low, high = np.percentile(values[key], [tail, 100 - tail]) if values[key] else (np.nan, np.nan)
            found[f"low{suffix}"], found[f"high{suffix}"] = float(low), float(high)
        result[key] = found
    return result


def decide(sph: dict[str, dict], ptb_results: dict[str, dict]) -> dict[str, Any]:
    """Apply Amendment 1's three-arm rule with 98.33% intervals."""
    result: dict[str, Any] = {}
    for arm in DECISION_ARMS:
        parts = {
            "A_selectivity": (sph[f"{arm}-P0/all_age_blend/selectivity"], "low", 0.0),
            "B_territory": (sph[f"{arm}-P0/all_age_blend/territory"], "low", study.MARGIN),
            "C_component_flip": (sph[f"{arm}-P0/all_age_blend/abnormal/component_flip"], "low", study.MARGIN),
            "D_ptb_selectivity": (ptb_results[f"{arm}-P0/all_age_blend/selectivity"], "point", 0.0),
            "D_ptb_territory": (ptb_results[f"{arm}-P0/all_age_blend/territory"], "low", study.MARGIN),
        }
        decided = {
            name: {
                "point": value["point"],
                "low": value["low"],
                "high": value["high"],
                "rule": f"{side} > {bound}",
                "passed": bool(value[side] > bound),
            }
            for name, (value, side, bound) in parts.items()
        }
        decided["D_confirmed"] = (
            decided["D_ptb_selectivity"]["passed"] and decided["D_ptb_territory"]["passed"]
        )
        decided["replaces_P0"] = decided["D_confirmed"] and all(
            decided[name]["passed"] for name in ("A_selectivity", "B_territory", "C_component_flip")
        )
        result[arm] = decided
    passing = [arm for arm in DECISION_ARMS if result[arm]["replaces_P0"]]
    result["passing"] = passing
    result["passing_differences"] = {
        f"{first}-{second}": sph[f"{first}-{second}/all_age_blend/selectivity"]
        for first in passing
        for second in passing
        if first != second and f"{first}-{second}/all_age_blend/selectivity" in sph
    }
    chosen = (
        max(passing, key=lambda arm: sph[f"{arm}/all_age_blend/selectivity"]["point"]) if passing else None
    )
    result["chosen"] = chosen
    result["exported_arm"] = chosen or "L6"
    return result


def analyze(input_root: Path) -> dict[str, Any]:
    """Compute every arm, decision and factorial statistic on both samples."""
    receipt = study.binding()
    study.require(input_root, "amendment_score")
    started = time.perf_counter()
    contrasts = tuple((arm, "P0") for arm in ("C6", *DECISION_ARMS))
    contrasts += tuple(
        (first, second) for first in DECISION_ARMS for second in DECISION_ARMS if first != second
    )
    results = {}
    for dataset in ("sph", "ptb"):
        tables, base = tables_for(input_root, dataset)
        groups, patients = base["group"], base["patient"]
        if dataset == "sph":

            def function(table: dict, rows: np.ndarray, groups: np.ndarray = groups) -> dict[str, float]:
                return na.statistics(table, groups, rows)
        else:
            primary = study03.primary_groups(
                groups, study03.require(input_root, "prepare")["primary_territories"]
            )

            def function(
                table: dict, rows: np.ndarray, groups: np.ndarray = groups, primary: np.ndarray = primary
            ) -> dict[str, float]:
                return study03.statistics(table, groups, primary, rows)

        def statistic(
            rows: np.ndarray, tables: dict = tables, function: Callable = function
        ) -> dict[str, float]:
            found = study.arm_statistics(tables, function, contrasts, rows)
            return {**found, **factorial(found)}

        results[dataset] = bootstrap(statistic, patients)
    reproduced = {
        "sph": results["sph"]["P0/all_age_blend/territory"]["point"],
        "ptb": results["ptb"]["P0/all_age_blend/territory"]["point"],
    }
    previous = study.load_results(input_root)
    same = {
        f"{dataset}/{key}": results[dataset][key]["point"] == previous[dataset][key]["point"]
        for dataset in ("sph", "ptb")
        for key in previous[dataset]
        if key.split("/")[0] in ("P0", "L6", "L6D") and key in results[dataset]
    }
    if reproduced != {"sph": study03.NBR01_PRIMARY, "ptb": study.NBR03_PRIMARY} or not all(same.values()):
        raise ValueError("Amendment analysis does not reproduce NBR-04's P0, L6 and L6D point statistics")
    decision = decide(results["sph"], results["ptb"])
    directory = stages.stage_dir(input_root, study.OUTPUT, "amendment_analyze")
    write_json_atomic(
        directory / "results.json",
        {
            "level": LEVEL,
            "factorial_level": FACTORIAL_LEVEL,
            "decision": decision,
            "p0_reproduced": reproduced,
            "nbr04_points_reproduced": len(same),
            **results,
        },
    )
    details = {
        "seconds": time.perf_counter() - started,
        "decision": decision,
        "nbr04_points_reproduced": len(same),
    }
    return study.complete(input_root, "amendment_analyze", receipt, details)


def load_results(input_root: Path) -> dict[str, Any]:
    """Read the amendment's analysis results after verifying the stage."""
    study.require(input_root, "amendment_analyze")
    return read_json(input_root / study.OUTPUT / "amendment_analyze/results.json")
