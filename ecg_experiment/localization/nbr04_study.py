"""NBR-04 stages: fitsets, profile, prepare, encode, score, analyze on SPH and PTB-XL fold 9."""

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
from ..heads01_features import lock_path, superclass_table
from ..intervals import patient_groups, patient_resample
from ..pr74_fresh_baseline import read_signal
from ..pr74_screening_features import load_encoder, measurements
from . import nbr01_analysis as na
from . import nbr01_features as nf
from . import nbr01_study as study01
from . import nbr02_analysis as na2
from . import nbr02_pipeline as pipeline02
from . import nbr02_study as study02
from . import nbr02_templates as templates02
from . import nbr03_ptbxl as ptb
from . import nbr03_study as study03
from . import nbr04_templates as nt
from . import stages

OUTPUT = Path("outputs/nbr04_leave_region_out_20261028")
WORK = Path("/tmp/nbr04-work")
PROTOCOL = Path("docs/nbr04-leave-region-out-protocol-2026-10-28.md")
ARMS = ("L6", "L6D")
LEVEL = 97.5
MARGIN = -0.05
NBR03_PRIMARY = 0.09848614265592237
PROFILE_RECORDS = 10
SPH_CONTRASTS = (("L6", "P0"), ("L6D", "P0"), ("L6D", "L6"), ("L6", "C6"), ("L6D", "C6"))
PTB_CONTRASTS = (("L6", "P0"), ("L6D", "P0"), ("L6D", "L6"))
GROUPS = ("inferior", "anterior", "lateral", "lbbb", "rbbb", "stt", "normal")
_STATE: dict[str, Any] = {}


def binding(require_committed: bool = True) -> dict[str, Any]:
    """Bind NBR-04's sources and protocol."""
    return stages.binding("nbr04", PROTOCOL, require_committed)


def complete(
    input_root: Path, stage: str, receipt: dict[str, Any], details: dict[str, Any]
) -> dict[str, Any]:
    """Complete one NBR-04 stage."""
    return stages.complete(input_root, OUTPUT, stage, receipt, details)


def require(input_root: Path, stage: str) -> dict[str, Any]:
    """Verify one completed NBR-04 stage."""
    return stages.require(input_root, OUTPUT, stage)


def c_models(input_root: Path) -> dict[str, nm.PopulationModel]:
    """Load NBR-02's verified C0 and C6 models and require shared coefficients and scale."""
    models = study02.models_of(input_root)
    c0, c6 = models["C0"], models["C6"]
    if not (np.array_equal(c0.coefficients, c6.coefficients) and np.array_equal(c0.scale, c6.scale)):
        raise ValueError("C0 and C6 do not share coefficients and scale")
    return {"C0": c0, "C6": c6}


def fitsets(input_root: Path, workers: int) -> dict[str, Any]:
    """Compute the superclass disease directions on the all-age cohort and build the L6D model."""
    receipt = binding(require_committed=False)
    started = time.perf_counter()
    model = c_models(input_root)["C6"]
    rows = pd.read_csv(input_root / templates02.COHORT, dtype=str, keep_default_na=False)
    pool = rows.assign(target=rows["target"].astype(int))
    flags = superclass_table(input_root, pool)
    used, standard = nt.cohort_standardized(input_root, rows, model, workers)
    flags = flags.iloc[used].reset_index(drop=True)
    normal = flags["target"].to_numpy() == 0
    directions = nt.disease_directions(standard, flags, normal)
    fitted, diagnostics = nt.remove_directions(model, directions["orthonormal"], standard[normal])
    directory = stages.stage_dir(input_root, OUTPUT, "fitsets")
    nf.save_model(directory / "L6D.npz", fitted)
    write_npz_atomic(
        directory / "directions.npz", raw=directions["raw"], orthonormal=directions["orthonormal"]
    )
    details = {
        "cohort_rows": len(rows),
        "used": len(used),
        "excluded": len(rows) - len(used),
        "normals": directions["normals"],
        "superclass_counts": directions["counts"],
        "direction_norms": directions["norms"],
        "l6d": diagnostics,
        "seconds": time.perf_counter() - started,
    }
    write_json_atomic(directory / "fitsets.json", details)
    if directions["normals"] != require_c_normals(input_root):
        raise ValueError("Normal count differs from NBR-02's C fit set")
    return complete(input_root, "fitsets", receipt, details)


def require_c_normals(input_root: Path) -> int:
    """Return the number of C fit-set normals recorded by NBR-02."""
    return int(study02.require(input_root, "fitsets")["fit_sets"]["C"]["used"])


def arm_models(input_root: Path) -> dict[str, nm.PopulationModel]:
    """Return the L6 (C6) and L6D models."""
    require(input_root, "fitsets")
    return {"L6": c_models(input_root)["C6"], "L6D": nf.load_model(input_root / OUTPUT / "fitsets/L6D.npz")}


def datasets(input_root: Path) -> dict[str, dict[str, Any]]:
    """
    Return the two samples: NBR-01's 1,037 SPH ECGs and NBR-03's 862 PTB-XL fold-9 ECGs.

    Returns
    -------
    dict[str, dict[str, Any]]
        Per dataset: rows, NBR plan directory, covariate function and the P0 scores file.
    """
    return {
        "sph": {
            "rows": study02.sample_rows(input_root),
            "plans": study01.WORK / "full/plans",
            "covariates": nf.covariates,
            "scores": study01.OUTPUT / "score/scores.npz",
        },
        "ptb": {
            "rows": study03.prepared_rows(input_root),
            "plans": study03.WORK / "full/plans",
            "covariates": ptb.covariates,
            "scores": study03.OUTPUT / "score/scores.npz",
        },
    }


def prepare_record(
    input_root: Path,
    row: pd.Series,
    dataset: str,
    models: dict[str, nm.PopulationModel],
    plans: Path,
    covariates: Callable[[pd.Series], tuple[float, float]],
) -> dict[str, Any]:
    """
    Region coordinates, diagnostics, energies, measurements and digests of both arms for one record.

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
    items = pipeline02.counterpart_variants()
    result: dict[str, Any] = {}
    for arm, model in models.items():
        regions = nt.region_differences(raw, age, sex, model, items)
        if regions is None or not np.array_equal(regions["knots"], plan["knots"]):
            raise ValueError(f"062 features differ from the NBR plan for {row['ecg_id']}")
        signals, energy = nt.region_signals(raw, plan["components"], regions["differences"], items)
        result.update(
            {
                f"{arm}_coordinates": regions["coordinates"],
                f"{arm}_clipped": regions["clipped"],
                f"{arm}_movement": regions["movement"],
                f"{arm}_energy": energy,
                f"{arm}_measurements": measurements(list(signals)).to_numpy(),
                f"{arm}_digest": np.array(nf.signal_digest(signals)),
            }
        )
        if arm == "L6" and dataset == "sph":
            with np.load(study02.WORK / "full/plans" / f"{row['ecg_id']}.npz") as previous:
                result["l6_full_equals_c0"] = np.array(
                    bool(np.array_equal(regions["differences"][-1], previous["C0_difference"]))
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
        prepared = prepare_record(
            _STATE["input_root"], row, dataset, _STATE["models"], spec["plans"], spec["covariates"]
        )
    write_npz_atomic(path, **prepared)
    return dataset, row["ecg_id"], time.perf_counter() - started


def prepare_jobs(
    input_root: Path, jobs: list[tuple[str, int]], workers: int, work: Path
) -> list[tuple[str, str, float]]:
    """Prepare (dataset, position) jobs in forked single-thread workers into ``work``."""
    specs = datasets(input_root)
    for name in specs:
        (work / name / "plans").mkdir(parents=True, exist_ok=True)
    _STATE.update(datasets=specs, input_root=input_root, models=arm_models(input_root), work=work)
    with get_context("fork").Pool(workers) as pool:
        return pool.map(_prepare_job, jobs, chunksize=1)


def encode_rows(input_root: Path, jobs: list[tuple[str, int]], device: int, work: Path) -> dict[str, Any]:
    """Encode both arms' 104 variants of each job on one GPU under its lock, checking every digest."""
    specs = datasets(input_root)
    models = arm_models(input_root)
    items = pipeline02.counterpart_variants()
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
            age, sex = spec["covariates"](row)
            plan = nf.load_plan(spec["plans"] / f"{row['ecg_id']}.npz")
            inputs = []
            with np.load(work / dataset / "plans" / f"{row['ecg_id']}.npz") as prepared:
                for arm, model in models.items():
                    regions = nt.region_differences(raw, age, sex, model, items)
                    signals, _ = nt.region_signals(raw, plan["components"], regions["differences"], items)
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


def all_jobs(input_root: Path) -> list[tuple[str, int]]:
    """Return every (dataset, position) job, SPH first."""
    return [
        (name, position)
        for name, spec in datasets(input_root).items()
        for position in range(len(spec["rows"]))
    ]


def profile(input_root: Path, device: int, workers: int) -> dict[str, Any]:
    """Time preparation and encoding on 10 SPH and 10 fold-9 ECGs, without scoring."""
    receipt = binding(require_committed=False)
    work = WORK / "profile"
    jobs = [(name, position) for name in ("sph", "ptb") for position in range(PROFILE_RECORDS)]
    started = time.perf_counter()
    timings = prepare_jobs(input_root, jobs, workers, work)
    prepare_wall = time.perf_counter() - started
    encoded = encode_rows(input_root, jobs, device, work)
    equal = [
        bool(np.load(work / "sph/plans" / f"{ecg_id}.npz")["l6_full_equals_c0"])
        for name, ecg_id, _ in timings
        if name == "sph"
    ]
    details = {
        "records": len(jobs),
        "prepare_wall_seconds": prepare_wall,
        "workers": workers,
        "prepare_cpu_seconds_per_record": float(np.mean([value for _, _, value in timings])),
        "encode": encoded,
        "encode_seconds_per_record": encoded["wall_seconds"] / max(encoded["records"], 1),
        "l6_full_equals_c0": f"{sum(equal)} of {len(equal)}",
    }
    write_json_atomic(stages.stage_dir(input_root, OUTPUT, "profile") / "profile.json", details)
    return complete(input_root, "profile", receipt, details)


def prepare(input_root: Path, workers: int) -> dict[str, Any]:
    """Prepare both samples and require L6's full-replacement counterpart to equal NBR-02's C0 one."""
    receipt = binding()
    started = time.perf_counter()
    timings = prepare_jobs(input_root, all_jobs(input_root), workers, WORK / "full")
    frame = pd.DataFrame(timings, columns=["dataset", "ecg_id", "seconds"])
    frame.to_csv(stages.stage_dir(input_root, OUTPUT, "prepare") / "prepare.csv", index=False)
    sph = frame.loc[frame["dataset"].eq("sph"), "ecg_id"]
    equal = int(sum(bool(np.load(WORK / "full/sph/plans" / f"{e}.npz")["l6_full_equals_c0"]) for e in sph))
    details = {
        "records": frame.groupby("dataset").size().to_dict(),
        "wall_seconds": time.perf_counter() - started,
        "workers": workers,
        "l6_full_equals_c0_records": equal,
        "sph_records": len(sph),
    }
    if equal != len(sph):
        raise ValueError("L6 full-replacement counterparts differ from NBR-02's C0")
    return complete(input_root, "prepare", receipt, details)


def encode(input_root: Path, device: int, shard: int, shards: int) -> dict[str, Any]:
    """Encode one shard of both samples on one GPU."""
    receipt = binding()
    require(input_root, "prepare")
    result = encode_rows(input_root, all_jobs(input_root)[shard::shards], device, WORK / "full")
    write_json_atomic(stages.stage_dir(input_root, OUTPUT, f"encode_{shard}") / "encode.json", result)
    return complete(input_root, f"encode_{shard}", receipt, {**result, "shard": shard, "shards": shards})


def score(input_root: Path) -> dict[str, Any]:
    """Score both arms on both samples and require L6 full-replacement scores to equal NBR-02's C0 ones."""
    receipt = binding()
    require(input_root, "prepare")
    directory = stages.stage_dir(input_root, OUTPUT, "score")
    details: dict[str, Any] = {}
    for name, spec in datasets(input_root).items():
        ids = spec["rows"]["ecg_id"].tolist()
        scored = score_arms(input_root, ids, WORK / "full" / name)
        extra = {
            key: np.stack([np.load(WORK / "full" / name / "plans" / f"{e}.npz")[key] for e in ids])
            for key in ("L6_clipped", "L6_movement", "L6D_clipped", "L6D_movement")
        }
        write_npz_atomic(directory / f"{name}_scores.npz", ecg_id=np.array(ids, dtype=str), **scored, **extra)
        details[name] = {"records": len(ids)}
    sph = np.load(directory / "sph_scores.npz")
    previous = np.load(input_root / study02.OUTPUT / "score/scores.npz")
    full = len(pipeline02.counterpart_variants()) - 1
    difference = np.abs(sph["scores"][:, 0, full] - previous["scores"][:, 1, full])
    gap = float(difference.max())
    hgb = float(difference[:, :, nf.KINDS.index("hgb")].max())
    details["l6_full_vs_nbr02_c0_full_max_abs_difference"] = gap
    details["l6_full_vs_nbr02_c0_full_hgb_max_abs_difference"] = hgb
    details["l6_full_scores_exact"] = gap == 0.0
    write_json_atomic(directory / "integrity.json", details)
    if hgb != 0.0:
        raise ValueError(f"L6 full-replacement HGB scores differ from NBR-02's C0 by {hgb}")
    return complete(input_root, "score", receipt, details)


def score_arms(input_root: Path, ecg_ids: list[str], work: Path) -> dict[str, np.ndarray]:
    """Score the 52 variants of each arm with both frozen models."""
    bundle = joblib.load(input_root / nf.BUNDLE)
    count = len(pipeline02.counterpart_variants())
    measured, energy = [], []
    for ecg_id in ecg_ids:
        with np.load(work / "plans" / f"{ecg_id}.npz") as prepared:
            measured.append(np.stack([prepared[f"{arm}_measurements"] for arm in ARMS]))
            energy.append(np.stack([prepared[f"{arm}_energy"] for arm in ARMS]))
    pooled = np.stack([np.load(work / "pooled" / f"{ecg_id}.npy") for ecg_id in ecg_ids])
    records = len(ecg_ids)
    frame = pd.DataFrame(
        np.stack(measured).reshape(records * len(ARMS) * count, -1), columns=list(RAW_FEATURES)
    )
    flat = pooled.reshape(records * len(ARMS) * count, -1)
    scores = np.empty((records, len(ARMS), count, len(nf.MODELS), len(nf.KINDS)))
    for position, name in enumerate(nf.MODELS):
        fitted = bundle["fitted"][name]
        hgb = fitted["hgb"].predict_proba(frame)[:, 1]
        xecg = predict(fitted["xecg"], flat)
        values = np.stack([hgb, xecg, 0.5 * hgb + 0.5 * xecg], axis=1)
        scores[:, :, :, position] = values.reshape(records, len(ARMS), count, len(nf.KINDS))
    return {"scores": scores, "energy": np.stack(energy)}


def arm_tables(
    base: dict[str, np.ndarray],
    new: dict[str, np.ndarray],
    thresholds: dict[str, float],
    extra: dict[str, tuple[np.ndarray, np.ndarray]],
) -> dict[str, dict[str, Any]]:
    """NBR-01-layout tables of P0, any extra arm and the two new arms."""
    tables = {"P0": na.record_table(base["scores"], base["energy"], thresholds)}
    for name, (scores, energy) in extra.items():
        tables[name] = na.record_table(
            *na2.arm_arrays(base["scores"], base["energy"], scores, energy), thresholds
        )
    for position, arm in enumerate(ARMS):
        scores, energy = na2.arm_arrays(
            base["scores"], base["energy"], new["scores"][:, position], new["energy"][:, position]
        )
        tables[arm] = na.record_table(scores, energy, thresholds)
    return tables


def diagnostics(groups: np.ndarray, new: dict[str, np.ndarray], rows: np.ndarray) -> dict[str, float]:
    """Mean template movement and region clip rate of each arm by group."""
    result = {}
    group = groups[rows]
    for arm in ARMS:
        clipped = new[f"{arm}_clipped"][rows][:, :-1] > 0
        movement = new[f"{arm}_movement"][rows][:, :-1].mean(axis=1)
        for name in GROUPS:
            mask = group == name
            if mask.any():
                result[f"clip/{arm}/{name}"] = float(clipped[mask].mean())
                result[f"template/{arm}/{name}"] = float(movement[mask].mean())
    return result


def arm_statistics(
    tables: dict[str, dict[str, Any]],
    function: Callable[..., dict[str, float]],
    contrasts: tuple[tuple[str, str], ...],
    rows: np.ndarray,
) -> dict[str, float]:
    """Arm-prefixed statistics, selectivities and paired arm contrasts on one set of record positions."""
    result: dict[str, float] = {}
    for arm, table in tables.items():
        for key, value in function(table, rows).items():
            result[f"{arm}/{key}"] = value
        for score in na2.SCORES:
            result[f"{arm}/{score}/selectivity"] = na2.selectivity(result, f"{arm}/", score)
    names = [f"{score}/{name}" for score in na2.SCORES for name in na2.CONTRAST_NAMES]
    names += [f"energy/{name}" for name in na2.ENERGY_NAMES]
    for first, second in contrasts:
        for key in names:
            if f"{first}/{key}" in result and f"{second}/{key}" in result:
                result[f"{first}-{second}/{key}"] = result[f"{first}/{key}"] - result[f"{second}/{key}"]
    return result


def bootstrap(
    statistic: Callable[[np.ndarray], dict[str, float]], patients: np.ndarray
) -> dict[str, dict[str, float]]:
    """Point values and 97.5% whole-patient bootstrap intervals, 2,000 draws, seed 20261028."""
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
    tail = (100.0 - LEVEL) / 2
    result = {}
    for key, value in point.items():
        low, high = np.percentile(values[key], [tail, 100 - tail]) if values[key] else (np.nan, np.nan)
        result[key] = {
            "point": value,
            "low": float(low),
            "high": float(high),
            "valid_draws": len(values[key]),
        }
    return result


def decide(sph: dict[str, dict[str, float]], ptb_results: dict[str, dict[str, float]]) -> dict[str, Any]:
    """Apply the frozen NBR-04 rule to both arms and the tie rule."""
    result: dict[str, Any] = {}
    for arm in ARMS:
        parts = {
            "A_selectivity": (sph[f"{arm}-P0/all_age_blend/selectivity"], "low", 0.0),
            "B_territory": (sph[f"{arm}-P0/all_age_blend/territory"], "low", MARGIN),
            "C_component_flip": (sph[f"{arm}-P0/all_age_blend/abnormal/component_flip"], "low", MARGIN),
            "D_ptb_selectivity": (ptb_results[f"{arm}-P0/all_age_blend/selectivity"], "point", 0.0),
            "D_ptb_territory": (ptb_results[f"{arm}-P0/all_age_blend/territory"], "low", MARGIN),
        }
        decided = {
            name: {**value, "rule": f"{side} > {bound}", "passed": bool(value[side] > bound)}
            for name, (value, side, bound) in parts.items()
        }
        decided["D_confirmed"] = (
            decided["D_ptb_selectivity"]["passed"] and decided["D_ptb_territory"]["passed"]
        )
        decided["replaces_P0"] = (
            all(decided[name]["passed"] for name in ("A_selectivity", "B_territory", "C_component_flip"))
            and decided["D_confirmed"]
        )
        result[arm] = decided
    tie = sph["L6D-L6/all_age_blend/selectivity"]
    passing = [arm for arm in ARMS if result[arm]["replaces_P0"]]
    if len(passing) == 2:
        chosen = "L6D" if tie["low"] > 0 else "L6"
    else:
        chosen = passing[0] if passing else None
    result["tie_L6D_minus_L6_selectivity"] = tie
    result["chosen"] = chosen
    result["exported_arm"] = chosen or "L6"
    return result


def analyze(input_root: Path) -> dict[str, Any]:
    """Compute every statistic on both samples, then the decision."""
    receipt = binding()
    require(input_root, "score")
    started = time.perf_counter()
    specs = datasets(input_root)
    results = {}
    for name, spec in specs.items():
        base = dict(np.load(input_root / spec["scores"]))
        new = dict(np.load(input_root / OUTPUT / f"score/{name}_scores.npz"))
        if not np.array_equal(base["ecg_id"], new["ecg_id"]):
            raise ValueError(f"{name}: P0 and NBR-04 scores are misaligned")
        groups, patients = base["group"], base["patient"]
        if name == "sph":
            previous = dict(np.load(input_root / study02.OUTPUT / "score/scores.npz"))
            extra = {"C6": (previous["scores"][:, 2], previous["energy"][:, 2])}
            tables = arm_tables(base, new, nf.thresholds(input_root), extra)

            def function(
                table: dict[str, Any], rows: np.ndarray, groups: np.ndarray = groups
            ) -> dict[str, float]:
                return na.statistics(table, groups, rows)

            contrasts = SPH_CONTRASTS
        else:
            primary = study03.primary_groups(
                groups, study03.require(input_root, "prepare")["primary_territories"]
            )
            tables = arm_tables(base, new, ptb.thresholds(input_root), {})

            def function(
                table: dict[str, Any],
                rows: np.ndarray,
                groups: np.ndarray = groups,
                primary: np.ndarray = primary,
            ) -> dict[str, float]:
                return study03.statistics(table, groups, primary, rows)

            contrasts = PTB_CONTRASTS

        def statistic(
            rows: np.ndarray,
            tables: dict = tables,
            function: Callable = function,
            contrasts: tuple = contrasts,
            groups: np.ndarray = groups,
            new: dict = new,
        ) -> dict[str, float]:
            return {**arm_statistics(tables, function, contrasts, rows), **diagnostics(groups, new, rows)}

        results[name] = bootstrap(statistic, patients)
    reproduced = {
        "sph": results["sph"]["P0/all_age_blend/territory"]["point"],
        "ptb": results["ptb"]["P0/all_age_blend/territory"]["point"],
    }
    if reproduced != {"sph": study03.NBR01_PRIMARY, "ptb": NBR03_PRIMARY}:
        raise ValueError(f"P0 primary statistics do not reproduce NBR-01 and NBR-03: {reproduced}")
    decision = decide(results["sph"], results["ptb"])
    directory = stages.stage_dir(input_root, OUTPUT, "analyze")
    write_json_atomic(
        directory / "results.json",
        {"level": LEVEL, "decision": decision, "p0_reproduced": reproduced, **results},
    )
    details = {
        "draws": na.DRAWS,
        "seed": na.SEED,
        "level": LEVEL,
        "seconds": time.perf_counter() - started,
        "decision": decision,
    }
    return complete(input_root, "analyze", receipt, details)


def load_results(input_root: Path) -> dict[str, Any]:
    """Read the analyze stage's results after verifying it."""
    require(input_root, "analyze")
    return read_json(input_root / OUTPUT / "analyze/results.json")
