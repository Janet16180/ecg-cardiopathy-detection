"""
Experiment 073: per-lead interval criteria and beat-level criteria.

Phases::

    run_criteria073 measure   # own per-lead intervals and beat features (8 processes), Uni-G / 12SL intervals
    run_criteria073 maps      # 069 rebuild check, evidence, beat references, scored maps
    run_criteria073 score     # integrity, LocBench, strata, standardized noise, descriptives, result.json

``--smoke`` uses 16 scored, 200 clean-normal and 40 sanity records in a separate work folder; nothing is
scored. See ``docs/experiment-073-per-lead-criteria.md``.
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
from threadpoolctl import threadpool_limits

from ecg_experiment import ROOT
from ecg_experiment import combination066 as cb
from ecg_experiment import criteria069 as cr
from ecg_experiment import criteria073 as pl
from ecg_experiment.ecgad import load_split
from ecg_experiment.eda.ptbxl import load_metadata
from ecg_experiment.files import sha256_file, write_json_atomic
from ecg_experiment.locbench import ecgad_record_metrics, load_benchmark, scored_ecg_ids
from ecg_experiment.locbench_baselines061 import amplitude_map, exact_signal
from ecg_experiment.locbench_cache import open_cache
from ecg_experiment.locbench_data import PARTITION_PATH, REGION_TABLE, load_partition
from ecg_experiment.locbench_noise import standardized_noise_table
from ecg_experiment.locbench_strata import strata_paired
from ecg_experiment.paths import to_stored
from ecg_experiment.provenance import git_head, utc_now
from ecg_experiment.waveforms import LEADS

OUT = ROOT / "outputs/experiment073_per_lead_criteria_v1"
WORK = Path(os.environ.get("ECG073_WORK", "/tmp/ecg073-work"))
WORK069 = Path(os.environ.get("ECG069_WORK", "/tmp/ecg069-work")) / "full"
FEATURES = ROOT / "data/raw/ptb-xl-plus/1.0.1/features"
OUT069 = ROOT / "outputs/experiment069_criteria_v1"
LOCBENCH061 = ROOT / "outputs/experiment061_locbench_v1"
BAR = "qrs_prior"
REFERENCES = {
    BAR: LOCBENCH061 / "tables" / BAR,
    "criteria": OUT069 / "locbench/criteria/tables",
    "criteria_qrs": OUT069 / "locbench/criteria_qrs/tables",
}
MAPS = ("perlead", "perlead_median", "perlead_shift", "perlead_agree", "perlead_qrs", "perlead_own")
PRIMARY = "perlead"
WITH_BEATS = ("perlead", "perlead_shift", "perlead_agree", "perlead_own")
SANITY = {
    "1AVB": ("pr_long", None),
    "CLBBB": ("lbbb_lat", None),
    "CRBBB": ("rbbb_v1", None),
    "LAFB": (("lafb", "lafb_rs", "lafb_qr"), None),
    "IMI": ("q_path", ("II", "aVF")),
    "ASMI": ("q_v2v3", None),
    "LVH": (("sokolow", "r_avl", "cornell"), None),
    "LNGQT": ("qtc_long", None),
}
WORKERS = 8
SMOKE = {"scored": 16, "normals": 200, "sanity": 40}
TOLERANCE = 1e-12
BAR_A2 = 0.7419383358406554
A2_069 = {"criteria": 0.7240953915617058, "criteria_qrs": 0.7631057507901341}
SOURCES = (
    "ecg_experiment/criteria073.py",
    "scripts/experiments/run_criteria073.py",
    "docs/experiment-073-per-lead-criteria.md",
    "ecg_experiment/criteria069.py",
    "ecg_experiment/combination066.py",
    "ecg_experiment/normative_map062.py",
    "ecg_experiment/morphology_boundaries.py",
    "ecg_experiment/locbench.py",
)

_STATE: dict[str, Any] = {}


def finite_json(value: Any) -> Any:
    """Replace NaN and infinite floats by None, recursively, so the value is strict JSON."""
    if isinstance(value, dict):
        return {k: finite_json(v) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [finite_json(v) for v in value]
    if isinstance(value, np.ndarray):
        return finite_json(value.tolist())
    if isinstance(value, float | np.floating):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, np.integer):
        return int(value)
    return value


def work_dir(smoke: bool) -> Path:
    """Local work folder of the run."""
    path = WORK / ("smoke" if smoke else "full")
    path.mkdir(parents=True, exist_ok=True)
    return path


def holdout_ids() -> set[int]:
    """ECGAD holdout records, used only to assert that none is loaded."""
    split = load_split()
    return set(split.ecg_id[split.partition != "development"].astype(int))


def forbidden_fit_ids() -> set[int]:
    """ECGAD records (both halves) and evaluation-partition records: never fitted on."""
    partition = load_partition()
    evaluation = partition.ecg_id[partition.partition == "evaluation"]
    return set(load_split().ecg_id.astype(int)) | set(evaluation.astype(int))


def allowed_ids() -> set[int]:
    """PTB-XL folds 1-8 (the LocBench partition) and the ECGAD development records."""
    split = load_split()
    return set(load_partition().ecg_id.astype(int)) | set(split.ecg_id[split.partition == "development"])


def load_069() -> dict[str, Any]:
    """Load the 069 ``measure`` output (sets, source tables, beat components)."""
    with (WORK069 / "measurements.pkl").open("rb") as handle:
        return pickle.load(handle)  # noqa: S301


def record_sets(sets069: dict[str, np.ndarray], smoke: bool) -> dict[str, np.ndarray]:
    """Scored, clean-normal and sanity record IDs from 069 (smoke: small subsets), with separation checks."""
    sets = {k: np.asarray(sets069[k], int) for k in ("scored", "normals", "sanity")}
    if smoke:
        sets = {k: v[: SMOKE[k]] for k, v in sets.items()}
    if not smoke and not np.array_equal(sets["scored"], scored_ecg_ids()):
        raise ValueError("Scored records differ from LocBench")
    if holdout_ids() & set(sets["scored"].tolist()):
        raise ValueError("An ECGAD holdout record would be mapped")
    if (set(sets["normals"].tolist()) | set(sets["sanity"].tolist())) & forbidden_fit_ids():
        raise ValueError("A fit-side record is an ECGAD or evaluation record")
    partition = load_partition().set_index("ecg_id")
    normals = partition.loc[sets["normals"]]
    if not ((normals.partition == "fit") & normals.clean_normal).all():
        raise ValueError("A reference record is not a fit clean normal")
    return sets


def _measure_init(geometry: dict[int, np.ndarray | None]) -> None:
    threadpool_limits(limits=1)
    signals, index = open_cache()
    _STATE["signals"] = signals
    _STATE["row"] = dict(zip(index.ecg_id.astype(int), range(len(index)), strict=True))
    _STATE["geometry"] = geometry


def _measure(task: tuple[int, float]) -> tuple[int, bool, dict[str, float], dict[str, np.ndarray]]:
    ecg_id, rr = task
    raw = np.asarray(_STATE["signals"][_STATE["row"][ecg_id]], np.float64)
    intervals = cb.delineated_components(raw)
    same = pl.same_components(intervals, _STATE["geometry"][ecg_id])
    bounds = pl.kept_bounds(raw) if len(intervals) else np.zeros((0, len(LEADS), 3, 2), np.int64)
    if len(bounds) != len(intervals):
        raise ValueError(f"Support and component beat counts differ for {ecg_id}")
    return ecg_id, same, pl.own_intervals(bounds, rr), pl.beat_features(raw, intervals, bounds)


def read_features(name: str, columns: list[str], ids: np.ndarray) -> pd.DataFrame:
    """Read the needed columns of one PTB-XL+ feature file, keep only the measured records, order by ids."""
    raw = pd.read_csv(FEATURES / f"{name}_features.csv", usecols=columns).set_index("ecg_id")
    if not set(ids.tolist()) <= allowed_ids():
        raise ValueError("A measured record is outside folds 1-8 and ECGAD development")
    raw = raw[raw.index.isin(set(ids.tolist()))]
    return raw.reindex(ids)


def measure_phase(smoke: bool) -> None:
    """Own per-lead intervals, beat features, geometry check, Uni-G and 12SL per-lead intervals."""
    work = work_dir(smoke)
    began = time.time()
    data = load_069()
    sets = record_sets(data["sets"], smoke)
    ids = np.unique(np.concatenate(list(sets.values())))
    own_rr = data["tables"]["own"]["RR"]
    tasks = [(int(e), float(own_rr.loc[e])) for e in ids]
    geometry = {int(e): data["geometry"][int(e)] for e in ids}
    own, beats, mismatched = {}, {}, []
    with Pool(WORKERS, initializer=_measure_init, initargs=(geometry,)) as pool:
        for k, (ecg_id, same, values, features) in enumerate(pool.imap_unordered(_measure, tasks, 8)):
            own[ecg_id], beats[ecg_id] = values, features
            if not same:
                mismatched.append(ecg_id)
            if (k + 1) % 2000 == 0:
                print(f"measured {k + 1}/{len(ids)} {time.time() - began:.0f}s", flush=True)
    if mismatched:
        raise ValueError(f"Beat components differ from 069 for {len(mismatched)} records")
    consensus_rr = cr.consensus([data["tables"][k] for k in ("own", "12sl", "unig")])["RR"].reindex(ids)
    unig = pl.unig_intervals(read_features("unig", pl.unig_columns(), ids), consensus_rr)
    sl12 = pl.sl12_intervals(read_features("12sl", pl.sl12_columns(), ids))
    tables = {
        "own": pd.DataFrame.from_dict(own, orient="index").reindex(ids),
        "commercial": pd.concat([unig, sl12], axis=1),
    }
    with (work / "measurements073.pkl").open("wb") as handle:
        pickle.dump({"sets": sets, "tables": tables, "beats": beats}, handle, pickle.HIGHEST_PROTOCOL)
    receipt = {
        "created": utc_now(),
        "git_head": git_head(ROOT),
        "records": {k: int(len(v)) for k, v in sets.items()},
        "measured": int(len(ids)),
        "geometry_equal_069": int(len(ids) - len(mismatched)),
        "measurements069_sha256": sha256_file(WORK069 / "measurements.pkl"),
        "missing_share": {k: t.isna().mean().round(4).to_dict() for k, t in tables.items()},
        "seconds": time.time() - began,
    }
    write_json_atomic(work / "measure_receipt.json", finite_json(receipt))
    print(json.dumps({k: v for k, v in receipt.items() if k != "missing_share"}), flush=True)


def demographics(ids: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Age and sex from the cache index, in ids order."""
    _, index = open_cache()
    table = index.set_index("ecg_id").loc[ids]
    return table.age.to_numpy(np.float64), table.sex.to_numpy(np.float64)


def reference_columns_069() -> list[str]:
    """Return the 069 normal-reference columns: criterion inputs and the agreement measures."""
    agreement = [f"{m}_{lead}" for m in cr.AGREEMENT for lead in LEADS]
    return list(dict.fromkeys(cr.used_columns() + agreement))


def build_evidence(data069: dict[str, Any], data: dict[str, Any]) -> dict[str, Any]:
    """
    069 rebuild, 073 tables, normal references, exceedances and median-beat lead x wave scores.

    Returns
    -------
    dict
        ``ids``, ``scores`` per map (records, 12, 5), ``z`` per table, ``ref``, ``offsets``, ``reliability``,
        ``criteria069`` scores and z.
    """
    sets = data["sets"]
    ids = data["tables"]["own"].index.to_numpy(int)
    sources = [data069["tables"][k].reindex(ids) for k in ("own", "12sl", "unig")]
    consensus = cr.add_derived(cr.consensus(sources))
    own069 = cr.add_derived(sources[0])
    age, sex = demographics(ids)
    normals = sets["normals"]
    ref069 = cr.normal_reference(consensus.loc[normals], reference_columns_069())
    z069 = cr.exceedances(consensus, ref069["scale"], age, sex)
    table = pd.concat([consensus, data["tables"]["commercial"]], axis=1)
    own = pd.concat([own069, data["tables"]["own"]], axis=1)
    agreement = [f"{m}_{lead}" for m in cr.AGREEMENT for lead in LEADS]
    columns = list(dict.fromkeys(pl.used_columns() + agreement))
    ref = {"perlead": pl.normal_reference(table.loc[normals], columns)}
    ref["own"] = pl.normal_reference(own.loc[normals], pl.used_columns())
    offsets = pl.shift_offsets(table.loc[normals])
    z = {
        "perlead": pl.exceedances(table, ref["perlead"]["scale"], age, sex),
        "shift": pl.exceedances(table, ref["perlead"]["scale"], age, sex, offsets=offsets),
        "own": pl.exceedances(own, ref["own"]["scale"], age, sex),
    }
    d = cr.disagreement(sources, ref069["scale"])
    q95 = np.nanpercentile(d[np.isin(ids, normals)], 95, axis=0)
    r = cr.reliability(d, q95)
    base = pl.lead_wave(cr.evidence(z["perlead"]))
    scores = {
        "perlead": base,
        "perlead_median": base,
        "perlead_shift": pl.lead_wave(cr.evidence(z["shift"])),
        "perlead_agree": base * r[:, :, None],
        "perlead_own": pl.lead_wave(cr.evidence(z["own"])),
    }
    return {
        "ids": ids,
        "scores": scores,
        "z": z,
        "z069": z069,
        "criteria069": cr.lead_wave(cr.evidence(z069)),
        "ref": ref,
        "offsets": offsets,
        "reliability": r,
        "q95": q95,
    }


def beat_scores(data: dict[str, Any], ref: dict[str, Any]) -> dict[int, np.ndarray]:
    """Beat x lead x wave scores of every measured record."""
    return {e: pl.beat_wave(pl.beat_exceedances(f, ref)) for e, f in data["beats"].items()}


def _paint_init(work: str, names: tuple[str, ...]) -> None:
    threadpool_limits(limits=1)
    signals, index = open_cache()
    _STATE["signals"] = signals
    _STATE["row"] = dict(zip(index.ecg_id.astype(int), range(len(index)), strict=True))
    _STATE["maps"] = {
        name: np.lib.format.open_memmap(Path(work) / f"{name}.npy", mode="r+") for name in names
    }


def record_maps(
    amplitude: np.ndarray, intervals: np.ndarray, scores: dict[str, np.ndarray], beats: np.ndarray
) -> dict[str, np.ndarray]:
    """Every 073 map of one record from its median-beat scores, beat scores and components."""
    fields = {}
    for name in ("perlead", "perlead_median", "perlead_shift", "perlead_agree", "perlead_own"):
        beat = beats if name in WITH_BEATS else None
        if name == "perlead_agree":
            beat = beats * scores["reliability"][None, :, None]
        fields[name] = pl.paint_beats(scores[name], beat, intervals)
    maps = {name: cr.finish(field, amplitude) for name, field in fields.items()}
    maps["perlead_qrs"] = pl.lexicographic(fields["perlead"], intervals, amplitude)
    return maps


def _paint(task: tuple[int, int, np.ndarray, dict[str, np.ndarray], np.ndarray]) -> int:
    position, ecg_id, intervals, scores, beats = task
    amplitude = amplitude_map(exact_signal(_STATE["signals"][_STATE["row"][ecg_id]]))
    for name, values in record_maps(amplitude, intervals, scores, beats).items():
        if not np.isfinite(values).all():
            raise ValueError(f"Nonfinite {name} map for {ecg_id}")
        _STATE["maps"][name][position] = values
    return position


def descriptives(ev: dict[str, Any], data: dict[str, Any], beats: dict[int, np.ndarray]) -> dict[str, Any]:
    """Fit clean-normal prevalence, sanity readings, beat-level firing and references."""
    sets = data["sets"]
    normals = sets["normals"]
    rows = np.searchsorted(ev["ids"], normals)
    names073 = [c.name for c in pl.CRITERIA]
    names069 = [c.name for c in cr.CRITERIA]
    prevalence = {
        "consensus_069": dict(
            zip(names069, cr.met_any_lead(ev["z069"][rows]).mean(axis=0).round(4).tolist(), strict=True)
        ),
    }
    for kind, z in ev["z"].items():
        met = cr.met_any_lead(z[rows]).mean(axis=0).round(4).tolist()
        prevalence[kind] = dict(zip(names073, met, strict=True))
    normal_beats = [beats[int(e)] for e in normals]
    fired = [b.max(axis=(1, 2)) > 0 for b in normal_beats if len(b)]
    beat_firing = {
        "records_with_beat_evidence": float(np.mean([f.any() for f in fired])),
        "beats_with_evidence": float(np.concatenate(fired).mean()),
        "records": len(fired),
    }
    return {
        "criterion_prevalence_fit_clean_normals": prevalence,
        "beat_firing_fit_clean_normals": beat_firing,
        "sanity_fit_statements": sanity_readings(ev, sets),
        "shift_offsets": ev["offsets"],
        "q95_disagreement": dict(zip(LEADS, ev["q95"].round(4).tolist(), strict=True)),
        "normal_reference": {k: v.round(6).to_dict(orient="index") for k, v in ev["ref"].items()},
    }


def sanity_readings(ev: dict[str, Any], sets: dict[str, np.ndarray]) -> dict[str, Any]:
    """Share of fit records with a statement that meet the matching criterion (069 and 073 tables)."""
    meta = load_metadata()
    tables = {"consensus_069": (ev["z069"], [c.name for c in cr.CRITERIA])}
    tables |= {k: (z, [c.name for c in pl.CRITERIA]) for k, z in ev["z"].items()}
    out = {}
    for statement, (criteria, leads) in SANITY.items():
        chosen = np.array([e for e in sets["sanity"] if meta.at[e, "scp_codes"].get(statement, 0) >= 50], int)
        if not len(chosen):
            continue
        rows = np.searchsorted(ev["ids"], chosen)
        lead_rows = [LEADS.index(lead) for lead in leads] if leads else list(range(len(LEADS)))
        wanted_names = (criteria,) if isinstance(criteria, str) else criteria
        reading: dict[str, Any] = {"records": int(len(chosen))}
        for kind, (z, names) in tables.items():
            wanted = [names.index(c) for c in wanted_names if c in names]
            block = np.nan_to_num(z[rows][:, wanted][:, :, lead_rows], nan=-np.inf)
            reading[kind] = float((block >= 0).any(axis=(1, 2)).mean())
        out[statement] = reading
    return out


def check_069_scores(ev: dict[str, Any], scored: np.ndarray) -> dict[str, Any]:
    """069 ``criteria`` lead x wave scores rebuilt here against 069's ``evidence_scored.npz``."""
    with np.load(WORK069 / "evidence_scored.npz") as saved:
        if not np.array_equal(saved["ecg_id"], scored):
            raise ValueError("069 scored order differs")
        theirs = saved["criteria"]
    mine = ev["criteria069"][np.searchsorted(ev["ids"], scored)]
    largest = float(np.abs(mine - theirs).max())
    return {"records": int(len(scored)), "max_abs_difference": largest, "pass": bool(largest == 0.0)}


def maps_phase(smoke: bool) -> None:
    """069 rebuild check, evidence of every record, beat references and the scored maps."""
    work = work_dir(smoke)
    began = time.time()
    data069 = load_069()
    with (work / "measurements073.pkl").open("rb") as handle:
        data = pickle.load(handle)  # noqa: S301
    sets = data["sets"]
    ev = build_evidence(data069, data)
    scored = sets["scored"]
    check069 = None if smoke else check_069_scores(ev, scored)
    if check069 is not None and not check069["pass"]:
        raise ValueError(f"069 scores not reproduced: {check069}")
    ref_beats = pl.beat_reference([data["beats"][int(e)] for e in sets["normals"]])
    beats = beat_scores(data, ref_beats)
    for name in MAPS:
        np.lib.format.open_memmap(
            work / f"{name}.npy", mode="w+", dtype=np.float32, shape=(len(scored), 12, 5000)
        )
    rows = np.searchsorted(ev["ids"], scored)
    tasks = []
    for position, (ecg_id, row) in enumerate(zip(scored.tolist(), rows, strict=True)):
        scores = {name: values[row] for name, values in ev["scores"].items()}
        scores["reliability"] = ev["reliability"][row]
        intervals = pl.components_of(data069["geometry"][ecg_id])
        tasks.append((position, ecg_id, intervals, scores, beats[ecg_id]))
    with Pool(WORKERS, initializer=_paint_init, initargs=(str(work), MAPS)) as pool:
        list(pool.imap_unordered(_paint, tasks, chunksize=16))
    (work / "scored_ids.json").write_text(json.dumps([int(e) for e in scored]))
    with (work / "evidence_scored.pkl").open("wb") as handle:
        pickle.dump(
            {
                "ecg_id": scored,
                "scores": {name: values[rows] for name, values in ev["scores"].items()},
                "criteria069": ev["criteria069"][rows],
                "beats": {int(e): beats[int(e)] for e in scored},
            },
            handle,
            pickle.HIGHEST_PROTOCOL,
        )
    receipt = {
        "created": utc_now(),
        "git_head": git_head(ROOT),
        "check_069_scores": check069,
        "beat_reference": ref_beats,
        "files": {
            n: {"file": str(work / f"{n}.npy"), "sha256": sha256_file(work / f"{n}.npy")} for n in MAPS
        },
        "descriptives": descriptives(ev, data, beats),
        "seconds": time.time() - began,
    }
    write_json_atomic(work / "maps_receipt.json", finite_json(receipt))
    print(json.dumps(receipt["descriptives"]["criterion_prevalence_fit_clean_normals"]), flush=True)
    print(json.dumps(receipt["descriptives"]["sanity_fit_statements"]), flush=True)
    print(json.dumps(receipt["descriptives"]["beat_firing_fit_clean_normals"]), flush=True)


def ecgad_a2(maps: np.ndarray, bench: Any) -> np.ndarray:
    """Per-record ECGAD development A2 of maps in benchmark order."""
    return np.array(
        [
            ecgad_record_metrics(np.asarray(m, np.float32), y)["A2"]
            for m, y in zip(maps, bench.ecgad_labels, strict=True)
        ]
    )


def compare_a2(values: np.ndarray, folder: Path, target: float) -> dict[str, Any]:
    """Per-record A2 against a saved ``ecgad.csv`` and the mean against its target."""
    saved = pd.read_csv(folder / "ecgad.csv")["A2"].to_numpy()
    largest = float(np.nanmax(np.abs(values - saved)))
    mean = float(np.nanmean(values))
    ok = np.array_equal(np.isnan(values), np.isnan(saved)) and largest <= TOLERANCE
    return {
        "per_record_max_abs_difference": largest,
        "A2": mean,
        "A2_target": target,
        "pass": bool(ok and abs(mean - target) <= TOLERANCE),
    }


def reproduce_bar() -> dict[str, Any]:
    """061 ``qrs_prior`` ECGAD development A2 through this run's scoring path."""
    bench = load_benchmark()
    folder = REFERENCES[BAR]
    if (
        json.loads((folder / "meta.json").read_text())["ecgad_ecg_ids"]
        != bench.ecgad.ecg_id.astype(int).tolist()
    ):
        raise ValueError("ECGAD order of the bar differs from the benchmark")
    result = compare_a2(ecgad_a2(np.load(folder / "ecgad_maps.npy", mmap_mode="r"), bench), folder, BAR_A2)
    if not result["pass"]:
        raise ValueError(f"Bar not reproduced: {result}")
    return result


def reproduce_069_maps(work: Path) -> dict[str, Any]:
    """Repaint 069 ``criteria`` and ``criteria_qrs`` ECGAD maps; compare with 069's maps and A2 tables."""
    bench = load_benchmark()
    data069 = load_069()
    with (work / "evidence_scored.pkl").open("rb") as handle:
        saved = pickle.load(handle)  # noqa: S301
    row = {int(e): k for k, e in enumerate(saved["ecg_id"].tolist())}
    signals, index = open_cache()
    cache_row = dict(zip(index.ecg_id.astype(int), range(len(index)), strict=True))
    rebuilt = {"criteria": [], "criteria_qrs": []}
    for ecg_id in bench.ecgad.ecg_id.astype(int).tolist():
        intervals = pl.components_of(data069["geometry"][ecg_id])
        amplitude = amplitude_map(exact_signal(signals[cache_row[ecg_id]]))
        field = cr.paint(saved["criteria069"][row[ecg_id]], intervals)
        rebuilt["criteria"].append(cr.finish(field, amplitude))
        rebuilt["criteria_qrs"].append(cr.lexicographic(field, intervals, amplitude))
    out = {}
    for name, maps in rebuilt.items():
        stacked = np.stack(maps)
        original = np.load(REFERENCES[name] / "ecgad_maps.npy", mmap_mode="r")
        result = compare_a2(ecgad_a2(stacked, bench), REFERENCES[name], A2_069[name])
        result["map_max_abs_difference"] = float(np.abs(stacked - original).max())
        result["pass"] = bool(result["pass"] and result["map_max_abs_difference"] == 0.0)
        out[name] = result
    if not all(r["pass"] for r in out.values()):
        raise ValueError(f"069 maps not reproduced: {out}")
    return out


def separation(sets: dict[str, np.ndarray]) -> dict[str, Any]:
    """Data-separation checks of the fitted quantities and of the mapped records."""
    forbidden = forbidden_fit_ids()
    fit = load_partition()
    fit = set(fit.ecg_id[fit.partition == "fit"].astype(int))
    measured = set(np.concatenate(list(sets.values())).tolist())
    checks = {
        "normal_reference_records": int(len(sets["normals"])),
        "normal_reference_forbidden": len(set(sets["normals"].tolist()) & forbidden),
        "normal_reference_outside_fit": len(set(sets["normals"].tolist()) - fit),
        "sanity_forbidden": len(set(sets["sanity"].tolist()) & forbidden),
        "measured_outside_allowed": len(measured - allowed_ids()),
        "mapped_ecgad_holdout": len(set(sets["scored"].tolist()) & holdout_ids()),
    }
    checks["pass"] = all(v == 0 for k, v in checks.items() if not k.endswith("_records"))
    if not checks["pass"]:
        raise ValueError(f"Data separation failed: {checks}")
    return checks


def run_score(maps: Path, ids: Path, name: str, target: Path, compare: list[Path]) -> dict[str, Any]:
    """Run the LocBench scoring CLI and return its summary."""
    command = [str(ROOT / ".venv/bin/python"), "-m", "scripts.experiments.run_locbench_score"]
    command += ["--maps", str(maps), "--ids", str(ids), "--name", name, "--out", str(target)]
    for folder in compare:
        command += ["--compare", str(folder)]
    subprocess.run(command, check=True, cwd=ROOT)
    return json.loads((target / "summary.json").read_text())


def copy_references() -> dict[str, Path]:
    """Copy the reference tables into this run's folder (the strata and noise CLIs write into them)."""
    copies = {}
    for name, source in REFERENCES.items():
        target = OUT / "ref" / name
        if not target.exists():
            shutil.copytree(source, target, ignore=shutil.ignore_patterns("strata.*", "noise_std.*"))
        copies[name] = target
    return copies


def score_maps(work: Path, refs: dict[str, Path]) -> dict[str, dict]:
    """Score every map with the LocBench CLI and its paired comparisons."""
    base = [refs[BAR], refs["criteria"], refs["criteria_qrs"]]
    summaries = {}
    for name in MAPS:
        compare = base + ([OUT / "locbench" / PRIMARY / "tables"] if name != PRIMARY else [])
        target = OUT / "locbench" / name
        summaries[name] = run_score(work / f"{name}.npy", work / "scored_ids.json", name, target, compare)
        print(f"{name}: A2 {summaries[name]['metrics']['A2']['value']:.4f}", flush=True)
    return summaries


def run_secondary_clis(refs: dict[str, Path]) -> None:
    """Strata (``qrs_prior`` reference first) and standardized noise readings for every map and reference."""
    tables = [OUT / "locbench" / name / "tables" for name in MAPS]
    others = [refs["criteria"], refs["criteria_qrs"]]
    python = str(ROOT / ".venv/bin/python")
    strata = [
        python,
        "-m",
        "scripts.experiments.run_locbench_strata",
        str(refs[BAR]),
        *map(str, tables + others),
    ]
    subprocess.run(strata, check=True, cwd=ROOT)
    noise = [python, "-m", "scripts.experiments.run_locbench_noise", *map(str, [*tables, *others, refs[BAR]])]
    subprocess.run(noise, check=True, cwd=ROOT)


def paired_secondary(refs: dict[str, Path]) -> dict[str, Any]:
    """Paired strata against ``qrs_prior`` and 069 ``criteria``; paired C1_std against the references."""
    folders = {name: OUT / "locbench" / name / "tables" for name in MAPS} | refs
    strata = {name: pd.read_csv(folder / "strata.csv") for name, folder in folders.items()}
    noise = {name: standardized_noise_table(folder) for name, folder in folders.items()}
    out: dict[str, Any] = {}
    for name in MAPS:
        out[name] = {
            f"{stratum}_vs_{other}": strata_paired(strata[name], strata[other], stratum)
            for stratum in ("A2_qrs", "A2_nonqrs")
            for other in (BAR, "criteria")
        }
        for other in (BAR, "criteria", PRIMARY):
            if other != name:
                out[name][f"C1_std_vs_{other}"] = strata_paired(noise[name], noise[other], "C1_std")
    return out


def lead_specificity(evidence: dict[str, Any]) -> dict[str, Any]:
    """Evidence-lead counts on marked ECGAD development records for ``perlead`` and 069 ``criteria``."""
    bench = load_benchmark()
    row = {int(e): k for k, e in enumerate(evidence["ecg_id"].tolist())}
    marked = bench.ecgad[bench.ecgad.marked.astype(bool)].ecg_id.astype(int).tolist()
    rows = [row[e] for e in marked]
    out: dict[str, Any] = {"marked_records": len(marked)}
    for name, scores in (("perlead", evidence["scores"]["perlead"]), ("criteria", evidence["criteria069"])):
        leads = cr.evidence_leads(scores[rows])
        count = leads.sum(axis=1)
        out[name] = {
            "all_12_share": float((count == len(LEADS)).mean()),
            "none_share": float((count == 0).mean()),
            "mean_evidence_leads": float(count.mean()),
        }
    beat_any = [bool(evidence["beats"][e].max() > 0) if len(evidence["beats"][e]) else False for e in marked]
    out["perlead_beat_evidence_share"] = float(np.mean(beat_any))
    return out


def region_specificity(scores: np.ndarray, ids: np.ndarray) -> dict[str, Any]:
    """Evidence leads against the textbook leads on region-set records whose lead set is not all leads."""
    bench = load_benchmark()
    region = bench.region.reset_index(drop=True)
    row = {int(e): k for k, e in enumerate(ids.tolist())}
    keep = ~region.all_leads.to_numpy(bool)
    textbook = bench.region_leads[keep]
    evidence = cr.evidence_leads(np.stack([scores[row[int(e)]] for e in region.ecg_id[keep]]))
    union = (evidence | textbook).sum(axis=1)
    jaccard = np.where(union > 0, (evidence & textbook).sum(axis=1) / np.maximum(union, 1), 0.0)
    frame = pd.DataFrame(
        {
            "family": region.family[keep].to_numpy(),
            "empty": ~evidence.any(axis=1),
            "jaccard": jaccard,
            "departure": (evidence & ~textbook).any(axis=1),
            "departure_leads": (evidence & ~textbook).sum(axis=1),
            "evidence_leads": evidence.sum(axis=1),
        }
    )
    summary = frame.groupby("family").agg(
        records=("empty", "size"),
        empty_share=("empty", "mean"),
        mean_jaccard=("jaccard", "mean"),
        departure_share=("departure", "mean"),
        mean_departure_leads=("departure_leads", "mean"),
        mean_evidence_leads=("evidence_leads", "mean"),
    )
    return {
        "by_family": summary.round(4).to_dict(orient="index"),
        "overall": frame.drop(columns="family").mean().to_dict(),
        "records": int(keep.sum()),
    }


def ecgad_departures(scores: np.ndarray, ids: np.ndarray) -> dict[str, Any]:
    """Cardiologist-marked share of departure, shared, textbook-only and other leads on ECGAD development."""
    bench = load_benchmark()
    meta = load_metadata()
    row = {int(e): k for k, e in enumerate(ids.tolist())}
    counts = {g: [0, 0] for g in ("departure", "shared", "textbook_only", "neither")}
    used = 0
    for k, record in enumerate(bench.ecgad.itertuples(index=False)):
        codes = meta.at[int(record.ecg_id), "scp_codes"]
        statements = [s for s in REGION_TABLE if codes.get(s, 0) >= 50]
        textbook = np.isin(LEADS, [lead for s in statements for lead in REGION_TABLE[s][0]])
        if not record.marked or not statements or textbook.all():
            continue
        used += 1
        marked = bench.ecgad_labels[k][:, 100:4900].any(axis=1)
        evidence = cr.evidence_leads(scores[row[int(record.ecg_id)]][None])[0]
        groups = {
            "departure": evidence & ~textbook,
            "shared": evidence & textbook,
            "textbook_only": ~evidence & textbook,
            "neither": ~evidence & ~textbook,
        }
        for name, mask in groups.items():
            counts[name][0] += int((mask & marked).sum())
            counts[name][1] += int(mask.sum())
    return {
        "records": used,
        "groups": {
            g: {"marked": m, "leads": n, "marked_share": (m / n) if n else None}
            for g, (m, n) in counts.items()
        },
    }


def readings(refs: dict[str, Path], summaries: dict[str, dict]) -> tuple[dict, dict]:
    """Every map's metrics, strata and noise readings, and the references' strata and noise."""
    mine = {}
    for name in MAPS:
        folder = OUT / "locbench" / name / "tables"
        mine[name] = {
            "metrics": summaries[name]["metrics"],
            "paired": summaries[name]["paired"],
            "strata": json.loads((folder / "strata.json").read_text()),
            "noise_std": json.loads((folder / "noise_std.json").read_text()),
            "maps_sha256": summaries[name]["inputs"]["maps_sha256"],
        }
    refs_out = {}
    for name, folder in refs.items():
        summary = REFERENCES[name].parent / "summary.json"
        refs_out[name] = {
            "metrics": json.loads(summary.read_text())["metrics"] if summary.exists() else None,
            "strata": json.loads((folder / "strata.json").read_text()),
            "noise_std": json.loads((folder / "noise_std.json").read_text()),
        }
    return mine, refs_out


def decision(interval: dict[str, Any]) -> str:
    """Primary decision words from a paired interval."""
    if interval["ci_low"] > 0:
        return "localizes"
    if interval["ci_high"] < 0:
        return "below"
    return "matches"


def decisions(summaries: dict[str, dict], paired: dict[str, Any]) -> dict[str, Any]:
    """Primary and key secondary comparisons with their declared words."""
    primary = summaries[PRIMARY]["paired"][BAR]["A2"]
    a3 = summaries[PRIMARY]["paired"]["criteria"]["A3"]
    agree_a2 = summaries["perlead_agree"]["paired"][PRIMARY]["A2"]
    agree_noise = paired["perlead_agree"][f"C1_std_vs_{PRIMARY}"]
    keep_factor = bool(agree_noise["ci_high"] < 0 and agree_a2["ci_high"] >= 0)
    return {
        "primary": {"map": PRIMARY, "against": BAR, "A2_difference": primary, "decision": decision(primary)},
        "key_secondary": {
            "map": PRIMARY,
            "against": "criteria",
            "A3_difference": a3,
            "decision": "lead localization recovers" if a3["ci_low"] > 0 else "does not recover",
        },
        "reliability_factor": {
            "A2_perlead_agree_minus_perlead": agree_a2,
            "C1_std_perlead_agree_minus_perlead": agree_noise,
            "decision": "keep" if keep_factor else "drop",
        },
        "exploratory": "ECGAD development has been compared many times; no reading here is confirmed.",
    }


def inputs_record(work: Path) -> dict[str, Any]:
    """Hashes of the inputs and sources of the run."""
    return {
        "partition": to_stored(PARTITION_PATH),
        "partition_sha256": sha256_file(PARTITION_PATH),
        "features_sha256": {n: sha256_file(FEATURES / f"{n}_features.csv") for n in ("12sl", "unig")},
        "measurements069_sha256": sha256_file(WORK069 / "measurements.pkl"),
        "evidence069_sha256": sha256_file(WORK069 / "evidence_scored.npz"),
        "measurements073_sha256": sha256_file(work / "measurements073.pkl"),
        "sources_sha256": {path: sha256_file(ROOT / path) for path in SOURCES},
    }


def score_phase() -> None:
    """Integrity checks, LocBench scoring, strata, standardized noise, descriptives, result.json."""
    work = work_dir(False)
    OUT.mkdir(parents=True, exist_ok=True)
    began = time.time()
    with (work / "measurements073.pkl").open("rb") as handle:
        sets = pickle.load(handle)["sets"]  # noqa: S301
    scored = np.array(json.loads((work / "scored_ids.json").read_text()), np.int64)
    if not np.array_equal(scored, scored_ecg_ids()):
        raise ValueError("Scored records changed")
    maps_receipt = json.loads((work / "maps_receipt.json").read_text())
    measure_receipt = json.loads((work / "measure_receipt.json").read_text())
    integrity = {
        "bar": reproduce_bar(),
        "criteria069_scores": maps_receipt["check_069_scores"],
        "criteria069_maps": reproduce_069_maps(work),
        "geometry": {
            "records": measure_receipt["measured"],
            "equal": measure_receipt["geometry_equal_069"],
            "pass": measure_receipt["measured"] == measure_receipt["geometry_equal_069"],
        },
        "separation": separation(sets),
    }
    refs = copy_references()
    summaries = score_maps(work, refs)
    run_secondary_clis(refs)
    paired = paired_secondary(refs)
    mine, refs_out = readings(refs, summaries)
    with (work / "evidence_scored.pkl").open("rb") as handle:
        evidence = pickle.load(handle)  # noqa: S301
    specificity = {
        "ecgad_lead_specificity": lead_specificity(evidence),
        "region_set": region_specificity(evidence["scores"]["perlead"], scored),
        "ecgad_departures": ecgad_departures(evidence["scores"]["perlead"], scored),
    }
    result = {
        "experiment": "073",
        "protocol": "docs/experiment-073-per-lead-criteria.md",
        "protocol_sha256": sha256_file(ROOT / "docs/experiment-073-per-lead-criteria.md"),
        "created": utc_now(),
        "git_head": git_head(ROOT),
        "inputs": inputs_record(work),
        "integrity": integrity,
        "measure": measure_receipt,
        "maps": maps_receipt,
        "readings": mine,
        "paired_secondary": paired,
        "reference_readings": refs_out,
        "specificity": specificity,
        "decisions": decisions(summaries, paired),
        "score_seconds": time.time() - began,
    }
    write_json_atomic(OUT / "result.json", finite_json(result))
    print(json.dumps(result["decisions"], indent=1), flush=True)


def main() -> None:
    """Command-line entry point."""
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=["measure", "maps", "score"])
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    if args.phase == "measure":
        measure_phase(args.smoke)
    elif args.phase == "maps":
        maps_phase(args.smoke)
    elif args.smoke:
        raise ValueError("The smoke run is not scored")
    else:
        score_phase()


if __name__ == "__main__":
    main()
