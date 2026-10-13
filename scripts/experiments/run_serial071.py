"""
Experiment 071: same-patient serial change maps.

Phases::

    run_serial071 count       # pairs, counts and comparator coverage from metadata only
    run_serial071 features    # frozen 062 beat summaries of every needed record (CPU, 8 processes)
    run_serial071 fit         # same-patient and cross-patient reference-difference models (fit pairs only)
    run_serial071 score       # change maps and readings on evaluation pairs, result_primary.json
    run_serial071 secondary   # change-supervised unit model on fit pairs, maps of the LocBench scored records
    run_serial071 locbench    # integrity, LocBench scoring of the secondary map, strata, noise
    run_serial071 report      # result.json from the phase outputs

``--smoke`` reads fit-partition pairs in place of evaluation pairs, writes to a separate folder and scores
nothing on evaluation or ECGAD records. See ``docs/experiment-071-serial.md``.
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
from ecg_experiment import combiner068 as cv
from ecg_experiment import normative_map062 as nm
from ecg_experiment import serial071 as sr
from ecg_experiment.ecgad import load_split
from ecg_experiment.eda.ptbxl import load_metadata, load_statements
from ecg_experiment.files import sha256_file, write_json_atomic
from ecg_experiment.locbench import (
    FAMILIES,
    ecgad_record_metrics,
    fiducial_path,
    load_benchmark,
    read_components,
    scored_ecg_ids,
)
from ecg_experiment.locbench_baselines061 import amplitude_map, exact_signal, qrs_prior_map
from ecg_experiment.locbench_cache import CACHE_DIR, open_cache
from ecg_experiment.locbench_data import PARTITION_PATH, load_partition
from ecg_experiment.paths import to_stored
from ecg_experiment.provenance import git_head, utc_now
from ecg_experiment.waveforms import LEADS

OUT = ROOT / "outputs/experiment071_serial_v1"
SMOKE_OUT = ROOT / "outputs/experiment071_serial_smoke"
WORK = Path(os.environ.get("ECG071_WORK", "/tmp/ecg071-work"))
LOCBENCH061 = ROOT / "outputs/experiment061_locbench_v1"
NORMATIVE062 = ROOT / "outputs/experiment062_normative_v1"
MIL065 = ROOT / "outputs/experiment065_mil_v1"
OUT066 = ROOT / "outputs/experiment066_combination_v1"
OUT068 = ROOT / "outputs/experiment068_combiner_v2"
WORK066 = Path("/tmp/ecg066-work/full")
WORK068 = Path("/tmp/ecg068-work/full")
WORKERS = 8
TOLERANCE = 1e-12
SETS = {
    "forward": ("normal", "forward"),
    "both": ("normal", "both"),
    "finding_free": ("finding_free", "forward"),
}
PRIMARY_SET = "forward"
SECONDARY_TRAIN = ("finding_free", "both")
SERIAL_MAPS = ("serial", "serial_interval", "cross")
COMPARATORS = ("normative_v1", "mil_lse", "textbook", "units_v2", "qrs_prior")
MAP_NAMES = (*SERIAL_MAPS, *COMPARATORS)
READINGS = ("B1u", "B1", "B2", "B3")
SECONDARY = "serial_units"
REFERENCES = {
    "qrs_prior": LOCBENCH061 / "tables/qrs_prior",
    "units_v2": OUT068 / "locbench/units_v2/tables",
}
SAVED_HASHES = {
    "normative_v1": (NORMATIVE062 / "maps_normative_v1.npy", NORMATIVE062 / "maps_receipt.json"),
    "mil_lse": (MIL065 / "maps_mil_lse_evaluation.npy", MIL065 / "result.json"),
    "textbook": (WORK066 / "textbook.npy", OUT066 / "maps_receipt.json"),
    "units_v2": (WORK068 / "units_v2.npy", OUT068 / "fit_receipt.json"),
}
SOURCES = (
    "ecg_experiment/serial071.py",
    "scripts/experiments/run_serial071.py",
    "docs/experiment-071-serial.md",
    "ecg_experiment/normative_map062.py",
    "ecg_experiment/combination066.py",
    "ecg_experiment/combiner068.py",
    "ecg_experiment/locbench.py",
)
SMOKE_PAIRS = 12
_STATE: dict[str, Any] = {}


def folders(smoke: bool) -> tuple[Path, Path]:
    """Output folder and local work folder of the run."""
    out = SMOKE_OUT if smoke else OUT
    work = WORK / ("smoke" if smoke else "full")
    out.mkdir(parents=True, exist_ok=True)
    work.mkdir(parents=True, exist_ok=True)
    return out, work


def holdout_ids() -> set[int]:
    """ECGAD holdout records, used only to assert that none is read."""
    split = load_split()
    return set(split.ecg_id[split.partition != "development"].astype(int))


def all_pairs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Record flags, every change-pair set (``set`` column) and the normal-normal pairs."""
    flags = sr.record_flags(load_metadata(), load_statements(), load_partition())
    frames = []
    for name, (rule, direction) in SETS.items():
        frame = sr.change_pairs(flags, rule, direction)
        frame.insert(0, "set", name)
        frames.append(frame)
    pairs = pd.concat(frames, ignore_index=True)
    pairs["ecgdeli"] = [Path(str(fiducial_path(e)) + ".atr").exists() for e in pairs.ecg_id]
    nulls = sr.null_pairs(flags)
    nulls["ecgdeli"] = [Path(str(fiducial_path(e)) + ".atr").exists() for e in nulls.ecg_id]
    return flags, pairs, nulls


def families_of(new: tuple[str, ...]) -> list[str]:
    """LocBench families of a statement set, sorted."""
    return sorted({FAMILIES[s] for s in new})


def coverage() -> dict[str, set[int]]:
    """Record IDs each saved comparator map covers."""
    scored = set(json.loads((WORK068 / "scored_ids.json").read_text()))
    return {
        "normative_v1": set(np.load(NORMATIVE062 / "ecg_ids.npy").astype(int).tolist()),
        "mil_lse": set(np.load(MIL065 / "maps_index_evaluation.npy").astype(int).tolist()),
        "textbook": set(json.loads((WORK066 / "scored_ids.json").read_text())),
        "units_v2": scored,
    }


def set_counts(rows: pd.DataFrame, cover: dict[str, set[int]]) -> dict[str, Any]:
    """Count one pair set in one partition."""
    fam: dict[str, int] = {}
    for new in rows.new:
        for family in families_of(new):
            fam[family] = fam.get(family, 0) + 1
    statements = pd.Series([s for x in rows.new for s in x], dtype=object)
    return {
        "pairs": len(rows),
        "patients": int(rows.patient_id.nunique()),
        "forward": int(rows.forward.sum()),
        "with_ecgdeli": int(rows.ecgdeli.sum()) if "ecgdeli" in rows else None,
        "multi_statement": int((rows.new.apply(len) > 1).sum()),
        "families": dict(sorted(fam.items())),
        "statements": statements.value_counts().sort_index().to_dict(),
        "gap_days_median": float(rows.gap_days.median()) if len(rows) else None,
        "gap_days_quartiles": rows.gap_days.quantile([0.25, 0.75]).tolist() if len(rows) else None,
        "coverage": {k: int(rows.ecg_id.isin(v).sum()) for k, v in cover.items()},
    }


def count_phase(smoke: bool) -> None:
    """Pair sets and counts from metadata only (no waveform, no score)."""
    out, _ = folders(smoke)
    flags, pairs, nulls = all_pairs()
    cover = coverage()
    multi = flags.groupby("patient_id").size()
    norm = flags.codes.apply(lambda c: "NORM" in c)
    mixed = flags.assign(norm=norm).groupby("patient_id").norm.agg(lambda x: bool(x.any() and not x.all()))
    counts: dict[str, Any] = {
        "records_folds_1_8": len(flags),
        "patients_folds_1_8": int(flags.patient_id.nunique()),
        "patients_with_several_ecgs": int((multi > 1).sum()),
        "ecgs_of_those_patients": int(multi[multi > 1].sum()),
        "patients_with_norm_and_non_norm_ecg": int(mixed.sum()),
        "normal_records": int(flags.normal.sum()),
        "finding_free_records": int(flags.finding_free.sum()),
        "sets": {
            name: {
                part: set_counts(pairs[(pairs.set == name) & (pairs.partition == part)], cover)
                for part in ("fit", "evaluation")
            }
            for name in SETS
        },
    }
    train = sr.change_pairs(flags, *SECONDARY_TRAIN)
    counts["secondary_training_pairs"] = set_counts(train[train.partition == "fit"], cover)
    counts["null_pairs"] = {
        part: {
            "pairs": int((nulls.partition == part).sum()),
            "patients": int(nulls.patient_id[nulls.partition == part].nunique()),
            "with_ecgdeli": int((nulls.ecgdeli & (nulls.partition == part)).sum()),
            "gap_days_median": float(nulls.gap_days[nulls.partition == part].median()),
        }
        for part in ("fit", "evaluation")
    }
    counts["cross_pool_fit_normals"] = int((flags.normal & (flags.partition == "fit")).sum())
    pairs.assign(new=pairs.new.str.join(";")).to_csv(out / "pairs.csv", index=False)
    nulls.to_csv(out / "null_pairs.csv", index=False)
    write_json_atomic(out / "counts.json", counts)
    print(json.dumps({k: v for k, v in counts.items() if k != "sets"}, indent=1, default=str), flush=True)


def _init_cache() -> None:
    threadpool_limits(limits=1)
    signals, index = open_cache()
    _STATE["signals"] = signals
    _STATE["row"] = dict(zip(index.ecg_id.astype(int), range(len(index)), strict=True))


def _beats(ecg_id: int) -> tuple[int, sr.Beats | None]:
    return ecg_id, sr.beat_summary(np.asarray(_STATE["signals"][_STATE["row"][ecg_id]], np.float64))


def needed_records(smoke: bool) -> list[int]:
    """Every record whose beat summary a later phase reads."""
    flags, pairs, nulls = all_pairs()
    if smoke:
        pairs = pairs[pairs.partition == "fit"].groupby("set").head(SMOKE_PAIRS)
        nulls = nulls[nulls.partition == "fit"].head(4 * SMOKE_PAIRS)
    train = sr.change_pairs(flags, *SECONDARY_TRAIN)
    train = train[train.partition == "fit"]
    if smoke:
        train = train.head(3 * SMOKE_PAIRS)
    ids = set(pairs.ecg_id) | set(pairs.partner) | set(nulls.ecg_id) | set(nulls.partner)
    ids |= set(train.ecg_id) | set(train.partner)
    ids |= set(flags.ecg_id[flags.normal & (flags.partition == "fit")])
    if ids & holdout_ids():
        raise ValueError("An ECGAD holdout record would be read")
    return sorted(int(e) for e in ids)


def features_phase(smoke: bool) -> None:
    """Beat summaries of every needed record."""
    out, work = folders(smoke)
    began = time.time()
    ids = needed_records(smoke)
    with Pool(WORKERS, initializer=_init_cache) as pool:
        beats = dict(pool.imap_unordered(_beats, ids, chunksize=16))
    with (work / "beats.pkl").open("wb") as handle:
        pickle.dump(beats, handle, protocol=pickle.HIGHEST_PROTOCOL)
    receipt = {
        "records": len(beats),
        "without_beats": sorted(e for e, b in beats.items() if b is None),
        "seconds": time.time() - began,
    }
    write_json_atomic(out / "features_receipt.json", receipt)
    print(json.dumps(receipt), flush=True)


def load_beats(work: Path) -> dict[int, sr.Beats | None]:
    """Load the saved beat summaries."""
    with (work / "beats.pkl").open("rb") as handle:
        return pickle.load(handle)  # noqa: S301


def cross_pool(flags: pd.DataFrame, beats: dict[int, sr.Beats | None]) -> pd.DataFrame:
    """Fit-partition normal ECGs with beats, their sex, age and heart rate."""
    pool = flags[flags.normal & (flags.partition == "fit")]
    pool = pool[[beats.get(int(e)) is not None for e in pool.ecg_id]].copy()
    pool["heart_rate"] = [beats[int(e)].intervals[0] for e in pool.ecg_id]
    return pool[["ecg_id", "patient_id", "sex", "age", "heart_rate"]].reset_index(drop=True)


def cross_partner(ecg_id: int, flags: pd.DataFrame, pool: pd.DataFrame, beats: dict) -> int:
    """Match a fit normal of another patient to one record."""
    row = flags.set_index("ecg_id").loc[ecg_id]
    target = pd.Series(
        {
            "patient_id": row.patient_id,
            "sex": row.sex,
            "age": row.age,
            "heart_rate": beats[ecg_id].intervals[0],
        }
    )
    return sr.match_partner(target, pool)


def differences(rows: pd.DataFrame, partners: list[int], beats: dict) -> tuple[np.ndarray, np.ndarray, int]:
    """Median-beat and interval differences of pairs whose two records both have beats."""
    grids, intervals, skipped = [], [], 0
    for ecg_id, partner in zip(rows.ecg_id, partners, strict=True):
        late, early = beats.get(int(ecg_id)), beats.get(int(partner))
        if late is None or early is None:
            skipped += 1
            continue
        grids.append(late.median - early.median)
        intervals.append(sr.interval_change(late.intervals, early.intervals))
    return np.array(grids), np.array(intervals), skipped


def model_summary(model: sr.NullModel) -> dict[str, Any]:
    """Spread quantiles per segment and the interval model, for the receipt."""
    segments = {
        name: np.quantile(model.spread[:, k == nm.SEGMENT_OF_BIN], [0.1, 0.5, 0.9]).tolist()
        for k, name in enumerate(nm.SEGMENTS)
    }
    return {
        "pairs": model.pairs,
        "floor_mV": model.floor,
        "floored_share": float((model.spread <= model.floor).mean()),
        "spread_quantiles_by_segment_mV": segments,
        "interval_centre_s": dict(zip(sr.INTERVAL_SEGMENTS, model.interval_centre.tolist(), strict=True)),
        "interval_spread_s": dict(zip(sr.INTERVAL_SEGMENTS, model.interval_spread.tolist(), strict=True)),
    }


def fit_phase(smoke: bool) -> None:
    """Fit the same-patient and cross-patient reference-difference models on fit pairs only."""
    out, work = folders(smoke)
    flags, _, nulls = all_pairs()
    beats = load_beats(work)
    fit = nulls[nulls.partition == "fit"]
    if smoke:
        fit = fit.head(4 * SMOKE_PAIRS)
    evaluation = set(flags.ecg_id[flags.partition == "evaluation"])
    if evaluation & (set(fit.ecg_id) | set(fit.partner)):
        raise ValueError("An evaluation record would enter the fit")
    grids, intervals, skipped = differences(fit, fit.partner.tolist(), beats)
    same = sr.fit_null(grids, intervals)
    pool = cross_pool(flags, beats)
    usable = fit[[beats.get(int(e)) is not None for e in fit.ecg_id]]
    matches = [cross_partner(int(e), flags, pool, beats) for e in usable.ecg_id]
    cgrids, cintervals, cskipped = differences(usable, matches, beats)
    cross = sr.fit_null(cgrids, cintervals)
    with (work / "null_models.pkl").open("wb") as handle:
        pickle.dump({"same": same, "cross": cross}, handle, protocol=pickle.HIGHEST_PROTOCOL)
    summary = {
        "same_patient": {**model_summary(same), "skipped_without_beats": skipped},
        "cross_patient": {**model_summary(cross), "skipped_without_beats": cskipped},
        "cross_pool_records": len(pool),
        "cross_fit_matches_distinct": len(set(matches)),
    }
    write_json_atomic(out / "fit_summary.json", summary)
    print(json.dumps(summary, indent=1), flush=True)


def load_models(work: Path) -> dict[str, sr.NullModel]:
    """Load the saved reference-difference models."""
    with (work / "null_models.pkl").open("rb") as handle:
        return pickle.load(handle)  # noqa: S301


def comparator_sources() -> dict[str, Any]:
    """Memory maps and row lookups of the saved comparator maps; 062 model for records it lacks."""
    sources = {}
    ids062 = np.load(NORMATIVE062 / "ecg_ids.npy").astype(int).tolist()
    sources["normative_v1"] = (np.load(NORMATIVE062 / "maps_normative_v1.npy", mmap_mode="r"), ids062)
    ids065 = np.load(MIL065 / "maps_index_evaluation.npy").astype(int).tolist()
    sources["mil_lse"] = (np.load(MIL065 / "maps_mil_lse_evaluation.npy", mmap_mode="r"), ids065)
    ids066 = json.loads((WORK066 / "scored_ids.json").read_text())
    sources["textbook"] = (np.load(WORK066 / "textbook.npy", mmap_mode="r"), ids066)
    ids068 = json.loads((WORK068 / "scored_ids.json").read_text())
    sources["units_v2"] = (np.load(WORK068 / "units_v2.npy", mmap_mode="r"), ids068)
    lookups = {name: (maps, {e: k for k, e in enumerate(ids)}) for name, (maps, ids) in sources.items()}
    with (NORMATIVE062 / "normative.pkl").open("rb") as handle:
        lookups["normative_model"] = pickle.load(handle)  # noqa: S301
    return lookups


def comparator_map(
    name: str, ecg_id: int, signal: np.ndarray, meta: pd.Series, sources: dict
) -> np.ndarray | None:
    """One comparator map of one record; None when it is not saved and cannot be rebuilt."""
    if name == "qrs_prior":
        return qrs_prior_map(exact_signal(signal))[0].astype(np.float32)
    maps, row = sources[name]
    if ecg_id in row:
        return np.asarray(maps[row[ecg_id]], np.float32)
    if name == "normative_v1":
        built, _ = nm.record_maps(
            np.asarray(signal, np.float64), float(meta.age), float(meta.sex), sources["normative_model"]
        )
        return built[nm.PRIMARY].astype(np.float16).astype(np.float32)
    return None


def record_maps(
    ecg_id: int,
    partner: int,
    cross: int | None,
    beats: dict,
    models: dict,
    comparators: bool,
    signal: np.ndarray,
    meta: pd.Series,
    sources: dict | None,
) -> dict[str, np.ndarray | None]:
    """Build the serial, cross-patient and comparator maps of one painted record."""
    late, early = beats.get(ecg_id), beats.get(partner)
    maps: dict[str, np.ndarray | None] = dict.fromkeys(MAP_NAMES)
    if late is not None and early is not None:
        maps["serial"] = sr.paint(sr.change_grid(late, early, models["same"]), late.knots)
        maps["serial_interval"] = sr.paint(
            sr.change_grid(late, early, models["same"], intervals=True), late.knots
        )
    if late is not None and cross is not None and beats.get(cross) is not None:
        maps["cross"] = sr.paint(sr.change_grid(late, beats[cross], models["cross"]), late.knots)
    if comparators:
        for name in COMPARATORS:
            maps[name] = comparator_map(name, ecg_id, signal, meta, sources)
    return maps


def noise_leads(flags: pd.DataFrame, a: int, b: int) -> np.ndarray:
    """Union of the noisy leads named for two records."""
    table = flags.set_index("ecg_id").noisy_leads
    named = {lead for e in (a, b) for lead in str(table.loc[e]).split(";") if lead}
    return np.isin(LEADS, sorted(named))


def describe(values: np.ndarray, patients: np.ndarray) -> dict[str, Any]:
    """Mean with a whole-patient bootstrap interval, or a count when nothing is defined."""
    values = np.asarray(values, float)
    if not np.isfinite(values).any():
        return {"value": None, "records": [0]}
    return sr.bootstrap_means([(values, patients)], lambda m: m[0])


def score_context(smoke: bool, develop: bool) -> dict[str, Any]:
    """Pairs to read, beat summaries, models, cache and comparator sources of the score phase."""
    _, work = folders(smoke)
    part = "fit" if smoke or develop else "evaluation"
    flags, pairs, nulls = all_pairs()
    beats = load_beats(work)
    pairs = pairs[(pairs.partition == part) & pairs.ecgdeli]
    nulls = nulls[(nulls.partition == part) & nulls.ecgdeli]
    if smoke:
        pairs = pairs.groupby("set").head(SMOKE_PAIRS)
        nulls = nulls.head(4 * SMOKE_PAIRS)
    if (set(pairs.ecg_id) | set(nulls.ecg_id)) & holdout_ids():
        raise ValueError("An ECGAD holdout record would be read")
    signals, index = open_cache()
    return {
        "part": part,
        "flags": flags,
        "pairs": pairs.reset_index(drop=True),
        "nulls": nulls.reset_index(drop=True),
        "beats": beats,
        "models": load_models(work),
        "pool": cross_pool(flags, beats),
        "signals": signals,
        "row": dict(zip(index.ecg_id.astype(int), range(len(index)), strict=True)),
        "meta": index.set_index("ecg_id"),
        "sources": comparator_sources(),
    }


def maps_of(
    ctx: dict[str, Any], ecg_id: int, partner: int, comparators: bool
) -> dict[str, np.ndarray | None]:
    """Every map of one painted record in the score phase."""
    beats = ctx["beats"]
    cross = cross_partner(ecg_id, ctx["flags"], ctx["pool"], beats) if beats.get(ecg_id) is not None else None
    signal = np.asarray(ctx["signals"][ctx["row"][ecg_id]])
    sources = ctx["sources"] if comparators else None
    return record_maps(
        ecg_id, partner, cross, beats, ctx["models"], comparators, signal, ctx["meta"].loc[ecg_id], sources
    )


def pair_readings(ctx: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Region readings of every map on every change pair, primary-set unit scores, and the painted maps."""
    rows, unit_rows, painted, masks_of = [], [], {}, {}
    for record in ctx["pairs"].itertuples(index=False):
        e, p = int(record.ecg_id), int(record.partner)
        if e not in masks_of:
            masks_of[e] = sr.unit_masks(read_components(e))
        if (e, p) not in painted:
            painted[(e, p)] = maps_of(ctx, e, p, True)
        maps = painted[(e, p)]
        base = {
            "set": record.set,
            "ecg_id": e,
            "partner": p,
            "patient_id": int(record.patient_id),
            "new": ";".join(record.new),
            "families": ";".join(families_of(record.new)),
            "forward": bool(record.forward),
            "gap_days": float(record.gap_days),
        }
        for name, field in maps.items():
            values = (
                sr.region_readings(field, record.new, masks_of[e])
                if field is not None
                else dict.fromkeys(READINGS, np.nan)
            )
            rows.append({**base, "method": name, **values})
        if record.set == PRIMARY_SET and maps["serial"] is not None:
            scores, target = sr.unit_scores(maps["serial"], masks_of[e]), sr.target_units(record.new)
            unit_rows += [
                {
                    "ecg_id": e,
                    "lead": lead,
                    "component": component,
                    "score": scores[k, c],
                    "target": bool(target[k, c]),
                }
                for k, lead in enumerate(LEADS)
                for c, component in enumerate(sr.UNIT_COMPONENTS)
            ]
    return pd.DataFrame(rows), pd.DataFrame(unit_rows), painted


def compositions_of(table: pd.DataFrame) -> dict[str, dict[tuple[str, ...], int]]:
    """Statement-set composition of each change-pair set, for the pseudo-statement null."""
    serial = table[table.method == "serial"]
    return {
        name: pd.Series([tuple(n.split(";")) for n in serial.new[serial.set == name]], dtype=object)
        .value_counts()
        .to_dict()
        for name in SETS
    }


def null_readings(ctx: dict[str, Any], compositions: dict, painted: dict) -> pd.DataFrame:
    """Pseudo-statement readings of the serial, wave-only and cross maps on normal-normal pairs."""
    rows = []
    for record in ctx["nulls"].itertuples(index=False):
        e, p = int(record.ecg_id), int(record.partner)
        masks = sr.unit_masks(read_components(e))
        maps = maps_of(ctx, e, p, False)
        entry = {
            "ecg_id": e,
            "partner": p,
            "patient_id": int(record.patient_id),
            "gap_days": float(record.gap_days),
        }
        for name in SERIAL_MAPS:
            for set_name, composition in compositions.items():
                defined = maps[name] is not None and composition
                entry[f"{name}_{set_name}"] = (
                    sr.null_reading(maps[name], masks, composition) if defined else np.nan
                )
        if maps["serial"] is not None:
            scores = sr.unit_scores(maps["serial"], masks)
            for code in sr.REGION_TABLE:
                entry[f"statement_{code}"] = sr.unit_auroc(scores, sr.target_units((code,)))
        rows.append(entry)
        painted[(e, p)] = {name: maps[name] for name in ("serial", "cross")}
    return pd.DataFrame(rows)


def score_phase(smoke: bool, develop: bool = False) -> None:
    """
    Change maps and every declared reading on the evaluation pairs.

    The smoke run reads a few fit pairs; ``develop`` reads every fit pair (design data, in-sample for the
    reference-difference model) into ``develop_fit/``.
    """
    out, _ = folders(smoke)
    if develop:
        out = out / "develop_fit"
        out.mkdir(exist_ok=True)
    began = time.time()
    if not smoke and not develop:
        integrity = {"bar": reproduce_bar(), "saved_maps": saved_hashes()}
        write_json_atomic(out / "integrity_primary.json", integrity)
    ctx = score_context(smoke, develop)
    table, units, painted = pair_readings(ctx)
    compositions = compositions_of(table)
    nulls = null_readings(ctx, compositions, painted)
    noise = noise_readings(ctx["flags"], ctx["pairs"], ctx["nulls"], painted)
    tables = out / "tables"
    tables.mkdir(exist_ok=True)
    table.to_csv(tables / "pair_readings.csv", index=False)
    nulls.to_csv(tables / "null_readings.csv", index=False)
    noise.to_csv(tables / "noise_readings.csv", index=False)
    units.to_csv(tables / "forward_serial_units.csv", index=False)
    readings = summarize_primary(table, nulls, noise, compositions)
    readings["partition_read"] = ctx["part"]
    readings["seconds"] = time.time() - began
    write_json_atomic(out / "result_primary.json", readings)
    print(json.dumps(readings["decisions"], indent=1), flush=True)


def noise_readings(
    flags: pd.DataFrame, pairs: pd.DataFrame, nulls: pd.DataFrame, painted: dict
) -> pd.DataFrame:
    """Noisy-lead AUROC of the serial and cross maps, raw and standardized by quiet null pairs."""
    quiet = [
        (int(e), int(p))
        for e, p in zip(nulls.ecg_id, nulls.partner, strict=True)
        if not noise_leads(flags, int(e), int(p)).any()
    ]
    reference = {}
    for name in ("serial", "cross"):
        scores = np.array(
            [
                np.percentile(painted[k][name][:, sr.START : sr.STOP], 95, axis=1)
                for k in quiet
                if painted[k][name] is not None
            ]
        )
        reference[name] = (np.median(scores, axis=0), np.subtract(*np.percentile(scores, [75, 25], axis=0)))
    keys = sorted(
        set(zip(pairs.ecg_id.astype(int), pairs.partner.astype(int), strict=True))
        | set(zip(nulls.ecg_id.astype(int), nulls.partner.astype(int), strict=True))
    )
    patient = flags.set_index("ecg_id").patient_id
    rows = []
    for e, p in keys:
        named = noise_leads(flags, e, p)
        if not 1 <= named.sum() <= len(LEADS) - 1:
            continue
        entry = {
            "ecg_id": e,
            "partner": p,
            "patient_id": int(patient.loc[e]),
            "named": int(named.sum()),
            "kind": "null" if (e, p) in set(zip(nulls.ecg_id, nulls.partner, strict=True)) else "change",
        }
        for name in ("serial", "cross"):
            field = painted[(e, p)][name]
            entry[f"{name}_C1"] = sr.lead_auroc(field, named) if field is not None else np.nan
            entry[f"{name}_C1_std"] = (
                sr.lead_auroc(field, named, *reference[name]) if field is not None else np.nan
            )
        rows.append(entry)
    return pd.DataFrame(rows)


def summarize_primary(
    table: pd.DataFrame, nulls: pd.DataFrame, noise: pd.DataFrame, compositions: dict
) -> dict[str, Any]:
    """Means, intervals, paired differences, families, the null and noise checks, and the decisions."""
    result: dict[str, Any] = {
        "compositions": {k: {";".join(s): n for s, n in v.items()} for k, v in compositions.items()}
    }
    wide = {
        reading: table.pivot_table(
            index=["set", "ecg_id", "patient_id"], columns="method", values=reading, dropna=False
        ).reset_index()
        for reading in READINGS
    }
    result["means"], result["paired"] = {}, {}
    for set_name in SETS:
        result["means"][set_name], result["paired"][set_name] = {}, {}
        for reading in READINGS:
            frame = wide[reading][wide[reading].set == set_name]
            if frame.empty:
                continue
            patients = frame.patient_id.to_numpy()
            result["means"][set_name][reading] = {
                name: describe(frame[name].to_numpy(), patients) for name in MAP_NAMES if name in frame
            }
            result["paired"][set_name][reading] = {
                f"serial_minus_{other}": sr.paired_mean_difference(frame["serial"], frame[other], patients)
                for other in MAP_NAMES[1:]
                if other in frame and np.isfinite(frame[other]).any()
            }
    result["null"] = {}
    for set_name in SETS:
        entry = {}
        for name in SERIAL_MAPS:
            column = f"{name}_{set_name}"
            if column in nulls and np.isfinite(nulls[column]).any():
                entry[name] = describe(nulls[column].to_numpy(), nulls.patient_id.to_numpy())
        b1u = wide["B1u"][wide["B1u"].set == set_name]
        for name in SERIAL_MAPS:
            column = f"{name}_{set_name}"
            if b1u.empty or column not in nulls or not np.isfinite(nulls[column]).any():
                continue
            entry[f"{name}_change_minus_null"] = sr.bootstrap_means(
                [
                    (b1u[name].to_numpy(), b1u.patient_id.to_numpy()),
                    (nulls[column].to_numpy(), nulls.patient_id.to_numpy()),
                ],
                lambda m: m[0] - m[1],
            )
        result["null"][set_name] = entry
    statements = [c for c in nulls.columns if c.startswith("statement_")]
    result["null_by_statement"] = {
        c.removeprefix("statement_"): describe(nulls[c].to_numpy(), nulls.patient_id.to_numpy())
        for c in statements
    }
    result["families"] = family_means(table)
    result["macro"] = macro_readings(table)
    result["gap"] = gap_means(table)
    result["noise"] = {
        f"{kind}_{column}": describe(noise[column].to_numpy(), noise.patient_id.to_numpy())
        for kind in ("all",)
        for column in ("serial_C1", "serial_C1_std", "cross_C1", "cross_C1_std")
        if column in noise and len(noise)
    }
    result["noise"]["records"] = len(noise)
    result["noise"]["records_null"] = int((noise.kind == "null").sum()) if len(noise) else 0
    result["decisions"] = decisions(result)
    return result


def macro_readings(table: pd.DataFrame) -> dict[str, Any]:
    """Family macro B1u (families with at least three records) of the serial maps and 062, and differences."""
    out: dict[str, Any] = {}
    for set_name, frame in table.groupby("set"):
        wide = frame.pivot_table(
            index=["ecg_id", "patient_id", "families"], columns="method", values="B1u", dropna=False
        ).reset_index()
        families = [f.split(";") for f in wide.families]
        patients = wide.patient_id.to_numpy()
        entry = {
            name: sr.family_macro(wide[name].to_numpy(), families, patients)
            for name in ("serial", "cross", "normative_v1")
            if name in wide
        }
        keep = np.isfinite(wide.serial) & np.isfinite(wide.normative_v1)
        entry["serial_minus_normative_v1"] = sr.family_macro(
            np.where(keep, wide.serial - wide.normative_v1, np.nan), families, patients
        )
        out[set_name] = entry
    return out


def family_means(table: pd.DataFrame) -> dict[str, Any]:
    """Mean B1u per family and map (a record counts in each family of its new statements)."""
    out: dict[str, Any] = {}
    exploded = table.assign(family=table.families.str.split(";")).explode("family")
    for (set_name, family), frame in exploded.groupby(["set", "family"]):
        wide = frame.pivot_table(index="ecg_id", columns="method", values="B1u", dropna=False)
        out.setdefault(set_name, {})[family] = {
            "records": len(wide),
            **{
                name: (float(wide[name].mean()) if name in wide and np.isfinite(wide[name]).any() else None)
                for name in MAP_NAMES
            },
        }
    return out


def gap_means(table: pd.DataFrame) -> dict[str, Any]:
    """Mean serial and normative B1u by time gap (at most 30 days against longer)."""
    out: dict[str, Any] = {}
    for set_name, frame in table.groupby("set"):
        for label, keep in (("gap_30d_or_less", frame.gap_days <= 30), ("gap_over_30d", frame.gap_days > 30)):
            rows = frame[keep]
            wide = rows.pivot_table(index="ecg_id", columns="method", values="B1u", dropna=False)
            out.setdefault(set_name, {})[label] = {
                "records": len(wide),
                **{
                    name: (float(wide[name].mean()) if name in wide else None)
                    for name in ("serial", "normative_v1")
                },
            }
    return out


def verdict(interval: dict[str, Any]) -> str:
    """Decision words from an interval of a difference."""
    if interval["ci_low"] > 0:
        return "above"
    if interval["ci_high"] < 0:
        return "below"
    return "not distinguishable"


def decisions(result: dict[str, Any]) -> dict[str, Any]:
    """Frozen decision rule of the protocol."""
    null = result["null"][PRIMARY_SET]["serial_change_minus_null"]
    paired = result["paired"][PRIMARY_SET]["B1u"]
    return {
        "primary_set": PRIMARY_SET,
        "serial_B1u": result["means"][PRIMARY_SET]["B1u"]["serial"],
        "null_B1u": result["null"][PRIMARY_SET]["serial"],
        "change_minus_null": {**null, "decision": verdict(null)},
        "serial_minus_normative_v1": {
            **paired["serial_minus_normative_v1"],
            "decision": verdict(paired["serial_minus_normative_v1"]),
        },
        "serial_minus_cross": {
            **paired["serial_minus_cross"],
            "decision": verdict(paired["serial_minus_cross"]),
        },
        "null_flag": not 0.45 <= result["null"][PRIMARY_SET]["serial"]["value"] <= 0.55,
    }


def _paint_init(work: str) -> None:
    _init_cache()
    _STATE["out"] = np.lib.format.open_memmap(Path(work) / f"{SECONDARY}.npy", mode="r+")


def _paint(task: tuple[int, int, np.ndarray, np.ndarray]) -> int:
    position, ecg_id, units, scores = task
    amplitude = amplitude_map(exact_signal(_STATE["signals"][_STATE["row"][ecg_id]]))
    painted = cb.units_map(scores, units, amplitude)
    if not np.isfinite(painted).all():
        raise ValueError(f"Nonfinite map for {ecg_id}")
    _STATE["out"][position] = painted
    return position


def load_068(kind: str) -> dict[int, dict]:
    """068 unit features of the training (fit) or scored records."""
    with (WORK068 / f"features_{kind}.pkl").open("rb") as handle:
        return pickle.load(handle)  # noqa: S301


def secondary_design(smoke: bool, work: Path) -> dict[str, Any]:
    """Stacked units_v2 design of the fit change pairs with change-derived unit labels."""
    flags = sr.record_flags(load_metadata(), load_statements(), load_partition())
    beats, models = load_beats(work), load_models(work)
    train = sr.change_pairs(flags, *SECONDARY_TRAIN)
    train = train[train.partition == "fit"].reset_index(drop=True)
    if smoke:
        train = train.head(3 * SMOKE_PAIRS)
    evaluation = set(flags.ecg_id[flags.partition == "evaluation"]) | set(load_split().ecg_id.astype(int))
    if evaluation & (set(train.ecg_id) | set(train.partner)):
        raise ValueError("An evaluation or ECGAD record would enter the secondary fit")
    features = load_068("train")
    cols = cv.columns(cv.feature_set("units_v2", cv.SOFT_GRID[0]))
    designs, labels, ids, patients, dropped = [], [], [], [], []
    for record in train.itertuples(index=False):
        e, p = int(record.ecg_id), int(record.partner)
        r = features.get(e)
        late, early = beats.get(e), beats.get(p)
        if r is None or late is None or early is None or not len(r["units"]):
            dropped.append(e)
            continue
        field = sr.paint(sr.change_grid(late, early, models["same"]), late.knots)
        labels.append(sr.change_unit_labels(cb.unit_means(r["units"], field)))
        designs.append(cv.unit_design(r["unit_raw"], r["q"], r["units"])[:, cols])
        ids.append(e)
        patients.append(int(record.patient_id))
    x, y, starts = cv.stack(designs, labels)
    folds = cb.patient_folds(np.array(patients), cv.FOLDS, sr.SEED)
    return {
        "x": x,
        "y": y,
        "starts": starts,
        "folds": folds,
        "ecg_ids": ids,
        "patients": patients,
        "dropped": dropped,
        "cols": cols,
    }


def secondary_phase(smoke: bool) -> None:
    """Fit the change-supervised unit model on fit pairs and paint it on every LocBench scored record."""
    out, work = folders(smoke)
    began = time.time()
    data = secondary_design(smoke, work)
    x, y, starts, folds = data["x"], data["y"], data["starts"], data["folds"]
    names = cv.feature_set("units_v2", cv.SOFT_GRID[0])
    weights_068 = json.loads((OUT068 / "fit_receipt.json").read_text())["models"]["units_v2"]["weights"]
    w068 = np.array([weights_068[n] for n in names])
    grid = []
    for l2 in cv.L2_GRID:
        loss = auc = auc068 = 0.0
        for fold in range(cv.FOLDS):
            train, test = np.flatnonzero(folds != fold), np.flatnonzero(folds == fold)
            w = cv.fit_listwise(*cv.subset(x, y, starts, train), l2)
            xt, yt, st = cv.subset(x, y, starts, test)
            loss += cv.listwise_loss(w, xt, yt, st, 0.0)[0] * len(test)
            auc += cv.within_record_auroc(xt @ w, yt, st) * len(test)
            auc068 += cv.within_record_auroc(xt @ w068, yt, st) * len(test)
        n = len(folds)
        grid.append({"l2": l2, "loss": loss / n, "auroc": auc / n, "auroc_units_v2_weights": auc068 / n})
        print(grid[-1], flush=True)
    best = min(grid, key=lambda r: (r["loss"], -r["l2"]))
    w = cv.fit_listwise(x, y, starts, best["l2"])
    scored = scored_ecg_ids()
    if smoke:
        scored = scored[:16]
    if set(scored.astype(int)) & holdout_ids():
        raise ValueError("An ECGAD holdout record would be mapped")
    features = load_068("scored")
    np.lib.format.open_memmap(
        work / f"{SECONDARY}.npy", mode="w+", dtype=np.float32, shape=(len(scored), 12, 5000)
    )
    tasks = []
    for k, e in enumerate(scored.astype(int).tolist()):
        r = features[e]
        design = (
            cv.unit_design(r["unit_raw"], r["q"], r["units"])[:, data["cols"]]
            if len(r["units"])
            else np.zeros((0, len(w)))
        )
        tasks.append((k, e, r["units"], design @ w))
    with Pool(WORKERS, initializer=_paint_init, initargs=(str(work),)) as pool:
        list(pool.imap_unordered(_paint, tasks, chunksize=16))
    (work / "scored_ids.json").write_text(json.dumps([int(e) for e in scored]))
    receipt = {
        "created": utc_now(),
        "git_head": git_head(ROOT),
        "training_records": len(data["ecg_ids"]),
        "training_patients": len(set(data["patients"])),
        "training_dropped": data["dropped"],
        "training_units": int(len(y)),
        "training_positive_units": int(y.sum()),
        "features": list(names),
        "cv_grid": grid,
        "l2": best["l2"],
        "weights": dict(zip(names, w.tolist(), strict=True)),
        "units_v2_weights": weights_068,
        "maps_file": str(work / f"{SECONDARY}.npy"),
        "maps_sha256": sha256_file(work / f"{SECONDARY}.npy"),
        "seconds": time.time() - began,
    }
    write_json_atomic(out / "secondary_receipt.json", receipt)
    print(
        json.dumps({k: v for k, v in receipt.items() if k not in ("weights", "units_v2_weights")}, indent=1),
        flush=True,
    )


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
    return {
        "per_record_max_abs_difference": largest,
        "records": int((~np.isnan(values)).sum()),
        "A2": float(np.nanmean(values)),
        "pass": bool(np.array_equal(np.isnan(values), np.isnan(saved)) and largest <= TOLERANCE),
    }


def reproduce_bar() -> dict[str, Any]:
    """061 ``qrs_prior`` ECGAD development A2, per record within 1e-12 and as a mean."""
    result = rescore_tables(REFERENCES["qrs_prior"])
    target = json.loads((LOCBENCH061 / "result.json").read_text())["bar"]["A2"]["value"]
    result["A2_saved"] = target
    result["pass"] = bool(result["pass"] and abs(result["A2"] - target) <= TOLERANCE)
    if not result["pass"]:
        raise ValueError(f"Bar not reproduced: {result}")
    return result


def stored(path: Path) -> str:
    """Repository-relative path for receipts, or the absolute path of a local work file."""
    return to_stored(path) if str(path).startswith(str(ROOT)) else str(path)


def saved_hashes() -> dict[str, Any]:
    """SHA-256 of every saved comparator map file against its producing receipt."""
    result = {}
    for name, (path, receipt) in SAVED_HASHES.items():
        text = receipt.read_text()
        actual = sha256_file(path)
        result[name] = {"file": stored(path), "sha256": actual, "in_receipt": actual in text}
    if not all(v["in_receipt"] for v in result.values()):
        raise ValueError(f"A comparator map differs from its receipt: {result}")
    return result


def units_v2_rebuild(records: int = 24) -> dict[str, Any]:
    """Repaint 068 ``units_v2`` from its saved features and weights on the first scored records."""
    features = load_068("scored")
    receipt = json.loads((OUT068 / "fit_receipt.json").read_text())["models"]["units_v2"]
    names = tuple(receipt["features"])
    w = np.array([receipt["weights"][n] for n in names])
    ids = json.loads((WORK068 / "scored_ids.json").read_text())
    saved = np.load(WORK068 / "units_v2.npy", mmap_mode="r")
    signals, index = open_cache()
    row = dict(zip(index.ecg_id.astype(int), range(len(index)), strict=True))
    largest = 0.0
    for k, e in enumerate(ids[:records]):
        r = features[e]
        scores = cv.unit_design(r["unit_raw"], r["q"], r["units"])[:, cv.columns(names)] @ w
        painted = cb.units_map(scores, r["units"], amplitude_map(exact_signal(signals[row[e]])))
        largest = max(largest, float(np.abs(painted - saved[k]).max()))
    return {"records": records, "max_abs_difference": largest, "pass": bool(largest <= 1e-6)}


def copy_references(out: Path) -> dict[str, Path]:
    """Copy the reference tables into this run (the strata and noise CLIs write into them)."""
    copies = {}
    for name, source in REFERENCES.items():
        target = out / "locbench" / "ref" / name
        if not target.exists():
            shutil.copytree(source, target, ignore=shutil.ignore_patterns("strata.*", "noise_std.*"))
        copies[name] = target
    return copies


def locbench_phase() -> None:
    """Integrity checks, LocBench scoring of the secondary map with paired comparisons, strata and noise."""
    out, work = folders(False)
    began = time.time()
    scored = np.array(json.loads((work / "scored_ids.json").read_text()), np.int64)
    if not np.array_equal(scored, scored_ecg_ids()):
        raise ValueError("Scored records changed")
    integrity = {"bar": reproduce_bar(), "saved_maps": saved_hashes(), "units_v2_rebuild": units_v2_rebuild()}
    if not integrity["units_v2_rebuild"]["pass"]:
        raise ValueError(f"068 features do not rebuild units_v2: {integrity['units_v2_rebuild']}")
    write_json_atomic(out / "integrity.json", integrity)
    refs = copy_references(out)
    target = out / "locbench" / SECONDARY
    python = str(ROOT / ".venv/bin/python")
    command = [
        python,
        "-m",
        "scripts.experiments.run_locbench_score",
        "--maps",
        str(work / f"{SECONDARY}.npy"),
        "--ids",
        str(work / "scored_ids.json"),
        "--name",
        SECONDARY,
        "--out",
        str(target),
    ]
    for folder in refs.values():
        command += ["--compare", str(folder)]
    subprocess.run(command, check=True, cwd=ROOT)
    tables = [target / "tables", refs["units_v2"]]
    subprocess.run(
        [python, "-m", "scripts.experiments.run_locbench_strata", str(refs["qrs_prior"]), *map(str, tables)],
        check=True,
        cwd=ROOT,
    )
    subprocess.run(
        [python, "-m", "scripts.experiments.run_locbench_noise", *map(str, tables), str(refs["qrs_prior"])],
        check=True,
        cwd=ROOT,
    )
    write_json_atomic(out / "locbench_receipt.json", {"seconds": time.time() - began})


def report_phase() -> None:
    """Assemble result.json from the phase outputs."""
    out, _ = folders(False)
    summary = json.loads((out / "locbench" / SECONDARY / "summary.json").read_text())
    readings = {"summary": summary}
    for name in ("strata", "noise_std"):
        path = out / "locbench" / SECONDARY / "tables" / f"{name}.json"
        readings[name] = json.loads(path.read_text())
    for ref in REFERENCES:
        folder = out / "locbench" / "ref" / ref
        readings[f"ref_{ref}"] = {
            name: json.loads((folder / f"{name}.json").read_text())
            for name in ("strata", "noise_std")
            if (folder / f"{name}.json").exists()
        }
    result = {
        "experiment": "071",
        "protocol": "docs/experiment-071-serial.md",
        "protocol_sha256": sha256_file(ROOT / "docs/experiment-071-serial.md"),
        "created": utc_now(),
        "git_head": git_head(ROOT),
        "inputs": {
            "partition": to_stored(PARTITION_PATH),
            "partition_sha256": sha256_file(PARTITION_PATH),
            "cache_dir": str(CACHE_DIR),
            "features068_sha256": {
                k: sha256_file(WORK068 / f"features_{k}.pkl") for k in ("train", "scored")
            },
            "normative062_model_sha256": sha256_file(NORMATIVE062 / "normative.pkl"),
            "sources_sha256": {path: sha256_file(ROOT / path) for path in SOURCES},
        },
        "counts": json.loads((out / "counts.json").read_text()),
        "features": json.loads((out / "features_receipt.json").read_text()),
        "fit": json.loads((out / "fit_summary.json").read_text()),
        "primary": json.loads((out / "result_primary.json").read_text()),
        "integrity": {
            "primary": json.loads((out / "integrity_primary.json").read_text()),
            "locbench": json.loads((out / "integrity.json").read_text()),
        },
        "secondary": {
            "receipt": json.loads((out / "secondary_receipt.json").read_text()),
            "locbench": readings,
        },
    }
    write_json_atomic(out / "result.json", result)
    print(json.dumps(result["primary"]["decisions"], indent=1), flush=True)


def main() -> None:
    """Command-line entry point."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "phase", choices=["count", "features", "fit", "develop", "score", "secondary", "locbench", "report"]
    )
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    if args.phase == "count":
        count_phase(args.smoke)
    elif args.phase == "features":
        features_phase(args.smoke)
    elif args.phase == "fit":
        fit_phase(args.smoke)
    elif args.phase == "develop":
        score_phase(args.smoke, develop=True)
    elif args.phase == "score":
        score_phase(args.smoke)
    elif args.phase == "secondary":
        secondary_phase(args.smoke)
    elif args.smoke:
        raise ValueError("The smoke run is not scored on LocBench")
    elif args.phase == "locbench":
        locbench_phase()
    else:
        report_phase()


if __name__ == "__main__":
    main()
