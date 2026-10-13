"""
Experiment 069: measurement-based criteria localizer.

Phases::

    run_criteria069 measure    # own median-beat measurements (8 processes) and harmonized 12SL / Uni-G tables
    run_criteria069 maps       # consensus, normal reference, criteria evidence, scored maps, unit means
    run_criteria069 combiner   # 068 rebuild and the units_v2_criteria combiner (CV as 068), its maps
    run_criteria069 score      # integrity, LocBench, strata, standardized noise, specificity, result.json

``--smoke`` uses 16 scored, 200 clean-normal and 60 training records in a separate work folder; nothing is
scored. See ``docs/experiment-069-criteria-localizer.md``.
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
from ecg_experiment import criteria069 as cr
from ecg_experiment.ecgad import load_split
from ecg_experiment.eda.ptbxl import load_metadata
from ecg_experiment.files import sha256_file, write_json_atomic
from ecg_experiment.locbench import ecgad_record_metrics, load_benchmark, scored_ecg_ids
from ecg_experiment.locbench_baselines061 import amplitude_map, exact_signal
from ecg_experiment.locbench_cache import CACHE_DIR, open_cache
from ecg_experiment.locbench_data import PARTITION_PATH, REGION_TABLE, load_partition
from ecg_experiment.locbench_noise import standardized_noise_table
from ecg_experiment.locbench_strata import strata_paired
from ecg_experiment.paths import to_stored
from ecg_experiment.provenance import git_head, utc_now
from ecg_experiment.waveforms import LEADS

OUT = ROOT / "outputs/experiment069_criteria_v1"
WORK = Path(os.environ.get("ECG069_WORK", "/tmp/ecg069-work"))
WORK068 = Path(os.environ.get("ECG068_WORK", "/tmp/ecg068-work")) / "full"
WORK066 = Path(os.environ.get("ECG066_WORK", "/tmp/ecg066-work")) / "full"
FEATURES = ROOT / "data/raw/ptb-xl-plus/1.0.1/features"
MIL065 = ROOT / "outputs/experiment065_mil_v1"
OUT068 = ROOT / "outputs/experiment068_combiner_v2"
LOCBENCH061 = ROOT / "outputs/experiment061_locbench_v1"
BAR = "qrs_prior"
REFERENCES = {
    BAR: LOCBENCH061 / "tables" / BAR,
    "units_v2": OUT068 / "locbench/units_v2/tables",
    "textbook": ROOT / "outputs/experiment066_combination_v1/locbench/textbook/tables",
}
CRITERIA_MAPS = ("criteria", "criteria_agree", "criteria_x_mil", "criteria_own", "criteria_qrs")
MAPS = (*CRITERIA_MAPS, "units_v2_criteria")
PRIMARY = "criteria"
SANITY = {
    "1AVB": ("pr_long", None),
    "CLBBB": ("lbbb_lat", None),
    "CRBBB": ("rbbb_v1", None),
    "LAFB": ("lafb", None),
    "IMI": ("q_path", ("II", "aVF")),
    "ASMI": ("q_v2v3", None),
    "LVH": (("sokolow", "r_avl", "cornell"), None),
    "LNGQT": ("qtc_long", None),
}
WORKERS = 8
SMOKE = {"scored": 16, "normals": 200, "train": 60}
TOLERANCE = 1e-12
BAR_A2 = 0.7419383358406554
SOURCES = (
    "ecg_experiment/criteria069.py",
    "scripts/experiments/run_criteria069.py",
    "docs/experiment-069-criteria-localizer.md",
    "ecg_experiment/combination066.py",
    "ecg_experiment/combiner068.py",
    "ecg_experiment/normative_map062.py",
    "ecg_experiment/locbench.py",
)

_STATE: dict[str, Any] = {}


def finite_json(value: Any) -> Any:
    """Replace NaN and infinite floats by None, recursively, so the value is strict JSON."""
    if isinstance(value, dict):
        return {k: finite_json(v) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [finite_json(v) for v in value]
    if isinstance(value, float | np.floating):
        return float(value) if np.isfinite(value) else None
    return value


def work_dir(smoke: bool) -> Path:
    """Local work folder of the run."""
    path = WORK / ("smoke" if smoke else "full")
    path.mkdir(parents=True, exist_ok=True)
    return path


def heads065() -> list[str]:
    """Return the 49 065 heads in network order."""
    return json.loads((MIL065 / "result.json").read_text())["heads"]


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
    """PTB-XL folds 1-8 and the ECGAD development records: the rows read from the feature files."""
    split = load_split()
    return set(load_partition().ecg_id.astype(int)) | set(split.ecg_id[split.partition == "development"])


def fit_table() -> pd.DataFrame:
    """Fit-partition records with the 066/068 patient folds."""
    partition = load_partition()
    fit = partition[partition.partition == "fit"].reset_index(drop=True)
    fit["fold"] = cb.patient_folds(fit.patient_id.to_numpy(), cv.FOLDS, cv.FOLD_SEED)
    return fit


def training_records(fit: pd.DataFrame) -> pd.DataFrame:
    """Return the 068 combiner records: fit records with a region statement, their statements and folds."""
    table = fit[fit.region_statements != ""].copy()
    table["statements"] = table.region_statements.str.split(";")
    return table[["ecg_id", "patient_id", "fold", "statements"]].reset_index(drop=True)


def sanity_ids(fit: pd.DataFrame) -> list[int]:
    """Fit records with a sanity statement at likelihood >= 50 (for the descriptive prevalence readings)."""
    meta = load_metadata()
    codes = meta.loc[fit.ecg_id, "scp_codes"]
    return [int(e) for e, c in codes.items() if any(c.get(s, 0) >= 50 for s in SANITY)]


def record_sets(smoke: bool) -> dict[str, np.ndarray]:
    """Scored, clean-normal, training and sanity record IDs (smoke: small subsets)."""
    fit = fit_table()
    sets = {
        "scored": scored_ecg_ids().astype(int),
        "normals": fit.ecg_id[fit.clean_normal].to_numpy(int),
        "train": training_records(fit).ecg_id.to_numpy(int),
        "sanity": np.array(sanity_ids(fit), int),
    }
    if smoke:
        sets = {k: v[: SMOKE.get(k, 40)] for k, v in sets.items()}
    if holdout_ids() & set(sets["scored"].tolist()):
        raise ValueError("An ECGAD holdout record would be mapped")
    fitted = set(sets["normals"].tolist()) | set(sets["train"].tolist()) | set(sets["sanity"].tolist())
    if fitted & forbidden_fit_ids():
        raise ValueError("A fit-side record is an ECGAD or evaluation record")
    return sets


def _measure_init() -> None:
    threadpool_limits(limits=1)
    signals, index = open_cache()
    _STATE["signals"] = signals
    _STATE["row"] = dict(zip(index.ecg_id.astype(int), range(len(index)), strict=True))


def _measure(ecg_id: int) -> tuple[int, dict[str, float], np.ndarray | None]:
    raw = np.asarray(_STATE["signals"][_STATE["row"][ecg_id]], np.float64)
    geometry = cr.beat_geometry(raw)
    intervals = None if geometry is None else geometry["intervals"]
    return ecg_id, cr.own_measurements(raw, geometry), intervals


def read_source(name: str, ids: np.ndarray) -> pd.DataFrame:
    """Read one PTB-XL+ feature file, keep folds 1-8 and ECGAD development rows, harmonize, order by ids."""
    raw = pd.read_csv(FEATURES / f"{name}_features.csv").set_index("ecg_id")
    raw = raw[raw.index.isin(allowed_ids())]
    table = cr.source_12sl(raw) if name == "12sl" else cr.source_unig(raw)
    return table.reindex(ids)


def ecgdeli_coverage(ids: np.ndarray) -> dict[str, int]:
    """ECGdeli feature rows present for the measured records (reported only)."""
    raw = pd.read_csv(FEATURES / "ecgdeli_features.csv", usecols=["ecg_id"])
    present = set(raw.ecg_id[raw.ecg_id.isin(allowed_ids())].astype(int))
    return {"records": len(ids), "present": int(sum(int(e) in present for e in ids))}


def measure_phase(smoke: bool) -> None:
    """Own measurements and harmonized commercial measurements of every needed record."""
    work = work_dir(smoke)
    began = time.time()
    sets = record_sets(smoke)
    ids = np.unique(np.concatenate(list(sets.values())))
    own, geometry = {}, {}
    with Pool(WORKERS, initializer=_measure_init) as pool:
        for k, (ecg_id, values, intervals) in enumerate(pool.imap_unordered(_measure, ids.tolist(), 8)):
            own[ecg_id], geometry[ecg_id] = values, intervals
            if (k + 1) % 2000 == 0:
                print(f"measured {k + 1}/{len(ids)} {time.time() - began:.0f}s", flush=True)
    tables = {"own": pd.DataFrame.from_dict(own, orient="index").reindex(ids)}
    tables["12sl"] = read_source("12sl", ids)
    tables["unig"] = read_source("unig", ids)
    with (work / "measurements.pkl").open("wb") as handle:
        pickle.dump({"sets": sets, "tables": tables, "geometry": geometry}, handle, pickle.HIGHEST_PROTOCOL)
    receipt = {
        "created": utc_now(),
        "git_head": git_head(ROOT),
        "records": {k: int(len(v)) for k, v in sets.items()},
        "measured": int(len(ids)),
        "without_beats": sorted(int(e) for e, g in geometry.items() if g is None),
        "missing_rows": {k: int(t.isna().all(axis=1).sum()) for k, t in tables.items()},
        "ecgdeli_coverage": ecgdeli_coverage(ids),
        "seconds": time.time() - began,
    }
    write_json_atomic(work / "measure_receipt.json", receipt)
    print(json.dumps({k: v for k, v in receipt.items() if k != "without_beats"}), flush=True)


def load_measurements(work: Path) -> dict[str, Any]:
    """Load the measure phase output."""
    with (work / "measurements.pkl").open("rb") as handle:
        return pickle.load(handle)  # noqa: S301


def demographics(ids: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Age and sex from the cache index, in ids order."""
    _, index = open_cache()
    table = index.set_index("ecg_id").loc[ids]
    return table.age.to_numpy(np.float64), table.sex.to_numpy(np.float64)


def bag_logits(ids: np.ndarray, kind: str) -> np.ndarray:
    """
    065 bag logits: three-seed ensemble for scored records, 068 cross-fit out-of-fold for training.

    Parameters
    ----------
    ids : np.ndarray
        Record IDs.
    kind : str
        ``"scored"`` or ``"train"``.

    Returns
    -------
    np.ndarray
        (records, 49) logits in ``heads065`` order.
    """
    logits: dict[int, np.ndarray] = {}
    if kind == "scored":
        for set_name in ("ecgad_dev", "evaluation"):
            with np.load(MIL065 / f"bags_{set_name}.npz") as saved:
                logits.update(zip(saved["ecg_id"].astype(int).tolist(), saved["mil_lse"], strict=True))
    else:
        oof = np.load(WORK068 / "oof_bags.npy")
        logits.update(zip(np.load(WORK068 / "oof_ids.npy").astype(int).tolist(), oof, strict=True))
    return np.stack([np.asarray(logits[int(e)], np.float64) for e in ids])


def probabilities(ids: np.ndarray, kind: str) -> np.ndarray:
    """Sigmoid of ``bag_logits``."""
    return 1.0 / (1.0 + np.exp(-bag_logits(ids, kind)))


def reference_columns() -> list[str]:
    """Columns of the normal reference: criterion inputs and the agreement measures."""
    agreement = [f"{m}_{lead}" for m in cr.AGREEMENT for lead in LEADS]
    return list(dict.fromkeys(cr.used_columns() + agreement))


def build_evidence(data: dict[str, Any]) -> dict[str, Any]:
    """
    Consensus and own tables, normal references, exceedances and lead x wave scores of every record.

    Returns
    -------
    dict
        ``ids``, per map (records, 12, 5) scores in ``scores``, ``z`` (consensus and own), references,
        ``q95``, ``reliability`` and ``disagreement``.
    """
    sets, tables = data["sets"], data["tables"]
    ids = tables["own"].index.to_numpy(int)
    sources = [tables["own"], tables["12sl"], tables["unig"]]
    consensus = cr.add_derived(cr.consensus(sources))
    own = cr.add_derived(tables["own"])
    columns = reference_columns()
    ref = {
        "consensus": cr.normal_reference(consensus.loc[sets["normals"]], columns),
        "own": cr.normal_reference(own.loc[sets["normals"]], columns),
    }
    age, sex = demographics(ids)
    z = {
        "consensus": cr.exceedances(consensus, ref["consensus"]["scale"], age, sex),
        "own": cr.exceedances(own, ref["own"]["scale"], age, sex),
    }
    d = cr.disagreement(sources, ref["consensus"]["scale"])
    normal_rows = np.isin(ids, sets["normals"])
    q95 = np.nanpercentile(d[normal_rows], 95, axis=0)
    r = cr.reliability(d, q95)
    base = cr.lead_wave(cr.evidence(z["consensus"]))
    scores = {
        "criteria": base,
        "criteria_agree": base * r[:, :, None],
        "criteria_own": cr.lead_wave(cr.evidence(z["own"])),
    }
    return {
        "ids": ids,
        "consensus": consensus,
        "own": own,
        "ref": ref,
        "z": z,
        "disagreement": d,
        "q95": q95,
        "reliability": r,
        "scores": scores,
    }


def add_mil_scores(ev: dict[str, Any], sets: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """``criteria_x_mil`` scores for scored and training records (keyed by ecg_id)."""
    heads = heads065()
    out: dict[int, np.ndarray] = {}
    for kind in ("scored", "train"):
        ids = sets[kind]
        rows = np.searchsorted(ev["ids"], ids)
        weights = cr.mil_weights(probabilities(ids, kind), heads)
        scores = cr.lead_wave(cr.evidence(ev["z"]["consensus"][rows]) * weights)
        out.update(zip(ids.tolist(), scores, strict=True))
    return out


def _paint_init(work: str, names: tuple[str, ...]) -> None:
    threadpool_limits(limits=1)
    signals, index = open_cache()
    _STATE["signals"] = signals
    _STATE["row"] = dict(zip(index.ecg_id.astype(int), range(len(index)), strict=True))
    _STATE["maps"] = {
        name: np.lib.format.open_memmap(Path(work) / f"{name}.npy", mode="r+") for name in names
    }


def _paint(task: tuple[int, int, np.ndarray, dict[str, np.ndarray]]) -> tuple[int, np.ndarray, np.ndarray]:
    position, ecg_id, intervals, scores = task
    amplitude = amplitude_map(exact_signal(_STATE["signals"][_STATE["row"][ecg_id]]))
    fields = {name: cr.paint(values, intervals) for name, values in scores.items()}
    maps = {name: cr.finish(field, amplitude) for name, field in fields.items()}
    maps["criteria_qrs"] = cr.lexicographic(fields["criteria"], intervals, amplitude)
    units = cb.unit_table(intervals)
    means = np.column_stack([cb.unit_means(units, maps[name].astype(np.float64)) for name in cr.EXTRA_MAPS])
    if position >= 0:
        for name, values in maps.items():
            if not np.isfinite(values).all():
                raise ValueError(f"Nonfinite {name} map for {ecg_id}")
            _STATE["maps"][name][position] = values
    return ecg_id, units, means


def paint_tasks(
    ids: np.ndarray, positions: np.ndarray, ev: dict[str, Any], mil: dict[int, np.ndarray], geometry: dict
) -> list[tuple]:
    """Painting tasks: (position or -1, ecg_id, intervals, per-map lead x wave scores)."""
    rows = np.searchsorted(ev["ids"], ids)
    tasks = []
    for position, ecg_id, row in zip(positions, ids, rows, strict=True):
        intervals = geometry[int(ecg_id)]
        if intervals is None:
            intervals = np.zeros((0, len(cr.COMPONENTS), 2), np.int64)
        scores = {name: values[row] for name, values in ev["scores"].items()}
        scores["criteria_x_mil"] = mil[int(ecg_id)]
        tasks.append((int(position), int(ecg_id), intervals, scores))
    return tasks


def descriptives(ev: dict[str, Any], data: dict[str, Any]) -> dict[str, Any]:
    """Fit-partition agreement, normal reference, criterion prevalence and reliability summaries."""
    sets, tables = data["sets"], data["tables"]
    normals = sets["normals"]
    agreement = {}
    for name in [f"{m}_{lead}" for m in cr.AGREEMENT for lead in LEADS] + [
        "QRSD",
        "PR",
        "QTC",
        "AXIS",
        "PDUR",
    ]:
        pair = {}
        for a, b in (("own", "12sl"), ("own", "unig"), ("12sl", "unig")):
            x, y = tables[a].loc[normals, name], tables[b].loc[normals, name]
            keep = x.notna() & y.notna()
            pair[f"{a}_vs_{b}"] = {
                "median_abs_difference": float(np.median(np.abs(x[keep] - y[keep]))) if keep.any() else None,
                "pearson": float(np.corrcoef(x[keep], y[keep])[0, 1]) if keep.sum() > 2 else None,
                "n": int(keep.sum()),
            }
        agreement[name] = pair
    rows = np.searchsorted(ev["ids"], normals)
    met = {k: cr.met_any_lead(v[rows]).mean(axis=0) for k, v in ev["z"].items()}
    prevalence = {
        k: dict(zip([c.name for c in cr.CRITERIA], v.round(4).tolist(), strict=True)) for k, v in met.items()
    }
    return {
        "agreement_fit_clean_normals": agreement,
        "criterion_prevalence_fit_clean_normals": prevalence,
        "sanity_fit_statements": sanity_readings(ev, sets),
        "q95_disagreement": dict(zip(LEADS, ev["q95"].round(4).tolist(), strict=True)),
        "reliability_below_1_share_normals": float(
            (ev["reliability"][np.isin(ev["ids"], normals)] < 1).mean()
        ),
        "normal_reference": {k: v.round(6).to_dict(orient="index") for k, v in ev["ref"].items()},
    }


def sanity_readings(ev: dict[str, Any], sets: dict[str, np.ndarray]) -> dict[str, Any]:
    """Share of fit records with a statement that meet the matching criterion (consensus and own)."""
    meta = load_metadata()
    names = [c.name for c in cr.CRITERIA]
    ids = np.unique(np.concatenate([sets["sanity"], sets["train"]]))
    out = {}
    for statement, (criteria, leads) in SANITY.items():
        chosen = np.array([e for e in ids if meta.at[e, "scp_codes"].get(statement, 0) >= 50], int)
        if not len(chosen):
            continue
        rows = np.searchsorted(ev["ids"], chosen)
        wanted = [names.index(c) for c in ((criteria,) if isinstance(criteria, str) else criteria)]
        lead_rows = [LEADS.index(lead) for lead in leads] if leads else list(range(len(LEADS)))
        reading = {"records": int(len(chosen))}
        for kind, z in ev["z"].items():
            block = np.nan_to_num(z[rows][:, wanted][:, :, lead_rows], nan=-np.inf)
            reading[kind] = float((block >= 0).any(axis=(1, 2)).mean())
        out[statement] = reading
    return out


def maps_phase(smoke: bool) -> None:
    """Build the evidence of every record, paint the scored maps and the unit means for the combiner."""
    work = work_dir(smoke)
    began = time.time()
    data = load_measurements(work)
    sets = data["sets"]
    ev = build_evidence(data)
    mil = add_mil_scores(ev, sets)
    scored = sets["scored"]
    names = CRITERIA_MAPS
    for name in names:
        np.lib.format.open_memmap(
            work / f"{name}.npy", mode="w+", dtype=np.float32, shape=(len(scored), 12, 5000)
        )
    tasks = paint_tasks(scored, np.arange(len(scored)), ev, mil, data["geometry"])
    train_only = np.setdiff1d(sets["train"], scored)
    tasks += paint_tasks(train_only, np.full(len(train_only), -1), ev, mil, data["geometry"])
    units, means = {}, {}
    with Pool(WORKERS, initializer=_paint_init, initargs=(str(work), names)) as pool:
        for ecg_id, unit_table, unit_means in pool.imap_unordered(_paint, tasks, chunksize=16):
            units[ecg_id], means[ecg_id] = unit_table, unit_means
    (work / "scored_ids.json").write_text(json.dumps([int(e) for e in scored]))
    with (work / "unit_means.pkl").open("wb") as handle:
        pickle.dump({"units": units, "means": means}, handle, pickle.HIGHEST_PROTOCOL)
    rows = np.searchsorted(ev["ids"], scored)
    np.savez_compressed(
        work / "evidence_scored.npz",
        ecg_id=scored,
        criteria=ev["scores"]["criteria"][rows],
        criteria_own=ev["scores"]["criteria_own"][rows],
        criteria_x_mil=np.stack([mil[int(e)] for e in scored]),
        reliability=ev["reliability"][rows],
    )
    receipt = {
        "created": utc_now(),
        "git_head": git_head(ROOT),
        "files": {
            n: {"file": str(work / f"{n}.npy"), "sha256": sha256_file(work / f"{n}.npy")} for n in names
        },
        "descriptives": descriptives(ev, data),
        "seconds": time.time() - began,
    }
    write_json_atomic(work / "maps_receipt.json", finite_json(receipt))
    print(json.dumps(receipt["descriptives"]["criterion_prevalence_fit_clean_normals"]), flush=True)
    print(json.dumps(receipt["descriptives"]["sanity_fit_statements"]), flush=True)


def load_068(kind: str) -> dict[int, dict]:
    """068 saved unit features of the training or scored records."""
    with (WORK068 / f"features_{kind}.pkl").open("rb") as handle:
        return pickle.load(handle)  # noqa: S301


def check_units(features: dict[int, dict], units: dict[int, np.ndarray], ids: np.ndarray) -> int:
    """Raise unless this run's unit tables equal 068's for every record; return the number compared."""
    for ecg_id in ids.tolist():
        if not np.array_equal(features[ecg_id]["units"], units[ecg_id]):
            raise ValueError(f"Unit tables differ from 068 for {ecg_id}")
    return len(ids)


def training_design(train: pd.DataFrame, features: dict[int, dict], means: dict[int, np.ndarray]) -> dict:
    """068 training design with the two criteria columns; drops records with no or only positive units."""
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
        designs.append(cr.design(r["unit_raw"], r["q"], r["units"], means[int(record.ecg_id)]))
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


def cross_validate(data: dict, cols: np.ndarray, l2: float) -> dict[str, float]:
    """Patient-grouped five-fold held-out listwise loss and within-record weak-label AUROC (068 procedure)."""
    x, y, starts, folds = data["x"][:, cols], data["y"], data["starts"], data["folds"]
    losses, aurocs = [], []
    for fold in range(cv.FOLDS):
        train, test = np.flatnonzero(folds != fold), np.flatnonzero(folds == fold)
        if not len(train) or not len(test):
            continue
        w = cv.fit_listwise(*cv.subset(x, y, starts, train), l2)
        xt, yt, st = cv.subset(x, y, starts, test)
        losses.append(cv.listwise_loss(w, xt, yt, st, 0.0)[0] * len(test))
        aurocs.append(cv.within_record_auroc(xt @ w, yt, st) * len(test))
    n = len(folds)
    return {"loss": float(sum(losses) / n), "auroc": float(sum(aurocs) / n)}


def select_and_fit(data: dict) -> dict[str, Any]:
    """Choose (a, L2) by the lowest CV loss over the 068 grid and fit on all training records."""
    rows = []
    for a in cv.SOFT_GRID:
        cols = cr.feature_columns(cr.criteria_features(a))
        for l2 in cv.L2_GRID:
            rows.append({"a": a, "l2": l2, **cross_validate(data, cols, l2)})
            print(f"units_v2_criteria a={a} l2={l2} {rows[-1]}", flush=True)
    best = min(rows, key=lambda r: (r["loss"], -r["l2"]))
    names = cr.criteria_features(best["a"])
    w = cv.fit_listwise(data["x"][:, cr.feature_columns(names)], data["y"], data["starts"], best["l2"])
    return {
        "grid": rows,
        "a": best["a"],
        "l2": best["l2"],
        "cv_loss": best["loss"],
        "cv_auroc": best["auroc"],
        "features": list(names),
        "weights": dict(zip(names, w.tolist(), strict=True)),
    }


def weights_068() -> dict[str, Any]:
    """068 ``units_v2`` features and weights from its fit receipt."""
    model = json.loads((OUT068 / "fit_receipt.json").read_text())["models"]["units_v2"]
    return {"features": model["features"], "weights": model["weights"]}


def _units_init(work: str, names: tuple[str, ...]) -> None:
    threadpool_limits(limits=1)
    signals, index = open_cache()
    _STATE["signals"] = signals
    _STATE["row"] = dict(zip(index.ecg_id.astype(int), range(len(index)), strict=True))
    _STATE["maps"] = {
        name: np.lib.format.open_memmap(Path(work) / f"{name}.npy", mode="r+") for name in names
    }


def _units_paint(task: tuple[int, int, np.ndarray, dict[str, np.ndarray]]) -> int:
    position, ecg_id, units, scores = task
    amplitude = amplitude_map(exact_signal(_STATE["signals"][_STATE["row"][ecg_id]]))
    for name, values in scores.items():
        painted = cb.units_map(values, units, amplitude)
        if not np.isfinite(painted).all():
            raise ValueError(f"Nonfinite {name} map for {ecg_id}")
        _STATE["maps"][name][position] = painted
    return position


def model_scores(design: np.ndarray, model: dict[str, Any]) -> np.ndarray:
    """Linear unit scores of a model on a full 069 design."""
    names = tuple(model["features"])
    return design[:, cr.feature_columns(names)] @ np.array([model["weights"][n] for n in names])


def combiner_phase(smoke: bool) -> None:
    """Rebuild 068 ``units_v2`` from its features and weights, fit ``units_v2_criteria``, paint both."""
    work = work_dir(smoke)
    began = time.time()
    sets = load_measurements(work)["sets"]
    with (work / "unit_means.pkl").open("rb") as handle:
        saved = pickle.load(handle)  # noqa: S301
    features_train, features_scored = load_068("train"), load_068("scored")
    compared = check_units(features_train, saved["units"], sets["train"])
    compared += check_units(features_scored, saved["units"], sets["scored"])
    train = training_records(fit_table())
    train = train[train.ecg_id.isin(sets["train"])].reset_index(drop=True)
    data = training_design(train, features_train, saved["means"])
    model = select_and_fit(data)
    old = weights_068()
    names = ("units_v2_rebuilt", "units_v2_criteria")
    scored = sets["scored"]
    for name in names:
        np.lib.format.open_memmap(
            work / f"{name}.npy", mode="w+", dtype=np.float32, shape=(len(scored), 12, 5000)
        )
    tasks = []
    for k, ecg_id in enumerate(scored.tolist()):
        r = features_scored[ecg_id]
        full = (
            cr.design(r["unit_raw"], r["q"], r["units"], saved["means"][ecg_id])
            if len(r["units"])
            else np.zeros((0, 34))
        )
        scores = {"units_v2_rebuilt": model_scores(full, old), "units_v2_criteria": model_scores(full, model)}
        tasks.append((k, ecg_id, r["units"], scores))
    with Pool(WORKERS, initializer=_units_init, initargs=(str(work), names)) as pool:
        list(pool.imap_unordered(_units_paint, tasks, chunksize=16))
    receipt = {
        "created": utc_now(),
        "git_head": git_head(ROOT),
        "units_compared_with_068": compared,
        "training_records": int(len(data["ecg_ids"])),
        "training_patients": int(len(np.unique(data["patients"]))),
        "training_dropped": data["dropped"],
        "training_units": int(len(data["y"])),
        "training_positive_units": int(data["y"].sum()),
        "model": model,
        "files": {
            n: {"file": str(work / f"{n}.npy"), "sha256": sha256_file(work / f"{n}.npy")} for n in names
        },
        "seconds": time.time() - began,
    }
    write_json_atomic(work / "combiner_receipt.json", finite_json(receipt))
    print(
        json.dumps({k: v for k, v in receipt.items() if k not in ("files", "training_dropped")}, indent=1),
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
    """061 ``qrs_prior`` ECGAD development A2 through this run's scoring path."""
    result = rescore_tables(REFERENCES[BAR])
    result["A2_target"] = BAR_A2
    result["pass"] = bool(result["pass"] and abs(result["A2"] - BAR_A2) <= TOLERANCE)
    if not result["pass"]:
        raise ValueError(f"Bar not reproduced: {result}")
    return result


def reproduce_068(work: Path) -> dict[str, Any]:
    """068 ``units_v2`` rebuilt from its features and weights against its map file and its tables."""
    scored = json.loads((work / "scored_ids.json").read_text())
    if scored != json.loads((WORK068 / "scored_ids.json").read_text()):
        raise ValueError("Scored records differ from 068")
    rebuilt = np.load(work / "units_v2_rebuilt.npy", mmap_mode="r")
    original = np.load(WORK068 / "units_v2.npy", mmap_mode="r")
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
    saved = pd.read_csv(REFERENCES["units_v2"] / "ecgad.csv")["A2"].to_numpy()
    difference = float(np.nanmax(np.abs(values - saved)))
    result = {
        "map_max_abs_difference": largest,
        "per_record_A2_max_abs_difference": difference,
        "A2": float(np.nanmean(values)),
        "pass": bool(difference <= TOLERANCE),
    }
    if not result["pass"]:
        raise ValueError(f"068 units_v2 not reproduced: {result}")
    return result


def check_bags(scored: np.ndarray) -> dict[str, Any]:
    """065 ``bags_*.npz`` logits against 066's ``mil_bags.npy`` (used by 068) for the scored records."""
    ids = np.load(WORK066 / "mil_ids.npy").astype(int)
    row = {e: k for k, e in enumerate(ids.tolist())}
    bags = np.load(WORK066 / "mil_bags.npy")
    mine = bag_logits(scored, "scored")
    theirs = np.stack([bags[row[int(e)]] for e in scored]).astype(np.float64)
    largest = float(np.abs(mine - theirs).max())
    return {"records": int(len(scored)), "max_abs_logit_difference": largest, "pass": bool(largest <= 1e-4)}


def separation(sets: dict[str, np.ndarray]) -> dict[str, Any]:
    """Data-separation checks of every fitted quantity and of the mapped records."""
    forbidden = forbidden_fit_ids()
    fit = set(fit_table().ecg_id.astype(int))
    checks = {
        "normal_reference_records": int(len(sets["normals"])),
        "normal_reference_forbidden": len(set(sets["normals"].tolist()) & forbidden),
        "normal_reference_outside_fit": len(set(sets["normals"].tolist()) - fit),
        "combiner_records": int(len(sets["train"])),
        "combiner_forbidden": len(set(sets["train"].tolist()) & forbidden),
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
    base = [refs[BAR], refs["units_v2"], refs["textbook"]]
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
    others = [refs["units_v2"], refs["textbook"]]
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
    """Paired strata against ``qrs_prior`` and ``units_v2``; paired C1_std against the references."""
    folders = {name: OUT / "locbench" / name / "tables" for name in MAPS} | refs
    strata = {name: pd.read_csv(folder / "strata.csv") for name, folder in folders.items()}
    noise = {name: standardized_noise_table(folder) for name, folder in folders.items()}
    out: dict[str, Any] = {}
    for name in MAPS:
        out[name] = {
            f"{stratum}_vs_{other}": strata_paired(strata[name], strata[other], stratum)
            for stratum in ("A2_qrs", "A2_nonqrs")
            for other in (BAR, "units_v2")
        }
        for other in (PRIMARY, BAR, "units_v2"):
            if other != name:
                out[name][f"C1_std_vs_{other}"] = strata_paired(noise[name], noise[other], "C1_std")
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
    overall = frame.drop(columns="family").mean().to_dict()
    return {
        "by_family": summary.round(4).to_dict(orient="index"),
        "overall": overall,
        "records": int(keep.sum()),
    }


def ecgad_specificity(scores: np.ndarray, ids: np.ndarray) -> dict[str, Any]:
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
    agree = paired["criteria_agree"]["C1_std_vs_criteria"]
    return {
        "primary": {"map": PRIMARY, "against": BAR, "A2_difference": primary, "decision": decision(primary)},
        "units_v2_criteria_vs_units_v2": summaries["units_v2_criteria"]["paired"]["units_v2"]["A2"],
        "criteria_x_mil_vs_textbook": summaries["criteria_x_mil"]["paired"]["textbook"]["A2"],
        "criteria_agree_C1_std_vs_criteria": agree,
        "criteria_agree_lowers_C1_std": bool(agree["ci_high"] < 0),
        "exploratory": "ECGAD development has been compared many times; no reading here is confirmed.",
    }


def score_phase() -> None:
    """Integrity checks, LocBench scoring, strata, standardized noise, specificity readings, result.json."""
    work = work_dir(False)
    OUT.mkdir(parents=True, exist_ok=True)
    began = time.time()
    sets = load_measurements(work)["sets"]
    scored = np.array(json.loads((work / "scored_ids.json").read_text()), np.int64)
    if not np.array_equal(scored, scored_ecg_ids()):
        raise ValueError("Scored records changed")
    integrity = {
        "bar": reproduce_bar(),
        "units_v2_068": reproduce_068(work),
        "bags065": check_bags(scored),
        "separation": separation(sets),
    }
    refs = copy_references()
    summaries = score_maps(work, refs)
    run_secondary_clis(refs)
    paired = paired_secondary(refs)
    mine, refs_out = readings(refs, summaries)
    with np.load(work / "evidence_scored.npz") as saved:
        criteria_scores = saved["criteria"]
    specificity = {
        "region_set": region_specificity(criteria_scores, scored),
        "ecgad_development": ecgad_specificity(criteria_scores, scored),
    }
    result = {
        "experiment": "069",
        "protocol": "docs/experiment-069-criteria-localizer.md",
        "protocol_sha256": sha256_file(ROOT / "docs/experiment-069-criteria-localizer.md"),
        "created": utc_now(),
        "git_head": git_head(ROOT),
        "inputs": inputs_record(),
        "integrity": integrity,
        "measure": json.loads((work / "measure_receipt.json").read_text()),
        "maps": json.loads((work / "maps_receipt.json").read_text()),
        "combiner": json.loads((work / "combiner_receipt.json").read_text()),
        "readings": mine,
        "paired_secondary": paired,
        "reference_readings": refs_out,
        "specificity": specificity,
        "decisions": decisions(summaries, paired),
        "score_seconds": time.time() - began,
    }
    write_json_atomic(OUT / "result.json", finite_json(result))
    print(json.dumps(result["decisions"], indent=1), flush=True)


def inputs_record() -> dict[str, Any]:
    """Hashes of the inputs and sources of the run."""
    return {
        "partition": to_stored(PARTITION_PATH),
        "partition_sha256": sha256_file(PARTITION_PATH),
        "cache_dir": str(CACHE_DIR),
        "features_sha256": {
            n: sha256_file(FEATURES / f"{n}_features.csv") for n in ("12sl", "unig", "ecgdeli")
        },
        "bags065_sha256": {n: sha256_file(MIL065 / f"bags_{n}.npz") for n in ("ecgad_dev", "evaluation")},
        "mil065_result_sha256": sha256_file(MIL065 / "result.json"),
        "work068_sha256": {
            n: sha256_file(WORK068 / n)
            for n in (
                "features_train.pkl",
                "features_scored.pkl",
                "oof_bags.npy",
                "oof_ids.npy",
                "units_v2.npy",
            )
        },
        "fit_receipt068_sha256": sha256_file(OUT068 / "fit_receipt.json"),
        "sources_sha256": {path: sha256_file(ROOT / path) for path in SOURCES},
    }


def main() -> None:
    """Command-line entry point."""
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=["measure", "maps", "combiner", "score"])
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    if args.phase == "measure":
        measure_phase(args.smoke)
    elif args.phase == "maps":
        maps_phase(args.smoke)
    elif args.phase == "combiner":
        combiner_phase(args.smoke)
    elif args.smoke:
        raise ValueError("The smoke run is not scored")
    else:
        score_phase()


if __name__ == "__main__":
    main()
