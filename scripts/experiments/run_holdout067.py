"""
Experiment 067: one-time confirmation of the frozen localization maps on the ECGAD holdout half.

Phases::

    run_holdout067 maps    # stage 2: reproduce the development maps, then the holdout maps and manifest
    run_holdout067 score   # stage 3: the single scoring run on the holdout marks

``maps`` uses GPU0 with the shared lock for the 065 MIL evidence and 8 CPU processes; it never reads ECGAD
labels. ``score`` checks every map hash against the committed manifest before reading the holdout marks.
See ``docs/experiment-067-holdout.md``.
"""

from __future__ import annotations

import argparse
import json
import os
import time
import traceback
from multiprocessing import get_context
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from ecg_experiment import ROOT
from ecg_experiment import combination066 as cb
from ecg_experiment import holdout067 as ho
from ecg_experiment.ecgad import load_labels
from ecg_experiment.eda.ptbxl import load_metadata
from ecg_experiment.files import sha256_file, write_json_atomic
from ecg_experiment.gpu import gpu_lock
from ecg_experiment.locbench import metric_interval, paired_difference
from ecg_experiment.locbench_baselines061 import exact_signal
from ecg_experiment.locbench_cache import CACHE_DIR, open_cache
from ecg_experiment.locbench_strata import strata_paired, strata_summary
from ecg_experiment.paths import to_stored
from ecg_experiment.provenance import git_head, utc_now
from ecg_experiment.waveforms import LEADS

OUT = ROOT / "outputs/experiment067_holdout_v1"
WORK = Path(os.environ.get("ECG067_WORK", "/tmp/ecg067-work"))
MANIFEST = ROOT / "docs/experiment-067-maps-manifest.json"
PROTOCOL = "docs/experiment-067-holdout.md"
SOURCES = (
    PROTOCOL,
    "ecg_experiment/holdout067.py",
    "scripts/experiments/run_holdout067.py",
    "ecg_experiment/combination066.py",
    "ecg_experiment/combiner068.py",
    "ecg_experiment/mil065.py",
    "ecg_experiment/normative_map062.py",
    "ecg_experiment/locbench_baselines061.py",
    "ecg_experiment/locbench.py",
    "ecg_experiment/locbench_strata.py",
)
EXACT = ("random", "amplitude", "qrs_prior", "U_B", "residual056", "qrs_lexicographic")
MAP_TOLERANCE = 1e-5
BAG_TOLERANCE = 1e-3
Q_TOLERANCE = 1e-12
A2_TOLERANCE = 1e-12
RANDOM_RANGE = (0.48, 0.52)
METRICS = ("A1", "A2", "A3", "A4", "A5", "A6", "marked_share")
PAIRED_METRICS = ("A1", "A2", "A3", "A4", "A5", "A6")
WORKERS = 8

_STATE: dict[str, Any] = {}


def folder(kind: str) -> Path:
    """Work folder of the development or holdout maps."""
    return WORK / kind


def _quality_one(stored: np.ndarray) -> np.ndarray:
    return cb.quality_features(exact_signal(stored))


def _worker_init() -> None:
    threadpool_limits(limits=1)
    _STATE["out"] = {name: np.load(_STATE["folder"] / f"maps_{name}.npy", mmap_mode="r+") for name in ho.MAPS}


def _map_row(row: int) -> int:
    ecg_id = int(_STATE["ecg_ids"][row])
    mil_row = {name: values[row] for name, values in _STATE["mil"].items()}
    maps = ho.record_maps(
        _STATE["signals"][row],
        ecg_id,
        float(_STATE["age"][row]),
        float(_STATE["sex"][row]),
        mil_row,
        _STATE["q"][row],
        _STATE["truth"][row],
        _STATE["frozen"],
    )
    for name, values in maps.items():
        _STATE["out"][name][row] = values
    return row


def build_set(records: pd.DataFrame, kind: str, frozen: dict[str, Any]) -> dict[str, Any]:
    """
    Map one record set with the frozen pipeline into ``WORK/<kind>/maps_<name>.npy``.

    Parameters
    ----------
    records : pd.DataFrame
        ``split_records`` output.
    kind : str
        ``development`` or ``holdout``.
    frozen : dict[str, Any]
        ``holdout067.load_frozen`` output.

    Returns
    -------
    dict[str, Any]
        MIL summaries, q, timings and the map file hashes.
    """
    began = time.time()
    target = folder(kind)
    target.mkdir(parents=True, exist_ok=True)
    signals, index = open_cache()
    row_of = dict(zip(index.ecg_id.astype(int), range(len(index)), strict=True))
    positions = np.array([row_of[int(e)] for e in records.ecg_id])
    stored = np.stack([np.asarray(signals[p]) for p in positions])
    covariates = index.iloc[positions]
    device = ho.device_setup()
    with gpu_lock(device):
        networks = ho.load_networks(len(frozen["heads"]), device)
        summaries = ho.mil_summaries(stored, networks, frozen["heads"], device)
        del networks
    mil_seconds = time.time() - began
    with get_context("fork").Pool(WORKERS, initializer=threadpool_limits, initargs=(1,)) as pool:
        features = np.stack(pool.map(_quality_one, list(stored), chunksize=8))
    q = ho.lead_quality(features, frozen["quality"])
    truth = ho.true_labels(records.ecg_id.to_numpy(), frozen["heads"])
    for name in ho.MAPS:
        np.lib.format.open_memmap(
            target / f"maps_{name}.npy", mode="w+", dtype=np.float32, shape=(len(records), len(LEADS), 5000)
        ).flush()
    _STATE.update(
        folder=target,
        ecg_ids=records.ecg_id.to_numpy(),
        signals=stored,
        age=covariates.age.to_numpy(),
        sex=covariates.sex.to_numpy(),
        mil=summaries,
        q=q,
        truth=truth,
        frozen=frozen,
    )
    with get_context("fork").Pool(WORKERS, initializer=_worker_init) as pool:
        done = list(pool.imap_unordered(_map_row, range(len(records)), chunksize=4))
    if sorted(done) != list(range(len(records))):
        raise RuntimeError(f"{kind}: not every record was mapped")
    np.savez(target / "mil_summaries.npz", ecg_id=records.ecg_id.to_numpy(), q=q, **summaries)
    return {
        "records": len(records),
        "device": device,
        "mil_seconds": mil_seconds,
        "seconds": time.time() - began,
        "summaries": summaries,
        "q": q,
        "maps_sha256": ho.manifest_hashes(target),
    }


def development_check(records: pd.DataFrame, built: dict[str, Any]) -> dict[str, Any]:
    """
    Compare the wrapper's development maps, bag logits and q with the saved development outputs.

    Parameters
    ----------
    records : pd.DataFrame
        Development records.
    built : dict[str, Any]
        ``build_set`` output for the development records.

    Returns
    -------
    dict[str, Any]
        Per map differences, the bag and q checks and ``pass``.
    """
    order = records.ecg_id.to_numpy()
    maps = {}
    for name in ho.MAPS:
        saved_order, saved = ho.saved_development_maps(name)
        if not np.array_equal(saved_order, order):
            raise ValueError(f"Saved development order of {name} differs")
        new = np.load(folder("development") / f"maps_{name}.npy", mmap_mode="r")
        result = ho.map_difference(new, saved)
        tolerance = 0.0 if name in EXACT else MAP_TOLERANCE
        result["tolerance"] = tolerance
        result["pass"] = bool(
            result["identical"] if name in EXACT else result["max_abs_difference"] <= tolerance
        )
        maps[name] = result
    with np.load(ho.MIL065 / "bags_ecgad_dev.npz") as saved:
        bag_of = dict(zip(saved["ecg_id"].astype(int).tolist(), saved["mil_lse"], strict=True))
    bags = np.stack([bag_of[int(e)] for e in order])
    bag_difference = float(np.abs(built["summaries"]["bags"] - bags).max())
    with np.load(ho.OUT066 / "quality/q.npz") as saved:
        q_of = dict(zip(saved["scored_ids"].astype(int).tolist(), saved["q_scored"], strict=True))
    q_saved = np.stack([q_of[int(e)] for e in order])
    q_difference = float(np.abs(built["q"] - q_saved).max())
    checks = {
        "maps": maps,
        "bags": {"max_abs_difference": bag_difference, "tolerance": BAG_TOLERANCE},
        "q": {"max_abs_difference": q_difference, "tolerance": Q_TOLERANCE},
    }
    checks["pass"] = bool(
        all(m["pass"] for m in maps.values())
        and bag_difference <= BAG_TOLERANCE
        and q_difference <= Q_TOLERANCE
    )
    return checks


def maps_phase() -> None:
    """Stage 2: development reproduction, then the holdout maps and the manifest. No label is read."""
    if (OUT / "result.json").exists():
        raise FileExistsError("The holdout has been scored; maps are never rebuilt")
    OUT.mkdir(parents=True, exist_ok=True)
    began = time.time()
    hashes = ho.frozen_hashes()
    receipt_check = ho.receipt_hash_check(hashes)
    if not receipt_check["pass"]:
        raise ValueError(f"Frozen inputs differ from the 066 receipt: {receipt_check}")
    frozen = ho.load_frozen()
    development = ho.split_records("development")
    holdout = ho.split_records("holdout")
    if len(holdout) != 201 or set(holdout.ecg_id) & set(development.ecg_id):
        raise ValueError("Unexpected holdout records")
    dev_built = development_check_and_build(development, frozen)
    print(json.dumps({k: v["max_abs_difference"] for k, v in dev_built["check"]["maps"].items()}), flush=True)
    hold_built = build_set(holdout, "holdout", frozen)
    manifest = {
        "experiment": "067",
        "protocol": PROTOCOL,
        "protocol_sha256": sha256_file(ROOT / PROTOCOL),
        "created": utc_now(),
        "git_head": git_head(ROOT),
        "dtype": "float32",
        "shape_per_record": [len(LEADS), 5000],
        "leads": list(LEADS),
        "holdout": {
            "folder": str(folder("holdout")),
            "ecg_ids": holdout.ecg_id.tolist(),
            "patient_ids": holdout.patient_id.tolist(),
            "bench_index": holdout.bench_index.tolist(),
            "maps_sha256": hold_built["maps_sha256"],
            "mil_summaries_sha256": sha256_file(folder("holdout") / "mil_summaries.npz"),
        },
        "development": {
            "folder": str(folder("development")),
            "ecg_ids": development.ecg_id.tolist(),
            "maps_sha256": dev_built["built"]["maps_sha256"],
        },
        "frozen_inputs_sha256": hashes,
        "sources_sha256": {path: sha256_file(ROOT / path) for path in SOURCES},
        "cache": {"directory": str(CACHE_DIR), "index_sha256": sha256_file(CACHE_DIR / "index.json")},
    }
    timing = {
        "development": {k: dev_built["built"][k] for k in ("records", "device", "mil_seconds", "seconds")},
        "holdout": {k: hold_built[k] for k in ("records", "device", "mil_seconds", "seconds")},
        "total_seconds": time.time() - began,
    }
    write_json_atomic(OUT / "maps_manifest.json", manifest)
    write_json_atomic(
        OUT / "maps_check.json", {**dev_built["check"], "receipt_hashes": receipt_check, **timing}
    )
    print(json.dumps(timing), flush=True)


def development_check_and_build(development: pd.DataFrame, frozen: dict[str, Any]) -> dict[str, Any]:
    """Map the development records and stop before any holdout record if they are not reproduced."""
    built = build_set(development, "development", frozen)
    check = development_check(development, built)
    if not check["pass"]:
        write_json_atomic(OUT / "maps_check.json", check)
        raise ValueError(f"Development maps not reproduced; no holdout record mapped: {check}")
    return {"built": built, "check": check}


def verify_manifest(manifest: dict[str, Any], holdout: pd.DataFrame, development: pd.DataFrame) -> dict:
    """Map file hashes and record orders against the committed manifest."""
    if manifest["holdout"]["ecg_ids"] != holdout.ecg_id.tolist():
        raise ValueError("Holdout record order differs from the manifest")
    if manifest["development"]["ecg_ids"] != development.ecg_id.tolist():
        raise ValueError("Development record order differs from the manifest")
    current = {kind: ho.manifest_hashes(folder(kind)) for kind in ("holdout", "development")}
    for kind, hashes in current.items():
        if hashes != manifest[kind]["maps_sha256"]:
            raise ValueError(f"{kind} map hashes differ from the committed manifest")
    return {"maps_checked": sum(len(h) for h in current.values()), "pass": True}


def development_tables(development: pd.DataFrame, labels: np.ndarray) -> tuple[dict, dict]:
    """Development tables from the wrapper maps and their check against the saved development A2."""
    marks = ho.label_rows(labels, development.bench_index.to_numpy())
    tables, check = {}, {}
    for name in ho.MAPS:
        maps = np.load(folder("development") / f"maps_{name}.npy", mmap_mode="r")
        tables[name] = ho.ecgad_tables(maps, marks, development, name)
        saved = pd.read_csv(ho.DEVELOPMENT_TABLES[name] / "ecgad.csv")
        mine = tables[name]["ecgad"]
        if not np.array_equal(saved.ecg_id.to_numpy(), mine.ecg_id.to_numpy()):
            raise ValueError(f"Development table order of {name} differs")
        a, b = mine.A2.to_numpy(), saved.A2.to_numpy()
        same_nan = bool(np.array_equal(np.isnan(a), np.isnan(b)))
        largest = float(np.nanmax(np.abs(a - b))) if same_nan else float("inf")
        target = ho.DEVELOPMENT_A2.get(name, float(np.nanmean(b)))
        mean = float(np.nanmean(a))
        check[name] = {
            "per_record_max_abs_difference": largest,
            "A2": mean,
            "A2_saved": target,
            "pass": bool(largest <= A2_TOLERANCE and abs(mean - target) <= A2_TOLERANCE),
        }
    return tables, check


def score_phase() -> None:
    """Stage 3: the single scoring run."""
    attempts = OUT / "score_attempts.jsonl"
    attempt = {"started": utc_now(), "git_head": git_head(ROOT)}
    with attempts.open("a") as handle:
        handle.write(json.dumps({**attempt, "event": "start"}) + "\n")
    try:
        result = score(attempt)
    except Exception:
        with attempts.open("a") as handle:
            handle.write(json.dumps({**attempt, "event": "crash", "error": traceback.format_exc()}) + "\n")
        raise
    with attempts.open("a") as handle:
        handle.write(json.dumps({**attempt, "event": "complete", "finished": result["finished"]}) + "\n")


def score(attempt: dict[str, Any]) -> dict[str, Any]:
    """Hash checks, development check, holdout metrics and result.json."""
    began = time.time()
    manifest = json.loads(MANIFEST.read_text())
    holdout = ho.split_records("holdout")
    development = ho.split_records("development")
    integrity: dict[str, Any] = {"manifest": verify_manifest(manifest, holdout, development)}
    integrity["labels"] = ho.label_hash_check()
    if not integrity["labels"]["pass"]:
        raise ValueError("ECGAD label file differs from the benchmark note")
    labels = load_labels()
    dev_tables, integrity["development_A2"] = development_tables(development, labels)
    if not all(c["pass"] for c in integrity["development_A2"].values()):
        raise ValueError(f"Development A2 not reproduced: {integrity['development_A2']}")
    print("integrity passed; reading holdout marks", flush=True)

    marks = ho.label_rows(labels, holdout.bench_index.to_numpy())
    maps = {name: np.load(folder("holdout") / f"maps_{name}.npy") for name in ho.MAPS}
    tables = {name: ho.ecgad_tables(maps[name], marks, holdout, name) for name in ho.MAPS}
    with threadpool_limits(limits=WORKERS):
        metrics = {name: {m: metric_interval(tables[name], m) for m in METRICS} for name in ho.MAPS}
        print("metrics done", flush=True)
        paired = {
            name: {m: paired_difference(tables[name], tables[ho.BAR], m) for m in PAIRED_METRICS}
            for name in ho.MAPS
            if name != ho.BAR
        }
    print("paired done", flush=True)
    extra = {
        f"{ho.PRIMARY}_minus_{other}": paired_difference(tables[ho.PRIMARY], tables[other], "A2")
        for other in ("textbook", "units_learned")
    }
    primary = paired[ho.PRIMARY]["A2"]

    qrs = maps[ho.BAR] >= 0.5
    strata_tables = {name: ho.strata_records(maps[name], marks, qrs, holdout) for name in ho.MAPS}
    strata = {
        name: {
            "summary": strata_summary(table),
            "paired_vs_qrs_prior": None
            if name == ho.BAR
            else {s: strata_paired(table, strata_tables[ho.BAR], s) for s in ho.STRATA},
        }
        for name, table in strata_tables.items()
    }
    groups = ho.statement_groups(holdout.ecg_id.to_numpy(), load_metadata())
    group_table = ho.group_means({name: tables[name]["ecgad"] for name in ho.MAPS}, groups)
    marked_ids = tables[ho.BAR]["ecgad"].query("marked").ecg_id.astype(int)
    group_counts = pd.Series([g for e in marked_ids for g in groups[int(e)]]).value_counts().to_dict()
    shift = {name: ho.independent_shift(tables[name]["ecgad"], dev_tables[name]["ecgad"]) for name in ho.MAPS}
    random_a2 = metrics["random"]["A2"]["value"]
    integrity["random_A2"] = {
        "value": random_a2,
        "range": list(RANDOM_RANGE),
        "pass": RANDOM_RANGE[0] <= random_a2 <= RANDOM_RANGE[1],
    }
    first = tables[ho.BAR]["ecgad"]
    counts = {
        "records": len(first),
        "patients": int(first.patient_id.nunique()),
        "marked": int(first.marked.sum()),
        "A2_records": int(first.A2.notna().sum()),
        "A2_patients": int(first[first.A2.notna()].patient_id.nunique()),
        "A3_records": int(first.A3.notna().sum()),
        "A4_records": int(first.A4.notna().sum()),
        "strata_records": {s: int(strata_tables[ho.BAR][s].notna().sum()) for s in ho.STRATA},
    }

    for name in ho.MAPS:
        target = OUT / "tables" / name
        target.mkdir(parents=True, exist_ok=True)
        tables[name]["ecgad"].to_csv(target / "ecgad_holdout.csv", index=False, float_format="%.17g")
        strata_tables[name].to_csv(target / "strata_holdout.csv", index=False, float_format="%.17g")
        dev_tables[name]["ecgad"].to_csv(target / "ecgad_development.csv", index=False, float_format="%.17g")
    group_table.to_csv(OUT / "statement_groups.csv", index=False, float_format="%.17g")
    result = {
        "experiment": "067",
        "protocol": PROTOCOL,
        "protocol_sha256": sha256_file(ROOT / PROTOCOL),
        "attempt": attempt,
        "manifest": to_stored(MANIFEST),
        "manifest_sha256": sha256_file(MANIFEST),
        "integrity": integrity,
        "counts": counts,
        "primary": {
            "map": ho.PRIMARY,
            "against": ho.BAR,
            "A2_difference": primary,
            "decision": ho.decision(primary),
        },
        "metrics": metrics,
        "paired_vs_qrs_prior": paired,
        "paired_extra": extra,
        "strata": strata,
        "statement_groups": group_table.to_dict(orient="records"),
        "statement_group_marked_counts": group_counts,
        "shift": shift,
        "candidates": list(ho.CANDIDATES),
        "references": list(ho.REFERENCES),
        "sources_sha256": {path: sha256_file(ROOT / path) for path in SOURCES},
        "seconds": time.time() - began,
        "finished": utc_now(),
    }
    write_json_atomic(OUT / "result.json", result)
    print(json.dumps(result["primary"], indent=1), flush=True)
    return result


def main() -> None:
    """Command-line entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=["maps", "score"])
    args = parser.parse_args()
    if args.phase == "maps":
        maps_phase()
    else:
        score_phase()


if __name__ == "__main__":
    main()
