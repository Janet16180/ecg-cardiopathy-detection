"""
Experiment 062: fit the normative phase-aligned model on LocBench fit clean normals and write maps.

Usage::

    PYTHONPATH=. python -m scripts.experiments.run_normative062 fit
    PYTHONPATH=. python -m scripts.experiments.run_normative062 maps [--ids FILE]

``fit`` writes ``normative.pkl`` and ``fit_summary.json``; ``maps`` writes one float16 memmap per
declared variant (``maps_<variant>.npy``, shape (n, 12, 5000)), ``ecg_ids.npy`` and
``record_summary.csv``. Without ``--ids`` the targets are ``locbench.scored_ecg_ids()``.
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import subprocess
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

from ecg_experiment import ROOT
from ecg_experiment import normative_map062 as nm
from ecg_experiment.ecgad import load_split
from ecg_experiment.files import sha256_file
from ecg_experiment.locbench import (
    ecgad_record_metrics,
    load_benchmark,
    load_tables,
    scored_ecg_ids,
    summarize,
)
from ecg_experiment.locbench_cache import open_cache
from ecg_experiment.locbench_data import PARTITION_PATH, load_partition
from ecg_experiment.paths import to_stored
from ecg_experiment.provenance import git_head

OUT = ROOT / "outputs/experiment062_normative_v1"
BASELINE_DIR = ROOT / "outputs/experiment061_locbench_v1"
BASELINES = ("random", "amplitude", "qrs_prior", "U_B", "residual056")
BAR = "qrs_prior"
TOLERANCE = 1e-12
WORKERS = 8
SOURCES = (
    "ecg_experiment/normative_map062.py",
    "ecg_experiment/morphology_boundaries.py",
    "ecg_experiment/fragment_localization.py",
    "scripts/experiments/run_normative062.py",
    "docs/experiment-062-normative-map.md",
)

_STATE: dict = {}


def _init(normative_path: str | None) -> None:
    signals, index = open_cache()
    _STATE["signals"] = signals
    _STATE["row"] = {int(e): i for i, e in enumerate(index.ecg_id)}
    _STATE["index"] = index.set_index("ecg_id")
    if normative_path:
        with open(normative_path, "rb") as handle:
            _STATE["normative"] = pickle.load(handle)  # noqa: S301


def _signal(ecg_id: int) -> np.ndarray:
    return np.asarray(_STATE["signals"][_STATE["row"][ecg_id]], dtype=float)


def _features(ecg_id: int) -> nm.RecordFeatures | None:
    return nm.record_features(_signal(ecg_id))


def _maps(ecg_id: int) -> tuple[int, dict, dict]:
    meta = _STATE["index"].loc[ecg_id]
    maps, summary = nm.record_maps(_signal(ecg_id), float(meta.age), float(meta.sex), _STATE["normative"])
    return ecg_id, {name: value.astype(np.float16) for name, value in maps.items()}, summary


def fit_ids(limit: int | None = None) -> np.ndarray:
    """
    LocBench ``fit`` partition clean normals.

    Parameters
    ----------
    limit : int or None
        Keep only the first records (smoke tests).

    Returns
    -------
    np.ndarray
        ECG IDs in partition-table order.
    """
    partition = load_partition()
    ids = partition[(partition.partition == "fit") & partition.clean_normal].ecg_id.to_numpy().astype(int)
    return ids[:limit] if limit else ids


def default_targets() -> np.ndarray:
    """
    Return the LocBench v1 scored records.

    Returns
    -------
    np.ndarray
        ``locbench.scored_ecg_ids()``, sorted.
    """
    return scored_ecg_ids()


def run_fit(out: Path, limit: int | None) -> None:
    """
    Compute fit-normal features, fit the normative model and save it.

    Parameters
    ----------
    out : Path
        Output directory.
    limit : int or None
        Smoke-test record limit.
    """
    start = time.time()
    ids = fit_ids(limit)
    with Pool(WORKERS, initializer=_init, initargs=(None,)) as pool:
        features = pool.map(_features, ids.tolist(), chunksize=16)
    kept = [index for index, value in enumerate(features) if value is not None]
    _init(None)
    meta = _STATE["index"].loc[ids[kept]]
    normative = nm.fit_normative(
        [features[index] for index in kept],
        meta.age.to_numpy(),
        meta.sex.to_numpy(),
        meta.patient_id.to_numpy(),
    )
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "normative.pkl", "wb") as handle:
        pickle.dump(normative, handle)
    intervals = np.stack([features[index].intervals for index in kept])
    summary = {
        "fit_records": int(len(ids)),
        "fit_used": int(len(kept)),
        "fit_failed_ids": [int(ids[index]) for index, value in enumerate(features) if value is None],
        "fit_patients": int(meta.patient_id.nunique()),
        "beats_total": int(sum(len(features[index].knots) for index in kept)),
        "interval_medians": dict(zip(nm.INTERVALS, np.nanmedian(intervals, axis=0).tolist(), strict=True)),
        "pr_missing": int(np.isnan(intervals[:, 3]).sum()),
        "subspace_eigenvalues": normative.models["full"].eigenvalues.tolist(),
        "subspace_noise": normative.models["full"].noise,
        "seconds": time.time() - start,
        "normative_sha256": sha256_file(out / "normative.pkl"),
        "partition": to_stored(PARTITION_PATH),
        "partition_sha256": sha256_file(PARTITION_PATH),
    }
    (out / "fit_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps({k: v for k, v in summary.items() if k != "subspace_eigenvalues"}, indent=2))


def run_maps(out: Path, ids: np.ndarray) -> None:
    """
    Write float16 maps of every declared variant for the target records.

    Parameters
    ----------
    out : Path
        Output directory holding ``normative.pkl``.
    ids : np.ndarray
        Target ECG IDs.
    """
    start = time.time()
    split = load_split()
    if set(ids) & set(split[split.partition == "holdout"].ecg_id):
        raise ValueError("ECGAD holdout records are never mapped in Experiment 062")
    maps = {
        name: np.lib.format.open_memmap(
            out / f"maps_{name}.npy.part", mode="w+", dtype=np.float16, shape=(len(ids), nm.LEADS, nm.SAMPLES)
        )
        for name in nm.VARIANTS
    }
    position = {int(e): i for i, e in enumerate(ids)}
    rows = []
    with Pool(WORKERS, initializer=_init, initargs=(str(out / "normative.pkl"),)) as pool:
        for ecg_id, record, summary in pool.imap_unordered(_maps, ids.tolist(), chunksize=8):
            for name, value in record.items():
                maps[name][position[ecg_id]] = value
            rows.append({"ecg_id": ecg_id, **summary})
    for name, array in maps.items():
        array.flush()
        del array
        os.replace(out / f"maps_{name}.npy.part", out / f"maps_{name}.npy")
    np.save(out / "ecg_ids.npy", ids)
    (out / "ecg_ids.json").write_text(json.dumps([int(e) for e in ids]))
    table = pd.DataFrame(rows).set_index("ecg_id").loc[ids].reset_index()
    table.to_csv(out / "record_summary.csv", index=False)
    receipt = {
        "records": int(len(ids)),
        "failed": int(table.failed.sum()),
        "failed_ids": table.ecg_id[table.failed].astype(int).tolist(),
        "variants": {name: [list(components), mode] for name, (components, mode) in nm.VARIANTS.items()},
        "primary": nm.PRIMARY,
        "seconds": time.time() - start,
        "maps_sha256": {name: sha256_file(out / f"maps_{name}.npy") for name in nm.VARIANTS},
        "ecg_ids_sha256": sha256_file(out / "ecg_ids.npy"),
        "normative_sha256": sha256_file(out / "normative.pkl"),
        "sources_sha256": {
            source: sha256_file(ROOT / source) for source in SOURCES if (ROOT / source).exists()
        },
    }
    (out / "maps_receipt.json").write_text(json.dumps(receipt, indent=2))
    print(json.dumps({k: v for k, v in receipt.items() if k not in ("failed_ids",)}, indent=2))


def reproduce061(baseline: str = "residual056") -> dict:
    """
    Integrity check of Amendment 1: rescore 061's scored float32 ECGAD maps and re-summarize its tables.

    Parameters
    ----------
    baseline : str
        061 baseline to reproduce.

    Returns
    -------
    dict
        Per-record maximum absolute differences, summary differences and ``passed``.
    """
    folder = BASELINE_DIR / "tables" / baseline
    maps = np.load(folder / "ecgad_maps.npy", mmap_mode="r")
    bench = load_benchmark()
    saved = pd.read_csv(folder / "ecgad.csv").set_index("ecg_id")
    columns = ("A2", "A3", "A4", "A6", "marked_share")
    worst = dict.fromkeys(columns, 0.0)
    for position, ecg_id in enumerate(bench.ecgad.ecg_id):
        metrics = ecgad_record_metrics(np.asarray(maps[position]), bench.ecgad_labels[position])
        for column in columns:
            a, b = metrics[column], saved.at[int(ecg_id), column]
            if np.isnan(a) != np.isnan(b):
                worst[column] = np.inf
            elif not np.isnan(a):
                worst[column] = max(worst[column], abs(a - b))
    receipt = json.loads((BASELINE_DIR / "result.json").read_text())["methods"][baseline]["metrics"]
    summary = summarize(load_tables(folder))["metrics"]
    keys = ("value", "ci_low", "ci_high")
    summary_difference = {
        metric: max(abs(summary[metric][key] - receipt[metric][key]) for key in keys)
        for metric in ("A2", "B1/macro", "C1", "D1")
    }
    return {
        "baseline": baseline,
        "ecgad_records": int(len(bench.ecgad)),
        "per_record_max_abs_difference": worst,
        "summary_max_abs_difference": summary_difference,
        "tolerance": TOLERANCE,
        "passed": bool(max(worst.values()) <= TOLERANCE and max(summary_difference.values()) <= TOLERANCE),
    }


def run_score(out: Path) -> None:
    """
    Integrity check, then score every variant with the LocBench CLI and write ``result.json``.

    Parameters
    ----------
    out : Path
        Output directory holding the maps.
    """
    start = time.time()
    integrity = reproduce061()
    print(json.dumps(integrity, indent=2))
    if not integrity["passed"]:
        raise RuntimeError("061 reproduction failed; nothing is scored")
    for name in nm.VARIANTS:
        command = [
            sys.executable,
            "-m",
            "scripts.experiments.run_locbench_score",
            "--maps",
            str(out / f"maps_{name}.npy"),
            "--ids",
            str(out / "ecg_ids.json"),
            "--name",
            name,
            "--out",
            str(out / "locbench" / name),
        ]
        for baseline in BASELINES:
            command += ["--compare", str(BASELINE_DIR / "tables" / baseline)]
        subprocess.run(command, check=True, cwd=ROOT)
    summaries = {
        name: json.loads((out / "locbench" / name / "summary.json").read_text()) for name in nm.VARIANTS
    }
    primary = summaries[nm.PRIMARY]["paired"][BAR]["A2"]
    result = {
        "experiment": "062",
        "protocol": "docs/experiment-062-normative-map.md",
        "code_commit": git_head(ROOT),
        "sources_sha256": {source: sha256_file(ROOT / source) for source in SOURCES},
        "fit": json.loads((out / "fit_summary.json").read_text()),
        "maps": json.loads((out / "maps_receipt.json").read_text()),
        "integrity": integrity,
        "bar": BAR,
        "primary": nm.PRIMARY,
        "primary_A2_difference": primary,
        "beats_bar": bool(primary["ci_low"] > 0),
        "variants": {name: summary["metrics"] for name, summary in summaries.items()},
        "paired": {name: summary["paired"] for name, summary in summaries.items()},
        "benchmark_counts": summaries[nm.PRIMARY]["benchmark_counts"],
        "scoring_seconds": time.time() - start,
    }
    (out / "result.json").write_text(json.dumps(result, indent=2))
    print(json.dumps({"primary_A2_difference": primary, "beats_bar": result["beats_bar"]}, indent=2))


def main() -> None:
    """Parse arguments and run a stage."""
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("fit", "maps", "score"))
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--ids", type=Path, default=None, help="Text file of target ECG IDs, one per line")
    args = parser.parse_args()
    if args.stage == "fit":
        run_fit(args.out, args.limit)
        return
    if args.stage == "score":
        run_score(args.out)
        return
    ids = np.loadtxt(args.ids, dtype=int, ndmin=1) if args.ids else default_targets()
    run_maps(args.out, ids[: args.limit] if args.limit else ids)


if __name__ == "__main__":
    main()
