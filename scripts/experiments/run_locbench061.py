"""
Experiment 061: LocBench v1 and baselines.

Runs the integrity checks, refits U_B on the 2,666 ``fit`` clean normals, maps every scored record with the
five baselines, scores them with ``ecg_experiment.locbench`` and writes
``outputs/experiment061_locbench_v1/``.
CPU only, at most 8 worker processes. ``--smoke`` profiles the maps on 40 ``fit`` records and scores nothing.

    OPENBLAS_CORETYPE=Haswell PYTHONPATH=. .venv/bin/python -m scripts.experiments.run_locbench061
"""

from __future__ import annotations

import argparse
import hashlib
import logging
import os
import pickle
import re
import subprocess
import time
from multiprocessing import get_context
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from ecg_experiment import ROOT
from ecg_experiment.eda.ptbxl import load_metadata
from ecg_experiment.external_encoders import read_ptb_float64
from ecg_experiment.files import sha256_file, write_json_atomic
from ecg_experiment.lead_wave_maps import fit_wave_references
from ecg_experiment.locbench import (
    START,
    STOP,
    build_benchmark,
    evaluate_method,
    fiducial_path,
    paired_difference,
    save_maps,
    save_tables,
    scored_ecg_ids,
)
from ecg_experiment.locbench_baselines061 import (
    METHODS,
    amplitude_map,
    collect_pieces,
    exact_signal,
    qrs_prior_map,
    random_map,
    reproduce042,
    reproduce056,
    residual_map,
    ub_map,
)
from ecg_experiment.locbench_cache import CACHE_DIR, open_cache
from ecg_experiment.locbench_data import load_partition
from ecg_experiment.paths import to_stored
from ecg_experiment.provenance import git_head, utc_now
from ecg_experiment.waveforms import LEADS

OUTPUT = ROOT / "outputs/experiment061_locbench_v1"
WORK = Path(os.environ.get("LOCBENCH061_WORK", "/tmp/ecg-locbench061-work"))
PROTOCOL = "docs/experiment-061-locbench.md"
ECGAD_DOC = "docs/ecgad-benchmark-v1.md"
SOURCES = (
    PROTOCOL,
    ECGAD_DOC,
    "docs/localization-program-2026-10-12.md",
    "ecg_experiment/locbench.py",
    "ecg_experiment/locbench_baselines061.py",
    "ecg_experiment/locbench_data.py",
    "ecg_experiment/locbench_cache.py",
    "ecg_experiment/ecgad.py",
    "ecg_experiment/intervals.py",
    "ecg_experiment/lead_wave_maps.py",
    "ecg_experiment/raw_residual052.py",
    "ecg_experiment/raw_residual054.py",
    "ecg_experiment/incart_localization056.py",
    "ecg_experiment/hybrid_boundaries059_v2.py",
    "ecg_experiment/morphology_boundaries.py",
    "ecg_experiment/fragment_localization.py",
    "ecg_experiment/full_development.py",
    "ecg_experiment/external_encoders.py",
    "ecg_experiment/eda/ptbxl.py",
    "ecg_experiment/waveforms.py",
    "scripts/experiments/run_locbench061.py",
    "scripts/experiments/run_locbench_score.py",
    "pyproject.toml",
    "uv.lock",
)
INPUTS = (
    "data/raw/ecgad/benchmark_label.npy",
    "data/raw/ecgad/ecgad_ptbxl_match.csv",
    "docs/ecgad-split-v1.json",
    "docs/locbench-partition-v1.json",
    "data/raw/ptb-xl/1.0.3/ptbxl_database.csv",
    "data/raw/ptb-xl/1.0.3/scp_statements.csv",
    "data/raw/ptb-xl-plus/1.0.1/SHA256SUMS.txt",
    "data/raw/incart/1.0.0/I01.dat",
    "data/raw/incart/1.0.0/I01.hea",
    "data/raw/incart/1.0.0/I02.dat",
    "data/raw/incart/1.0.0/I02.hea",
    "data/raw/incart/1.0.0/I03.dat",
    "data/raw/incart/1.0.0/I03.hea",
)
ECGAD_HASHED = ("benchmark_label.npy", "ecgad_ptbxl_match.csv")
WORKERS = 8
CHUNK = 32
SMOKE_RECORDS = 40
SIGNAL_CHECK_RECORDS = 20
RANDOM_A2_RANGE = (0.48, 0.52)
BAR_METRICS = ("A2", "B1/macro", "C1", "D1")
INFERIOR = ("II", "III", "aVF")
ANTERIOR = ("V1", "V2", "V3", "V4")
LOG = logging.getLogger("experiment061")

_STATE: dict[str, Any] = {}


def identity() -> dict[str, Any]:
    """
    Hashes of every source and input, code commit and tracked modifications.

    Returns
    -------
    dict[str, Any]
        Receipt identity block.

    Raises
    ------
    ValueError
        If the ECGAD labels or match table differ from ``docs/ecgad-benchmark-v1.md``.
    """
    inputs = {path: sha256_file(ROOT / path) for path in INPUTS}
    recorded = (ROOT / ECGAD_DOC).read_text()
    for name in ECGAD_HASHED:
        expected = re.search(rf"`{re.escape(name)}` \| `([0-9a-f]{{64}})`", recorded).group(1)
        if inputs[f"data/raw/ecgad/{name}"] != expected:
            raise ValueError(f"{name} differs from the hash in {ECGAD_DOC}")
    dirty = subprocess.run(
        ["git", "-C", str(ROOT), "status", "--porcelain", "--untracked-files=no"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.split("\n")
    return {
        "code_commit": git_head(ROOT),
        "tracked_modifications": [line for line in dirty if line],
        "sources_sha256": {path: sha256_file(ROOT / path) for path in SOURCES},
        "inputs_sha256": inputs,
        "cache": {
            "directory": str(CACHE_DIR),
            "index_sha256": sha256_file(CACHE_DIR / "index.json"),
            "signals_sha256": sha256_file(CACHE_DIR / "signals.npy"),
        },
        "started": utc_now(),
    }


def fiducial_digest(ecg_ids: np.ndarray) -> dict[str, Any]:
    """
    Hash the ECGdeli files of the region set into one SHA-256.

    Parameters
    ----------
    ecg_ids : np.ndarray
        Region-set records.

    Returns
    -------
    dict[str, Any]
        File count and the SHA-256 of the sorted ``name sha256`` lines.
    """
    lines = sorted(
        f"{fiducial_path(int(ecg_id)).name}.atr {sha256_file(str(fiducial_path(int(ecg_id))) + '.atr')}"
        for ecg_id in ecg_ids
    )
    return {"files": len(lines), "sha256": hashlib.sha256("\n".join(lines).encode()).hexdigest()}


def check_signals(signals: np.ndarray, index: pd.DataFrame) -> dict[str, Any]:
    """
    Compare the exact float64 recovery from the cache with ``wfdb.rdsamp`` on spread records.

    Parameters
    ----------
    signals : np.ndarray
        Cache signals.
    index : pd.DataFrame
        Cache index.

    Returns
    -------
    dict[str, Any]
        Checked records and ``passed``.
    """
    meta = load_metadata()
    positions = np.linspace(0, len(index) - 1, SIGNAL_CHECK_RECORDS).astype(int)
    equal = [
        bool(
            np.array_equal(
                exact_signal(signals[position]),
                read_ptb_float64(meta.at[int(index.ecg_id[position]), "filename_hr"]),
            )
        )
        for position in positions
    ]
    return {"records": index.ecg_id[positions].astype(int).tolist(), "passed": all(equal)}


def fit_references(signals: np.ndarray, index: pd.DataFrame) -> tuple[dict, dict[str, Any]]:
    """
    Refit the 48 U_B references on every kept beat of the ``fit`` clean normals.

    Parameters
    ----------
    signals : np.ndarray
        Cache signals.
    index : pd.DataFrame
        Cache index.

    Returns
    -------
    tuple[dict, dict[str, Any]]
        References and counts.
    """
    partition = load_partition()
    fit = partition[(partition["partition"] == "fit") & partition["clean_normal"]]
    position = pd.Series(np.arange(len(index)), index=index["ecg_id"].to_numpy())
    pieces, beats, dropped = collect_pieces([exact_signal(signals[int(position[e])]) for e in fit["ecg_id"]])
    references = fit_wave_references(pieces)
    return references, {
        "records": len(fit),
        "patients": int(fit["patient_id"].nunique()),
        "kept_beats": beats,
        "records_without_two_beats": dropped,
    }


def _worker_init(references: dict, ecg_ids: np.ndarray, work: str) -> None:
    """Open the cache and the output maps once per worker, with one BLAS thread."""
    threadpool_limits(limits=1)
    signals, index = open_cache()
    _STATE["signals"] = signals
    _STATE["position"] = pd.Series(np.arange(len(index)), index=index["ecg_id"].to_numpy())
    _STATE["references"] = references
    _STATE["ecg_ids"] = ecg_ids
    _STATE["maps"] = {name: np.load(Path(work) / f"{name}.npy", mmap_mode="r+") for name in METHODS}


def _map_rows(rows: list[int]) -> list[dict[str, Any]]:
    """Compute the five baseline maps of some rows and write them; return per-row build notes."""
    notes = []
    for row in rows:
        ecg_id = int(_STATE["ecg_ids"][row])
        signal = exact_signal(_STATE["signals"][int(_STATE["position"][ecg_id])])
        references = _STATE["references"]
        amplitude = amplitude_map(signal)
        prior, intervals = qrs_prior_map(signal)
        ub = ub_map(signal, references)
        residual = residual_map(signal, references)
        built = {
            "random": random_map(ecg_id),
            "amplitude": amplitude,
            "qrs_prior": prior,
            "U_B": np.zeros_like(signal) if ub is None else ub,
            "residual056": np.zeros_like(signal) if residual is None else residual,
        }
        for name, values in built.items():
            if not np.isfinite(values).all():
                raise ValueError(f"Nonfinite {name} map for {ecg_id}")
            _STATE["maps"][name][row] = values.astype(np.float32)
        notes.append(
            {
                "ecg_id": ecg_id,
                "qrs_intervals": intervals,
                "U_B_constant": ub is None,
                "residual056_constant": residual is None,
            }
        )
    for values in _STATE["maps"].values():
        values.flush()
    return notes


def build_maps(ecg_ids: np.ndarray, references: dict, work: Path) -> pd.DataFrame:
    """
    Compute every baseline map of the given records into float32 memmaps ``<work>/<method>.npy``.

    Parameters
    ----------
    ecg_ids : np.ndarray
        Records, in row order.
    references : dict
        U_B references.
    work : Path
        Local scratch folder.

    Returns
    -------
    pd.DataFrame
        Per-record build notes.
    """
    work.mkdir(parents=True, exist_ok=True)
    for name in METHODS:
        np.lib.format.open_memmap(
            work / f"{name}.npy", mode="w+", dtype=np.float32, shape=(len(ecg_ids), len(LEADS), 5000)
        ).flush()
    chunks = [list(range(start, min(start + CHUNK, len(ecg_ids)))) for start in range(0, len(ecg_ids), CHUNK)]
    notes = []
    with get_context("fork").Pool(WORKERS, _worker_init, (references, ecg_ids, str(work))) as pool:
        for done, found in enumerate(pool.imap(_map_rows, chunks), start=1):
            notes.extend(found)
            if done % 20 == 0:
                LOG.info("mapped %d/%d records", min(done * CHUNK, len(ecg_ids)), len(ecg_ids))
    return pd.DataFrame(notes)


def sanity_reading(bench: Any) -> dict[str, Any]:
    """
    Share of ECGAD development marks in inferior and anterior leads for marked IMI and ASMI records.

    Parameters
    ----------
    bench : Benchmark
        Benchmark.

    Returns
    -------
    dict[str, Any]
        Per statement: records, mean share of marked lead-samples in inferior and in anterior leads.
    """
    meta = load_metadata()
    inferior = np.isin(LEADS, INFERIOR)
    anterior = np.isin(LEADS, ANTERIOR)
    result = {}
    for code in ("IMI", "ASMI"):
        shares = []
        for row, ecg_id in enumerate(bench.ecgad["ecg_id"]):
            marks = bench.ecgad_labels[row][:, START:STOP]
            if marks.any() and meta.at[int(ecg_id), "scp_codes"].get(code, 0) >= 50:
                total = marks.sum()
                shares.append((marks[inferior].sum() / total, marks[anterior].sum() / total))
        shares = np.array(shares)
        result[code] = {
            "records": len(shares),
            "inferior_share": float(shares[:, 0].mean()),
            "anterior_share": float(shares[:, 1].mean()),
            "inferior_lead_fraction": float(inferior.mean()),
            "anterior_lead_fraction": float(anterior.mean()),
        }
    return result


def smoke(signals: np.ndarray, index: pd.DataFrame, references: dict, work: Path) -> dict[str, Any]:
    """
    Time the maps on 40 ``fit`` records that are not clean normals; nothing is scored.

    Parameters
    ----------
    signals : np.ndarray
        Cache signals.
    index : pd.DataFrame
        Cache index.
    references : dict
        U_B references.
    work : Path
        Scratch folder.

    Returns
    -------
    dict[str, Any]
        Seconds per record and build notes.
    """
    partition = load_partition()
    rows = partition[(partition["partition"] == "fit") & ~partition["clean_normal"]]
    ecg_ids = rows["ecg_id"].to_numpy()[:: len(rows) // SMOKE_RECORDS][:SMOKE_RECORDS]
    began = time.perf_counter()
    notes = build_maps(ecg_ids, references, work / "smoke")
    seconds = time.perf_counter() - began
    return {
        "records": len(ecg_ids),
        "seconds": seconds,
        "seconds_per_record_per_worker": seconds * WORKERS / len(ecg_ids),
        "U_B_constant": int(notes["U_B_constant"].sum()),
        "residual056_constant": int(notes["residual056_constant"].sum()),
        "min_qrs_intervals": int(notes["qrs_intervals"].min()),
    }


def run(smoke_only: bool) -> None:
    """
    Run the experiment end to end.

    Parameters
    ----------
    smoke_only : bool
        Profile the maps on 40 ``fit`` records, write a smoke receipt and stop.
    """
    output = WORK / "smoke_output" if smoke_only else OUTPUT
    partial = output.with_name(output.name + ".partial")
    if output.exists() or partial.exists():
        raise FileExistsError(output)
    partial.mkdir(parents=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(message)s",
        handlers=[logging.FileHandler(partial / "run.log"), logging.StreamHandler()],
    )
    began = time.perf_counter()
    result: dict[str, Any] = {"experiment": "061", "protocol": PROTOCOL, "smoke": smoke_only}
    result["identity"] = identity()
    LOG.info("identity hashed; commit %s", result["identity"]["code_commit"])
    signals, index = open_cache()
    integrity: dict[str, Any] = {"signal_recovery": check_signals(signals, index)}
    LOG.info("signal recovery %s", integrity["signal_recovery"]["passed"])
    with threadpool_limits(limits=WORKERS):
        integrity["reproduce042"] = reproduce042(signals, index)
        LOG.info("042 reproduction %s", integrity["reproduce042"]["passed"])
        integrity["reproduce056"] = reproduce056()
        LOG.info("056 reproduction %s", integrity["reproduce056"]["passed"])
        result["integrity"] = integrity
        failed = [name for name, check in integrity.items() if not check["passed"]]
        if failed:
            write_json_atomic(partial / "result.json", {**result, "status": f"stopped: {failed} failed"})
            raise RuntimeError(f"Integrity checks failed: {failed}")
        references, result["U_B_fit"] = fit_references(signals, index)
        LOG.info("U_B refitted: %s", result["U_B_fit"])
    with (partial / "ub_references.pkl").open("wb") as target:
        pickle.dump(references, target)
    if smoke_only:
        result["smoke_profile"] = smoke(signals, index, references, WORK)
        result["seconds"] = time.perf_counter() - began
        write_json_atomic(partial / "result.json", result)
        os.replace(partial, output)
        LOG.info("smoke done: %s", result["smoke_profile"])
        return

    bench = build_benchmark()
    ecg_ids = scored_ecg_ids(bench)
    result["counts"] = {**bench.counts, "scored_records": len(ecg_ids)}
    result["identity"]["region_fiducials"] = fiducial_digest(bench.region["ecg_id"].to_numpy())
    LOG.info("benchmark built: %d scored records", len(ecg_ids))
    started = time.perf_counter()
    notes = build_maps(ecg_ids, references, WORK / "maps")
    result["map_build"] = {
        "seconds": time.perf_counter() - started,
        "U_B_constant_records": notes.loc[notes["U_B_constant"], "ecg_id"].astype(int).tolist(),
        "residual056_constant_records": notes.loc[notes["residual056_constant"], "ecg_id"]
        .astype(int)
        .tolist(),
        "qrs_prior_records_without_interval": notes.loc[notes["qrs_intervals"] == 0, "ecg_id"]
        .astype(int)
        .tolist(),
    }
    notes.to_csv(partial / "map_build.csv", index=False)
    LOG.info("maps built in %.0f s", result["map_build"]["seconds"])

    display = np.unique(
        np.concatenate([bench.ecgad["ecg_id"], bench.region["ecg_id"], bench.noise["ecg_id"]])
    )
    display_rows = np.searchsorted(ecg_ids, display)
    summaries, tables = {}, {}
    with threadpool_limits(limits=WORKERS):
        for name in METHODS:
            maps = np.load(WORK / "maps" / f"{name}.npy", mmap_mode="r")
            summaries[name], tables[name] = evaluate_method(maps, ecg_ids, name, bench)
            save_tables(tables[name], partial / "tables" / name)
            save_maps(partial / "maps" / f"{name}.npy", maps[display_rows], display, dtype=np.float16)
            metrics = summaries[name]["metrics"]
            LOG.info(
                "%s: A2 %.4f, B1 macro %.4f, C1 %.4f, D1 %.4f",
                name,
                metrics["A2"]["value"],
                metrics["B1/macro"]["value"],
                metrics["C1"]["value"],
                metrics["D1"]["value"],
            )
    random_a2 = summaries["random"]["metrics"]["A2"]["value"]
    integrity["random_A2"] = {
        "value": random_a2,
        "range": list(RANDOM_A2_RANGE),
        "passed": RANDOM_A2_RANGE[0] <= random_a2 <= RANDOM_A2_RANGE[1],
    }
    if not integrity["random_A2"]["passed"]:
        write_json_atomic(partial / "result.json", {**result, "status": "stopped: random A2 out of range"})
        raise RuntimeError(f"random A2 {random_a2} outside {RANDOM_A2_RANGE}; metric code is wrong")
    integrity["ecgad_sanity_reading"] = sanity_reading(bench)

    bar = max(METHODS, key=lambda name: summaries[name]["metrics"]["A2"]["value"])
    result["bar"] = {"method": bar, **{metric: summaries[bar]["metrics"][metric] for metric in BAR_METRICS}}
    result["paired_against_bar"] = {
        name: {metric: paired_difference(tables[name], tables[bar], metric) for metric in BAR_METRICS}
        for name in METHODS
        if name != bar
    }
    result["methods"] = summaries
    result["outputs"] = {
        "tables": {name: to_stored(OUTPUT / "tables" / name) for name in METHODS},
        "maps": {name: to_stored(OUTPUT / "maps" / f"{name}.npy") for name in METHODS},
        "maps_note": (
            "float16 maps of the ECGAD development, region and noise records; the float32 ECGAD maps that "
            "were scored are tables/<method>/ecgad_maps.npy; detection-only records keep lead and record "
            "scores in tables/<method>/records.csv."
        ),
        "ub_references": to_stored(OUTPUT / "ub_references.pkl"),
    }
    result["adopted"] = all(check.get("passed", True) for check in integrity.values())
    result["status"] = "complete"
    result["seconds"] = time.perf_counter() - began
    result["finished"] = utc_now()
    write_json_atomic(partial / "result.json", result)
    os.replace(partial, output)
    LOG.info("done in %.0f s; bar %s", result["seconds"], bar)


def main() -> None:
    """Parse arguments and run."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true")
    run(parser.parse_args().smoke)


if __name__ == "__main__":
    main()
