"""Build train-only, full-record xECG inputs for the audited 40k MIMIC study."""

from __future__ import annotations

import argparse
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
import scipy

from ecg_experiment.files import read_csv, sha256_file, write_csv_atomic, write_json_atomic
from ecg_experiment.xecg import preprocess_xecg
from scripts.data.prepare_xecg import VIEW_SHAPE, read_record_float64

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = ROOT / "data/processed/xecg_ssl_40k"
POOL = ROOT / "data/processed/cpc_pool_40k"
PTB = ROOT / "data/processed/ptbxl/seed42_fraction1"
MIMIC = ROOT / "data/processed/mimic_ssl_40k_cpc"
FIELDS = ("ecg_id", "patient_id", "source", "split", "raw_dir", "filename_hr")
TRAINING_RECORDS = 56875
BATCH_ROWS = 128
PROGRESS_INTERVAL = 1024


def _check_audited_inputs(pool_info: dict[str, Any], audit: dict[str, Any]) -> None:
    if sha256_file(POOL / "rows.csv") != pool_info["rows_sha256"]:
        raise ValueError("Audited pool row identities changed")
    if sha256_file(MIMIC / "ssl_manifest.csv") != audit["manifest_sha256"]:
        raise ValueError("Audited MIMIC manifest changed")
    if sha256_file(PTB / "all_train_ssl.csv") != pool_info["ptb_manifest_sha256"]["all_train_ssl.csv"]:
        raise ValueError("PTB training manifest changed")


def _source_lookup() -> dict[str, dict[str, str]]:
    """Map each training ECG ID to its full source row."""
    lookup = {}
    for row in read_csv(PTB / "all_train_ssl.csv"):
        lookup[row["ecg_id"]] = {"ecg_id": row["ecg_id"], "patient_id": row["patient_id"],
                                 "source": "ptbxl", "split": "train",
                                 "raw_dir": str(ROOT / "data/raw/ptb-xl/1.0.3"),
                                 "filename_hr": row["filename_hr"]}
    for row in read_csv(MIMIC / "ssl_manifest.csv"):
        if row["ecg_id"] in lookup:
            raise ValueError("Duplicate source ECG identity")
        lookup[row["ecg_id"]] = {**row, "split": "train"}
    return lookup


def _check_row_mapping(expected: list[dict[str, str]], rows: list[dict[str, str]]) -> None:
    for before, after in zip(expected, rows, strict=True):
        if any(before[key] != after[key] for key in ("ecg_id", "patient_id", "source", "split")):
            raise ValueError("Audited training row mapping changed")


def selected_rows() -> tuple[list[dict[str, str]], dict[str, str]]:
    """
    Return the audited training rows in pool order, and the hashes of their sources.

    Returns
    -------
    tuple[list[dict[str, str]], dict[str, str]]
        Rows with :data:`FIELDS` keys, and SHA-256 keyed by repository-relative
        path of every input and code file the cache depends on.

    Raises
    ------
    ValueError
        If an audited input changed, the pool size differs, or held-out PTB-XL
        patients appear in the pool.
    """
    pool_info = json.loads((POOL / "complete.json").read_text())
    audit = json.loads((MIMIC / "metadata.json").read_text())
    _check_audited_inputs(pool_info, audit)
    lookup = _source_lookup()
    expected = [row for row in read_csv(POOL / "rows.csv") if row["split"] == "train"]
    rows = [lookup[row["ecg_id"]] for row in expected]
    if len(rows) != TRAINING_RECORDS or len(lookup) != TRAINING_RECORDS:
        raise ValueError("Expected exactly the audited 56,875 training ECGs")
    _check_row_mapping(expected, rows)
    heldout = {row["patient_id"] for name in ("validation.csv", "test.csv") for row in read_csv(PTB / name)}
    if heldout & {row["patient_id"] for row in rows if row["source"] == "ptbxl"}:
        raise ValueError("Held-out PTB patients entered the SSL pool")
    paths = [POOL / "complete.json", POOL / "rows.csv", PTB / "all_train_ssl.csv",
             PTB / "validation.csv", PTB / "test.csv", MIMIC / "ssl_manifest.csv", MIMIC / "metadata.json",
             Path(__file__), ROOT / "scripts/data/prepare_xecg.py", ROOT / "ecg_experiment/xecg.py"]
    return rows, {str(path.relative_to(ROOT)): sha256_file(path) for path in paths}


def _write_rows(row_path: Path, rows: list[dict[str, str]]) -> None:
    """Write ``rows.csv`` once; afterwards require it to be unchanged."""
    projected = [{key: row[key] for key in FIELDS} for row in rows]
    if row_path.exists():
        if read_csv(row_path) != projected:
            raise ValueError("Existing xECG SSL rows differ")
        return
    write_csv_atomic(row_path, projected, FIELDS)


def _verified_completion(output: Path, identity: dict[str, Any]) -> dict[str, Any]:
    old = json.loads((output / "metadata.json").read_text())
    if any(old.get(key) != value for key, value in identity.items()):
        raise ValueError("Completed SSL cache identity differs")
    if (sha256_file(output / "views.npy") != old["views_sha256"] or
            sha256_file(output / "raw_sha256.npy") != old["raw_sha256_file_sha256"]):
        raise ValueError("Completed SSL cache checksum mismatch")
    return old


def _views_path(output: Path) -> Path:
    """Return the partial array, or the final one if it was already renamed."""
    partial = output / "views.partial.npy"
    return partial if partial.exists() else output / "views.npy"


def _resume_position(output: Path, identity: dict[str, Any], count: int) -> int:
    """Validate a checkpoint and return its position, or create new arrays."""
    progress = output / "progress.json"
    partial, array_path = output / "views.partial.npy", output / "views.npy"
    raw_hash_path = output / "raw_sha256.npy"
    if not progress.exists():
        if partial.exists() or array_path.exists() or raw_hash_path.exists():
            raise ValueError("Partial SSL cache lacks progress record")
        shape = tuple(identity["shape"])
        np.lib.format.open_memmap(partial, mode="w+", dtype=np.float32, shape=shape).flush()
        np.lib.format.open_memmap(raw_hash_path, mode="w+", dtype="S64", shape=(count, 2)).flush()
        write_json_atomic(progress, {"identity": identity, "completed_rows": 0})
        return 0
    saved = json.loads(progress.read_text())
    if saved["identity"] != identity:
        raise ValueError("Partial SSL cache identity differs")
    done = saved["completed_rows"]
    matrix = np.lib.format.open_memmap(_views_path(output), mode="r+")
    raw_hashes = np.lib.format.open_memmap(raw_hash_path, mode="r+")
    if matrix.shape != tuple(identity["shape"]) or raw_hashes.shape != (count, 2) or not 0 <= done <= count:
        raise ValueError("Malformed partial SSL cache")
    return done


def _decode(row: dict[str, str]) -> tuple[np.ndarray, list[str]]:
    """Return the preprocessed view and the SHA-256 of the record's header and data files."""
    raw_dir = Path(row["raw_dir"])
    base = (raw_dir / row["filename_hr"]).resolve()
    if not base.is_relative_to(raw_dir.resolve()):
        raise ValueError("Unsafe source path")
    hashes = [sha256_file(base.with_suffix(suffix)) for suffix in (".hea", ".dat")]
    return preprocess_xecg(read_record_float64(raw_dir, row["filename_hr"])), hashes


def _fill(output: Path, rows: list[dict[str, str]], done: int, workers: int,
          identity: dict[str, Any]) -> None:
    """Decode rows from ``done`` onward, checkpointing after every batch."""
    matrix = np.lib.format.open_memmap(_views_path(output), mode="r+")
    raw_hashes = np.lib.format.open_memmap(output / "raw_sha256.npy", mode="r+")
    started = time.monotonic()
    with ThreadPoolExecutor(max_workers=workers) as executor:
        for start in range(done, len(rows), BATCH_ROWS):
            stop = min(start + BATCH_ROWS, len(rows))
            for index, (view, hashes) in enumerate(executor.map(_decode, rows[start:stop]), start):
                matrix[index], raw_hashes[index] = view, hashes
            matrix.flush()
            raw_hashes.flush()
            write_json_atomic(output / "progress.json", {"identity": identity, "completed_rows": stop})
            if stop % PROGRESS_INTERVAL == 0 or stop == len(rows):
                print(json.dumps({"stage": "xecg_ssl_cache", "completed_rows": stop, "total_rows": len(rows),
                                  "seconds_this_run": time.monotonic() - started}), flush=True)


def prepare(output: Path, workers: int = 4) -> dict[str, Any]:
    """
    Build or resume the train-only xECG SSL view cache.

    Parameters
    ----------
    output : Path
        Cache directory.
    workers : int
        Decoding threads.

    Returns
    -------
    dict[str, Any]
        Completion metadata, including row identities and array hashes.

    Raises
    ------
    ValueError
        If existing files or sources differ from the audited inputs.
    """
    rows, sources = selected_rows()
    identity = {"source_sha256": sources, "scipy_version": scipy.__version__,
                "shape": [len(rows), *VIEW_SHAPE], "dtype": "float32",
                "preprocessing": ("Full 10 seconds, canonical 12 leads, physical mV; "
                                  "official 500-to-100 Hz FFT; no normalization")}
    output.mkdir(parents=True, exist_ok=True)
    row_path = output / "rows.csv"
    _write_rows(row_path, rows)
    identity["rows_sha256"] = sha256_file(row_path)
    if (output / "metadata.json").exists():
        return _verified_completion(output, identity)
    done = _resume_position(output, identity, len(rows))
    _fill(output, rows, done, workers, identity)
    partial, array_path = output / "views.partial.npy", output / "views.npy"
    if partial.exists():
        os.replace(partial, array_path)
    _, final_sources = selected_rows()
    if final_sources != sources:
        raise ValueError("Sources changed during xECG SSL preparation")
    result = {**identity, "record_count": len(rows), "all_train_only": True,
              "ecg_ids": [row["ecg_id"] for row in rows],
              "patient_ids": [row["patient_id"] for row in rows],
              "sources": [row["source"] for row in rows], "views_sha256": sha256_file(array_path),
              "raw_sha256_file_sha256": sha256_file(output / "raw_sha256.npy")}
    write_json_atomic(output / "metadata.json", result)
    (output / "progress.json").unlink()
    return result


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """
    Parse the command line.

    Parameters
    ----------
    argv : list[str] | None
        Arguments, or ``None`` for ``sys.argv``.

    Returns
    -------
    argparse.Namespace
        Parsed arguments.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--workers", type=int, choices=range(1, 9), default=4)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    """
    Build the cache from the command line and print a short completion line.

    Parameters
    ----------
    argv : list[str] | None
        Arguments, or ``None`` for ``sys.argv``.
    """
    args = parse_args(argv)
    info = prepare(args.output_dir, args.workers)
    print(json.dumps({"complete": True, "records": info["record_count"],
                      "views_sha256": info["views_sha256"]}), flush=True)


if __name__ == "__main__":
    main()
