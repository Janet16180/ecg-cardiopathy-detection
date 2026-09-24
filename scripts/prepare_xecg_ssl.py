"""Build train-only, full-record xECG inputs for the audited 40k MIMIC study."""

import argparse
from concurrent.futures import ThreadPoolExecutor
import csv
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np
import scipy

from ecg_experiment.xecg import preprocess_xecg, sha256
from scripts.prepare_xecg import read_record

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "data/processed/xecg_ssl_40k"
FIELDS = ("ecg_id", "patient_id", "source", "split", "raw_dir", "filename_hr")


def read_csv(path):
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def atomic_json(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def selected_rows():
    pool = ROOT / "data/processed/cpc_pool_40k"
    ptb = ROOT / "data/processed/ptbxl/seed42_fraction1"
    mimic = ROOT / "data/processed/mimic_ssl_40k_cpc"
    pool_info = json.loads((pool / "complete.json").read_text())
    audit = json.loads((mimic / "metadata.json").read_text())
    if sha256(pool / "rows.csv") != pool_info["rows_sha256"]:
        raise ValueError("Audited pool row identities changed")
    if sha256(mimic / "ssl_manifest.csv") != audit["manifest_sha256"]:
        raise ValueError("Audited MIMIC manifest changed")
    if sha256(ptb / "all_train_ssl.csv") != pool_info["ptb_manifest_sha256"]["all_train_ssl.csv"]:
        raise ValueError("PTB training manifest changed")
    lookup = {}
    for row in read_csv(ptb / "all_train_ssl.csv"):
        lookup[row["ecg_id"]] = {"ecg_id": row["ecg_id"], "patient_id": row["patient_id"],
            "source": "ptbxl", "split": "train", "raw_dir": str(ROOT / "data/raw/ptb-xl/1.0.3"),
            "filename_hr": row["filename_hr"]}
    for row in read_csv(mimic / "ssl_manifest.csv"):
        if row["ecg_id"] in lookup:
            raise ValueError("Duplicate source ECG identity")
        lookup[row["ecg_id"]] = {**row, "split": "train"}
    expected = [r for r in read_csv(pool / "rows.csv") if r["split"] == "train"]
    rows = [lookup[r["ecg_id"]] for r in expected]
    if len(rows) != 56875 or len(lookup) != 56875:
        raise ValueError("Expected exactly the audited 56,875 training ECGs")
    for before, after in zip(expected, rows):
        if any(before[k] != after[k] for k in ("ecg_id", "patient_id", "source", "split")):
            raise ValueError("Audited training row mapping changed")
    heldout = {r["patient_id"] for name in ("validation.csv", "test.csv") for r in read_csv(ptb / name)}
    if heldout & {r["patient_id"] for r in rows if r["source"] == "ptbxl"}:
        raise ValueError("Held-out PTB patients entered the SSL pool")
    paths = [pool / "complete.json", pool / "rows.csv", ptb / "all_train_ssl.csv",
             ptb / "validation.csv", ptb / "test.csv", mimic / "ssl_manifest.csv", mimic / "metadata.json",
             Path(__file__), ROOT / "scripts/prepare_xecg.py", ROOT / "ecg_experiment/xecg.py"]
    return rows, {str(p.relative_to(ROOT)): sha256(p) for p in paths}


def prepare(output, workers=4):
    rows, sources = selected_rows()
    identity = {"source_sha256": sources, "scipy_version": scipy.__version__,
                "shape": [len(rows), 1000, 12], "dtype": "float32",
                "preprocessing": "Full 10 seconds, canonical 12 leads, physical mV; official 500-to-100 Hz FFT; no normalization"}
    output.mkdir(parents=True, exist_ok=True)
    row_path = output / "rows.csv"
    if row_path.exists():
        if read_csv(row_path) != [{k: r[k] for k in FIELDS} for r in rows]:
            raise ValueError("Existing xECG SSL rows differ")
    else:
        with row_path.with_suffix(".tmp").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=FIELDS)
            writer.writeheader()
            writer.writerows({k: r[k] for k in FIELDS} for r in rows)
        row_path.with_suffix(".tmp").replace(row_path)
    identity["rows_sha256"] = sha256(row_path)
    complete, progress = output / "metadata.json", output / "progress.json"
    array_path, partial = output / "views.npy", output / "views.partial.npy"
    raw_hash_path = output / "raw_sha256.npy"
    if complete.exists():
        old = json.loads(complete.read_text())
        if any(old.get(k) != v for k, v in identity.items()):
            raise ValueError("Completed SSL cache identity differs")
        if sha256(array_path) != old["views_sha256"] or sha256(raw_hash_path) != old["raw_sha256_file_sha256"]:
            raise ValueError("Completed SSL cache checksum mismatch")
        return old
    if progress.exists():
        saved = json.loads(progress.read_text())
        if saved["identity"] != identity:
            raise ValueError("Partial SSL cache identity differs")
        done = saved["completed_rows"]
        matrix = np.lib.format.open_memmap(partial if partial.exists() else array_path, mode="r+")
        raw_hashes = np.lib.format.open_memmap(raw_hash_path, mode="r+")
        if matrix.shape != tuple(identity["shape"]) or raw_hashes.shape != (len(rows), 2) or not 0 <= done <= len(rows):
            raise ValueError("Malformed partial SSL cache")
    else:
        if partial.exists() or array_path.exists() or raw_hash_path.exists():
            raise ValueError("Partial SSL cache lacks progress record")
        done = 0
        matrix = np.lib.format.open_memmap(partial, mode="w+", dtype=np.float32, shape=tuple(identity["shape"]))
        raw_hashes = np.lib.format.open_memmap(raw_hash_path, mode="w+", dtype="S64", shape=(len(rows), 2))
        atomic_json(progress, {"identity": identity, "completed_rows": 0})

    def decode(row):
        raw_dir = Path(row["raw_dir"])
        base = (raw_dir / row["filename_hr"]).resolve()
        if not base.is_relative_to(raw_dir.resolve()):
            raise ValueError("Unsafe source path")
        hashes = [sha256(base.with_suffix(suffix)) for suffix in (".hea", ".dat")]
        return preprocess_xecg(read_record(raw_dir, row["filename_hr"])), hashes

    started = time.monotonic()
    with ThreadPoolExecutor(max_workers=workers) as executor:
        for start in range(done, len(rows), 128):
            stop = min(start + 128, len(rows))
            for index, (view, hashes) in enumerate(executor.map(decode, rows[start:stop]), start):
                matrix[index], raw_hashes[index] = view, hashes
            matrix.flush()
            raw_hashes.flush()
            atomic_json(progress, {"identity": identity, "completed_rows": stop})
            if stop % 1024 == 0 or stop == len(rows):
                print(json.dumps({"stage": "xecg_ssl_cache", "completed_rows": stop, "total_rows": len(rows),
                                  "seconds_this_run": time.monotonic() - started}), flush=True)
    del matrix, raw_hashes
    if partial.exists():
        os.replace(partial, array_path)
    _, final_sources = selected_rows()
    if final_sources != sources:
        raise ValueError("Sources changed during xECG SSL preparation")
    result = {**identity, "record_count": len(rows), "all_train_only": True,
              "ecg_ids": [r["ecg_id"] for r in rows], "patient_ids": [r["patient_id"] for r in rows],
              "sources": [r["source"] for r in rows], "views_sha256": sha256(array_path),
              "raw_sha256_file_sha256": sha256(raw_hash_path)}
    atomic_json(complete, result)
    progress.unlink()
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--workers", type=int, choices=range(1, 9), default=4)
    args = parser.parse_args()
    info = prepare(args.output_dir, args.workers)
    print(json.dumps({"complete": True, "records": info["record_count"], "views_sha256": info["views_sha256"]}), flush=True)
