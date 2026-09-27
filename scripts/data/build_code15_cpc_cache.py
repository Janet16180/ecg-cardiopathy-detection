"""Stream the verified CODE-15% archive once and write full-length tracings as 250 Hz CPC inputs."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

from ecg_experiment.code15_cpc import LABELS, central_window, patient_split, to_cpc, window_quality
from ecg_experiment.eda.code15 import EXPORT_ARCHIVE, extracted_parts
from ecg_experiment.files import sha256_file, write_json_atomic

ROOT = Path(__file__).resolve().parents[2]
EXAMS = ROOT / "data/raw/code-15pct/zenodo-4916206/exams.csv"
EXPORT = ROOT / "outputs/data_export"
MONITORING_FRACTION = 0.05
SPLIT_SEED = 21001
CHUNK = 500


def verify_sources() -> dict[str, str]:
    """Check the archive and exams.csv against their recorded checksums."""
    recorded = (EXPORT / "CODE_SHA256SUMS").read_text().split()[0]
    archive = sha256_file(EXPORT_ARCHIVE)
    if archive != recorded:
        raise ValueError("CODE-15 archive differs from its recorded SHA-256")
    receipt = json.loads((ROOT / "data/acquisition/code_15pct.json").read_text())
    expected = next(item["checksum"] for item in receipt["verified_files"] if item["name"] == "exams.csv")
    exams = "md5:" + hashlib.md5(EXAMS.read_bytes()).hexdigest()
    if exams != expected:
        raise ValueError("exams.csv differs from the Zenodo checksum")
    return {"archive_sha256": archive, "exams_csv": exams}


def convert_part(path: Path, part: int, output: Path) -> list[dict[str, object]]:
    """Write one part's full-length tracings to a shard and describe each row."""
    rows, signals = [], []
    with h5py.File(path, "r") as handle:
        ids, tracings = handle["exam_id"][:], handle["tracings"]
        for start in range(0, len(ids), CHUNK):
            for offset, tracing in enumerate(tracings[start:start + CHUNK]):
                window = central_window(tracing)
                if ids[start + offset] == 0 or window is None:
                    continue
                rows.append({"exam_id": int(ids[start + offset]), "shard": f"part{part}.npy",
                             "index": len(signals), **window_quality(window)})
                signals.append(to_cpc(np.nan_to_num(window)))
    np.save(output / f"part{part}.npy", np.stack(signals))
    return rows


def build(output: Path) -> dict[str, object]:
    """Convert every part, join labels and patients, and write the receipt."""
    if output.exists():
        raise ValueError(f"Destination exists: {output}")
    start = time.monotonic()
    sources = verify_sources()
    stage = output.with_name(output.name + ".partial")
    stage.mkdir(parents=True, exist_ok=False)
    rows = []
    for part, path in extracted_parts():
        rows += convert_part(path, part, stage)
        print(json.dumps({"part": part, "rows": len(rows), "seconds": time.monotonic() - start}), flush=True)

    exams = pd.read_csv(EXAMS).set_index("exam_id")
    table = pd.DataFrame(rows).join(exams[[*LABELS, "patient_id", "age", "is_male"]], on="exam_id",
                                    validate="1:1")
    table[list(LABELS)] = table[list(LABELS)].astype(int)
    table["split"] = patient_split(table["patient_id"], MONITORING_FRACTION, SPLIT_SEED)
    table.to_csv(stage / "rows.csv", index=False)
    shards = sorted(stage.glob("part*.npy"))
    metadata = {
        "complete": True, "sources": sources, "records": len(table),
        "patients": int(table["patient_id"].nunique()),
        "split_counts": table["split"].value_counts().to_dict(),
        "quality_counts": {name: int(table[name].sum())
                           for name in ("nonfinite", "constant_lead", "flat_segment")},
        "label_counts": {name: int(table[name].sum()) for name in LABELS},
        "transform": "central 4,000 active 400 Hz samples; each 5 s half resample_poly(5, 8); native units",
        "split_seed": SPLIT_SEED, "monitoring_fraction": MONITORING_FRACTION,
        "shards": {path.name: sha256_file(path) for path in shards},
        "rows_sha256": sha256_file(stage / "rows.csv"),
        "source_sha256": {name: sha256_file(ROOT / name) for name in (
            "ecg_experiment/code15_cpc.py", "ecg_experiment/eda/code15.py", "ecg_experiment/ecg_quality.py",
            "scripts/data/build_code15_cpc_cache.py")},
        "build_seconds": time.monotonic() - start,
    }
    write_json_atomic(stage / "metadata.json", metadata, sort_keys=True)
    stage.rename(output)
    return metadata


def main() -> None:
    """Parse the output directory and build the cache."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "data/processed/code15_cpc_250hz_v1")
    args = parser.parse_args()
    metadata = build(args.output_dir)
    print(json.dumps({key: metadata[key] for key in ("records", "patients", "split_counts",
                                                     "quality_counts", "label_counts")}, indent=2))


if __name__ == "__main__":
    main()
