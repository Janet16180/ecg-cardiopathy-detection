"""Verify the EchoNext release and write clean-flagged train and val waveforms as CPC-ready arrays.

The test waveforms are hash checked only; they are never decoded.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from ecg_experiment import echonext
from ecg_experiment.eda.echonext import (
    ARCHIVE,
    COMPONENTS,
    OPEN_SPLITS,
    checksums,
    load_metadata,
    member_sha256,
    waveform_chunks,
)
from ecg_experiment.files import sha256_file, write_json_atomic

ROOT = Path(__file__).resolve().parents[2]
SOURCES = ("ecg_experiment/echonext.py", "ecg_experiment/eda/echonext.py", "ecg_experiment/ecg_quality.py",
           "scripts/data/build_echonext_cache.py")
ROW_COLUMNS = ["patient_key", "split", "row", "age_at_ecg", "sex", "acquisition_year", "location_setting",
               "ventricular_rate", "atrial_rate", "pr_interval", "qrs_duration", "qt_corrected",
               "shd_moderate_or_greater_flag", *COMPONENTS, *dict.fromkeys(COMPONENTS.values())]


def verify_release() -> dict[str, str]:
    """Check every file of the release against its own SHA-256 list."""
    verified = {}
    for name, expected in checksums().items():
        digest = member_sha256(name)
        if digest != expected:
            raise ValueError(f"EchoNext file differs from SHA256SUMS: {name}")
        verified[name] = digest
    return verified


def convert_split(split: str, stage: Path) -> tuple[list[list[str]], np.ndarray, np.ndarray, np.ndarray]:
    """Write one split as float32 ``(N, 12, 2500)`` and return reasons and per-lead sums of passing rows."""
    count = int((load_metadata()["split"] == split).sum())
    shard = np.lib.format.open_memmap(stage / f"{split}.npy", mode="w+", dtype=np.float32,
                                      shape=(count, 12, 2500))
    reasons, total, squares, samples, offset = [], np.zeros(12), np.zeros(12), 0, 0
    for chunk in waveform_chunks(split):
        leads = np.ascontiguousarray(chunk.transpose(0, 2, 1))
        for index, signal in enumerate(leads):
            found = echonext.assess(signal)
            reasons.append(found)
            if not found:
                total += signal.sum(axis=1)
                squares += (signal ** 2).sum(axis=1)
                samples += signal.shape[1]
            shard[offset + index] = signal
        offset += len(leads)
    shard.flush()
    del shard
    if offset != count:
        raise ValueError(f"{split}: expected {count} waveforms, read {offset}")
    return reasons, total, squares, np.array(samples)


def build(output: Path) -> dict[str, object]:
    """Verify, convert the open splits, and write rows, statistics and the receipt."""
    if output.exists():
        raise ValueError(f"Destination exists: {output}")
    start = time.monotonic()
    verified = verify_release()
    stage = output.with_name(output.name + ".partial")
    stage.mkdir(parents=True)
    meta = load_metadata()
    rows = meta.loc[meta["split"].isin(OPEN_SPLITS), ROW_COLUMNS].copy()
    lead_stats = {}
    for split in OPEN_SPLITS:
        reasons, total, squares, samples = convert_split(split, stage)
        in_split = rows["split"] == split
        rows.loc[in_split, "exclusion_reasons"] = [";".join(items) for items in reasons]
        if split == "train":
            mean = total / samples
            lead_stats = {"mean": mean.tolist(), "std": np.sqrt(squares / samples - mean ** 2).tolist(),
                          "samples_per_lead": int(samples)}
        print(json.dumps({"split": split, "seconds": time.monotonic() - start}), flush=True)
    rows["exclusion_reasons"] = rows["exclusion_reasons"].fillna("")
    rows["use"] = rows["exclusion_reasons"] == ""
    rows.to_csv(stage / "rows.csv")
    metadata = {
        "complete": True, "archive": ARCHIVE.name, "release_sha256": verified,
        "records": rows.groupby("split").size().to_dict(),
        "usable": rows[rows["use"]].groupby("split").size().to_dict(),
        "exclusion_reason_counts": rows["exclusion_reasons"].str.split(";").explode()
        .loc[lambda values: values != ""].value_counts().to_dict(),
        "train_lead_statistics": lead_stats,
        "lead_statistics_rows": "train rows passing the quality rules",
        "layout": "float32 (N, 12, 2500), EchoNext standardized values, canonical lead order, 250 Hz",
        "arrays_sha256": {f"{split}.npy": sha256_file(stage / f"{split}.npy") for split in OPEN_SPLITS},
        "rows_sha256": sha256_file(stage / "rows.csv"),
        "source_sha256": {name: sha256_file(ROOT / name) for name in SOURCES},
        "build_seconds": time.monotonic() - start,
    }
    write_json_atomic(stage / "metadata.json", metadata, sort_keys=True)
    stage.rename(output)
    return metadata


def main() -> None:
    """Parse the output directory and build the cache."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "data/processed/echonext_250hz_v1")
    args = parser.parse_args()
    metadata = build(args.output_dir)
    print(json.dumps({key: metadata[key] for key in ("records", "usable", "exclusion_reason_counts")},
                     indent=2))


if __name__ == "__main__":
    main()
