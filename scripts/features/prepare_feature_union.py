"""Prepare label-free manifests for extracting features once across label samples.

These manifests are for feature extraction only, never classifier fitting.
"""

import argparse
import csv
import hashlib
import json
from pathlib import Path


FIELDS = ["ecg_id", "patient_id", "filename_lr", "filename_hr"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    parser.add_argument("--root", type=Path, default=Path("data/processed/ptbxl"))
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError("Feature union output must be empty")
    partitions, hashes = {}, {}
    for split in ("labeled_train", "validation", "test"):
        merged = {}
        reference = None
        for seed in args.seeds:
            path = args.root / f"seed{seed}_fraction0.1" / f"{split}.csv"
            hashes[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
            with path.open() as handle:
                rows = [{key: row[key] for key in FIELDS} for row in csv.DictReader(handle)]
            current = {row["ecg_id"]: row for row in rows}
            if len(current) != len(rows):
                raise ValueError(f"Duplicate IDs in {path}")
            if split != "labeled_train" and reference is not None and current != reference:
                raise ValueError(f"Evaluation partition differs for seed {seed}")
            reference = current
            for key, row in current.items():
                if key in merged and merged[key] != row:
                    raise ValueError(f"Conflicting waveform provenance for ECG {key}")
                merged[key] = row
        partitions[split] = list(sorted(merged.values(), key=lambda row: int(row["ecg_id"])))
    seen_records, seen_patients = set(), set()
    for split, rows in partitions.items():
        record_ids = {row["ecg_id"] for row in rows}
        patient_ids = {row["patient_id"] for row in rows}
        if seen_records & record_ids or seen_patients & patient_ids:
            raise ValueError(f"Cross-partition overlap in {split}")
        seen_records |= record_ids
        seen_patients |= patient_ids
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for split, rows in partitions.items():
        with (args.output_dir / f"{split}.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=FIELDS)
            writer.writeheader()
            writer.writerows(rows)
    (args.output_dir / "purpose.json").write_text(json.dumps({
        "purpose": "Feature extraction only; no target labels. Fit each seed on its original labeled manifest.",
        "source_sha256": hashes, "seeds": args.seeds,
        "records": {key: len(value) for key, value in partitions.items()},
    }, indent=2) + "\n")


if __name__ == "__main__":
    main()
