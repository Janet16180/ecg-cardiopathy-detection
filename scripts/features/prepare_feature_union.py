"""Prepare label-free manifests for extracting features once across label samples.

These manifests are for feature extraction only, never classifier fitting.
"""

import argparse
import csv
from pathlib import Path

from ecg_experiment.files import sha256_file, write_csv_atomic, write_json_atomic

ROOT = Path(__file__).resolve().parents[2]
FIELDS = ["ecg_id", "patient_id", "filename_lr", "filename_hr"]
SPLITS = ("labeled_train", "validation", "test")

Rows = list[dict[str, str]]


def read_split(path: Path) -> dict[str, dict[str, str]]:
    """
    Read one manifest's label-free columns keyed by ECG identifier.

    Parameters
    ----------
    path : Path
        Split manifest.

    Returns
    -------
    dict[str, dict[str, str]]
        Label-free rows keyed by ``ecg_id``.

    Raises
    ------
    ValueError
        If an ECG identifier repeats.
    """
    with path.open() as handle:
        rows = [{key: row[key] for key in FIELDS} for row in csv.DictReader(handle)]
    current = {row["ecg_id"]: row for row in rows}
    if len(current) != len(rows):
        raise ValueError(f"Duplicate IDs in {path}")
    return current


def merge_split(root: Path, split: str, seeds: list[int], hashes: dict[str, str]) -> Rows:
    """
    Merge one split across label seeds, sorted by ECG identifier.

    Parameters
    ----------
    root : Path
        Directory holding the per-seed manifests, as given on the command line.
    split : str
        Split name.
    seeds : list[int]
        Label-sampling seeds.
    hashes : dict[str, str]
        Source digests keyed by manifest path, updated in place.

    Returns
    -------
    list[dict[str, str]]
        Union of the split's label-free rows.

    Raises
    ------
    ValueError
        If evaluation partitions differ across seeds or an ECG's waveform
        provenance conflicts.
    """
    merged: dict[str, dict[str, str]] = {}
    reference = None
    for seed in seeds:
        path = root / f"seed{seed}_fraction0.1" / f"{split}.csv"
        hashes[str(path)] = sha256_file(ROOT / path)
        current = read_split(ROOT / path)
        if split != "labeled_train" and reference is not None and current != reference:
            raise ValueError(f"Evaluation partition differs for seed {seed}")
        reference = current
        for key, row in current.items():
            if key in merged and merged[key] != row:
                raise ValueError(f"Conflicting waveform provenance for ECG {key}")
            merged[key] = row
    return sorted(merged.values(), key=lambda row: int(row["ecg_id"]))


def check_disjoint(partitions: dict[str, Rows]) -> None:
    """
    Require the merged splits to share no record and no patient.

    Parameters
    ----------
    partitions : dict[str, list[dict[str, str]]]
        Merged rows per split.

    Raises
    ------
    ValueError
        If a record or patient appears in two splits.
    """
    seen_records, seen_patients = set(), set()
    for split, rows in partitions.items():
        record_ids = {row["ecg_id"] for row in rows}
        patient_ids = {row["patient_id"] for row in rows}
        if seen_records & record_ids or seen_patients & patient_ids:
            raise ValueError(f"Cross-partition overlap in {split}")
        seen_records |= record_ids
        seen_patients |= patient_ids


def main() -> None:
    """
    Write the union manifests and their provenance.

    Raises
    ------
    FileExistsError
        If the output directory is not empty.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    parser.add_argument("--root", type=Path, default=Path("data/processed/ptbxl"),
                        help="Per-seed manifest directory; relative paths start at the repository root")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError("Feature union output must be empty")
    hashes: dict[str, str] = {}
    partitions = {split: merge_split(args.root, split, args.seeds, hashes) for split in SPLITS}
    check_disjoint(partitions)
    for split, rows in partitions.items():
        write_csv_atomic(args.output_dir / f"{split}.csv", rows, FIELDS)
    write_json_atomic(args.output_dir / "purpose.json", {
        "purpose": ("Feature extraction only; no target labels. "
                    "Fit each seed on its original labeled manifest."),
        "source_sha256": hashes, "seeds": args.seeds,
        "records": {key: len(value) for key, value in partitions.items()},
    })


if __name__ == "__main__":
    main()
