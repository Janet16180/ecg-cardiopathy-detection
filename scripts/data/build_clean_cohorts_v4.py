"""Publish cohorts v4: the v3 tiers with the usable EchoNext training ECGs right after the curated sources.

Two steps, see ``docs/clean-cohorts-v4.md``:

- ``build`` orders every candidate once, checks that v4 without EchoNext is v3, and writes each tier as the
  first N rows of that order.
- ``resolve`` drops failed pending MIMIC records from the saved order and writes new tier directories. The
  results come from the v2 ``score-pending`` step run with ``--work-dir`` pointing at the v4 work directory.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd

from ecg_experiment import clean_cohorts_v2 as v2
from ecg_experiment import clean_cohorts_v3 as v3
from ecg_experiment import clean_cohorts_v4 as v4
from ecg_experiment.clean_cohorts import UNION
from ecg_experiment.eda.echonext import member_sha256
from ecg_experiment.files import sha256_file, write_json_atomic
from ecg_experiment.paths import to_stored

ROOT = v2.ROOT
SIZES = {"25k": 25_000, "50k": 50_000, "100k": 100_000, "150k": 150_000, "200k": 200_000, "500k": 500_000,
         "1m": 1_000_000}
SOURCES = ("ecg_experiment/clean_cohorts_v4.py", "ecg_experiment/clean_cohorts_v3.py",
           "ecg_experiment/cohort_tiers.py", "ecg_experiment/clean_cohorts_v2.py",
           "ecg_experiment/clean_cohorts.py", "ecg_experiment/challenge_labels.py",
           "ecg_experiment/ecg_quality.py", "ecg_experiment/echonext.py", "ecg_experiment/eda/echonext.py",
           "scripts/data/build_clean_cohorts_v4.py")
INPUTS = (v3.CHALLENGE_SPLITS, v2.QUALITY_V1, v2.NINGBO, v2.NINGBO_RECEIPT, v2.CODE15, v2.MIMIC_RECORDS,
          v2.PTBXL_REFERENCE, ROOT / UNION / "heldout_references.csv", ROOT / UNION / "labels_fraction1.csv",
          ROOT / UNION / "labels_fraction0.1.csv", v4.ECHONEXT / "rows.csv", v4.ECHONEXT / "metadata.json",
          v4.ECHONEXT / "train.npy")
V3_DIR = ROOT / "outputs/data_quality/clean_cohorts_v3"
EXCLUSION = ("Challenge records of the test and calibration groups of docs/challenge-splits-v1.md, and any "
             "candidate with one of their waveform hashes, are never candidates; of EchoNext only usable "
             "train rows are candidates, and no training patient appears in val, test or no_split")


def check_echonext_inputs() -> dict[str, object]:
    """Check the EchoNext cache against the release and the patient split; return what was checked."""
    cache = json.loads((v4.ECHONEXT / "metadata.json").read_text())
    if sha256_file(v4.ECHONEXT / "train.npy") != cache["arrays_sha256"]["train.npy"]:
        raise ValueError("EchoNext train.npy differs from its cache metadata")
    name = "echonext_metadata_100k.csv"
    if member_sha256(name) != cache["release_sha256"][name]:
        raise ValueError("The EchoNext metadata differs from the verified release")
    splits = v4.read_release_splits()
    v4.check_patients_disjoint(splits)
    return {"release_split_ecgs": splits["split"].value_counts().to_dict(),
            "train_patients": int(splits.loc[splits["split"] == "train", "patient_key"].nunique()),
            "train_patients_in_other_splits": 0}


def shared_metadata(seed: int) -> dict[str, object]:
    """Metadata common to every v4 tier."""
    return {
        "schema_version": 4, "complete": True, "seed": seed, "nested_sizes": SIZES,
        "ordering": v4.__doc__.strip(), "exclusion": EXCLUSION,
        "storage_policy": "Manifest only; rows reference union or Chapman shards, raw WFDB files, CODE-15 "
                          "archive members, or rows of the local EchoNext train.npy; no waveform is copied",
        "label_policy": "labels_fraction files hold the clean PTB-XL training proxy labels; every other row "
                        "is SSL only; label_available marks rows whose ECG-annotation label is defined, so "
                        "it is false for EchoNext, whose echo labels stay in its own rows.csv",
        "source_sha256": {name: sha256_file(ROOT / name) for name in SOURCES},
    }


def build(output_root: Path, work_dir: Path, seed: int) -> dict[str, object]:
    """Order every candidate, check it against v3, save the order and publish the tiers."""
    echonext_checks = check_echonext_inputs()
    splits = v3.read_splits()
    _, held_ids, held_hashes = v3.challenge_references(splits)
    order = v4.candidate_order(seed, splits, v4.echonext_rows())
    if order["record_id"].isin(held_ids).any() or order["signal_sha256"].isin(held_hashes).any():
        raise ValueError("A Challenge evaluation record is still a candidate")
    work_dir.mkdir(parents=True, exist_ok=True)
    order_path = work_dir / "candidate_order.csv.gz"
    order.to_csv(order_path, index=False, compression={"method": "gzip", "mtime": 0})
    stored = pd.read_csv(order_path, dtype=str, keep_default_na=False)
    previous = pd.read_csv(V3_DIR / "candidate_order.csv.gz", dtype=str, keep_default_na=False)
    comparison = v4.compare_with_v3(stored, previous)
    shared = shared_metadata(seed) | {
        "candidate_order_sha256": sha256_file(order_path),
        "candidate_order_csv_sha256": hashlib.sha256(order.to_csv(index=False).encode()).hexdigest(),
        "input_sha256": {to_stored(path): sha256_file(path) for path in INPUTS},
        "v3_candidate_order_sha256": sha256_file(V3_DIR / "candidate_order.csv.gz"),
    }
    receipt = {
        "candidates": len(order),
        "candidates_per_block": order["block"].value_counts().to_dict(),
        "candidates_per_source": order["source"].value_counts().to_dict(),
        "echonext_checks": echonext_checks,
        "v3_comparison": comparison,
        "tiers": v4.publish_tiers(order, SIZES, output_root, "v4", shared, seed),
    } | {key: shared[key] for key in ("seed", "candidate_order_sha256", "candidate_order_csv_sha256",
                                      "input_sha256", "source_sha256", "v3_candidate_order_sha256")}
    write_json_atomic(work_dir / "receipt.json", receipt, sort_keys=True)
    return receipt


def resolve(work_dir: Path, results: Path, output_root: Path, suffix: str, seed: int) -> dict[str, object]:
    """Drop failed pending records from the saved order and publish new tier directories."""
    order = pd.read_csv(work_dir / "candidate_order.csv.gz", dtype=str, keep_default_na=False)
    order["label_available"] = order["label_available"] == "True"
    scored = pd.read_csv(results, dtype=str, keep_default_na=False)
    resolved = v2.resolve_pending(order, scored)
    shared = shared_metadata(seed) | {
        "resolved_from_sha256": sha256_file(work_dir / "candidate_order.csv.gz"),
        "pending_results_sha256": sha256_file(results),
    }
    return {"tiers": v4.publish_tiers(resolved, SIZES, output_root, suffix, shared, seed),
            "still_pending": int((resolved["quality_status"] == "pending").sum())}


def main() -> None:
    """Parse arguments and run one step."""
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("step", choices=["build", "resolve"])
    parser.add_argument("--output-root", type=Path, default=ROOT / "data/processed")
    parser.add_argument("--work-dir", type=Path, default=ROOT / "outputs/data_quality/clean_cohorts_v4")
    parser.add_argument("--results", type=Path, help="pending MIMIC quality results (resolve)")
    parser.add_argument("--suffix", default="v4_resolved", help="tier directory suffix for resolve")
    parser.add_argument("--seed", type=int, default=20260929)
    args = parser.parse_args()
    if args.step == "build":
        result = build(args.output_root, args.work_dir, args.seed)
    else:
        result = resolve(args.work_dir, args.results, args.output_root, args.suffix, args.seed)
    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()
