"""Publish cohorts v3: the nested v2 tiers without any Challenge test or calibration record.

Two steps, see ``docs/clean-cohorts-v3.md``:

- ``build`` orders every candidate once with the v2 rules after the Challenge exclusion and writes each tier
  as the first N rows of that order.
- ``resolve`` drops failed pending MIMIC records from the saved order and writes new tier directories. The
  results come from the v2 ``score-pending`` step run with ``--work-dir`` pointing at the v3 work directory.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd

from ecg_experiment import clean_cohorts_v2 as v2
from ecg_experiment import clean_cohorts_v3 as v3
from ecg_experiment.clean_cohorts import UNION
from ecg_experiment.cohort_tiers import publish_tiers
from ecg_experiment.files import sha256_file, write_json_atomic
from ecg_experiment.paths import to_stored

ROOT = v2.ROOT
SIZES = {"25k": 25_000, "50k": 50_000, "100k": 100_000, "150k": 150_000, "200k": 200_000, "500k": 500_000,
         "1m": 1_000_000}
SOURCES = ("ecg_experiment/clean_cohorts_v3.py", "ecg_experiment/cohort_tiers.py",
           "ecg_experiment/clean_cohorts_v2.py", "ecg_experiment/clean_cohorts.py",
           "ecg_experiment/challenge_labels.py", "ecg_experiment/ecg_quality.py",
           "scripts/data/build_clean_cohorts_v3.py")
INPUTS = (v3.CHALLENGE_SPLITS, v2.QUALITY_V1, v2.NINGBO, v2.NINGBO_RECEIPT, v2.CODE15, v2.MIMIC_RECORDS,
          v2.PTBXL_REFERENCE, ROOT / UNION / "heldout_references.csv", ROOT / UNION / "labels_fraction1.csv",
          ROOT / UNION / "labels_fraction0.1.csv")
V2_ORDER = ROOT / "outputs/data_quality/clean_cohorts_v2/candidate_order.csv.gz"
EXCLUSION = ("Challenge records of the test and calibration groups of docs/challenge-splits-v1.md, and any "
             "candidate with one of their waveform hashes, are never candidates")


def evaluation_counts(order: pd.DataFrame, held_ids: set[str], splits: pd.DataFrame) -> dict[str, dict]:
    """
    Challenge evaluation records in an order, per split group and source.

    Parameters
    ----------
    order : pd.DataFrame
        A candidate order with ``record_id``.
    held_ids : set[str]
        Record IDs of the test and calibration groups.
    splits : pd.DataFrame
        Output of ``clean_cohorts_v3.read_splits``.

    Returns
    -------
    dict[str, dict]
        Counts keyed by split group, then source.
    """
    groups = splits.assign(record_id=splits["source"] + ":" + splits["record"]).set_index("record_id")
    held = order.loc[order["record_id"].isin(held_ids), "record_id"]
    found = groups.loc[held, ["split", "source"]]
    return {split: rows["source"].value_counts().sort_index().to_dict()
            for split, rows in found.groupby("split")}


def shared_metadata(seed: int) -> dict[str, object]:
    """Metadata common to every v3 tier."""
    return {
        "schema_version": 3, "complete": True, "seed": seed, "nested_sizes": SIZES,
        "ordering": v2.__doc__.strip(), "exclusion": EXCLUSION,
        "storage_policy": "Manifest only; rows reference union or Chapman shards, raw WFDB files, or CODE-15 "
                          "archive members; no waveform is copied",
        "label_policy": "labels_fraction files hold the clean PTB-XL training proxy labels; every other row "
                        "is SSL only; label_available marks rows whose source label is defined",
        "source_sha256": {name: sha256_file(ROOT / name) for name in SOURCES},
    }


def build(output_root: Path, work_dir: Path, seed: int) -> dict[str, object]:
    """Order every candidate, check no evaluation record remains, save the order and publish the tiers."""
    splits = v3.read_splits()
    _, held_ids, held_hashes = v3.challenge_references(splits)
    order = v3.candidate_order(seed, splits)
    if order["record_id"].isin(held_ids).any() or order["signal_sha256"].isin(held_hashes).any():
        raise ValueError("A Challenge evaluation record is still a candidate")
    v2_order = pd.read_csv(V2_ORDER, dtype=str, keep_default_na=False, usecols=["record_id"])
    work_dir.mkdir(parents=True, exist_ok=True)
    order_path = work_dir / "candidate_order.csv.gz"
    order.to_csv(order_path, index=False, compression={"method": "gzip", "mtime": 0})
    csv_sha256 = hashlib.sha256(order.to_csv(index=False).encode()).hexdigest()
    shared = shared_metadata(seed) | {
        "candidate_order_sha256": sha256_file(order_path),
        "candidate_order_csv_sha256": csv_sha256,
        "input_sha256": {to_stored(path): sha256_file(path) for path in INPUTS},
    }
    receipt = {
        "candidates": len(order),
        "candidates_per_block": order["block"].value_counts().to_dict(),
        "candidates_per_source": order["source"].value_counts().to_dict(),
        "v2_candidates": len(v2_order),
        "v2_evaluation_records": evaluation_counts(v2_order, held_ids, splits),
        "tiers": publish_tiers(order, SIZES, output_root, "v3", shared, seed),
    } | {key: shared[key] for key in ("seed", "candidate_order_sha256", "candidate_order_csv_sha256",
                                      "input_sha256", "source_sha256")}
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
    return {"tiers": publish_tiers(resolved, SIZES, output_root, suffix, shared, seed),
            "still_pending": int((resolved["quality_status"] == "pending").sum())}


def main() -> None:
    """Parse arguments and run one step."""
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("step", choices=["build", "resolve"])
    parser.add_argument("--output-root", type=Path, default=ROOT / "data/processed")
    parser.add_argument("--work-dir", type=Path, default=ROOT / "outputs/data_quality/clean_cohorts_v3")
    parser.add_argument("--results", type=Path, help="pending MIMIC quality results (resolve)")
    parser.add_argument("--suffix", default="v3_resolved", help="tier directory suffix for resolve")
    parser.add_argument("--seed", type=int, default=20260929)
    args = parser.parse_args()
    if args.step == "build":
        result = build(args.output_root, args.work_dir, args.seed)
    else:
        result = resolve(args.work_dir, args.results, args.output_root, args.suffix, args.seed)
    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()
