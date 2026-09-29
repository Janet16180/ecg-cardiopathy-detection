"""Publish a manifest-only clean Ningbo table: files verified, duplicates resolved, labels mapped."""

from __future__ import annotations

import argparse
import hashlib
import json
import sqlite3
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

from ecg_experiment import challenge_labels, ningbo
from ecg_experiment.ecg_quality import EXCLUSION_REASONS, REVIEW_FLAGS
from ecg_experiment.eda import challenge
from ecg_experiment.eda import ningbo as ningbo_eda
from ecg_experiment.files import sha256_file, write_json_atomic
from ecg_experiment.public_sources import signal_sha256

ROOT = Path(__file__).resolve().parents[2]
SOURCES = ("ecg_experiment/ningbo.py", "ecg_experiment/eda/ningbo.py", "ecg_experiment/challenge_labels.py",
           "ecg_experiment/ecg_quality.py", "scripts/data/build_ningbo_clean.py")
VERIFICATION = ROOT / "data/acquisition/ningbo_verification.json"
EXPECTED_ABSENT = {"training/ningbo/g3/JS13118.mat", "training/ningbo/g13/JS23074.mat",
                   "training/ningbo/g13/S23074.hea"}
PTBXL_REFERENCE = ROOT / "outputs/data_quality/ptbxl_reference_hashes_v2.json"
UNION = ROOT / "data/processed/training_union_500hz_v1/train_manifest.csv"
MIMIC_AUDIT = ROOT / "data/processed/mimic_ssl_200k/audit.sqlite3"
CLEAN_COHORTS = [ROOT / f"data/processed/clean_{size}_plus_labels_v1/train_manifest.csv"
                 for size in ("25k", "50k", "100k")]
NEAR_DUPLICATE_CORRELATION = 0.99


def verify_files() -> dict[str, object]:
    """
    Check every local Ningbo file against the official SHA256SUMS.

    Returns
    -------
    dict[str, object]
        Manifest hash, number of verified files and the absent official files.

    Raises
    ------
    ValueError
        If a file differs, or the absent files are not exactly the known upstream problems.
    """
    manifest_path = ningbo_eda.RAW_ROOT / "SHA256SUMS.txt"
    lines = [line.split(maxsplit=1) for line in manifest_path.read_text().splitlines()]
    expected = {name.strip(): digest for digest, name in lines if name.startswith("training/ningbo/")}
    wanted = {name: digest for name, digest in expected.items() if name.endswith((".hea", ".mat"))}
    absent = {name for name in wanted if not (ningbo_eda.RAW_ROOT / name).exists()}
    if absent != EXPECTED_ABSENT:
        raise ValueError(f"Unexpected absent Ningbo files: {sorted(absent ^ EXPECTED_ABSENT)}")
    for name in sorted(set(wanted) - absent):
        if hashlib.sha256((ningbo_eda.RAW_ROOT / name).read_bytes()).hexdigest() != wanted[name]:
            raise ValueError(f"Ningbo file differs from SHA256SUMS: {name}")
    return {"sha256sums_sha256": sha256_file(manifest_path), "verified_files": len(wanted) - len(absent),
            "absent_files": sorted(absent)}


def _chapman_item(path: str) -> tuple[str, str, np.ndarray]:
    signal, _ = challenge.read_signal(path)
    microvolts = np.round(np.nan_to_num(signal, nan=-99999) * 1000).astype(np.int32)
    window = signal[:5000].T.astype(np.float32)
    return (hashlib.sha256(microvolts.tobytes()).hexdigest(), signal_sha256(window),
            ningbo_eda.fingerprint(signal[:5000]))


def chapman_table(official: pd.DataFrame, workers: int) -> tuple[pd.DataFrame, np.ndarray]:
    """
    Hashes, labels, sex and fingerprints of every Chapman record, read from the raw files.

    Parameters
    ----------
    official : pd.DataFrame
        ``challenge_labels.load_official`` output.
    workers : int
        Number of processes.

    Returns
    -------
    tuple[pd.DataFrame, np.ndarray]
        Table indexed by record name and the fingerprints in its order.
    """
    headers = challenge.load_headers()
    headers = headers[headers["source"] == "chapman_shaoxing"].set_index("record")
    paths = [f"chapman_shaoxing:{path}" for path in headers["path"]]
    with Pool(workers) as pool:
        items = pool.map(_chapman_item, paths, chunksize=64)
    labels = challenge_labels.label_table(headers["dx_codes"], official)
    table = pd.DataFrame({"signal_sha256": [item[0] for item in items],
                          "window_sha256": [item[1] for item in items],
                          "label_key": ningbo.label_key(labels), "male": ningbo.male(headers["sex"]),
                          "age": ningbo.clean_age(headers["age"])}, index=headers.index)
    return table, np.stack([item[2] for item in items])


def near_duplicate_pairs(names: pd.Index, fingerprints: np.ndarray, hashes: pd.Series) -> pd.DataFrame:
    """
    Family pairs whose lead II fingerprints correlate above the threshold but are not identical.

    Parameters
    ----------
    names : pd.Index
        Record names in fingerprint order.
    fingerprints : np.ndarray
        Stacked fingerprints.
    hashes : pd.Series
        Signal hash per record name.

    Returns
    -------
    pd.DataFrame
        ``first``, ``second`` and ``correlation`` of the non-identical pairs.
    """
    pairs = ningbo_eda.near_duplicates(names, fingerprints, NEAR_DUPLICATE_CORRELATION)
    identical = pairs["first"].map(hashes).to_numpy() == pairs["second"].map(hashes).to_numpy()
    return pairs.loc[~identical].reset_index(drop=True)


def external_overlaps(window_sha256: pd.Series, micro_sha256: pd.Series) -> dict[str, object]:
    """
    Exact overlaps of Ningbo windows with the other verified sources.

    Parameters
    ----------
    window_sha256 : pd.Series
        Float32 window hash per Ningbo record.
    micro_sha256 : pd.Series
        Microvolt hash per Ningbo record.

    Returns
    -------
    dict[str, object]
        Reference sizes and the overlapping Ningbo records per source, with the input hashes.
    """
    reference = set(json.loads(PTBXL_REFERENCE.read_text())["hashes"])
    union = pd.read_csv(UNION, dtype=str, usecols=["source", "signal_sha256"])
    with sqlite3.connect(f"file:{MIMIC_AUDIT}?mode=ro", uri=True) as connection:
        mimic = {row[0] for row in connection.execute(
            "SELECT signal_sha256 FROM outcomes WHERE signal_sha256 IS NOT NULL")}
    challenge_hashes = challenge.signal_hashes(challenge.load_headers())
    other_hashes = challenge_hashes[~challenge_hashes.index.str.startswith("chapman_shaoxing:")]
    cohorts = {path.parent.name: set(pd.read_csv(path, dtype=str, usecols=["signal_sha256"])["signal_sha256"])
               for path in CLEAN_COHORTS}
    checks = {"ptbxl_all_records": reference, "training_union_500hz_v1": set(union["signal_sha256"]),
              "mimic_ssl_200k_audit": mimic, **cohorts}
    result = {name: {"reference_hashes": len(hashes),
                     "overlapping_ningbo_records": sorted(window_sha256.index[window_sha256.isin(hashes)])}
              for name, hashes in checks.items()}
    result["georgia_cpsc_cpsc_extra"] = {
        "reference_hashes": int(other_hashes.nunique()),
        "overlapping_ningbo_records": sorted(micro_sha256.index[micro_sha256.isin(set(other_hashes))])}
    result["inputs_sha256"] = {str(path.relative_to(ROOT)): sha256_file(path)
                               for path in (PTBXL_REFERENCE, UNION, MIMIC_AUDIT, *CLEAN_COHORTS)}
    return result


def ningbo_rows(workers: int, official: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray]:
    """
    Score, hash and label every Ningbo record with a signal file.

    Parameters
    ----------
    workers : int
        Number of processes.
    official : pd.DataFrame
        ``challenge_labels.load_official`` output.

    Returns
    -------
    tuple[pd.DataFrame, np.ndarray]
        One row per record, indexed by record name, and the fingerprints in its order.
    """
    headers = ningbo_eda.load_headers(use_cache=False)
    headers = headers[headers["has_signal"]]
    scan, leads, fingerprints = ningbo_eda.record_scan(headers, workers, use_cache=False)
    zero = leads[(leads["run_length"] == 5000) & (leads["run_value"] == 0)]
    zero_leads = zero.groupby("record")["lead"].agg(";".join)
    labels = challenge_labels.label_table(headers["dx_codes"], official)
    rows = pd.DataFrame({
        "path": headers["path"], "age": ningbo.clean_age(headers["age"]), "male": ningbo.male(headers["sex"]),
        "snomed_codes": headers["dx_codes"].apply(";".join),
    }, index=headers.index).join(labels)
    rows["label_key"] = ningbo.label_key(labels)
    rows = rows.join(scan[["signal_sha256", "window_sha256"]])
    rows["exclusion_reasons"] = scan[list(EXCLUSION_REASONS)].apply(
        lambda row: ";".join(name for name in EXCLUSION_REASONS if row[name]), axis=1)
    rows["review_flags"] = scan[list(REVIEW_FLAGS)].apply(
        lambda row: ";".join(name for name in REVIEW_FLAGS if row[name]), axis=1)
    rows["zero_leads"] = zero_leads.reindex(rows.index).fillna("")
    return rows, fingerprints


def build(output: Path, receipt_path: Path, workers: int) -> dict[str, object]:
    """Verify, score and resolve every record, then write ``rows.csv``, ``metadata.json`` and the receipt."""
    if output.exists():
        raise ValueError(f"Destination exists: {output}")
    files = verify_files()
    official = challenge_labels.load_official()
    rows, ningbo_prints = ningbo_rows(workers, official)
    chapman, chapman_prints = chapman_table(official, workers)

    family = pd.concat([chapman[["signal_sha256", "label_key", "male"]],
                        rows[["signal_sha256", "label_key", "male"]]])
    status = ningbo.duplicate_status(family)
    rows = rows.join(status)
    rows["use_evaluation"] = rows["duplicate_status"].isin(["unique", "kept"])
    rows["use_training"] = rows["use_evaluation"] & (rows["exclusion_reasons"] == "")
    names = pd.Index([*chapman.index, *rows.index])
    near = near_duplicate_pairs(names, np.concatenate([chapman_prints, ningbo_prints]),
                                family["signal_sha256"])
    near_partner = pd.concat([near.set_index("first")["second"], near.set_index("second")["first"]])
    rows["near_duplicate_of"] = near_partner.groupby(level=0).agg(";".join).reindex(rows.index).fillna("")

    chapman_status = status.loc[chapman.index]
    chapman_names = set(chapman.index)
    of_chapman = rows["duplicate_of"].str.split(";").apply(lambda names: bool(chapman_names & set(names)))
    cross = rows[of_chapman]
    receipt = {
        "within_family_statuses": {
            "ningbo": rows["duplicate_status"].value_counts().to_dict(),
            "chapman_shaoxing": chapman_status["duplicate_status"].value_counts().to_dict()},
        "ningbo_copies_of_chapman": len(cross),
        "ningbo_copies_of_chapman_conflicting": int((cross["duplicate_status"] == "dropped_conflict").sum()),
        "chapman_records_dropped_for_conflict": sorted(
            chapman_status.index[chapman_status["duplicate_status"] == "dropped_conflict"]),
        "chapman_records_dropped_as_copy": sorted(
            chapman_status.index[chapman_status["duplicate_status"] == "dropped_copy"]),
        "within_ningbo_groups": int(rows.loc[~rows.index.isin(cross.index)]
                                    .query("duplicate_status != 'unique'")["signal_sha256"].nunique()),
        "external": external_overlaps(rows["window_sha256"], rows["signal_sha256"]),
        "near_duplicates": {"correlation_threshold": NEAR_DUPLICATE_CORRELATION,
                            "fingerprint": "lead II, 50 Hz means, z-scored, zero lag",
                            "non_identical_pairs": near.to_dict("records")},
        "policy": "exact int32-microvolt hash groups across Chapman and Ningbo; the lowest record name is "
                  "kept when mapped labels and known sex agree, otherwise every copy is dropped",
    }
    receipt_path.parent.mkdir(parents=True, exist_ok=True)
    write_json_atomic(receipt_path, receipt, sort_keys=True)

    rows = rows.drop(columns=["label_key"])
    stage = output.with_name(output.name + ".partial")
    stage.mkdir(parents=True)
    rows.to_csv(stage / "rows.csv", index_label="record")
    metadata = {
        "complete": True, "records": len(rows),
        "window": "whole record (10 s, 5,000 samples at 500 Hz), mV, canonical lead order",
        "files": files, "verification_receipt_sha256": sha256_file(VERIFICATION),
        "official_mapping_sha256": challenge_labels.MAPPING_SHA256,
        "duplicate_status": rows["duplicate_status"].value_counts().to_dict(),
        "exclusion_reason_counts": rows["exclusion_reasons"].str.split(";").explode()
        .loc[lambda values: values != ""].value_counts().to_dict(),
        "review_flag_counts": rows["review_flags"].str.split(";").explode()
        .loc[lambda values: values != ""].value_counts().to_dict(),
        "records_with_zero_leads": int((rows["zero_leads"] != "").sum()),
        "evaluation": {"records": int(rows["use_evaluation"].sum()),
                       "primary": rows.loc[rows["use_evaluation"], "primary"].value_counts(dropna=False)
                       .rename(str).to_dict()},
        "training": {"records": int(rows["use_training"].sum()),
                     "primary": rows.loc[rows["use_training"], "primary"].value_counts(dropna=False)
                     .rename(str).to_dict()},
        "overlap_receipt_sha256": sha256_file(receipt_path),
        "rows_sha256": sha256_file(stage / "rows.csv"),
        "source_sha256": {name: sha256_file(ROOT / name) for name in SOURCES},
    }
    write_json_atomic(stage / "metadata.json", metadata, sort_keys=True)
    stage.rename(output)
    return metadata


def main() -> None:
    """Parse arguments and build the manifest."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "data/processed/ningbo_clean_v1")
    parser.add_argument("--receipt", type=Path,
                        default=ROOT / "outputs/data_quality/ningbo_v1/exact_overlap_receipt.json")
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    metadata = build(args.output_dir, args.receipt, args.workers)
    summary = ("records", "duplicate_status", "exclusion_reason_counts", "evaluation", "training")
    print(json.dumps({key: metadata[key] for key in summary}, indent=2))


if __name__ == "__main__":
    main()
