"""Publish a seeded, manifest-backed ECG cohort without copying raw waveforms."""

from __future__ import annotations

import argparse
import json
import sqlite3
from collections import Counter
from pathlib import Path

import pandas as pd

from ecg_experiment.files import sha256_file, write_json_atomic
from ecg_experiment.sampled_cohort import sample_sources
from ecg_experiment.sampled_training_dataset import SampledTrainingECGDataset
from ecg_experiment.staging import published_directory

ROOT = Path(__file__).resolve().parents[2]
UNION = Path("data/processed/training_union_500hz_v1")
CHAPMAN = Path("data/processed/challenge_ecg_views/chapman_strict_10s_v1")
MIMIC = Path("data/processed/mimic_ssl_200k")
MIMIC_RAW = Path("data/raw/mimic-iv-ecg/1.0")
FIELDS = (
    "record_id", "patient_id", "source", "split", "label_scope", "backend",
    "path", "index", "signal_sha256",
)
INPUTS = (
    UNION / "metadata.json",
    UNION / "train_manifest.csv",
    UNION / "labels_fraction1.csv",
    UNION / "labels_fraction0.1.csv",
    UNION / "heldout_references.csv",
    CHAPMAN / "metadata.json",
    CHAPMAN / "manifest.csv",
    MIMIC / "metadata.json",
    MIMIC / "ssl_manifest.csv",
    MIMIC / "audit.sqlite3",
)


def table(path: Path) -> pd.DataFrame:
    """Read a repository CSV while preserving patient and record IDs as text."""
    return pd.read_csv(ROOT / path, dtype=str, keep_default_na=False)


def union_rows() -> tuple[pd.DataFrame, pd.DataFrame, dict[str, dict[str, str]]]:
    """Split the verified union into labeled PTB rows and other candidates."""
    metadata = json.loads((ROOT / UNION / "metadata.json").read_text())
    for name in (
        "train_manifest.csv", "labels_fraction1.csv", "labels_fraction0.1.csv",
        "heldout_references.csv",
    ):
        if sha256_file(ROOT / UNION / name) != metadata["table_sha256"][name]:
            raise ValueError(f"Union table hash mismatch: {name}")
    rows = table(UNION / "train_manifest.csv")
    labels = table(UNION / "labels_fraction1.csv")
    labeled_ids = set(labels["record_id"])
    if len(labels) != 15359 or not labeled_ids <= set(rows["record_id"]):
        raise ValueError("Clean PTB label selection changed")

    rows = rows.assign(
        backend="shard",
        path=(str(UNION) + "/" + rows["shard"]),
        index=rows["shard_index"],
    )
    labeled = rows.loc[rows["record_id"].isin(labeled_ids), list(FIELDS)].copy()
    candidates = rows.loc[
        (rows["source"] != "mimic") & ~rows["record_id"].isin(labeled_ids), list(FIELDS)
    ].copy()
    mimic_shards = rows.loc[rows["source"] == "mimic", ["record_id", "path", "index", "signal_sha256"]]
    lookup = mimic_shards.set_index("record_id").to_dict("index")
    return labeled, candidates, lookup


def mimic_rows(union_lookup: dict[str, dict[str, str]]) -> pd.DataFrame:
    """Join the 200k audit's accepted canonical hashes to source pointers."""
    metadata = json.loads((ROOT / MIMIC / "metadata.json").read_text())
    manifest_path = ROOT / MIMIC / "ssl_manifest.csv"
    if sha256_file(manifest_path) != metadata["manifest_sha256"]:
        raise ValueError("MIMIC accepted manifest hash mismatch")
    rows = table(MIMIC / "ssl_manifest.csv")
    database = ROOT / MIMIC / "audit.sqlite3"
    with sqlite3.connect(f"file:{database}?mode=ro", uri=True) as connection:
        hashes = dict(connection.execute(
            "SELECT name, signal_sha256 FROM outcomes WHERE status='accepted'"
        ))
    if len(rows) != metadata["accepted_records"] or len(rows) != len(hashes):
        raise ValueError("MIMIC audit and accepted manifest disagree")

    records = []
    for row in rows.itertuples(index=False):
        digest = hashes[row.filename_hr]
        stored = union_lookup.get(row.ecg_id)
        if stored and stored["signal_sha256"] != digest:
            raise ValueError(f"MIMIC union/audit waveform mismatch: {row.ecg_id}")
        records.append({
            "record_id": row.ecg_id,
            "patient_id": row.patient_id,
            "source": "mimic",
            "split": "train",
            "label_scope": "ssl_only",
            "backend": "shard" if stored else "mimic_wfdb",
            "path": stored["path"] if stored else str(MIMIC_RAW / row.filename_hr),
            "index": stored["index"] if stored else "",
            "signal_sha256": digest,
        })
    return pd.DataFrame.from_records(records, columns=FIELDS)


def chapman_rows() -> pd.DataFrame:
    """Describe verified 500 Hz Chapman shards as SSL-only rows."""
    metadata = json.loads((ROOT / CHAPMAN / "metadata.json").read_text())
    manifest_path = ROOT / CHAPMAN / "manifest.csv"
    if sha256_file(manifest_path) != metadata["manifest_sha256"]:
        raise ValueError("Chapman manifest hash mismatch")
    rows = table(CHAPMAN / "manifest.csv")
    if len(rows) != metadata["accepted_records"]:
        raise ValueError("Chapman accepted count changed")
    return pd.DataFrame({
        "record_id": rows["ecg_id"],
        "patient_id": "",
        "source": "chapman_shaoxing",
        "split": "train",
        "label_scope": "ssl_only",
        "backend": "shard",
        "path": str(CHAPMAN) + "/" + rows["shard"],
        "index": rows["shard_index"],
        "signal_sha256": rows["signal_sha256"],
    })


def unique_candidates(labeled: pd.DataFrame, pools: list[pd.DataFrame]) -> tuple[pd.DataFrame, int]:
    """Remove exact waveform copies, preferring curated views over raw MIMIC."""
    candidates = pd.concat(pools, ignore_index=True)
    if candidates["record_id"].duplicated().any():
        raise ValueError("Source record ID collision")
    candidates = candidates.loc[~candidates["signal_sha256"].isin(labeled["signal_sha256"])]
    before = len(candidates)
    candidates = candidates.drop_duplicates("signal_sha256", keep="first")
    return candidates.reset_index(drop=True), before - len(candidates)


def source_shards(rows: pd.DataFrame) -> dict[str, dict[str, object]]:
    """Bind every referenced source shard to its published SHA and shape."""
    union_meta = json.loads((ROOT / UNION / "metadata.json").read_text())
    chapman_meta = json.loads((ROOT / CHAPMAN / "metadata.json").read_text())
    chapman_shards = {item["file"]: item for item in chapman_meta["shards"]}
    result = {}
    for relative in sorted(rows.loc[rows["backend"] == "shard", "path"].unique()):
        path = Path(relative)
        if path.parent == UNION:
            result[relative] = union_meta["shards"][path.name]
        elif path.parent == CHAPMAN:
            item = chapman_shards[path.name]
            result[relative] = {"sha256": item["sha256"], "shape": [item["records"], 12, 5000]}
        else:
            raise ValueError(f"Unexpected shard source: {relative}")
    return result


def build(output: Path, unlabeled_count: int, seed: int, mimic_fraction: float) -> dict[str, object]:
    """Create the sampled cohort and smoke-check every storage backend."""
    if output.exists() or output.resolve().is_relative_to((ROOT / "data/raw").resolve()):
        raise ValueError("Destination exists or lies under raw data")
    labeled, union_pool, lookup = union_rows()
    candidates, duplicates = unique_candidates(
        labeled, [union_pool, chapman_rows(), mimic_rows(lookup)]
    )
    unlabeled, quotas = sample_sources(candidates, unlabeled_count, seed, mimic_fraction)
    selected = pd.concat([labeled, unlabeled], ignore_index=True)
    if selected["record_id"].duplicated().any() or selected["signal_sha256"].duplicated().any():
        raise ValueError("Selected duplicate record or waveform")
    heldout = set(table(UNION / "heldout_references.csv")["record_id"])
    if heldout.intersection(selected["record_id"]):
        raise ValueError("PTB held-out record entered training")

    inputs = {str(path): sha256_file(ROOT / path) for path in INPUTS}
    sources = Counter(selected["source"])
    metadata: dict[str, object] = {
        "schema_version": 1,
        "complete": True,
        "seed": seed,
        "sampling_policy": "stratified source quotas; seeded uniform records within each source",
        "mimic_fraction": mimic_fraction,
        "unlabeled_target": unlabeled_count,
        "unlabeled_source_quotas": quotas,
        "candidate_source_counts": candidates["source"].value_counts().to_dict(),
        "duplicate_candidate_waveforms_removed": duplicates,
        "labeled_records": len(labeled),
        "record_count": len(selected),
        "source_counts": dict(sources),
        "shape_per_record": [12, 5000],
        "sampling_rate_hz": 500,
        "units": "mV",
        "label_policy": "Only the 15,359 resolved PTB-XL training proxy labels; no CODE or Challenge mapping",
        "storage_policy": "References verified source shards or original MIMIC WFDB; no waveform copy",
        "input_sha256": inputs,
        "source_sha256": {
            name: sha256_file(ROOT / name)
            for name in (
                "ecg_experiment/sampled_cohort.py",
                "ecg_experiment/sampled_training_dataset.py",
                "ecg_experiment/training_contracts.py",
                "ecg_experiment/waveforms.py",
                "ecg_experiment/public_sources.py",
                "scripts/data/build_sampled_training_dataset.py",
            )
        },
        "source_shards": source_shards(selected),
    }
    with published_directory(output) as stage:
        selected.to_csv(stage / "train_manifest.csv", columns=FIELDS, index=False)
        for budget in ("1", "0.1"):
            source = ROOT / UNION / f"labels_fraction{budget}.csv"
            (stage / source.name).write_bytes(source.read_bytes())
        metadata["table_sha256"] = {
            name: sha256_file(stage / name)
            for name in ("train_manifest.csv", "labels_fraction1.csv", "labels_fraction0.1.csv")
        }
        write_json_atomic(stage / "metadata.json", metadata, sort_keys=True)
        ssl = SampledTrainingECGDataset(stage)
        supervised = SampledTrainingECGDataset(stage, purpose="supervised")
        limited = SampledTrainingECGDataset(stage, purpose="supervised", label_budget="0.1")
        if len(ssl) != unlabeled_count + len(labeled) or len(supervised) != 15359 or len(limited) != 1518:
            raise ValueError("Published cohort label or record count mismatch")
        for source in sorted(sources):
            index = next(i for i, row in enumerate(ssl.rows) if row["source"] == source)
            if ssl[index]["target_available"] or ssl[index]["target"] != -1:
                raise ValueError("SSL target leakage")
        if not supervised[0]["target_available"]:
            raise ValueError("Supervised target unavailable")
        metadata["smoke_checked_sources"] = sorted(sources)
        write_json_atomic(stage / "metadata.json", metadata, sort_keys=True)
    return metadata


def main() -> None:
    """Parse the seed and sample size, then publish a new immutable directory."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--unlabeled-records", type=int, default=100000)
    parser.add_argument("--seed", type=int, default=20260926)
    parser.add_argument("--mimic-fraction", type=float, default=0.7)
    args = parser.parse_args()
    metadata = build(args.output_dir, args.unlabeled_records, args.seed, args.mimic_fraction)
    print(json.dumps({key: metadata[key] for key in (
        "seed", "record_count", "labeled_records", "unlabeled_source_quotas", "source_counts"
    )}, indent=2))


if __name__ == "__main__":
    main()
