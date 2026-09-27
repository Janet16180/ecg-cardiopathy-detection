"""Publish nested, quality-filtered 25k, 50k and 100k ECG cohorts without copying waveforms."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import pandas as pd

from ecg_experiment import ecg_quality
from ecg_experiment.clean_cohorts import (
    FIELDS,
    INPUTS,
    ROOT,
    UNION,
    candidate_pool,
    nested_samples,
    score_rows,
    source_shards,
    table,
)
from ecg_experiment.files import sha256_file, write_json_atomic
from ecg_experiment.sampled_training_dataset import SampledTrainingECGDataset
from ecg_experiment.staging import published_directory

SOURCES = (
    "ecg_experiment/ecg_quality.py",
    "ecg_experiment/clean_cohorts.py",
    "ecg_experiment/sampled_cohort.py",
    "ecg_experiment/sampled_training_dataset.py",
    "ecg_experiment/training_contracts.py",
    "ecg_experiment/waveforms.py",
    "ecg_experiment/public_sources.py",
    "scripts/data/build_clean_cohorts.py",
)
POLICY = {
    "exclusion_reasons": list(ecg_quality.EXCLUSION_REASONS),
    "review_flags": list(ecg_quality.REVIEW_FLAGS),
    "rail_mv": ecg_quality.RAIL_MV,
    "max_amplitude_mv": ecg_quality.MAX_AMPLITUDE_MV,
    "review_amplitude_mv": ecg_quality.REVIEW_AMPLITUDE_MV,
    "flat_seconds": ecg_quality.FLAT_SECONDS,
    "near_flat_median_std_mv": ecg_quality.NEAR_FLAT_STD_MV,
    "noise_band_hz": list(ecg_quality.NOISE_BAND_HZ),
    "noise_median_fraction": ecg_quality.NOISE_FRACTION,
    "baseline_band_hz": list(ecg_quality.BASELINE_BAND_HZ),
    "baseline_median_fraction": ecg_quality.BASELINE_FRACTION,
    "limb_residual_mv": ecg_quality.LIMB_RESIDUAL_MV,
}


def quality_table(rows: pd.DataFrame, path: Path, workers: int) -> pd.DataFrame:
    """
    Score every row, or reuse an earlier table that covers exactly the same records.

    Parameters
    ----------
    rows : pd.DataFrame
        Labeled and candidate rows.
    path : Path
        Compressed CSV holding one quality row per record.
    workers : int
        Number of scoring processes.

    Returns
    -------
    pd.DataFrame
        Quality rows aligned with ``rows``.

    Raises
    ------
    ValueError
        If a reused table does not cover exactly the same records.
    """
    if path.exists():
        scores = pd.read_csv(path, dtype=str, keep_default_na=False)
        if list(scores["record_id"]) != list(rows["record_id"]):
            raise ValueError(f"{path} was built for different records; remove it to rescore")
        return scores
    scores = score_rows(rows, workers)
    path.parent.mkdir(parents=True, exist_ok=True)
    scores.to_csv(path, index=False)
    return scores


def counts_by_source(scores: pd.DataFrame, column: str) -> dict[str, dict[str, int]]:
    """
    Count each semicolon-separated reason or flag per source.

    Parameters
    ----------
    scores : pd.DataFrame
        Output of ``quality_table``.
    column : str
        ``"exclusion"`` or ``"review_flags"``.

    Returns
    -------
    dict[str, dict[str, int]]
        Counts keyed by source, then by reason.
    """
    result: dict[str, dict[str, int]] = {}
    for source, values in scores.groupby("source")[column]:
        reasons = Counter(name for value in values for name in value.split(";") if name)
        result[source] = dict(sorted(reasons.items()))
    return result


def publish(output: Path, labeled: pd.DataFrame, unlabeled: pd.DataFrame,
            labels: dict[str, pd.DataFrame], metadata: dict[str, object]) -> dict[str, object]:
    """
    Write one cohort directory atomically and smoke-check it with the loader.

    Parameters
    ----------
    output : Path
        New cohort directory; it must not exist.
    labeled : pd.DataFrame
        Clean labeled PTB-XL rows.
    unlabeled : pd.DataFrame
        Sampled clean unlabeled rows.
    labels : dict[str, pd.DataFrame]
        Label tables keyed by budget (``"1"`` and ``"0.1"``).
    metadata : dict[str, object]
        Shared metadata; counts, shards and table hashes are added here.

    Returns
    -------
    dict[str, object]
        The published metadata.

    Raises
    ------
    ValueError
        If the output exists or the loader serves unexpected counts.
    """
    if output.exists():
        raise ValueError(f"Destination exists: {output}")
    selected = pd.concat([labeled, unlabeled], ignore_index=True)
    if selected["record_id"].duplicated().any() or selected["signal_sha256"].duplicated().any():
        raise ValueError("Selected duplicate record or waveform")
    metadata = metadata | {
        "unlabeled_target": len(unlabeled),
        "labeled_records": len(labeled),
        "record_count": len(selected),
        "source_counts": selected["source"].value_counts().sort_index().to_dict(),
        "source_shards": source_shards(selected),
    }
    with published_directory(output) as stage:
        selected.to_csv(stage / "train_manifest.csv", columns=FIELDS, index=False)
        for budget, frame in labels.items():
            frame.to_csv(stage / f"labels_fraction{budget}.csv", index=False)
        metadata["table_sha256"] = {
            name: sha256_file(stage / name)
            for name in ("train_manifest.csv", "labels_fraction1.csv", "labels_fraction0.1.csv")
        }
        write_json_atomic(stage / "metadata.json", metadata, sort_keys=True)
        ssl = SampledTrainingECGDataset(stage)
        supervised = SampledTrainingECGDataset(stage, purpose="supervised")
        limited = SampledTrainingECGDataset(stage, purpose="supervised", label_budget="0.1")
        if (len(ssl), len(supervised), len(limited)) != (len(selected), len(labels["1"]), len(labels["0.1"])):
            raise ValueError("Published cohort count mismatch")
        for source in sorted(metadata["source_counts"]):
            item = ssl[next(i for i, row in enumerate(ssl.rows) if row["source"] == source)]
            if item["target_available"]:
                raise ValueError("SSL target leakage")
    return metadata


def build(output_root: Path, quality_dir: Path, sizes: list[int], seed: int, mimic_fraction: float,
          workers: int) -> dict[str, object]:
    """
    Score all candidates, sample the nested cohorts and publish each one.

    Parameters
    ----------
    output_root : Path
        Directory that receives ``clean_{size}_plus_labels_v1`` folders.
    quality_dir : Path
        Directory for the per-record quality table and the aggregate receipt.
    sizes : list[int]
        Unlabeled cohort sizes.
    seed : int
        Selection seed.
    mimic_fraction : float
        Planned MIMIC share of each unlabeled sample.
    workers : int
        Number of scoring processes.

    Returns
    -------
    dict[str, object]
        Aggregate receipt.

    Raises
    ------
    ValueError
        If a held-out PTB-XL record entered a cohort.
    """
    labeled, candidates, duplicates = candidate_pool()
    everything = pd.concat([labeled, candidates], ignore_index=True)
    scores = quality_table(everything, quality_dir / "record_quality.csv.gz", workers)
    excluded = set(scores.loc[scores["exclusion"] != "", "record_id"])
    clean_labeled = labeled.loc[~labeled["record_id"].isin(excluded)]
    clean_candidates = candidates.loc[~candidates["record_id"].isin(excluded)]
    labels = {budget: table(UNION / f"labels_fraction{budget}.csv") for budget in ("1", "0.1")}
    labels = {budget: frame.loc[frame["record_id"].isin(clean_labeled["record_id"])]
              for budget, frame in labels.items()}
    heldout = set(table(UNION / "heldout_references.csv")["record_id"])
    if heldout & set(everything["record_id"]):
        raise ValueError("PTB held-out record entered the candidate pool")

    samples, quotas = nested_samples(clean_candidates, sizes, seed, mimic_fraction)
    shared = {
        "schema_version": 1, "complete": True, "seed": seed, "mimic_fraction": mimic_fraction,
        "sampling_policy": "nested source quotas; one seeded permutation per source; smaller cohorts "
                           "are prefixes of the larger ones",
        "quality_policy": POLICY,
        "shape_per_record": [12, 5000], "sampling_rate_hz": 500, "units": "mV",
        "label_policy": "Clean subset of the 15,359 resolved PTB-XL training proxy labels",
        "storage_policy": "References verified source shards or original MIMIC WFDB; no waveform copy",
        "input_sha256": {str(path): sha256_file(ROOT / path) for path in INPUTS},
        "source_sha256": {name: sha256_file(ROOT / name) for name in SOURCES},
        "quality_table_sha256": sha256_file(quality_dir / "record_quality.csv.gz"),
        "nested_sizes": sorted(sizes),
    }
    cohorts = {}
    for size in sorted(sizes):
        name = f"clean_{size // 1000}k_plus_labels_v1"
        metadata = publish(output_root / name, clean_labeled, samples[size], labels,
                           shared | {"unlabeled_source_quotas": quotas[size]})
        cohort_ids = set(samples[size]["record_id"]) | set(clean_labeled["record_id"])
        flags = scores.loc[scores["record_id"].isin(cohort_ids)]
        cohorts[name] = {
            "record_count": metadata["record_count"], "source_counts": metadata["source_counts"],
            "unlabeled_source_quotas": quotas[size], "table_sha256": metadata["table_sha256"],
            "metadata_sha256": sha256_file(output_root / name / "metadata.json"),
            "review_flags": counts_by_source(flags, "review_flags"),
        }

    receipt = {
        "candidate_records": len(candidates), "labeled_records": len(labeled),
        "duplicate_candidate_waveforms_removed": duplicates,
        "candidate_source_counts": candidates["source"].value_counts().sort_index().to_dict(),
        "clean_candidate_source_counts": clean_candidates["source"].value_counts().sort_index().to_dict(),
        "clean_label_counts": {budget: len(frame) for budget, frame in labels.items()},
        "exclusions_by_source": counts_by_source(scores, "exclusion"),
        "excluded_records_by_source": scores.loc[scores["exclusion"] != "", "source"].value_counts()
        .sort_index().to_dict(),
        "review_flags_by_source": counts_by_source(scores, "review_flags"),
        "cohorts": cohorts,
    } | {key: shared[key] for key in ("seed", "mimic_fraction", "quality_policy", "source_sha256",
                                      "quality_table_sha256", "nested_sizes")}
    write_json_atomic(quality_dir / "receipt.json", receipt, sort_keys=True)
    return receipt


def main() -> None:
    """Parse arguments and publish the cohorts."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=ROOT / "data/processed")
    parser.add_argument("--quality-dir", type=Path, default=ROOT / "outputs/data_quality/clean_cohorts_v1")
    parser.add_argument("--sizes", type=int, nargs="+", default=[25000, 50000, 100000])
    parser.add_argument("--seed", type=int, default=20260927)
    parser.add_argument("--mimic-fraction", type=float, default=0.7)
    parser.add_argument("--workers", type=int, default=6)
    args = parser.parse_args()
    receipt = build(args.output_root, args.quality_dir, args.sizes, args.seed, args.mimic_fraction,
                    args.workers)
    print(json.dumps({key: receipt[key] for key in (
        "excluded_records_by_source", "clean_label_counts", "cohorts")}, indent=2, default=str))


if __name__ == "__main__":
    main()
