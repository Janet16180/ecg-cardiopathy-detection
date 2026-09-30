"""Write nested cohort tiers as manifest directories: the first N rows of one saved candidate order.

Copied once from the frozen ``scripts/data/build_clean_cohorts_v2.py`` so later cohort versions import it
instead of copying it again.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from .clean_cohorts import UNION, read_signal, table
from .clean_cohorts_v2 import CHALLENGE_ROOT, ROOT
from .files import sha256_file, write_json_atomic
from .public_sources import signal_sha256
from .waveforms import read_record

CHECKED_PER_BACKEND = 20
TABLES = ("train_manifest.csv", "labels_fraction1.csv", "labels_fraction0.1.csv")
SUMMARY_KEYS = ("records", "source_counts", "block_counts", "quality_status", "patients_known",
                "records_without_patient_id", "label_available")


def check_waveforms(rows: pd.DataFrame, seed: int) -> dict[str, int]:
    """
    Read a few rows of every locally readable backend and compare their waveform hash with the manifest.

    Parameters
    ----------
    rows : pd.DataFrame
        Passed rows of a tier.
    seed : int
        Sampling seed.

    Returns
    -------
    dict[str, int]
        Rows checked per backend.

    Raises
    ------
    ValueError
        If a waveform differs from its manifest hash.
    """
    checked = {}
    for backend, group in rows.groupby("backend"):
        if backend == "code15_archive":
            continue
        sample = group.sample(min(CHECKED_PER_BACKEND, len(group)), random_state=seed)
        for row in sample.itertuples(index=False):
            if backend == "challenge_wfdb":
                signal = read_record(ROOT / CHALLENGE_ROOT, str(Path(row.path).relative_to(CHALLENGE_ROOT)))
            else:
                signal = read_signal(backend, row.path, row.index)
            if signal_sha256(signal) != row.signal_sha256:
                raise ValueError(f"Waveform differs from the manifest: {row.record_id}")
        checked[backend] = len(sample)
    return checked


def tier_summary(rows: pd.DataFrame) -> dict[str, object]:
    """
    Composition of one tier.

    Parameters
    ----------
    rows : pd.DataFrame
        The tier's rows.

    Returns
    -------
    dict[str, object]
        Counts per source, block and quality status, patients, labels and review flags.
    """
    known = rows["patient_id"] != ""
    return {
        "records": len(rows),
        "source_counts": rows["source"].value_counts().sort_index().to_dict(),
        "block_counts": rows["block"].value_counts().to_dict(),
        "quality_status": rows["quality_status"].value_counts().to_dict(),
        "patients_known": int(rows.loc[known, "patient_id"].nunique()),
        "records_without_patient_id": int((~known).sum()),
        "label_available": int(rows["label_available"].sum()),
        "review_flagged": int((rows["review_flags"] != "").sum()),
    }


def publish_tier(output: Path, rows: pd.DataFrame, labels: dict[str, pd.DataFrame],
                 shared: dict[str, object], seed: int) -> dict[str, object]:
    """
    Write one tier directory: its manifest, the PTB-XL label files and metadata.

    Parameters
    ----------
    output : Path
        New directory; it must not exist.
    rows : pd.DataFrame
        The tier's rows in selection order.
    labels : dict[str, pd.DataFrame]
        PTB-XL label tables keyed by budget.
    shared : dict[str, object]
        Metadata common to all tiers.
    seed : int
        Seed for the waveform spot check.

    Returns
    -------
    dict[str, object]
        The tier metadata.

    Raises
    ------
    ValueError
        If the output exists, a label record is missing from the tier, or a record or waveform repeats.
    """
    if output.exists():
        raise ValueError(f"Destination exists: {output}")
    passed = rows[rows["quality_status"] == "passed"]
    if rows["record_id"].duplicated().any() or passed["signal_sha256"].duplicated().any():
        raise ValueError("A tier repeats a record or a waveform")
    if not set(labels["1"]["record_id"]) <= set(rows["record_id"]):
        raise ValueError("A PTB-XL label record is missing from the tier")
    stage = output.with_name(output.name + ".partial")
    stage.mkdir(parents=True)
    rows.to_csv(stage / "train_manifest.csv", index=False)
    for budget, frame in labels.items():
        frame.to_csv(stage / f"labels_fraction{budget}.csv", index=False)
    metadata = shared | tier_summary(rows) | {
        "waveforms_checked": check_waveforms(passed, seed),
        "table_sha256": {name: sha256_file(stage / name) for name in TABLES},
    }
    write_json_atomic(stage / "metadata.json", metadata, sort_keys=True)
    stage.rename(output)
    return metadata


def tier_sizes(sizes: dict[str, int], candidates: int) -> dict[str, int]:
    """
    Feasible tiers, plus a ``max`` tier with every candidate when the largest size is out of reach.

    Parameters
    ----------
    sizes : dict[str, int]
        Requested tier sizes by name.
    candidates : int
        Number of rows in the candidate order.

    Returns
    -------
    dict[str, int]
        Tier sizes to write.
    """
    tiers = {name: size for name, size in sizes.items() if size <= candidates}
    if len(tiers) < len(sizes):
        tiers["max"] = candidates
    return tiers


def publish_tiers(order: pd.DataFrame, sizes: dict[str, int], output_root: Path, suffix: str,
                  shared: dict[str, object], seed: int) -> dict[str, object]:
    """
    Write every feasible tier of a candidate order.

    Parameters
    ----------
    order : pd.DataFrame
        The candidate order.
    sizes : dict[str, int]
        Requested tier sizes by name.
    output_root : Path
        Parent of the tier directories.
    suffix : str
        Directory suffix, such as ``v3``.
    shared : dict[str, object]
        Metadata common to all tiers.
    seed : int
        Seed for the waveform spot checks.

    Returns
    -------
    dict[str, object]
        Composition and metadata hash per written tier.
    """
    clean_labeled = set(order.loc[order["source"] == "ptbxl", "record_id"])
    labels = {budget: table(UNION / f"labels_fraction{budget}.csv") for budget in ("1", "0.1")}
    labels = {budget: frame[frame["record_id"].isin(clean_labeled)] for budget, frame in labels.items()}
    written = {}
    for name, size in tier_sizes(sizes, len(order)).items():
        output = output_root / f"clean_{name}_{suffix}"
        tier = shared | {"tier": name, "size": size}
        metadata = publish_tier(output, order.iloc[:size], labels, tier, seed)
        written[name] = {key: metadata[key] for key in SUMMARY_KEYS}
        written[name]["metadata_sha256"] = sha256_file(output / "metadata.json")
    return written
